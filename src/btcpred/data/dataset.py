"""
Timeline assembly: turn a directory of monthly ``.btcz`` shards into one
contiguous, gap-free 1 Hz view of the market, plus an on-disk memmap cache.

Two access patterns are supported:

``ShardIndex`` / ``load_range``
    Decode on demand.  Good for notebooks, evaluation and rendering, where you
    only ever touch a few days at a time.

``MarketStore``
    A set of ``numpy.memmap`` columns covering the entire history, produced once
    by ``scripts/materialize.py``.  Training reads this: no decompression in the
    hot loop, the OS page cache does the work, and random access across six
    years costs a page fault instead of a zstd frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .btcz import PRICE_SCALE, Shard, read_header, read_shard

# Columns materialised to disk, in a fixed order.
#
# NOTE ON logp's dtype.  Everything the model sees is a *difference* of log
# prices.  At BTC's price level a float32 log price carries only ~1e-6 absolute
# precision, while a typical 1-second return is ~1e-5 -- measured, that is a
# 4.25% error on every single return, which silently caps accuracy no matter
# how good the network is.  So logp is float64 (1.7 GB over the full history,
# irrelevant next to the feature matrix) and every *offset* from it is float32,
# where the values are small and relative precision is full.
RAW_COLUMNS: Dict[str, str] = {
    "logp": "float64",       # log(close)  -- the anchor, must be float64
    "close": "float32",      # USDT, for plotting and reporting only
    "hi_off": "float32",     # log(high / close)
    "lo_off": "float32",     # log(low  / close)
    "lv": "float32",         # log1p(base volume)
    "lntr": "float32",       # log1p(trade count)
    "imb": "float32",        # 2 * (taker-buy share - 0.5), 0 on empty seconds
    "valid": "uint8",
}


# --------------------------------------------------------------------- indexing
@dataclass(frozen=True)
class ShardMeta:
    path: Path
    t0: int
    n: int

    @property
    def t_end(self) -> int:          # inclusive
        return self.t0 + self.n - 1


class ShardIndex:
    """Discovers shards and maps absolute unix seconds to (shard, offset)."""

    def __init__(self, shard_dir: str | Path):
        self.dir = Path(shard_dir)
        files = sorted(self.dir.glob("*.btcz"))
        if not files:
            raise FileNotFoundError(
                f"no .btcz shards in {self.dir}. Run scripts/download_pack.py first."
            )
        metas: List[ShardMeta] = []
        for f in files:
            h = read_header(f)
            metas.append(ShardMeta(path=f, t0=h["t0"], n=h["n"]))
        metas.sort(key=lambda m: m.t0)

        # verify the months butt up against each other with no hole
        for a, b in zip(metas, metas[1:]):
            if b.t0 != a.t_end + 1:
                raise ValueError(
                    f"timeline break between {a.path.name} (ends {a.t_end}) and "
                    f"{b.path.name} (starts {b.t0}); re-run the packer for the gap"
                )
        self.shards = metas
        self.t0 = metas[0].t0
        self.t_end = metas[-1].t_end
        self.n = self.t_end - self.t0 + 1

    def __repr__(self) -> str:
        from datetime import datetime, timezone
        f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        return (f"<ShardIndex {len(self.shards)} shards {f(self.t0)}..{f(self.t_end)} "
                f"n={self.n:,}s>")

    def load_range(self, t_start: int, t_stop: int) -> Dict[str, np.ndarray]:
        """Decode ``[t_start, t_stop)`` as physical-unit numpy arrays."""
        if t_start < self.t0 or t_stop > self.t_end + 1 or t_stop <= t_start:
            raise ValueError(
                f"range [{t_start},{t_stop}) outside [{self.t0},{self.t_end + 1})")
        n = t_stop - t_start
        out = {k: np.zeros(n, dtype=np.dtype(v)) for k, v in RAW_COLUMNS.items()}
        for m in self.shards:
            lo, hi = max(t_start, m.t0), min(t_stop, m.t_end + 1)
            if lo >= hi:
                continue
            sh: Shard = read_shard(m.path)
            a, b = lo - m.t0, hi - m.t0
            d, e = lo - t_start, hi - t_start
            c64 = sh.close[a:b].astype(np.float64) / PRICE_SCALE
            h64 = sh.high[a:b].astype(np.float64) / PRICE_SCALE
            l64 = sh.low[a:b].astype(np.float64) / PRICE_SCALE
            lp = np.log(np.maximum(c64, 1e-9))
            out["logp"][d:e] = lp
            out["close"][d:e] = c64.astype(np.float32)
            out["hi_off"][d:e] = (np.log(np.maximum(h64, 1e-9)) - lp).astype(np.float32)
            out["lo_off"][d:e] = (np.log(np.maximum(l64, 1e-9)) - lp).astype(np.float32)
            out["lv"][d:e] = np.log1p(np.maximum(sh.vol[a:b], 0.0)).astype(np.float32)
            out["lntr"][d:e] = np.log1p(np.maximum(sh.ntrades[a:b], 0.0)).astype(np.float32)
            out["imb"][d:e] = np.where(
                sh.vol[a:b] > 0, (sh.takerfrac[a:b] - 0.5) * 2.0, 0.0).astype(np.float32)
            out["valid"][d:e] = sh.valid[a:b]
            del sh, c64, h64, l64, lp
        out["t0"] = t_start  # type: ignore[assignment]
        return out


# ------------------------------------------------------------------ memmap store
class MarketStore:
    """Whole-history columns backed by ``numpy.memmap`` files on local disk."""

    META = "store.json"

    def __init__(self, root: str | Path, mode: str = "r"):
        self.root = Path(root)
        meta_path = self.root / self.META
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Build it with scripts/materialize.py")
        self.meta = json.loads(meta_path.read_text())
        self.t0: int = self.meta["t0"]
        self.n: int = self.meta["n"]
        self.t_end: int = self.t0 + self.n - 1
        self.cols: Dict[str, np.memmap] = {}
        for name, dtype in self.meta["columns"].items():
            self.cols[name] = np.memmap(self.root / f"{name}.bin",
                                        dtype=np.dtype(dtype), mode=mode,
                                        shape=(self.n,))
        feat_path = self.root / "features.bin"
        self.feature_names: List[str] = self.meta.get("feature_names", [])
        self.features: Optional[np.memmap] = None
        if feat_path.exists() and self.feature_names:
            self.features = np.memmap(
                feat_path, dtype=np.dtype(self.meta["feature_dtype"]), mode=mode,
                shape=(self.n, len(self.feature_names)))

    # -- conversions -------------------------------------------------------
    def idx(self, t: int) -> int:
        if not (self.t0 <= t <= self.t_end):
            raise IndexError(f"t={t} outside store [{self.t0},{self.t_end}]")
        return t - self.t0

    def time(self, i: int) -> int:
        return self.t0 + i

    @property
    def close(self) -> np.memmap:
        return self.cols["close"]

    def __repr__(self) -> str:
        from datetime import datetime, timezone
        f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")
        nf = len(self.feature_names)
        return (f"<MarketStore {f(self.t0)}..{f(self.t_end)} n={self.n:,}s "
                f"cols={list(self.cols)} features={nf}>")

    @staticmethod
    def create(root: str | Path, t0: int, n: int,
               feature_names: Sequence[str], feature_dtype: str = "float16"
               ) -> "MarketStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name, dtype in RAW_COLUMNS.items():
            np.memmap(root / f"{name}.bin", dtype=np.dtype(dtype), mode="w+",
                      shape=(n,)).flush()
        if feature_names:
            np.memmap(root / "features.bin", dtype=np.dtype(feature_dtype), mode="w+",
                      shape=(n, len(feature_names))).flush()
        (root / MarketStore.META).write_text(json.dumps({
            "t0": int(t0), "n": int(n), "columns": dict(RAW_COLUMNS),
            "feature_names": list(feature_names), "feature_dtype": feature_dtype,
        }, indent=2))
        return MarketStore(root, mode="r+")


def utc(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def day_bounds(day: str) -> Tuple[int, int]:
    """'2024-03-17' -> (unix start, unix end-exclusive)."""
    from datetime import datetime, timezone
    d = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    s = int(d.timestamp())
    return s, s + 86400
