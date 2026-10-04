"""
btcz -- a compact, self-describing container for dense 1 Hz OHLCV market data.

Why not Parquet/CSV?
--------------------
We need ~207 million consecutive seconds of BTCUSDT data to live inside a git
repository (GitHub hard-caps single files at 100 MB) and to be memory-mappable
at training time on a Snowflake node with no internet access.

Layout (little-endian throughout)::

    magic        6 bytes   b"BTCZ1\\0"
    header_len   uint32
    header       header_len bytes of UTF-8 JSON
    payload      concatenated zstd frames, one per column, in header order

The grid is *dense*: the file covers every second in ``[t0, t0 + n)`` with no
holes.  Seconds the exchange never reported are forward-filled (flat candle,
zero volume) and flagged in the ``valid`` bitmask, so a model consuming a fixed
43 200-step (12 h) window never has to deal with ragged time axes, yet can still
tell synthetic bars from real ones.

Column encoding (chosen by measurement, see docs/DATA_FORMAT.md):

===========  =======  ==========================================================
column       dtype    meaning
===========  =======  ==========================================================
valid        bitmask  1 = second was really reported by the exchange
dclose       int32    first difference of close price in cents (integer)
open_off     int32    open  - close, cents
high_off     int32    high  - close, cents
low_off      int32    low   - close, cents
vol          int32    base volume in units of 1e-5 BTC (exact for Binance ticks)
qvol         int32    quote volume in whole USDT
ntrades      int32    number of trades in the second
takerfrac    uint16   taker-buy base volume / base volume, scaled by 65535
===========  =======  ==========================================================

Delta/offset coding turns slowly-varying prices into near-zero integers, which
zstd then packs ~5-9x.  Measured on 2024-01: 378 MB CSV -> 84 MB Binance zip ->
~25 MB btcz, fully lossless for price and base volume.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover
    raise ImportError("btcz requires the 'zstandard' package (pip install zstandard)") from exc

MAGIC = b"BTCZ1\0"
PRICE_SCALE = 100          # cents
VOL_SCALE = 100_000        # 1e-5 BTC
TAKER_SCALE = 65535

# name -> numpy dtype of the *stored* representation
COLUMNS: Dict[str, str] = {
    "valid": "uint8",      # bit-packed
    "dclose": "int32",
    "open_off": "int32",
    "high_off": "int32",
    "low_off": "int32",
    "vol": "int32",
    "qvol": "int32",
    "ntrades": "int32",
    "takerfrac": "uint16",
}


def _compressor(level: int, threads: int = 0) -> "zstd.ZstdCompressor":
    params = zstd.ZstdCompressionParameters.from_level(
        level, enable_ldm=True, window_log=27, threads=threads
    )
    return zstd.ZstdCompressor(compression_params=params)


@dataclass
class Shard:
    """One decoded btcz file held as plain numpy arrays."""

    t0: int                      # unix seconds of the first row
    n: int                       # number of rows (== seconds)
    close: np.ndarray            # int64 cents
    open: np.ndarray             # int64 cents
    high: np.ndarray             # int64 cents
    low: np.ndarray              # int64 cents
    vol: np.ndarray              # float64 BTC
    qvol: np.ndarray             # float64 USDT
    ntrades: np.ndarray          # int32
    takerfrac: np.ndarray        # float32 in [0, 1]
    valid: np.ndarray            # bool
    meta: Dict[str, Any]

    @property
    def t_end(self) -> int:
        return self.t0 + self.n - 1

    def close_float(self) -> np.ndarray:
        return self.close.astype(np.float64) / PRICE_SCALE


def write_shard(
    path: str | Path,
    *,
    t0: int,
    close_cents: np.ndarray,
    open_cents: np.ndarray,
    high_cents: np.ndarray,
    low_cents: np.ndarray,
    vol: np.ndarray,
    qvol: np.ndarray,
    ntrades: np.ndarray,
    valid: np.ndarray,
    takerbuy_base: Optional[np.ndarray] = None,
    takerfrac: Optional[np.ndarray] = None,
    level: int = 17,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode one dense 1 Hz block.  All arrays must share the same length."""
    n = int(close_cents.shape[0])
    if (takerbuy_base is None) == (takerfrac is None):
        raise ValueError("pass exactly one of takerbuy_base= or takerfrac=")
    for name, arr in (
        ("open", open_cents), ("high", high_cents), ("low", low_cents),
        ("vol", vol), ("qvol", qvol), ("ntrades", ntrades), ("valid", valid),
        ("taker", takerbuy_base if takerbuy_base is not None else takerfrac),
    ):
        if arr.shape[0] != n:
            raise ValueError(f"column {name} has length {arr.shape[0]}, expected {n}")

    if takerfrac is not None:
        frac_q = np.asarray(takerfrac)
        if frac_q.dtype != np.uint16:   # caller may hand us a float ratio
            frac_q = np.rint(np.clip(frac_q, 0.0, 1.0) * TAKER_SCALE).astype(np.uint16)
    else:
        frac_q = np.rint(
            np.clip(
                np.divide(
                    np.asarray(takerbuy_base, dtype=np.float64),
                    np.maximum(np.asarray(vol, dtype=np.float64), 1e-12),
                ),
                0.0, 1.0,
            ) * TAKER_SCALE
        ).astype(np.uint16)

    close_cents = close_cents.astype(np.int64, copy=False)
    # prepend 0 (NOT close[0]) so that cumsum() on the decode side recovers the
    # absolute price level: dclose[0] == close[0].
    dclose = np.diff(close_cents, prepend=np.int64(0)).astype(np.int32)
    if np.abs(close_cents).max() > np.iinfo(np.int32).max:
        raise ValueError("price in cents overflows int32")
    enc = {
        "valid": np.packbits(valid.astype(bool)),
        "dclose": dclose,
        "open_off": (open_cents.astype(np.int64) - close_cents).astype(np.int32),
        "high_off": (high_cents.astype(np.int64) - close_cents).astype(np.int32),
        "low_off": (low_cents.astype(np.int64) - close_cents).astype(np.int32),
        "vol": np.rint(np.asarray(vol, dtype=np.float64) * VOL_SCALE).astype(np.int32),
        "qvol": np.rint(np.asarray(qvol, dtype=np.float64)).astype(np.int32),
        "ntrades": np.asarray(ntrades).astype(np.int32),
        "takerfrac": frac_q,
    }

    cctx = _compressor(level)
    frames, cols = [], []
    for name in COLUMNS:
        raw = np.ascontiguousarray(enc[name]).tobytes()
        blob = cctx.compress(raw)
        frames.append(blob)
        cols.append({
            "name": name,
            "dtype": COLUMNS[name],
            "raw_bytes": len(raw),
            "comp_bytes": len(blob),
        })

    header = {
        "format": "btcz1",
        "symbol": "BTCUSDT",
        "interval": "1s",
        "t0": int(t0),
        "n": n,
        "price_scale": PRICE_SCALE,
        "vol_scale": VOL_SCALE,
        "taker_scale": TAKER_SCALE,
        "zstd_level": level,
        "n_valid": int(valid.sum()),
        "columns": cols,
    }
    if extra_meta:
        header.update(extra_meta)

    hb = json.dumps(header, separators=(",", ":")).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hb)))
        fh.write(hb)
        for blob in frames:
            fh.write(blob)
    tmp.replace(path)
    header["file_bytes"] = path.stat().st_size
    return header


def read_header(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        if fh.read(6) != MAGIC:
            raise ValueError(f"{path}: not a btcz file")
        (hlen,) = struct.unpack("<I", fh.read(4))
        return json.loads(fh.read(hlen).decode())


def read_shard(path: str | Path, columns: Optional[Iterable[str]] = None) -> Shard:
    """Decode a btcz file back into physical units."""
    path = Path(path)
    want = set(columns) if columns is not None else None
    with open(path, "rb") as fh:
        if fh.read(6) != MAGIC:
            raise ValueError(f"{path}: not a btcz file")
        (hlen,) = struct.unpack("<I", fh.read(4))
        header = json.loads(fh.read(hlen).decode())
        dctx = zstd.ZstdDecompressor()
        out: Dict[str, np.ndarray] = {}
        for col in header["columns"]:
            blob = fh.read(col["comp_bytes"])
            if want is not None and col["name"] not in want and col["name"] != "valid":
                continue
            raw = dctx.decompress(blob, max_output_size=col["raw_bytes"] + 64)
            out[col["name"]] = np.frombuffer(raw, dtype=np.dtype(col["dtype"]))

    n = header["n"]
    close = np.cumsum(out["dclose"].astype(np.int64))
    valid = np.unpackbits(out["valid"])[:n].astype(bool)
    vol = out["vol"].astype(np.float64) / header["vol_scale"]
    return Shard(
        t0=header["t0"],
        n=n,
        close=close,
        open=close + out["open_off"].astype(np.int64),
        high=close + out["high_off"].astype(np.int64),
        low=close + out["low_off"].astype(np.int64),
        vol=vol,
        qvol=out["qvol"].astype(np.float64),
        ntrades=out["ntrades"].astype(np.int32),
        takerfrac=(out["takerfrac"].astype(np.float32) / header["taker_scale"]),
        valid=valid,
        meta=header,
    )
