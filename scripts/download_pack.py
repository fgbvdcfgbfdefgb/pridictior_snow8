#!/usr/bin/env python3
"""
Download BTCUSDT 1-second klines from Binance Data Vision and pack them into
dense, git-friendly ``.btcz`` monthly shards.

This is the ONLY step that needs internet access.  Run it on any machine that
can reach data.binance.vision; everything downstream (training, evaluation,
rendering) reads the shards offline.

    python scripts/download_pack.py --start 2020-01 --end 2026-10 --out data/shards

Design notes
------------
* Binance publishes *monthly* archives a few days after month end and *daily*
  archives within ~24 h.  We use monthly where available and fall back to
  stitching dailies for the trailing partial month.
* Timestamp units changed from **milliseconds to microseconds** in the 2025
  archives.  We sniff the magnitude of the first timestamp per file instead of
  assuming, otherwise 2025+ data silently lands ~55 000 years in the future.
* Some archives carry a CSV header row, some do not.  We sniff that too.
* Output is a *dense* 1 Hz grid: every second in the month exists.  Seconds the
  exchange did not report are forward-filled and marked invalid in a bitmask.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.data.btcz import PRICE_SCALE, TAKER_SCALE, VOL_SCALE, write_shard  # noqa: E402

BASE = "https://data.binance.vision/data/spot"
SYMBOL = "BTCUSDT"
CHUNK = 500_000           # CSV rows per read chunk -> keeps peak RSS ~150 MB
RETRIES = 5


# --------------------------------------------------------------------------- io
def _fetch(url: str, timeout: int = 180) -> bytes | None:
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btcpred/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _ts_divisor(first_ts: int) -> int:
    """Return the factor that converts the archive's open_time to unix seconds."""
    if first_ts > 10**17:
        return 1_000_000_000      # nanoseconds
    if first_ts > 10**14:
        return 1_000_000          # microseconds (Binance, 2025+)
    if first_ts > 10**11:
        return 1_000              # milliseconds (Binance, pre-2025)
    return 1                      # already seconds


def _iter_csv(raw_zip: bytes):
    """Yield (ts_seconds, o, h, l, c, v, qv, n, tb) numpy chunks from a zip."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as fh:
            head = fh.read(64)
        skip = 0 if head[:1].isdigit() else 1
        with zf.open(name) as fh:
            reader = pd.read_csv(
                fh, header=None, skiprows=skip, chunksize=CHUNK,
                usecols=[0, 1, 2, 3, 4, 5, 7, 8, 9],
                names=["ot", "o", "h", "l", "c", "v", "qv", "n", "tb"],
                dtype={"ot": np.int64, "o": np.float64, "h": np.float64,
                       "l": np.float64, "c": np.float64, "v": np.float64,
                       "qv": np.float64, "n": np.int64, "tb": np.float64},
            )
            div = None
            for ch in reader:
                ot = ch["ot"].to_numpy()
                if div is None:
                    div = _ts_divisor(int(ot[0]))
                yield (ot // div, ch["o"].to_numpy(), ch["h"].to_numpy(),
                       ch["l"].to_numpy(), ch["c"].to_numpy(), ch["v"].to_numpy(),
                       ch["qv"].to_numpy(), ch["n"].to_numpy(), ch["tb"].to_numpy())
                del ch


# ----------------------------------------------------------------------- packing
def _month_bounds(year: int, month: int) -> tuple[int, int]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def _urls_for(year: int, month: int, until: date | None):
    """Prefer the monthly archive; otherwise stitch the daily archives."""
    m = f"{year:04d}-{month:02d}"
    monthly = f"{BASE}/monthly/klines/{SYMBOL}/1s/{SYMBOL}-1s-{m}.zip"
    try:
        req = urllib.request.Request(monthly, method="HEAD",
                                     headers={"User-Agent": "btcpred/1.0"})
        with urllib.request.urlopen(req, timeout=60):
            return [monthly], "monthly"
    except Exception:  # noqa: BLE001
        pass
    out, d = [], date(year, month, 1)
    while d.month == month and d.year == year:
        if until is not None and d > until:
            break
        out.append(f"{BASE}/daily/klines/{SYMBOL}/1s/{SYMBOL}-1s-{d.isoformat()}.zip")
        d += timedelta(days=1)
    return out, "daily"


def pack_month(year: int, month: int, out_dir: Path, level: int,
               until: date | None) -> dict:
    out_path = out_dir / f"{SYMBOL}-1s-{year:04d}-{month:02d}.btcz"
    if out_path.exists():
        return {"month": f"{year:04d}-{month:02d}", "status": "cached",
                "bytes": out_path.stat().st_size}

    urls, mode = _urls_for(year, month, until)
    if not urls:
        return {"month": f"{year:04d}-{month:02d}", "status": "unavailable"}

    t0, t_end_excl = _month_bounds(year, month)
    if mode == "daily" and until is not None:
        last = min(date(year, month, 1) + timedelta(days=len(urls) - 1), until)
        t_end_excl = int(datetime(last.year, last.month, last.day,
                                  tzinfo=timezone.utc).timestamp()) + 86400
    n = t_end_excl - t0

    # dense accumulators (int32 keeps peak RSS ~85 MB for a 31-day month)
    close = np.zeros(n, dtype=np.int32)
    opn = np.zeros(n, dtype=np.int32)
    high = np.zeros(n, dtype=np.int32)
    low = np.zeros(n, dtype=np.int32)
    vol_q = np.zeros(n, dtype=np.int32)
    qvol_q = np.zeros(n, dtype=np.int32)
    ntr = np.zeros(n, dtype=np.int32)
    tfrac = np.zeros(n, dtype=np.uint16)
    valid = np.zeros(n, dtype=bool)

    rows = 0
    for url in urls:
        raw = _fetch(url)
        if raw is None:
            continue
        for ts, o, h, l, c, v, qv, nt, tb in _iter_csv(raw):
            idx = (ts - t0).astype(np.int64)
            keep = (idx >= 0) & (idx < n)
            if not keep.all():
                idx, o, h, l, c, v, qv, nt, tb = (
                    idx[keep], o[keep], h[keep], l[keep], c[keep],
                    v[keep], qv[keep], nt[keep], tb[keep])
            if idx.size == 0:
                continue
            close[idx] = np.rint(c * PRICE_SCALE).astype(np.int32)
            opn[idx] = np.rint(o * PRICE_SCALE).astype(np.int32)
            high[idx] = np.rint(h * PRICE_SCALE).astype(np.int32)
            low[idx] = np.rint(l * PRICE_SCALE).astype(np.int32)
            vol_q[idx] = np.rint(v * VOL_SCALE).astype(np.int32)
            qvol_q[idx] = np.rint(qv).astype(np.int32)
            ntr[idx] = nt.astype(np.int32)
            with np.errstate(divide="ignore", invalid="ignore"):
                fr = np.where(v > 0, tb / np.maximum(v, 1e-12), 0.0)
            tfrac[idx] = np.rint(np.clip(fr, 0.0, 1.0) * TAKER_SCALE).astype(np.uint16)
            valid[idx] = True
            rows += int(idx.size)
        del raw

    if rows == 0:
        return {"month": f"{year:04d}-{month:02d}", "status": "empty"}

    # forward-fill price through gaps; flat candle, zero flow
    pos = np.where(valid, np.arange(n, dtype=np.int64), -1)
    np.maximum.accumulate(pos, out=pos)
    first = int(np.argmax(valid))
    pos[pos < 0] = first                      # leading gap -> back-fill
    gaps = ~valid
    if gaps.any():
        close[gaps] = close[pos[gaps]]
        opn[gaps] = close[gaps]
        high[gaps] = close[gaps]
        low[gaps] = close[gaps]

    hdr = write_shard(
        out_path,
        t0=t0,
        close_cents=close, open_cents=opn, high_cents=high, low_cents=low,
        vol=vol_q.astype(np.float64) / VOL_SCALE,
        qvol=qvol_q.astype(np.float64),
        ntrades=ntr,
        takerfrac=tfrac,
        valid=valid,
        level=level,
        extra_meta={"source": mode, "rows_seen": rows,
                    "built": datetime.now(timezone.utc).isoformat()},
    )
    return {"month": f"{year:04d}-{month:02d}", "status": "ok",
            "bytes": hdr["file_bytes"], "n": n, "n_valid": hdr["n_valid"],
            "gap_pct": round(100.0 * (n - hdr["n_valid"]) / n, 4), "source": mode}


def _job(args):
    y, m, out_dir, level, until = args
    try:
        r = pack_month(y, m, Path(out_dir), level, until)
    except Exception as e:  # noqa: BLE001
        r = {"month": f"{y:04d}-{m:02d}", "status": "error",
             "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}
    print(json.dumps(r), flush=True)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2020-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default=None, help="last month, YYYY-MM (default: today)")
    ap.add_argument("--until", default=None,
                    help="last calendar day to include, YYYY-MM-DD (default: yesterday UTC)")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--level", type=int, default=17, help="zstd compression level")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    sy, sm = (int(x) for x in a.start.split("-"))
    if a.end:
        ey, em = (int(x) for x in a.end.split("-"))
    else:
        now = datetime.now(timezone.utc)
        ey, em = now.year, now.month
    until = (date.fromisoformat(a.until) if a.until
             else (datetime.now(timezone.utc).date() - timedelta(days=1)))

    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        y, m = (y + (m == 12), (m % 12) + 1)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# packing {len(months)} months -> {out_dir} "
          f"(workers={a.workers}, zstd={a.level}, until={until})", flush=True)

    t_start = time.time()
    jobs = [(yy, mm, str(out_dir), a.level, until) for yy, mm in months]
    results = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            results.append(r)

    total = sum(r.get("bytes", 0) for r in results)
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    manifest = {
        "symbol": SYMBOL, "interval": "1s", "format": "btcz1",
        "built": datetime.now(timezone.utc).isoformat(),
        "months": results, "total_bytes": total, "n_shards": len(ok),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"# done: {len(ok)}/{len(months)} shards, {total/1e9:.2f} GB, "
          f"{manifest['elapsed_s']}s", flush=True)
    return 0 if len(ok) == len(months) else 1


if __name__ == "__main__":
    raise SystemExit(main())
