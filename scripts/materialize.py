#!/usr/bin/env python3
"""
Turn ``.btcz`` shards into a memmap ``MarketStore``: raw columns plus the full
precomputed Market Analyser feature matrix.

This is the "store everything" step.  It is embarrassingly parallel and entirely
offline, so on the Snowflake node (48 vCPU) it saturates the box and then
training never pays for decompression or feature maths again -- the GPUs read
pages, not zstd frames.

    python scripts/materialize.py --shards data/shards --out /tmp/btcstore --workers 16

Sizing for the full 2020-2026 history (~2.1e8 seconds):
    raw columns      ~3.4 GB   (logp float64 dominates)
    features float16 ~24  GB
    features float32 ~49  GB
Both fit comfortably in the 100 GB page cache, which is the point.

Feature rows need ``WARMUP`` seconds of context behind them, so each chunk is
computed with an overlapping prefix that is then discarded.  Chunk boundaries
are therefore invisible in the output -- ``--verify`` checks exactly that.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.data.btcz import PRICE_SCALE  # noqa: E402
from btcpred.data.dataset import RAW_COLUMNS, MarketStore, ShardIndex, utc  # noqa: E402
from btcpred.data.features import WARMUP, compute_block, feature_names  # noqa: E402


def _raw_from_index(index: ShardIndex, t_start: int, t_stop: int):
    """Decode shards and rebuild the physical quantities compute_block wants."""
    d = index.load_range(t_start, t_stop)
    logp = d["logp"]
    close = np.exp(logp)
    high = np.exp(logp + d["hi_off"].astype(np.float64))
    low = np.exp(logp + d["lo_off"].astype(np.float64))
    vol = np.expm1(d["lv"].astype(np.float64))
    ntr = np.expm1(d["lntr"].astype(np.float64))
    takerfrac = d["imb"].astype(np.float64) / 2.0 + 0.5
    qvol = vol * close                      # VWAP-consistent quote volume
    return d, close, high, low, vol, qvol, ntr, takerfrac


def _worker(task) -> dict:
    (shard_dir, out_root, a, b, store_t0, feat_dtype) = task
    t0 = time.time()
    index = ShardIndex(shard_dir)
    store = MarketStore(out_root, mode="r+")

    ctx = min(WARMUP, a - index.t0)         # as much warm-up as exists
    lo = a - ctx
    d, close, high, low, vol, qvol, ntr, tf = _raw_from_index(index, lo, b)

    feats = compute_block(close=close, high=high, low=low, vol=vol, qvol=qvol,
                          ntrades=ntr, takerfrac=tf, valid=d["valid"], t0=lo)

    s, e = a - store_t0, b - store_t0
    for name in RAW_COLUMNS:
        store.cols[name][s:e] = d[name][ctx:]
    store.features[s:e] = feats[ctx:].astype(np.dtype(feat_dtype))
    for m in list(store.cols.values()):
        m.flush()
    store.features.flush()
    del store, index, d, feats
    return {"a": a, "b": b, "sec": b - a, "warm": ctx, "s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--out", default="store")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk", type=int, default=1_000_000, help="seconds per task")
    ap.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: all)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD exclusive")
    ap.add_argument("--verify", action="store_true",
                    help="recompute a window across a chunk seam and compare")
    a = ap.parse_args()

    index = ShardIndex(a.shards)
    print(index, flush=True)
    from datetime import datetime, timezone
    to_ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())
    t_start = max(index.t0, to_ts(a.start)) if a.start else index.t0
    t_stop = min(index.t_end + 1, to_ts(a.end)) if a.end else index.t_end + 1
    n = t_stop - t_start
    names = feature_names()
    print(f"materialising {utc(t_start)} .. {utc(t_stop)}  ({n:,} s, "
          f"{len(names)} features, {a.feature_dtype})", flush=True)

    MarketStore.create(a.out, t0=t_start, n=n, feature_names=names,
                       feature_dtype=a.feature_dtype)

    tasks = []
    for s in range(t_start, t_stop, a.chunk):
        tasks.append((a.shards, a.out, s, min(s + a.chunk, t_stop), t_start,
                      a.feature_dtype))
    print(f"{len(tasks)} chunks on {a.workers} workers", flush=True)

    t0 = time.time()
    done = 0
    if a.workers == 1:
        for t in tasks:
            r = _worker(t)
            done += 1
            print(f"  [{done}/{len(tasks)}] {utc(r['a'])} {r['s']}s", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(_worker, tasks):
                done += 1
                print(f"  [{done}/{len(tasks)}] {utc(r['a'])} {r['s']}s", flush=True)

    store = MarketStore(a.out)
    total = sum((Path(a.out) / f"{c}.bin").stat().st_size for c in RAW_COLUMNS)
    total += (Path(a.out) / "features.bin").stat().st_size
    print(f"done in {(time.time()-t0)/60:.1f} min, {total/1e9:.2f} GB", flush=True)
    print(store, flush=True)

    if a.verify and len(tasks) > 1:
        seam = tasks[1][2]
        w = 4096
        ref_lo = seam - WARMUP - w
        d, close, high, low, vol, qvol, ntr, tf = _raw_from_index(index, ref_lo, seam + w)
        ref = compute_block(close=close, high=high, low=low, vol=vol, qvol=qvol,
                            ntrades=ntr, takerfrac=tf, valid=d["valid"], t0=ref_lo)[WARMUP:]
        got = np.asarray(store.features[seam - w - store.t0: seam + w - store.t0],
                         dtype=np.float32)
        err = np.abs(ref.astype(np.float32) - got).max()
        tol = 2e-2 if a.feature_dtype == "float16" else 1e-4
        print(f"seam check at {utc(seam)}: max abs diff = {err:.2e} "
              f"(tol {tol}) -> {'OK' if err < tol else 'FAIL'}", flush=True)
        if err >= tol:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
