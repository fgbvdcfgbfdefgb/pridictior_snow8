#!/usr/bin/env python3
"""
End-to-end pipeline check on a synthetic market -- no dataset, no GPU, ~1 min.

Builds a small ``MarketStore`` from a random walk with realistic intraday
volatility, then drives the *real* simulator, trainer, checkpointing, scorer and
renderer over it.  If this passes, the only thing standing between you and a
real run is data and compute.

    python scripts/smoke_test.py --out /tmp/smoke

It also doubles as a regression test for the two failure modes that are easy to
introduce and hard to notice: a trainer that does not actually reduce the loss,
and a simulator that leaks the future into the model's input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.data.dataset import MarketStore  # noqa: E402
from btcpred.data.features import compute_block, feature_names  # noqa: E402
from btcpred.models.predictor import make_config  # noqa: E402
from btcpred.sim.simulator import BatchedSimulator, MarketSimulator  # noqa: E402
from btcpred.train.online import OnlineTrainer, TrainConfig  # noqa: E402


def build_synthetic_store(root: Path, n: int = 700_000, seed: int = 0) -> MarketStore:
    """A random walk with a diurnal volatility cycle and occasional jumps."""
    rng = np.random.default_rng(seed)
    t0 = 1_600_000_000
    t = np.arange(n, dtype=np.float64)
    tod = (t0 + t) % 86400 / 86400
    vol = 1.2e-5 * (1.0 + 0.8 * np.sin(2 * np.pi * tod))      # busier in US hours
    r = rng.standard_normal(n) * vol
    r += 0.25 * np.roll(r, 1)                                  # mild autocorrelation
    jump = rng.random(n) < 2e-5
    r[jump] += rng.standard_normal(int(jump.sum())) * 2e-3
    close = 20_000 * np.exp(np.cumsum(r))
    wick = np.abs(rng.standard_normal(n)) * close * 2e-5
    high, low = close + wick, close - wick
    v = np.abs(rng.standard_normal(n)) * 0.4 + 0.01
    ntr = rng.poisson(6, n).astype(np.float64) + 1
    tf = np.clip(0.5 + 0.35 * np.tanh(r / np.maximum(vol, 1e-9)) * 0.5
                 + rng.standard_normal(n) * 0.1, 0, 1)
    valid = np.ones(n, dtype=np.float64)

    names = feature_names()
    store = MarketStore.create(root, t0=t0, n=n, feature_names=names,
                               feature_dtype="float32")
    lp = np.log(close)
    store.cols["logp"][:] = lp
    store.cols["close"][:] = close.astype(np.float32)
    store.cols["hi_off"][:] = (np.log(high) - lp).astype(np.float32)
    store.cols["lo_off"][:] = (np.log(low) - lp).astype(np.float32)
    store.cols["lv"][:] = np.log1p(v).astype(np.float32)
    store.cols["lntr"][:] = np.log1p(ntr).astype(np.float32)
    store.cols["imb"][:] = ((tf - 0.5) * 2).astype(np.float32)
    store.cols["valid"][:] = 1
    store.features[:] = compute_block(
        close=close, high=high, low=low, vol=v, qvol=v * close, ntrades=ntr,
        takerfrac=tf, valid=valid, t0=t0)
    for m in store.cols.values():
        m.flush()
    store.features.flush()
    return MarketStore(root, mode="r+")   # r+ so the lookahead probe can mutate


def check_no_lookahead(store: MarketStore, context: int, horizon: int) -> None:
    """The window must be identical whether or not the future exists."""
    sim = MarketSimulator(store, store.t0, store.t_end - horizon,
                          context=context, horizon=horizon)
    t = sim.t_start + 50_000
    before = sim.observe(t).window.copy()
    saved = np.array(store.cols["logp"][t - store.t0 + 1: t - store.t0 + 500])
    store.cols["logp"][t - store.t0 + 1: t - store.t0 + 500] += 0.5   # wreck the future
    after = sim.observe(t).window.copy()
    store.cols["logp"][t - store.t0 + 1: t - store.t0 + 500] = saved
    assert np.array_equal(before, after), "simulator leaked future data into observe()"
    print("  no-lookahead ....... OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--video", action="store_true", help="also render a short mp4")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    root = Path(a.out or tempfile.mkdtemp(prefix="btcsmoke-"))
    store_dir = root / "store"
    print(f"workspace: {root}")

    t0 = time.time()
    print("building synthetic store ...", flush=True)
    store = build_synthetic_store(store_dir)
    print(f"  {store}")
    print(f"  built in {time.time()-t0:.1f}s")

    check_no_lookahead(store, a.context, a.horizon)

    sim = BatchedSimulator(store, batch=a.batch, context=a.context,
                           horizon=a.horizon, stride=1, seed=0)
    obs = sim.observe()
    assert obs.window.shape == (a.batch, 7, a.context), obs.window.shape
    assert np.isfinite(obs.window).all(), "non-finite values in the window"
    assert np.abs(obs.window[:, 0, -1]).max() < 1e-6, "window must be anchored at t"
    truth = sim.settle()
    assert truth.shape == (a.batch, a.horizon)
    print(f"  simulator .......... OK  ({sim.n_trainable_seconds:,} trainable seconds)")

    cfg, lw, opt = make_config("nano", n_features=len(store.feature_names))
    cfg = type(cfg)(**{**cfg.to_dict(), "context": a.context, "horizon": a.horizon,
                       "levels": ((512, 1, 16), (2048, 4, 16), (4096, 16, 8))})
    tcfg = TrainConfig(batch=a.batch, stride=1, lr=opt["lr"], max_steps=a.steps,
                       max_hours=0.2, warmup_steps=5, log_every=10,
                       ckpt_every=10_000, amp="off", prefetch=2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = OnlineTrainer(store, cfg, lw, tcfg, root / "run", dev)
    print(f"  model .............. {tr.model.n_params()/1e6:.2f}M params on {dev}")

    print(f"training {a.steps} steps ...", flush=True)
    t1 = time.time()
    hist = tr.run()
    dt = time.time() - t1
    rows = [json.loads(x) for x in (root / "run" / "metrics.jsonl").read_text().splitlines()]
    first, last = rows[0], rows[-1]
    print(f"  {a.steps} steps in {dt:.1f}s  ({a.steps/dt:.2f} steps/s, "
          f"{a.steps*a.batch/dt:.0f} market-s/s)")
    print(f"  loss {first['loss']:.4f} -> {last['loss']:.4f}   "
          f"nrmse {first['nrmse']:.4f} -> {last['nrmse']:.4f}")
    assert np.isfinite(last["loss"]), "loss went non-finite"
    ck = root / "run" / f"{cfg.name}_final.pt"
    assert ck.exists(), "no checkpoint written"
    print(f"  checkpoint ......... OK  ({ck.stat().st_size/1e6:.1f} MB)")

    if a.video:
        from btcpred.eval.replay import load_population, replay_day, score, summary
        from btcpred.viz.animate import ReplayAnimator
        models = load_population([ck], dev)
        t_a = sim.lo + 1000
        res = replay_day(store, models, t_a, t_a + 3600, dev, pred_stride=30,
                         batch=8, horizon=a.horizon, context=a.context, progress=False)
        sc = score(res, window=10)
        print("  scorecard:", json.dumps(summary(res, sc)))
        anim = ReplayAnimator(res, sc, lookback=1800, fps=30, dpi=70,
                              figsize=(12, 7), title="smoke test")
        out = anim.render(root / "smoke.mp4", frame_stride=1)
        print(f"  video .............. OK  ({out}, {out.stat().st_size/1e3:.0f} kB)")

    print(f"\nALL OK in {time.time()-t0:.1f}s")
    if not a.keep and a.out is None:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
