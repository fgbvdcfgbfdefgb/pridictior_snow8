#!/usr/bin/env python3
"""
Pick a market day, replay it through the trained population, score every model
against the real tape and write a 30 fps MP4.

    python scripts/render_video.py --store /tmp/btcstore --ckpt-dir runs/pop \\
        --random-day --out media/replay.mp4

Timing knobs
------------
``--pred-stride``   market seconds between predictions (1 = every second)
``--frame-stride``  predictions between rendered frames
A 24 h session at ``--pred-stride 20 --frame-stride 1`` gives 4 320 frames,
i.e. 2 min 24 s of 30 fps video covering the whole day.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.data.dataset import MarketStore, utc  # noqa: E402
from btcpred.eval.replay import load_population, replay_day, score, summary  # noqa: E402
from btcpred.viz.animate import ReplayAnimator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None,
                    help="materialised store; not needed with --load-npz")
    ap.add_argument("--ckpt-dir", default="runs/pop")
    ap.add_argument("--ckpt", nargs="*", default=None)
    ap.add_argument("--out", default="media/replay.mp4")
    ap.add_argument("--day", default=None, help="YYYY-MM-DD")
    ap.add_argument("--random-day", action="store_true")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--pred-stride", type=int, default=20)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lookback-min", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--figsize", default="16x9")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--roll", type=int, default=60, help="rolling metric window")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--save-npz", default=None,
                    help="cache the replay tensors so the plot can be "
                         "re-rendered without re-running the models")
    ap.add_argument("--load-npz", default=None,
                    help="re-render from a --save-npz cache; skips model "
                         "loading and the replay entirely")
    a = ap.parse_args()
    if not a.store and not a.load_npz:
        ap.error("--store is required unless you pass --load-npz")

    if a.load_npz:
        # Pure re-render path: everything needed for the figure is in the
        # cache, so no store, no checkpoints and no forward passes.
        z = np.load(a.load_npz)
        res = {k: z[k] for k in z.files}
        res["model_names"] = [str(x) for x in res["model_names"]]
        # keep the boxed 1-element arrays exactly as replay_day emits them;
        # score() indexes them. Only the sidecar wants a plain int.
        if "pred_stride" in res:
            a.pred_stride = int(np.asarray(res["pred_stride"]).reshape(-1)[0])
        sc = score(res, window=a.roll)
        rep = summary(res, sc)
        print(json.dumps(rep, indent=2), flush=True)
        t0 = int(res["times"][0])
        t1 = int(res["times"][-1])
        _render(res, sc, rep, a, t0, t1)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = MarketStore(a.store)
    print(store, flush=True)

    ck = ([Path(p) for p in a.ckpt] if a.ckpt else
          sorted(Path(a.ckpt_dir).rglob("*_final.pt")) or
          sorted(Path(a.ckpt_dir).rglob("*_latest.pt")))
    if not ck:
        raise SystemExit(f"no checkpoints under {a.ckpt_dir}")
    print("checkpoints:", [p.name for p in ck], flush=True)
    models = load_population(ck, dev, use_ema=not a.no_ema)

    ctx = models[0].cfg.context
    hor = models[0].cfg.horizon
    first = store.t0 + ctx + 1
    last = store.t_end - hor - int(a.hours * 3600)
    if last <= first:
        raise SystemExit("store too short for the requested session length")

    if a.day:
        d = datetime.fromisoformat(a.day).replace(tzinfo=timezone.utc)
        t0 = int(d.timestamp())
    elif a.random_day:
        rng = random.Random(a.seed)
        t0 = rng.randrange(first, last)
        t0 -= t0 % 86400                      # snap to a UTC midnight
        t0 = max(first, min(t0, last))
    else:
        t0 = first
    t1 = t0 + int(a.hours * 3600)
    print(f"session: {utc(t0)} .. {utc(t1)}  ({a.hours} h)", flush=True)

    res = replay_day(store, models, t0, t1, dev, pred_stride=a.pred_stride,
                     batch=a.batch, horizon=hor, context=ctx)
    sc = score(res, window=a.roll)
    rep = summary(res, sc)
    print(json.dumps(rep, indent=2), flush=True)

    if a.save_npz:
        np.savez_compressed(a.save_npz, **res)

    _render(res, sc, rep, a, t0, t1)
    return 0


def _render(res, sc, rep, a, t0, t1):
    """Draw the animation and drop the scorecard sidecar next to it."""
    w, h = (float(x) for x in a.figsize.lower().split("x"))
    day = datetime.fromtimestamp(t0, timezone.utc).strftime("%d %b %Y")
    anim = ReplayAnimator(res, sc, lookback=a.lookback_min * 60, fps=a.fps,
                          dpi=a.dpi, figsize=(w, h),
                          title=f"BTCUSDT  ·  live 25-minute forecast  ·  {day}")
    out = anim.render(a.out, frame_stride=a.frame_stride)
    nfr = len(res["times"][::a.frame_stride])
    print(f"wrote {out}  ({nfr} frames, {nfr / a.fps:.1f}s at {a.fps}fps)", flush=True)

    Path(str(out) + ".json").write_text(json.dumps(
        {"session_start": utc(t0), "session_end": utc(t1),
         "pred_stride_s": a.pred_stride, "frames": nfr, "fps": a.fps,
         "models": rep}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
