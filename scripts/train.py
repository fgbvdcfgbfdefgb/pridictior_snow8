#!/usr/bin/env python3
"""
Train the population of Price Predictors.

The spec asks for a *different* model on each GPU rather than data-parallel
replicas of one model, so this is not DDP: each rank owns an independent
predictor with its own architecture, learning rate, path-basis size and
stability weight, and its own cursors into history.  Nothing is all-reduced.
The payoff comes at evaluation time, where the four disagreeing models are
combined into an ensemble whose spread is itself a useful uncertainty signal.

Launch on all 4 A10s:

    torchrun --nproc_per_node=4 scripts/train.py --store /tmp/btcstore \\
             --out runs/pop --max-hours 8 --batch 24

Single GPU (or CPU smoke test):

    python scripts/train.py --store /tmp/btcstore --out runs/dbg \\
           --max-steps 50 --batch 2 --variant agile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from btcpred.data.dataset import MarketStore, utc  # noqa: E402
from btcpred.models.predictor import (LossWeights, ModelConfig,  # noqa: E402
                                      make_config, population_configs)
from btcpred.train.online import OnlineTrainer, TrainConfig  # noqa: E402


def pick_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default="runs/pop")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--context", type=int, default=43200)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-start", default=None,
                    help="YYYY-MM-DD; restrict cursors to on/after this date")
    ap.add_argument("--train-end", default=None,
                    help="YYYY-MM-DD; hold everything from here out of training "
                         "so render_video.py can score on unseen tape")
    ap.add_argument("--variant", default=None,
                    help="force one recipe by name: base|deep|wide|agile|micro|nano")
    ap.add_argument("--n-variants", type=int, default=None,
                    help="population size (default: world size)")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = pick_device(rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)

    store = MarketStore(a.store)
    n_feat = len(store.feature_names)
    if a.variant is not None:
        cfg, lw, opt = make_config(a.variant, n_features=n_feat)
    else:
        pop = population_configs(a.n_variants or max(world, 1), n_features=n_feat)
        cfg, lw, opt = pop[rank % len(pop)]

    cfg = ModelConfig(**{**cfg.to_dict(), "context": a.context, "horizon": a.horizon})
    from datetime import datetime, timezone
    to_ts = lambda d: int(datetime.fromisoformat(d).replace(
        tzinfo=timezone.utc).timestamp())
    tcfg = TrainConfig(
        batch=a.batch, stride=a.stride, lr=opt["lr"], weight_decay=opt["weight_decay"],
        max_steps=a.max_steps, max_hours=a.max_hours, amp=a.amp, compile=a.compile,
        log_every=a.log_every, ckpt_every=a.ckpt_every, seed=a.seed,
        t_start=to_ts(a.train_start) if a.train_start else None,
        t_stop=to_ts(a.train_end) if a.train_end else None)

    out = Path(a.out) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    tr = OnlineTrainer(store, cfg, lw, tcfg, out, device, rank=rank, world=world)

    print(json.dumps({
        "rank": rank, "world": world, "device": str(device), "variant": cfg.name,
        "params_M": round(tr.model.n_params() / 1e6, 1),
        "d_model": cfg.d_model, "depth": cfg.depth, "n_basis": cfg.n_basis,
        "lr": opt["lr"], "stability_w": lw.stability, "batch": a.batch,
        "trainable_seconds": tr.sim.n_trainable_seconds,
        "train_window": f"{utc(tr.sim.lo)}..{utc(tr.sim.hi)}",
        "store": f"{utc(store.t0)}..{utc(store.t_end)}",
    }), flush=True)

    t0 = time.time()
    hist = tr.run()
    print(json.dumps({"rank": rank, "variant": cfg.name, "status": "done",
                      "minutes": round((time.time() - t0) / 60, 2),
                      "steps": tr.step,
                      "market_days_seen": round(tr.sim.market_seconds_seen() / 86400, 2),
                      **{k: round(v, 5) for k, v in hist.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
