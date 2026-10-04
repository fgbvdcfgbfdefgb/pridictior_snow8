"""
Online, epoch-free trainer.

The spec is explicit: there are no epochs.  Training is a single pass through
market time with a per-second reward.  Concretely:

* ``B`` cursors are scattered across the whole history and all advance one
  market second per optimiser step;
* at each second the predictor emits a 25-minute path;
* the reward for that second is computed against what the market **actually**
  did over the following 1 500 seconds, read from the stored (non-simulated)
  tape;
* the loss is backpropagated immediately and the cursors move on.

The one subtlety worth stating plainly: a prediction made at *t* cannot be
scored until *t + 1500*.  Rather than hold 1 500 autograd graphs alive, the
learning cursor runs 25 minutes behind the simulator's "now", which is exactly
how an online learner behaves in production -- it learns from settled
predictions while serving unsettled ones.  ``MarketSimulator`` enforces that
split: ``observe()`` only ever reads the past, ``settle()`` is the only call
that touches the future.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import MarketStore
from ..models.inputs import WindowNormalizer
from ..models.predictor import LossWeights, ModelConfig, PredictionLoss, PricePredictor
from ..sim.simulator import BatchedSimulator


@dataclass
class TrainConfig:
    batch: int = 24
    stride: int = 1                 # market seconds per optimiser step
    lr: float = 3e-4
    weight_decay: float = 0.02
    warmup_steps: int = 500
    max_steps: int = 1_000_000
    max_hours: float = 6.0
    grad_clip: float = 1.0
    log_every: int = 25
    ckpt_every: int = 2000
    ema_decay: float = 0.999
    amp: str = "bf16"               # bf16 | fp16 | off
    compile: bool = False
    prefetch: int = 3
    seed: int = 0
    t_start: Optional[int] = None
    t_stop: Optional[int] = None


class _Prefetcher(threading.Thread):
    """Gather the next batch of windows on CPU while the GPU is busy.

    The 12-hour window is the expensive part of a step on the host side
    (B x 7 x 43200 float32 ~ 29 MB at batch 24), so it overlaps with compute on
    a worker thread.  numpy releases the GIL inside the memmap copy, so this
    genuinely parallelises.
    """

    def __init__(self, sim: BatchedSimulator, depth: int = 3):
        super().__init__(daemon=True)
        self.sim = sim
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            obs = self.sim.observe()
            item = (obs.t.copy(), obs.window.copy(), obs.feats.copy(),
                    obs.sigref.copy(), obs.last_price.copy(), self.sim.settle())
            wrapped_before = self.sim.cursors.copy()
            self.sim.advance()
            cont = (self.sim.cursors == wrapped_before + self.sim.stride)
            try:
                self.q.put((item, cont), timeout=5)
            except queue.Full:
                continue

    def close(self) -> None:
        self.stop_flag.set()


class OnlineTrainer:
    def __init__(self, store: MarketStore, model_cfg: ModelConfig,
                 loss_w: LossWeights, tcfg: TrainConfig, out_dir: Path,
                 device: torch.device, rank: int = 0, world: int = 1):
        self.store = store
        self.cfg = model_cfg
        self.tcfg = tcfg
        self.device = device
        self.rank = rank
        self.world = world
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(tcfg.seed + rank)
        np.random.seed(tcfg.seed + rank)

        self.model = PricePredictor(model_cfg).to(device)
        self.norm = WindowNormalizer().to(device)
        self.crit = PredictionLoss(model_cfg, loss_w).to(device)
        if tcfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith("pos") or n.endswith("cls")
             else decay).append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": tcfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=tcfg.lr, betas=(0.9, 0.95), eps=1e-8,
            fused=(device.type == "cuda"))

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "off": None}[tcfg.amp]
        self.scaler = torch.amp.GradScaler(
            device.type, enabled=(tcfg.amp == "fp16" and device.type == "cuda"))

        self.sim = BatchedSimulator(
            store, batch=tcfg.batch, context=model_cfg.context,
            horizon=model_cfg.horizon, t_start=tcfg.t_start, t_stop=tcfg.t_stop,
            stride=tcfg.stride, seed=tcfg.seed + 977 * rank)

        self.ema = {k: v.detach().clone().float()
                    for k, v in self.model.state_dict().items()
                    if v.dtype.is_floating_point}
        self.step = 0
        self.prev_pred: Optional[torch.Tensor] = None
        self.hist: Dict[str, float] = {}
        self.metrics_path = self.out / "metrics.jsonl"
        self.t_begin = time.time()

    # -- schedule ---------------------------------------------------------
    def _lr(self) -> float:
        w = self.tcfg.warmup_steps
        if self.step < w:
            return self.tcfg.lr * (self.step + 1) / w
        # slow cosine floor; the stream never ends, so never decay to zero
        prog = min(1.0, (self.step - w) / max(self.tcfg.max_steps - w, 1))
        return self.tcfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    @torch.no_grad()
    def _update_ema(self) -> None:
        d = self.tcfg.ema_decay
        sd = self.model.state_dict()
        for k, v in self.ema.items():
            v.mul_(d).add_(sd[k].detach().float(), alpha=1 - d)

    # -- one market second ------------------------------------------------
    def train_step(self, item, cont: np.ndarray) -> Dict[str, float]:
        t_arr, win, feats, sig, price, truth = item
        dev = self.device
        win_t = torch.from_numpy(win).to(dev, non_blocking=True)
        feats_t = torch.from_numpy(feats).to(dev, non_blocking=True)
        sig_t = torch.from_numpy(sig).to(dev, non_blocking=True)
        truth_t = torch.from_numpy(truth).to(dev, non_blocking=True)
        cont_t = torch.from_numpy(cont.astype(np.float32)).to(dev, non_blocking=True)

        for g in self.opt.param_groups:
            g["lr"] = self._lr()

        ctx = (torch.autocast(dev.type, dtype=self.amp_dtype)
               if self.amp_dtype is not None else torch.autocast(dev.type, enabled=False))
        with ctx:
            x = self.norm(win_t, sig_t)
            pred = self.model(x, feats_t, sig_t)
            prev = None
            if self.prev_pred is not None and self.prev_pred.shape == pred.shape:
                prev = self.prev_pred * cont_t.view(-1, 1, 1)
                pred_masked = pred * cont_t.view(-1, 1, 1)
            else:
                pred_masked = pred
            loss, stats = self.crit(pred.float(), truth_t.float(), sig_t.float(),
                                    prev_pred=None if prev is None else prev.float())

        self.opt.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.opt.step()

        self.prev_pred = pred.detach()
        self._update_ema()
        stats["grad_norm"] = float(gn)
        stats["lr"] = self._lr()
        return stats

    # -- main loop --------------------------------------------------------
    def run(self) -> Dict[str, float]:
        pf = _Prefetcher(self.sim, depth=self.tcfg.prefetch)
        pf.start()
        t_log = time.time()
        seen_at_log = 0
        try:
            while self.step < self.tcfg.max_steps:
                if (time.time() - self.t_begin) / 3600.0 > self.tcfg.max_hours:
                    break
                try:
                    item, cont = pf.q.get(timeout=120)
                except queue.Empty:
                    break
                stats = self.train_step(item, cont)
                self.step += 1

                for k, v in stats.items():
                    self.hist[k] = v if k not in self.hist else 0.98 * self.hist[k] + 0.02 * v

                if self.step % self.tcfg.log_every == 0:
                    now = time.time()
                    sps = self.tcfg.log_every / max(now - t_log, 1e-6)
                    seen = self.sim.market_seconds_seen()
                    rec = {
                        "rank": self.rank, "variant": self.cfg.name, "step": self.step,
                        "steps_per_s": round(sps, 2),
                        "market_s_per_s": round((seen - seen_at_log) / max(now - t_log, 1e-6)),
                        "market_days_seen": round(seen / 86400.0, 2),
                        "elapsed_min": round((now - self.t_begin) / 60.0, 2),
                        **{k: round(float(v), 5) for k, v in self.hist.items()},
                    }
                    with open(self.metrics_path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    if self.rank == 0 or self.world == 1:
                        print(json.dumps(rec), flush=True)
                    t_log, seen_at_log = now, seen

                if self.step % self.tcfg.ckpt_every == 0:
                    self.save("latest")
        finally:
            pf.close()
            self.save("final")
        return {k: float(v) for k, v in self.hist.items()}

    def save(self, tag: str) -> Path:
        p = self.out / f"{self.cfg.name}_{tag}.pt"
        sd = self.model.state_dict()
        torch.save({
            "model": {k: v.cpu() for k, v in sd.items()},
            "ema": {k: v.cpu() for k, v in self.ema.items()},
            "model_cfg": self.cfg.to_dict(),
            "train_cfg": asdict(self.tcfg),
            "step": self.step,
            "feature_names": self.store.feature_names,
            "hist": self.hist,
        }, p)
        return p
