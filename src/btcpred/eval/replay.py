"""
Replay a chosen market day through every trained predictor and score each one
against what the market actually did.

The scoring rule is the one from the spec: the path predicted at second *t* is
compared with the **real stored tape** over ``[t+1, t+1500]`` -- not with
anything the simulator generated.  A prediction is only counted once it has
fully settled, i.e. 25 minutes of real future exist for it.

Output is a single ``.npz`` that the renderer turns into video, so replay (GPU)
and rendering (CPU, matplotlib) can run on different machines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import MarketStore, utc
from ..models.inputs import WindowNormalizer
from ..models.predictor import ModelConfig, PricePredictor
from ..sim.simulator import N_WINDOW_CHANNELS, WINDOW_COLUMNS, MarketSimulator


@dataclass
class LoadedModel:
    name: str
    model: PricePredictor
    cfg: ModelConfig


def load_population(ckpts: Sequence[str | Path], device: torch.device,
                    use_ema: bool = True) -> List[LoadedModel]:
    out: List[LoadedModel] = []
    for p in ckpts:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**blob["model_cfg"])
        m = PricePredictor(cfg)
        sd = blob["ema"] if (use_ema and blob.get("ema")) else blob["model"]
        missing = m.load_state_dict({k: v for k, v in sd.items()}, strict=False)
        if missing.missing_keys:
            # EMA only tracks floating-point tensors; buffers come from init
            m.load_state_dict(blob["model"], strict=False)
        m.to(device).eval()
        out.append(LoadedModel(name=cfg.name, model=m, cfg=cfg))
    return out


def _gather_windows(store: MarketStore, times: np.ndarray, context: int) -> np.ndarray:
    lp = store.cols["logp"]
    win = np.empty((len(times), N_WINDOW_CHANNELS, context), dtype=np.float32)
    for j, t in enumerate(times):
        i = int(t) - store.t0
        a, b = i - context + 1, i + 1
        win[j, 0] = (np.asarray(lp[a:b], dtype=np.float64) - float(lp[i])).astype(np.float32)
        for c, name in enumerate(WINDOW_COLUMNS[1:], start=1):
            win[j, c] = store.cols[name][a:b]
    return win


@torch.no_grad()
def replay_day(store: MarketStore, models: List[LoadedModel], t_start: int,
               t_stop: int, device: torch.device, pred_stride: int = 20,
               batch: int = 32, horizon: int = 1500, context: int = 43200,
               tape_pad: int = 7200, progress: bool = True) -> Dict[str, np.ndarray]:
    """Predict every ``pred_stride`` seconds over ``[t_start, t_stop)``."""
    lo = max(t_start, store.t0 + context + 1)
    hi = min(t_stop, store.t_end - horizon + 1)
    times = np.arange(lo, hi, pred_stride, dtype=np.int64)
    if len(times) == 0:
        raise ValueError("replay window is empty after clamping to the store")

    norm = WindowNormalizer().to(device)
    sig_idx = store.feature_names.index("logvol_300s")
    n_q = len(models[0].cfg.quantiles)

    preds = np.zeros((len(models), len(times), n_q, horizon), dtype=np.float16)
    sig_all = np.zeros(len(times), dtype=np.float32)

    for s in range(0, len(times), batch):
        tb = times[s:s + batch]
        win = torch.from_numpy(_gather_windows(store, tb, context)).to(device)
        idx = (tb - store.t0).astype(np.int64)
        feats = torch.from_numpy(
            np.asarray(store.features[idx], dtype=np.float32)).to(device)
        sig = torch.from_numpy(
            np.exp(np.asarray(store.features[idx], dtype=np.float32)[:, sig_idx] - 9.0)
            .astype(np.float32)).to(device)
        sig_all[s:s + len(tb)] = sig.cpu().numpy()
        x = norm(win, sig)
        for mi, lm in enumerate(models):
            p = lm.model(x[:, :lm.cfg.in_channels], feats, sig)
            preds[mi, s:s + len(tb)] = p.float().cpu().numpy().astype(np.float16)
        if progress and (s // max(batch, 1)) % 20 == 0:
            print(f"  replay {s}/{len(times)}", flush=True)

    # ground truth: the real stored tape
    lp = store.cols["logp"]
    i0 = (times - store.t0).astype(np.int64)
    truth = np.empty((len(times), horizon), dtype=np.float32)
    for j, i in enumerate(i0):
        truth[j] = (np.asarray(lp[i + 1:i + 1 + horizon], dtype=np.float64)
                    - float(lp[i])).astype(np.float32)

    price = np.asarray(store.cols["close"][i0], dtype=np.float32)
    # The actual traded price over the whole rendered span at 1 Hz, padded
    # *backwards* so the very first frame already has scrolling history behind
    # it, and forwards by one horizon so the last frame's forecast can be
    # compared against real tape.  (These are store indices, not timestamps --
    # mixing the two silently produces an empty span.)
    i_lo = max(0, lo - tape_pad - store.t0)
    i_hi = min(hi + horizon - store.t0, store.n)
    span_i = np.arange(i_lo, i_hi, dtype=np.int64)
    tape_t = (span_i + store.t0).astype(np.int64)
    tape_p = np.asarray(store.cols["close"][span_i], dtype=np.float32)

    return {
        "times": times, "preds": preds, "truth": truth, "price": price,
        "sigref": sig_all, "tape_t": tape_t, "tape_p": tape_p,
        "model_names": np.array([m.name for m in models]),
        "quantiles": np.array(models[0].cfg.quantiles, dtype=np.float32),
        "horizon": np.array([horizon]), "pred_stride": np.array([pred_stride]),
    }


def score(res: Dict[str, np.ndarray], window: int = 60) -> Dict[str, np.ndarray]:
    """Per-model accuracy, settled against real data.

    ``dir_hit``  sign of the 25-minute move, predicted vs realised
    ``nrmse``    path RMSE normalised by the volatility scale sigma*sqrt(H)
    ``acc``      100 * (1 - min(1, nrmse)), a bounded "how close was the path"
    ``jump``     median absolute change of the 25-min point forecast between
                 consecutive predictions, in sigma units -- the stability metric
    """
    preds = res["preds"].astype(np.float32)
    truth = res["truth"].astype(np.float32)
    sig = res["sigref"].astype(np.float32)
    H = int(res["horizon"][0])
    qs = list(res["quantiles"])
    qi = qs.index(0.5) if 0.5 in qs else len(qs) // 2
    scale = np.maximum(sig * math.sqrt(H), 1e-9)

    med = preds[:, :, qi, :]                                   # (M, T, H)
    err = (med - truth[None]) / scale[None, :, None]
    nrmse = np.sqrt((err ** 2).mean(axis=2))                   # (M, T)
    hit = (np.sign(med[:, :, -1]) == np.sign(truth[None, :, -1])).astype(np.float32)
    endpoint = np.abs(med[:, :, -1] - truth[None, :, -1]) / scale[None]

    jump = np.zeros_like(nrmse)
    jump[:, 1:] = np.abs(med[:, 1:, -1] - med[:, :-1, -1]) / scale[None, 1:]

    def roll(x, w):
        k = np.ones(w, dtype=np.float32) / w
        return np.stack([np.convolve(r, k, mode="full")[:len(r)] for r in x])

    return {
        "nrmse": nrmse, "dir_hit": hit, "endpoint": endpoint, "jump": jump,
        "roll_hit": roll(hit, window), "roll_nrmse": roll(nrmse, window),
        "acc": 100.0 * np.clip(1.0 - nrmse, 0.0, 1.0),
        "roll_acc": 100.0 * np.clip(1.0 - roll(nrmse, window), 0.0, 1.0),
    }


def summary(res: Dict[str, np.ndarray], sc: Dict[str, np.ndarray]) -> List[Dict]:
    out = []
    for i, name in enumerate(res["model_names"]):
        out.append({
            "model": str(name),
            "dir_hit_pct": round(float(sc["dir_hit"][i].mean()) * 100, 2),
            "path_nrmse": round(float(sc["nrmse"][i].mean()), 4),
            "endpoint_nrmse": round(float(sc["endpoint"][i].mean()), 4),
            "accuracy_pct": round(float(sc["acc"][i].mean()), 2),
            "stability_jump_sigma": round(float(np.median(sc["jump"][i])), 5),
        })
    return out
