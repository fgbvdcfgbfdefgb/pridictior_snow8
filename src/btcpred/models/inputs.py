"""
GPU-side normalisation of the raw 12-hour window.

The simulator hands over channels that are already *differences* of log prices
(computed in float64 before being cast down -- see data/dataset.py), so nothing
here has to take a logarithm of a large number.  All that remains is to divide
by the current volatility scale, which is what makes a 2020 window and a 2026
window statistically comparable to the network.

Input  (B, 7, L):  rel_logp, hi_off, lo_off, lv, lntr, imb, valid
Output (B, 8, L):

0  path   rel_logp / (sigref * sqrt(L))     shape of the last 12 h
1  ret    d(rel_logp) / sigref              per-second return, unit variance
2  hi     hi_off / sigref                   upper wick
3  lo     lo_off / sigref                   lower wick
4  act    lv  - mean(lv)                    volume anomaly
5  ntr    lntr - mean(lntr)                 trade-count anomaly
6  imb    order-flow imbalance, already in [-1, 1]
7  valid  1 real bar, 0 forward-filled
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

CLIP = 10.0
N_MODEL_CHANNELS = 8


class WindowNormalizer(nn.Module):
    """Raw (B, 7, L) tape -> bounded, scale-free (B, 8, L) model input."""

    def __init__(self, clip: float = CLIP):
        super().__init__()
        self.clip = clip

    def forward(self, raw: torch.Tensor, sigref: torch.Tensor) -> torch.Tensor:
        rel, hi_off, lo_off = raw[:, 0], raw[:, 1], raw[:, 2]
        lv, lntr, imb, valid = raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]
        L = rel.shape[-1]
        s = sigref.clamp_min(1e-8).unsqueeze(-1)

        path = rel / (s * math.sqrt(L))
        ret = torch.zeros_like(rel)
        ret[:, 1:] = (rel[:, 1:] - rel[:, :-1]) / s
        hi = hi_off / s
        lo = lo_off / s
        act = lv - lv.mean(dim=-1, keepdim=True)
        ntr = lntr - lntr.mean(dim=-1, keepdim=True)

        x = torch.stack([path, ret, hi, lo, act, ntr, imb, valid], dim=1)
        return torch.nan_to_num(x, nan=0.0, posinf=self.clip, neginf=-self.clip) \
                    .clamp(-self.clip, self.clip)
