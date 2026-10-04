"""
Price Predictor -- the GPU half of the model.

Inputs per market second *t*
    window   (B, C, 43200)  the last 12 hours of raw 1 Hz tape
    feats    (B, F)         the Market Analyser vector at *t*
    sigref   (B,)           per-second realised volatility, the natural price scale

Output
    a full **25-minute path** of cumulative log-returns, 1 500 points at 1 Hz,
    for each requested quantile, anchored so that the path is exactly 0 at h=0.

Two design decisions do the heavy lifting
-----------------------------------------
**1. Multi-resolution pyramid instead of a 43 200-token transformer.**
Attention over 43 200 steps is ~1.9e9 pairs per head -- hopeless.  A trader does
not look at 12 hours tick by tick either; they look at the last few minutes in
detail and the last twelve hours as a shape.  So the window is split into four
levels, each covering a longer span at a coarser stride, then patch-embedded.
~240 tokens total, full 12 h reach, 1-second resolution where it matters.

**2. A Karhunen-Loeve path basis instead of 1 500 free outputs.**
The network emits K coefficients of the basis
``phi_k(u) = sqrt(2) * sin((k - 1/2) * pi * u) / ((k - 1/2) * pi)``
which is *exactly* the KL expansion of Brownian motion on [0, 1].  Three
consequences, all of them things the spec asks for:

* every basis function is 0 at u=0, so the predicted path starts at the current
  price by construction -- no discontinuity at the anchor;
* the basis is ordered by smoothness and truncated at K, so high-frequency
  jitter is *structurally impossible* rather than merely penalised -- this is
  what makes the output stable enough to trade on;
* coefficients are pre-scaled by their KL standard deviation, so a unit-variance
  output from the net corresponds to a realistic price path, and the whole thing
  is scale-free across the 4k -> 126k USDT range of the dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON = 1500          # 25 minutes, in seconds
CONTEXT = 43200         # 12 hours, in seconds


# --------------------------------------------------------------------- config
@dataclass
class ModelConfig:
    name: str = "base"
    in_channels: int = 8
    n_features: int = 59
    d_model: int = 640
    depth: int = 12
    n_heads: int = 10
    d_ff_mult: int = 4
    dropout: float = 0.05
    n_basis: int = 24
    horizon: int = HORIZON
    context: int = CONTEXT
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    # (span_seconds, stride, patch) per pyramid level
    levels: Tuple[Tuple[int, int, int], ...] = (
        (1024, 1, 16),        # last ~17 min at 1 s  -> 64 tokens
        (4096, 4, 16),        # last ~68 min at 4 s  -> 64 tokens
        (16384, 16, 16),      # last ~4.5 h  at 16 s -> 64 tokens
        (43200, 64, 16),      # last 12 h    at 64 s -> 42 tokens
    )
    n_analyser_tokens: int = 4

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------------ KL basis
def kl_basis(horizon: int, n_basis: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(horizon, n_basis) Karhunen-Loeve basis of Brownian motion on (0, 1]."""
    u = torch.arange(1, horizon + 1, device=device, dtype=torch.float64) / horizon
    k = torch.arange(1, n_basis + 1, device=device, dtype=torch.float64)
    w = (k - 0.5) * math.pi
    phi = math.sqrt(2.0) * torch.sin(u[:, None] * w[None, :]) / w[None, :]
    return phi.to(dtype)


# ------------------------------------------------------------------- backbone
class PatchEmbed(nn.Module):
    """Average-pool a window level to its stride, then embed fixed-size patches."""

    def __init__(self, c_in: int, d_model: int, span: int, stride: int, patch: int):
        super().__init__()
        self.span, self.stride, self.patch = span, stride, patch
        self.n_steps = span // stride
        self.n_tokens = self.n_steps // patch
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        x = w[:, :, -self.span:]
        if self.stride > 1:
            x = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        x = x[:, :, -self.n_tokens * self.patch:]
        x = self.proj(x).transpose(1, 2)          # (B, tokens, d)
        return self.norm(x)


class Block(nn.Module):
    """Pre-LN transformer block with GEGLU feed-forward."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        hidden = d * ff_mult
        self.fc1 = nn.Linear(d, hidden * 2)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.n2(x)
        u, v = self.fc1(h).chunk(2, dim=-1)
        x = x + self.drop(self.fc2(F.gelu(u) * v))
        return x


class PricePredictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.embeds = nn.ModuleList([
            PatchEmbed(cfg.in_channels, d, span, stride, patch)
            for (span, stride, patch) in cfg.levels
        ])
        self.level_emb = nn.Parameter(torch.zeros(len(cfg.levels), d))
        n_tok = sum(e.n_tokens for e in self.embeds)
        self.n_window_tokens = n_tok

        self.analyser = nn.Sequential(
            nn.LayerNorm(cfg.n_features),
            nn.Linear(cfg.n_features, d * 2), nn.GELU(),
            nn.Linear(d * 2, d * cfg.n_analyser_tokens),
        )
        self.pos = nn.Parameter(
            torch.zeros(1, n_tok + cfg.n_analyser_tokens + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.level_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [Block(d, cfg.n_heads, cfg.d_ff_mult, cfg.dropout) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)

        self.n_q = len(cfg.quantiles)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.n_q * cfg.n_basis))

        self.register_buffer("basis", kl_basis(cfg.horizon, cfg.n_basis), persistent=False)
        self.register_buffer("h_sqrt",
                             torch.sqrt(torch.arange(1, cfg.horizon + 1,
                                                     dtype=torch.float32)),
                             persistent=False)
        try:
            self.q_index = list(cfg.quantiles).index(0.5)
        except ValueError:
            self.q_index = self.n_q // 2
        self._init_head()

    def _init_head(self) -> None:
        """Start life as a *calibrated driftless random walk*.

        Zeroing the last layer gives a zero median path -- the right prior for a
        price series, since "no change" beats any untrained guess.  But the
        quantile offsets go through a softplus, and softplus(0) = 0.693 is an
        arbitrary non-zero band.  So the biases are solved instead: pick them so
        that at h = H the predicted 10/90 band is exactly the Gaussian band of a
        random walk with the current volatility, i.e. +/- z(q) * sigma * sqrt(H).

        The model therefore begins at a sensible, well-calibrated baseline and
        only has to learn the *departure* from a random walk.
        """
        from statistics import NormalDist

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # value at h=H of a path built from an all-ones coefficient vector
        unit = float(self.basis[-1].sum())
        if abs(unit) < 1e-8:
            return
        qs = list(self.cfg.quantiles)
        z = [abs(NormalDist().inv_cdf(min(max(q, 1e-4), 1 - 1e-4))) for q in qs]
        m = self.q_index
        bias = self.head[-1].bias.view(self.n_q, self.cfg.n_basis)
        with torch.no_grad():
            for i in range(self.n_q):
                if i == m:
                    continue
                prev = z[i - 1] if i > m else z[i + 1]
                step = max(z[i] - (prev if (m < i - 1 or i + 1 < m) else 0.0), 1e-3)
                target = step / unit                       # need softplus(b)=target
                bias[i] = math.log(math.expm1(max(target, 1e-6)))

    # -- helpers ----------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, window: torch.Tensor, feats: torch.Tensor,
                sigref: torch.Tensor) -> torch.Tensor:
        """Returns cumulative log-return paths ``(B, n_quantiles, horizon)``."""
        B = window.shape[0]
        toks = []
        for i, emb in enumerate(self.embeds):
            toks.append(emb(window) + self.level_emb[i])
        a = self.analyser(feats).view(B, self.cfg.n_analyser_tokens, self.cfg.d_model)
        x = torch.cat([self.cls.expand(B, -1, -1), a] + toks, dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        z = self.norm(x[:, 0])

        coef = self.head(z).view(B, self.n_q, self.cfg.n_basis)
        # monotone quantile widths: median + softplus offsets, so q10 <= q50 <= q90
        coef = self._order_quantiles(coef)

        # scale: expected drift/diffusion over the horizon at the current vol
        scale = (sigref.clamp_min(1e-8) * math.sqrt(self.cfg.horizon)).view(B, 1, 1)
        path = torch.einsum("bqk,hk->bqh", coef, self.basis.to(coef.dtype))
        return path * scale

    def _order_quantiles(self, coef: torch.Tensor) -> torch.Tensor:
        if self.n_q == 1:
            return coef
        m = self.q_index
        out = [None] * self.n_q
        out[m] = coef[:, m]
        run = coef[:, m]
        for i in range(m + 1, self.n_q):               # upper quantiles
            run = run + F.softplus(coef[:, i])
            out[i] = run
        run = coef[:, m]
        for i in range(m - 1, -1, -1):                 # lower quantiles
            run = run - F.softplus(coef[:, i])
            out[i] = run
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def predict_prices(self, window, feats, sigref, last_price) -> torch.Tensor:
        """(B, n_quantiles, horizon) absolute prices."""
        path = self.forward(window, feats, sigref)
        return last_price.view(-1, 1, 1) * torch.exp(path.clamp(-0.5, 0.5))


# --------------------------------------------------------------------- losses
@dataclass
class LossWeights:
    quantile: float = 1.0
    huber: float = 1.0
    direction: float = 0.15
    stability: float = 0.35
    smooth: float = 0.02
    horizon_tilt: float = 0.5      # 0 = flat, 1 = fully 1/sqrt(h) weighted
    dir_horizons: Tuple[int, ...] = (60, 300, 900, 1500)


class PredictionLoss(nn.Module):
    """Realtime reward, expressed as a differentiable loss.

    Each settled second contributes
      * pinball loss over the whole 1 500-point path (calibrated uncertainty),
      * Huber on the median path, normalised by the volatility scale,
      * a directional term at trade-relevant horizons,
      * a stability term tying this second's path to the previous second's,
      * a curvature penalty on the path.
    The scalar "reward" logged during training is the negative volatility
    normalised RMSE, which is comparable across regimes and across years.
    """

    def __init__(self, cfg: ModelConfig, w: LossWeights | None = None):
        super().__init__()
        self.cfg = cfg
        self.w = w or LossWeights()
        q = torch.tensor(cfg.quantiles, dtype=torch.float32).view(1, -1, 1)
        self.register_buffer("q", q, persistent=False)
        h = torch.arange(1, cfg.horizon + 1, dtype=torch.float32)
        tilt = (1.0 / torch.sqrt(h)) ** self.w.horizon_tilt
        self.register_buffer("hw", (tilt / tilt.mean()).view(1, 1, -1), persistent=False)
        self.q_index = (list(cfg.quantiles).index(0.5)
                        if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2)

    def forward(self, pred: torch.Tensor, truth: torch.Tensor, sigref: torch.Tensor,
                prev_pred: torch.Tensor | None = None,
                prev_mask: torch.Tensor | None = None,
                shift: int = 1) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``pred`` (B,Q,H) and ``truth`` (B,H) are cumulative log-returns."""
        B, Q, H = pred.shape
        scale = (sigref.clamp_min(1e-8) * math.sqrt(H)).view(B, 1, 1)
        t = truth.unsqueeze(1)
        err = (t - pred) / scale                                   # normalised
        pin = torch.maximum(self.q * err, (self.q - 1.0) * err)
        l_q = (pin * self.hw).mean()

        med = pred[:, self.q_index]
        e_med = (truth - med) / scale.squeeze(1)
        l_h = (F.huber_loss(e_med, torch.zeros_like(e_med), reduction="none",
                            delta=1.0) * self.hw.squeeze(1)).mean()

        idx = [min(h, H) - 1 for h in self.w.dir_horizons]
        d_pred, d_true = med[:, idx], truth[:, idx]
        conf = torch.tanh(d_pred / (scale.view(B, 1) * 0.5))
        l_dir = F.softplus(-conf * torch.sign(d_true) * 4.0).mean()

        # Stability: the forecast made `shift` seconds ago, re-anchored to now,
        # is what this forecast should look like.  prev_pred[k] is the previous
        # cursor's prediction for h'=k+1; the point that lands on absolute time
        # t+h is k = h-1+shift, and its anchor (absolute time t) is k = shift-1.
        l_stab = pred.new_zeros(())
        if prev_pred is not None and shift < H:
            tgt = (prev_pred[:, :, shift:]
                   - prev_pred[:, :, shift - 1:shift]).detach()
            d = ((pred[:, :, :H - shift] - tgt) / scale) ** 2
            if prev_mask is None:
                l_stab = d.mean()
            else:
                # cursors that wrapped have no meaningful predecessor: drop
                # those rows entirely rather than pulling them toward zero
                w = prev_mask.view(-1, 1, 1).to(d.dtype)
                l_stab = (d * w).sum() / w.sum().clamp_min(1.0) / (d.shape[1] * d.shape[2])

        curv = med[:, 2:] - 2 * med[:, 1:-1] + med[:, :-2]
        l_sm = ((curv / scale.squeeze(1)) ** 2).mean() * H

        total = (self.w.quantile * l_q + self.w.huber * l_h
                 + self.w.direction * l_dir + self.w.stability * l_stab
                 + self.w.smooth * l_sm)

        with torch.no_grad():
            nrmse = torch.sqrt(((truth - med) / scale.squeeze(1)).pow(2).mean())
            hit = ((torch.sign(d_pred) == torch.sign(d_true)).float().mean())
            stats = {
                "loss": float(total), "q": float(l_q), "huber": float(l_h),
                "dir": float(l_dir), "stab": float(l_stab), "smooth": float(l_sm),
                "nrmse": float(nrmse), "hit": float(hit),
                "reward": float(-nrmse + 0.5 * (hit - 0.5)),
            }
        return total, stats


# ---------------------------------------------------------------- population
#
# One recipe per GPU.  Sized for a 23 GB A10: the largest ("wide", 178 M
# parameters) needs ~2.5 GB of AdamW state plus activations, leaving plenty of
# room for batch 24-48 at bf16.  "nano"/"micro" exist so the whole pipeline can
# be smoke-tested on a laptop or CPU-only box before burning GPU hours.
RECIPES: Dict[str, Dict] = {
    "base":  dict(name="base",  d_model=640,  depth=12, n_heads=10, n_basis=24,
                  dropout=0.05, lr=3.0e-4, stab=0.35, tilt=0.50, wd=0.02),
    "deep":  dict(name="deep",  d_model=768,  depth=16, n_heads=12, n_basis=32,
                  dropout=0.10, lr=2.0e-4, stab=0.50, tilt=0.35, wd=0.05),
    "wide":  dict(name="wide",  d_model=1024, depth=10, n_heads=16, n_basis=32,
                  dropout=0.05, lr=2.5e-4, stab=0.25, tilt=0.65, wd=0.02),
    "agile": dict(name="agile", d_model=512,  depth=14, n_heads=8,  n_basis=40,
                  dropout=0.15, lr=5.0e-4, stab=0.20, tilt=0.80, wd=0.01),
    "micro": dict(name="micro", d_model=192,  depth=4,  n_heads=6,  n_basis=16,
                  dropout=0.05, lr=6.0e-4, stab=0.30, tilt=0.50, wd=0.01),
    "nano":  dict(name="nano",  d_model=96,   depth=2,  n_heads=4,  n_basis=12,
                  dropout=0.00, lr=1.0e-3, stab=0.30, tilt=0.50, wd=0.01),
}
GPU_RECIPES: Tuple[str, ...] = ("base", "deep", "wide", "agile")


def make_config(recipe: str, n_features: int, in_channels: int = 8,
                **overrides) -> Tuple[ModelConfig, LossWeights, Dict]:
    """Build one (model, loss, optimiser) triple from a named recipe."""
    if recipe not in RECIPES:
        raise KeyError(f"unknown recipe {recipe!r}; have {sorted(RECIPES)}")
    s = {**RECIPES[recipe], **overrides}
    cfg = ModelConfig(
        name=s["name"], n_features=n_features, in_channels=in_channels,
        d_model=s["d_model"], depth=s["depth"], n_heads=s["n_heads"],
        n_basis=s["n_basis"], dropout=s["dropout"],
        context=s.get("context", CONTEXT), horizon=s.get("horizon", HORIZON))
    lw = LossWeights(stability=s["stab"], horizon_tilt=s["tilt"])
    return cfg, lw, {"lr": s["lr"], "weight_decay": s["wd"]}


def population_configs(n: int, n_features: int, in_channels: int = 8
                       ) -> List[Tuple[ModelConfig, LossWeights, Dict]]:
    """One distinct (architecture, loss, optimiser) recipe per GPU.

    The spec asks for a different model per device rather than data-parallel
    replicas of one model, so each rank trains an independent predictor and the
    ensemble is formed at evaluation time.  Variants differ in capacity, in how
    smooth the path basis forces them to be, and in how hard the stability term
    bites -- that spread is what makes the ensemble worth more than its members.
    """
    out = []
    k = len(GPU_RECIPES)
    for i in range(n):
        key = GPU_RECIPES[i % k]
        tag = key if i < k else f"{key}{i // k}"
        cfg, lw, opt = make_config(key, n_features, in_channels, name=tag)
        out.append((cfg, lw, opt))
    return out
