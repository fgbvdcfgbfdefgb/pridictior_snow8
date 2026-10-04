"""
Render the replay as a 30 fps animation.

Layout
------
    +---------------------------------------------------+
    |  price tape (solid) + each model's 25-min forecast |
    |  (dotted, projected to the right of "now")         |
    +------------------------+--------------------------+
    |  live accuracy bars    |  rolling error / history  |
    +------------------------+--------------------------+

Frames are produced by mutating artists on a single reused Figure rather than
redrawing it -- matplotlib's draw path is the bottleneck here, and this keeps a
4 000-frame render to minutes instead of an hour.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

BG = "#0b0f17"
FG = "#e8edf7"
GRID = "#1e2735"
ACTUAL = "#f5f7fa"
PALETTE = ["#4ea8ff", "#ff8a4e", "#54e08a", "#d78bff", "#ffd24e", "#ff6f91"]


def _fmt_price(v, _p=None) -> str:
    return f"{v:,.0f}"


class ReplayAnimator:
    def __init__(self, res: Dict[str, np.ndarray], sc: Dict[str, np.ndarray],
                 lookback: int = 7200, fps: int = 30, dpi: int = 100,
                 figsize=(16, 9), title: str = ""):
        self.res, self.sc = res, sc
        self.lookback = lookback
        self.fps = fps
        self.H = int(res["horizon"][0])
        self.names = [str(x) for x in res["model_names"]]
        self.M = len(self.names)
        self.qs = list(res["quantiles"])
        self.qi = self.qs.index(0.5) if 0.5 in self.qs else len(self.qs) // 2

        self.tape_t = res["tape_t"].astype(np.int64)
        self.tape_p = res["tape_p"].astype(np.float64)
        self.times = res["times"].astype(np.int64)
        self.price = res["price"].astype(np.float64)
        self.preds = res["preds"].astype(np.float32)

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
            "text.color": FG, "axes.labelcolor": FG, "xtick.color": "#8b97ab",
            "ytick.color": "#8b97ab", "axes.edgecolor": GRID, "font.size": 11,
            "axes.titlesize": 13, "grid.color": GRID, "figure.dpi": dpi,
        })
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0],
                                   width_ratios=[1.0, 1.25],
                                   hspace=0.30, wspace=0.18,
                                   left=0.055, right=0.985, top=0.90, bottom=0.085)
        self.ax = self.fig.add_subplot(gs[0, :])
        self.ax_bar = self.fig.add_subplot(gs[1, 0])
        self.ax_err = self.fig.add_subplot(gs[1, 1])
        self._build(title)

    # ------------------------------------------------------------------ setup
    def _build(self, title: str) -> None:
        ax = self.ax
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_price))
        ax.set_ylabel("BTCUSDT")
        (self.l_actual,) = ax.plot([], [], color=ACTUAL, lw=1.7, zorder=6,
                                   label="Actual price")
        # the stretch of real tape the current forecast covers, drawn faint
        (self.l_future,) = ax.plot([], [], color=ACTUAL, lw=1.1, alpha=0.30,
                                   zorder=4, label="Actual (next 25 min)")
        self.l_pred = []
        for i, nm in enumerate(self.names):
            (ln,) = ax.plot([], [], color=PALETTE[i % len(PALETTE)], lw=1.9,
                            ls=(0, (2.2, 2.0)), zorder=7, label=f"{nm} forecast")
            self.l_pred.append(ln)
        self.band = None
        self.vline = ax.axvline(0, color="#64748b", lw=1.0, ls="--", alpha=0.8, zorder=5)
        # faint wash over the forecast region so "past" and "future" read apart
        self.fzone = ax.axvspan(0, self.H / 60.0 * 1.02, color="#4ea8ff",
                                alpha=0.045, lw=0, zorder=1)
        self.now_dot = ax.scatter([], [], s=34, color=ACTUAL, zorder=9,
                                  edgecolors=BG, linewidths=1.0)
        ax.legend(loc="upper left", ncols=min(self.M + 2, 4), framealpha=0.0,
                  fontsize=9.5, handlelength=2.4)
        self.title = ax.set_title(title, loc="left", pad=12, fontweight="bold")
        self.sub = ax.text(0.995, 1.022, "", transform=ax.transAxes, ha="right",
                           va="bottom", fontsize=11, color="#9fb0c9")

        # accuracy bars
        axb = self.ax_bar
        axb.set_title("Live accuracy  ·  100·(1−nRMSE) vs real tape",
                      loc="left", fontsize=11, color="#9fb0c9")
        y = np.arange(self.M)
        self.bars = axb.barh(y, np.zeros(self.M),
                             color=[PALETTE[i % len(PALETTE)] for i in range(self.M)],
                             height=0.62, alpha=0.9)
        axb.set_yticks(y, self.names, fontsize=10)
        axb.set_xlim(0, 100)
        axb.invert_yaxis()
        axb.grid(True, axis="x", alpha=0.22, linewidth=0.6)
        axb.set_xlabel("accuracy %")
        self.bar_txt = [axb.text(1.5, i, "", va="center", ha="left", fontsize=9.5,
                                 color=BG, fontweight="bold") for i in range(self.M)]

        # rolling error history
        axe = self.ax_err
        axe.set_title("Directional hit-rate, 25-min move  (rolling)",
                      loc="left", fontsize=11, color="#9fb0c9")
        axe.grid(True, alpha=0.22, linewidth=0.6)
        axe.set_ylim(0, 100)
        axe.set_ylabel("%")
        axe.axhline(50, color="#64748b", lw=0.9, ls=":", alpha=0.9)
        self.l_hit = []
        for i in range(self.M):
            (ln,) = axe.plot([], [], color=PALETTE[i % len(PALETTE)], lw=1.5)
            self.l_hit.append(ln)

    # ----------------------------------------------------------------- frames
    def n_frames(self, frame_stride: int) -> int:
        return len(self.times[::frame_stride])

    def _draw(self, k: int) -> None:
        t_now = int(self.times[k])
        p_now = float(self.price[k])

        lo = t_now - self.lookback
        m = (self.tape_t >= lo) & (self.tape_t <= t_now)
        if not m.any():                       # not enough history padded in
            m = self.tape_t <= t_now
            if not m.any():
                m = np.zeros_like(m); m[0] = True
        xs = (self.tape_t[m] - t_now) / 60.0
        self.l_actual.set_data(xs, self.tape_p[m])

        mf = (self.tape_t > t_now) & (self.tape_t <= t_now + self.H)
        self.l_future.set_data((self.tape_t[mf] - t_now) / 60.0, self.tape_p[mf])

        hx = np.arange(1, self.H + 1) / 60.0
        ymin, ymax = self.tape_p[m].min(), self.tape_p[m].max()
        for i in range(self.M):
            path = self.preds[i, k, self.qi].astype(np.float64)
            yy = p_now * np.exp(np.clip(path, -0.5, 0.5))
            self.l_pred[i].set_data(hx, yy)
            ymin, ymax = min(ymin, yy.min()), max(ymax, yy.max())

        if self.band is not None:
            self.band.remove()
            self.band = None
        if len(self.qs) >= 3:
            lo_p = p_now * np.exp(np.clip(self.preds[:, k, 0].astype(np.float64), -.5, .5))
            hi_p = p_now * np.exp(np.clip(self.preds[:, k, -1].astype(np.float64), -.5, .5))
            # average the members' bands rather than taking their union: the
            # union is dominated by whichever model is least confident and
            # renders as an opaque blob that hides the forecasts themselves
            lo_b, hi_b = lo_p.mean(axis=0), hi_p.mean(axis=0)
            self.band = self.ax.fill_between(hx, lo_b, hi_b, color="#4ea8ff",
                                             alpha=0.09, lw=0, zorder=2)
            ymin, ymax = min(ymin, lo_b.min()), max(ymax, hi_b.max())

        pad = max((ymax - ymin) * 0.10, p_now * 2e-4)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_xlim(-self.lookback / 60.0, self.H / 60.0 * 1.02)
        self.ax.set_xlabel("minutes from now   (0 = now, right of 0 = forecast)")
        self.vline.set_xdata([0, 0])
        self.now_dot.set_offsets([[0.0, p_now]])

        ts = datetime.fromtimestamp(t_now, timezone.utc)
        self.sub.set_text(f"{ts:%Y-%m-%d  %H:%M:%S} UTC     "
                          f"BTC {p_now:,.2f}     forecast horizon 25 min")

        acc = self.sc["roll_acc"][:, k]
        hit = self.sc["roll_hit"][:, k] * 100.0
        for i, b in enumerate(self.bars):
            b.set_width(float(acc[i]))
            self.bar_txt[i].set_text(f"{acc[i]:5.1f}%   hit {hit[i]:4.1f}%")

        kk = slice(0, k + 1)
        tx = (self.times[kk] - self.times[0]) / 3600.0
        for i in range(self.M):
            self.l_hit[i].set_data(tx, self.sc["roll_hit"][i, kk] * 100.0)
        self.ax_err.set_xlim(0, max((self.times[-1] - self.times[0]) / 3600.0, 1e-6))
        self.ax_err.set_xlabel("hours into session")

    def render(self, path: str | Path, frame_stride: int = 1,
               bitrate: str = "8M", quality: int = 8) -> Path:
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        idx = list(range(0, len(self.times), frame_stride))
        writer = imageio.get_writer(
            str(path), fps=self.fps, codec="libx264", quality=quality,
            macro_block_size=8, pixelformat="yuv420p")
        try:
            for n, k in enumerate(idx):
                self._draw(k)
                self.fig.canvas.draw()
                buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(np.ascontiguousarray(buf))
                if n % 100 == 0:
                    print(f"  frame {n}/{len(idx)}", flush=True)
        finally:
            writer.close()
            plt.close(self.fig)
        return path
