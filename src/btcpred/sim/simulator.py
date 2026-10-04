"""
Live market simulator.

Replays stored history one second at a time, exactly as if it were arriving
from a websocket.  The contract that matters:

* ``observe()`` may only ever touch seconds <= the cursor;
* ``settle()`` is the only method that looks forward, it is used by the reward
  function and the scoreboard, never by the model.

Keeping those two on separate methods is what stops lookahead bias creeping in.

``BatchedSimulator`` runs ``B`` independent cursors scattered across the whole
history so one optimiser step sees 2020 crash tape, 2021 bull tape and 2026 tape
at once.  Each cursor advances in market time; there are no epochs, the stream
just keeps running and wraps around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.dataset import MarketStore
from ..data.features import WARMUP

# Channels handed to the GPU, in this order.  Channel 0 is the log price
# *relative to the cursor* -- the subtraction happens here in float64 and only
# the small difference is cast to float32, which is what keeps 1-second returns
# exact (see the dtype note in data/dataset.py).
WINDOW_COLUMNS: Tuple[str, ...] = (
    "rel_logp", "hi_off", "lo_off", "lv", "lntr", "imb", "valid")
N_WINDOW_CHANNELS = len(WINDOW_COLUMNS)
_PASSTHROUGH = WINDOW_COLUMNS[1:]      # everything except rel_logp


@dataclass
class Observation:
    t: np.ndarray            # (B,) unix seconds of the cursor
    window: np.ndarray       # (B, C, context) raw tape up to and including t
    feats: np.ndarray        # (B, F) Market Analyser output at t
    sigref: np.ndarray       # (B,) per-second realised vol at t
    last_price: np.ndarray   # (B,) close at t


class MarketSimulator:
    """Single-cursor replay over ``[t_start, t_stop)``."""

    def __init__(self, store: MarketStore, t_start: int, t_stop: int,
                 context: int = 43200, horizon: int = 1500):
        self.store = store
        self.context = context
        self.horizon = horizon
        lo = store.t0 + context + 1
        hi = store.t_end - horizon
        self.t_start = max(t_start, lo)
        self.t_stop = min(t_stop, hi + 1)
        if self.t_start >= self.t_stop:
            raise ValueError(
                f"empty replay range: store covers [{store.t0},{store.t_end}], "
                f"need {context}s of context and {horizon}s of future")
        self.t = self.t_start
        self._sig_idx = store.feature_names.index("logvol_300s")

    def __len__(self) -> int:
        return self.t_stop - self.t_start

    def reset(self, t: Optional[int] = None) -> None:
        self.t = self.t_start if t is None else int(t)

    # -- past only ---------------------------------------------------------
    def observe(self, t: Optional[int] = None) -> Observation:
        t = self.t if t is None else int(t)
        i = self.store.idx(t)
        a, b = i - self.context + 1, i + 1
        if a < 0:
            raise IndexError("not enough context before t")
        win = np.empty((1, N_WINDOW_CHANNELS, self.context), dtype=np.float32)
        lp = self.store.cols["logp"]
        win[0, 0] = (np.asarray(lp[a:b], dtype=np.float64) - float(lp[i])).astype(np.float32)
        for c, name in enumerate(_PASSTHROUGH, start=1):
            win[0, c] = self.store.cols[name][a:b]
        feats = np.asarray(self.store.features[i], dtype=np.float32)[None, :]
        sig = float(np.exp(feats[0, self._sig_idx] - 9.0))
        return Observation(
            t=np.array([t], dtype=np.int64), window=win, feats=feats,
            sigref=np.array([sig], dtype=np.float32),
            last_price=np.array([self.store.cols["close"][i]], dtype=np.float32))

    def step(self, n: int = 1) -> bool:
        self.t += n
        if self.t >= self.t_stop:
            self.t = self.t_start
            return False
        return True

    # -- future: reward and scoreboard only --------------------------------
    def settle(self, t: Optional[int] = None) -> np.ndarray:
        """Realised cumulative log-returns ``log(P[t+h]/P[t])`` for h=1..horizon."""
        t = self.t if t is None else int(t)
        i = self.store.idx(t)
        lp = self.store.cols["logp"]
        fut = np.asarray(lp[i + 1: i + 1 + self.horizon], dtype=np.float64)
        return (fut - float(lp[i])).astype(np.float32)


class BatchedSimulator:
    """``B`` cursors advancing together; the unit of work for one training step."""

    def __init__(self, store: MarketStore, batch: int, context: int = 43200,
                 horizon: int = 1500, t_start: Optional[int] = None,
                 t_stop: Optional[int] = None, stride: int = 1,
                 seed: int = 0, settle_lag: Optional[int] = None):
        self.store = store
        self.batch = batch
        self.context = context
        self.horizon = horizon
        self.stride = stride
        self.rng = np.random.default_rng(seed)
        # a cursor is trainable only if it has full context behind it and the
        # whole 25-minute future in front of it
        self.lo = max(store.t0 + context + WARMUP, store.t0 + context + 1)
        self.hi = store.t_end - horizon
        if t_start is not None:
            self.lo = max(self.lo, int(t_start))
        if t_stop is not None:
            self.hi = min(self.hi, int(t_stop) - 1)
        span = self.hi - self.lo
        if span <= batch:
            raise ValueError(
                f"replay span too short ({span}s) for batch={batch}; "
                f"materialise more history or lower --batch")
        self.settle_lag = horizon if settle_lag is None else int(settle_lag)
        # spread the cursors uniformly so each step mixes market regimes
        self.cursors = (self.lo + (np.arange(batch) * (span // batch))).astype(np.int64)
        self.cursors += self.rng.integers(0, max(span // batch, 1), size=batch)
        self.cursors = np.clip(self.cursors, self.lo, self.hi)
        self._sig_idx = store.feature_names.index("logvol_300s")
        self._win = np.empty((batch, N_WINDOW_CHANNELS, context), dtype=np.float32)
        self.steps = 0

    @property
    def n_trainable_seconds(self) -> int:
        return self.hi - self.lo

    def observe(self) -> Observation:
        cols = self.store.cols
        lp = cols["logp"]
        for j, t in enumerate(self.cursors):
            i = int(t) - self.store.t0
            a, b = i - self.context + 1, i + 1
            self._win[j, 0] = (np.asarray(lp[a:b], dtype=np.float64)
                               - float(lp[i])).astype(np.float32)
            for c, name in enumerate(_PASSTHROUGH, start=1):
                self._win[j, c] = cols[name][a:b]
        idx = (self.cursors - self.store.t0).astype(np.int64)
        feats = np.asarray(self.store.features[idx], dtype=np.float32)
        sig = np.exp(feats[:, self._sig_idx] - 9.0).astype(np.float32)
        price = np.asarray(cols["close"][idx], dtype=np.float32)
        return Observation(t=self.cursors.copy(), window=self._win, feats=feats,
                           sigref=sig, last_price=price)

    def settle(self) -> np.ndarray:
        """(B, horizon) realised cumulative log-returns for the current cursors.

        This is the only place training touches the future, and it reads the
        stored real tape -- not anything the simulator synthesised.
        """
        lp = self.store.cols["logp"]
        out = np.empty((self.batch, self.horizon), dtype=np.float32)
        for j, t in enumerate(self.cursors):
            i = int(t) - self.store.t0
            fut = np.asarray(lp[i + 1: i + 1 + self.horizon], dtype=np.float64)
            out[j] = fut - float(lp[i])
        return out

    def advance(self) -> None:
        """Move market time forward by ``stride`` seconds, wrapping each cursor."""
        self.cursors += self.stride
        over = self.cursors > self.hi
        if over.any():
            self.cursors[over] = self.lo + self.rng.integers(
                0, max(self.hi - self.lo - 1, 1), size=int(over.sum()))
        self.steps += 1

    def market_seconds_seen(self) -> int:
        return self.steps * self.batch * self.stride
