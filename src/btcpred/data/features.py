"""
Market Analyser -- the CPU half of the model.

It converts the raw 1 Hz tape into a bounded, scale-free feature vector that the
GPU Price Predictor consumes.  Everything here is strictly **causal**: the value
at second *t* uses only seconds <= *t*.

Two implementations, one definition
-----------------------------------
``compute_block``   vectorised over a whole array (training / materialisation).
``StreamingAnalyser`` O(1) per second, carries state (live trading / simulator).

They are the same recursions written two ways, so they must agree bit-for-bit up
to float round-off -- ``tests/test_feature_parity.py`` enforces that.  This
matters: a model trained on the vectorised features and deployed against the
streaming ones would otherwise silently eat a train/serve skew.

Why these features
------------------
Prices are non-stationary and span 4k -> 126k USDT over the dataset, so nothing
raw is fed in.  Every feature is either a log-return normalised by its own
realised volatility, a ratio of two EWMAs, or a bounded fraction.  That makes a
2020 bar and a 2026 bar look statistically comparable to the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

# ---------------------------------------------------------------- configuration
RETURN_LAGS: Tuple[int, ...] = (1, 5, 15, 60, 300, 900, 3600, 10800, 43200)
VOL_HL: Tuple[int, ...] = (10, 60, 300, 1800, 7200)      # realised-vol half-lives
TREND_HL: Tuple[int, ...] = (60, 300, 1800, 7200, 43200)  # price EWMA half-lives
FLOW_HL: Tuple[int, ...] = (30, 300, 3600)                # order-flow half-lives
CLIP = 8.0
EPS = 1e-12
SEC_PER_DAY = 86400.0

# Seconds of context to burn before a feature row is trustworthy.
#
# An EWMA still carries 2^-k of its seed after k half-lives, so the warm-up has
# to be counted against the *slowest* half-life in use (43 200 s = 12 h), not
# the fastest.  At 8 half-lives the seed contributes 0.4%, which is below the
# float16 storage resolution -- scripts/materialize.py --verify asserts that
# chunk seams are invisible, and that check fails loudly if this is too small.
_SLOWEST_HL = max(VOL_HL + TREND_HL + FLOW_HL + (43200, 7200, 3600))
WARMUP = max(RETURN_LAGS) + 8 * _SLOWEST_HL   # 43 200 + 345 600 = 388 800 s


def _alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / float(halflife))


def _ewma(x: np.ndarray, halflife: float, init: float | None = None) -> np.ndarray:
    """Exact EWMA via a first-order IIR filter: y[t] = a*x[t] + (1-a)*y[t-1]."""
    a = _alpha(halflife)
    y0 = float(x[0]) if init is None else float(init)
    zi = np.array([(1.0 - a) * y0], dtype=np.float64)
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x.astype(np.float64), zi=zi)
    return y


def _lag_ratio(logp: np.ndarray, k: int) -> np.ndarray:
    """log(P[t]/P[t-k]) with the first k entries back-filled from what exists."""
    out = np.empty_like(logp)
    out[k:] = logp[k:] - logp[:-k]
    out[:k] = logp[:k] - logp[0]
    return out


def feature_names() -> List[str]:
    names: List[str] = []
    names += [f"ret_{k}s_z" for k in RETURN_LAGS]
    names += [f"ret_{k}s_tanh" for k in RETURN_LAGS]
    names += [f"logvol_{h}s" for h in VOL_HL]
    names += [f"volratio_{a}_{b}" for a, b in zip(VOL_HL[:-1], VOL_HL[1:])]
    names += [f"trend_z_{h}s" for h in TREND_HL]
    names += [f"range_{h}s" for h in (60, 900, 7200)]
    names += ["vol_z", "logvol_ratio_60_3600", "logvol_ratio_300_43200",
              "ntrades_z", "ntrades_ratio_60_3600"]
    names += [f"flow_imb_{h}s" for h in FLOW_HL]
    names += [f"signed_vol_{h}s" for h in FLOW_HL]
    names += [f"vwap_dev_{h}s" for h in (60, 900, 7200)]
    names += ["amihud_300s", "amihud_3600s"]
    names += ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    names += ["valid_frac_300s", "gap_flag"]
    names += ["accel_60s", "rsi_900s"]
    return names


N_FEATURES = len(feature_names())


# ------------------------------------------------------------------- vectorised
def compute_block(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    qvol: np.ndarray,
    ntrades: np.ndarray,
    takerfrac: np.ndarray,
    valid: np.ndarray,
    t0: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Causal features for a contiguous 1 Hz block starting at unix second ``t0``.

    Returns ``(n, N_FEATURES)`` float32.  The first ``WARMUP`` rows are warm-up
    and should be discarded by the caller (feed an overlapping prefix).
    """
    n = close.shape[0]
    close = np.asarray(close, dtype=np.float64)
    logp = np.log(np.maximum(close, EPS))
    r1 = np.empty(n, dtype=np.float64)
    r1[0] = 0.0
    r1[1:] = np.diff(logp)

    # realised volatility ladder
    var = {h: _ewma(r1 * r1, h, init=0.0) for h in VOL_HL}
    sig = {h: np.sqrt(np.maximum(var[h], 0.0)) for h in VOL_HL}
    sig_ref = np.maximum(sig[300], 1e-7)          # 5-min realised vol per second

    # Stream columns straight into the float32 output instead of building a
    # list of 59 float64 arrays: at a 1 M-second chunk that is the difference
    # between ~470 MB and ~40 MB of peak RSS per worker, which decides how many
    # materialiser processes fit on the box.
    feats = np.empty((n, N_FEATURES), dtype=np.float32) if out is None else out

    class _Emit:
        """Writes each finished column into the output buffer and drops it."""

        __slots__ = ("i",)

        def __init__(self) -> None:
            self.i = 0

        def append(self, c: np.ndarray) -> None:
            col = np.nan_to_num(np.asarray(c, dtype=np.float64),
                                nan=0.0, posinf=CLIP, neginf=-CLIP)
            np.clip(col, -CLIP, CLIP, out=col)
            feats[:, self.i] = col
            self.i += 1

        def __iadd__(self, items):
            for it in items:
                self.append(it)
            return self

    cols = _Emit()

    # 1. multi-horizon returns, normalised by sqrt-of-time scaled realised vol
    rets = {k: _lag_ratio(logp, k) for k in RETURN_LAGS}
    for k in RETURN_LAGS:
        cols.append(rets[k] / (sig_ref * np.sqrt(k)))
    # 2. the same returns, squashed -- keeps raw magnitude information
    for k in RETURN_LAGS:
        cols.append(np.tanh(rets[k] * 200.0))

    # 3. volatility level and term structure
    for h in VOL_HL:
        cols.append(np.log(np.maximum(sig[h], 1e-9)) + 9.0)
    for a, b in zip(VOL_HL[:-1], VOL_HL[1:]):
        cols.append(np.log(np.maximum(sig[a], 1e-9) / np.maximum(sig[b], 1e-9)))

    # 4. distance from trend, in units of realised vol over the same horizon
    for h in TREND_HL:
        ma = _ewma(logp, h, init=logp[0])
        cols.append((logp - ma) / (sig_ref * np.sqrt(h)))

    # 5. intrabar range (Parkinson-style) -- a volatility estimate the close misses
    hl = np.log(np.maximum(high, EPS) / np.maximum(low, EPS))
    for h in (60, 900, 7200):
        cols.append(np.log(np.maximum(_ewma(hl, h, init=0.0), 1e-9)) + 9.0)

    # 6. activity
    lv = np.log1p(np.maximum(vol.astype(np.float64), 0.0))
    lv_slow = _ewma(lv, 3600, init=0.0)
    lv_fast = _ewma(lv, 60, init=0.0)
    lv_vslow = _ewma(lv, 43200, init=0.0)
    lv_mid = _ewma(lv, 300, init=0.0)
    lv_sd = np.sqrt(np.maximum(_ewma((lv - lv_slow) ** 2, 3600, init=0.0), 1e-9))
    cols.append((lv - lv_slow) / lv_sd)
    cols.append(lv_fast - lv_slow)
    cols.append(lv_mid - lv_vslow)
    ln = np.log1p(np.maximum(ntrades.astype(np.float64), 0.0))
    ln_slow = _ewma(ln, 3600, init=0.0)
    ln_sd = np.sqrt(np.maximum(_ewma((ln - ln_slow) ** 2, 3600, init=0.0), 1e-9))
    cols.append((ln - ln_slow) / ln_sd)
    cols.append(_ewma(ln, 60, init=0.0) - ln_slow)

    # 7. order-flow imbalance: who is crossing the spread
    imb = np.where(vol > 0, (takerfrac.astype(np.float64) - 0.5) * 2.0, 0.0)
    for h in FLOW_HL:
        cols.append(_ewma(imb, h, init=0.0))
    sv = imb * lv
    for h in FLOW_HL:
        sv_e = _ewma(sv, h, init=0.0)
        sv_sd = np.sqrt(np.maximum(_ewma(sv * sv, h, init=0.0), 1e-9))
        cols.append(sv_e / sv_sd)

    # 8. deviation from VWAP
    qv = np.maximum(qvol.astype(np.float64), 0.0)
    bv = np.maximum(vol.astype(np.float64), 0.0)
    for h in (60, 900, 7200):
        num = _ewma(qv, h, init=0.0)
        den = _ewma(bv, h, init=0.0)
        vwap = np.where(den > EPS, num / np.maximum(den, EPS), close)
        cols.append(np.log(np.maximum(close, EPS) / np.maximum(vwap, EPS))
                    / (sig_ref * np.sqrt(h)))

    # 9. Amihud illiquidity: price impact per unit of traded value
    ami = np.abs(r1) / np.maximum(qv, 1.0)
    for h in (300, 3600):
        cols.append(np.log(np.maximum(_ewma(ami, h, init=0.0), 1e-15)) + 18.0)

    # 10. calendar -- crypto has a strong and stable intraday/weekly seasonality
    t = t0 + np.arange(n, dtype=np.float64)
    tod = (t % SEC_PER_DAY) / SEC_PER_DAY
    dow = ((t // SEC_PER_DAY) % 7.0) / 7.0
    cols += [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
             np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)]

    # 11. data quality -- lets the net discount forward-filled stretches
    v = valid.astype(np.float64)
    cols.append(_ewma(v, 300, init=1.0))
    cols.append(1.0 - v)

    # 12. acceleration and a bounded momentum oscillator
    cols.append((rets[60] - _lag_ratio(logp, 120) + rets[60]) / (sig_ref * np.sqrt(60)))
    up = _ewma(np.maximum(r1, 0.0), 900, init=0.0)
    dn = _ewma(np.maximum(-r1, 0.0), 900, init=0.0)
    cols.append((up - dn) / np.maximum(up + dn, 1e-12))

    assert cols.i == N_FEATURES, f"emitted {cols.i} columns, expected {N_FEATURES}"
    return feats


# -------------------------------------------------------------------- streaming
class _Ewma:
    __slots__ = ("a", "y", "primed")

    def __init__(self, halflife: float, init: float = 0.0, prime: bool = True):
        self.a = _alpha(halflife)
        self.y = float(init)
        self.primed = prime

    def update(self, x: float) -> float:
        if not self.primed:
            self.y = float(x)
            self.primed = True
        else:
            self.y += self.a * (float(x) - self.y)
        return self.y


@dataclass
class Bar:
    t: int
    close: float
    high: float
    low: float
    vol: float
    qvol: float
    ntrades: float
    takerfrac: float
    valid: float


class StreamingAnalyser:
    """Per-second O(1) Market Analyser.  Same recursions as ``compute_block``."""

    def __init__(self) -> None:
        self.names = feature_names()
        self.hist: List[float] = []           # log prices, capped at max lag
        self.maxlag = max(RETURN_LAGS)
        self.var = {h: _Ewma(h, 0.0, prime=True) for h in VOL_HL}
        self.trend = {h: _Ewma(h, 0.0, prime=False) for h in TREND_HL}
        self.rng = {h: _Ewma(h, 0.0, prime=True) for h in (60, 900, 7200)}
        self.lv = {h: _Ewma(h, 0.0, prime=True) for h in (60, 300, 3600, 43200)}
        self.lv_var = _Ewma(3600, 0.0, prime=True)
        self.ln = {h: _Ewma(h, 0.0, prime=True) for h in (60, 3600)}
        self.ln_var = _Ewma(3600, 0.0, prime=True)
        self.flow = {h: _Ewma(h, 0.0, prime=True) for h in FLOW_HL}
        self.sv = {h: _Ewma(h, 0.0, prime=True) for h in FLOW_HL}
        self.sv2 = {h: _Ewma(h, 0.0, prime=True) for h in FLOW_HL}
        self.qv = {h: _Ewma(h, 0.0, prime=True) for h in (60, 900, 7200)}
        self.bv = {h: _Ewma(h, 0.0, prime=True) for h in (60, 900, 7200)}
        self.ami = {h: _Ewma(h, 0.0, prime=True) for h in (300, 3600)}
        self.validf = _Ewma(300, 1.0, prime=True)
        self.up = _Ewma(900, 0.0, prime=True)
        self.dn = _Ewma(900, 0.0, prime=True)
        self._prev_lp: float | None = None

    def _lag(self, k: int) -> float:
        """log price k seconds ago, clamped to the oldest value we hold."""
        if len(self.hist) > k:
            return self.hist[-1 - k]
        return self.hist[0]

    def update(self, bar: Bar) -> np.ndarray:
        lp = float(np.log(max(bar.close, EPS)))
        r1 = 0.0 if self._prev_lp is None else lp - self._prev_lp
        self._prev_lp = lp
        self.hist.append(lp)
        if len(self.hist) > self.maxlag + 2:
            self.hist.pop(0)

        sig = {h: np.sqrt(max(e.update(r1 * r1), 0.0)) for h, e in self.var.items()}
        sig_ref = max(sig[300], 1e-7)
        f: List[float] = []

        rets = {k: lp - self._lag(k) for k in RETURN_LAGS}
        f += [rets[k] / (sig_ref * np.sqrt(k)) for k in RETURN_LAGS]
        f += [np.tanh(rets[k] * 200.0) for k in RETURN_LAGS]
        f += [np.log(max(sig[h], 1e-9)) + 9.0 for h in VOL_HL]
        f += [np.log(max(sig[a], 1e-9) / max(sig[b], 1e-9))
              for a, b in zip(VOL_HL[:-1], VOL_HL[1:])]
        f += [(lp - self.trend[h].update(lp)) / (sig_ref * np.sqrt(h)) for h in TREND_HL]

        hl = float(np.log(max(bar.high, EPS) / max(bar.low, EPS)))
        f += [np.log(max(self.rng[h].update(hl), 1e-9)) + 9.0 for h in (60, 900, 7200)]

        lv = float(np.log1p(max(bar.vol, 0.0)))
        e60, e300 = self.lv[60].update(lv), self.lv[300].update(lv)
        e3600, e43200 = self.lv[3600].update(lv), self.lv[43200].update(lv)
        lv_sd = np.sqrt(max(self.lv_var.update((lv - e3600) ** 2), 1e-9))
        f += [(lv - e3600) / lv_sd, e60 - e3600, e300 - e43200]
        ln = float(np.log1p(max(bar.ntrades, 0.0)))
        n3600 = self.ln[3600].update(ln)
        ln_sd = np.sqrt(max(self.ln_var.update((ln - n3600) ** 2), 1e-9))
        f += [(ln - n3600) / ln_sd, self.ln[60].update(ln) - n3600]

        imb = (bar.takerfrac - 0.5) * 2.0 if bar.vol > 0 else 0.0
        f += [self.flow[h].update(imb) for h in FLOW_HL]
        svv = imb * lv
        for h in FLOW_HL:
            m = self.sv[h].update(svv)
            s = np.sqrt(max(self.sv2[h].update(svv * svv), 1e-9))
            f.append(m / s)

        for h in (60, 900, 7200):
            num = self.qv[h].update(max(bar.qvol, 0.0))
            den = self.bv[h].update(max(bar.vol, 0.0))
            vwap = num / max(den, EPS) if den > EPS else bar.close
            f.append(np.log(max(bar.close, EPS) / max(vwap, EPS)) / (sig_ref * np.sqrt(h)))

        a = abs(r1) / max(bar.qvol, 1.0)
        f += [np.log(max(self.ami[h].update(a), 1e-15)) + 18.0 for h in (300, 3600)]

        tod = (bar.t % SEC_PER_DAY) / SEC_PER_DAY
        dow = ((bar.t // SEC_PER_DAY) % 7.0) / 7.0
        f += [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
              np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)]
        f += [self.validf.update(bar.valid), 1.0 - bar.valid]

        f.append((2.0 * rets[60] - (lp - self._lag(120))) / (sig_ref * np.sqrt(60)))
        u, d = self.up.update(max(r1, 0.0)), self.dn.update(max(-r1, 0.0))
        f.append((u - d) / max(u + d, 1e-12))

        arr = np.asarray(f, dtype=np.float64)
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=CLIP, neginf=-CLIP),
                       -CLIP, CLIP).astype(np.float32)
