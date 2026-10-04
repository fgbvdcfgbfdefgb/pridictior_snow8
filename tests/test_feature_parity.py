"""Vectorised and streaming Market Analyser must agree."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from btcpred.data.features import (Bar, StreamingAnalyser, compute_block,
                                   feature_names, N_FEATURES)

def _synth(n=9000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 2e-4
    r[n//3:n//3+50] += 3e-3                      # a shock
    close = 30000 * np.exp(np.cumsum(r))
    spread = np.abs(rng.standard_normal(n)) * 2.0
    high, low = close + spread, close - spread
    vol = np.abs(rng.standard_normal(n)) * 0.5
    vol[rng.random(n) < 0.1] = 0.0               # quiet seconds
    qvol = vol * close
    ntr = rng.poisson(8, n).astype(float)
    tf = np.clip(rng.random(n), 0, 1)
    valid = (rng.random(n) > 0.01).astype(float)
    return dict(close=close, high=high, low=low, vol=vol, qvol=qvol,
                ntrades=ntr, takerfrac=tf, valid=valid)

def test_parity():
    d = _synth(); t0 = 1_700_000_000
    block = compute_block(t0=t0, **d)
    sa = StreamingAnalyser()
    stream = np.stack([
        sa.update(Bar(t=t0+i, close=d["close"][i], high=d["high"][i],
                      low=d["low"][i], vol=d["vol"][i], qvol=d["qvol"][i],
                      ntrades=d["ntrades"][i], takerfrac=d["takerfrac"][i],
                      valid=d["valid"][i]))
        for i in range(len(d["close"]))])
    assert block.shape == stream.shape == (len(d["close"]), N_FEATURES)
    # ignore warm-up: lag windows longer than the synthetic series clamp differently
    s = 5000
    diff = np.abs(block[s:].astype(np.float64) - stream[s:].astype(np.float64))
    worst = diff.max(axis=0)
    names = feature_names()
    bad = [(names[i], float(worst[i])) for i in np.argsort(-worst)[:6] if worst[i] > 2e-3]
    assert not bad, f"streaming/vectorised mismatch: {bad}"
    print(f"parity OK over {N_FEATURES} features, max abs diff = {worst.max():.2e}")

def test_causality():
    """Perturbing the future must not change a past feature row."""
    d = _synth(n=7000); t0 = 1_700_000_000
    a = compute_block(t0=t0, **d)
    d2 = {k: v.copy() for k, v in d.items()}
    d2["close"][6000:] *= 1.5
    d2["qvol"] = d2["vol"] * d2["close"]
    b = compute_block(t0=t0, **d2)
    assert np.allclose(a[:6000], b[:6000], atol=1e-6), "feature leaks future data"
    print("causality OK")

if __name__ == "__main__":
    test_parity(); test_causality(); print("ALL OK")
