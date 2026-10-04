"""Shape, round-trip and sanity checks that do not need the full dataset."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from btcpred.data.btcz import PRICE_SCALE, read_header, read_shard, write_shard


def test_btcz_roundtrip():
    rng = np.random.default_rng(7)
    n = 50_000
    close = np.cumsum(rng.integers(-40, 41, n)) + 6_200_000     # cents
    close = np.maximum(close, 100_000).astype(np.int64)
    high = close + rng.integers(0, 60, n)
    low = close - rng.integers(0, 60, n)
    opn = close + rng.integers(-30, 31, n)
    vol = np.abs(rng.standard_normal(n)) * 0.7
    qvol = vol * (close / PRICE_SCALE)
    ntr = rng.poisson(9, n)
    tf = rng.random(n)
    valid = rng.random(n) > 0.02

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.btcz"
        hdr = write_shard(p, t0=1_700_000_000, close_cents=close, open_cents=opn,
                          high_cents=high, low_cents=low, vol=vol, qvol=qvol,
                          ntrades=ntr, takerbuy_base=tf * vol, valid=valid, level=10)
        assert read_header(p)["n"] == n
        sh = read_shard(p)
        assert np.array_equal(sh.close, close), "close must be exact"
        assert np.array_equal(sh.open, opn)
        assert np.array_equal(sh.high, high)
        assert np.array_equal(sh.low, low)
        assert np.array_equal(sh.ntrades, ntr.astype(np.int32))
        assert np.array_equal(sh.valid, valid)
        assert np.abs(sh.vol - vol).max() < 1e-5
        ratio = hdr["file_bytes"] / (n * 12)
        print(f"roundtrip OK  exact OHLC+trades, {hdr['file_bytes']/1e3:.0f} kB "
              f"({ratio:.2f} B/row)")


def test_kl_basis_properties():
    import torch
    from btcpred.models.predictor import kl_basis
    H, K = 1500, 24
    phi = kl_basis(H, K).double()
    # anchored: the first step is tiny, and extrapolating back to u=0 gives 0
    assert phi[0].abs().max() < 1e-2, "basis must start at ~0"
    # ordered by smoothness: later components wiggle more
    tv = (phi[1:] - phi[:-1]).abs().sum(0)
    assert torch.all(tv[1:] > tv[:-1] * 0.9), "components should get rougher"
    # near-orthogonal (KL components are orthogonal on the continuum)
    g = (phi.T @ phi) / H
    off = (g - torch.diag(torch.diag(g))).abs().max()
    assert off < 1e-3, f"components not orthogonal: {off}"
    print(f"KL basis OK  K={K}, max off-diagonal Gram = {off:.2e}")


def test_model_shapes_and_loss():
    import torch
    from btcpred.models.inputs import WindowNormalizer
    from btcpred.models.predictor import PredictionLoss, make_config, PricePredictor

    ctx, hor, B, F = 4096, 300, 2, 59
    cfg, lw, _ = make_config("nano", n_features=F)
    cfg = type(cfg)(**{**cfg.to_dict(), "context": ctx, "horizon": hor,
                       "levels": ((512, 1, 16), (2048, 4, 16), (4096, 16, 8))})
    m = PricePredictor(cfg)
    norm = WindowNormalizer()

    raw = torch.randn(B, 7, ctx) * 0.001
    raw[:, 6] = 1.0
    sig = torch.full((B,), 1e-4)
    feats = torch.randn(B, F)
    x = norm(raw, sig)
    assert x.shape == (B, 8, ctx), x.shape
    out = m(x, feats, sig)
    assert out.shape == (B, len(cfg.quantiles), hor), out.shape

    # quantiles must be ordered everywhere
    assert torch.all(out[:, 0] <= out[:, 1] + 1e-6)
    assert torch.all(out[:, 1] <= out[:, 2] + 1e-6)
    # an untrained model is a calibrated driftless random walk:
    # zero median, and a 10/90 band of +/- 1.2816 * sigma * sqrt(H)
    med = out[:, 1]
    assert med.abs().max() < 1e-6, f"median should start at zero, got {med.abs().max()}"
    want = 1.2816 * float(sig[0]) * np.sqrt(hor)
    got = float(out[0, 2, -1])
    assert abs(got - want) / want < 0.05, f"band {got:.3e} != gaussian {want:.3e}"
    assert abs(float(out[0, 0, -1]) + want) / want < 0.05, "band must be symmetric"
    print(f"  init band at H: {got:.3e} vs gaussian {want:.3e}")

    crit = PredictionLoss(cfg, lw)
    truth = torch.randn(B, hor) * 1e-3
    loss, stats = crit(out, truth, sig, prev_pred=out.detach())
    assert torch.isfinite(loss)
    loss.backward()
    gs = [p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None]
    assert sum(gs) > 0, "no gradient reached the parameters"
    print(f"model OK  out={tuple(out.shape)}  loss={float(loss):.4f}  "
          f"nrmse={stats['nrmse']:.3f}  params={m.n_params()/1e6:.2f}M")


def test_anchor_continuity():
    """A predicted path must start at the current price, by construction."""
    import torch
    from btcpred.models.inputs import WindowNormalizer
    from btcpred.models.predictor import make_config, PricePredictor
    cfg, _, _ = make_config("nano", n_features=59)
    cfg = type(cfg)(**{**cfg.to_dict(), "context": 2048, "horizon": 600,
                       "levels": ((512, 1, 16), (2048, 4, 16))})
    m = PricePredictor(cfg)
    with torch.no_grad():                      # perturb the head off zero
        for p in m.head[-1].parameters():
            p.add_(torch.randn_like(p) * 0.05)
    raw = torch.randn(3, 7, 2048) * 1e-3
    raw[:, 6] = 1.0
    sig = torch.full((3,), 2e-4)
    path = m(WindowNormalizer()(raw, sig), torch.randn(3, 59), sig)
    step0 = path[:, :, 0].abs().max().item()
    typical = path[:, :, -1].abs().max().item()
    assert step0 < typical * 0.05, f"path jumps at h=1: {step0} vs {typical}"
    print(f"anchor OK  |path[h=1]|={step0:.2e} vs |path[H]|={typical:.2e}")


if __name__ == "__main__":
    test_btcz_roundtrip()
    test_kl_basis_properties()
    test_model_shapes_and_loss()
    test_anchor_continuity()
    print("ALL OK")
