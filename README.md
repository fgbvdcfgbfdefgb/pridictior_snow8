# pridictior_snow8 — BTCUSDT 25-minute price predictor

A second-by-second Bitcoin forecasting stack: ~6.75 years of 1 Hz BTCUSDT tape,
a CPU **Market Analyser**, a GPU **Price Predictor** that emits a full
25-minute price path and refreshes it **every second**, online training with a
per-second reward (no epochs), and a 30 fps replay animation scored against the
real market.

Built to run **fully offline on Snowflake** — 4 × A10G (23 GB), 48 vCPU, 100 GB
RAM. The only network access needed at run time is `pip install`.

---

## TL;DR

```bash
git clone https://github.com/fgbvdcfgbfdefgb/pridictior_snow8.git
cd pridictior_snow8
pip install -r requirements.txt

# 1. decompress shards + precompute all features  (CPU, ~20 min, ~28 GB)
python scripts/materialize.py --shards data/shards --out /tmp/btcstore \
       --workers 16 --chunk 4000000 --verify

# 2. train one distinct model per GPU, no epochs   (4 × A10G)
torchrun --nproc_per_node=4 scripts/train.py \
       --store /tmp/btcstore --out runs/pop --batch 24 --max-hours 8

# 3. replay a random day and render the 30 fps video
python scripts/render_video.py --store /tmp/btcstore --ckpt-dir runs/pop \
       --random-day --pred-stride 20 --fps 30 --out media/replay.mp4
```

Or open `notebooks/snowflake_btc_predictor.ipynb` and run it top to bottom.

---

## 1. The dataset

`data/shards/` holds **every second of BTCUSDT from 2020-01-01 to 2026-10-03**
— about 2.13 × 10⁸ rows — as 82 monthly `.btcz` files, one per month, each
~20–39 MB (every file is comfortably under GitHub's 100 MB limit).

| | |
|---|---|
| Source | Binance Data Vision, 1 s spot klines |
| Span | 2020-01-01 00:00:00 → 2026-10-03 23:59:59 UTC |
| Rows | **213,235,200** consecutive seconds, no holes |
| Raw size | ~29 GB of CSV / ~6.9 GB of Binance zips |
| In this repo | **2.19 GB** (10.3 bytes per second of market) |
| Largest file | 38.9 MB (GitHub's cap is 100 MB) |
| Genuinely reported | 213,086,061 s — **99.930%** |
| Forward-filled | 149,139 s — 0.070%, flagged in the `valid` bitmask |
| Fidelity | **bit-exact** for OHLC, volume and trade count |

The grid is *dense*: every second exists. Seconds the exchange never reported
are forward-filled as flat candles and flagged in a `valid` bitmask, so a fixed
43 200-step window is always well-defined while the model can still tell a real
bar from a filled one. Gaps are concentrated in 2020 (worst month 2020-02 at
0.99%); 62 of the 82 months have none at all.

`ShardIndex` refuses to load a directory whose months do not butt up exactly
against each other, so a partial download fails loudly instead of training on a
silently discontinuous timeline.

### Why a custom container and not Parquet

Prices are nearly constant second to second, so storing *differences* as
integers makes them almost free to compress. Measured on 2024-01:

| encoding | one month |
|---|---|
| Binance CSV | 378 MB |
| Binance zip | 84 MB |
| Parquet + zstd (float columns) | ~31 MB |
| **`.btcz`** | **~25 MB** |

`.btcz` stores `close` as a first-difference in integer cents, O/H/L as integer
offsets from the close, volume as an integer at 1e-5 BTC resolution, taker-buy
as a `uint16` fraction, and validity as a bitmask — then zstd-19 over each
column separately. Full spec and the round-trip test: `docs/DATA_FORMAT.md`.

### Rebuilding or extending it

`data/shards/` is already populated. To refresh it (needs internet — this is
the *only* step that does):

```bash
python scripts/download_pack.py --start 2020-01 --end 2026-10 --out data/shards
```

It is resumable, skips months already on disk, and auto-detects two Binance
quirks that silently corrupt naive parsers: kline timestamps switched from
**milliseconds to microseconds** during 2025, and some archives carry a CSV
header row while others do not.

---

## 2. Architecture

```
          ┌──────────────── CPU ────────────────┐   ┌────────── GPU ──────────┐
 1 Hz     │  Market Analyser                    │   │  Price Predictor        │
 tape ───►│  59 causal, scale-free features     ├──►│  multi-scale encoder    │
          │  (returns · realised vol · order    │   │  + transformer          │──► 25-min
          │   flow · VWAP · Amihud · calendar)  │   │  + KL path basis        │    path
          └─────────────────────────────────────┘   └─────────────────────────┘
                                                        ▲
             last 12 h of raw tape (43 200 × 8 ch) ─────┘
```

### Market Analyser — `src/btcpred/data/features.py`

59 features per second, each either a log-return normalised by its own realised
volatility, a ratio of two EWMAs, or a bounded fraction. That is what lets a
2020 bar (BTC ≈ $7 k) and a 2026 bar (BTC ≈ $63 k) look statistically
comparable to the network.

Groups: multi-horizon returns (1 s → 12 h) both vol-normalised and squashed;
a realised-volatility ladder with its term structure; distance from trend at
five half-lives; Parkinson intrabar range; volume and trade-count anomalies;
taker-buy order-flow imbalance; VWAP deviation at three scales; Amihud
illiquidity; intraday/weekly seasonality; and data-quality flags.

It exists **twice** — `compute_block` (vectorised, used for training) and
`StreamingAnalyser` (O(1) per second, used for live replay). They are the same
recursions written two ways, and `tests/test_feature_parity.py` asserts they
agree to **6 × 10⁻⁸**. Without that check, a model trained on one and served
the other would quietly eat a train/serve skew.

### Price Predictor — `src/btcpred/models/predictor.py`

**Input.** The Market Analyser vector at second *t*, plus the raw last 12 hours
at 1 Hz (43 200 × 8 channels).

**Multi-resolution pyramid, not a 43 200-token transformer.** Attention over
43 200 steps is ~1.9 × 10⁹ pairs per head. Instead the window is cut into four
levels — the last 17 min at 1 s, 68 min at 4 s, 4.5 h at 16 s, 12 h at 64 s —
each patch-embedded into 64-ish tokens. **239 tokens total, full 12-hour reach,
1-second resolution where it matters.**

**A Karhunen–Loève path basis, not 1 500 free outputs.** The head emits `K`
coefficients of

```
φ_k(u) = √2 · sin((k − ½)πu) / ((k − ½)π),     u = h/1500
```

which is exactly the KL expansion of Brownian motion on [0,1]. Three things
fall out of that, all of them requirements from the spec:

1. every basis function is zero at `u = 0`, so the path **starts at the current
   price by construction** — no discontinuity at the anchor;
2. the basis is ordered by smoothness and truncated at `K`, so high-frequency
   jitter is **structurally impossible** rather than merely penalised — this is
   what makes the output stable enough to drive trading;
3. coefficients are pre-scaled by their KL standard deviation and multiplied by
   `σ·√H`, so the network predicts unit-variance numbers and the whole model is
   scale-free across the 4 k → 126 k USDT range of the dataset.

Three quantiles (0.1 / 0.5 / 0.9) are emitted and forced monotone via softplus
offsets, giving a calibrated band for free.

### The population — one different model per GPU

The spec asks for *different models on each GPU*, so this is deliberately
**not** DDP. Each rank trains an independent predictor with its own
architecture, learning rate, basis size and stability weight; nothing is
all-reduced. They are combined at evaluation time, where their disagreement is
itself a useful uncertainty signal.

| rank | variant | d_model | depth | K | params | lr | stability |
|---|---|---|---|---|---|---|---|
| 0 | `base` | 640 | 12 | 24 | **83.1 M** | 3e-4 | 0.35 |
| 1 | `deep` | 768 | 16 | 32 | **157.3 M** | 2e-4 | 0.50 |
| 2 | `wide` | 1024 | 10 | 32 | **178.4 M** | 2.5e-4 | 0.25 |
| 3 | `agile` | 512 | 14 | 40 | **61.7 M** | 5e-4 | 0.20 |

**480 M parameters across the four cards.** The largest needs ~2.5 GB of AdamW
state, so batch 24 at bf16 sits around 9 GB of the 23 GB available — batch 48
is comfortable. `micro` (2.9 M) and `nano` (0.47 M) recipes exist for CPU smoke
tests.

---

## 3. Training: no epochs, a per-second reward

`src/btcpred/train/online.py`. There is no dataset shuffle and no epoch
counter. `B` cursors are scattered across the whole history and every optimiser
step advances all of them by one **market second**. At each second the model
emits a 25-minute path, and the reward for that second is computed against what
the market actually did over the following 1 500 seconds — read from the stored
real tape, never from anything the simulator generated.

> **The one honest subtlety.** A prediction made at *t* cannot be scored until
> *t + 1500*. Rather than keep 1 500 autograd graphs alive, the learning cursor
> runs 25 minutes behind the simulator's "now" — exactly how a production
> online learner behaves: it learns from settled predictions while serving
> unsettled ones. `MarketSimulator` enforces the split in its API —
> `observe()` only ever reads the past, `settle()` is the only method that
> touches the future.

The per-second loss combines:

| term | purpose |
|---|---|
| pinball over all 1 500 points | calibrated uncertainty band |
| Huber on the median path, σ-normalised | robust to fat tails |
| directional term at 1/5/15/25 min | the part a trader actually monetises |
| **stability** vs the previous second's path | no erratic second-to-second jumps |
| curvature penalty | smooth paths |

Logged `reward = −nRMSE + 0.5·(hit − 0.5)` is volatility-normalised, so it is
comparable across 2020 and 2026 regimes.

### The simulator — `src/btcpred/sim/simulator.py`

Replays stored history one second at a time as if streaming. `BatchedSimulator`
runs `B` cursors spread across the full history, so a single step mixes the
March-2020 crash, the 2021 bull run and 2026 tape — the model never sees a
regime in isolation, and there is no epoch boundary to overfit to.

---

## 4. Output: the 30 fps replay

```bash
python scripts/render_video.py --store /tmp/btcstore --ckpt-dir runs/pop \
    --random-day --hours 24 --pred-stride 20 --frame-stride 1 --fps 30 \
    --out media/replay.mp4
```

Picks a random day between 2020 and 2026, walks it second by second, and
renders:

* the **actual price** as a solid line, scrolling;
* each model's **25-minute forecast as a dotted line** projected to the right
  of "now", with the ensemble quantile band shaded;
* the real tape over the forecast span, drawn faint, so you can watch the
  dotted line be right or wrong;
* a **live accuracy bar per model**, `100·(1 − nRMSE)` against the stored real
  data, plus rolling directional hit-rate.

`--pred-stride 20 --frame-stride 1` turns a 24-hour session into 4 320 frames =
2 min 24 s at 30 fps. `--pred-stride 1` predicts every single second (86 400
frames, 48 min of video) if you want the full-rate version.

A JSON scorecard is written next to the video:

```json
[{"model": "base", "dir_hit_pct": 54.1, "path_nrmse": 0.83,
  "accuracy_pct": 17.2, "stability_jump_sigma": 0.004}]
```

`stability_jump_sigma` is the median change in the 25-minute point forecast from
one second to the next, in σ units — the direct measure of "no erratic jumps".

---

## 5. Repository layout

```
pridictior_snow8/
├── data/shards/                 82 × .btcz  — the full 2020→2026 1 Hz dataset
├── docs/DATA_FORMAT.md          .btcz container spec
├── notebooks/
│   └── snowflake_btc_predictor.ipynb     the Snowflake runbook
├── scripts/
│   ├── download_pack.py         Binance → .btcz          (only online step)
│   ├── materialize.py           .btcz → memmap + features (offline, parallel)
│   ├── train.py                 torchrun entrypoint, one model per GPU
│   └── render_video.py          replay + score + 30 fps MP4
├── src/btcpred/
│   ├── data/   btcz.py · dataset.py · features.py        Market Analyser
│   ├── sim/    simulator.py                              live replay
│   ├── models/ predictor.py · inputs.py                  Price Predictor
│   ├── train/  online.py                                 per-second reward loop
│   ├── eval/   replay.py                                 scoring vs real tape
│   └── viz/    animate.py                                30 fps renderer
└── tests/      feature parity · causality · round-trip · shapes
```

---

## 6. Expectations, stated plainly

Second-scale crypto forecasting is close to the noise floor. A genuinely useful
model lands a few points above 50% directional accuracy on the 25-minute move,
and `path_nrmse` below 1.0 means it beats a flat "price won't move" baseline.
Treat any run reporting 90%+ directional accuracy as a bug — most likely
lookahead. The two tests in `tests/` exist specifically to catch that class of
error, and `settle()` being the only forward-looking call in the simulator is
the structural guard.

Nothing here is financial advice or a trading system.

## 7. Tests

```bash
python tests/test_feature_parity.py   # vectorised vs streaming, causality
python tests/test_pipeline.py         # btcz round-trip, model shapes, loss, basis
```
