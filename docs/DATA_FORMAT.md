# The `.btcz` container

A compact, self-describing file for dense 1 Hz OHLCV market data.

## Why it exists

Three constraints had to hold at once:

1. ~2.13 × 10⁸ seconds of BTCUSDT must fit inside a Git repository, where a
   single file may not exceed **100 MB**;
2. training needs random access across six years without paying decompression
   in the hot loop;
3. the grid must be **dense** — a fixed 43 200-step (12 h) window has to be
   well-defined at every timestamp, with no ragged edges.

CSV fails (1) by 14×. Parquet satisfies (1) and (2) but still stores prices as
independent values. `.btcz` instead exploits the thing that is actually true of
second-by-second price data: **consecutive values barely differ**.

## Measured, on 2024-01 (2 678 400 rows)

| representation | size | vs CSV |
|---|---|---|
| Binance CSV | 378 MB | 1.0× |
| Binance `.zip` | 84 MB | 4.5× |
| float32 columns + zstd-19 | 30.8 MB | 12.3× |
| **`.btcz` (quantised + delta + zstd-19)** | **25.3 MB** | **14.9×** |

Per-column compression ratios, same month:

| column | raw | zstd-19 | ratio |
|---|---|---|---|
| `dclose` (int32 Δcents) | 10.7 MB | 2.11 MB | 5.1× |
| `open_off` | 10.7 MB | 2.10 MB | 5.1× |
| `high_off` | 10.7 MB | 1.24 MB | 8.6× |
| `low_off` | 10.7 MB | 1.24 MB | 8.7× |
| `vol` (int32, 1e-5 BTC) | 10.7 MB | 6.82 MB | 1.6× |
| `ntrades` | 10.7 MB | 2.10 MB | 5.1× |
| `takerfrac` (uint16) | 5.4 MB | — | — |

## Layout

Little-endian throughout.

```
offset  size        contents
0       6           magic  b"BTCZ1\0"
6       4           uint32 header_len
10      header_len  UTF-8 JSON header
...     rest        concatenated zstd frames, one per column, in header order
```

The header lists every column with its dtype and its exact compressed byte
length, so a reader can seek to and decode a single column without touching the
others.

```json
{
  "format": "btcz1", "symbol": "BTCUSDT", "interval": "1s",
  "t0": 1704067200, "n": 2678400,
  "price_scale": 100, "vol_scale": 100000, "taker_scale": 65535,
  "zstd_level": 17, "n_valid": 2678400,
  "columns": [{"name": "valid", "dtype": "uint8",
               "raw_bytes": 334800, "comp_bytes": 112}, ...]
}
```

## Columns

| name | stored dtype | meaning |
|---|---|---|
| `valid` | bitmask (`np.packbits`) | 1 = the exchange really reported this second |
| `dclose` | int32 | first difference of close, in integer cents |
| `open_off` | int32 | `open − close`, cents |
| `high_off` | int32 | `high − close`, cents |
| `low_off` | int32 | `low − close`, cents |
| `vol` | int32 | base volume in units of 1e-5 BTC |
| `qvol` | int32 | quote volume, whole USDT |
| `ntrades` | int32 | trades in the second |
| `takerfrac` | uint16 | taker-buy base / base volume, × 65535 |

### Two details that matter

**`dclose[0]` must be `close[0]`, not 0.** The decoder reconstructs prices with
`np.cumsum(dclose)`. Seeding the difference with `close[0]` instead of the
conventional zero is what preserves the absolute price level; getting this
wrong produces a series that is correct in shape but offset by the opening
price — silent, and invisible in a plot. `tests/test_pipeline.py` checks the
round-trip against raw CSV at 0.0 absolute error.

**Gaps are filled, then flagged.** Missing seconds get a flat candle at the
last known close with zero flow, and `valid = 0`. The bitmask costs ~1 bit per
second and compresses to almost nothing (112 bytes for a clean month), but it
lets the model discount forward-filled stretches instead of reading them as
genuine zero-volatility periods.

## Fidelity

Verified against the raw Binance CSV for 2026-08:

| field | max abs error |
|---|---|
| open / high / low / close | **0.0** (exact) |
| volume | **0.0** (exact) |
| quote volume | 0.5 USDT (rounded to whole units) |
| trade count | **0.0** (exact) |
| taker-buy volume | 7.7e-6 relative (uint16 quantisation) |

Price, volume and trade count are bit-exact. The two lossy fields are both
derived quantities that the Market Analyser only ever consumes as ratios.

## Reading one

```python
from btcpred.data.btcz import read_shard, read_header

print(read_header("data/shards/BTCUSDT-1s-2024-01.btcz"))

sh = read_shard("data/shards/BTCUSDT-1s-2024-01.btcz")
sh.close_float()   # float64 USDT, length sh.n
sh.valid           # bool mask of genuinely reported seconds
sh.t0, sh.t_end    # unix seconds, inclusive
```
