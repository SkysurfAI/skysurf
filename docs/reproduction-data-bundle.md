# Reproduction data bundle

`skysurf.reproduction` reads every input from a single flat directory — the
**reproduction data bundle**. It is distributed separately from the code
(~450 MB) because it is large and pre-computed. This page is its exact contract:
match this layout and the backtest reproduces the published numbers bit-for-bit.

Point the library at the bundle with `--data DIR`, `reproduce(data_dir=DIR)`, or
the `SKYSURF_REPRO_DATA` environment variable.

There are two ways to obtain the bundle:

1. **Download it** (pre-computed) and run `skysurf-reproduce`.
2. **Regenerate it from raw daily OHLCV** with `skysurf-build` — the
   open-sourced data pipeline. This is the license-clean path: bring your own
   NSE OHLCV, and the code derives every intermediate. See
   [Raw OHLCV inputs](#raw-ohlcv-inputs-for-skysurf-build) below.

## Raw OHLCV inputs (for `skysurf-build`)

`skysurf-build --ohlcv DIR --out BUNDLE` reads raw **daily** OHLCV from `DIR`
and writes the derived intermediates into `BUNDLE`. Two files (`.parquet`
preferred, `.csv` accepted):

### `stocks_daily.parquet` / `.csv`
One row per ticker per trading day.

| column | type | notes |
|---|---|---|
| `ticker` | str | NSE symbol, e.g. `RELIANCE.NS` |
| `date` | date | trading day |
| `open` `high` `low` `close` | float | split-adjusted, not dividend-adjusted |
| `volume` | int | |
| `primary_sector` | str | e.g. `NIFTY IT` (required for universe + sector mapping) |
| `market_cap` | float | INR (required; drives the quarterly universe filter) |
| `has_demerger` | bool | optional corporate-action flags |
| `demerger_no_trade_until` | date/null | optional |
| `has_sme_history` | bool | optional; SME names are excluded when present |

### `benchmark_daily.parquet` / `.csv`
Nifty 50 daily, one row per trading day: `date, open, high, low, close, volume`.

From these, `skysurf-build` regenerates: `stock_weekly_cache/`, `nifty_weekly.csv`,
`regime_weekly.csv`, `breadth_weekly.csv`, `universe_quarterly.csv`,
`entries_all.csv`, and `entries_all_lagged.csv`.

### `sector_daily.parquet` / `.csv` (extra raw input for sector regimes)
Per-index daily OHLCV for the NIFTY sector and cap-tier indices (and `NIFTY 50`,
used as the relative-strength benchmark). One row per index per trading day:
`index_symbol, trade_date, open, high, low, close, volume`.

The remaining bundle CSVs now regenerate too — the OHLCV-only loop is closed:
- `skysurf-build-sectors` (`pipeline.sector_regime`) reads `sector_daily` +
  `stocks_daily` → `sector_regime_weekly.csv`, `captier_regime_weekly.csv`,
  `ticker_metadata.csv` (+ diagnostic `sector_breadth_weekly.csv`).
- `skysurf-build-stats` (`pipeline.entry_stats`) reads `entries_all.csv` +
  `stock_weekly_cache/` → `trade_stats_all.csv`.

So the full build is: `skysurf-build` → `skysurf-build-sectors` →
`skysurf-build-stats` → `skysurf-reproduce`. You may still download the
pre-computed bundle instead of rebuilding.

---

## Derived bundle layout (what `skysurf.reproduction` consumes)

```
skysurf-repro-data/
├── stock_weekly_cache/         # one CSV per ticker, e.g. RELIANCE.NS.csv  (~2,155 files)
├── nifty_weekly.csv            # Nifty 50 weekly OHLCV + indicators
├── regime_weekly.csv           # weekly market-breadth / regime timeline
├── sector_regime_weekly.csv    # per-sector weekly regime + RS quartiles
├── captier_regime_weekly.csv   # per-cap-tier weekly regime
├── ticker_metadata.csv         # ticker → sector / cap-tier / index mapping
├── entries_all.csv             # all candidate entries (baseline config)
├── entries_all_lagged.csv      # look-ahead-fixed entries (Phase-4 configs)
├── trade_stats_all.csv         # per-trade MFE/MAE stats (dynamic type-prior)
├── manifest.json               # row counts, date range, ticker count, bundle version
└── SHA256SUMS                  # integrity checksums
```

## File schemas

All dates are ISO `YYYY-MM-DD`. All CSVs are comma-separated with a header row.

### `stock_weekly_cache/<TICKER>.csv`
Pre-computed weekly bars **with indicators already calculated** (this is why
reproduction is exact — no indicator recompute drift). One file per ticker,
named by NSE symbol with the `.NS` suffix.

| column | type | notes |
|---|---|---|
| `date` | date | week ending |
| `Open` `High` `Low` `Close` | float | split-adjusted, **not** dividend-adjusted |
| `Volume` | int | |
| `atr_14` | float | 14-week ATR |
| `rs_13w` | float | 13-week relative strength vs Nifty |
| `rsi_14` | float | |
| `momentum` | float | |
| `volume_ratio_20w` | float | volume / 20w avg |
| `ema_20` `ema_25` `ema_30` `ema_40` | float | |
| `sma_20` `sma_25` `sma_30` `sma_40` | float | |

### `nifty_weekly.csv`
`date, Open, High, Low, Close, Volume, atr_14, sma_20, sma_25, sma_30, sma_40, ema_20, ema_25, ema_30, ema_40`

### `regime_weekly.csv`
Weekly market-breadth and regime timeline. This file also drives the simulation
timeline (one row per week).

`week_date, breadth_pct_ema_20, breadth_pct_ema_25, breadth_pct_ema_30, breadth_pct_ema_40, breadth_pct_sma_20, breadth_pct_sma_25, breadth_pct_sma_30, breadth_pct_sma_40, regime_breadth_ema_20, …, regime_breadth_sma_40, regime_weinstein, nifty_close, nifty_ema20, nifty_atr`

(`regime_*` columns are categorical: `bull` / `bear` / `sideways`.)

### `sector_regime_weekly.csv`
`week_date, index_symbol, primary_sector, close, ema_20, atr_14, ema_direction, regime, sector_return_13w, nifty50_return_13w, sector_rs_vs_nifty, sector_rs_rank, sector_rs_quartile`

### `captier_regime_weekly.csv`
`week_date, index_symbol, cap_tier, close, ema_20, atr_14, ema_direction, regime`
(`cap_tier` ∈ `LARGE` / `MID` / `SMALL`.)

### `ticker_metadata.csv`
`ticker, primary_sector, index_symbol, market_cap, cap_tier, sector_has_index_data, sector_data_start_date`

### `entries_all.csv` / `entries_all_lagged.csv`
Candidate entries, one row per (ticker, week, entry-type, MA config). Required
columns include `week_date, ticker, entry_type, ma_type, ma_period,
entry_price_close, rr, …`. `entries_all_lagged.csv` is the look-ahead-fixed
version (uses `argrelextrema` confirmed swings) and is what the Phase-4 configs
consume.

### `trade_stats_all.csv`
Per-trade outcome stats used to compute the dynamic type-prior out-of-sample.
Required columns: `week_date, entry_type, mfe_pct, mae_pct` (others ignored).

## `manifest.json` (recommended)

```json
{
  "bundle_version": "1.0.0",
  "generated_utc": "2026-06-04T00:00:00Z",
  "date_range": ["2007-01-05", "2026-03-21"],
  "tickers": 2155,
  "files": { "trade_stats_all.csv": {"rows": 1234567}, "...": {} }
}
```

Consumers should verify `SHA256SUMS` before running:

```bash
cd skysurf-repro-data && sha256sum -c SHA256SUMS
```

## What format should I publish it in?

* **Canonical: CSV** (as above). It's diff-able, language-agnostic, and matches
  the engine's reader exactly — keep this as the source of truth.
* **Optional: Parquet** of `stock_weekly_cache/` and `trade_stats_all.csv` for a
  ~10× smaller download. Ship it as a convenience alongside CSV, not instead of.
* **Distribution:** a single `skysurf-repro-data-v1.tar.gz`, hosted where large
  research artifacts belong — a GitHub Release asset, Hugging Face Dataset, or
  Zenodo (Zenodo gives you a citable DOI, which is ideal for a reproducible
  research claim). Keep it **out of git** (it's ~450 MB).

## Building the bundle from a Skysurf research checkout

If you maintain the research repo, `tools/make_repro_bundle.py` assembles a
bundle from `val_engine_a_v2_results/` + `val_entry_stats_results/`, flattens the
layout above, and writes `manifest.json` + `SHA256SUMS`. See that script's
`--help`.
