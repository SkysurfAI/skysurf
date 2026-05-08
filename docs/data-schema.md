# Data schema

Skysurf consumes data through the abstract `DataProvider` contract. Every connector returns DataFrames in the schema below; every custom connector you write must too. This page is the single source of truth for column names, dtypes, and shapes.

If your data is in any of: a directory of CSVs, a directory of Parquet files, a SQL database, or already in pandas DataFrames — use one of the bundled connectors and only worry about the column names. If your data lives somewhere else, subclass `DataProvider` directly; this page tells you what every method must return.

## Schema reference

### Weekly OHLCV (per ticker)

Returned by `get_weekly_ohlcv(tickers, start, end)` as `dict[str, pd.DataFrame]`.

| Column | Type | Notes |
|---|---|---|
| Index | `DatetimeIndex` | Friday-close convention. Weekly bars only. |
| `Open` | float | |
| `High` | float | |
| `Low` | float | |
| `Close` | float | |
| `Volume` | float or int | Either is fine; the engine coerces. |

Tickers without data for the requested window must be **omitted** from the returned dict — do not return them with empty DataFrames.

### Daily OHLCV (per ticker)

Same columns as weekly. Returned by `get_daily_ohlcv(tickers, start, end)`.

Used internally for ADTV (average daily traded value) computation; about 20 trading days of history is enough.

### Nifty 50 weekly

Returned by `get_nifty_weekly(start, end)` as a single `pd.DataFrame`.

| Column | Type | Notes |
|---|---|---|
| Index | `DatetimeIndex` | Weekly bars. |
| `Close` | float | Required. |
| `Open`, `High`, `Low`, `Volume` | float | Optional. Tolerated and ignored. |

### Sector indices weekly (per sector)

Returned by `get_sector_indices_weekly(sectors, start, end)` as `dict[str, pd.DataFrame]`.

Same shape as Nifty 50. Sector names must match those used in the universe table (see below).

### Universe (single snapshot)

Returned by `get_universe(as_of)` as a single `pd.DataFrame`. One row per eligible ticker.

| Column | Type | Notes |
|---|---|---|
| `ticker` | str | E.g., `"RELIANCE.NS"`. |
| `sector` | str | Skysurf sector taxonomy; must match sector index keys. |
| `market_cap` | float | **Absolute INR**, not crores. |
| `adtv_20d` | float | 20-day average daily traded value, **absolute INR**. |

The brain applies the `market_cap_floor_inr` and `adtv_floor_inr` filters from `StrategyConfig` on this table. Eligibility (suspended tickers, recent listings) is your responsibility — the brain trusts what you return.

### Historical trades

Returned by `get_historical_trades(before_date)` as a single `pd.DataFrame`.

| Column | Type | Notes |
|---|---|---|
| `ticker` | str | |
| `week_date` | `pd.Timestamp` | Entry week. |
| `entry_type` | str | E.g., `"PULLBACK_S2"`, `"VCP_CONTINUATION"`. |
| `mfe_pct` | float | Max favorable excursion, never negative. |
| `mae_pct` | float | Max adverse excursion, never positive. |

The brain filters strictly on `entry_week_date < before_date`.

> **Documented caveat**: trades that *entered* before the cutoff but *exited* after still contribute their realized MFE/MAE. This is a known small look-ahead leak (research call CF1) that was part of the validated 1.96 MAR result. Production callers don't need to do anything about it — just provide whatever completed-trade history you have.

For the very first run on a fresh dataset, an empty DataFrame is acceptable; the brain falls back to `type_prior_default` (1.0) until enough history accumulates.

### Overall regime snapshot (optional)

Returned by `get_overall_regime_snapshot(week_date)` as a `dict | None`.

```python
{"regime": "weakening_bull", "breadth_pct": 62.5}
```

`regime` is one of: `"strong_bull"`, `"weakening_bull"`, `"recovering"`, `"deteriorating"`, `"bear"`. `breadth_pct` is the percentage (0–100) of the universe trading above its SMA-25.

Returns `None` if regime data is not available for the requested week. The brain treats `None` as a regime-gate **block**.

### Sector regime + sector RS (optional)

Two helpers:

* `get_sector_regime_for(week_date, ticker) -> str | None` — Weinstein-style regime for the ticker's sector. One of `"bull"`, `"bear"`, `"sideways"`, or `None`.
* `get_sector_rs_quartile_for(week_date, ticker) -> str | None` — quartile for the ticker's sector by 13-week relative strength. One of `"TOP"`, `"UPPER"`, `"LOWER"`, `"BOTTOM"`, or `None`.

PHASE_4_V1 admits only `"TOP"`-quartile sectors; `None` falls through (no sector data ⇒ no sector veto).

## File layouts for the file-system connectors

### CSV directory layout

```
root/
├── weekly_ohlcv/
│   ├── RELIANCE.NS.csv
│   ├── TCS.NS.csv
│   └── ...
├── daily_ohlcv/
│   ├── RELIANCE.NS.csv
│   └── ...
├── nifty_weekly.csv
├── sector_indices_weekly/
│   ├── NIFTY ENERGY.csv
│   └── ...
├── universe.csv
├── historical_trades.csv
└── overall_regime.csv          # optional
```

OHLCV-shaped CSVs need a `date` column (parsed as datetime) plus the OHLCV columns. The connector sets `date` as the index.

### Parquet directory layout

Identical to CSV but with `.parquet` extensions. Parquet files may either store the date as a regular `date` column or as a `DatetimeIndex` — both work, and `DatetimeIndex` is faster.

### SQL schema (default table names)

Override these names by passing a `TableMap` to `SQLAlchemyDataProvider`.

```sql
-- weekly OHLCV — long form
weekly_ohlcv  (ticker TEXT, date DATE, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT)
daily_ohlcv   (ticker TEXT, date DATE, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT)

-- index time series
nifty_weekly         (date DATE, close FLOAT, ...)
sector_indices_weekly (sector TEXT, date DATE, close FLOAT, ...)

-- snapshots
universe (ticker TEXT, sector TEXT, market_cap FLOAT, adtv_20d FLOAT)
historical_trades (ticker TEXT, week_date DATE, entry_type TEXT, mfe_pct FLOAT, mae_pct FLOAT)

-- optional
overall_regime (date DATE, regime TEXT, breadth_pct FLOAT)
```

OHLCV columns are case-insensitive at the SQL layer — the connector renames lower-case to canonical case after read.
