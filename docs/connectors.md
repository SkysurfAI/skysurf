# Connectors

Connectors are how Skysurf reads your data. Pick one that matches your shape; if none fit, write your own — they're small (50–200 lines).

## Decision tree

```
Where does your OHLCV live?
│
├── In memory as pandas DataFrames already
│   └── PandasDataProvider          (zero config)
│
├── In a directory of CSVs
│   └── CsvDataProvider             (point at the directory)
│
├── In a directory of Parquet files
│   └── ParquetDataProvider         (point at the directory; recommended for prod)
│
├── In a SQL database (Postgres / MySQL / SQLite / ...)
│   └── SQLAlchemyDataProvider      (pass a connection URL)
│
└── Somewhere else (broker SDK, REST API, Redis, ...)
    └── Subclass DataProvider yourself
```

All connectors satisfy the same `DataProvider` contract, so you can swap them without changing the rest of your code.

## Bundled connectors

### `InMemoryDataProvider` — for tests and demos

```python
from skysurf import InMemoryDataProvider

provider = InMemoryDataProvider(
    weekly_ohlcv={"ALPHA.NS": weekly_df, ...},
    daily_ohlcv={"ALPHA.NS": daily_df, ...},
    nifty_weekly=nifty_df,
    sector_indices_weekly={"TECH": tech_df, ...},
    universe=universe_df,
    historical_trades=trades_df,
)
```

Best for unit tests and tutorials. The synthetic-data fixtures in `tests/conftest.py` use this connector.

### `PandasDataProvider` — bring your own DataFrames

Same interface as `InMemoryDataProvider`, exposed under a more discoverable name. Use this if your data already lives in pandas in your application — no I/O required.

### `CsvDataProvider` — directory of CSVs

```python
from skysurf import CsvDataProvider

provider = CsvDataProvider("/path/to/data")
```

Files are read lazily on first access and cached. See [data-schema.md](data-schema.md) for the expected directory layout.

Good for: getting started fast, sharing a dataset with a teammate, version-controlling your data alongside your code.

Avoid for: production. CSVs lose dtypes, are slow to parse, and are large on disk.

### `ParquetDataProvider` — directory of Parquet files

```python
from skysurf import ParquetDataProvider

provider = ParquetDataProvider("/path/to/parquet")
```

Same interface as the CSV connector, swapping the file format. Requires `pyarrow`:

```bash
pip install skysurf[parquet]
```

Good for: production. Parquet preserves dtypes, is small on disk, and reads ~10x faster than CSV. Data scientists usually have Parquet pipelines already.

### `SQLAlchemyDataProvider` — any SQL database

```python
from skysurf import SQLAlchemyDataProvider

# Postgres
provider = SQLAlchemyDataProvider("postgresql://user:pass@host/db")

# Local SQLite
provider = SQLAlchemyDataProvider("sqlite:///./skysurf.db")

# Or pass a pre-configured engine
from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg://...", pool_size=5)
provider = SQLAlchemyDataProvider(engine)
```

If your tables are named differently from the canonical defaults, pass a `TableMap`:

```python
from skysurf.data import TableMap

provider = SQLAlchemyDataProvider(
    "postgresql://...",
    tables=TableMap(
        weekly_ohlcv="market_data_weekly",
        daily_ohlcv="market_data_daily",
        universe="security_master",
    ),
)
```

Requires `sqlalchemy`:

```bash
pip install skysurf[sqlalchemy]
```

Good for: production with an existing data pipeline. The connector uses parameterized queries (no SQL injection risk) and connection pooling via SQLAlchemy.

## Writing your own connector

Subclass `DataProvider` and implement the eight required methods. Skeleton:

```python
from collections.abc import Iterable

import pandas as pd
from skysurf import DataProvider
from skysurf.data import OverallRegimeSnapshot


class MyKiteConnector(DataProvider):
    """Read OHLCV from Zerodha Kite Connect."""

    def __init__(self, kite_client, ...):
        self._kite = kite_client
        # ...

    def get_weekly_ohlcv(self, tickers, start, end):
        out = {}
        for t in tickers:
            df = self._fetch_kite(t, start, end, interval="week")
            if not df.empty:
                out[t] = df
        return out

    def get_daily_ohlcv(self, tickers, start, end): ...
    def get_nifty_weekly(self, start, end): ...
    def get_sector_indices_weekly(self, sectors, start, end): ...
    def get_universe(self, as_of): ...
    def get_historical_trades(self, before_date): ...
    def get_overall_regime_snapshot(self, week_date): ...
    def get_sector_regime_for(self, week_date, ticker): ...
    def get_sector_rs_quartile_for(self, week_date, ticker): ...
```

The contract test suite in `tests/` runs the same assertions against every bundled connector. Run those tests against your custom connector too — that's the cheapest way to catch schema drift.

### Tips for writing connectors

* **Cache aggressively.** The brain calls every method many times within a single `generate_weekly_signals` call. A LRU cache on `get_weekly_ohlcv(ticker, start, end)` saves a lot of round-trips.
* **Be lenient on input shapes, strict on output.** Your code will see strange ticker formats, weird date types, missing columns — accept gracefully. But always return the canonical schema.
* **Return `None` rather than raising** for the regime / sector lookups. The brain treats `None` as a graceful "no data, fall through" signal.
* **Don't include partial weeks.** If a ticker has data through Tuesday and you're queried for the Friday close, omit it for that week.

## Combining connectors

A common production pattern: weekly OHLCV from Parquet (cheap to refresh nightly), live regime data from SQL (computed by another job), historical trades from a small CSV. You can do this with a thin facade:

```python
from skysurf import (
    DataProvider, ParquetDataProvider, SQLAlchemyDataProvider, CsvDataProvider,
)


class CompositeProvider(DataProvider):
    def __init__(self, ohlcv_root, sql_url, trades_csv_root):
        self._ohlcv = ParquetDataProvider(ohlcv_root)
        self._sql = SQLAlchemyDataProvider(sql_url)
        self._trades = CsvDataProvider(trades_csv_root)

    def get_weekly_ohlcv(self, *args, **kwargs):
        return self._ohlcv.get_weekly_ohlcv(*args, **kwargs)

    def get_daily_ohlcv(self, *args, **kwargs):
        return self._ohlcv.get_daily_ohlcv(*args, **kwargs)

    def get_nifty_weekly(self, *args, **kwargs):
        return self._ohlcv.get_nifty_weekly(*args, **kwargs)

    def get_sector_indices_weekly(self, *args, **kwargs):
        return self._ohlcv.get_sector_indices_weekly(*args, **kwargs)

    def get_universe(self, *args, **kwargs):
        return self._sql.get_universe(*args, **kwargs)

    def get_historical_trades(self, *args, **kwargs):
        return self._trades.get_historical_trades(*args, **kwargs)

    def get_overall_regime_snapshot(self, *args, **kwargs):
        return self._sql.get_overall_regime_snapshot(*args, **kwargs)

    def get_sector_regime_for(self, *args, **kwargs):
        return self._sql.get_sector_regime_for(*args, **kwargs)

    def get_sector_rs_quartile_for(self, *args, **kwargs):
        return self._sql.get_sector_rs_quartile_for(*args, **kwargs)
```

## What we don't ship (and why)

* **Zerodha Kite / Truedata / Dhan adapters.** They each have their own SDK update cycles, paid auth tokens, and idiosyncrasies. Pinning to a particular SDK version in the library would couple users' upgrade timelines to ours, and embedding broker auth would create a security liability. Easier to copy-paste 50 lines and own them yourself.
* **Yfinance adapter.** Yfinance's Indian-equity coverage has known corporate-action adjustment issues. Using it for production strategy execution is a foot-gun. We deliberately don't make it easy.
* **Live data feeds (websockets).** Out of scope — Skysurf operates on weekly bars on Friday close.

If you write a high-quality adapter for one of the above and would like to contribute it as `skysurf-kite` (a separate package), open an issue.
