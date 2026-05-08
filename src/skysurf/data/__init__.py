"""Data-provider abstraction and reference connectors.

The brain consumes data through the abstract :class:`DataProvider`
contract. Pick the connector that matches your data shape:

* :class:`InMemoryDataProvider` — pre-loaded DataFrames; best for tests.
* (additional connectors are added in :mod:`skysurf.data.pandas`,
  :mod:`skysurf.data.csv`, :mod:`skysurf.data.parquet`,
  :mod:`skysurf.data.sqlalchemy`)

See ``docs/connectors.md`` for the decision tree and
``docs/data-schema.md`` for the canonical column names.
"""
from __future__ import annotations

from skysurf.data.provider import (
    DAILY_OHLCV_COLUMNS,
    HISTORICAL_TRADES_COLUMNS,
    NIFTY_WEEKLY_COLUMNS,
    SECTOR_INDEX_WEEKLY_COLUMNS,
    UNIVERSE_COLUMNS,
    WEEKLY_OHLCV_COLUMNS,
    DataProvider,
    InMemoryDataProvider,
    OverallRegimeSnapshot,
)

__all__ = [
    "DAILY_OHLCV_COLUMNS",
    "HISTORICAL_TRADES_COLUMNS",
    "NIFTY_WEEKLY_COLUMNS",
    "SECTOR_INDEX_WEEKLY_COLUMNS",
    "UNIVERSE_COLUMNS",
    "WEEKLY_OHLCV_COLUMNS",
    "DataProvider",
    "InMemoryDataProvider",
    "OverallRegimeSnapshot",
]
