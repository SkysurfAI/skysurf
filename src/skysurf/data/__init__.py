"""Data-provider abstraction and reference connectors.

The brain consumes data through the abstract :class:`DataProvider`
contract. Pick the connector that matches your data shape:

* :class:`InMemoryDataProvider` — pre-loaded DataFrames (this module).
* :class:`~skysurf.data.pandas.PandasDataProvider` — same, exposed under
  a friendlier name when you want to BYO DataFrames.
* :class:`~skysurf.data.csv.CsvDataProvider` — directory of CSV files.
* :class:`~skysurf.data.parquet.ParquetDataProvider` — directory of
  Parquet files (recommended for production).
* :class:`~skysurf.data.sqlalchemy.SQLAlchemyDataProvider` — any SQL DB.

See ``docs/connectors.md`` for the decision tree and
``docs/data-schema.md`` for the canonical column names.
"""

from __future__ import annotations

from skysurf.data.csv import CsvDataProvider
from skysurf.data.pandas import PandasDataProvider
from skysurf.data.parquet import ParquetDataProvider
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
from skysurf.data.sqlalchemy import SQLAlchemyDataProvider, TableMap

__all__ = [
    "DAILY_OHLCV_COLUMNS",
    "HISTORICAL_TRADES_COLUMNS",
    "NIFTY_WEEKLY_COLUMNS",
    "SECTOR_INDEX_WEEKLY_COLUMNS",
    "UNIVERSE_COLUMNS",
    "WEEKLY_OHLCV_COLUMNS",
    "CsvDataProvider",
    "DataProvider",
    "InMemoryDataProvider",
    "OverallRegimeSnapshot",
    "PandasDataProvider",
    "ParquetDataProvider",
    "SQLAlchemyDataProvider",
    "TableMap",
]
