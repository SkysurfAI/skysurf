"""CsvDataProvider — load data from a directory of CSV files.

Expected layout under ``root``::

    root/
    ├── weekly_ohlcv/<TICKER>.csv
    ├── daily_ohlcv/<TICKER>.csv
    ├── nifty_weekly.csv
    ├── sector_indices_weekly/<SECTOR>.csv
    ├── universe.csv
    ├── historical_trades.csv
    └── overall_regime.csv      (optional)

Each OHLCV-shaped CSV must include a ``date`` column (any pandas-parseable
format) plus the usual ``Open, High, Low, Close, Volume`` columns. The
``date`` column is set as the index. ``universe.csv`` and
``historical_trades.csv`` follow the schemas in
:mod:`skysurf.data.provider`.

Files are read lazily on first access and cached for the lifetime of
the provider instance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skysurf.data._filesystem import _FileSystemDataProvider


class CsvDataProvider(_FileSystemDataProvider):
    """Read tabular data from a directory of CSV files."""

    _extension = "csv"

    def _read_table(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


__all__ = ["CsvDataProvider"]
