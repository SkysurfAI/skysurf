"""ParquetDataProvider — load data from a directory of Parquet files.

Expected layout under ``root``::

    root/
    ├── weekly_ohlcv/<TICKER>.parquet
    ├── daily_ohlcv/<TICKER>.parquet
    ├── nifty_weekly.parquet
    ├── sector_indices_weekly/<SECTOR>.parquet
    ├── universe.parquet
    ├── historical_trades.parquet
    └── overall_regime.parquet      (optional)

Parquet is the recommended production format: smaller on disk, faster to
read, and preserves dtypes (so the ``date`` column or the existing
``DatetimeIndex`` round-trips without manual parsing).

OHLCV-shaped Parquet files may either store the date as a regular
column named ``date`` *or* as a :class:`pandas.DatetimeIndex` — both
work.

Requires :mod:`pyarrow`. Install with ``pip install skysurf[parquet]``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skysurf.data._filesystem import _FileSystemDataProvider


class ParquetDataProvider(_FileSystemDataProvider):
    """Read tabular data from a directory of Parquet files."""

    _extension = "parquet"

    def _read_table(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)


__all__ = ["ParquetDataProvider"]
