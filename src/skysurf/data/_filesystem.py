"""Shared base class for file-system DataProviders (CSV and Parquet).

Both the CSV and the Parquet connector load the same on-disk layout —
only the file format differs. This module factors out the layout logic
so the format-specific subclasses are tiny.

Layout (under ``root``)::

    root/
    ├── weekly_ohlcv/<TICKER>.<ext>
    ├── daily_ohlcv/<TICKER>.<ext>
    ├── nifty_weekly.<ext>
    ├── sector_indices_weekly/<SECTOR>.<ext>
    ├── universe.<ext>
    ├── historical_trades.<ext>
    └── overall_regime.<ext>            (optional)

For tabular formats with a wide row layout (CSV, Parquet, Feather), all
OHLCV files must have a ``date`` column that becomes the row index.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from skysurf.data.provider import DataProvider, OverallRegimeSnapshot

_LOG = logging.getLogger(__name__)


class _FileSystemDataProvider(DataProvider):
    """Common scaffolding for CSV / Parquet / Feather-style providers.

    Subclasses implement :meth:`_read_table` (read a path → DataFrame)
    and override :attr:`_extension` (e.g., ``"csv"``, ``"parquet"``).
    Everything else is shared.
    """

    _extension: str = ""
    """File extension *without* a leading dot (e.g., ``"csv"``)."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"{type(self).__name__} root not found: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"{type(self).__name__} root is not a directory: {root_path}")
        self._root = root_path
        self._weekly_cache: dict[str, pd.DataFrame] = {}
        self._daily_cache: dict[str, pd.DataFrame] = {}
        self._sector_cache: dict[str, pd.DataFrame] = {}
        self._nifty_cache: pd.DataFrame | None = None
        self._universe_cache: pd.DataFrame | None = None
        self._trades_cache: pd.DataFrame | None = None
        self._overall_regime_cache: pd.DataFrame | None = None

    # ── Subclass hook ─────────────────────────────────────────────────

    @abstractmethod
    def _read_table(self, path: Path) -> pd.DataFrame:
        """Read a single file at ``path`` and return a DataFrame.

        Subclasses are responsible for any format-specific parsing
        (e.g., dtype hints, compression). For OHLCV-shaped tables the
        returned frame will be passed to :meth:`_apply_date_index`.
        """

    # ── DataProvider implementation ──────────────────────────────────

    def get_weekly_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._load_weekly(ticker)
            if df is None or df.empty:
                continue
            sliced = df.loc[start:end]
            if not sliced.empty:
                out[ticker] = sliced
        return out

    def get_daily_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._load_daily(ticker)
            if df is None or df.empty:
                continue
            sliced = df.loc[start:end]
            if not sliced.empty:
                out[ticker] = sliced
        return out

    def get_nifty_weekly(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self._nifty_cache is None:
            self._nifty_cache = self._apply_date_index(
                self._read_table(self._path_for("nifty_weekly"))
            )
        return self._nifty_cache.loc[start:end]

    def get_sector_indices_weekly(
        self,
        sectors: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sector in sectors:
            df = self._load_sector(sector)
            if df is None or df.empty:
                continue
            sliced = df.loc[start:end]
            if not sliced.empty:
                out[sector] = sliced
        return out

    def get_universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if self._universe_cache is None:
            self._universe_cache = self._read_table(self._path_for("universe"))
        return self._universe_cache

    def get_historical_trades(self, before_date: pd.Timestamp) -> pd.DataFrame:
        if self._trades_cache is None:
            df = self._read_table(self._path_for("historical_trades"))
            df["week_date"] = pd.to_datetime(df["week_date"])
            self._trades_cache = df
        return self._trades_cache[self._trades_cache["week_date"] < before_date]

    def get_overall_regime_snapshot(self, week_date: pd.Timestamp) -> OverallRegimeSnapshot | None:
        if self._overall_regime_cache is None:
            path = self._path_for("overall_regime")
            if not path.exists():
                return None
            self._overall_regime_cache = self._apply_date_index(self._read_table(path))
        df = self._overall_regime_cache
        if week_date not in df.index:
            return None
        row = df.loc[week_date]
        regime = row.get("regime")
        breadth = row.get("breadth_pct")
        if regime is None or pd.isna(regime):
            return None
        return {
            "regime": str(regime),
            "breadth_pct": float(breadth) if breadth is not None and not pd.isna(breadth) else 0.0,
        }

    def get_sector_regime_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        # The file-system connectors do not ship sector-regime lookups
        # by default (the canonical research dataset stores them in a
        # cross-tabulated shape that doesn't fit one file per sector).
        # Override this method in a subclass to wire your own.
        return None

    def get_sector_rs_quartile_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        return None

    # ── Internal helpers ──────────────────────────────────────────────

    def _path_for(self, basename: str) -> Path:
        return self._root / f"{basename}.{self._extension}"

    def _load_weekly(self, ticker: str) -> pd.DataFrame | None:
        if ticker in self._weekly_cache:
            return self._weekly_cache[ticker]
        path = self._root / "weekly_ohlcv" / f"{ticker}.{self._extension}"
        if not path.exists():
            _LOG.debug("weekly OHLCV missing for %s at %s", ticker, path)
            return None
        df = self._apply_date_index(self._read_table(path))
        self._weekly_cache[ticker] = df
        return df

    def _load_daily(self, ticker: str) -> pd.DataFrame | None:
        if ticker in self._daily_cache:
            return self._daily_cache[ticker]
        path = self._root / "daily_ohlcv" / f"{ticker}.{self._extension}"
        if not path.exists():
            return None
        df = self._apply_date_index(self._read_table(path))
        self._daily_cache[ticker] = df
        return df

    def _load_sector(self, sector: str) -> pd.DataFrame | None:
        if sector in self._sector_cache:
            return self._sector_cache[sector]
        path = self._root / "sector_indices_weekly" / f"{sector}.{self._extension}"
        if not path.exists():
            return None
        df = self._apply_date_index(self._read_table(path))
        self._sector_cache[sector] = df
        return df

    @staticmethod
    def _apply_date_index(df: pd.DataFrame) -> pd.DataFrame:
        """If ``df`` has a ``date`` column, set it as a sorted DatetimeIndex.

        If the DataFrame already has a ``DatetimeIndex`` (e.g., a
        Parquet file that stored the index), it is returned as-is.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        if "date" not in df.columns:
            raise ValueError(
                "DataFrame is missing a 'date' column and is not "
                "DatetimeIndex'd; cannot determine bar index"
            )
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
