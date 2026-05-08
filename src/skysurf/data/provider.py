"""DataProvider — how the brain asks for data.

The brain never reads files, never queries databases, never calls APIs.
It declares what data it needs through the abstract :class:`DataProvider`
contract. Callers implement a concrete subclass against whatever data
store they have.

The schema for every method's return value is documented at the top of
this module. Returns are pandas DataFrames; they are not type-checked at
runtime, so implementations are responsible for matching the schema.

Five reference implementations live in this package:

* :class:`InMemoryDataProvider` — backed by pre-loaded DataFrames
  (this module). Best for tests and minimal-plumbing demos.
* :class:`~skysurf.data.pandas.PandasDataProvider` — backed by user-
  supplied DataFrames (BYO).
* :class:`~skysurf.data.csv.CsvDataProvider` — backed by a directory of
  CSVs in the canonical schema.
* :class:`~skysurf.data.parquet.ParquetDataProvider` — same, but Parquet.
* :class:`~skysurf.data.sqlalchemy.SQLAlchemyDataProvider` — backed by
  any SQL database.

Production callers (e.g., a managed-service runtime) typically write
their own DataProvider that translates brain requests into DB queries
or cached fetches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd

# ── Schema constants ─────────────────────────────────────────────────

WEEKLY_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
"""Columns expected in every weekly-OHLCV DataFrame returned to the
brain. The index is a :class:`pandas.DatetimeIndex` of weekly bars
(Friday-close convention). All columns are floats; ``Volume`` may also
be integral."""

DAILY_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
"""Columns expected in every daily-OHLCV DataFrame. Index is the trading
day."""

NIFTY_WEEKLY_COLUMNS: tuple[str, ...] = ("Close",)
"""At minimum, the Nifty 50 weekly DataFrame must carry a ``Close``
column; other OHLC columns are tolerated and ignored."""

SECTOR_INDEX_WEEKLY_COLUMNS: tuple[str, ...] = ("Close",)
"""Same shape as :data:`NIFTY_WEEKLY_COLUMNS`, per sector index."""

UNIVERSE_COLUMNS: tuple[str, ...] = ("ticker", "sector", "market_cap", "adtv_20d")
"""One row per eligible ticker as of the reference date.

* ``ticker`` — string symbol.
* ``sector`` — string in the Skysurf sector taxonomy.
* ``market_cap`` — float, **absolute INR** (not crores).
* ``adtv_20d`` — float, 20-day average daily traded value in INR.
"""

HISTORICAL_TRADES_COLUMNS: tuple[str, ...] = (
    "ticker",
    "week_date",
    "entry_type",
    "mfe_pct",
    "mae_pct",
)
"""Completed historical trades used for dynamic type-prior computation.

* ``week_date`` — entry week (``pd.Timestamp``).
* ``mfe_pct`` — max favorable excursion (%, never negative).
* ``mae_pct`` — max adverse excursion (%, never positive).

Per the canonical CF1 design (a documented residual look-ahead leak),
the brain filters on entry ``week_date < cutoff``. Trades that entered
before the cutoff but exited after still contribute their realized
MFE/MAE — this is intentional and was part of the validated 1.96 MAR
result.
"""


class OverallRegimeSnapshot(TypedDict):
    """The shape returned by :meth:`DataProvider.get_overall_regime_snapshot`.

    * ``regime`` — one of the five-state classifier labels:
      ``"strong_bull"``, ``"weakening_bull"``, ``"recovering"``,
      ``"deteriorating"``, ``"bear"``.
    * ``breadth_pct`` — percentage of universe above SMA-25 (0–100).
    """

    regime: str
    breadth_pct: float


class DataProvider(ABC):
    """Abstract data-fetch interface used by the brain.

    Each method has an exact return-shape contract documented in its
    docstring. Implementations may cache aggressively, fetch lazily, or
    load all data upfront — the brain is agnostic.
    """

    @abstractmethod
    def get_weekly_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Return weekly OHLCV for ``tickers`` over ``[start, end]``.

        Tickers without data for the window should be **omitted** from
        the result, not included with empty DataFrames. The brain
        typically requests around 104 weeks (2 years) of history to
        cover the longest moving-average period plus a buffer.
        """

    @abstractmethod
    def get_daily_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Return daily OHLCV for ``tickers`` over ``[start, end]``.

        Used primarily for ADTV computation (~20 trading days).
        """

    @abstractmethod
    def get_nifty_weekly(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return Nifty 50 weekly bars over ``[start, end]``."""

    @abstractmethod
    def get_sector_indices_weekly(
        self,
        sectors: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Return per-sector weekly bars over ``[start, end]``.

        Used for sector-regime evaluation and sector-RS quartile
        assignment.
        """

    @abstractmethod
    def get_universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Return the eligible universe as of ``as_of``.

        The brain applies the ``market_cap_floor_inr`` and
        ``adtv_floor_inr`` filters from
        :class:`~skysurf.config.StrategyConfig` on this universe.
        Implementations are responsible for handling listing-based
        eligibility (suspended tickers, recent listings, etc.).
        """

    @abstractmethod
    def get_historical_trades(self, before_date: pd.Timestamp) -> pd.DataFrame:
        """Return all completed strategy trades that *entered* before ``before_date``.

        Exits may post-date the cutoff (the documented CF1 residual
        leak); see the module-level docstring of
        :data:`HISTORICAL_TRADES_COLUMNS`.
        """

    @abstractmethod
    def get_overall_regime_snapshot(self, week_date: pd.Timestamp) -> OverallRegimeSnapshot | None:
        """Return the overall market regime + breadth for ``week_date``.

        Returns ``None`` if the regime has not been computed for that
        week (e.g., the week is in the future or before regime data
        starts). The brain treats ``None`` as a regime-gate block.
        """

    @abstractmethod
    def get_sector_regime_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        """Return the sector-level Weinstein regime for ``ticker`` this week.

        Returns one of ``"bull"``, ``"bear"``, ``"sideways"``, or
        ``None`` if not classifiable (no sector index data, ticker
        missing, etc.). Used by the ``OVERALL_OR_SECTOR`` combined
        regime gate.
        """

    @abstractmethod
    def get_sector_rs_quartile_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        """Return the sector-RS quartile for ``ticker`` this week.

        Returns one of ``"TOP"``, ``"UPPER"``, ``"LOWER"``,
        ``"BOTTOM"``, or ``None`` if unclassifiable. Used by the
        ``sector_rs_filter`` gate (PHASE_4_V1 admits only ``"TOP"``;
        ``None`` falls through per research convention).
        """

    def is_ticker_in_universe_for(self, ticker: str, week_date: pd.Timestamp) -> bool:
        """Return whether ``ticker`` qualifies in the universe at ``week_date``.

        Default returns ``True``. Concrete providers may override to
        enforce per-week universe filtering — for example, a
        research-style quarterly Nifty-scaled market-cap floor.
        """
        return True


def _quarter_str(date: pd.Timestamp) -> str:
    """Format ``date`` as ``"YYYYQn"`` (e.g., ``"2024Q2"``)."""
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year}Q{quarter}"


@dataclass
class InMemoryDataProvider(DataProvider):
    """Reference DataProvider holding pre-loaded DataFrames.

    Use this for tests and demos. Construct with the full dataset
    upfront; queries slice the in-memory data.

    Inputs are expected to satisfy the schema contracts documented at
    module level. No validation is performed at construction — invalid
    inputs surface as ``KeyError`` or ``AttributeError`` when the brain
    reads them.

    Optional fields (default ``None``) support the regime / sector
    queries used in the public-API path. Tests that don't exercise the
    regime gate can omit them.
    """

    weekly_ohlcv: dict[str, pd.DataFrame]
    daily_ohlcv: dict[str, pd.DataFrame]
    nifty_weekly: pd.DataFrame
    sector_indices_weekly: dict[str, pd.DataFrame]
    universe: pd.DataFrame
    historical_trades: pd.DataFrame

    overall_regime_df: pd.DataFrame | None = None
    """Optional. DataFrame with columns ``[regime, breadth_pct]`` indexed by week."""

    sector_regime_lookup: dict[tuple[str, str], str] | None = None
    """Optional. Map ``(date_str, index_symbol) → regime``."""

    sector_rs_lookup: dict[tuple[str, str], str] | None = None
    """Optional. Map ``(date_str, index_symbol) → quartile``."""

    captier_regime_lookup: dict[tuple[str, str], str] | None = None
    """Optional. Map ``(date_str, cap_tier) → regime``."""

    ticker_metadata: dict[str, dict[str, str]] | None = None
    """Optional. Map ``ticker → {primary_sector, index_symbol, ...}``."""

    quarterly_universe: dict[str, set[str]] | None = field(default=None)
    """Optional. If set, :meth:`get_universe` filters to qualifying tickers
    for the given quarter (research-parity behaviour)."""

    def get_weekly_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.weekly_ohlcv.get(ticker)
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
            df = self.daily_ohlcv.get(ticker)
            if df is None or df.empty:
                continue
            sliced = df.loc[start:end]
            if not sliced.empty:
                out[ticker] = sliced
        return out

    def get_nifty_weekly(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.nifty_weekly.loc[start:end]

    def get_sector_indices_weekly(
        self,
        sectors: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sector in sectors:
            df = self.sector_indices_weekly.get(sector)
            if df is None or df.empty:
                continue
            sliced = df.loc[start:end]
            if not sliced.empty:
                out[sector] = sliced
        return out

    def get_universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        if self.quarterly_universe is None:
            return self.universe
        qkey = _quarter_str(pd.Timestamp(as_of))
        qualifying = self.quarterly_universe.get(qkey)
        if qualifying is None:
            return self.universe
        return self.universe[self.universe["ticker"].isin(qualifying)]

    def get_historical_trades(self, before_date: pd.Timestamp) -> pd.DataFrame:
        return self.historical_trades[self.historical_trades["week_date"] < before_date]

    def get_overall_regime_snapshot(self, week_date: pd.Timestamp) -> OverallRegimeSnapshot | None:
        if self.overall_regime_df is None:
            return None
        if week_date not in self.overall_regime_df.index:
            return None
        row = self.overall_regime_df.loc[week_date]
        regime = row.get("regime")
        breadth = row.get("breadth_pct")
        if regime is None or pd.isna(regime):
            return None
        return {
            "regime": str(regime),
            "breadth_pct": float(breadth) if breadth is not None and not pd.isna(breadth) else 0.0,
        }

    def get_sector_regime_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        if self.sector_regime_lookup is None or self.ticker_metadata is None:
            return None
        meta = self.ticker_metadata.get(ticker)
        if meta is None:
            return None
        index_symbol = meta.get("index_symbol")
        if not index_symbol:
            return None
        return self.sector_regime_lookup.get((str(week_date)[:10], index_symbol))

    def is_ticker_in_universe_for(self, ticker: str, week_date: pd.Timestamp) -> bool:
        if self.quarterly_universe is None:
            return True
        qkey = _quarter_str(pd.Timestamp(week_date))
        qualifying = self.quarterly_universe.get(qkey)
        if qualifying is None:
            return True
        return ticker in qualifying

    def get_sector_rs_quartile_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        if self.sector_rs_lookup is None or self.ticker_metadata is None:
            return None
        meta = self.ticker_metadata.get(ticker)
        if meta is None:
            return None
        index_symbol = meta.get("index_symbol")
        if not index_symbol:
            return None
        return self.sector_rs_lookup.get((str(week_date)[:10], index_symbol))
