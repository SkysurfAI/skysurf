"""SQLAlchemyDataProvider — load data from any SQL database.

Wraps a SQLAlchemy :class:`~sqlalchemy.engine.Engine` (or connection
URL) and reads the brain's required tables on demand. Works with
PostgreSQL, MySQL / MariaDB, SQLite, and any other dialect supported by
SQLAlchemy.

Default table names match the canonical schema; pass a
:class:`TableMap` to override.

Default schema (one row per bar / per ticker / etc.):

* ``weekly_ohlcv(ticker, date, open, high, low, close, volume)``
* ``daily_ohlcv(ticker, date, open, high, low, close, volume)``
* ``nifty_weekly(date, close, ...)``
* ``sector_indices_weekly(sector, date, close, ...)``
* ``universe(ticker, sector, market_cap, adtv_20d)``
* ``historical_trades(ticker, week_date, entry_type, mfe_pct, mae_pct)``
* ``overall_regime(date, regime, breadth_pct)``  *(optional)*

Requires :mod:`sqlalchemy`. Install with ``pip install skysurf[sqlalchemy]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from skysurf.data.provider import DataProvider, OverallRegimeSnapshot

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class TableMap:
    """Override the default table names used by :class:`SQLAlchemyDataProvider`.

    All fields default to the canonical names; override only the ones
    your schema differs on.
    """

    weekly_ohlcv: str = "weekly_ohlcv"
    daily_ohlcv: str = "daily_ohlcv"
    nifty_weekly: str = "nifty_weekly"
    sector_indices_weekly: str = "sector_indices_weekly"
    universe: str = "universe"
    historical_trades: str = "historical_trades"
    overall_regime: str = "overall_regime"


# OHLCV column names case-fold to match the canonical schema. Some
# back-ends store them lower-case; we normalize after read.
_OHLCV_RENAME = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


class SQLAlchemyDataProvider(DataProvider):
    """Read tabular data from any SQL database via SQLAlchemy."""

    def __init__(
        self,
        engine_or_url: Engine | str,
        tables: TableMap | None = None,
    ) -> None:
        """Initialize from a SQLAlchemy engine or connection URL.

        Args:
            engine_or_url: Either a live :class:`~sqlalchemy.engine.Engine`
                or a connection URL string (e.g.,
                ``"postgresql://user:pass@host/db"`` or
                ``"sqlite:///./skysurf.db"``).
            tables: Optional :class:`TableMap` overriding default table
                names.

        Raises:
            ImportError: If :mod:`sqlalchemy` is not installed.
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine import Engine as _Engine
        except ImportError as exc:
            raise ImportError(
                "SQLAlchemyDataProvider requires sqlalchemy. "
                "Install with `pip install skysurf[sqlalchemy]`."
            ) from exc

        if isinstance(engine_or_url, str):
            self._engine = create_engine(engine_or_url)
        elif isinstance(engine_or_url, _Engine):
            self._engine = engine_or_url
        else:
            raise TypeError(
                f"engine_or_url must be a SQLAlchemy Engine or URL string; "
                f"got {type(engine_or_url).__name__}"
            )

        self._tables = tables or TableMap()

    # ── DataProvider implementation ───────────────────────────────────

    def get_weekly_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        return self._read_per_ticker_ohlcv(self._tables.weekly_ohlcv, tickers, start, end)

    def get_daily_ohlcv(
        self,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        return self._read_per_ticker_ohlcv(self._tables.daily_ohlcv, tickers, start, end)

    def get_nifty_weekly(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        sql = (
            f"SELECT * FROM {self._tables.nifty_weekly} "
            f"WHERE date >= :start AND date <= :end ORDER BY date"
        )
        df = self._read_sql(sql, params={"start": _iso(start), "end": _iso(end)})
        return self._index_by_date(df)

    def get_sector_indices_weekly(
        self,
        sectors: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        sectors_list = list(sectors)
        if not sectors_list:
            return {}
        # Use IN-clause via parameter expansion (SQLAlchemy text + bindparam)
        from sqlalchemy import bindparam, text

        stmt = text(
            f"SELECT * FROM {self._tables.sector_indices_weekly} "
            f"WHERE sector IN :sectors AND date >= :start AND date <= :end "
            f"ORDER BY sector, date"
        ).bindparams(bindparam("sectors", expanding=True))

        params: dict[str, object] = {
            "sectors": sectors_list,
            "start": _iso(start),
            "end": _iso(end),
        }
        with self._engine.connect() as conn:
            df: pd.DataFrame = pd.read_sql(stmt, conn, params=params)  # type: ignore[arg-type]
        if df.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for sector, group in df.groupby("sector"):
            out[str(sector)] = self._index_by_date(group.drop(columns=["sector"]))
        return out

    def get_universe(self, as_of: pd.Timestamp) -> pd.DataFrame:
        sql = f"SELECT ticker, sector, market_cap, adtv_20d FROM {self._tables.universe}"
        return self._read_sql(sql)

    def get_historical_trades(self, before_date: pd.Timestamp) -> pd.DataFrame:
        sql = (
            f"SELECT ticker, week_date, entry_type, mfe_pct, mae_pct "
            f"FROM {self._tables.historical_trades} "
            f"WHERE week_date < :before ORDER BY week_date"
        )
        df = self._read_sql(sql, params={"before": _iso(before_date)})
        if not df.empty:
            df["week_date"] = pd.to_datetime(df["week_date"])
        return df

    def get_overall_regime_snapshot(self, week_date: pd.Timestamp) -> OverallRegimeSnapshot | None:
        sql = (
            f"SELECT regime, breadth_pct FROM {self._tables.overall_regime} "
            f"WHERE date = :date LIMIT 1"
        )
        try:
            df = self._read_sql(sql, params={"date": _iso(week_date)})
        except Exception:
            return None
        if df.empty:
            return None
        row = df.iloc[0]
        regime = row.get("regime")
        breadth = row.get("breadth_pct")
        if regime is None or pd.isna(regime):
            return None
        return {
            "regime": str(regime),
            "breadth_pct": float(breadth) if breadth is not None and not pd.isna(breadth) else 0.0,
        }

    def get_sector_regime_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        return None

    def get_sector_rs_quartile_for(self, week_date: pd.Timestamp, ticker: str) -> str | None:
        return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _read_per_ticker_ohlcv(
        self,
        table: str,
        tickers: Iterable[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        tickers_list = list(tickers)
        if not tickers_list:
            return {}
        from sqlalchemy import bindparam, text

        stmt = text(
            f"SELECT * FROM {table} "
            f"WHERE ticker IN :tickers AND date >= :start AND date <= :end "
            f"ORDER BY ticker, date"
        ).bindparams(bindparam("tickers", expanding=True))

        params: dict[str, object] = {
            "tickers": tickers_list,
            "start": _iso(start),
            "end": _iso(end),
        }
        with self._engine.connect() as conn:
            df: pd.DataFrame = pd.read_sql(stmt, conn, params=params)  # type: ignore[arg-type]
        if df.empty:
            return {}

        out: dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker"):
            out[str(ticker)] = self._index_by_date(group.drop(columns=["ticker"]))
        return out

    def _read_sql(
        self,
        sql: str,
        params: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            # pandas-stubs is strict about the params shape; SQLAlchemy
            # accepts a plain dict at runtime so the cast is safe.
            df: pd.DataFrame = pd.read_sql(text(sql), conn, params=params or {})  # type: ignore[arg-type]
            return df

    @staticmethod
    def _index_by_date(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        # Normalize lower-case OHLCV column names if present
        rename = {k: v for k, v in _OHLCV_RENAME.items() if k in df.columns}
        if rename:
            df = df.rename(columns=rename)
        return df


def _iso(value: pd.Timestamp | str) -> str:
    """Return an ISO-formatted string suitable for SQL bind parameters.

    SQLite doesn't bind Python ``datetime`` / ``pd.Timestamp`` directly,
    so we serialize at the boundary. PostgreSQL and MySQL accept both
    forms; ISO strings work for everyone.
    """
    if isinstance(value, str):
        return value
    return str(pd.Timestamp(value).isoformat())


__all__ = ["SQLAlchemyDataProvider", "TableMap"]
