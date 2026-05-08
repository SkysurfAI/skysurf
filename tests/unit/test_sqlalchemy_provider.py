"""Tests for SQLAlchemyDataProvider — uses an in-memory SQLite DB."""

from __future__ import annotations

import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from skysurf import SQLAlchemyDataProvider  # noqa: E402
from skysurf.data import TableMap  # noqa: E402


@pytest.fixture
def sqlite_engine_with_data(
    synthetic_weekly_ohlcv: dict[str, pd.DataFrame],
    synthetic_nifty_weekly: pd.DataFrame,
    synthetic_universe: pd.DataFrame,
):
    """Create an in-memory SQLite engine and populate the canonical tables.

    Materializes weekly_ohlcv (long form), nifty_weekly, universe, and
    historical_trades. Daily OHLCV is omitted to keep the fixture small.
    """
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")

    # weekly_ohlcv as a long table: (ticker, date, Open, High, Low, Close, Volume)
    rows = []
    for ticker, df in synthetic_weekly_ohlcv.items():
        long = df.reset_index().rename(columns={df.index.name or "index": "date"})
        long["ticker"] = ticker
        rows.append(long)
    weekly_long = pd.concat(rows, ignore_index=True)
    # Lower-case the OHLCV columns to exercise the rename path
    weekly_long = weekly_long.rename(
        columns={c: c.lower() for c in ("Open", "High", "Low", "Close", "Volume")}
    )
    weekly_long.to_sql("weekly_ohlcv", engine, index=False)
    weekly_long.to_sql("daily_ohlcv", engine, index=False)

    # nifty_weekly
    nifty = synthetic_nifty_weekly.reset_index().rename(
        columns={synthetic_nifty_weekly.index.name or "index": "date"}
    )
    nifty.to_sql("nifty_weekly", engine, index=False)

    # universe
    synthetic_universe.to_sql("universe", engine, index=False)

    # historical_trades — empty
    pd.DataFrame(columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]).to_sql(
        "historical_trades", engine, index=False
    )

    return engine


@pytest.fixture
def sqla_provider(sqlite_engine_with_data) -> SQLAlchemyDataProvider:
    return SQLAlchemyDataProvider(sqlite_engine_with_data)


def test_init_accepts_url_string() -> None:
    provider = SQLAlchemyDataProvider("sqlite:///:memory:")
    assert provider is not None


def test_init_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="Engine"):
        SQLAlchemyDataProvider(12345)  # type: ignore[arg-type]


def test_init_with_custom_table_map() -> None:
    provider = SQLAlchemyDataProvider(
        "sqlite:///:memory:",
        tables=TableMap(weekly_ohlcv="my_weekly_bars"),
    )
    assert provider._tables.weekly_ohlcv == "my_weekly_bars"


def test_get_weekly_ohlcv_returns_present_tickers(sqla_provider, friday) -> None:
    out = sqla_provider.get_weekly_ohlcv(
        ["ALPHA.NS", "ZETA.NS"],
        start=friday - pd.Timedelta(weeks=20),
        end=friday,
    )
    assert "ALPHA.NS" in out
    assert "ZETA.NS" not in out
    df = out["ALPHA.NS"]
    # OHLCV column names should be normalized to canonical case
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_get_weekly_ohlcv_empty_ticker_list(sqla_provider, friday) -> None:
    out = sqla_provider.get_weekly_ohlcv([], friday - pd.Timedelta(weeks=10), friday)
    assert out == {}


def test_get_nifty_weekly_returns_close(sqla_provider, friday) -> None:
    df = sqla_provider.get_nifty_weekly(
        friday - pd.Timedelta(weeks=10),
        friday,
    )
    assert "Close" in df.columns
    assert not df.empty


def test_get_universe_returns_canonical_columns(sqla_provider, friday) -> None:
    df = sqla_provider.get_universe(friday)
    for col in ("ticker", "sector", "market_cap", "adtv_20d"):
        assert col in df.columns


def test_get_historical_trades_when_empty(sqla_provider, friday) -> None:
    df = sqla_provider.get_historical_trades(before_date=friday)
    assert df.empty


def test_get_overall_regime_snapshot_returns_none_when_table_missing(
    sqla_provider,
    friday,
) -> None:
    """No overall_regime table created → returns None without raising."""
    assert sqla_provider.get_overall_regime_snapshot(friday) is None


def test_sqlalchemy_import_error_message_when_unavailable(monkeypatch) -> None:
    """If sqlalchemy is gone at runtime, init raises a helpful ImportError.

    Simulated by hiding sqlalchemy from import."""
    import sys

    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", None)
    with pytest.raises(ImportError, match="sqlalchemy"):
        SQLAlchemyDataProvider("sqlite:///:memory:")
