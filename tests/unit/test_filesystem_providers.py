"""Tests for CsvDataProvider and ParquetDataProvider — file-system connectors.

Both connectors share the ``_FileSystemDataProvider`` base, so the same
test cases run against both via parametrization. Synthetic data is
written to a tmp_path on the fly; nothing real is shipped with the repo.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from skysurf import CsvDataProvider, ParquetDataProvider

# Skip the parquet tests if pyarrow isn't installed
pyarrow = pytest.importorskip("pyarrow")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


@pytest.fixture(
    params=[
        ("csv", CsvDataProvider, _write_csv),
        ("parquet", ParquetDataProvider, _write_parquet),
    ],
    ids=["csv", "parquet"],
)
def filesystem_provider_factory(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    synthetic_weekly_ohlcv: dict[str, pd.DataFrame],
    synthetic_nifty_weekly: pd.DataFrame,
    synthetic_universe: pd.DataFrame,
    synthetic_historical_trades: pd.DataFrame,
):
    """Materialize synthetic data on disk in the requested format and
    return an initialized provider."""
    extension, provider_class, writer = request.param
    writer: Callable[[pd.DataFrame, Path], None]

    weekly_dir = tmp_path / "weekly_ohlcv"
    weekly_dir.mkdir()
    for ticker, df in synthetic_weekly_ohlcv.items():
        df_with_idx = df.copy()
        df_with_idx.index.name = "date"
        if extension == "csv":
            writer(df_with_idx.reset_index(), weekly_dir / f"{ticker}.csv")
        else:
            writer(df_with_idx, weekly_dir / f"{ticker}.parquet")

    daily_dir = tmp_path / "daily_ohlcv"
    daily_dir.mkdir()
    for ticker, df in synthetic_weekly_ohlcv.items():
        df_with_idx = df.copy()
        df_with_idx.index.name = "date"
        if extension == "csv":
            writer(df_with_idx.reset_index(), daily_dir / f"{ticker}.csv")
        else:
            writer(df_with_idx, daily_dir / f"{ticker}.parquet")

    nifty_path = tmp_path / f"nifty_weekly.{extension}"
    nifty = synthetic_nifty_weekly.copy()
    nifty.index.name = "date"
    if extension == "csv":
        writer(nifty.reset_index(), nifty_path)
    else:
        writer(nifty, nifty_path)

    if extension == "csv":
        synthetic_universe.to_csv(tmp_path / "universe.csv", index=False)
        synthetic_historical_trades.to_csv(
            tmp_path / "historical_trades.csv",
            index=False,
        )
    else:
        synthetic_universe.to_parquet(tmp_path / "universe.parquet")
        synthetic_historical_trades.to_parquet(
            tmp_path / "historical_trades.parquet",
        )

    return provider_class(tmp_path)


def test_init_rejects_nonexistent_root() -> None:
    with pytest.raises(FileNotFoundError):
        CsvDataProvider("/nonexistent/path")


def test_init_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hello")
    with pytest.raises(NotADirectoryError):
        CsvDataProvider(file_path)


def test_get_weekly_ohlcv_returns_present_tickers(filesystem_provider_factory, friday) -> None:
    out = filesystem_provider_factory.get_weekly_ohlcv(
        ["ALPHA.NS", "BETA.NS", "ZETA.NS"],
        start=friday - pd.Timedelta(weeks=10),
        end=friday,
    )
    assert "ALPHA.NS" in out
    assert "BETA.NS" in out
    assert "ZETA.NS" not in out  # missing file → omitted


def test_get_weekly_ohlcv_caches_results(filesystem_provider_factory, friday) -> None:
    """Second call doesn't re-read disk — provider caches DataFrames."""
    p = filesystem_provider_factory
    p.get_weekly_ohlcv(["ALPHA.NS"], friday - pd.Timedelta(weeks=4), friday)
    # The internal cache attribute is named _weekly_cache.
    assert "ALPHA.NS" in p._weekly_cache


def test_get_nifty_weekly_returns_close(filesystem_provider_factory, friday) -> None:
    df = filesystem_provider_factory.get_nifty_weekly(
        friday - pd.Timedelta(weeks=10),
        friday,
    )
    assert "Close" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_get_universe_returns_canonical_columns(filesystem_provider_factory, friday) -> None:
    df = filesystem_provider_factory.get_universe(friday)
    for col in ("ticker", "sector", "market_cap", "adtv_20d"):
        assert col in df.columns


def test_get_historical_trades_when_empty(filesystem_provider_factory, friday) -> None:
    df = filesystem_provider_factory.get_historical_trades(before_date=friday)
    assert df.empty


def test_get_overall_regime_snapshot_missing_returns_none(
    filesystem_provider_factory,
    friday,
) -> None:
    """No overall_regime file shipped → returns None gracefully."""
    assert filesystem_provider_factory.get_overall_regime_snapshot(friday) is None


def test_csv_missing_date_column_raises(tmp_path: Path) -> None:
    """An OHLCV CSV without a 'date' column should raise ValueError on load."""
    weekly_dir = tmp_path / "weekly_ohlcv"
    weekly_dir.mkdir()
    pd.DataFrame({"Open": [1, 2], "Close": [1, 2]}).to_csv(
        weekly_dir / "BAD.NS.csv",
        index=False,
    )
    provider = CsvDataProvider(tmp_path)
    with pytest.raises(ValueError, match="date"):
        provider.get_weekly_ohlcv(
            ["BAD.NS"],
            start=pd.Timestamp("2023-01-01"),
            end=pd.Timestamp("2024-01-01"),
        )
