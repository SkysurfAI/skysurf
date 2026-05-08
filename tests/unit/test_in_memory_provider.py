"""Tests for InMemoryDataProvider — the reference DataProvider.

The other connectors (CSV, Parquet, SQLAlchemy) reuse most of this
behaviour by sharing a layer; their tests live in their own modules and
focus on the format-specific I/O.
"""

from __future__ import annotations

import pandas as pd
import pytest

from skysurf import InMemoryDataProvider


def test_get_weekly_ohlcv_returns_only_present_tickers(in_memory_provider, friday) -> None:
    out = in_memory_provider.get_weekly_ohlcv(
        ["ALPHA.NS", "ZETA.NS"],  # ZETA.NS is intentionally not in the fixture
        start=friday - pd.Timedelta(weeks=10),
        end=friday,
    )
    assert "ALPHA.NS" in out
    assert "ZETA.NS" not in out


def test_get_weekly_ohlcv_slices_by_date_range(in_memory_provider, friday) -> None:
    out = in_memory_provider.get_weekly_ohlcv(
        ["ALPHA.NS"],
        start=friday - pd.Timedelta(weeks=4),
        end=friday,
    )
    df = out["ALPHA.NS"]
    assert len(df) <= 5  # ≤ 5 weekly bars in a 4-week window
    assert df.index.min() >= friday - pd.Timedelta(weeks=4)
    assert df.index.max() <= friday


def test_get_universe_returns_all_tickers_when_no_quarterly_filter(
    in_memory_provider, friday
) -> None:
    universe = in_memory_provider.get_universe(friday)
    assert len(universe) == 5
    assert set(universe["ticker"]) == {
        "ALPHA.NS",
        "BETA.NS",
        "GAMMA.NS",
        "DELTA.NS",
        "EPSILON.NS",
    }


def test_quarterly_universe_filter(synthetic_universe, friday) -> None:
    """When quarterly_universe is supplied, get_universe filters by the
    ticker-set qualifying in the given quarter."""
    qmap = {
        "2024Q2": {"ALPHA.NS", "BETA.NS"},  # only two qualify in the friday quarter
    }
    provider = InMemoryDataProvider(
        weekly_ohlcv={},
        daily_ohlcv={},
        nifty_weekly=pd.DataFrame(),
        sector_indices_weekly={},
        universe=synthetic_universe,
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
        quarterly_universe=qmap,
    )
    filtered = provider.get_universe(friday)
    assert set(filtered["ticker"]) == {"ALPHA.NS", "BETA.NS"}


def test_quarterly_universe_returns_full_universe_for_unknown_quarter(
    synthetic_universe, friday
) -> None:
    """If the quarterly map has no entry for as_of's quarter, fall back
    to the full universe."""
    provider = InMemoryDataProvider(
        weekly_ohlcv={},
        daily_ohlcv={},
        nifty_weekly=pd.DataFrame(),
        sector_indices_weekly={},
        universe=synthetic_universe,
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
        quarterly_universe={"1999Q1": {"DOES_NOT_MATTER"}},
    )
    filtered = provider.get_universe(friday)
    assert len(filtered) == 5


def test_get_historical_trades_filters_strictly_before_cutoff(synthetic_universe) -> None:
    cutoff = pd.Timestamp("2024-06-07")
    trades = pd.DataFrame(
        [
            {
                "ticker": "A",
                "week_date": pd.Timestamp("2024-05-31"),
                "entry_type": "PULLBACK_S2",
                "mfe_pct": 12.0,
                "mae_pct": -3.0,
            },
            {
                "ticker": "B",
                "week_date": pd.Timestamp("2024-06-07"),  # exactly on cutoff
                "entry_type": "VCP_CONTINUATION",
                "mfe_pct": 8.0,
                "mae_pct": -5.0,
            },
            {
                "ticker": "C",
                "week_date": pd.Timestamp("2024-06-14"),
                "entry_type": "RETEST_SUPPORT",
                "mfe_pct": 5.0,
                "mae_pct": -8.0,
            },
        ]
    )
    provider = InMemoryDataProvider(
        weekly_ohlcv={},
        daily_ohlcv={},
        nifty_weekly=pd.DataFrame(),
        sector_indices_weekly={},
        universe=synthetic_universe,
        historical_trades=trades,
    )
    filtered = provider.get_historical_trades(before_date=cutoff)
    assert set(filtered["ticker"]) == {"A"}  # B and C excluded by strict <


def test_get_overall_regime_snapshot_returns_none_when_not_configured(
    in_memory_provider,
    friday,
) -> None:
    """The fixture provider has no overall_regime_df, so this returns None."""
    assert in_memory_provider.get_overall_regime_snapshot(friday) is None


def test_get_overall_regime_snapshot_when_configured(synthetic_universe, friday) -> None:
    regime_df = pd.DataFrame(
        {"regime": ["weakening_bull"], "breadth_pct": [62.5]},
        index=[friday],
    )
    provider = InMemoryDataProvider(
        weekly_ohlcv={},
        daily_ohlcv={},
        nifty_weekly=pd.DataFrame(),
        sector_indices_weekly={},
        universe=synthetic_universe,
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
        overall_regime_df=regime_df,
    )
    snap = provider.get_overall_regime_snapshot(friday)
    assert snap == {"regime": "weakening_bull", "breadth_pct": 62.5}


def test_is_ticker_in_universe_default_true(in_memory_provider, friday) -> None:
    assert in_memory_provider.is_ticker_in_universe_for("ANY.NS", friday) is True


@pytest.mark.parametrize(
    ("date", "expected_quarter"),
    [
        (pd.Timestamp("2024-01-15"), "2024Q1"),
        (pd.Timestamp("2024-04-15"), "2024Q2"),
        (pd.Timestamp("2024-07-15"), "2024Q3"),
        (pd.Timestamp("2024-10-15"), "2024Q4"),
        (pd.Timestamp("2024-12-31"), "2024Q4"),
        (pd.Timestamp("2025-01-01"), "2025Q1"),
    ],
)
def test_quarter_str_helper(date: pd.Timestamp, expected_quarter: str) -> None:
    """Internal helper — quarter-string formatting."""
    from skysurf.data.provider import _quarter_str

    assert _quarter_str(date) == expected_quarter
