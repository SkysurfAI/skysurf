"""Integration smoke tests — full pipeline against synthetic data.

Exercises ``generate_weekly_signals`` and ``manage_positions`` end-to-end
with realistic-shaped synthetic OHLCV and asserts shape / invariants of
the output. These tests don't require external data; they run in CI.

The synthetic data is randomized but seeded deterministically, so the
exact signal counts may shift if random-number-generation changes
upstream — assertions therefore focus on shape and monotonic
properties rather than exact counts.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from skysurf import (
    InMemoryDataProvider,
    Position,
    generate_weekly_signals,
    manage_positions,
)


@pytest.fixture
def populated_provider() -> InMemoryDataProvider:
    """A larger synthetic universe biased toward Stage-2 trends.

    104 weeks of price action across 20 tickers, with an upward drift to
    increase the chance that some Stage-2 patterns fire.
    """
    rng = np.random.default_rng(seed=123)
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    n = len(weeks)

    weekly_ohlcv: dict[str, pd.DataFrame] = {}
    universe_rows = []
    sectors = ["TECH", "FINANCIAL", "ENERGY", "FMCG", "AUTO"]
    for i in range(20):
        ticker = f"STOCK{i:02d}.NS"
        sector = sectors[i % len(sectors)]
        start_price = 50.0 + i * 25
        # Add trend bias to encourage Stage-2 patterns.
        trend = 0.005
        returns = rng.normal(trend, 0.04, size=n)
        close = start_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(rng.normal(0, 0.02, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.02, n)))
        open_ = close * (1 + rng.normal(0, 0.01, n))
        volume = rng.integers(50_000, 500_000, n)
        weekly_ohlcv[ticker] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=weeks,
        )
        universe_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "market_cap": 1e10 * (i + 1),
                "adtv_20d": 1e7 * (i + 1),
            }
        )

    nifty_returns = rng.normal(0.003, 0.02, n)
    nifty_close = 17_000 * np.cumprod(1 + nifty_returns)
    nifty = pd.DataFrame({"Close": nifty_close}, index=weeks)

    return InMemoryDataProvider(
        weekly_ohlcv=weekly_ohlcv,
        daily_ohlcv=weekly_ohlcv,
        nifty_weekly=nifty,
        sector_indices_weekly={},
        universe=pd.DataFrame(universe_rows),
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
    )


def test_generate_signals_returns_list(populated_provider: InMemoryDataProvider) -> None:
    signals = generate_weekly_signals(
        provider=populated_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    assert isinstance(signals, list)
    # Even when no signals fire, the call must complete cleanly.


def test_generate_signals_respects_max_positions(populated_provider: InMemoryDataProvider) -> None:
    """If the universe has 20 tickers and max_positions=30, we never exceed 20."""
    signals = generate_weekly_signals(
        provider=populated_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=10_000_000.0,
    )
    assert len(signals) <= 20


def test_signal_fields_well_formed(populated_provider: InMemoryDataProvider) -> None:
    """Every emitted signal carries required fields with sensible values."""
    signals = generate_weekly_signals(
        provider=populated_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=10_000_000.0,
    )
    for s in signals:
        assert s.ticker
        assert s.sector
        assert s.entry_type in (
            "BREAKOUT_S1_TO_S2",
            "PULLBACK_S2",
            "VCP_CONTINUATION",
            "RETEST_SUPPORT",
            "TRENDLINE_BOUNCE",
        )
        assert s.entry_price > 0
        assert 0 < s.initial_stop < s.entry_price
        assert s.qty > 0
        assert s.tier in ("starter", "half", "full")
        assert s.entry_cost > 0


def test_manage_positions_emits_one_action_per_input(
    populated_provider: InMemoryDataProvider,
) -> None:
    """The output list mirrors the input list 1:1."""
    held = [
        Position(
            ticker="STOCK00.NS",
            sector="TECH",
            entry_date=date(2023, 12, 1),
            entry_price=100.0,
            qty=50,
            entry_type="PULLBACK_S2",
            tier="full",
            cost=5_000.0,
            stop_at_entry=92.0,
            stop_level=92.0,
            current_close=110.0,
            peak_close=115.0,
        ),
        Position(
            ticker="STOCK01.NS",
            sector="FINANCIAL",
            entry_date=date(2023, 12, 1),
            entry_price=200.0,
            qty=25,
            entry_type="BREAKOUT_S1_TO_S2",
            tier="full",
            cost=5_000.0,
            stop_at_entry=185.0,
            stop_level=190.0,
            current_close=220.0,
            peak_close=225.0,
        ),
    ]
    actions = manage_positions(
        provider=populated_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=held,
    )
    assert len(actions) == 2
    assert {a.ticker for a in actions} == {"STOCK00.NS", "STOCK01.NS"}


def test_no_signals_when_universe_empty() -> None:
    """Empty universe produces no signals."""
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    nifty = pd.DataFrame({"Close": np.full(104, 17_000.0)}, index=weeks)
    provider = InMemoryDataProvider(
        weekly_ohlcv={},
        daily_ohlcv={},
        nifty_weekly=nifty,
        sector_indices_weekly={},
        universe=pd.DataFrame(columns=["ticker", "sector", "market_cap", "adtv_20d"]),
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
    )
    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    assert signals == []


def test_manage_positions_empty_input_returns_empty(
    populated_provider: InMemoryDataProvider,
) -> None:
    actions = manage_positions(
        provider=populated_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
    )
    assert actions == []
