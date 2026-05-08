"""Shared pytest fixtures.

Provides synthetic OHLCV / regime / universe data so unit tests run on a
fresh clone with no external setup. The fixtures generate small,
deterministic datasets seeded from a fixed RNG so test outputs are
reproducible across runs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator (seed = 42)."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_weekly_ohlcv(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Synthetic weekly OHLCV for 5 fake tickers across 104 weeks (2 years).

    Prices are a geometric random walk; high/low are ±2% bands around
    close; volume is integer in [10_000, 100_000].
    """
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    out: dict[str, pd.DataFrame] = {}
    for ticker, start_price in [
        ("ALPHA.NS", 100.0),
        ("BETA.NS", 250.0),
        ("GAMMA.NS", 800.0),
        ("DELTA.NS", 45.0),
        ("EPSILON.NS", 1500.0),
    ]:
        returns = rng.normal(loc=0.001, scale=0.04, size=len(weeks))
        close = start_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(rng.normal(0, 0.02, size=len(weeks))))
        low = close * (1 - np.abs(rng.normal(0, 0.02, size=len(weeks))))
        open_ = close * (1 + rng.normal(0, 0.01, size=len(weeks)))
        volume = rng.integers(10_000, 100_000, size=len(weeks))
        out[ticker] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=weeks,
        )
    return out


@pytest.fixture
def synthetic_universe() -> pd.DataFrame:
    """Synthetic universe with the 5 tickers from ``synthetic_weekly_ohlcv``."""
    return pd.DataFrame(
        [
            {"ticker": "ALPHA.NS", "sector": "TECH", "market_cap": 5e10, "adtv_20d": 5e7},
            {"ticker": "BETA.NS", "sector": "FINANCIAL", "market_cap": 8e10, "adtv_20d": 1e8},
            {"ticker": "GAMMA.NS", "sector": "ENERGY", "market_cap": 1.5e11, "adtv_20d": 8e7},
            {"ticker": "DELTA.NS", "sector": "TECH", "market_cap": 2e10, "adtv_20d": 3e7},
            {"ticker": "EPSILON.NS", "sector": "FMCG", "market_cap": 1.2e11, "adtv_20d": 6e7},
        ]
    )


@pytest.fixture
def synthetic_nifty_weekly() -> pd.DataFrame:
    """Synthetic Nifty 50 weekly Close series."""
    rng = np.random.default_rng(seed=7)
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    returns = rng.normal(loc=0.001, scale=0.02, size=len(weeks))
    close = 17_000 * np.cumprod(1 + returns)
    return pd.DataFrame({"Close": close}, index=weeks)


@pytest.fixture
def synthetic_sector_indices(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Synthetic per-sector weekly Close series matching the universe sectors."""
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    out: dict[str, pd.DataFrame] = {}
    for sector, start in [
        ("TECH", 30_000),
        ("FINANCIAL", 20_000),
        ("ENERGY", 35_000),
        ("FMCG", 50_000),
    ]:
        returns = rng.normal(loc=0.001, scale=0.025, size=len(weeks))
        close = start * np.cumprod(1 + returns)
        out[sector] = pd.DataFrame({"Close": close}, index=weeks)
    return out


@pytest.fixture
def synthetic_historical_trades() -> pd.DataFrame:
    """Empty historical-trades DataFrame in the canonical schema."""
    return pd.DataFrame(columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"])


@pytest.fixture
def in_memory_provider(
    synthetic_weekly_ohlcv: dict[str, pd.DataFrame],
    synthetic_nifty_weekly: pd.DataFrame,
    synthetic_sector_indices: dict[str, pd.DataFrame],
    synthetic_universe: pd.DataFrame,
    synthetic_historical_trades: pd.DataFrame,
):
    """An InMemoryDataProvider populated with all the synthetic fixtures."""
    from skysurf.data import InMemoryDataProvider

    # Daily OHLCV: re-use the weekly frames (good enough for shape-level tests).
    return InMemoryDataProvider(
        weekly_ohlcv=synthetic_weekly_ohlcv,
        daily_ohlcv=synthetic_weekly_ohlcv,
        nifty_weekly=synthetic_nifty_weekly,
        sector_indices_weekly=synthetic_sector_indices,
        universe=synthetic_universe,
        historical_trades=synthetic_historical_trades,
    )


@pytest.fixture
def friday() -> pd.Timestamp:
    """A reference Friday inside the synthetic-data window."""
    return pd.Timestamp(date(2024, 6, 7))
