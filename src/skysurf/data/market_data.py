"""Internal market-data bundle.

Internal plumbing between the data layer and the strategy pipeline. Not
a public API — clients should not construct or inspect this; it lives
solely so the brain can pass a single object into the entry-detection
and exit-evaluation paths.

The brain assembles a :class:`MarketData` inside
:func:`skysurf.signals.generate_weekly_signals` (and the
positions counterpart), populates it from
:class:`~skysurf.data.provider.DataProvider` calls, and discards it when
the call returns. The brain holds no state across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class MarketData:
    """Snapshot of market data needed for one decision-pipeline run.

    See :class:`~skysurf.data.provider.DataProvider` for the canonical
    schema of each field.
    """

    weekly_ohlcv: dict[str, pd.DataFrame]
    """Ticker → DataFrame with columns ``[Open, High, Low, Close, Volume]``."""

    daily_ohlcv: dict[str, pd.DataFrame]
    """Ticker → DataFrame with columns ``[Open, High, Low, Close, Volume]``."""

    nifty_weekly: pd.DataFrame
    """DataFrame with at least a ``Close`` column, indexed by weekly date."""

    sector_indices_weekly: dict[str, pd.DataFrame]
    """Sector name → DataFrame with at least a ``Close`` column."""

    universe: pd.DataFrame
    """DataFrame with columns ``[ticker, sector, market_cap, adtv_20d]``."""

    historical_trades: pd.DataFrame
    """DataFrame with columns ``[ticker, week_date, entry_type, mfe_pct, mae_pct]``."""

    as_of_date: date | pd.Timestamp
    """The Friday being evaluated."""
