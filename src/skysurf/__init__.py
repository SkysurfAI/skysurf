"""Skysurf — production trading-signal library for Indian equities.

This is the open-source core of the Skysurf Phase 4 swing-trading
strategy. The public surface is intentionally small:

* :class:`StrategyConfig` and the canonical :data:`PHASE_4_V1` config.
* The :class:`DataProvider` abstract base + reference connectors.
* The :func:`generate_weekly_signals` and :func:`manage_positions`
  decision functions (added in subsequent releases of this package).
* Public types: :class:`Position`, :class:`EntrySignal`,
  :class:`PositionAction`, etc.
* The Zerodha cost model.
* Indicator helpers: ATR, RSI, moving averages.

See ``README.md`` for a 5-minute quickstart and ``docs/guide.md`` for
the end-to-end user guide.
"""

from __future__ import annotations

from skysurf.config import PHASE_4_V1, SUSPENDED_ENTRY_TYPES, StrategyConfig
from skysurf.costs import (
    ZERODHA_BUY_PCT,
    ZERODHA_DP_FLAT,
    ZERODHA_SELL_PCT,
    buy_cost,
    buy_cost_factor,
    sell_proceeds,
)
from skysurf.data import (
    CsvDataProvider,
    DataProvider,
    InMemoryDataProvider,
    OverallRegimeSnapshot,
    PandasDataProvider,
    ParquetDataProvider,
    SQLAlchemyDataProvider,
)
from skysurf.indicators import calculate_atr, calculate_rsi, compute_ma_series
from skysurf.positions import manage_positions
from skysurf.signals import generate_weekly_signals
from skysurf.types import (
    ActionType,
    EntrySignal,
    Position,
    PositionAction,
    TrailMode,
    TripleStackTighten,
    ValidationResult,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "PHASE_4_V1",
    "StrategyConfig",
    "SUSPENDED_ENTRY_TYPES",
    # Types
    "Position",
    "EntrySignal",
    "PositionAction",
    "ActionType",
    "TrailMode",
    "TripleStackTighten",
    "ValidationResult",
    # Data
    "DataProvider",
    "InMemoryDataProvider",
    "OverallRegimeSnapshot",
    "PandasDataProvider",
    "CsvDataProvider",
    "ParquetDataProvider",
    "SQLAlchemyDataProvider",
    # Costs
    "ZERODHA_BUY_PCT",
    "ZERODHA_SELL_PCT",
    "ZERODHA_DP_FLAT",
    "buy_cost",
    "buy_cost_factor",
    "sell_proceeds",
    # Indicators
    "calculate_atr",
    "calculate_rsi",
    "compute_ma_series",
    # API
    "generate_weekly_signals",
    "manage_positions",
]
