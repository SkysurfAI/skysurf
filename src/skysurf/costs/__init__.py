"""Cost models for trade execution.

Currently supports the Zerodha equity-delivery cost model used by the
canonical Phase 4 strategy. Additional cost models (e.g., other Indian
brokers) can be added as new modules; the public :class:`StrategyConfig`
selects between them via the ``cost_model`` field.
"""
from __future__ import annotations

from skysurf.costs.zerodha import (
    ZERODHA_BUY_PCT,
    ZERODHA_DP_FLAT,
    ZERODHA_SELL_PCT,
    buy_cost,
    buy_cost_factor,
    sell_proceeds,
)

__all__ = [
    "ZERODHA_BUY_PCT",
    "ZERODHA_DP_FLAT",
    "ZERODHA_SELL_PCT",
    "buy_cost",
    "buy_cost_factor",
    "sell_proceeds",
]
