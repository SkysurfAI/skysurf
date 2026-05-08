"""Zerodha equity-delivery cost model.

Implements the per-share cost factors and DP charge that Zerodha bills
on Indian equity delivery trades, per the canonical Phase 4 cost model.
The constants are taken from `Zerodha's published charge sheet
<https://zerodha.com/charges>`_; small rounding has been folded into the
percentage so a single multiplication yields the exact net.

Cost composition

* Buy: STT 0.10% + NSE transaction 0.00307% + SEBI 0.0001% + stamp 0.015%
  + GST on (txn + SEBI) = **0.11874%** of trade value, plus a per-side
  market-impact slippage (default 0.10%).
* Sell: STT 0.10% + NSE transaction 0.00307% + SEBI 0.0001% +
  GST on (txn + SEBI) = **0.10374%** of trade value, plus a per-side
  slippage, plus a flat **₹15.93** DP charge per sell transaction per
  scrip (₹13.50 + 18% GST). DP applies to partial sells as well — every
  sell is a separate DP-charged transaction.

The functions accept either a :class:`~skysurf.brain.config.StrategyConfig`
instance (attribute access) or a plain mapping (dict-like), so they can
be used both inside the brain and from external test harnesses without
coercion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Buy-side regulatory percentage: STT + txn + SEBI + stamp + GST.
ZERODHA_BUY_PCT: float = 0.0011874

#: Sell-side regulatory percentage: STT + txn + SEBI + GST. No stamp on sells.
ZERODHA_SELL_PCT: float = 0.0010374

#: Flat DP (depository participant) charge per sell transaction per scrip,
#: in INR. Includes 18% GST.
ZERODHA_DP_FLAT: float = 15.93


def buy_cost_factor(cfg: Mapping[str, Any] | object) -> float:
    """Return the per-share buy multiplier.

    The result is ``effective_outflow_per_share / price``: multiply the
    intended ``qty * price`` by this number to get the actual cash
    outflow. The inverse multiplier is used to back out an original
    entry price from a cumulative cost (e.g., for break-even stops).

    Reads three fields from ``cfg`` (with sensible defaults):

    * ``cost_model`` — ``"ZERODHA"`` or ``"FLAT"`` (default ``"FLAT"``).
    * ``slippage_pct`` — per-side slippage (default ``0.001``).
    * ``brokerage_pct`` — per-side brokerage in the FLAT model only
      (default ``0.0011``). Ignored under ``"ZERODHA"`` (delivery is free).
    """
    slippage = _read(cfg, "slippage_pct", 0.001)
    if _read(cfg, "cost_model", "FLAT") == "ZERODHA":
        return 1 + ZERODHA_BUY_PCT + slippage
    return 1 + slippage + _read(cfg, "brokerage_pct", 0.0011)


def buy_cost(qty: float, price: float, cfg: Mapping[str, Any] | object) -> float:
    """Return total cash outflow for a buy transaction."""
    return qty * price * buy_cost_factor(cfg)


def sell_proceeds(qty: float, price: float, cfg: Mapping[str, Any] | object) -> float:
    """Return cash inflow for a sell transaction.

    Under the ZERODHA model: subtracts ``ZERODHA_SELL_PCT`` plus slippage
    from the gross proceeds, then deducts a flat ``ZERODHA_DP_FLAT`` per
    transaction. Under the FLAT model: subtracts only slippage and
    brokerage.
    """
    slippage = _read(cfg, "slippage_pct", 0.001)
    if _read(cfg, "cost_model", "FLAT") == "ZERODHA":
        return qty * price * (1 - ZERODHA_SELL_PCT - slippage) - ZERODHA_DP_FLAT
    return qty * price * (1 - slippage - _read(cfg, "brokerage_pct", 0.0011))


def _read(cfg: Mapping[str, Any] | object, key: str, default: Any) -> Any:
    """Read ``key`` from either a mapping (``cfg.get``) or a dataclass (``getattr``).

    This lets the cost helpers accept both ``StrategyConfig`` instances
    and plain dicts without coercion.
    """
    if hasattr(cfg, "get") and not hasattr(cfg, "__dataclass_fields__"):
        return cfg.get(key, default)  # type: ignore[union-attr]
    return getattr(cfg, key, default)
