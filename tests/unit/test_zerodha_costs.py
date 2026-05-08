"""Tests for the Zerodha equity-delivery cost model.

The constants are taken from Zerodha's published charge sheet for
equity delivery. Reconciliation against the canonical strategy is
tested separately in the engine integration tests.
"""

from __future__ import annotations

import math

import pytest

from skysurf import (
    PHASE_4_V1,
    ZERODHA_BUY_PCT,
    ZERODHA_DP_FLAT,
    ZERODHA_SELL_PCT,
    buy_cost,
    buy_cost_factor,
    sell_proceeds,
)


def test_zerodha_constants_match_canonical_doc() -> None:
    """Cost constants must match the canonical config exactly."""
    assert ZERODHA_BUY_PCT == 0.0011874
    assert ZERODHA_SELL_PCT == 0.0010374
    assert ZERODHA_DP_FLAT == 15.93


# ── PHASE_4_V1 (a StrategyConfig instance) — attribute-access path ────


def test_buy_cost_factor_zerodha_via_strategy_config() -> None:
    factor = buy_cost_factor(PHASE_4_V1)
    expected = 1 + ZERODHA_BUY_PCT + PHASE_4_V1.slippage_pct
    assert math.isclose(factor, expected)


def test_buy_cost_zerodha_via_strategy_config() -> None:
    cost = buy_cost(qty=100, price=500.0, cfg=PHASE_4_V1)
    expected = 100 * 500.0 * (1 + ZERODHA_BUY_PCT + PHASE_4_V1.slippage_pct)
    assert math.isclose(cost, expected)


def test_sell_proceeds_zerodha_via_strategy_config() -> None:
    proceeds = sell_proceeds(qty=100, price=600.0, cfg=PHASE_4_V1)
    expected = 100 * 600.0 * (1 - ZERODHA_SELL_PCT - PHASE_4_V1.slippage_pct) - ZERODHA_DP_FLAT
    assert math.isclose(proceeds, expected)


# ── Plain-dict path (used by the research engine) ────────────────────


def test_buy_cost_zerodha_via_dict() -> None:
    cfg = {"cost_model": "ZERODHA", "slippage_pct": 0.001}
    cost = buy_cost(qty=10, price=1000.0, cfg=cfg)
    expected = 10 * 1000.0 * (1 + ZERODHA_BUY_PCT + 0.001)
    assert math.isclose(cost, expected)


def test_sell_proceeds_zerodha_via_dict_includes_dp_charge() -> None:
    cfg = {"cost_model": "ZERODHA", "slippage_pct": 0.001}
    proceeds = sell_proceeds(qty=10, price=1000.0, cfg=cfg)
    expected = 10 * 1000.0 * (1 - ZERODHA_SELL_PCT - 0.001) - ZERODHA_DP_FLAT
    assert math.isclose(proceeds, expected)


# ── FLAT-model fallback ───────────────────────────────────────────────


def test_buy_cost_flat_model_no_zerodha_pct() -> None:
    cfg = {"cost_model": "FLAT", "slippage_pct": 0.001, "brokerage_pct": 0.0011}
    cost = buy_cost(qty=10, price=1000.0, cfg=cfg)
    expected = 10 * 1000.0 * (1 + 0.001 + 0.0011)
    assert math.isclose(cost, expected)


def test_sell_proceeds_flat_model_no_dp_charge() -> None:
    """The FLAT model has NO flat DP charge — only percentage costs."""
    cfg = {"cost_model": "FLAT", "slippage_pct": 0.001, "brokerage_pct": 0.0011}
    proceeds = sell_proceeds(qty=10, price=1000.0, cfg=cfg)
    expected = 10 * 1000.0 * (1 - 0.001 - 0.0011)
    assert math.isclose(proceeds, expected)


def test_buy_cost_unknown_cost_model_falls_back_to_flat() -> None:
    """Any cost_model other than ZERODHA uses the FLAT formula."""
    cfg = {"cost_model": "MYSTERY"}
    cost = buy_cost(qty=10, price=1000.0, cfg=cfg)
    # Default slippage 0.001 + default brokerage 0.0011
    expected = 10 * 1000.0 * (1 + 0.001 + 0.0011)
    assert math.isclose(cost, expected)


@pytest.mark.parametrize("qty", [1, 10, 1000, 50_000])
@pytest.mark.parametrize("price", [10.0, 100.0, 1_000.0, 10_000.0])
def test_round_trip_zerodha_loses_money(qty: int, price: float) -> None:
    """Buying then immediately selling at the same price must cost money.

    The buy adds buy_pct + slippage; the sell subtracts sell_pct +
    slippage and a flat DP. So a round-trip is always a net loss under
    Zerodha — this is a sanity invariant.
    """
    cfg = PHASE_4_V1
    out = buy_cost(qty, price, cfg)
    in_ = sell_proceeds(qty, price, cfg)
    assert in_ < out, f"qty={qty} price={price} buy={out} sell={in_}"
