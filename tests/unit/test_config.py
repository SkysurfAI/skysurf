"""Tests for :class:`StrategyConfig` and the canonical PHASE_4_V1 lock."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from skysurf import PHASE_4_V1, SUSPENDED_ENTRY_TYPES, StrategyConfig


def test_phase_4_v1_is_a_strategy_config() -> None:
    assert isinstance(PHASE_4_V1, StrategyConfig)


def test_phase_4_v1_is_frozen() -> None:
    """Mutating PHASE_4_V1 must raise — protects against accidental state."""
    with pytest.raises(FrozenInstanceError):
        PHASE_4_V1.risk_pct = 0.02  # type: ignore[misc]


def test_phase_4_v1_replace_returns_new_instance() -> None:
    """`dataclasses.replace` is the canonical way to derive new configs."""
    derived = replace(PHASE_4_V1, risk_pct=0.005)
    assert derived is not PHASE_4_V1
    assert derived.risk_pct == 0.005
    assert PHASE_4_V1.risk_pct == 0.01  # original unchanged


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        # Universe filters — absolute INR
        ("market_cap_floor_inr", 1_500 * 1e7),
        ("adtv_floor_inr", 2 * 1e7),
        # Regime gate
        ("regime_combination", "OVERALL_OR_SECTOR"),
        ("overall_breadth_threshold", 60),
        ("sector_rs_filter", "TOP"),
        # Stock-level filters
        ("rs_gate", 0.5),
        ("rsi_gate", None),
        ("rr_floor", 0),
        # Entry detection
        ("swing_order", 8),
        ("swing_order_minor", 5),
        ("swing_lag_weeks", 8),
        ("retest_proximity_atr", 1.5),
        ("retest_max_weeks", 12),
        ("entry_price_method", "week_close"),
        # Per-detector
        ("breakout_base_min_weeks", 4),
        ("pullback_min_stage2", 4),
        ("vcp_consol_min_weeks", 3),
        # Ranking
        ("ranking_method", "type_prior"),
        ("type_prior_mode", "dynamic"),
        ("type_prior_min_type_n", 20),
        ("type_prior_min_total_n", 100),
        ("type_prior_default", 1.0),
        # Sizing
        ("risk_pct", 0.01),
        ("sizing_on", "total_equity"),
        # Portfolio
        ("sector_limit", 5),
        ("max_positions", 30),
        ("pyramid_enabled", False),
        ("time_stop_enabled", False),
        # Initial trail
        ("exit_ma_period", 27),
        ("exit_atr_buffer", 0.75),
        # Triple-stack
        ("triple_stack_enabled", True),
        ("stall_tighten_week", 12),
        ("stall_tighten_threshold", 0.07),
        ("extension_atr_mult", 3.5),
        ("climactic_vol_threshold", 2.0),
        # Partial profit
        ("partial_profit_trigger_pct", 30),
        ("partial_profit_sell_pct", 50),
        ("partial_profit_move_stop", "HALF_BACK"),
        # Progressive trail
        ("trail_tighten_trigger_pct", 50),
        ("trail_tighten_ma_period", 20),
        # Costs
        ("cost_model", "ZERODHA"),
        ("zerodha_buy_cost_pct", 0.0011874),
        ("zerodha_sell_cost_pct", 0.0010374),
        ("zerodha_dp_charge_inr", 15.93),
        ("slippage_pct", 0.001),
        # Walk-forward
        ("starting_capital", 500_000),
        ("risk_free_annual", 0.06),
    ],
)
def test_phase_4_v1_canonical_value(attr: str, expected: object) -> None:
    """Every value in PHASE_4_V1 must match the canonical lock exactly.

    These assertions guard against accidental drift away from the values
    that produced MAR 1.96 / CAGR 24.13% / MaxDD −12.31% / 604 trades on
    the validation walk-forward.
    """
    assert getattr(PHASE_4_V1, attr) == expected, (
        f"PHASE_4_V1.{attr} drifted from canonical: "
        f"got {getattr(PHASE_4_V1, attr)!r}, expected {expected!r}"
    )


def test_phase_4_v1_entry_mas_per_type_canonical() -> None:
    expected = {
        "PULLBACK_S2": ("EMA", 20),
        "BREAKOUT_S1_TO_S2": ("SMA", 40),
        "VCP_CONTINUATION": ("EMA", 40),
        "RETEST_SUPPORT": ("SMA", 40),
        "TRENDLINE_BOUNCE": ("EMA", 40),
    }
    assert PHASE_4_V1.entry_mas_per_type == expected


def test_phase_4_v1_base_min_weeks_canonical() -> None:
    expected = {
        "PULLBACK_S2": 12,
        "BREAKOUT_S1_TO_S2": 0,
        "VCP_CONTINUATION": 4,
        "RETEST_SUPPORT": 8,
        "TRENDLINE_BOUNCE": 8,
    }
    assert PHASE_4_V1.base_min_weeks_per_type == expected


def test_phase_4_v1_tier_tuples() -> None:
    assert PHASE_4_V1.tier_rr_thresholds == (1.5, 2.0)
    assert PHASE_4_V1.tier_multipliers == (0.5, 0.75, 1.0)
    assert PHASE_4_V1.concentration_caps == (0.08, 0.15, 0.25)


def test_suspended_entry_types_constant() -> None:
    """The suspended-entry-types tuple is referenced both standalone and inside the config."""
    assert SUSPENDED_ENTRY_TYPES == ("ATH_BREAKOUT", "PULLBACK_STRUCTURAL")
    assert PHASE_4_V1.suspended_entry_types == SUSPENDED_ENTRY_TYPES
