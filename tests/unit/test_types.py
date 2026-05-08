"""Tests for the public dataclasses and enums in :mod:`skysurf.types`."""

from __future__ import annotations

from datetime import date

from skysurf import (
    ActionType,
    EntrySignal,
    Position,
    PositionAction,
    TrailMode,
    TripleStackTighten,
    ValidationResult,
)


def test_action_type_values() -> None:
    """ActionType is the closed set of decisions the brain emits."""
    assert {member.value for member in ActionType} == {
        "HOLD",
        "UPDATE_STOP",
        "SWITCH_TRAIL_MA",
        "PARTIAL_SELL",
        "EXIT_FULL",
    }


def test_trail_mode_values() -> None:
    assert {member.value for member in TrailMode} == {"INITIAL", "PROGRESSIVE"}


def test_triple_stack_tighten_values() -> None:
    """STALL and CLIMACTIC are persisted; EXTENSION is transient and not represented."""
    assert {member.value for member in TripleStackTighten} == {
        "NONE",
        "STALL",
        "CLIMACTIC",
    }


def test_position_minimal_construction() -> None:
    """A Position can be constructed with the required fields and sensible defaults."""
    pos = Position(
        ticker="ALPHA.NS",
        sector="TECH",
        entry_date=date(2024, 6, 7),
        entry_price=100.0,
        qty=50,
        entry_type="PULLBACK_S2",
        tier="full",
        cost=5_059.37,
        stop_at_entry=92.0,
        stop_level=92.0,
        current_close=100.0,
        peak_close=100.0,
    )
    assert pos.ticker == "ALPHA.NS"
    assert pos.qty == 50
    # Defaults
    assert pos.trail_mode is TrailMode.INITIAL
    assert pos.triple_stack_state is TripleStackTighten.NONE
    assert pos.partial_taken is False
    assert pos.partial_sale_qty == 0


def test_entry_signal_construction() -> None:
    sig = EntrySignal(
        ticker="ALPHA.NS",
        sector="TECH",
        entry_type="PULLBACK_S2",
        entry_price=100.0,
        initial_stop=92.0,
        qty=50,
        tier="full",
        entry_cost=5_059.37,
        type_prior_score=2.92,
        rs_13w=0.65,
        target_level=120.0,
        rr_at_entry=2.5,
    )
    assert sig.diagnostic == {}  # default factory
    assert sig.entry_price == 100.0


def test_position_action_default_optionals() -> None:
    action = PositionAction(
        ticker="ALPHA.NS",
        action_type=ActionType.HOLD,
        reason="no change",
    )
    assert action.new_stop_level is None
    assert action.sell_qty is None
    assert action.proceeds is None


def test_validation_result_construction() -> None:
    result = ValidationResult(
        achieved_metrics={"mar": 1.96},
        expected_metrics={"mar": 1.96},
        deltas={"mar": 0.0},
        tolerances={"mar": 0.01},
        passed=True,
    )
    assert result.notes == []
    assert result.passed is True
