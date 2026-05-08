"""Smoke tests for the v0.1.0 public API surface.

In v0.1.0 the strategy engine is not yet vendored, so the two top-level
decision functions raise :class:`NotImplementedError` with a helpful
message. The function signatures, types, and configuration surface are
stable, so callers can integrate against them today.

These tests verify that the placeholder behaviour is in place and that
nothing else in the public surface accidentally raises.
"""

from __future__ import annotations

from datetime import date

import pytest

from skysurf import (
    PHASE_4_V1,
    EntrySignal,
    Position,
    PositionAction,
    StrategyConfig,
    generate_weekly_signals,
    manage_positions,
)


def test_top_level_imports_resolve() -> None:
    """Every name re-exported at package level binds to something."""
    import skysurf

    for name in skysurf.__all__:
        assert hasattr(skysurf, name), f"{name} listed in __all__ but not bound"


def test_phase_4_v1_is_strategy_config_instance() -> None:
    assert isinstance(PHASE_4_V1, StrategyConfig)


def test_generate_weekly_signals_raises_not_implemented_in_v0_1(in_memory_provider) -> None:
    """The signal generator is wired but the engine is not vendored yet."""
    with pytest.raises(NotImplementedError, match="ROADMAP"):
        generate_weekly_signals(
            provider=in_memory_provider,
            as_of_date=date(2024, 6, 7),
            current_positions=[],
            total_equity=500_000.0,
        )


def test_manage_positions_raises_not_implemented_in_v0_1(in_memory_provider) -> None:
    """The position manager is wired but the engine is not vendored yet."""
    with pytest.raises(NotImplementedError, match="ROADMAP"):
        manage_positions(
            provider=in_memory_provider,
            as_of_date=date(2024, 6, 7),
            current_positions=[],
        )


def test_position_and_signal_types_are_dataclasses() -> None:
    """The public types are real dataclasses callers can construct directly."""
    pos = Position(
        ticker="ALPHA.NS",
        sector="TECH",
        entry_date=date(2024, 6, 7),
        entry_price=100.0,
        qty=10,
        entry_type="PULLBACK_S2",
        tier="full",
        cost=1_011.87,
        stop_at_entry=92.0,
        stop_level=92.0,
        current_close=100.0,
        peak_close=100.0,
    )
    assert pos.qty == 10

    sig = EntrySignal(
        ticker="ALPHA.NS",
        sector="TECH",
        entry_type="PULLBACK_S2",
        entry_price=100.0,
        initial_stop=92.0,
        qty=10,
        tier="full",
        entry_cost=1_011.87,
        type_prior_score=2.92,
        rs_13w=0.65,
        target_level=120.0,
        rr_at_entry=2.5,
    )
    assert sig.entry_price == 100.0


def test_position_action_construction_minimal() -> None:
    from skysurf import ActionType

    action = PositionAction(
        ticker="ALPHA.NS",
        action_type=ActionType.HOLD,
        reason="no change",
    )
    assert action.action_type is ActionType.HOLD
