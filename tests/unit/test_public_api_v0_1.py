"""Smoke tests for the public API surface.

These tests verify that the public surface is importable, types are
constructible, and the two top-level decision functions run end-to-end
on synthetic data without raising.

The full behavioural test suite for ``generate_weekly_signals`` and
``manage_positions`` lives in ``tests/integration/``; this module just
checks the API contract.
"""

from __future__ import annotations

from datetime import date

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


def test_generate_weekly_signals_runs_end_to_end_on_synthetic_data(
    in_memory_provider,
) -> None:
    """Synthetic synthesizer-generated data runs through the full pipeline.

    The synthetic universe / OHLCV is too short and too random to fire
    PHASE_4_V1 entry detectors most of the time, but the function must
    still complete cleanly and return a list (possibly empty).
    """
    signals = generate_weekly_signals(
        provider=in_memory_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    assert isinstance(signals, list)
    for s in signals:
        assert isinstance(s, EntrySignal)


def test_manage_positions_returns_empty_list_when_no_positions(in_memory_provider) -> None:
    """No held positions → empty action list."""
    actions = manage_positions(
        provider=in_memory_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
    )
    assert actions == []


def test_manage_positions_returns_one_action_per_position(in_memory_provider) -> None:
    """The returned list mirrors the input positions 1:1."""
    held = [
        Position(
            ticker="ALPHA.NS",
            sector="TECH",
            entry_date=date(2023, 12, 1),
            entry_price=100.0,
            qty=50,
            entry_type="PULLBACK_S2",
            tier="full",
            cost=5_059.37,
            stop_at_entry=92.0,
            stop_level=92.0,
            current_close=110.0,
            peak_close=115.0,
        )
    ]
    actions = manage_positions(
        provider=in_memory_provider,
        as_of_date=date(2024, 6, 7),
        current_positions=held,
    )
    assert len(actions) == 1
    assert actions[0].ticker == "ALPHA.NS"


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
