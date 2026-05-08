"""Public ↔ internal Position translation.

The public :class:`skysurf.types.Position` exposes a clean,
caller-facing dataclass with enum-based state fields. The engine
internally uses :class:`skysurf._internal.engine._Position` — a wider
record carrying per-position exit-config carry-over (so config swaps
between calls don't retroactively retune the trail anchor on existing
positions).

This module bridges the two. Translation is purely structural — every
field maps 1:1 or is filled from the active
:class:`~skysurf.config.StrategyConfig`. No behaviour change, only
renaming and enum ↔ string conversion.

State enum mapping
-------------------

============================  ===============================
internal ``tighten_state``    public ``triple_stack_state``
============================  ===============================
``None``                      ``TripleStackTighten.NONE``
``"stall"``                   ``TripleStackTighten.STALL``
``"climactic"``               ``TripleStackTighten.CLIMACTIC``
============================  ===============================

============================  ===============================
internal ``trail_tightened``  public ``trail_mode``
============================  ===============================
``False``                     ``TrailMode.INITIAL``
``True``                      ``TrailMode.PROGRESSIVE``
============================  ===============================

The public :class:`TripleStackTighten` does not carry an
``EXTENSION`` member — research's extension tightening is applied
transiently to the local ``use_ma_col`` inside ``decide_exits_for_week``
and is never persisted onto position state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from skysurf._internal.engine import _Position
from skysurf.config import StrategyConfig
from skysurf.types import Position, TrailMode, TripleStackTighten

_BRAIN_TIGHTEN_TO_INTERNAL: dict[TripleStackTighten, str | None] = {
    TripleStackTighten.NONE: None,
    TripleStackTighten.STALL: "stall",
    TripleStackTighten.CLIMACTIC: "climactic",
}

_INTERNAL_TIGHTEN_TO_BRAIN: dict[str | None, TripleStackTighten] = {
    None: TripleStackTighten.NONE,
    "stall": TripleStackTighten.STALL,
    "climactic": TripleStackTighten.CLIMACTIC,
}


def _trail_mode_to_internal(mode: TrailMode) -> bool:
    """``PROGRESSIVE`` → ``True`` (trail tightened); ``INITIAL`` → ``False``."""
    return mode == TrailMode.PROGRESSIVE


def _internal_to_trail_mode(trail_tightened: bool) -> TrailMode:
    """``True`` → ``PROGRESSIVE``; ``False`` → ``INITIAL``."""
    return TrailMode.PROGRESSIVE if trail_tightened else TrailMode.INITIAL


def _exit_ma_col(config: StrategyConfig) -> str:
    """``"sma_27"`` for PHASE_4_V1 (exit_ma_type SMA, exit_ma_period 27)."""
    return f"{config.exit_ma_type.lower()}_{config.exit_ma_period}"


def _per_position_overrides(config: StrategyConfig) -> dict[str, Any]:
    """Read the per-position exit-config block out of a StrategyConfig."""
    return {
        "exit_ma_col": _exit_ma_col(config),
        "exit_atr_buffer": float(config.exit_atr_buffer),
        "pos_triple_stack_enabled": bool(config.triple_stack_enabled),
        "pos_stall_tighten_week": int(config.stall_tighten_week),
        "pos_stall_tighten_threshold": float(config.stall_tighten_threshold),
        "pos_stall_tighten_ma": str(config.stall_tighten_ma),
        "pos_extension_atr_mult": float(config.extension_atr_mult),
        "pos_extension_tighten_ma": str(config.extension_tighten_ma),
        "pos_climactic_vol_threshold": float(config.climactic_vol_threshold),
        "pos_climactic_tighten_ma": str(config.climactic_tighten_ma),
        "config_id": "skysurf-public",
    }


def public_to_internal(pos: Position, config: StrategyConfig) -> _Position:
    """Translate a public :class:`Position` into a fresh internal record.

    Per-position exit-config fields come from ``config`` so the engine
    uses the active config's tightening parameters even after a config
    swap.
    """
    overrides = _per_position_overrides(config)
    return _Position(
        ticker=pos.ticker,
        sector=pos.sector,
        # Engine arithmetic uses pd.Timestamp; coerce here so callers
        # can pass either ``datetime.date`` or ``pd.Timestamp``.
        entry_date=pd.Timestamp(pos.entry_date),
        entry_price=float(pos.entry_price),
        qty=int(pos.qty),
        cost=float(pos.cost),
        stop_level=float(pos.stop_level),
        tier=pos.tier,
        peak_close=float(pos.peak_close),
        entry_type=pos.entry_type,
        target_type=pos.target_type,
        risk_pct_at_entry=float(pos.risk_pct_at_entry),
        rs_13w_at_entry=float(pos.rs_13w_at_entry),
        regime_at_entry=pos.regime_at_entry,
        rr_at_entry=float(pos.rr_at_entry),
        current_close=float(pos.current_close),
        stop_at_entry=float(pos.stop_at_entry),
        min_low=float(pos.min_low),
        mfe_week=pd.Timestamp(pos.mfe_week) if pos.mfe_week is not None else None,
        tighten_state=_BRAIN_TIGHTEN_TO_INTERNAL[pos.triple_stack_state],
        prev_close=float(pos.prev_close),
        partial_taken=bool(pos.partial_taken),
        partial_sale_proceeds=float(pos.partial_sale_proceeds),
        partial_sale_qty=int(pos.partial_sale_qty),
        partial_sale_price=float(pos.partial_sale_price),
        partial_sale_date=(
            pd.Timestamp(pos.partial_sale_date) if pos.partial_sale_date is not None else None
        ),
        trail_tightened=_trail_mode_to_internal(pos.trail_mode),
        time_stop_triggered=False,
        **overrides,
    )


def internal_to_public(rpos: _Position) -> Position:
    """Translate an engine-internal position back to the public dataclass.

    Internal-only fields (``exit_ma_col``, ``pos_*``, ``config_id``) are
    intentionally dropped — they're carry-over state, not part of the
    public contract.
    """
    if rpos.tighten_state not in _INTERNAL_TIGHTEN_TO_BRAIN:
        raise ValueError(
            f"internal_to_public: unknown tighten_state {rpos.tighten_state!r}; "
            f"update _INTERNAL_TIGHTEN_TO_BRAIN."
        )
    return Position(
        ticker=rpos.ticker,
        sector=rpos.sector,
        entry_date=rpos.entry_date,
        entry_price=float(rpos.entry_price),
        qty=int(rpos.qty),
        entry_type=rpos.entry_type,
        tier=rpos.tier,
        cost=float(rpos.cost),
        stop_at_entry=float(rpos.stop_at_entry),
        stop_level=float(rpos.stop_level),
        current_close=float(rpos.current_close),
        peak_close=float(rpos.peak_close),
        min_low=float(rpos.min_low),
        prev_close=float(rpos.prev_close),
        trail_mode=_internal_to_trail_mode(bool(rpos.trail_tightened)),
        triple_stack_state=_INTERNAL_TIGHTEN_TO_BRAIN[rpos.tighten_state],
        partial_taken=bool(rpos.partial_taken),
        partial_sale_qty=int(rpos.partial_sale_qty),
        partial_sale_price=float(rpos.partial_sale_price),
        partial_sale_date=rpos.partial_sale_date,
        partial_sale_proceeds=float(rpos.partial_sale_proceeds),
        risk_pct_at_entry=float(rpos.risk_pct_at_entry),
        rs_13w_at_entry=float(rpos.rs_13w_at_entry),
        regime_at_entry=rpos.regime_at_entry,
        target_type=rpos.target_type,
        rr_at_entry=float(rpos.rr_at_entry),
        mfe_week=rpos.mfe_week,
    )


__all__ = ["public_to_internal", "internal_to_public"]
