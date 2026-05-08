"""Public dataclasses at the brain's API boundary.

These types are the contract between the brain and its callers. Internal
brain modules use them too, but anything beyond these dataclasses is
implementation detail.

Design notes:
    * :class:`Position` is a dataclass (not frozen) because state mutates
      as the position-management loop ratchets stops, applies partials,
      etc. Callers should treat instances they receive as outputs and
      not mutate them; new state is expressed by emitting fresh objects,
      not by editing the inputs in place. Inputs to the brain are
      likewise treated as immutable from the brain's perspective.
    * :class:`EntrySignal` and :class:`PositionAction` are conceptually
      immutable result types.
    * :class:`ActionType` is the closed set of decisions the brain emits
      per held position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from enum import Enum
from typing import Any

import pandas as pd

# Type alias: most callers pass a `datetime.date`; the engine internally
# converts to `pd.Timestamp`. Both are accepted.
DateLike = _date | pd.Timestamp


class ActionType(str, Enum):
    """Closed set of decisions the brain may emit for a held position."""

    HOLD = "HOLD"
    UPDATE_STOP = "UPDATE_STOP"
    SWITCH_TRAIL_MA = "SWITCH_TRAIL_MA"
    PARTIAL_SELL = "PARTIAL_SELL"
    EXIT_FULL = "EXIT_FULL"


class TrailMode(str, Enum):
    """Which trailing-stop mechanism is currently active on a position.

    * ``INITIAL`` — SMA-27 minus 0.75×ATR. Default at entry.
    * ``PROGRESSIVE`` — SMA-20 minus 0.75×ATR. Activated once unrealized
      gain crosses 50%.
    """

    INITIAL = "INITIAL"
    PROGRESSIVE = "PROGRESSIVE"


class TripleStackTighten(str, Enum):
    """Which triple-stack tightening (if any) is currently active.

    The triple-stack adds three independent triggers that switch the
    trailing-MA baseline to a faster MA:

    * ``STALL`` — week ≥ 12 and current return < 7%, switches to SMA-20.
    * ``CLIMACTIC`` — volume ratio > 2.0 and close < previous close,
      switches to SMA-15.

    The third trigger ("extension": close > MA + 3.5×ATR → SMA-15) is
    applied transiently inside the per-week exit evaluator and is never
    persisted on :class:`Position` state, so it is not represented here.
    """

    NONE = "NONE"
    STALL = "STALL"
    CLIMACTIC = "CLIMACTIC"


@dataclass
class Position:
    """A currently-held strategy-managed position.

    The plumbing layer maintains the canonical store of strategy
    positions and passes a fresh list on every brain call. The brain
    treats inputs as immutable; any state changes are emitted as
    :class:`PositionAction` objects for the caller to apply.

    Fields fall into four groups:

    1. Identity & entry context — what we bought and why.
    2. Mark-to-market state — live price, peak / low tracking.
    3. Exit state — current stop level, which trail mode is active.
    4. Mechanism flags — partial taken, triple-stack state, etc.
    """

    # Identity & entry context
    ticker: str
    sector: str
    entry_date: DateLike
    entry_price: float
    qty: int
    """Current qty (post-partial if applicable)."""
    entry_type: str
    """Detector that fired the entry, e.g. ``"PULLBACK_S2"``."""
    tier: str
    """One of ``"starter"``, ``"half"``, ``"full"``."""
    cost: float
    """Actual cash outflow at entry, including fees."""

    # Stop tracking
    stop_at_entry: float
    """Original stop, never changes."""
    stop_level: float
    """Current stop (ratcheted up over time)."""

    # Mark-to-market
    current_close: float
    """This week's close."""
    peak_close: float
    """Highest weekly close since entry."""
    min_low: float = 0.0
    """Lowest intraweek low since entry (for MAE)."""
    prev_close: float = 0.0
    """Previous week's close (for climactic-tighten check)."""

    # Trail state
    trail_mode: TrailMode = TrailMode.INITIAL
    triple_stack_state: TripleStackTighten = TripleStackTighten.NONE

    # Partial profit state
    partial_taken: bool = False
    partial_sale_qty: int = 0
    partial_sale_price: float = 0.0
    partial_sale_date: DateLike | None = None
    partial_sale_proceeds: float = 0.0

    # Diagnostic / lineage
    risk_pct_at_entry: float = 0.0
    rs_13w_at_entry: float = 0.0
    regime_at_entry: str = ""
    target_type: str = ""
    rr_at_entry: float = 0.0
    mfe_week: DateLike | None = None
    """Week when ``peak_close`` was last updated."""


@dataclass
class EntrySignal:
    """A ranked recommendation to open a new position this Friday.

    The brain returns these from
    :func:`skysurf.brain.signals.generate_weekly_signals`. The caller
    decides whether to act on each (e.g., by submitting a broker order).
    The brain has no visibility into whether the order fills.
    """

    ticker: str
    sector: str
    entry_type: str
    """Which detector fired."""
    entry_price: float
    """This week's close (signal week)."""
    initial_stop: float
    """``entry − 2×ATR`` or structural, whichever is tighter."""
    qty: int
    """Sized recommendation."""
    tier: str
    """One of ``"starter"``, ``"half"``, ``"full"``."""
    entry_cost: float
    """Cash outflow including Zerodha fees (computed by the sizing logic)."""
    type_prior_score: float
    """Dynamic type-prior score used for ranking."""
    rs_13w: float
    """13-week relative strength at entry, for diagnostics."""
    target_level: float
    """Nearest historical resistance above entry, or ``entry × 1.20`` fallback."""
    rr_at_entry: float
    """``(target − entry) / (entry − stop)`` at the time of entry."""
    diagnostic: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for which filters passed, dedup decisions, etc."""


@dataclass
class PositionAction:
    """A required action on a held position emitted by manage_positions.

    The fields populated depend on ``action_type``:

    * ``HOLD`` — no other fields meaningful.
    * ``UPDATE_STOP`` — :attr:`new_stop_level`.
    * ``SWITCH_TRAIL_MA`` — :attr:`new_trail_mode` and
      :attr:`new_stop_level`.
    * ``PARTIAL_SELL`` — :attr:`sell_qty`, :attr:`sell_price`, and
      :attr:`new_stop_level` (post-HALF_BACK).
    * ``EXIT_FULL`` — :attr:`sell_qty` (== remaining qty),
      :attr:`sell_price`, and :attr:`reason`.
    """

    ticker: str
    action_type: ActionType
    reason: str
    new_stop_level: float | None = None
    new_trail_mode: TrailMode | None = None
    new_triple_stack_state: TripleStackTighten | None = None
    sell_qty: int | None = None
    sell_price: float | None = None
    proceeds: float | None = None
    """Zerodha-adjusted cash inflow for ``PARTIAL_SELL`` or ``EXIT_FULL``."""


@dataclass
class ValidationResult:
    """Result of a validation harness run."""

    achieved_metrics: dict[str, float]
    expected_metrics: dict[str, float]
    deltas: dict[str, float]
    tolerances: dict[str, float]
    passed: bool
    notes: list[str] = field(default_factory=list)
