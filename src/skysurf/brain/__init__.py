"""Brain — strategy types, configuration, and decision functions.

This subpackage contains the immutable :class:`StrategyConfig`, the
runtime types (:class:`Position`, :class:`EntrySignal`,
:class:`PositionAction`), and (in subsequent modules) the per-week
signal-generation and position-management functions.

Most callers import directly from :mod:`skysurf` (the top-level package
re-exports everything stable) — this submodule namespace exists so the
internal layout is discoverable.
"""
from __future__ import annotations

from skysurf.brain.config import PHASE_4_V1, SUSPENDED_ENTRY_TYPES, StrategyConfig
from skysurf.brain.types import (
    ActionType,
    EntrySignal,
    Position,
    PositionAction,
    TrailMode,
    TripleStackTighten,
    ValidationResult,
)

__all__ = [
    "PHASE_4_V1",
    "SUSPENDED_ENTRY_TYPES",
    "ActionType",
    "EntrySignal",
    "Position",
    "PositionAction",
    "StrategyConfig",
    "TrailMode",
    "TripleStackTighten",
    "ValidationResult",
]
