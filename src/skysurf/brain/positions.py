"""Position management — public entry point for exit-side decisions.

Given a :class:`~skysurf.data.provider.DataProvider`, an ``as_of_date``
(a Friday close), and the list of currently-held positions, returns a
list of :class:`PositionAction` objects describing what (if anything) to
do with each position this week: hold, ratchet the stop, take partial
profit, switch to a tighter trailing MA, or fully exit.

The brain emits decisions; the caller places the broker orders. The
brain has no knowledge of whether orders fill.

In the v0.1.0 release of this package, the underlying exit-evaluation
logic is being vendored from the production research repository. Until
that work lands (tracked in :doc:`/ROADMAP.md`), calling this function
raises :class:`NotImplementedError`. The function signature, types,
and configuration surface are stable and safe to integrate against.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from skysurf.brain.config import PHASE_4_V1, StrategyConfig
from skysurf.brain.types import Position, PositionAction
from skysurf.data.provider import DataProvider


def manage_positions(
    provider: DataProvider,
    as_of_date: date | pd.Timestamp,
    current_positions: list[Position],
    config: StrategyConfig = PHASE_4_V1,
) -> list[PositionAction]:
    """Return the action to take on each held position this week.

    Args:
        provider: Concrete :class:`DataProvider` supplying OHLCV and
            regime data.
        as_of_date: Friday close. The actions returned are valid for the
            following week's open.
        current_positions: List of currently-held positions. The brain
            does not mutate these; new state is expressed via
            :class:`PositionAction` objects for the caller to apply.
        config: Strategy configuration; defaults to the canonical
            :data:`PHASE_4_V1` lock.

    Returns:
        One :class:`PositionAction` per input position. The set of
        possible action types is documented on :class:`ActionType`.

    Raises:
        NotImplementedError: In v0.1.0, until the engine vendoring
            lands. See :doc:`/ROADMAP.md`.
    """
    raise NotImplementedError(
        "manage_positions is wired up to its public surface, but the "
        "strategy engine is being vendored from the production research repo. "
        "It will be available in v0.2.0. Track ROADMAP.md."
    )
