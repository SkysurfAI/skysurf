"""Weekly signal generation — public entry point for new-position recommendations.

Given a :class:`~skysurf.data.provider.DataProvider`, an ``as_of_date``
(a Friday close), the current portfolio of held positions, and the
total equity, returns a ranked list of :class:`EntrySignal` objects
representing recommended new positions for the upcoming week.

The brain itself is stateless. The caller maintains the canonical store
of held positions and re-passes them on every call.

In the v0.1.0 release of this package, the underlying strategy engine
is being vendored from the production research repository. Until that
work lands (tracked in :doc:`/ROADMAP.md`), calling this function
raises :class:`NotImplementedError`. The function signature, types,
and configuration surface are stable and safe to integrate against.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from skysurf.brain.config import PHASE_4_V1, StrategyConfig
from skysurf.brain.types import EntrySignal, Position
from skysurf.data.provider import DataProvider


def generate_weekly_signals(
    provider: DataProvider,
    as_of_date: date | pd.Timestamp,
    current_positions: list[Position],
    total_equity: float,
    config: StrategyConfig = PHASE_4_V1,
) -> list[EntrySignal]:
    """Return ranked entry signals for the week ending ``as_of_date``.

    Args:
        provider: Concrete :class:`DataProvider` supplying OHLCV, regime,
            sector, universe, and historical-trades data.
        as_of_date: Friday close. The signals returned are valid for the
            following week's open.
        current_positions: List of currently-held positions. Used to
            apply per-sector and total-position constraints.
        total_equity: Current portfolio equity (cash + invested).
        config: Strategy configuration; defaults to the canonical
            :data:`PHASE_4_V1` lock.

    Returns:
        Ranked list of :class:`EntrySignal` instances, highest-priority
        first. The list respects ``config.sector_limit`` and
        ``config.max_positions`` after counting the existing
        ``current_positions``.

    Raises:
        NotImplementedError: In v0.1.0, until the engine vendoring
            lands. See :doc:`/ROADMAP.md`.
    """
    raise NotImplementedError(
        "generate_weekly_signals is wired up to its public surface, but the "
        "strategy engine is being vendored from the production research repo. "
        "It will be available in v0.2.0. Track ROADMAP.md."
    )
