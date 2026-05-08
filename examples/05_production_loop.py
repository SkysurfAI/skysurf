"""05 — Production weekly loop.

A cron-ready Friday-evening loop that:

1. Confirms today is Friday.
2. Loads held positions and current equity from your records.
3. Calls ``manage_positions`` to decide what to do with each held
   position (ratchet stops, take partials, exit).
4. Calls ``generate_weekly_signals`` to pick new entries.
5. Hands the resulting actions to your broker integration.

The broker integration is your responsibility. Skysurf emits decisions;
your code places the orders. This example uses placeholders for both.

Run weekly via cron, e.g. (in IST, after the cash market close)::

    5 16 * * 5  cd /opt/skysurf && .venv/bin/python examples/05_production_loop.py >> logs/skysurf.log 2>&1
"""

from __future__ import annotations

import logging
import sys
from datetime import date

from skysurf import (
    PHASE_4_V1,
    ActionType,
    DataProvider,
    EntrySignal,
    Position,
    PositionAction,
    generate_weekly_signals,
    manage_positions,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("skysurf-runner")

    today = date.today()
    if today.weekday() != 4:  # 4 == Friday
        log.warning("Not Friday (weekday=%d); skipping run.", today.weekday())
        return

    provider = build_provider()
    held = load_held_positions()
    equity = load_total_equity()
    log.info("Starting weekly run: %d held positions, equity=%.0f", len(held), equity)

    # ── Manage exits first (Rule 1: exits before entries) ──────────
    actions = manage_positions(provider=provider, as_of_date=today, current_positions=held)
    log.info("manage_positions → %d actions", len(actions))
    handle_actions(actions)

    # ── Generate new entries ───────────────────────────────────────
    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=today,
        current_positions=held,
        total_equity=equity,
    )
    log.info("generate_weekly_signals → %d signals", len(signals))

    # Cap how many to take this week so we don't exceed max_positions.
    slots_open = PHASE_4_V1.max_positions - len(held)
    handle_signals(signals[:slots_open])
    log.info("Done.")


# ── Wiring stubs — replace with your real plumbing ────────────────


def build_provider() -> DataProvider:
    """Construct your :class:`DataProvider`. See examples 02–04."""
    raise NotImplementedError("Wire your data source here. See examples 01–04 for connector demos.")


def load_held_positions() -> list[Position]:
    """Load current held positions from your records."""
    return []


def load_total_equity() -> float:
    """Load current total portfolio equity (cash + invested)."""
    raise NotImplementedError("Read equity from your broker / records.")


def handle_actions(actions: list[PositionAction]) -> None:
    """Place broker orders for the brain's exit-side decisions."""
    for action in actions:
        if action.action_type == ActionType.HOLD:
            continue
        if action.action_type == ActionType.UPDATE_STOP:
            print(f"UPDATE_STOP {action.ticker} → {action.new_stop_level}")
        elif action.action_type == ActionType.SWITCH_TRAIL_MA:
            print(f"SWITCH_TRAIL_MA {action.ticker} → {action.new_stop_level}")
        elif action.action_type == ActionType.PARTIAL_SELL:
            print(f"PARTIAL_SELL {action.ticker} qty={action.sell_qty} @ {action.sell_price}")
        elif action.action_type == ActionType.EXIT_FULL:
            print(f"EXIT_FULL {action.ticker} qty={action.sell_qty} @ {action.sell_price}")


def handle_signals(signals: list[EntrySignal]) -> None:
    """Place broker orders for the brain's entry signals."""
    for s in signals:
        print(
            f"BUY {s.ticker:14s} qty={s.qty:>4d} @ {s.entry_price:.2f}  "
            f"stop={s.initial_stop:.2f}  ({s.entry_type})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Production loop failed")
        sys.exit(1)
