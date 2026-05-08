"""02 — CSV directory quickstart.

Point :class:`CsvDataProvider` at a directory of CSVs in the canonical
schema (see ``docs/data-schema.md``) and run the strategy. The
directory layout the connector expects::

    /path/to/data/
    ├── weekly_ohlcv/
    │   ├── RELIANCE.NS.csv
    │   └── ...
    ├── daily_ohlcv/
    │   └── ...
    ├── nifty_weekly.csv
    ├── universe.csv
    └── historical_trades.csv

Each OHLCV CSV must have columns: ``date, Open, High, Low, Close,
Volume`` (date can be any pandas-parseable format).

Run with::

    SKYSURF_DATA_DIR=/path/to/data python examples/02_csv_quickstart.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

from skysurf import CsvDataProvider, generate_weekly_signals


def main() -> None:
    data_dir = os.environ.get("SKYSURF_DATA_DIR")
    if not data_dir:
        print("Set SKYSURF_DATA_DIR=/path/to/data and re-run.")
        print("See docs/data-schema.md for the expected layout.")
        sys.exit(1)

    provider = CsvDataProvider(data_dir)
    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    print(f"Generated {len(signals)} signal(s).")
    for s in signals:
        print(
            f"  {s.ticker:14s}  {s.entry_type:18s}  "
            f"entry={s.entry_price:>8.2f}  stop={s.initial_stop:>8.2f}  qty={s.qty}"
        )


if __name__ == "__main__":
    main()
