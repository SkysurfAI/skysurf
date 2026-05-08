"""03 — Parquet directory quickstart.

Same as the CSV example but with Parquet. Recommended for production:
faster reads, smaller on disk, dtypes preserved.

Requires the ``parquet`` extra::

    pip install "skysurf[parquet]"

Layout (under ``$SKYSURF_DATA_DIR``)::

    weekly_ohlcv/<TICKER>.parquet
    daily_ohlcv/<TICKER>.parquet
    nifty_weekly.parquet
    universe.parquet
    historical_trades.parquet

Run with::

    SKYSURF_DATA_DIR=/path/to/parquet python examples/03_parquet_quickstart.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

from skysurf import ParquetDataProvider, generate_weekly_signals


def main() -> None:
    data_dir = os.environ.get("SKYSURF_DATA_DIR")
    if not data_dir:
        print("Set SKYSURF_DATA_DIR=/path/to/parquet and re-run.")
        sys.exit(1)

    provider = ParquetDataProvider(data_dir)
    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    print(f"Generated {len(signals)} signal(s).")
    for s in signals:
        print(f"  {s.ticker:14s}  {s.entry_type:18s}  qty={s.qty}")


if __name__ == "__main__":
    main()
