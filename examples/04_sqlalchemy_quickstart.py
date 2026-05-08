"""04 — SQL database quickstart.

Wire :class:`SQLAlchemyDataProvider` to any SQL database (SQLite,
Postgres, MySQL, …) and run the strategy. Tables expected to exist::

    weekly_ohlcv  (ticker, date, open, high, low, close, volume)
    daily_ohlcv   (ticker, date, open, high, low, close, volume)
    nifty_weekly  (date, close, ...)
    universe      (ticker, sector, market_cap, adtv_20d)
    historical_trades (ticker, week_date, entry_type, mfe_pct, mae_pct)

If your tables are named differently, pass a :class:`TableMap` to
override the defaults.

Requires the ``sqlalchemy`` extra::

    pip install "skysurf[sqlalchemy]"

Run with::

    DATABASE_URL=postgresql://user:pass@host/db python examples/04_sqlalchemy_quickstart.py

Or for a local SQLite file::

    DATABASE_URL=sqlite:///./skysurf.db python examples/04_sqlalchemy_quickstart.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

from skysurf import SQLAlchemyDataProvider, generate_weekly_signals


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL and re-run.")
        sys.exit(1)

    provider = SQLAlchemyDataProvider(url)
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
