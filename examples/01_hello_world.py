"""01 — Hello World on synthetic data.

The smallest possible end-to-end example. Generates a tiny synthetic
universe in-memory, builds an :class:`InMemoryDataProvider`, and runs
both ``generate_weekly_signals`` and ``manage_positions`` against it.

No external data, no setup. Run with::

    python examples/01_hello_world.py

You'll typically see ``0`` signals on this synthetic data — random walks
rarely fire PHASE_4_V1 detectors. The point of this example is to prove
the install + pipeline works on your machine. For real signals, point
the strategy at real OHLCV via one of the file/SQL connectors (see
examples 02–04).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from skysurf import (
    InMemoryDataProvider,
    generate_weekly_signals,
    manage_positions,
)


def main() -> None:
    rng = np.random.default_rng(seed=42)
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")

    # Synthetic 5-ticker universe.
    weekly_ohlcv: dict[str, pd.DataFrame] = {}
    for ticker, start_price in [
        ("ALPHA.NS", 100.0),
        ("BETA.NS", 250.0),
        ("GAMMA.NS", 800.0),
        ("DELTA.NS", 45.0),
        ("EPSILON.NS", 1500.0),
    ]:
        returns = rng.normal(0.002, 0.04, size=len(weeks))
        close = start_price * np.cumprod(1 + returns)
        weekly_ohlcv[ticker] = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.01, len(weeks))),
                "High": close * (1 + np.abs(rng.normal(0, 0.02, len(weeks)))),
                "Low": close * (1 - np.abs(rng.normal(0, 0.02, len(weeks)))),
                "Close": close,
                "Volume": rng.integers(50_000, 500_000, len(weeks)),
            },
            index=weeks,
        )

    nifty_close = 17_000 * np.cumprod(1 + rng.normal(0.001, 0.02, len(weeks)))
    nifty = pd.DataFrame({"Close": nifty_close}, index=weeks)

    universe = pd.DataFrame(
        [
            {"ticker": "ALPHA.NS", "sector": "TECH", "market_cap": 5e10, "adtv_20d": 5e7},
            {"ticker": "BETA.NS", "sector": "FINANCIAL", "market_cap": 8e10, "adtv_20d": 1e8},
            {"ticker": "GAMMA.NS", "sector": "ENERGY", "market_cap": 1.5e11, "adtv_20d": 8e7},
            {"ticker": "DELTA.NS", "sector": "TECH", "market_cap": 2e10, "adtv_20d": 3e7},
            {"ticker": "EPSILON.NS", "sector": "FMCG", "market_cap": 1.2e11, "adtv_20d": 6e7},
        ]
    )

    provider = InMemoryDataProvider(
        weekly_ohlcv=weekly_ohlcv,
        daily_ohlcv=weekly_ohlcv,
        nifty_weekly=nifty,
        sector_indices_weekly={},
        universe=universe,
        historical_trades=pd.DataFrame(
            columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
        ),
    )

    print("Generating signals for 2024-06-07...")
    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    print(f"  → {len(signals)} entry signal(s)")
    for s in signals:
        print(
            f"    {s.ticker:14s}  {s.entry_type:18s}  "
            f"entry={s.entry_price:>8.2f}  stop={s.initial_stop:>8.2f}  qty={s.qty}"
        )

    print("\nManaging held positions for 2024-06-07...")
    actions = manage_positions(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
    )
    print(f"  → {len(actions)} action(s) (no held positions in this demo)")


if __name__ == "__main__":
    main()
