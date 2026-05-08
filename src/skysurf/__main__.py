"""Command-line entry point for ``python -m skysurf``.

Currently exposes one subcommand:

* ``python -m skysurf doctor`` — installation smoke check. Verifies
  that core dependencies import, runs a synthetic-data hello-world
  through the full pipeline, and reports any issues. Modeled on
  ``pip check`` and ``mypy --install-types``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from datetime import date

_REQUIRED_PACKAGES: tuple[str, ...] = ("pandas", "numpy", "scipy", "ta")
_OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("pyarrow", "skysurf[parquet]"),
    ("sqlalchemy", "skysurf[sqlalchemy]"),
    ("trendln", "skysurf[trendlines]"),
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {mark} {label}{suffix}")
    return ok


def doctor(argv: list[str]) -> int:
    """Run installation diagnostics. Returns 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(
        prog="python -m skysurf doctor",
        description="Skysurf installation health check.",
    )
    parser.parse_args(argv)

    print("Skysurf doctor — installation health check")
    print("=" * 50)

    overall_ok = True

    # ── Skysurf import ──────────────────────────────────────────
    try:
        import skysurf

        _check("skysurf imports", True, f"version {skysurf.__version__}")
    except ImportError as exc:
        _check("skysurf imports", False, str(exc))
        return 1

    print("\n  Required dependencies:")
    for pkg in _REQUIRED_PACKAGES:
        ok = importlib.util.find_spec(pkg) is not None
        if not ok:
            overall_ok = False
        _check(pkg, ok, "" if ok else "missing — run 'pip install skysurf'")

    print("\n  Optional dependencies:")
    for pkg, install_hint in _OPTIONAL_PACKAGES:
        ok = importlib.util.find_spec(pkg) is not None
        _check(
            pkg,
            ok,
            "" if ok else f"not installed — install with 'pip install {install_hint}'",
        )

    print("\n  Hello-world pipeline:")
    try:
        ok = _run_hello_world()
        if not ok:
            overall_ok = False
    except Exception as exc:
        _check("end-to-end pipeline", False, str(exc))
        traceback.print_exc()
        overall_ok = False

    print("\n" + "=" * 50)
    if overall_ok:
        print("All checks passed. Skysurf is ready to use.")
        return 0
    print("One or more checks failed. See messages above.")
    return 1


def _run_hello_world() -> bool:
    """Build a tiny synthetic dataset and run the full pipeline."""
    import numpy as np
    import pandas as pd

    from skysurf import (
        InMemoryDataProvider,
        generate_weekly_signals,
        manage_positions,
    )

    rng = np.random.default_rng(seed=0)
    weeks = pd.date_range("2023-01-06", periods=104, freq="W-FRI")
    weekly_ohlcv: dict[str, pd.DataFrame] = {}
    for ticker, start in [("ALPHA.NS", 100.0), ("BETA.NS", 250.0), ("GAMMA.NS", 800.0)]:
        returns = rng.normal(0.001, 0.04, size=len(weeks))
        close = start * np.cumprod(1 + returns)
        weekly_ohlcv[ticker] = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.01, len(weeks))),
                "High": close * (1 + np.abs(rng.normal(0, 0.02, len(weeks)))),
                "Low": close * (1 - np.abs(rng.normal(0, 0.02, len(weeks)))),
                "Close": close,
                "Volume": rng.integers(10_000, 100_000, len(weeks)),
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
        ]
    )
    historical_trades = pd.DataFrame(
        columns=["ticker", "week_date", "entry_type", "mfe_pct", "mae_pct"]
    )
    provider = InMemoryDataProvider(
        weekly_ohlcv=weekly_ohlcv,
        daily_ohlcv=weekly_ohlcv,
        nifty_weekly=nifty,
        sector_indices_weekly={},
        universe=universe,
        historical_trades=historical_trades,
    )

    signals = generate_weekly_signals(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
        total_equity=500_000.0,
    )
    _check(
        "generate_weekly_signals on synthetic data",
        True,
        f"returned {len(signals)} signal(s)",
    )

    actions = manage_positions(
        provider=provider,
        as_of_date=date(2024, 6, 7),
        current_positions=[],
    )
    _check(
        "manage_positions on synthetic data",
        True,
        f"returned {len(actions)} action(s)",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m skysurf <command>")
        print()
        print("Commands:")
        print("  doctor     Run installation health check.")
        return 0 if argv else 1

    command, *rest = argv
    if command == "doctor":
        return doctor(rest)

    print(f"Unknown command: {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
