"""Command-line interface for reproducing the Skysurf Phase-4 backtest.

Exposed two ways, both shipped by ``pip install skysurf``:

    python -m skysurf.reproduction --data /path/to/skysurf-repro-data
    skysurf-reproduce --data /path/to/skysurf-repro-data --config phase4_best
"""
from __future__ import annotations

import argparse
import sys

from . import CONFIGS, reproduce, run_full_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skysurf-reproduce",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Reproduce the published Skysurf Phase-4 walk-forward backtest\n"
            "(headline: MAR 1.96 / CAGR 24.1% / MaxDD -12.3% / 604 trades)."
        ),
    )
    p.add_argument(
        "--data",
        metavar="DIR",
        default=None,
        help=(
            "Path to the reproduction data bundle. Defaults to $SKYSURF_REPRO_DATA, "
            "then ./skysurf-repro-data."
        ),
    )
    p.add_argument(
        "--config",
        choices=list(CONFIGS),
        default="phase4_best",
        help="Which canonical config to reproduce (default: phase4_best).",
    )
    p.add_argument(
        "--full-report",
        action="store_true",
        help="Run all three configs and print the full comparison report.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-segment progress and summary table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.full_report:
            run_full_report(data_dir=args.data)
            return 0
        result = reproduce(data_dir=args.data, config=args.config, verbose=not args.quiet)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Exit non-zero if a published target exists and we missed it.
    if result["passed"] is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
