"""CLI for regenerating the reproduction data bundle from raw OHLCV.

    skysurf-build --ohlcv /path/to/raw_ohlcv --out /path/to/skysurf-repro-data
    python -m skysurf.reproduction.pipeline --ohlcv ... --out ...
"""
from __future__ import annotations

import argparse
import sys

from . import build


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skysurf-build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Regenerate the Skysurf reproduction data bundle from raw daily OHLCV.\n"
            "Runs the verbatim research pipeline: OHLCV -> weekly cache -> regimes\n"
            "-> entry detectors -> entries_all.csv."
        ),
    )
    p.add_argument(
        "--ohlcv",
        metavar="DIR",
        default=None,
        help=(
            "Directory with stocks_daily + benchmark_daily (.parquet/.csv). "
            "Defaults to $SKYSURF_REPRO_OHLCV, then ./skysurf-repro-ohlcv."
        ),
    )
    p.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help=(
            "Bundle output directory for the regenerated intermediates. "
            "Defaults to $SKYSURF_REPRO_DATA, then ./skysurf-repro-data."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = build(ohlcv_dir=args.ohlcv, data_dir=args.out)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nIntermediates written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
