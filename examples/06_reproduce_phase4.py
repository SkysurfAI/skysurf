"""06 — Reproduce the published Phase-4 walk-forward backtest.

Reproduces the headline Skysurf numbers (MAR 1.96 / CAGR 24.1% / MaxDD -12.3% /
604 trades) from the reproduction data bundle.

Prerequisites:
    pip install skysurf
    # download the reproduction data bundle (~450 MB) — see
    # docs/reproduction-data-bundle.md — and set its path below or via env var.

Run:
    SKYSURF_REPRO_DATA=/path/to/skysurf-repro-data python examples/06_reproduce_phase4.py
    # ...or pass the path directly in code (see below).
"""
from __future__ import annotations

import os
import sys

import skysurf.reproduction as repro


def main() -> int:
    data_dir = os.environ.get("SKYSURF_REPRO_DATA")  # or hard-code a path here

    result = repro.reproduce(data_dir=data_dir, config="phase4_best", verbose=True)

    m = result["metrics"]
    print(f"\nReproduced: MAR={m['mar']} CAGR={m['cagr']}% MaxDD={m['maxdd']}% trades={m['total_trades']}")

    if result["passed"]:
        print("Exact reproduction confirmed (within published tolerance).")
        return 0
    print("Numbers fell outside tolerance — check the data bundle version.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
