"""skysurf.reproduction — reproduce the published Phase-4 backtest.

This subpackage ships the *exact* research walk-forward engine that produced the
headline Skysurf numbers:

    Config            MAR    CAGR     MaxDD    Trades
    Phase4 Best       1.96   24.1%    -12.3%   604

It is the unmodified research code (engine + driver + dynamic type-prior),
wired to read a single flat *data bundle* instead of the original repo paths.
Bring the bundle (distributed separately — ~450 MB; see ``docs/data-schema.md``)
and you can reproduce the numbers bit-for-bit.

Quick start
-----------
    import skysurf.reproduction as repro

    result = repro.reproduce(data_dir="/path/to/skysurf-repro-data")
    print(result["metrics"]["mar"])   # -> 1.96
    print(result["passed"])           # -> True

Or from the command line (ships with `pip install skysurf`)::

    python -m skysurf.reproduction --data /path/to/skysurf-repro-data
    skysurf-reproduce --data /path/to/skysurf-repro-data --config phase4_best

Charts are optional; install ``skysurf[reproduction]`` for matplotlib output.
The reproduction backtest itself runs on the base install.
"""
from __future__ import annotations

from ._paths import data_dir, set_data_dir

__all__ = ["reproduce", "set_data_dir", "data_dir", "CONFIGS"]

CONFIGS = ("phase4_best", "phase4_time_stop", "baseline")


def reproduce(
    data_dir: str | None = None,  # noqa: A002 - matches the public kwarg name
    config: str = "phase4_best",
    verbose: bool = True,
) -> dict:
    """Run a canonical Phase-4 walk-forward and return its metrics.

    Args:
        data_dir: path to the reproduction data bundle. If omitted, falls back
            to the ``SKYSURF_REPRO_DATA`` env var, then ``./skysurf-repro-data``.
        config: ``"phase4_best"`` (default, the published headline),
            ``"phase4_time_stop"``, or ``"baseline"``.
        verbose: print an achieved-vs-expected summary table.

    Returns:
        dict with ``config``, ``metrics``, ``per_segment``, ``expected``,
        ``deltas`` and ``passed`` (see :func:`._research.walkforward.reproduce`).
    """
    if data_dir is not None:
        set_data_dir(data_dir)
    # Imported lazily so the data dir is resolved before the research modules
    # bind their path constants at import time.
    from ._paths import require_data_dir

    require_data_dir()
    from ._research import walkforward

    return walkforward.reproduce(config=config, verbose=verbose)


def run_full_report(data_dir: str | None = None) -> None:
    """Run all three configs and print the full comparison report (research main)."""
    if data_dir is not None:
        set_data_dir(data_dir)
    from ._paths import require_data_dir

    require_data_dir()
    from ._research import walkforward

    walkforward.main()
