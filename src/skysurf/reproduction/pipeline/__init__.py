"""skysurf.reproduction.pipeline — regenerate the data bundle from raw OHLCV.

This is the open-sourced *data-building* half of the reproduction: it takes raw
daily OHLCV (stocks + benchmark) and runs the verbatim research pipeline
(indicators -> weekly cache -> regimes -> swing/breakout/pullback/VCP detectors
-> ``entries_all.csv``), writing every intermediate into the reproduction data
bundle. Combined with ``skysurf.reproduction``, this gives a full
**raw OHLCV -> entries -> headline result** chain, all in open code.

Quick start
-----------
    import skysurf.reproduction.pipeline as build

    build.build(
        ohlcv_dir="/path/to/raw_ohlcv",       # stocks_daily + benchmark_daily
        data_dir="/path/to/skysurf-repro-data",  # bundle output
    )

Or from the command line::

    skysurf-build --ohlcv /path/to/raw_ohlcv --out /path/to/skysurf-repro-data

The raw daily-OHLCV input schema is documented in
``docs/reproduction-data-bundle.md``.

Note: building also requires the lagged-entries and trade-stats steps to produce
a fully runnable bundle (``entries_all_lagged.csv`` via ``rebuild_entries_lagged``
and ``trade_stats_all.csv`` via the entry-stats generator). See the bundle doc
for the remaining steps.
"""
from __future__ import annotations

from .. import _paths

__all__ = ["build", "build_all", "build_lagged", "build_entry_stats", "build_sector_dimensions"]


def build_all(
    ohlcv_dir: str | None = None,
    data_dir: str | None = None,
    verbose: bool = True,
):
    """Run the **entire** OHLCV → bundle pipeline in one call.

    Chains every builder so the resulting ``data_dir`` is a complete, runnable
    reproduction bundle:

      1. ``build``                  → cache, regimes, nifty, universe,
                                       ``entries_all.csv`` + ``entries_all_lagged.csv``
      2. ``build_sector_dimensions``→ ``sector_regime_weekly.csv``,
                                       ``captier_regime_weekly.csv``, ``ticker_metadata.csv``
      3. ``build_entry_stats``      → ``trade_stats_all.csv`` (dynamic type-prior input)

    Args:
        ohlcv_dir: raw daily OHLCV dir (``stocks_daily`` + ``benchmark_daily`` +
            ``sector_daily`` + ``sector_index_master``).
        data_dir: bundle output dir.

    Returns:
        The output ``data_dir`` path — ready for ``skysurf.reproduction.reproduce``.
    """
    if data_dir is not None:
        _paths.set_data_dir(data_dir)
    if ohlcv_dir is not None:
        _paths.set_ohlcv_dir(ohlcv_dir)

    out = build(ohlcv_dir=_paths.ohlcv_dir(), data_dir=_paths.data_dir(), lagged=True, verbose=verbose)
    build_sector_dimensions(ohlcv_dir=_paths.ohlcv_dir(), data_dir=_paths.data_dir())
    build_entry_stats(data_dir=_paths.data_dir())
    return out


def build_entry_stats(data_dir: str | None = None, output_dir: str | None = None):
    """Regenerate ``trade_stats_all.csv`` (dynamic type-prior input) from the bundle.

    Wraps :func:`skysurf.reproduction.pipeline.entry_stats.build_entry_stats`.
    Requires ``build`` to have produced ``entries_all.csv`` + ``stock_weekly_cache/``.
    """
    from . import entry_stats

    return entry_stats.build_entry_stats(data_dir=data_dir, output_dir=output_dir)


def build_sector_dimensions(ohlcv_dir: str | None = None, data_dir: str | None = None):
    """Regenerate sector/cap-tier regime + ticker-metadata bundle CSVs from raw OHLCV.

    Wraps :func:`skysurf.reproduction.pipeline.sector_regime.build_sector_dimensions`.
    Needs ``sector_daily`` (+ ``stocks_daily``) in ``ohlcv_dir`` and the
    ``stock_weekly_cache/`` from a prior ``build``.
    """
    from . import sector_regime

    return sector_regime.build_sector_dimensions(ohlcv_dir=ohlcv_dir, data_dir=data_dir)


def build(
    ohlcv_dir: str | None = None,
    data_dir: str | None = None,
    lagged: bool = True,
    verbose: bool = True,
):
    """Regenerate the reproduction intermediates from raw daily OHLCV.

    Args:
        ohlcv_dir: directory with ``stocks_daily`` + ``benchmark_daily``
            (``.parquet``/``.csv``). Falls back to ``SKYSURF_REPRO_OHLCV`` then
            ``./skysurf-repro-ohlcv``.
        data_dir: bundle output directory (where intermediates are written).
            Falls back to ``SKYSURF_REPRO_DATA`` then ``./skysurf-repro-data``.
        verbose: currently always prints step progress (the research pipeline
            is chatty); reserved for future suppression.

    Returns:
        The output ``data_dir`` path.
    """
    if data_dir is not None:
        _paths.set_data_dir(data_dir)
    if ohlcv_dir is not None:
        _paths.set_ohlcv_dir(ohlcv_dir)

    # Imported lazily so the data/ohlcv dirs are resolved before the builder
    # binds its module-level path constants at import time.
    from . import builder

    out = builder.main(ohlcv_dir=_paths.ohlcv_dir())
    if lagged:
        from . import lagged as _lagged

        _lagged.main()
    return out


def build_lagged(data_dir: str | None = None):
    """Build ``entries_all_lagged.csv`` from ``entries_all.csv`` + the cache.

    The look-ahead-fixed entries the Phase-4 configs consume. Requires
    ``build`` (or the original builder) to have produced ``entries_all.csv``.
    """
    if data_dir is not None:
        _paths.set_data_dir(data_dir)
    from . import lagged as _lagged

    return _lagged.main()
