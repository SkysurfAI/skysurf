"""Data-directory resolution for the reproduction package.

Every research module in :mod:`skysurf.reproduction._research` reads its input
CSVs from a single *flat* directory — the Phase-4 reproduction data bundle
(distributed separately from the code; see ``docs/data-schema.md``).

Resolution order for the data directory:

  1. An explicit override set via :func:`set_data_dir` (what the public
     ``reproduce(data_dir=...)`` API and the CLI ``--data`` flag use).
  2. The ``SKYSURF_REPRO_DATA`` environment variable.
  3. ``./skysurf-repro-data`` relative to the current working directory.

The override / env var **must be set before the research modules are imported**,
because they bind their path constants at import time. The public API enforces
this by importing the research modules lazily, only after the data directory is
resolved.
"""
from __future__ import annotations

import os
from pathlib import Path

_OVERRIDE: Path | None = None
_OHLCV_OVERRIDE: Path | None = None


def set_data_dir(path: str | os.PathLike[str]) -> None:
    """Set an explicit data-bundle directory (takes precedence over the env var)."""
    global _OVERRIDE
    _OVERRIDE = Path(path).expanduser().resolve()


def set_ohlcv_dir(path: str | os.PathLike[str]) -> None:
    """Set the raw daily-OHLCV input directory used by the builder."""
    global _OHLCV_OVERRIDE
    _OHLCV_OVERRIDE = Path(path).expanduser().resolve()


def ohlcv_dir() -> Path:
    """Return the raw daily-OHLCV input directory (builder input).

    Resolution: :func:`set_ohlcv_dir` → ``SKYSURF_REPRO_OHLCV`` env var →
    ``./skysurf-repro-ohlcv``.
    """
    if _OHLCV_OVERRIDE is not None:
        return _OHLCV_OVERRIDE
    env = os.environ.get("SKYSURF_REPRO_OHLCV")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "skysurf-repro-ohlcv").resolve()


def data_dir() -> Path:
    """Return the resolved reproduction data-bundle directory."""
    if _OVERRIDE is not None:
        return _OVERRIDE
    env = os.environ.get("SKYSURF_REPRO_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "skysurf-repro-data").resolve()


def output_dir() -> Path:
    """Return the directory for any optional run artifacts (trades/equity/charts).

    Never written to by the walk-forward itself; only the engines' own
    ``write_outputs`` / chart helpers use it. Defaults next to the data bundle's
    sibling so it never tries to write inside the installed package.
    """
    env = os.environ.get("SKYSURF_REPRO_OUTPUT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "skysurf-repro-output").resolve()


def require_data_dir() -> Path:
    """Return the data dir, raising a clear error if it does not exist."""
    d = data_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"Reproduction data bundle not found at: {d}\n"
            "Download the Phase-4 data bundle and point to it with either\n"
            "  - reproduce(data_dir=...) / the --data CLI flag, or\n"
            "  - the SKYSURF_REPRO_DATA environment variable.\n"
            "See docs/data-schema.md for the bundle layout."
        )
    return d
