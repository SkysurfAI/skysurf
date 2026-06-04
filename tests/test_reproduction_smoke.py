"""Smoke + (gated) full-reproduction tests for skysurf.reproduction.

The import/wiring tests run everywhere (no data, no matplotlib needed). The full
backtest is gated behind the SKYSURF_REPRO_DATA env var pointing at a data
bundle, so CI without the bundle skips it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_package_imports_without_data_or_matplotlib():
    """The subpackage and its research modules import on the base install."""
    import skysurf.reproduction as repro
    from skysurf.reproduction._research import autosweep, engine, engine_locked, type_prior, walkforward

    assert callable(repro.reproduce)
    assert repro.CONFIGS == ("phase4_best", "phase4_time_stop", "baseline")
    # engine imports must not pull in matplotlib eagerly
    assert engine.plt is not None and engine_locked.plt is not None
    assert hasattr(walkforward, "reproduce")
    assert hasattr(type_prior, "compute_type_prior")
    assert hasattr(autosweep, "build_engine_b_config")


def test_published_targets_are_declared():
    from skysurf.reproduction._research import walkforward

    best = walkforward.EXPECTED["phase4_best"]
    assert best == {"mar": 1.96, "cagr": 24.1, "maxdd": -12.3, "trades": 604}


def test_set_data_dir_is_respected():
    from skysurf.reproduction import _paths

    _paths.set_data_dir("/tmp/some-bundle")
    assert _paths.data_dir() == Path("/tmp/some-bundle").resolve()


def test_missing_bundle_raises_clear_error(tmp_path):
    import skysurf.reproduction as repro

    with pytest.raises(FileNotFoundError, match="data bundle not found"):
        repro.reproduce(data_dir=str(tmp_path / "does-not-exist"))


@pytest.mark.reproduction
@pytest.mark.skipif(
    not os.environ.get("SKYSURF_REPRO_DATA"),
    reason="set SKYSURF_REPRO_DATA to the data bundle to run the full reproduction",
)
def test_phase4_best_reproduces_headline():
    import skysurf.reproduction as repro

    result = repro.reproduce(config="phase4_best", verbose=False)
    assert result["passed"] is True, result["deltas"]
    assert abs(result["metrics"]["mar"] - 1.96) <= 0.01
    assert result["metrics"]["total_trades"] == pytest.approx(604, abs=3)
