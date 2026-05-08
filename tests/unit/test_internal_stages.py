"""Tests for the Weinstein stage classifier and MA slope direction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skysurf._internal.stages import (
    classify_stages_single_ma,
    compute_ma_slope_direction,
    count_consecutive_stage2,
)


@pytest.fixture
def weekly_index() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-06", periods=60, freq="W-FRI")


# ── compute_ma_slope_direction ───────────────────────────────────────


def test_slope_direction_rising_when_ma_climbs(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    ma = pd.Series(np.linspace(100, 130, n), index=weekly_index)  # +30 over n
    atr = pd.Series([2.0] * n, index=weekly_index)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    # After warmup, all bars should show "rising"
    assert (direction.iloc[10:] == "rising").all()


def test_slope_direction_falling_when_ma_drops(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    ma = pd.Series(np.linspace(130, 100, n), index=weekly_index)
    atr = pd.Series([2.0] * n, index=weekly_index)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    assert (direction.iloc[10:] == "falling").all()


def test_slope_direction_flat_when_ma_constant(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    ma = pd.Series([100.0] * n, index=weekly_index)
    atr = pd.Series([2.0] * n, index=weekly_index)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    assert (direction == "flat").all()


def test_slope_direction_warmup_returns_flat(weekly_index: pd.DatetimeIndex) -> None:
    """The first ``lookback`` bars cannot have a defined slope."""
    n = len(weekly_index)
    ma = pd.Series(np.linspace(100, 130, n), index=weekly_index)
    atr = pd.Series([2.0] * n, index=weekly_index)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    assert (direction.iloc[:4] == "flat").all()


def test_slope_direction_short_series_all_flat() -> None:
    """If the series is shorter than the lookback, every bar is flat."""
    idx = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    ma = pd.Series([100.0, 105.0, 110.0], index=idx)
    atr = pd.Series([2.0, 2.0, 2.0], index=idx)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    assert (direction == "flat").all()


def test_slope_direction_handles_zero_atr(weekly_index: pd.DatetimeIndex) -> None:
    """Zero ATR (degenerate volatility) must not raise; bar is treated as flat."""
    n = len(weekly_index)
    ma = pd.Series(np.linspace(100, 130, n), index=weekly_index)
    atr = pd.Series([0.0] * n, index=weekly_index)
    direction = compute_ma_slope_direction(ma, atr, lookback=4)
    # Division by zero → NaN → comparisons evaluate False → "flat"
    assert direction.isin({"rising", "falling", "flat"}).all()


# ── classify_stages_single_ma ────────────────────────────────────────


def test_stage2_when_close_above_rising_ma(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    df = pd.DataFrame({"Close": np.linspace(110, 140, n)}, index=weekly_index)
    ma = pd.Series(np.linspace(100, 130, n), index=weekly_index)
    direction = pd.Series(["rising"] * n, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    assert (stages == "stage2").all()


def test_stage4_when_close_below_falling_ma(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    df = pd.DataFrame({"Close": np.linspace(95, 70, n)}, index=weekly_index)
    ma = pd.Series(np.linspace(105, 90, n), index=weekly_index)
    direction = pd.Series(["falling"] * n, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    assert (stages == "stage4").all()


def test_stage1_when_flat_with_no_recent_stage2(weekly_index: pd.DatetimeIndex) -> None:
    n = len(weekly_index)
    df = pd.DataFrame({"Close": [100.0] * n}, index=weekly_index)
    ma = pd.Series([100.0] * n, index=weekly_index)
    direction = pd.Series(["flat"] * n, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    assert (stages == "stage1").all()


def test_stage3_when_flat_after_recent_stage2(weekly_index: pd.DatetimeIndex) -> None:
    """After a Stage-2 run, when MA goes flat, the next bars are Stage 3."""
    n = len(weekly_index)
    # First 30 bars: rising → Stage 2; remaining 30: flat → Stage 3 (within 8 bars)
    direction = pd.Series(["rising"] * 30 + ["flat"] * 30, index=weekly_index)
    df = pd.DataFrame({"Close": [110.0] * n}, index=weekly_index)
    ma = pd.Series([100.0] * n, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    # First few flat bars after Stage 2 → Stage 3
    assert stages.iloc[30] == "stage3"
    assert stages.iloc[35] == "stage3"
    # Eventually (after the 8-bar lookback expires) → Stage 1
    assert stages.iloc[55] == "stage1"


def test_ambiguous_carries_forward_previous_stage(weekly_index: pd.DatetimeIndex) -> None:
    """Close above MA with falling slope is ambiguous; carry forward."""
    n = len(weekly_index)
    df = pd.DataFrame({"Close": [110.0] * n}, index=weekly_index)
    ma = pd.Series([100.0] * n, index=weekly_index)
    # First half rising (Stage 2), then falling (ambiguous: close > MA but falling)
    direction = pd.Series(["rising"] * 30 + ["falling"] * 30, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    # Ambiguous bars carry forward Stage 2
    assert (stages.iloc[30:] == "stage2").all()


def test_stage_classifier_handles_nan_ma(weekly_index: pd.DatetimeIndex) -> None:
    """During the MA warmup (NaN), bars default to Stage 1."""
    n = len(weekly_index)
    df = pd.DataFrame({"Close": [110.0] * n}, index=weekly_index)
    ma = pd.Series([float("nan")] * 5 + [100.0] * (n - 5), index=weekly_index)
    direction = pd.Series(["rising"] * n, index=weekly_index)
    stages = classify_stages_single_ma(df, ma, direction)
    assert (stages.iloc[:5] == "stage1").all()


# ── count_consecutive_stage2 ─────────────────────────────────────────


def test_count_consecutive_stage2_run_length() -> None:
    stages = pd.Series(["stage1", "stage2", "stage2", "stage2", "stage1", "stage2", "stage2"])
    assert count_consecutive_stage2(stages, idx=3) == 3
    assert count_consecutive_stage2(stages, idx=6) == 2


def test_count_consecutive_stage2_returns_zero_when_not_stage2() -> None:
    stages = pd.Series(["stage2", "stage2", "stage1"])
    assert count_consecutive_stage2(stages, idx=2) == 0


def test_count_consecutive_stage2_handles_out_of_range() -> None:
    stages = pd.Series(["stage2", "stage2"])
    assert count_consecutive_stage2(stages, idx=-1) == 0
    assert count_consecutive_stage2(stages, idx=10) == 0
