"""Base-to-breakout transition detection for Test 2."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _compute_volume_ratio(df: pd.DataFrame, week_idx: int, lookback: int = 20) -> float:
    """Ratio of breakout-week volume to trailing average volume."""
    start = max(0, week_idx - lookback)
    avg_vol = df["Volume"].iloc[start:week_idx].mean()
    if avg_vol is None or avg_vol == 0 or np.isnan(avg_vol):
        return 1.0
    return float(df["Volume"].iloc[week_idx] / avg_vol)


def _ceiling_from_swing_highs(
    df: pd.DataFrame,
    base_start: int,
    base_end: int,
    swing_high_indices: list[int],
) -> Optional[float]:
    """Highest swing high level within ``[base_start, base_end)``."""
    in_base = [idx for idx in swing_high_indices if base_start <= idx < base_end]
    if not in_base:
        return None  # no swing highs found in base
    return float(max(df["High"].iloc[idx] for idx in in_base))


def detect_breakouts(
    df: pd.DataFrame,
    stages_series: pd.Series,
    base_min_weeks: int,
    ceiling_def: str,
    *,
    swing_high_indices: Optional[list[int]] = None,
    ma_series: Optional[pd.Series] = None,
    atr_series: Optional[pd.Series] = None,
) -> list[dict]:
    """Find every confirmed base-to-breakout transition.

    A breakout occurs when a stock transitions INTO ``"stage2"`` from a
    non-Stage-2 state (``"stage1"``, ``"stage3"``, or ``"not_stage2"``).

    For each transition the function:
    1. Walks backwards to find the base start (first bar in the preceding
       non-stage2 run).
    2. Checks that the base duration >= *base_min_weeks*.
    3. Computes a ceiling value per *ceiling_def*.
    4. Confirms the breakout: Close at the transition week > ceiling.

    Args:
        df: Weekly OHLCV DataFrame.
        stages_series: Per-bar stage labels from the stage classifier.
        base_min_weeks: Minimum weeks in non-stage2 before transition.
        ceiling_def: One of the six ceiling definitions.
        swing_high_indices: Integer iloc indices of swing highs (needed
            for ``swing_high_*`` ceilings).
        ma_series: MA values (needed for ``ma_plus_atr`` ceiling).
        atr_series: ATR values (needed for ``ma_plus_atr`` ceiling).

    Returns:
        List of breakout event dicts.
    """
    stages = stages_series.values
    n = len(df)
    events: list[dict] = []
    non_stage2 = {"stage1", "stage3", "not_stage2"}

    for i in range(1, n):
        # Detect transition INTO stage2
        if stages[i] != "stage2":
            continue
        if stages[i - 1] not in non_stage2:
            continue

        # Walk backwards to find base start
        base_end = i  # exclusive — first bar of stage2
        base_start = i - 1
        while base_start > 0 and stages[base_start - 1] in non_stage2:
            base_start -= 1

        base_duration = base_end - base_start
        if base_duration < base_min_weeks:
            continue

        # Compute ceiling
        ceiling = _compute_ceiling(
            df, base_start, base_end, ceiling_def,
            swing_high_indices=swing_high_indices,
            ma_series=ma_series,
            atr_series=atr_series,
        )
        if ceiling is None:
            continue

        # Confirm breakout: close > ceiling
        close_at_breakout = float(df["Close"].iloc[i])
        if close_at_breakout <= ceiling:
            continue

        events.append(
            {
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": close_at_breakout,
                "ceiling": ceiling,
                "ceiling_def": ceiling_def,
                "base_start_idx": base_start,
                "base_duration": base_duration,
                "volume_ratio": _compute_volume_ratio(df, i),
            }
        )

    return events


def _compute_ceiling(
    df: pd.DataFrame,
    base_start: int,
    base_end: int,
    ceiling_def: str,
    *,
    swing_high_indices: Optional[list[int]] = None,
    ma_series: Optional[pd.Series] = None,
    atr_series: Optional[pd.Series] = None,
) -> Optional[float]:
    """Compute ceiling level for a given base period and definition."""

    if ceiling_def.startswith("swing_high"):
        if swing_high_indices is None:
            return None
        val = _ceiling_from_swing_highs(df, base_start, base_end, swing_high_indices)
        if val is None:
            # Fallback: max High in base period
            val = float(df["High"].iloc[base_start:base_end].max())
        return val

    elif ceiling_def == "weekly_high":
        return float(df["High"].iloc[base_start:base_end].max())

    elif ceiling_def == "weekly_close":
        return float(df["Close"].iloc[base_start:base_end].max())

    elif ceiling_def == "ma_plus_atr":
        if ma_series is None or atr_series is None:
            return None
        ma_val = ma_series.iloc[base_end]   # MA at breakout week
        atr_val = atr_series.iloc[base_end]
        if pd.isna(ma_val) or pd.isna(atr_val):
            return None
        return float(ma_val + 1.0 * atr_val)

    else:
        raise ValueError(f"Unknown ceiling_def: {ceiling_def}")
