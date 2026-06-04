"""Weinstein 4-stage classification for Test 2."""
from __future__ import annotations

import pandas as pd


def classify_stages_single_ma(
    df: pd.DataFrame,
    ma_series: pd.Series,
    direction_series: pd.Series,
) -> pd.Series:
    """Classify each weekly bar into one of four Weinstein stages.

    Stage rules (evaluated in order):
        - **Stage 2 (Advance)**: Close > MA AND direction == "rising"
        - **Stage 4 (Decline)**: Close < MA AND direction == "falling"
        - **Stage 3 (Top/Distribution)**: direction == "flat" AND the bar
          was in Stage 2 within the last 8 bars
        - **Stage 1 (Base/Accumulation)**: direction == "flat" AND NOT
          Stage 3
        - **Ambiguous**: anything else — carry forward the previous stage

    Stages are NOT forced sequential.  A stock can cycle
    stage1 → stage2 → stage1 → stage2 (failed breakout → re-base →
    successful breakout).

    Args:
        df: Weekly OHLCV DataFrame (needs ``Close`` column).
        ma_series: Pre-computed MA values, aligned with ``df.index``.
        direction_series: Per-bar slope direction (``"rising"`` /
            ``"falling"`` / ``"flat"``), from ``compute_ma_slope_direction``.

    Returns:
        pd.Series of stage labels (``"stage1"`` / ``"stage2"`` /
        ``"stage3"`` / ``"stage4"``), index-aligned with *df*.
    """
    n = len(df)
    stages = ["stage1"] * n
    close = df["Close"].values
    ma = ma_series.values
    dirs = direction_series.values

    for i in range(n):
        # Skip bars where MA isn't yet computed (NaN)
        if pd.isna(ma[i]):
            stages[i] = "stage1"
            continue

        d = dirs[i]

        if close[i] > ma[i] and d == "rising":
            stages[i] = "stage2"
        elif close[i] < ma[i] and d == "falling":
            stages[i] = "stage4"
        elif d == "flat":
            # Check if stage2 existed within last 8 bars
            lookback_start = max(0, i - 8)
            was_stage2 = any(stages[j] == "stage2" for j in range(lookback_start, i))
            stages[i] = "stage3" if was_stage2 else "stage1"
        else:
            # Ambiguous (e.g. close > MA but falling, or close < MA but
            # rising) — carry forward the previous stage
            stages[i] = stages[i - 1] if i > 0 else "stage1"

    return pd.Series(stages, index=df.index, dtype=object)


def classify_stages_trend_template(
    df: pd.DataFrame,
    fast_series: pd.Series,
    mid_series: pd.Series,
    slow_series: pd.Series,
    slow_direction_series: pd.Series,
) -> pd.Series:
    """Binary Stage 2 classification using a Trend Template cascade.

    Stage 2 requires ALL of:
        - Close > fast MA > mid MA > slow MA
        - slow MA direction == "rising"

    Everything else is ``"not_stage2"`` (treated as base-eligible for
    breakout detection).

    Args:
        df: Weekly OHLCV DataFrame.
        fast_series: Fast MA values.
        mid_series: Mid MA values.
        slow_series: Slow MA values.
        slow_direction_series: Per-bar slope direction for the slow MA.

    Returns:
        pd.Series of ``"stage2"`` / ``"not_stage2"``.
    """
    close = df["Close"]
    cascade = (
        (close > fast_series)
        & (fast_series > mid_series)
        & (mid_series > slow_series)
        & (slow_direction_series == "rising")
    )
    return cascade.map({True: "stage2", False: "not_stage2"}).fillna("not_stage2")
