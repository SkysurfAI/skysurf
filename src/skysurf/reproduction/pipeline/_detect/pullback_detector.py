"""Stage 2 pullback entry detection for Test 4."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_pullbacks(
    df: pd.DataFrame,
    stages_series: pd.Series,
    ma_series: pd.Series,
    atr_series: pd.Series,
    minor_swing_lows: list[int],
    min_stage2_duration: int,
    depth_def: str,
    confirmation: str,
    min_gap_weeks: int = 4,
) -> list[dict]:
    """Detect pullback entries within established Stage 2 trends.

    A pullback is a temporary dip toward support within Stage 2.
    The stock must remain in Stage 2 throughout.

    Args:
        df: Weekly OHLCV DataFrame.
        stages_series: Per-bar stage labels.
        ma_series: SMA_25w series.
        atr_series: 14-week ATR series.
        minor_swing_lows: Indices of A_order5 swing lows.
        min_stage2_duration: Weeks in Stage 2 before pullback is valid.
        depth_def: How close to support counts as a pullback.
        confirmation: Bounce confirmation type.
        min_gap_weeks: Min weeks between pullback entries.

    Returns:
        List of event dicts with ``week_idx``, ``date``, ``close``,
        ``entry_type``, and pullback-specific fields.
    """
    stages = stages_series.values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    ma = ma_series.values
    atr = atr_series.values
    n = len(df)

    # Build consecutive Stage 2 counter
    consec = np.zeros(n, dtype=int)
    for i in range(n):
        if stages[i] == "stage2":
            consec[i] = (consec[i - 1] + 1) if i > 0 else 1

    events: list[dict] = []
    last_entry_idx = -min_gap_weeks - 1

    for i in range(1, n):
        if consec[i] < min_stage2_duration:
            continue
        if i - last_entry_idx < min_gap_weeks:
            continue
        if pd.isna(ma[i]) or pd.isna(atr[i]):
            continue

        close_i = float(closes[i])
        ma_i = float(ma[i])
        atr_i = float(atr[i])

        # ── Depth check ─────────────────────────────────────────
        in_zone = False
        if depth_def == "pb_ma_3pct":
            in_zone = ma_i <= close_i <= ma_i * 1.03
        elif depth_def == "pb_ma_5pct":
            in_zone = ma_i <= close_i <= ma_i * 1.05
        elif depth_def == "pb_ma_atr":
            in_zone = ma_i <= close_i <= ma_i + 1.0 * atr_i
        elif depth_def == "pb_ma_1.5atr":
            in_zone = ma_i <= close_i <= ma_i + 1.5 * atr_i
        elif depth_def == "pb_swing_low":
            # Find most recent minor swing low below close
            nearest_low = None
            for j in reversed(minor_swing_lows):
                if j < i and float(lows[j]) < close_i:
                    nearest_low = float(lows[j])
                    break
            if nearest_low is not None:
                in_zone = abs(close_i - nearest_low) <= 1.0 * atr_i

        if not in_zone:
            continue

        # ── Confirmation ────────────────────────────────────────
        confirmed = False
        if confirmation == "no_confirm":
            confirmed = True
        elif confirmation == "close_up":
            confirmed = close_i > float(closes[i - 1])
        elif confirmation == "close_above_ma":
            confirmed = close_i > ma_i

        if not confirmed:
            continue

        # ── Record entry ────────────────────────────────────────
        events.append({
            "week_idx": i,
            "date": str(df.index[i].date()),
            "close": round(close_i, 2),
            "entry_type": "PULLBACK_S2",
            "depth_def": depth_def,
            "confirmation": confirmation,
            "consecutive_s2_weeks": int(consec[i]),
            "ma_value": round(ma_i, 2),
            "atr_value": round(atr_i, 2),
        })
        last_entry_idx = i

    return events
