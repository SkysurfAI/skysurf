"""Quality metrics computation for all entry types."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_entry_quality(
    df: pd.DataFrame,
    event: dict,
    major_swing_highs: list[int],
    minor_swing_lows: list[int],
    atr_series: pd.Series,
    base_start_idx: int | None = None,
) -> dict:
    """Enrich an entry event with volume ratio, stop, target, and R/R.

    Modifies *event* in-place and returns it.

    Args:
        df: Weekly OHLCV DataFrame.
        event: Entry event dict (must have ``week_idx``, ``close``, ``entry_type``).
        major_swing_highs: A_order8 swing high indices (for target).
        minor_swing_lows: A_order5 swing low indices (for pullback stop).
        atr_series: 14-week ATR series.
        base_start_idx: Start of base period (for breakout stop). None for
            pullback/VCP (they compute stops differently).
    """
    idx = event["week_idx"]
    close = event["close"]
    entry_type = event["entry_type"]
    lows = df["Low"].values
    highs = df["High"].values
    volumes = df["Volume"].values

    # ── Volume ratio ────────────────────────────────────────────
    start = max(0, idx - 20)
    trailing_vol = volumes[start:idx]
    avg_vol = float(np.mean(trailing_vol)) if len(trailing_vol) > 0 else 1.0
    event["volume_ratio"] = round(float(volumes[idx]) / avg_vol, 2) if avg_vol > 0 else 0.0

    # ── Stop level ──────────────────────────────────────────────
    atr_val = float(atr_series.iloc[idx]) if not pd.isna(atr_series.iloc[idx]) else 0.0

    if entry_type == "BREAKOUT_S1S2" and base_start_idx is not None:
        # Floor of the base period
        event["stop_level"] = round(float(np.min(lows[base_start_idx:idx])), 2)
    elif entry_type == "PULLBACK_S2":
        # Most recent minor swing low below entry
        stop = None
        for j in reversed(minor_swing_lows):
            if j < idx and float(lows[j]) < close:
                stop = float(lows[j])
                break
        event["stop_level"] = round(stop, 2) if stop is not None else round(close - 1.5 * atr_val, 2)
    elif entry_type == "VCP_CONTINUATION":
        # Lowest low during consolidation
        consol_low = event.get("consolidation_low")
        event["stop_level"] = round(float(consol_low), 2) if consol_low is not None else round(close - 1.5 * atr_val, 2)
    else:
        event["stop_level"] = round(close - 1.5 * atr_val, 2)

    # ── Target level ────────────────────────────────────────────
    target = None
    for j in major_swing_highs:
        if j > idx and float(highs[j]) > close:
            target = float(highs[j])
            break
    if target is None:
        target = close * 1.20  # 20% above entry if no swing high found
    event["target_level"] = round(target, 2)

    # ── R/R ratio ───────────────────────────────────────────────
    risk = close - event["stop_level"]
    reward = event["target_level"] - close
    event["rr_ratio"] = round(reward / risk, 2) if risk > 0 else 0.0

    return event
