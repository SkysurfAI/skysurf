"""Swing high / swing low detection.

Locates significant pivots in a price series — points that are local
maxima of ``high`` (swing highs) or local minima of ``low`` (swing
lows). PHASE_4_V1 uses :func:`detect_swings_argrelextrema` (scipy's
:func:`scipy.signal.argrelextrema`) followed by an alternating filter
that enforces strict H-L-H-L ordering.

Other swing-detection methods (percentage reversal, ATR reversal,
five-bar pivot) exist in the research codebase but are not used by
:data:`PHASE_4_V1`, so they are not vendored here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def detect_swings_argrelextrema(df: pd.DataFrame, order: int = 5) -> tuple[list[int], list[int]]:
    """Detect swing highs and lows via :func:`scipy.signal.argrelextrema`.

    Args:
        df: DataFrame with at least ``High`` and ``Low`` columns,
            indexed by bar (typically weekly).
        order: Window order — a bar at index ``i`` qualifies as a swing
            high (low) only if it is strictly greater (less) than the
            ``order`` bars on either side. Larger ``order`` means fewer,
            more significant swings.

    Returns:
        Tuple ``(swing_high_indices, swing_low_indices)`` — strictly
        alternating positional indices (``H-L-H-L`` or ``L-H-L-H``).
        When raw extrema overlap, the most extreme of each consecutive
        same-type run is kept.
    """
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    raw_high_idx: list[int] = list(argrelextrema(highs, np.greater, order=order)[0])
    raw_low_idx: list[int] = list(argrelextrema(lows, np.less, order=order)[0])

    return _alternating_filter(raw_high_idx, raw_low_idx, highs, lows)


def _alternating_filter(
    high_indices: list[int],
    low_indices: list[int],
    high_values: np.ndarray,
    low_values: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Enforce strict H-L-H-L alternation between two index lists.

    Merges highs and lows chronologically. When consecutive same-type
    swings appear, keeps the most extreme one (highest high / lowest
    low). Applied universally as post-processing on every detection
    method.
    """
    if not high_indices and not low_indices:
        return [], []

    events: list[tuple[int, str, float]] = []
    for idx in high_indices:
        events.append((idx, "H", float(high_values[idx])))
    for idx in low_indices:
        events.append((idx, "L", float(low_values[idx])))
    events.sort(key=lambda x: x[0])

    if not events:
        return [], []

    merged: list[tuple[int, str, float]] = [events[0]]
    for evt in events[1:]:
        if evt[1] == merged[-1][1]:
            # Same type — keep the more extreme one.
            if (evt[1] == "H" and evt[2] > merged[-1][2]) or (
                evt[1] == "L" and evt[2] < merged[-1][2]
            ):
                merged[-1] = evt
        else:
            merged.append(evt)

    out_highs = [e[0] for e in merged if e[1] == "H"]
    out_lows = [e[0] for e in merged if e[1] == "L"]
    return out_highs, out_lows
