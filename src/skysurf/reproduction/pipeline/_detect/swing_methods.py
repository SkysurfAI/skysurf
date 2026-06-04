"""Swing detection methods for Test 1 comparison.

Each public method takes a weekly OHLCV DataFrame and returns
(swing_high_indices, swing_low_indices) — lists of integer indices
into the DataFrame, strictly alternating (H-L-H-L or L-H-L-H).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .._calc.technical_indicators import calculate_atr


# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════


def _alternating_filter(
    high_indices: list[int],
    low_indices: list[int],
    high_values: np.ndarray,
    low_values: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Enforce strict H-L-H-L alternation.

    Merges highs and lows chronologically.  When consecutive same-type
    swings appear, keeps the most extreme one (highest high / lowest low).

    Applied universally to ALL methods as post-processing.
    """
    if not high_indices and not low_indices:
        return [], []

    # Build tagged list: (index, type, value)
    events: list[tuple[int, str, float]] = []
    for idx in high_indices:
        events.append((idx, "H", float(high_values[idx])))
    for idx in low_indices:
        events.append((idx, "L", float(low_values[idx])))
    events.sort(key=lambda x: x[0])

    if not events:
        return [], []

    # Merge consecutive same-type: keep highest H or lowest L
    merged: list[tuple[int, str, float]] = [events[0]]
    for evt in events[1:]:
        if evt[1] == merged[-1][1]:
            # Same type — keep the more extreme one
            if evt[1] == "H" and evt[2] > merged[-1][2]:
                merged[-1] = evt
            elif evt[1] == "L" and evt[2] < merged[-1][2]:
                merged[-1] = evt
        else:
            merged.append(evt)

    out_highs = [e[0] for e in merged if e[1] == "H"]
    out_lows = [e[0] for e in merged if e[1] == "L"]
    return out_highs, out_lows


def _five_bar_pivot_lows(lows: np.ndarray, min_spacing: int = 2) -> list[int]:
    """5-bar pivot swing low detection (skysurf algorithm, no [-5:] limit).

    low[i] < low[i-1] AND low[i] < low[i-2] AND
    low[i] < low[i+1] AND low[i] < low[i+2]
    """
    if len(lows) < 5:
        return []
    indices: list[int] = []
    for i in range(2, len(lows) - 2):
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            if not indices or (i - indices[-1]) >= min_spacing:
                indices.append(i)
    return indices


def _five_bar_pivot_highs(highs: np.ndarray, min_spacing: int = 2) -> list[int]:
    """5-bar pivot swing high detection (mirror of lows)."""
    if len(highs) < 5:
        return []
    indices: list[int] = []
    for i in range(2, len(highs) - 2):
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            if not indices or (i - indices[-1]) >= min_spacing:
                indices.append(i)
    return indices


# ═══════════════════════════════════════════════════════════════════════
# Method A: argrelextrema
# ═══════════════════════════════════════════════════════════════════════


def method_a_argrelextrema(
    df: pd.DataFrame, order: int = 5
) -> tuple[list[int], list[int]]:
    """scipy argrelextrema on High/Low, then alternating filter."""
    highs = df["High"].values
    lows = df["Low"].values

    raw_high_idx = list(argrelextrema(highs, np.greater, order=order)[0])
    raw_low_idx = list(argrelextrema(lows, np.less, order=order)[0])

    return _alternating_filter(raw_high_idx, raw_low_idx, highs, lows)


# ═══════════════════════════════════════════════════════════════════════
# Method B: Percentage reversal
# ═══════════════════════════════════════════════════════════════════════


def method_b_pct_reversal(
    df: pd.DataFrame, pct_threshold: float = 0.05
) -> tuple[list[int], list[int]]:
    """Track running peak/trough, confirm swing on X% reversal."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    if n < 3:
        return [], []

    raw_high_idx: list[int] = []
    raw_low_idx: list[int] = []

    # State: track running peak and trough
    peak_idx, peak_val = 0, highs[0]
    trough_idx, trough_val = 0, lows[0]
    seeking = "either"  # "high", "low", or "either" at start

    for i in range(1, n):
        # Update running peak
        if highs[i] > peak_val:
            peak_idx, peak_val = i, highs[i]
        # Update running trough
        if lows[i] < trough_val:
            trough_idx, trough_val = i, lows[i]

        # Check for swing high confirmation (price dropped enough from peak)
        if peak_val > 0 and (peak_val - closes[i]) / peak_val >= pct_threshold:
            if seeking in ("high", "either"):
                raw_high_idx.append(peak_idx)
                # Reset: start tracking from this point
                trough_idx, trough_val = i, lows[i]
                peak_idx, peak_val = i, highs[i]
                seeking = "low"

        # Check for swing low confirmation (price rose enough from trough)
        if trough_val > 0 and (closes[i] - trough_val) / trough_val >= pct_threshold:
            if seeking in ("low", "either"):
                raw_low_idx.append(trough_idx)
                # Reset
                peak_idx, peak_val = i, highs[i]
                trough_idx, trough_val = i, lows[i]
                seeking = "high"

    return _alternating_filter(raw_high_idx, raw_low_idx, highs, lows)


# ═══════════════════════════════════════════════════════════════════════
# Method C: ATR reversal
# ═══════════════════════════════════════════════════════════════════════


def method_c_atr_reversal(
    df: pd.DataFrame, atr_multiplier: float = 1.5
) -> tuple[list[int], list[int]]:
    """Same as Method B but threshold = N * 14-week ATR (adaptive)."""
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    if n < 15:
        return [], []

    atr_series = calculate_atr(df["High"], df["Low"], df["Close"], window=14)
    atr_vals = atr_series.values

    raw_high_idx: list[int] = []
    raw_low_idx: list[int] = []

    peak_idx, peak_val = 0, highs[0]
    trough_idx, trough_val = 0, lows[0]
    seeking = "either"

    for i in range(1, n):
        if highs[i] > peak_val:
            peak_idx, peak_val = i, highs[i]
        if lows[i] < trough_val:
            trough_idx, trough_val = i, lows[i]

        atr_val = atr_vals[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        threshold = atr_multiplier * atr_val

        # Swing high: price dropped threshold from peak
        if (peak_val - closes[i]) >= threshold:
            if seeking in ("high", "either"):
                raw_high_idx.append(peak_idx)
                trough_idx, trough_val = i, lows[i]
                peak_idx, peak_val = i, highs[i]
                seeking = "low"

        # Swing low: price rose threshold from trough
        if (closes[i] - trough_val) >= threshold:
            if seeking in ("low", "either"):
                raw_low_idx.append(trough_idx)
                peak_idx, peak_val = i, highs[i]
                trough_idx, trough_val = i, lows[i]
                seeking = "high"

    return _alternating_filter(raw_high_idx, raw_low_idx, highs, lows)


# ═══════════════════════════════════════════════════════════════════════
# Method D: argrelextrema + ATR minimum filter
# ═══════════════════════════════════════════════════════════════════════


def method_d_argrelextrema_atr(
    df: pd.DataFrame, order: int = 5, atr_multiplier: float = 1.0
) -> tuple[list[int], list[int]]:
    """argrelextrema candidates, then discard moves < M * ATR."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    if n < 15:
        return [], []

    atr_series = calculate_atr(df["High"], df["Low"], df["Close"], window=14)
    atr_vals = atr_series.values

    # Stage 1: raw candidates
    raw_high_idx = list(argrelextrema(highs, np.greater, order=order)[0])
    raw_low_idx = list(argrelextrema(lows, np.less, order=order)[0])

    # Stage 2: alternating filter first to get pairs
    alt_highs, alt_lows = _alternating_filter(
        raw_high_idx, raw_low_idx, highs, lows
    )

    # Stage 3: ATR magnitude filter on alternated pairs
    # Merge into chronological events
    events: list[tuple[int, str]] = []
    for idx in alt_highs:
        events.append((idx, "H"))
    for idx in alt_lows:
        events.append((idx, "L"))
    events.sort(key=lambda x: x[0])

    if len(events) < 2:
        return alt_highs, alt_lows

    # Filter: keep pairs where move exceeds ATR threshold
    kept: list[tuple[int, str]] = [events[0]]
    for i in range(1, len(events)):
        prev_idx, prev_type = kept[-1]
        curr_idx, curr_type = events[i]

        if prev_type == "H" and curr_type == "L":
            move = highs[prev_idx] - lows[curr_idx]
        elif prev_type == "L" and curr_type == "H":
            move = highs[curr_idx] - lows[prev_idx]
        else:
            move = float("inf")  # same type — should not happen after alternation

        atr_at = atr_vals[prev_idx]
        if np.isnan(atr_at) or atr_at <= 0:
            kept.append(events[i])
            continue

        if move >= atr_multiplier * atr_at:
            kept.append(events[i])
        # else: discard this swing (too small)

    # Re-alternate after filtering (filtering may break alternation)
    filt_highs = [e[0] for e in kept if e[1] == "H"]
    filt_lows = [e[0] for e in kept if e[1] == "L"]
    return _alternating_filter(filt_highs, filt_lows, highs, lows)


# ═══════════════════════════════════════════════════════════════════════
# Method E: Skysurf baseline (5-bar pivot)
# ═══════════════════════════════════════════════════════════════════════


def method_e_skysurf_baseline(
    df: pd.DataFrame, use_atr_filter: bool = False
) -> tuple[list[int], list[int]]:
    """5-bar pivot (skysurf algorithm) without the [-5:] truncation.

    Config 1: use_atr_filter=False — raw 5-bar pivot with min_spacing=2
    Config 2: use_atr_filter=True — add ATR significance filter
    """
    highs = df["High"].values
    lows = df["Low"].values

    raw_low_idx = _five_bar_pivot_lows(lows, min_spacing=2)
    raw_high_idx = _five_bar_pivot_highs(highs, min_spacing=2)

    if use_atr_filter and len(df) >= 15:
        atr_series = calculate_atr(df["High"], df["Low"], df["Close"], window=14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else None

        if atr_val is not None and atr_val > 0 and not np.isnan(atr_val):
            # Significance filter for lows: drop from preceding high >= 1*ATR
            filtered_lows: list[int] = []
            for idx in raw_low_idx:
                if idx > 0:
                    preceding_high = float(highs[:idx].max())
                    if (preceding_high - lows[idx]) >= atr_val:
                        filtered_lows.append(idx)
            raw_low_idx = filtered_lows

            # Significance filter for highs: rise from preceding low >= 1*ATR
            filtered_highs: list[int] = []
            for idx in raw_high_idx:
                if idx > 0:
                    preceding_low = float(lows[:idx].min())
                    if (highs[idx] - preceding_low) >= atr_val:
                        filtered_highs.append(idx)
            raw_high_idx = filtered_highs

    return _alternating_filter(raw_high_idx, raw_low_idx, highs, lows)


# ═══════════════════════════════════════════════════════════════════════
# Configuration registry
# ═══════════════════════════════════════════════════════════════════════


def get_all_configs() -> list[dict]:
    """Return all 20 configurations to test."""
    configs: list[dict] = []

    # Method A: argrelextrema (4 configs)
    for order in [3, 5, 8, 10]:
        configs.append({
            "method": "A",
            "label": f"A_order{order}",
            "func": method_a_argrelextrema,
            "kwargs": {"order": order},
        })

    # Method B: percentage reversal (4 configs)
    for pct in [0.03, 0.05, 0.08, 0.10]:
        configs.append({
            "method": "B",
            "label": f"B_pct{int(pct*100)}",
            "func": method_b_pct_reversal,
            "kwargs": {"pct_threshold": pct},
        })

    # Method C: ATR reversal (4 configs)
    for mult in [1.0, 1.5, 2.0, 2.5]:
        configs.append({
            "method": "C",
            "label": f"C_atr{mult}",
            "func": method_c_atr_reversal,
            "kwargs": {"atr_multiplier": mult},
        })

    # Method D: argrelextrema + ATR filter (6 configs)
    for order in [3, 5]:
        for mult in [0.5, 1.0, 1.5]:
            configs.append({
                "method": "D",
                "label": f"D_o{order}_a{mult}",
                "func": method_d_argrelextrema_atr,
                "kwargs": {"order": order, "atr_multiplier": mult},
            })

    # Method E: Skysurf baseline (2 configs)
    configs.append({
        "method": "E",
        "label": "E_base",
        "func": method_e_skysurf_baseline,
        "kwargs": {"use_atr_filter": False},
    })
    configs.append({
        "method": "E",
        "label": "E_atr",
        "func": method_e_skysurf_baseline,
        "kwargs": {"use_atr_filter": True},
    })

    return configs
