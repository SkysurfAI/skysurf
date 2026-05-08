"""The five entry detectors used by PHASE_4_V1.

Each detector takes a stock's weekly OHLCV plus per-bar indicator series
(stages, MA, ATR, swings) and returns a list of "event" dicts — one per
bar where the entry pattern fires. Events carry detector-specific fields
that downstream stages (sizing, ranking) consume.

The five detectors:

* :func:`detect_breakouts` — base-to-Stage-2 transition where the
  breakout week's close clears the base ceiling.
* :func:`detect_pullbacks` — pullback to support within an established
  Stage 2 trend, confirmed by a close above the MA.
* :func:`detect_vcp_continuations` — Volatility Contraction Pattern: a
  consolidation within Stage 2 with declining volume, broken upward.
* :func:`detect_retest_support` — broken swing high, retested as
  support and bounced off.
* :func:`detect_trendline_bounce` — bounce off a rising support
  trendline (requires the optional ``trendln`` package; gracefully
  returns ``[]`` if not installed).

After firing, every event is enriched by :func:`compute_entry_quality`
with stop level, target level, and risk/reward ratio.

Two historic detectors (``ATH_BREAKOUT`` and ``PULLBACK_STRUCTURAL``)
exist in the research codebase but are explicitly suspended in
PHASE_4_V1 (see ``SUSPENDED_ENTRY_TYPES``); they are not vendored here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

#: Stage labels that are NOT Stage 2 — used to detect base-to-breakout
#: transitions.
_NON_STAGE2: frozenset[str] = frozenset({"stage1", "stage3", "not_stage2"})


# ── Breakout detector ────────────────────────────────────────────────


def detect_breakouts(
    df: pd.DataFrame,
    stages_series: pd.Series,
    base_min_weeks: int,
    ceiling_def: str,
    *,
    swing_high_indices: list[int] | None = None,
    ma_series: pd.Series | None = None,
    atr_series: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Find every confirmed base-to-breakout transition.

    A breakout fires when a stock transitions *into* Stage 2 from a
    non-Stage-2 state (Stage 1, Stage 3, or unclassified). The function:

    1. Walks backwards to find the base start (first bar in the
       preceding non-Stage-2 run).
    2. Checks the base duration is at least ``base_min_weeks``.
    3. Computes a ceiling per ``ceiling_def``.
    4. Confirms the close at the transition week clears the ceiling.

    Args:
        df: Weekly OHLCV DataFrame.
        stages_series: Per-bar stage labels.
        base_min_weeks: Minimum bars in non-Stage-2 before transition.
        ceiling_def: One of ``"swing_high"``, ``"weekly_high"``,
            ``"weekly_close"``, or ``"ma_plus_atr"``. PHASE_4_V1 uses
            ``"ma_plus_atr"`` (MA + 1 × ATR).
        swing_high_indices: Required when ``ceiling_def`` is a
            ``swing_high`` variant.
        ma_series: Required when ``ceiling_def == "ma_plus_atr"``.
        atr_series: Required when ``ceiling_def == "ma_plus_atr"``.

    Returns:
        List of breakout-event dicts.
    """
    stages = stages_series.to_numpy()
    n = len(df)
    events: list[dict[str, Any]] = []

    for i in range(1, n):
        if stages[i] != "stage2" or stages[i - 1] not in _NON_STAGE2:
            continue

        # Walk back to find base start.
        base_end = i  # exclusive
        base_start = i - 1
        while base_start > 0 and stages[base_start - 1] in _NON_STAGE2:
            base_start -= 1

        base_duration = base_end - base_start
        if base_duration < base_min_weeks:
            continue

        ceiling = _compute_ceiling(
            df,
            base_start,
            base_end,
            ceiling_def,
            swing_high_indices=swing_high_indices,
            ma_series=ma_series,
            atr_series=atr_series,
        )
        if ceiling is None:
            continue

        close_at_breakout = float(df["Close"].iloc[i])
        if close_at_breakout <= ceiling:
            continue

        events.append(
            {
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": close_at_breakout,
                "entry_type": "BREAKOUT_S1S2",
                "ceiling": ceiling,
                "ceiling_def": ceiling_def,
                "base_start_idx": base_start,
                "base_duration": base_duration,
                "volume_ratio": _volume_ratio(df, i),
            }
        )

    return events


def _compute_ceiling(
    df: pd.DataFrame,
    base_start: int,
    base_end: int,
    ceiling_def: str,
    *,
    swing_high_indices: list[int] | None = None,
    ma_series: pd.Series | None = None,
    atr_series: pd.Series | None = None,
) -> float | None:
    """Compute the ceiling level for a given base period and definition."""
    if ceiling_def.startswith("swing_high"):
        if swing_high_indices is None:
            return None
        in_base = [idx for idx in swing_high_indices if base_start <= idx < base_end]
        if in_base:
            return float(max(df["High"].iloc[idx] for idx in in_base))
        # Fallback: max High in base period.
        return float(df["High"].iloc[base_start:base_end].max())

    if ceiling_def == "weekly_high":
        return float(df["High"].iloc[base_start:base_end].max())

    if ceiling_def == "weekly_close":
        return float(df["Close"].iloc[base_start:base_end].max())

    if ceiling_def == "ma_plus_atr":
        if ma_series is None or atr_series is None:
            return None
        ma_val = ma_series.iloc[base_end]
        atr_val = atr_series.iloc[base_end]
        if pd.isna(ma_val) or pd.isna(atr_val):
            return None
        return float(ma_val + 1.0 * atr_val)

    raise ValueError(f"Unknown ceiling_def: {ceiling_def!r}")


def _volume_ratio(df: pd.DataFrame, week_idx: int, lookback: int = 20) -> float:
    """Ratio of week ``week_idx`` volume to trailing-``lookback`` mean."""
    start = max(0, week_idx - lookback)
    avg_vol = df["Volume"].iloc[start:week_idx].mean()
    if avg_vol is None or avg_vol == 0 or np.isnan(avg_vol):
        return 1.0
    return float(df["Volume"].iloc[week_idx] / avg_vol)


# ── Pullback detector ────────────────────────────────────────────────


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
) -> list[dict[str, Any]]:
    """Detect pullback entries within established Stage 2 trends.

    A pullback is a temporary dip toward support within Stage 2; the
    stock must remain in Stage 2 throughout. PHASE_4_V1 uses
    ``depth_def="pb_swing_low"`` (most-recent minor swing low) and
    ``confirmation="close_above_ma"``.
    """
    stages = stages_series.to_numpy()
    closes = df["Close"].to_numpy()
    lows = df["Low"].to_numpy()
    ma = ma_series.to_numpy()
    atr = atr_series.to_numpy()
    n = len(df)

    consec = np.zeros(n, dtype=int)
    for i in range(n):
        if stages[i] == "stage2":
            consec[i] = (consec[i - 1] + 1) if i > 0 else 1

    events: list[dict[str, Any]] = []
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

        in_zone = _pullback_in_zone(depth_def, close_i, ma_i, atr_i, lows, minor_swing_lows, i)
        if not in_zone:
            continue

        if not _pullback_confirmed(confirmation, close_i, closes, ma_i, i):
            continue

        events.append(
            {
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": round(close_i, 2),
                "entry_type": "PULLBACK_S2",
                "depth_def": depth_def,
                "confirmation": confirmation,
                "consecutive_s2_weeks": int(consec[i]),
                "ma_value": round(ma_i, 2),
                "atr_value": round(atr_i, 2),
            }
        )
        last_entry_idx = i

    return events


def _pullback_in_zone(
    depth_def: str,
    close_i: float,
    ma_i: float,
    atr_i: float,
    lows: np.ndarray,
    minor_swing_lows: list[int],
    i: int,
) -> bool:
    if depth_def == "pb_ma_3pct":
        return ma_i <= close_i <= ma_i * 1.03
    if depth_def == "pb_ma_5pct":
        return ma_i <= close_i <= ma_i * 1.05
    if depth_def == "pb_ma_atr":
        return ma_i <= close_i <= ma_i + 1.0 * atr_i
    if depth_def == "pb_ma_1.5atr":
        return ma_i <= close_i <= ma_i + 1.5 * atr_i
    if depth_def == "pb_swing_low":
        for j in reversed(minor_swing_lows):
            if j < i and float(lows[j]) < close_i:
                nearest_low = float(lows[j])
                return abs(close_i - nearest_low) <= 1.0 * atr_i
        return False
    raise ValueError(f"Unknown pullback depth_def: {depth_def!r}")


def _pullback_confirmed(
    confirmation: str,
    close_i: float,
    closes: np.ndarray,
    ma_i: float,
    i: int,
) -> bool:
    if confirmation == "no_confirm":
        return True
    if confirmation == "close_up":
        return close_i > float(closes[i - 1])
    if confirmation == "close_above_ma":
        return close_i > ma_i
    raise ValueError(f"Unknown pullback confirmation: {confirmation!r}")


# ── VCP continuation detector ─────────────────────────────────────────


def detect_vcp_continuations(
    df: pd.DataFrame,
    stages_series: pd.Series,
    atr_series: pd.Series,
    minor_swing_highs: list[int],
    minor_swing_lows: list[int],
    consol_min_weeks: int,
    contraction_req: str,
    volume_req: str,
    min_stage2_duration: int = 8,
    min_gap_weeks: int = 4,
) -> list[dict[str, Any]]:
    """Detect VCP continuation entries within Stage 2 trends.

    A VCP (Volatility Contraction Pattern) is a consolidation within
    Stage 2 that shows contracting price ranges and (optionally)
    declining volume, broken upward when close clears the consolidation
    ceiling. PHASE_4_V1 uses ``contraction_req="vcp_any"`` and
    ``volume_req="vol_declining"``.
    """
    stages = stages_series.to_numpy()
    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    volumes = df["Volume"].to_numpy()
    atr = atr_series.to_numpy()
    n = len(df)

    consec = np.zeros(n, dtype=int)
    s2_peak = np.full(n, np.nan)
    for i in range(n):
        if stages[i] == "stage2":
            consec[i] = (consec[i - 1] + 1) if i > 0 else 1
            if i == 0 or consec[i] == 1:
                s2_peak[i] = float(highs[i])
            else:
                s2_peak[i] = max(s2_peak[i - 1], float(highs[i]))
        else:
            consec[i] = 0

    events: list[dict[str, Any]] = []
    last_entry_idx = -min_gap_weeks - 1
    i = 0

    while i < n:
        if consec[i] < min_stage2_duration or pd.isna(s2_peak[i]):
            i += 1
            continue

        peak = s2_peak[i]
        if float(highs[i]) >= peak:
            i += 1
            continue

        # Walk forward through consolidation (Stage 2 below the peak).
        consol_start = i
        consol_end = i
        while consol_end < n and stages[consol_end] == "stage2":
            if float(highs[consol_end]) >= peak:
                break
            consol_end += 1

        consol_len = consol_end - consol_start
        if consol_len < consol_min_weeks:
            i = consol_end
            continue

        if consol_end >= n or stages[consol_end] != "stage2":
            i = consol_end
            continue
        if float(highs[consol_end]) < peak:
            i = consol_end
            continue

        breakout_idx = consol_end
        breakout_close = float(closes[breakout_idx])
        consol_ceiling = float(np.max(highs[consol_start:consol_end]))
        consol_low = float(np.min(lows[consol_start:consol_end]))

        if breakout_close <= consol_ceiling:
            i = breakout_idx + 1
            continue

        # Contraction check.
        sh_in = [j for j in minor_swing_highs if consol_start <= j < consol_end]
        sl_in = [j for j in minor_swing_lows if consol_start <= j < consol_end]
        pairs = _contraction_pairs(sh_in, sl_in)
        depths = (
            [(float(highs[h]) - float(lows[low_idx])) / float(highs[h]) for h, low_idx in pairs]
            if pairs
            else []
        )
        if not _contraction_passes(contraction_req, depths, df, breakout_idx, atr):
            i = breakout_idx + 1
            continue

        # Volume check.
        consol_vols = volumes[consol_start:consol_end].astype(float)
        breakout_vol = float(volumes[breakout_idx])
        vol_slope = _volume_slope(consol_vols)
        consol_avg_vol = float(np.mean(consol_vols)) if len(consol_vols) > 0 else 1.0
        breakout_vol_ratio = breakout_vol / consol_avg_vol if consol_avg_vol > 0 else 0.0
        if not _volume_passes(volume_req, vol_slope, breakout_vol_ratio):
            i = breakout_idx + 1
            continue

        if breakout_idx - last_entry_idx < min_gap_weeks:
            i = breakout_idx + 1
            continue

        events.append(
            {
                "week_idx": breakout_idx,
                "date": str(df.index[breakout_idx].date()),
                "close": round(breakout_close, 2),
                "entry_type": "VCP_CONTINUATION",
                "consolidation_start_idx": consol_start,
                "consolidation_weeks": consol_len,
                "consolidation_ceiling": round(consol_ceiling, 2),
                "consolidation_low": round(consol_low, 2),
                "contraction_count": len(depths),
                "contraction_depths": [round(d, 4) for d in depths],
                "volume_slope": round(vol_slope, 4) if not np.isnan(vol_slope) else None,
                "breakout_volume_ratio": round(breakout_vol_ratio, 2),
            }
        )
        last_entry_idx = breakout_idx
        i = breakout_idx + 1

    return events


def _contraction_pairs(
    swing_high_indices: list[int],
    swing_low_indices: list[int],
) -> list[tuple[int, int]]:
    """Build chronological ``(high_idx, low_idx)`` pairs from minor swings."""
    all_swings = [(idx, "H") for idx in swing_high_indices] + [
        (idx, "L") for idx in swing_low_indices
    ]
    all_swings.sort(key=lambda x: x[0])

    pairs: list[tuple[int, int]] = []
    j = 0
    while j < len(all_swings) - 1:
        if all_swings[j][1] == "H" and all_swings[j + 1][1] == "L":
            pairs.append((all_swings[j][0], all_swings[j + 1][0]))
            j += 2
        else:
            j += 1
    return pairs


def _contraction_passes(
    req: str,
    depths: list[float],
    df: pd.DataFrame,
    breakout_idx: int,
    atr: np.ndarray,
) -> bool:
    if req == "vcp_any":
        return True
    if req == "vcp_2plus":
        return len(depths) >= 2 and all(depths[k] > depths[k + 1] for k in range(len(depths) - 1))
    if req == "vcp_tight_final":
        if len(depths) < 2 or not all(depths[k] > depths[k + 1] for k in range(len(depths) - 1)):
            return False
        final_idx = breakout_idx - 1
        if final_idx < 0 or final_idx >= len(df) or pd.isna(atr[final_idx]):
            return False
        final_range = float(df["High"].iloc[final_idx]) - float(df["Low"].iloc[final_idx])
        return final_range < 1.5 * float(atr[final_idx])
    raise ValueError(f"Unknown VCP contraction_req: {req!r}")


def _volume_passes(req: str, vol_slope: float, breakout_vol_ratio: float) -> bool:
    if req == "vol_any":
        return True
    if req == "vol_declining":
        return vol_slope < 0
    if req == "vol_dry_breakout":
        return vol_slope < 0 and breakout_vol_ratio > 1.5
    raise ValueError(f"Unknown VCP volume_req: {req!r}")


def _volume_slope(volumes: np.ndarray) -> float:
    """Linear-regression slope of a volume series."""
    if len(volumes) < 2:
        return 0.0
    x = np.arange(len(volumes), dtype=float)
    try:
        slope = float(np.polyfit(x, volumes, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return slope


# ── Retest-of-broken-resistance detector ─────────────────────────────


def detect_retest_support(
    df: pd.DataFrame,
    stages_series: pd.Series,
    major_swing_highs: list[int],
    atr_series: pd.Series,
    min_gap: int = 4,
) -> list[dict[str, Any]]:
    """Detect a retest of a broken swing high level acting as support.

    Logic: find a major swing high that price has previously closed
    above (level became support); flag the bar where price comes back
    near that level, holds above it, and is not a bearish candle.
    """
    events: list[dict[str, Any]] = []
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    opens = df["Open"].to_numpy()
    n = len(df)

    used_levels: set[float] = set()
    last_entry_idx = -min_gap - 1

    for i in range(2, n):
        if stages_series.iloc[i] != "stage2":
            continue
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr <= 0 or i - last_entry_idx < min_gap:
            continue

        close_i = closes[i]
        low_i = lows[i]
        open_i = opens[i]

        for sh_idx in major_swing_highs:
            sh_level = float(highs[sh_idx])
            if sh_level >= close_i or sh_level in used_levels:
                continue

            # First break of the level (close > level).
            break_week = next(
                (b for b in range(sh_idx + 1, i) if closes[b] > sh_level),
                None,
            )
            if break_week is None or break_week >= i - 1:
                continue

            # Price pulling back toward the level.
            if low_i > sh_level * (1 + 1.5 * atr / sh_level):
                continue
            if close_i <= sh_level:
                continue
            if close_i <= open_i and close_i <= sh_level * 1.005:
                # Bearish candle at the level — no bounce.
                continue

            post_break_high = float(np.max(highs[break_week:i]))
            pullback_depth_pct = (
                round((post_break_high - low_i) / post_break_high * 100, 2)
                if post_break_high > 0
                else 0
            )
            vol_start = max(0, i - 20)
            trailing_vol = df["Volume"].to_numpy()[vol_start:i]
            avg_vol = float(np.mean(trailing_vol)) if len(trailing_vol) > 0 else 1.0
            bounce_vol = round(float(df["Volume"].to_numpy()[i]) / avg_vol, 2) if avg_vol > 0 else 0

            used_levels.add(sh_level)
            last_entry_idx = i
            events.append(
                {
                    "week_idx": i,
                    "date": str(df.index[i].date()),
                    "close": round(float(close_i), 2),
                    "entry_type": "RETEST_SUPPORT",
                    "retest_level": round(sh_level, 2),
                    "weeks_since_breakout": i - break_week,
                    "pullback_depth_pct": pullback_depth_pct,
                    "bounce_volume_ratio": bounce_vol,
                    "stop_atr_level": round(sh_level - 1.0 * atr, 2),
                }
            )
            break  # one entry per week

    return events


# ── Trendline-bounce detector ─────────────────────────────────────────


def precompute_trendlines(
    df: pd.DataFrame,
    atr_series: pd.Series,
    swing_lows_data: list[dict[str, Any]] | None,
    interval: int = 26,
) -> list[dict[str, Any] | None]:
    """Precompute trendline detection at fixed intervals.

    Returns a list of length ``len(df)`` where each element is the
    trendline dict valid at that bar (or ``None``). Calls into the
    optional :mod:`trendln` library; if unavailable, all entries are
    ``None`` and :func:`detect_trendline_bounce` returns ``[]``.

    The expensive trendln call runs only every ``interval`` bars; the
    cached result is held forward until the next refresh.
    """
    n = len(df)
    cache: list[dict[str, Any] | None] = [None] * n

    if n < 26:
        return cache

    last_result: dict[str, Any] | None = None
    last_week = -interval - 1
    max_lookback = 104  # cap at 2 years for trendln performance

    for i in range(26, n):
        atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr_val <= 0:
            cache[i] = last_result
            continue

        if i - last_week >= interval:
            start_idx = max(0, i + 1 - max_lookback)
            last_result = _detect_trendlines(
                df.iloc[start_idx : i + 1],
                float(df["Close"].iloc[i]),
                float(df["Close"].iloc[i - 1]) if i > 0 else float(df["Close"].iloc[i]),
                atr_val,
                swing_lows_data,
            )
            last_week = i

        cache[i] = last_result

    return cache


def detect_trendline_bounce(
    df: pd.DataFrame,
    stages_series: pd.Series,
    atr_series: pd.Series,
    tl_cache: list[dict[str, Any] | None],
    min_gap: int = 4,
) -> list[dict[str, Any]]:
    """Detect bounces off rising support trendlines.

    Consumes the cache produced by :func:`precompute_trendlines`.
    Returns ``[]`` when the cache contains no trendlines (e.g.,
    ``trendln`` is not installed or no rising support was found).
    """
    events: list[dict[str, Any]] = []
    closes = df["Close"].to_numpy()
    lows = df["Low"].to_numpy()
    n = len(df)
    if n < 26:
        return events

    last_entry_idx = -min_gap - 1

    for i in range(26, n):
        if stages_series.iloc[i] != "stage2":
            continue
        if i - last_entry_idx < min_gap:
            continue

        atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr_val <= 0:
            continue

        cached_tl = tl_cache[i]
        if cached_tl is None:
            continue

        rising = cached_tl.get("rising_support")
        if rising is None:
            continue

        projected = rising.get("current_projected_level")
        if projected is None or projected <= 0:
            continue

        slope = rising.get("slope_per_week", 0) or 0
        # Forward-project the cached level a few bars (cache may be stale).
        projected_at_i = projected + slope * 2

        close_i = closes[i]
        low_i = lows[i]
        if low_i > projected_at_i * (1 + 1.0 * atr_val / projected_at_i):
            continue
        if close_i <= projected_at_i * 0.97:
            continue

        span = rising.get("span_weeks", 0) or 0
        anchors = rising.get("anchor_points", 0)
        anchor_count = len(anchors) if isinstance(anchors, list | tuple) else int(anchors or 0)
        slope_pct_month = round(slope / projected_at_i * 100 * 4.33, 2) if projected_at_i > 0 else 0
        dist_atr = round((low_i - projected_at_i) / atr_val, 2) if atr_val > 0 else 0

        last_entry_idx = i
        events.append(
            {
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": round(float(close_i), 2),
                "entry_type": "TRENDLINE_BOUNCE",
                "trendline_slope_pct_month": slope_pct_month,
                "trendline_anchor_count": anchor_count,
                "trendline_age_weeks": span,
                "distance_from_trendline_atr": dist_atr,
                "stop_atr_level": round(projected_at_i - 1.0 * atr_val, 2),
            }
        )

    return events


def _detect_trendlines(
    df_weekly: pd.DataFrame,
    current_price: float,
    last_completed_close: float,
    weekly_atr: float | None,
    swing_lows_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Lightweight wrapper around the optional ``trendln`` library.

    Returns ``None`` if ``trendln`` is not installed (the
    TRENDLINE_BOUNCE detector then returns no signals). Returns a dict
    with ``rising_support`` and ``falling_resistance`` keys when
    trendlines are detected, or those keys may be ``None`` when no
    qualifying line is found.

    To enable this detector, install with ``pip install skysurf[trendlines]``.
    """
    try:
        import trendln
    except ImportError:
        _LOG.debug("trendln not installed; TRENDLINE_BOUNCE will return no signals")
        return None

    if weekly_atr is None or len(df_weekly) < 21:
        return _null_trendlines()

    try:
        completed = df_weekly.iloc[:-1]
        n = len(completed)
        last_idx = n - 1
        lows = completed["Low"].to_numpy()
        highs = completed["High"].to_numpy()

        support_candidates: list[dict[str, Any]] = []
        resistance_candidates: list[dict[str, Any]] = []

        for method in (trendln.METHOD_NSQUREDLOGN, trendln.METHOD_NCUBED):
            try:
                mins, _maxs = trendln.calc_support_resistance(
                    lows, extmethod=trendln.METHOD_NAIVE, method=method
                )
                for pts, (slope, intercept, _ssr, slope_err, _int_err, _area_avg) in mins[2]:
                    support_candidates.append(
                        {
                            "slope": slope,
                            "intercept": intercept,
                            "anchor_indices": list(pts),
                            "slope_err": slope_err,
                        }
                    )
                break
            except Exception:
                continue

        for method in (trendln.METHOD_NSQUREDLOGN, trendln.METHOD_NCUBED):
            try:
                _mins2, maxs2 = trendln.calc_support_resistance(
                    highs, extmethod=trendln.METHOD_NAIVE, method=method
                )
                for pts, (slope, intercept, _ssr, slope_err, _int_err, _area_avg) in maxs2[2]:
                    resistance_candidates.append(
                        {
                            "slope": slope,
                            "intercept": intercept,
                            "anchor_indices": list(pts),
                            "slope_err": slope_err,
                        }
                    )
                break
            except Exception:
                continue

        rising_support = _best_rising_support(
            support_candidates, last_completed_close, weekly_atr, last_idx
        )
        return {
            "rising_support": rising_support,
            "falling_resistance": None,  # not used by PHASE_4_V1
            "trendline_status": "ok" if rising_support else "no_qualifying_line",
        }
    except Exception:
        _LOG.exception("trendln detection raised; returning null")
        return _null_trendlines()


def _best_rising_support(
    candidates: list[dict[str, Any]],
    last_close: float,
    atr: float,
    last_idx: int,
    *,
    min_anchors: int = 3,
    min_span_weeks: int = 12,
    support_atr_mult: float = 4.0,
) -> dict[str, Any] | None:
    """Pick the strongest rising-support trendline from candidates."""
    qualified: list[tuple[float, float, dict[str, Any]]] = []
    for c in candidates:
        anchors = c["anchor_indices"]
        if len(anchors) < min_anchors:
            continue
        span = anchors[-1] - anchors[0]
        if span < min_span_weeks or c["slope"] <= 0:
            continue
        projected = c["slope"] * last_idx + c["intercept"]
        if abs(last_close - projected) > support_atr_mult * atr:
            continue
        score = float(len(anchors) * span)
        qualified.append((-score, float(c.get("slope_err", float("inf"))), c))

    if not qualified:
        return None

    qualified.sort(key=lambda x: (x[0], x[1]))
    best = qualified[0][2]
    projected = best["slope"] * last_idx + best["intercept"]
    return {
        "current_projected_level": float(projected),
        "slope_per_week": float(best["slope"]),
        "anchor_points": list(best["anchor_indices"]),
        "span_weeks": int(best["anchor_indices"][-1] - best["anchor_indices"][0]),
        "significance_score": float(
            len(best["anchor_indices"]) * (best["anchor_indices"][-1] - best["anchor_indices"][0])
        ),
    }


def _null_trendlines() -> dict[str, Any]:
    return {
        "rising_support": None,
        "falling_resistance": None,
        "trendline_status": "insufficient_data",
    }


# ── Entry-quality enrichment ─────────────────────────────────────────


def compute_entry_quality(
    df: pd.DataFrame,
    event: dict[str, Any],
    major_swing_highs: list[int],
    minor_swing_lows: list[int],
    atr_series: pd.Series,
    base_start_idx: int | None = None,
) -> dict[str, Any]:
    """Enrich an entry event with volume ratio, stop, target, and R/R.

    Mutates ``event`` in place and returns it. Each entry type derives
    its stop differently:

    * BREAKOUT — base-period low
    * PULLBACK — most recent minor swing low below entry
    * VCP — consolidation low
    * Other — ``close − 1.5 × ATR`` (fallback)

    Target is the nearest future major swing high above entry, falling
    back to ``entry × 1.20`` when none exists.
    """
    idx = event["week_idx"]
    close = event["close"]
    entry_type = event["entry_type"]
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    volumes = df["Volume"].to_numpy()

    # Volume ratio (skipped if event already carries one).
    if "volume_ratio" not in event:
        start = max(0, idx - 20)
        trailing = volumes[start:idx]
        avg_vol = float(np.mean(trailing)) if len(trailing) > 0 else 1.0
        event["volume_ratio"] = round(float(volumes[idx]) / avg_vol, 2) if avg_vol > 0 else 0.0

    # Stop level.
    atr_val = float(atr_series.iloc[idx]) if not pd.isna(atr_series.iloc[idx]) else 0.0

    if entry_type == "BREAKOUT_S1S2" and base_start_idx is not None:
        event["stop_level"] = round(float(np.min(lows[base_start_idx:idx])), 2)
    elif entry_type == "PULLBACK_S2":
        stop = next(
            (
                float(lows[j])
                for j in reversed(minor_swing_lows)
                if j < idx and float(lows[j]) < close
            ),
            None,
        )
        event["stop_level"] = (
            round(stop, 2) if stop is not None else round(close - 1.5 * atr_val, 2)
        )
    elif entry_type == "VCP_CONTINUATION":
        consol_low = event.get("consolidation_low")
        event["stop_level"] = (
            round(float(consol_low), 2)
            if consol_low is not None
            else round(close - 1.5 * atr_val, 2)
        )
    else:
        # RETEST_SUPPORT, TRENDLINE_BOUNCE, and any others — use the
        # event's own stop_atr_level if provided, else fallback.
        existing_stop = event.get("stop_level") or event.get("stop_atr_level")
        event["stop_level"] = (
            round(float(existing_stop), 2)
            if existing_stop is not None
            else round(close - 1.5 * atr_val, 2)
        )

    # Target level.
    target = next(
        (float(highs[j]) for j in major_swing_highs if j > idx and float(highs[j]) > close),
        None,
    )
    event["target_level"] = round(target, 2) if target is not None else round(close * 1.20, 2)

    # Risk / reward.
    risk = close - event["stop_level"]
    reward = event["target_level"] - close
    event["rr_ratio"] = round(reward / risk, 2) if risk > 0 else 0.0

    return event
