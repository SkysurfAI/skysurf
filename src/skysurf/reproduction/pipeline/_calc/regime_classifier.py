"""Skysurf V2 — Market regime classification engine.

Pure functions for Nifty 50 regime classification (bull / bear / sideways)
based on weekly chart structure, 20-week EMA analysis, and breadth metrics.

This module is the V2 replacement for the regime logic in
``stock_analysis_service._fetch_market_context()`` and the
``MarketRegimeDaily`` table.  It does NOT modify any V1 code — existing
``detect_higher_lows()``, ``detect_lower_highs()``, ``classify_trend()``,
``detect_swing_lows()``, ``detect_swing_highs()``, and ``calculate_atr()``
are reused unchanged.

Architecture rule: every computation is a standalone, stateless function.
No DB reads, no side effects.  The caller provides all data, the function
returns a typed result (plain Python dicts/lists/floats/strings/bools/None).

ROUNDING RULE: Never round intermediate calculations.  Only round at the
final return value of each public function.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ── Regime classification thresholds (Skysurf V2) ─────────────────────
EMA_DIRECTION_THRESHOLD = 0.3       # ATR-normalized slope for rising/falling
EMA_POSITION_LOOKBACK = 4           # weeks for price vs EMA consistency
EMA_DIRECTION_LOOKBACK = 4          # weeks for slope calculation
RECENT_WEEKS_COUNT = 3              # weeks in recent character summary
BREADTH_BULL_FLOOR = 35             # below this %, override bull → sideways
BREADTH_BEAR_CEILING = 65           # above this %, override bear → sideways
BREADTH_NARROWING_THRESHOLD = 15    # 200d − 50d gap above this = narrowing
BREADTH_BROADENING_THRESHOLD = -5   # 200d − 50d gap below this = broadening
_MIN_WEEKLY_ROWS = 52               # minimum rows for compute_nifty_regime


# ── Function 1 ────────────────────────────────────────────────────────


def detect_higher_lows_regime(
    swing_lows: list[dict], min_swings: int = 3
) -> Optional[bool]:
    """Check if swing lows form a higher-lows pattern using majority rule.

    Unlike the strict ``detect_higher_lows()`` in *skysurf_calc* (which
    requires **every** successive pair to be ascending), this version
    counts how many pairs are ascending and uses a >50% threshold.  One
    messy pullback in a sequence of rising lows does not break the
    pattern — only a majority of declining pairs does.

    This relaxed version is used exclusively for Nifty regime
    classification, where a single noisy swing can cascade into every
    stock analysis via prompt selection.

    Note: at exactly *min_swings=3* (2 pairs), majority rule requires
    both pairs to agree — tolerance only kicks in with 4+ swing lows.

    Args:
        swing_lows: List of swing-low dicts, each with at minimum a
            ``"level"`` key (float).  Same format as
            ``detect_swing_lows()`` output.
        min_swings: Minimum number of swing lows required.  If fewer
            exist, the function returns ``None`` (insufficient data).

    Returns:
        ``None`` if fewer than *min_swings* entries.
        ``True`` if more than half the successive pairs show a higher low.
        ``False`` otherwise (including exact ties and equal levels).
    """
    if len(swing_lows) < min_swings:
        return None

    recent = swing_lows[-min(4, len(swing_lows)) :]
    total_pairs = len(recent) - 1
    higher_count = sum(
        1
        for i in range(1, len(recent))
        if recent[i]["level"] > recent[i - 1]["level"]
    )
    return higher_count > total_pairs / 2


# ── Function 2 ────────────────────────────────────────────────────────


def detect_lower_highs_regime(
    swing_highs: list[dict], min_swings: int = 3
) -> Optional[bool]:
    """Check if swing highs form a lower-highs pattern using majority rule.

    Mirror of :func:`detect_higher_lows_regime`.  Counts how many
    successive pairs show a *lower* high and returns ``True`` when more
    than half do.

    Args:
        swing_highs: List of swing-high dicts, each with a ``"level"``
            key (float).
        min_swings: Minimum entries required; returns ``None`` if fewer.

    Returns:
        ``None`` if insufficient data.
        ``True`` if majority of pairs show a lower high.
        ``False`` otherwise.
    """
    if len(swing_highs) < min_swings:
        return None

    recent = swing_highs[-min(4, len(swing_highs)) :]
    total_pairs = len(recent) - 1
    lower_count = sum(
        1
        for i in range(1, len(recent))
        if recent[i]["level"] < recent[i - 1]["level"]
    )
    return lower_count > total_pairs / 2


# ── Function 3 ────────────────────────────────────────────────────────


def compute_ema_position(
    weekly_closes: pd.Series,
    ema_series: pd.Series,
    lookback_weeks: int = EMA_POSITION_LOOKBACK,
) -> dict:
    """Determine whether price is consistently above or below the EMA.

    Instead of checking just the latest week (which can flicker), this
    function checks the pattern over the last *lookback_weeks* weeks.

    Args:
        weekly_closes: Weekly closing prices (pd.Series).
        ema_series: Pre-computed EMA values (same index as *weekly_closes*).
        lookback_weeks: Number of recent weeks to examine (default 4).

    Returns:
        Dict with keys:
        - ``"position"``: ``"above"`` | ``"below"`` | ``"at_ema"``
        - ``"distance_pct"``: current close vs current EMA, rounded 1 dp
        - ``"weeks_above"``: count of weeks close > EMA in lookback
        - ``"weeks_below"``: count of weeks close ≤ EMA in lookback
    """
    n = min(lookback_weeks, len(weekly_closes), len(ema_series))
    if n == 0:
        return {
            "position": "at_ema",
            "distance_pct": 0.0,
            "weeks_above": 0,
            "weeks_below": 0,
        }

    closes_tail = weekly_closes.iloc[-n:]
    ema_tail = ema_series.iloc[-n:]

    weeks_above = int((closes_tail > ema_tail).sum())
    weeks_below = n - weeks_above

    threshold_above = n * 0.75
    threshold_below = n * 0.25

    if weeks_above >= threshold_above:
        position = "above"
    elif weeks_above <= threshold_below:
        position = "below"
    else:
        position = "at_ema"

    current_close = float(weekly_closes.iloc[-1])
    current_ema = float(ema_series.iloc[-1])
    if current_ema != 0:
        distance_pct = round((current_close - current_ema) / current_ema * 100, 1)
    else:
        distance_pct = 0.0

    return {
        "position": position,
        "distance_pct": distance_pct,
        "weeks_above": weeks_above,
        "weeks_below": weeks_below,
    }


# ── Function 4 ────────────────────────────────────────────────────────


def compute_ema_direction(
    ema_series: pd.Series,
    atr_value: float,
    lookback_weeks: int = EMA_DIRECTION_LOOKBACK,
) -> dict:
    """Determine whether the 20-week EMA is rising, falling, or flat.

    The slope threshold is normalised by ATR so it adapts to market
    volatility — the same absolute slope means different things in calm
    vs volatile markets.

    Args:
        ema_series: Pre-computed EMA values (pd.Series).
        atr_value: Current weekly ATR (from ``calculate_atr()``).
        lookback_weeks: Window for slope calculation (default 4).

    Returns:
        Dict with keys:
        - ``"direction"``: ``"rising"`` | ``"falling"`` | ``"flat"``
        - ``"slope_pct"``: percentage change of EMA over the lookback,
          rounded to 2 dp
        - ``"normalized_slope"``: slope divided by ATR, rounded to 2 dp
    """
    if len(ema_series) < lookback_weeks + 1 or atr_value is None:
        return {"direction": "flat", "slope_pct": 0.0, "normalized_slope": 0.0}

    ema_now = float(ema_series.iloc[-1])
    ema_ago = float(ema_series.iloc[-1 - lookback_weeks])

    slope_abs = ema_now - ema_ago

    if atr_value > 0:
        normalized_slope = round(slope_abs / atr_value, 2)
    else:
        normalized_slope = 0.0

    if ema_ago != 0:
        slope_pct = round(slope_abs / ema_ago * 100, 2)
    else:
        slope_pct = 0.0

    if normalized_slope > EMA_DIRECTION_THRESHOLD:
        direction = "rising"
    elif normalized_slope < -EMA_DIRECTION_THRESHOLD:
        direction = "falling"
    else:
        direction = "flat"

    return {
        "direction": direction,
        "slope_pct": slope_pct,
        "normalized_slope": normalized_slope,
    }


# ── Function 5 ────────────────────────────────────────────────────────


def classify_recent_character(
    weekly_df: pd.DataFrame, num_weeks: int = RECENT_WEEKS_COUNT
) -> dict:
    """Classify the recent weekly price action into a descriptive label.

    Instead of returning raw weekly changes, this function interprets
    them into one of seven vocabulary values that the AI can use directly.

    The classification logic operates on changes in chronological order
    (oldest first), so ``changes[-1]`` is the most recent week.

    Args:
        weekly_df: DataFrame with at minimum a ``Close`` column and a
            ``DatetimeIndex``.
        num_weeks: Number of recent weeks to characterise (default 3).

    Returns:
        Dict with keys:
        - ``"character"``: one of ``"strong_rally"``,
          ``"grinding_higher"``, ``"sharp_decline"``,
          ``"grinding_lower"``, ``"recovering"``, ``"fading"``,
          ``"choppy"``
        - ``"weekly_changes"``: list of pct-change floats (rounded 1 dp),
          **most recent first** (reversed from computation order)
    """
    if len(weekly_df) < num_weeks + 1:
        return {"character": "choppy", "weekly_changes": []}

    tail = weekly_df.iloc[-(num_weeks + 1) :]
    closes = tail["Close"]
    pct_changes = closes.pct_change().dropna() * 100  # chronological order

    # Build list in chronological order for pattern matching
    changes = [round(float(v), 1) for v in pct_changes]

    all_positive = all(c > 0 for c in changes)
    all_negative = all(c < 0 for c in changes)
    avg_change = float(np.mean(changes))
    net_change = float(np.sum(changes))
    any_large_drop = any(c < -3.0 for c in changes)
    last_positive = changes[-1] > 0  # most recent week
    last_negative = changes[-1] < 0

    if all_positive and avg_change > 1.0:
        character = "strong_rally"
    elif all_positive:
        character = "grinding_higher"
    elif all_negative and (avg_change < -1.0 or any_large_drop):
        character = "sharp_decline"
    elif all_negative:
        character = "grinding_lower"
    elif last_positive and net_change > 0 and not all_positive:
        character = "recovering"
    elif last_negative and net_change < 0 and not all_negative:
        character = "fading"
    else:
        character = "choppy"

    # Return weekly_changes most-recent-first for display / _debug
    return {"character": character, "weekly_changes": list(reversed(changes))}


# ── Function 6 ────────────────────────────────────────────────────────


def classify_breadth_health(
    breadth_200dma_pct: Optional[int],
    breadth_50dma_pct: Optional[int],
    breadth_20wma_pct: Optional[int],
) -> str:
    """Classify breadth as broadening, stable, or narrowing.

    Compares medium-term breadth (50 DMA) against long-term breadth
    (200 DMA).  A large positive gap means many stocks maintain
    long-term uptrends but are losing medium-term momentum — classic
    narrowing.

    A secondary check uses the 20-week EMA breadth: if the 200d–50d gap
    is moderate but the weekly breadth is deteriorating sharply relative
    to the 50 DMA, that's an early narrowing signal.

    Args:
        breadth_200dma_pct: % of Nifty 500 above 200 DMA.
        breadth_50dma_pct: % of Nifty 500 above 50 DMA.
        breadth_20wma_pct: % of Nifty 500 above 20-week MA.

    Returns:
        ``"broadening"`` | ``"stable"`` | ``"narrowing"`` | ``"unknown"``
    """
    if breadth_200dma_pct is None or breadth_50dma_pct is None:
        return "unknown"

    gap = breadth_200dma_pct - breadth_50dma_pct

    if gap > BREADTH_NARROWING_THRESHOLD:
        return "narrowing"
    if gap < BREADTH_BROADENING_THRESHOLD:
        return "broadening"

    # Moderate gap (5–15): check 20w EMA tiebreaker
    if (
        breadth_20wma_pct is not None
        and 5 <= gap <= BREADTH_NARROWING_THRESHOLD
        and (breadth_50dma_pct - breadth_20wma_pct) >= 10
    ):
        return "narrowing"

    return "stable"


# ── Function 7 ────────────────────────────────────────────────────────


def classify_regime(
    ema_position: str,
    ema_direction: str,
    breadth_200dma_pct: Optional[int] = None,
) -> dict:
    """Classify market regime using Weinstein-style two-signal logic.

    Aligns with Stan Weinstein's Stage Analysis:

    - **Bull (Stage 2)**: price consistently above a rising 20-week EMA
    - **Bear (Stage 4)**: price consistently below a falling 20-week EMA
    - **Sideways (Stage 1/3)**: everything else — EMA flat, price tangled
      with EMA, or conflicting signals (above but falling, below but rising)

    Swing detection (higher lows / lower highs) is computed separately for
    contextual debug data but is **not** used for classification.  Historical
    validation showed that adding swing detection as a third gate caused the
    classifier to spend 65% of the time in sideways, missing confirmed bull
    runs (2021, 2023) due to minor pullbacks breaking the swing pattern.

    A breadth safety override can downgrade bull/bear to sideways when the
    breadth picture contradicts the structural classification.

    Args:
        ema_position: ``"above"`` | ``"below"`` | ``"at_ema"``
            (from :func:`compute_ema_position`).
        ema_direction: ``"rising"`` | ``"falling"`` | ``"flat"``
            (from :func:`compute_ema_direction`).
        breadth_200dma_pct: Optional; % of Nifty 500 above 200 DMA.

    Returns:
        Dict with keys:
        - ``"classification"``: ``"bull"`` | ``"bear"`` | ``"sideways"``
        - ``"breadth_override"``: ``True`` if breadth changed the result
        - ``"raw_classification"``: classification before breadth check
    """
    # Two-signal Weinstein-style classification
    if ema_position == "above" and ema_direction == "rising":
        raw = "bull"
    elif ema_position == "below" and ema_direction == "falling":
        raw = "bear"
    else:
        raw = "sideways"

    # Breadth safety override
    classification = raw
    breadth_override = False

    if breadth_200dma_pct is not None:
        if raw == "bull" and breadth_200dma_pct < BREADTH_BULL_FLOOR:
            classification = "sideways"
            breadth_override = True
        elif raw == "bear" and breadth_200dma_pct > BREADTH_BEAR_CEILING:
            classification = "sideways"
            breadth_override = True

    return {
        "classification": classification,
        "breadth_override": breadth_override,
        "raw_classification": raw,
    }


# ── Function 8 (orchestrator) ─────────────────────────────────────────


def compute_nifty_regime(
    index_df_weekly: pd.DataFrame,
    breadth_data: Optional[dict] = None,
) -> dict:
    """Compute the complete Nifty 50 regime record from weekly candles.

    This is the orchestrator function that ties everything together.
    It calls standalone functions in sequence and assembles the result
    dict ready for storage and AI consumption.

    The function does NOT compute ``weeks_in_current_condition`` or
    ``breadth_pct_4_weeks_ago`` — those require historical state that
    the caller manages via the database.

    Args:
        index_df_weekly: DataFrame with ``Open``, ``High``, ``Low``,
            ``Close``, ``Volume`` columns and a ``DatetimeIndex``.
            At least 52 rows expected (1 year minimum).
        breadth_data: Optional dict with breadth metrics::

                {
                    "breadth_200dma_pct": int,
                    "breadth_50dma_pct": int,
                    "breadth_20wma_pct": int,
                    "new_52w_highs": int,
                    "new_52w_lows": int,
                    "new_highs_vs_lows_ratio": float,
                    "breadth_pct_4_weeks_ago": int,  # optional, passed through
                }

    Returns:
        Dict with AI-facing payload fields and a ``_debug`` section.
        See module-level docstring for full shape.
    """
    # Late imports to keep standalone functions import-free
    from .skysurf_calc import (
        classify_trend,
        detect_higher_lows,
        detect_lower_highs,
        detect_swing_highs,
        detect_swing_lows,
    )
    from .technical_indicators import calculate_atr

    # ── Insufficient data guard ────────────────────────────────────
    if len(index_df_weekly) < _MIN_WEEKLY_ROWS:
        return {
            "_context": "Nifty 50 weekly structure and Nifty 500 breadth as of last Saturday",
            "market_condition": "insufficient_data",
            "weeks_in_current_condition": None,
            "nifty_above_nearest_support_pct": None,
            "nifty_above_20w_moving_avg_pct": None,
            "nifty_recent_weeks_character": None,
            "pct_of_nifty500_above_200d_avg": None,
            "breadth_health": "unknown",
            "breadth_pct_4_weeks_ago": None,
            "new_52w_highs_to_lows_ratio": None,
            "_debug": {},
        }

    # ── 1. ATR on weekly candles ───────────────────────────────────
    atr_series = calculate_atr(
        index_df_weekly["High"],
        index_df_weekly["Low"],
        index_df_weekly["Close"],
        window=14,
    )
    atr_value = (
        float(atr_series.iloc[-1])
        if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1])
        else None
    )

    # ── 2. Swing detection (reuse existing V1 functions) ───────────
    swing_lows = detect_swing_lows(
        index_df_weekly, min_spacing=2, atr=atr_value
    )
    swing_highs = detect_swing_highs(
        index_df_weekly, min_spacing=2, atr=atr_value
    )

    # ── 3. Majority-rule swing structure (V2) ──────────────────────
    higher_lows = detect_higher_lows_regime(swing_lows)
    lower_highs = detect_lower_highs_regime(swing_highs)

    # ── 4. Classic trend label for _debug (V1 strict functions) ────
    strict_hl = detect_higher_lows(swing_lows)
    strict_lh = detect_lower_highs(swing_highs)
    weekly_trend = classify_trend(strict_hl, strict_lh)

    # ── 5. 20-week EMA ────────────────────────────────────────────
    ema_series = index_df_weekly["Close"].ewm(span=20, adjust=False).mean()

    # ── 6. EMA position (4-week consistency) ───────────────────────
    ema_pos = compute_ema_position(index_df_weekly["Close"], ema_series)

    # ── 7. EMA direction (ATR-normalized slope) ────────────────────
    ema_dir = compute_ema_direction(ema_series, atr_value)

    # ── 8. Recent weeks character ──────────────────────────────────
    recent = classify_recent_character(index_df_weekly)

    # ── 9. Classify regime ─────────────────────────────────────────
    breadth_200 = breadth_data.get("breadth_200dma_pct") if breadth_data else None
    regime = classify_regime(
        ema_pos["position"],
        ema_dir["direction"],
        breadth_200,
    )

    # ── 10. Price distances ────────────────────────────────────────
    current_price = float(index_df_weekly["Close"].iloc[-1])
    current_ema = float(ema_series.iloc[-1])

    last_swing_low = swing_lows[-1] if swing_lows else None
    last_swing_high = swing_highs[-1] if swing_highs else None

    distance_from_swing_low_pct = None
    if last_swing_low:
        distance_from_swing_low_pct = round(
            (current_price - last_swing_low["level"])
            / last_swing_low["level"]
            * 100,
            1,
        )

    distance_from_swing_high_pct = None
    if last_swing_high:
        distance_from_swing_high_pct = round(
            (current_price - last_swing_high["level"])
            / last_swing_high["level"]
            * 100,
            1,
        )

    # ── 11. Breadth fields ─────────────────────────────────────────
    breadth_50 = breadth_data.get("breadth_50dma_pct") if breadth_data else None
    breadth_20w = breadth_data.get("breadth_20wma_pct") if breadth_data else None
    breadth_health = classify_breadth_health(breadth_200, breadth_50, breadth_20w)

    highs_vs_lows = (
        breadth_data.get("new_highs_vs_lows_ratio") if breadth_data else None
    )
    breadth_4w_ago = (
        breadth_data.get("breadth_pct_4_weeks_ago") if breadth_data else None
    )

    # ── 12. Strip internal "index" key from swing dicts ────────────
    debug_swing_low = None
    if last_swing_low:
        debug_swing_low = {
            "level": last_swing_low["level"],
            "date": last_swing_low["date"],
        }

    debug_swing_high = None
    if last_swing_high:
        debug_swing_high = {
            "level": last_swing_high["level"],
            "date": last_swing_high["date"],
        }

    # ── 13. Assemble return dict ───────────────────────────────────
    return {
        # Inline context for the AI
        "_context": "Nifty 50 weekly structure and Nifty 500 breadth as of last Saturday",
        # AI-facing payload
        "market_condition": regime["classification"],
        "weeks_in_current_condition": None,  # caller manages via DB
        "nifty_above_nearest_support_pct": distance_from_swing_low_pct,
        "nifty_above_20w_moving_avg_pct": ema_pos["distance_pct"],
        "nifty_recent_weeks_character": recent["character"],
        "pct_of_nifty500_above_200d_avg": breadth_200,
        "breadth_health": breadth_health,
        "breadth_pct_4_weeks_ago": breadth_4w_ago,
        "new_52w_highs_to_lows_ratio": highs_vs_lows,
        # Debug / transparency (not sent to AI, stored in DB)
        "_debug": {
            "raw_classification": regime["raw_classification"],
            "breadth_override": regime["breadth_override"],
            "nifty_weekly_trend": weekly_trend,
            "nifty_higher_lows": higher_lows,
            "nifty_lower_highs": lower_highs,
            "nifty_price_vs_20w_ema": ema_pos["position"],
            "nifty_20w_ema_direction": ema_dir["direction"],
            "nifty_current_price": current_price,
            "nifty_20w_ema_value": current_ema,
            "nifty_last_swing_low": debug_swing_low,
            "nifty_last_swing_high": debug_swing_high,
            "nifty_distance_from_swing_high_pct": distance_from_swing_high_pct,
            "ema_slope_pct": ema_dir["slope_pct"],
            "ema_normalized_slope": ema_dir["normalized_slope"],
            "ema_weeks_above": ema_pos["weeks_above"],
            "ema_weeks_below": ema_pos["weeks_below"],
            "atr_value": atr_value,
            "swing_low_count": len(swing_lows),
            "swing_high_count": len(swing_highs),
            "recent_weeks_raw": recent["weekly_changes"],
        },
    }
