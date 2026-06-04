"""Skysurf core calculation engine — Phase 1.

Pure functions for swing detection, trend classification, regime analysis,
and all derived metrics defined in the JSON restructure spec.

All functions are independently testable with no database, API, or service
dependencies.  They accept pandas DataFrames (DatetimeIndex + OHLCV columns)
and return plain Python types (dicts, lists, floats, strings, booleans, None).

ROUNDING RULE: Never round intermediate calculations.  Only round at the final
return value of each public function.
"""
from __future__ import annotations

import math
from typing import Optional

import logging
import numpy as np
import pandas as pd

try:
    import trendln
    _HAS_TRENDLN = True
except ImportError:
    _HAS_TRENDLN = False

_logger = logging.getLogger(__name__)

# ── Trendline constants ────────────────────────────────────────────────
_TL_MIN_ANCHORS = 3
_TL_MIN_SPAN_WEEKS = 12
_TL_SUPPORT_ATR_MULT = 4.0
_TL_RESISTANCE_ATR_MULT = 2.0

# ── Resampling ──────────────────────────────────────────────────────────


def resample_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily candles into weekly candles (ISO week, Mon-Fri).

    Args:
        df_daily: DataFrame with DatetimeIndex and OHLCV columns.

    Returns:
        Weekly DataFrame. Short weeks (holidays) still produce one candle.
    """
    if df_daily.empty:
        return df_daily.iloc[0:0].copy()

    weekly = df_daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    return weekly


def is_last_week_complete(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    today: "date",
) -> bool:
    """Check whether the last weekly candle in *df_weekly* is complete.

    The last W-FRI bucket is complete when:
      1. *today* is strictly after the last trading day in that bucket, OR
      2. *today* falls in the bucket but no daily data exists for *today*
         (market closed — weekend or holiday).

    This is data-driven: it uses the actual trading days present in
    *df_daily* rather than a hardcoded holiday calendar.
    """
    if df_weekly.empty or df_daily.empty:
        return False

    last_weekly_label = df_weekly.index[-1]  # the Friday that labels this bucket
    last_daily_date = df_daily.index[-1].date() if hasattr(df_daily.index[-1], "date") else df_daily.index[-1]

    # If today is after the last trading day we have data for,
    # that last daily candle is final.  The weekly bucket containing it
    # is complete because no more trading will happen in it.
    if today > last_daily_date:
        return True

    return False


def split_weekly_completed(
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    today: "date",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split *df_weekly* into completed candles and the current partial week.

    Returns
    -------
    (completed, current_partial)
        *current_partial* is a single-row DataFrame or ``None`` when the
        last week is already complete (e.g. Saturday/Sunday/holiday).
    """
    if df_weekly.empty:
        return df_weekly, None

    if is_last_week_complete(df_weekly, df_daily, today):
        # All weeks are complete — no partial current week
        return df_weekly, None
    else:
        # Last row is the in-progress week
        completed = df_weekly.iloc[:-1] if len(df_weekly) > 1 else df_weekly.iloc[0:0]
        current = df_weekly.iloc[[-1]]
        return completed, current


def resample_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily candles into monthly candles (calendar month).

    Args:
        df_daily: DataFrame with DatetimeIndex and OHLCV columns.

    Returns:
        Monthly DataFrame.
    """
    if df_daily.empty:
        return df_daily.iloc[0:0].copy()

    monthly = df_daily.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    return monthly


# ── Swing Detection ─────────────────────────────────────────────────────


def detect_swing_lows(
    df: pd.DataFrame,
    min_spacing: int = 2,
    atr: Optional[float] = None,
) -> list[dict]:
    """Detect significant swing lows using strict < comparison.

    A candle at index i is a swing low if:
        low[i] < low[i-1] AND low[i] < low[i-2] AND
        low[i] < low[i+1] AND low[i] < low[i+2]

    Latest 2 candles can never be confirmed swings.

    Args:
        df: DataFrame with DatetimeIndex and Low/High columns.
        min_spacing: Minimum index distance between consecutive swings.
        atr: If provided, apply significance filter — drop from preceding
             high must exceed 1.0x ATR.

    Returns:
        List of dicts (most recent 5):
        [{"index": int, "level": float, "date": str}, ...]
    """
    if len(df) < 5:
        return []

    lows = df["Low"].values
    highs = df["High"].values

    raw_swings: list[dict] = []
    for i in range(2, len(lows) - 2):
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            date_val = df.index[i]
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
            raw_swings.append({"index": i, "level": float(lows[i]), "date": date_str})

    # Significance filter
    if atr is not None and atr > 0:
        significant: list[dict] = []
        for sw in raw_swings:
            if sw["index"] > 0:
                preceding_high = float(highs[: sw["index"]].max())
                if (preceding_high - sw["level"]) >= atr:
                    significant.append(sw)
        raw_swings = significant

    # Minimum spacing
    filtered: list[dict] = []
    for sw in raw_swings:
        if not filtered or (sw["index"] - filtered[-1]["index"]) >= min_spacing:
            filtered.append(sw)

    return filtered[-5:]


def detect_swing_highs(
    df: pd.DataFrame,
    min_spacing: int = 2,
    atr: Optional[float] = None,
) -> list[dict]:
    """Detect significant swing highs using strict > comparison.

    A candle at index i is a swing high if:
        high[i] > high[i-1] AND high[i] > high[i-2] AND
        high[i] > high[i+1] AND high[i] > high[i+2]

    Latest 2 candles can never be confirmed swings.

    Args:
        df: DataFrame with DatetimeIndex and High/Low columns.
        min_spacing: Minimum index distance between consecutive swings.
        atr: If provided, apply significance filter — rise from preceding
             low must exceed 1.0x ATR.

    Returns:
        List of dicts (most recent 5):
        [{"index": int, "level": float, "date": str}, ...]
    """
    if len(df) < 5:
        return []

    highs = df["High"].values
    lows = df["Low"].values

    raw_swings: list[dict] = []
    for i in range(2, len(highs) - 2):
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            date_val = df.index[i]
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
            raw_swings.append({"index": i, "level": float(highs[i]), "date": date_str})

    # Significance filter
    if atr is not None and atr > 0:
        significant: list[dict] = []
        for sw in raw_swings:
            if sw["index"] > 0:
                preceding_low = float(lows[: sw["index"]].min())
                if (sw["level"] - preceding_low) >= atr:
                    significant.append(sw)
        raw_swings = significant

    # Minimum spacing
    filtered: list[dict] = []
    for sw in raw_swings:
        if not filtered or (sw["index"] - filtered[-1]["index"]) >= min_spacing:
            filtered.append(sw)

    return filtered[-5:]


def count_touches(
    df: pd.DataFrame,
    level: float,
    level_type: str = "support",
    min_spacing: int = 2,
) -> int:
    """Count candles that touch a level within 1%, with minimum spacing.

    For support: candle low within 1% of level AND close stays above level.
    For resistance: candle high within 1% of level AND close stays below level.

    Touches separated by fewer than min_spacing bars are counted as one.

    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns.
        level: The price level to check touches against.
        level_type: "support" or "resistance" (aliases: "low"/"high").
        min_spacing: Minimum bars between counted touches.

    Returns:
        Integer count of touches.
    """
    count, _ = count_touches_and_dates(
        df=df,
        level=level,
        level_type=level_type,
        min_spacing=min_spacing,
    )
    return count


def count_touches_and_dates(
    df: pd.DataFrame,
    level: float,
    level_type: str = "support",
    min_spacing: int = 2,
) -> tuple[int, list[str]]:
    """Count level touches and return touch dates using spacing de-duplication."""
    if df.empty or level <= 0:
        return 0, []

    band = level * 0.01
    upper = level + band
    lower = level - band

    normalized_level_type = str(level_type).strip().lower()
    if normalized_level_type in {"support", "low"}:
        mask = (df["Low"] >= lower) & (df["Low"] <= upper) & (df["Close"] > level)
    else:
        mask = (df["High"] >= lower) & (df["High"] <= upper) & (df["Close"] < level)

    touch_dates: list[str] = []
    last_pos = -min_spacing  # ensure first always counts
    for idx in range(len(df)):
        if mask.iloc[idx] and (idx - last_pos) >= min_spacing:
            dt = df.index[idx]
            touch_dates.append(
                dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            )
            last_pos = idx

    return len(touch_dates), touch_dates


# ── Trend Detection ─────────────────────────────────────────────────────


def detect_higher_lows(
    swing_lows: list[dict], min_swings: int = 2
) -> Optional[bool]:
    """Check if swing lows form a higher-lows pattern.

    Returns:
        None if fewer than min_swings exist (insufficient data, NOT False).
        True if last 2-3 swings are strictly ascending.
        False otherwise.
    """
    if len(swing_lows) < min_swings:
        return None

    recent = swing_lows[-min(3, len(swing_lows)) :]
    for i in range(1, len(recent)):
        if recent[i]["level"] <= recent[i - 1]["level"]:
            return False
    return True


def detect_lower_highs(
    swing_highs: list[dict], min_swings: int = 2
) -> Optional[bool]:
    """Check if swing highs form a lower-highs pattern.

    Returns:
        None if fewer than min_swings exist (insufficient data, NOT False).
        True if last 2-3 swings are strictly descending.
        False otherwise.
    """
    if len(swing_highs) < min_swings:
        return None

    recent = swing_highs[-min(3, len(swing_highs)) :]
    for i in range(1, len(recent)):
        if recent[i]["level"] >= recent[i - 1]["level"]:
            return False
    return True


def classify_trend(
    higher_lows: Optional[bool], lower_highs: Optional[bool]
) -> str:
    """Classify trend from higher_lows and lower_highs.

    Truth table:
        higher_lows=True,  lower_highs=False  → "uptrend"
        higher_lows=True,  lower_highs=True   → "sideways" (converging)
        higher_lows=False, lower_highs=True    → "downtrend"
        higher_lows=False, lower_highs=False   → "sideways" (expanding)
        Either is None                         → "insufficient_data"
    """
    if higher_lows is None or lower_highs is None:
        return "insufficient_data"
    if higher_lows and not lower_highs:
        return "uptrend"
    if lower_highs and not higher_lows:
        return "downtrend"
    return "sideways"


def calculate_trend_bias(
    higher_lows: Optional[bool],
    lower_highs: Optional[bool],
    swing_lows: list[dict],
    swing_highs: list[dict],
) -> str:
    """Return directional bias for converging structure: bullish/bearish/neutral."""
    if not (higher_lows is True and lower_highs is True):
        return "neutral"
    if len(swing_lows) < 3 or len(swing_highs) < 3:
        return "neutral"

    low_slope = float(swing_lows[-1]["level"]) - float(swing_lows[-3]["level"])
    high_slope = float(swing_highs[-1]["level"]) - float(swing_highs[-3]["level"])
    high_drop = abs(high_slope)

    if low_slope > high_drop:
        return "bullish"
    if high_drop > low_slope:
        return "bearish"
    return "neutral"


def price_above_recent_swing_highs(
    current_price: float,
    swing_highs: list[dict],
    lookback: int = 3,
) -> Optional[bool]:
    """Return True when price is above all recent swing highs, else False/None."""
    if current_price <= 0 or not swing_highs:
        return None
    recent = swing_highs[-max(1, lookback):]
    recent_max = max(float(sw["level"]) for sw in recent)
    return bool(current_price > recent_max)


def calculate_trend_duration_weeks(
    weekly_candles_len: int,
    current_trend: str,
    swing_lows: list[dict],
    swing_highs: list[dict],
) -> int:
    """Weeks since the current swing sequence began.

    For uptrend: walk backward through swing_lows until the higher-low
    sequence breaks.  Duration = total_weeks - 1 - start_index.
    For downtrend: same with swing_highs and lower-high sequence.
    For sideways/insufficient_data: 0.

    Args:
        weekly_candles_len: Total number of weekly candles.
        current_trend: Output of classify_trend().
        swing_lows: Chronologically ordered swing lows with "index" key.
        swing_highs: Chronologically ordered swing highs with "index" key.

    Returns:
        Integer number of weeks.
    """
    if current_trend == "uptrend" and swing_lows:
        # Walk backward: reversed list goes newest→oldest.
        # For uptrend, each older low should be LOWER than the newer one.
        # Break when an older low is NOT lower (>= the newer one).
        lows = list(reversed(swing_lows))
        start_week = lows[0]["index"]
        for i in range(1, len(lows)):
            if lows[i]["level"] >= lows[i - 1]["level"]:
                # This older low broke the higher-low sequence.
                # Sequence started at lows[i-1] (the newer one).
                start_week = lows[i - 1]["index"]
                break
            # Sequence still holds — extend start to this older swing
            start_week = lows[i]["index"]
        return max(0, weekly_candles_len - 1 - start_week)

    elif current_trend == "downtrend" and swing_highs:
        # For downtrend, each older high should be HIGHER than the newer one.
        # Break when an older high is NOT higher (<= the newer one).
        highs = list(reversed(swing_highs))
        start_week = highs[0]["index"]
        for i in range(1, len(highs)):
            if highs[i]["level"] <= highs[i - 1]["level"]:
                start_week = highs[i - 1]["index"]
                break
            start_week = highs[i]["index"]
        return max(0, weekly_candles_len - 1 - start_week)

    return 0


# ── Volume Character ────────────────────────────────────────────────────


def classify_volume_character(df_20d: pd.DataFrame) -> str:
    """Classify volume as heavier_on_up_days, heavier_on_down_days, or neutral.

    Up day: close > open (intraday direction, not gap-adjusted).
    Requires >= 4 up days AND >= 4 down days; otherwise "neutral".
    Doji days (close == open) excluded from both.
    Threshold: 1.3x.
    """
    if df_20d.empty:
        return "neutral"

    up_mask = df_20d["Close"] > df_20d["Open"]
    down_mask = df_20d["Close"] < df_20d["Open"]

    up_vols = df_20d.loc[up_mask, "Volume"]
    down_vols = df_20d.loc[down_mask, "Volume"]

    if len(up_vols) < 4 or len(down_vols) < 4:
        return "neutral"

    avg_up = float(up_vols.mean())
    avg_down = float(down_vols.mean())

    if avg_up > 1.3 * avg_down:
        return "heavier_on_up_days"
    elif avg_down > 1.3 * avg_up:
        return "heavier_on_down_days"
    return "neutral"


# ── Distance Functions ──────────────────────────────────────────────────


def distance_to_nearest_support(
    current_price: float, swing_lows: list[dict]
) -> Optional[float]:
    """Percentage distance to nearest swing low BELOW current price.

    Returns None if no swing lows exist below (freefall territory).
    Always positive — how far below is the floor.
    """
    below = [s for s in swing_lows if s["level"] < current_price]
    if not below:
        return None
    nearest = max(below, key=lambda s: s["level"])
    return round(((current_price - nearest["level"]) / current_price) * 100, 1)


def distance_to_nearest_resistance(
    current_price: float, swing_highs: list[dict]
) -> Optional[float]:
    """Percentage distance to nearest swing high ABOVE current price.

    Returns None if no swing highs exist above (all-time high territory).
    Always positive — how far above is the ceiling.
    """
    above = [s for s in swing_highs if s["level"] > current_price]
    if not above:
        return None
    nearest = min(above, key=lambda s: s["level"])
    return round(((nearest["level"] - current_price) / current_price) * 100, 1)


# ── Fibonacci Targets ───────────────────────────────────────────────────


def calculate_fib_targets(
    swing_lows: list[dict],
    swing_highs: list[dict],
    weekly_atr: float,
    current_week_idx: int,
    current_price: float,
) -> list[dict]:
    """Find most recent A-B-C pattern and return fib extension targets.

    Pattern: C = last swing low, B = last high before C, A = last low before B.
    Seven validation checks must all pass.
    Returns 0-2 target dicts with {level, method, distance_above_pct, distance_in_atr}.
    """
    lows = sorted(swing_lows, key=lambda s: s["index"])
    highs = sorted(swing_highs, key=lambda s: s["index"])

    if not lows:
        return []

    C = lows[-1]

    B_candidates = [h for h in highs if h["index"] < C["index"]]
    if not B_candidates:
        return []
    B = B_candidates[-1]

    A_candidates = [low for low in lows if low["index"] < B["index"]]
    if not A_candidates:
        return []
    A = A_candidates[-1]

    swing_size = B["level"] - A["level"]
    retracement = B["level"] - C["level"]

    checks = [
        B["level"] > A["level"],
        C["level"] > A["level"],
        C["level"] < B["level"],
        swing_size > 2.0 * weekly_atr,
        retracement > 0.236 * swing_size,
        retracement < 0.786 * swing_size,
        (current_week_idx - C["index"]) <= 12,
    ]

    if not all(checks):
        return []

    targets: list[dict] = []

    fib_1_0 = C["level"] + 1.0 * swing_size
    fib_1_618 = C["level"] + 1.618 * swing_size

    if fib_1_0 > current_price:
        targets.append(
            {
                "level": round(fib_1_0, 2),
                "method": "fib_extension_1_0",
                "distance_above_pct": round(((fib_1_0 / current_price) - 1) * 100, 1),
                "distance_in_atr": round(abs(fib_1_0 - current_price) / weekly_atr, 2) if weekly_atr and weekly_atr > 0 else None,
            }
        )

    if fib_1_618 > current_price:
        targets.append(
            {
                "level": round(fib_1_618, 2),
                "method": "fib_extension_1_618",
                "distance_above_pct": round(((fib_1_618 / current_price) - 1) * 100, 1),
                "distance_in_atr": round(abs(fib_1_618 - current_price) / weekly_atr, 2) if weekly_atr and weekly_atr > 0 else None,
            }
        )

    return targets


# ── Market Regime ───────────────────────────────────────────────────────


def calculate_regime(
    nifty_weekly_close: float,
    nifty_20wma: float,
    nifty_weekly_trend: str,
) -> str:
    """Classify market regime as 'bull', 'bear', or 'neutral'."""
    if nifty_weekly_close > nifty_20wma and nifty_weekly_trend == "uptrend":
        return "bull"
    if nifty_weekly_close < nifty_20wma and nifty_weekly_trend == "downtrend":
        return "bear"
    return "neutral"


def calculate_regime_strength(
    pct_from_20wma: float, trend_duration: int
) -> str:
    """Classify regime strength as 'strong', 'moderate', or 'weak'.

    All strict inequalities:
        strong: abs(pct) > 3.0 AND duration > 8
        weak:   abs(pct) < 2.0 OR duration < 4
        moderate: everything else
    """
    abs_pct = abs(pct_from_20wma)
    if abs_pct > 3.0 and trend_duration > 8:
        return "strong"
    if abs_pct < 2.0 or trend_duration < 4:
        return "weak"
    return "moderate"


# ── Significant Events ──────────────────────────────────────────────────


def detect_significant_events(
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
    lookback: int = 20,
) -> list[dict]:
    """Detect volume spikes >2x 20-day average in last `lookback` trading days.

    Each event includes weekly_candle_confirmed: whether the weekly candle
    containing this event closed in the same direction (close vs open) as
    the daily move.

    Returns:
        List of event dicts sorted by volume_multiplier descending.
    """
    if len(df_daily) < 22:
        return []

    avg_vol = float(df_daily["Volume"].iloc[-lookback - 20 : -lookback].mean()) if len(df_daily) > lookback + 20 else float(df_daily["Volume"].iloc[:-lookback].mean()) if len(df_daily) > lookback else float(df_daily["Volume"].mean())
    if avg_vol <= 0:
        return []

    scan = df_daily.tail(lookback)
    events: list[dict] = []

    for i in range(len(scan)):
        row = scan.iloc[i]
        vol = float(row["Volume"])
        vol_mult = vol / avg_vol

        if vol_mult <= 2.0:
            continue

        o, c = float(row["Open"]), float(row["Close"])
        price_change_pct = round(((c - o) / o) * 100, 1) if o > 0 else 0.0
        daily_direction = "up" if c >= o else "down"

        # Find containing weekly candle
        event_date = scan.index[i]
        weekly_confirmed = _is_weekly_confirmed(event_date, daily_direction, df_weekly)

        events.append(
            {
                "date": event_date.strftime("%Y-%m-%d") if hasattr(event_date, "strftime") else str(event_date),
                "type": "volume_spike",
                "price_close": round(c, 2),
                "price_change_pct": price_change_pct,
                "volume": int(vol),
                "volume_multiplier": round(vol_mult, 2),
                "weekly_candle_confirmed": weekly_confirmed,
            }
        )

    events.sort(key=lambda e: e["volume_multiplier"], reverse=True)
    return events


def _is_weekly_confirmed(
    event_date: pd.Timestamp, daily_direction: str, df_weekly: pd.DataFrame
) -> bool:
    """Check if the weekly candle containing event_date closed in the same
    direction as the daily event.

    Weekly direction = close vs open (NOT vs previous week's close).
    """
    if df_weekly.empty:
        return False

    for idx in range(len(df_weekly)):
        wc = df_weekly.iloc[idx]
        # Weekly candle date is period-end (Friday). Daily dates within
        # the same week fall on or before this Friday.
        week_end = df_weekly.index[idx]
        week_start = week_end - pd.Timedelta(days=6)  # approximate Mon

        if week_start <= event_date <= week_end:
            w_open, w_close = float(wc["Open"]), float(wc["Close"])
            if w_close > w_open and daily_direction == "up":
                return True
            elif w_close < w_open and daily_direction == "down":
                return True
            else:
                return False

    return False


# ── Daily Alignment ────────────────────────────────────────────────────


def compute_daily_alignment(weekly_trend: str, daily_trend: str) -> str:
    """Classify the relationship between weekly and daily trend.

    Returns a descriptive string instead of a boolean conflict flag,
    giving the AI proper context for weeks-to-months swing trading.
    """
    if "insufficient_data" in (weekly_trend, daily_trend):
        return "insufficient_data"
    if weekly_trend == "sideways":
        return "range_bound"
    if weekly_trend == "uptrend":
        if daily_trend == "uptrend":
            return "confirming"
        return "pullback_within_uptrend"      # daily down or sideways
    if weekly_trend == "downtrend":
        if daily_trend == "downtrend":
            return "confirming"
        return "counter_trend_bounce"         # daily up or sideways
    return "range_bound"


# ── Flag Derivation ─────────────────────────────────────────────────────


def derive_flags(
    weekly_trend: str,
    daily_trend: str,
    monthly_trend: str,
    has_position: bool,
    pnl_pct: Optional[float],
    index_change_1m: Optional[float],
    distance_to_support_pct: Optional[float],
    support_touches: Optional[int],
    significant_events: list[dict],
    weekly_candle_count: int,
    volume_avg_20d: Optional[float],
    near_round_number: Optional[float] = None,
    market_regime: Optional[str] = None,
) -> dict:
    """Derive 9 boolean flags per the spec (Phase 3A).

    breakeven_triggered removed — replaced by breakeven_context.breakeven_eligible
    inside safety_line_candidates.

    Args:
        weekly_trend: classify_trend output for weekly.
        daily_trend: classify_trend output for daily.
        monthly_trend: classify_trend output for monthly.
        has_position: Whether user holds a position.
        pnl_pct: Position P&L percentage (None if no position).
        index_change_1m: Market index 1-month change.
        distance_to_support_pct: distance_to_nearest_support output.
        support_touches: Touch count of nearest support level.
        significant_events: Output of detect_significant_events.
        weekly_candle_count: Number of weekly candles available.
        volume_avg_20d: 20-day average daily volume.
        near_round_number: Round number near price (None if not near).

    Returns:
        Dict with 9 boolean flags.
    """
    # in_noise_zone: has position AND abs(pnl_pct) < 5
    in_noise_zone = has_position and pnl_pct is not None and abs(pnl_pct) < 5

    # market_bearish: regime is bear/correcting, or index down > 3%
    market_bearish = market_regime in ("bear", "correcting") or (index_change_1m or 0) < -3

    # high_volume_event: any event in last 10 trading days
    high_volume_event = len(significant_events) > 0

    # near_major_support: distance not null AND < 3% AND touches >= 3
    near_major_support = (
        distance_to_support_pct is not None
        and distance_to_support_pct < 3
        and support_touches is not None
        and support_touches >= 3
    )

    # full_alignment: all three timeframes same non-sideways direction
    non_directional = {"sideways", "insufficient_data"}
    full_alignment = (
        weekly_trend == daily_trend == monthly_trend
        and weekly_trend not in non_directional
    )

    # near_round_number
    near_round = near_round_number is not None

    # limited_history: less than 52 weekly candles
    limited_history = weekly_candle_count < 52

    # low_liquidity: 20-day avg volume < 50,000
    low_liquidity = volume_avg_20d is not None and volume_avg_20d < 50_000

    return {
        "in_noise_zone": in_noise_zone,
        "market_bearish": market_bearish,
        "high_volume_event": high_volume_event,
        "near_major_support": near_major_support,
        "full_alignment": full_alignment,
        "near_round_number": near_round,
        "limited_history": limited_history,
        "low_liquidity": low_liquidity,
    }


# ── Level Significance ─────────────────────────────────────────────────


def compute_level_significance(
    level: float,
    completed_weekly: pd.DataFrame,
    weekly_atr: float | None,
) -> dict:
    """Count how many completed weekly candles closed near *level*.

    Returns
    -------
    {"consolidation_weeks": int, "interaction_span_weeks": int}

    *consolidation_weeks* — number of completed weekly candles whose close
    falls within the zone around *level*.

    *interaction_span_weeks* — weeks between the first and last such candle.
    Zero when fewer than two candles are in the zone.

    The zone width is ``min(weekly_atr, level * 0.05)`` to prevent absurdly
    wide zones on volatile stocks.
    """
    empty = {"consolidation_weeks": 0, "interaction_span_weeks": 0}

    if (
        weekly_atr is None
        or weekly_atr <= 0
        or completed_weekly is None
        or completed_weekly.empty
        or level <= 0
    ):
        return empty

    zone_width = min(weekly_atr, level * 0.05)
    closes = completed_weekly["Close"]
    mask = (closes - level).abs() <= zone_width
    in_zone = closes[mask]

    consolidation_weeks = len(in_zone)
    if consolidation_weeks < 2:
        return {
            "consolidation_weeks": consolidation_weeks,
            "interaction_span_weeks": 0,
        }

    first_date = in_zone.index[0]
    last_date = in_zone.index[-1]
    interaction_span_weeks = int((last_date - first_date).days // 7)

    return {
        "consolidation_weeks": consolidation_weeks,
        "interaction_span_weeks": interaction_span_weeks,
    }


def compute_bounce_volume_ratio(
    df_weekly: pd.DataFrame,
    bounce_date: str | None,
) -> float | None:
    """Ratio of bounce-week volume to its trailing 20-week average volume.

    Parameters
    ----------
    df_weekly : pd.DataFrame
        Weekly OHLCV DataFrame with DatetimeIndex.
    bounce_date : str | None
        Date string (YYYY-MM-DD) of the bounce candle.

    Returns
    -------
    float | None
        Rounded ratio (2 decimals), or ``None`` if data is missing.
    """
    if bounce_date is None or df_weekly is None or df_weekly.empty:
        return None
    if "Volume" not in df_weekly.columns:
        return None

    try:
        target = pd.Timestamp(bounce_date).normalize()
    except (ValueError, TypeError):
        return None

    # Find the row matching bounce_date
    idx = df_weekly.index.normalize()
    matches = idx == target
    if not matches.any():
        return None

    iloc_pos = int(matches.argmax())
    # Verify it's a true match (argmax returns 0 for all-False)
    if not matches[iloc_pos]:
        return None

    bounce_vol = float(df_weekly.iloc[iloc_pos]["Volume"])
    if bounce_vol <= 0:
        return None

    # Trailing 20-week average (up to 20 candles BEFORE the bounce)
    start = max(0, iloc_pos - 20)
    if start == iloc_pos:
        return None  # no preceding data
    trailing = df_weekly.iloc[start:iloc_pos]["Volume"]
    if trailing.empty:
        return None
    avg_vol = float(trailing.mean())
    if avg_vol <= 0:
        return None

    return round(bounce_vol / avg_vol, 2)


# ── Safety Line Candidates ──────────────────────────────────────────────


def build_safety_line_candidates(
    current_price: float,
    weekly_swing_lows: list[dict],
    daily_swing_lows: list[dict],
    weekly_ma_20: Optional[float],
    daily_atr: float,
    weekly_atr: float,
    df_weekly: Optional["pd.DataFrame"] = None,
    has_position: bool = False,
    entry_price: Optional[float] = None,
    fibonacci: Optional[list] = None,
    weekly_swing_highs: Optional[list] = None,
) -> dict:
    """Build safety line candidate dict for the JSON payload (Phase 3A).

    Returns:
        {
            "chandelier": {basis, highest_weekly_close, ..., levels: {2x_atr, 2.5x_atr, 3x_atr}},
            "weekly_swing_lows": [{level, date, touches, is_current_week, nearby_swing_highs}, ...],
            "daily_swing_lows": [{level, date, touches}, ...],
            "ma_trailing": {ema_10w_close, ema_20w_close},
            "atr_based": {daily_atr, weekly_atr, levels: [...]},
            "fibonacci": [],
            "breakeven_context": {...}  # only when has_position=True and entry_price set
        }
    """

    def _distance_below_pct(level: float) -> float:
        return round(abs(current_price - level) / current_price * 100, 1) if current_price > 0 else 0.0

    def _distance_in_atr(level: float):
        if not weekly_atr or weekly_atr <= 0:
            return None
        return round(abs(current_price - level) / weekly_atr, 2)

    def _apply_buffer(raw_level: float) -> tuple[float, float]:
        """Return (buffered_level, structural_level). Buffer = 1× daily ATR."""
        if daily_atr and daily_atr > 0:
            return round(raw_level - daily_atr, 2), round(raw_level, 2)
        return round(raw_level, 2), round(raw_level, 2)

    # ── Chandelier block ────────────────────────────────────────────────
    chandelier = None
    ma_trailing = None
    if df_weekly is not None and len(df_weekly) > 1 and weekly_atr and weekly_atr > 0:
        # Exclude the last (current running) week — use only completed candles
        completed_weekly = df_weekly.iloc[:-1]
        lookback = completed_weekly.tail(10)
        highest_weekly_close = float(lookback["Close"].max())
        highest_weekly_close_date = str(lookback["Close"].idxmax().date())
        ch_levels = {}
        for label, mult in [("2x_atr", 2.0), ("2.5x_atr", 2.5), ("3x_atr", 3.0)]:
            raw = highest_weekly_close - mult * weekly_atr
            buffered, structural = _apply_buffer(raw)
            ch_levels[label] = {
                "level": buffered,
                "structural_level": structural,
                "distance_below_pct": _distance_below_pct(buffered),
                "distance_in_atr": _distance_in_atr(buffered),
            }
        chandelier = {
            "timeframe": "weekly",
            "basis": "highest_weekly_close",
            "highest_weekly_close": round(highest_weekly_close, 2),
            "highest_weekly_close_date": highest_weekly_close_date,
            "lookback_weeks": len(lookback),
            "weekly_atr": round(weekly_atr, 2),
            "levels": ch_levels,
        }

        # ── ma_trailing block (EMA 10w + EMA 20w on completed weekly closes) ──
        completed_closes = completed_weekly["Close"]
        ema_10w = float(completed_closes.ewm(span=10, adjust=False).mean().iloc[-1])
        ema_20w = float(completed_closes.ewm(span=20, adjust=False).mean().iloc[-1])
        ema_10w_buf, ema_10w_struct = _apply_buffer(ema_10w)
        ema_20w_buf, ema_20w_struct = _apply_buffer(ema_20w)
        ma_trailing = {
            "timeframe": "weekly",
            "ema_10w_level": ema_10w_buf,
            "ema_10w_structural_level": ema_10w_struct,
            "ema_10w_distance_below_pct": _distance_below_pct(ema_10w_buf),
            "ema_10w_distance_in_atr": _distance_in_atr(ema_10w_buf),
            "ema_20w_level": ema_20w_buf,
            "ema_20w_structural_level": ema_20w_struct,
            "ema_20w_distance_below_pct": _distance_below_pct(ema_20w_buf),
            "ema_20w_distance_in_atr": _distance_in_atr(ema_20w_buf),
        }

    # ── Completed weekly candles (for level significance) ───────────────
    _completed = (
        df_weekly.iloc[:-1]
        if df_weekly is not None and len(df_weekly) > 1
        else pd.DataFrame()
    )

    # ── Weekly swing lows — add is_current_week + nearby_swing_highs ────
    current_week_date = str(df_weekly.index[-1].date()) if df_weekly is not None and len(df_weekly) > 0 else None
    _highs = weekly_swing_highs or []
    w_lows = []
    for sw in weekly_swing_lows:
        if sw["level"] < current_price:
            # Find swing highs older than this swing low and within 1× weekly ATR
            nearby = []
            if weekly_atr:
                sw_date = sw.get("date", "")
                for sh in _highs:
                    if sh.get("date", "") < sw_date:
                        dist = round(abs(sh["level"] - sw["level"]), 2)
                        if dist <= weekly_atr:
                            nearby.append({"level": round(sh["level"], 2), "date": sh["date"], "distance": dist})
                nearby.sort(key=lambda x: x["distance"])
            buffered, structural = _apply_buffer(sw["level"])
            sig = compute_level_significance(structural, _completed, weekly_atr)
            _td = sw.get("touch_dates", [])
            _bounce_dt = _td[0] if _td else sw.get("date")
            w_lows.append({
                "timeframe": "weekly",
                "level": buffered,
                "structural_level": structural,
                "date": sw.get("date"),
                "touches": sw.get("touches", 0),
                "distance_below_pct": _distance_below_pct(buffered),
                "distance_in_atr": _distance_in_atr(buffered),
                "is_current_week": sw.get("date") == current_week_date,
                "nearby_swing_highs": nearby,
                "bounce_volume_ratio": compute_bounce_volume_ratio(df_weekly, _bounce_dt),
                **sig,
            })

    # ── Daily swing lows (top 3 closest, below current price) ───────────
    d_lows_below = [sw for sw in daily_swing_lows if sw["level"] < current_price]
    d_lows_below.sort(key=lambda s: current_price - s["level"])
    d_lows = []
    for sw in d_lows_below[:3]:
        buffered, structural = _apply_buffer(sw["level"])
        d_lows.append({
            "timeframe": "daily",
            "level": buffered,
            "structural_level": structural,
            "date": sw.get("date"),
            "touches": sw.get("touches", 1),  # daily swings don't have touch counting — default 1
            "distance_below_pct": _distance_below_pct(buffered),
            "distance_in_atr": _distance_in_atr(buffered),
        })

    # ── ATR-based levels (weekly only — daily ATR too tight for swing holds) ─
    atr_based = None
    if weekly_atr and weekly_atr > 0:
        atr_raw = current_price - 2.0 * weekly_atr
        atr_buf, atr_struct = _apply_buffer(atr_raw)
        atr_based = {
            "timeframe": "weekly",
            "weekly_atr": round(weekly_atr, 2),
            "levels": [
                {
                    "label": "2x_weekly_atr",
                    "level": atr_buf,
                    "structural_level": atr_struct,
                    "distance_below_pct": _distance_below_pct(atr_buf),
                    "distance_in_atr": _distance_in_atr(atr_buf),
                },
            ],
        }

    # ── breakeven_context (only when position exists) ────────────────────
    breakeven_context = None
    if has_position and entry_price is not None and weekly_atr > 0:
        profit_in_atr_units = round((current_price - entry_price) / weekly_atr, 2)
        atr_breakeven_threshold = 3.0
        breakeven_eligible = profit_in_atr_units >= atr_breakeven_threshold
        profit_pct = round((current_price - entry_price) / entry_price * 100, 2)
        required_move = round(atr_breakeven_threshold * weekly_atr, 2)
        required_pct = round(required_move / entry_price * 100, 1)
        if breakeven_eligible:
            reason = (
                f"Profit ({profit_in_atr_units} ATR) has reached {atr_breakeven_threshold}x "
                f"weekly ATR — breakeven stop eligible."
            )
        else:
            reason = (
                f"Profit ({profit_in_atr_units} ATR) has not yet reached "
                f"{atr_breakeven_threshold}x weekly ATR "
                f"(₹{required_move} / {required_pct}%)"
            )
        breakeven_context = {
            "timeframe": "position",
            "entry_price": entry_price,
            "structural_level": entry_price,   # no buffer — psychological level
            "current_price": current_price,
            "unrealized_pnl_pct": profit_pct,
            "distance_below_pct": _distance_below_pct(entry_price),
            "distance_in_atr": _distance_in_atr(entry_price),
            "weekly_atr": round(weekly_atr, 2),
            "profit_in_atr_units": profit_in_atr_units,
            "atr_breakeven_threshold": atr_breakeven_threshold,
            "breakeven_eligible": breakeven_eligible,
            "reason": reason,
        }

    # ── Resistance turned support — former ceilings now below price ────
    from datetime import date as _date

    def _weeks_ago(date_str: str | None) -> int | None:
        if not date_str:
            return None
        try:
            return (_date.today() - _date.fromisoformat(date_str)).days // 7
        except (ValueError, TypeError):
            return None

    _all_highs = weekly_swing_highs or []
    rts_candidates = []
    for sh in _all_highs:
        if sh["level"] >= current_price:
            continue  # still above price, not turned support
        if weekly_atr and weekly_atr > 0:
            dist_atr = (current_price - sh["level"]) / weekly_atr
            if dist_atr > 6:
                continue  # too far for any practical stop
        buffered, structural = _apply_buffer(sh["level"])
        sh_date = sh.get("date")
        sig = compute_level_significance(structural, _completed, weekly_atr)
        rts_candidates.append({
            "timeframe": "weekly",
            "level": buffered,
            "structural_level": structural,
            "original_date": sh_date,
            "touches_as_resistance": sh.get("touches", 1),
            "weeks_ago": _weeks_ago(sh_date),
            "distance_below_pct": _distance_below_pct(buffered),
            "distance_in_atr": _distance_in_atr(buffered),
            **sig,
        })
    rts_candidates.sort(key=lambda x: x["level"], reverse=True)  # nearest first
    rts_candidates = rts_candidates[:3]  # max 3

    result: dict = {
        "chandelier": chandelier,
        "weekly_swing_lows": w_lows,
        "daily_swing_lows": d_lows,
        "resistance_turned_support": rts_candidates,
        "ma_trailing": ma_trailing,
        "atr_based": atr_based,
        "fibonacci": fibonacci or [],
    }
    if breakeven_context is not None:
        result["breakeven_context"] = breakeven_context
    return result


# ── Noise Floor Filter ──────────────────────────────────────────────────

_NOISE_FLOOR_ATR = 1.0


def apply_noise_floor_filter(
    safety_line_candidates: dict,
    *,
    threshold: float = _NOISE_FLOOR_ATR,
) -> tuple[dict, list[dict]]:
    """Remove safety-line candidates whose distance from price < *threshold* weekly ATR.

    Uses the pre-computed ``distance_in_atr`` field on each candidate.

    Returns
    -------
    (filtered_candidates, removed)
        *filtered_candidates* is a **new** dict with the same keys; lists are
        filtered in-place copies, dict-type categories have sub-entries pruned.
        *removed* is a list of ``{level, method, distance_atr, reason}`` dicts.
    """
    removed: list[dict] = []

    def _below(distance_in_atr) -> bool:
        """True when the candidate is below the noise floor."""
        return distance_in_atr is not None and distance_in_atr < threshold

    def _record(level, method, distance_atr):
        removed.append({
            "level": level,
            "method": method,
            "distance_atr": distance_atr,
            "reason": "below_noise_floor",
        })

    out: dict = {}

    # --- list-type categories: weekly_swing_lows, daily_swing_lows,
    #     resistance_turned_support, fibonacci, trendline_support ---
    for key in ("weekly_swing_lows", "daily_swing_lows",
                "resistance_turned_support", "fibonacci",
                "trendline_support"):
        items = safety_line_candidates.get(key, [])
        kept = []
        for item in items:
            d = item.get("distance_in_atr")
            if _below(d):
                _record(item.get("level"), key, d)
            else:
                kept.append(item)
        out[key] = kept

    # --- chandelier (dict with nested levels dict) ---
    ch = safety_line_candidates.get("chandelier")
    if ch is not None:
        levels = ch.get("levels", {})
        filtered_levels = {}
        for label, entry in levels.items():
            d = entry.get("distance_in_atr")
            if _below(d):
                _record(entry.get("level"), f"chandelier_{label}", d)
            else:
                filtered_levels[label] = entry
        ch_copy = {k: v for k, v in ch.items() if k != "levels"}
        ch_copy["levels"] = filtered_levels
        out["chandelier"] = ch_copy if filtered_levels else None
    else:
        out["chandelier"] = None

    # --- ma_trailing (single dict with ema_10w / ema_20w pairs) ---
    ma = safety_line_candidates.get("ma_trailing")
    if ma is not None:
        ma_copy = dict(ma)
        for suffix in ("10w", "20w"):
            d = ma.get(f"ema_{suffix}_distance_in_atr")
            if _below(d):
                _record(ma.get(f"ema_{suffix}_level"), f"ma_trailing_{suffix}", d)
                ma_copy[f"ema_{suffix}_level"] = None
                ma_copy[f"ema_{suffix}_structural_level"] = None
        # If both EMAs are nulled out, drop the whole block
        if ma_copy.get("ema_10w_level") is None and ma_copy.get("ema_20w_level") is None:
            out["ma_trailing"] = None
        else:
            out["ma_trailing"] = ma_copy
    else:
        out["ma_trailing"] = None

    # --- atr_based (dict with nested levels list) ---
    ab = safety_line_candidates.get("atr_based")
    if ab is not None:
        levels = ab.get("levels", [])
        filtered_levels = []
        for entry in levels:
            d = entry.get("distance_in_atr")
            if _below(d):
                _record(entry.get("level"), f"atr_based_{entry.get('label', '')}", d)
            else:
                filtered_levels.append(entry)
        ab_copy = {k: v for k, v in ab.items() if k != "levels"}
        ab_copy["levels"] = filtered_levels
        out["atr_based"] = ab_copy
    else:
        out["atr_based"] = None

    # --- breakeven_context (pass-through — never filtered) ---
    if "breakeven_context" in safety_line_candidates:
        out["breakeven_context"] = safety_line_candidates["breakeven_context"]

    return out, removed


# ── R/R Viability Filter ───────────────────────────────────────────────

_MIN_RR = 1.5


def apply_rr_viability_filter(
    safety_line_candidates: dict,
    target_candidates: list[dict],
    entry_price: float,
    *,
    min_rr: float = _MIN_RR,
) -> tuple[dict, list[dict]]:
    """Remove safety-line candidates where no target achieves R/R ≥ *min_rr*.

    If *target_candidates* is empty (e.g. stock at all-time high), the filter
    is skipped entirely and all candidates pass through unchanged.

    Returns
    -------
    (filtered_candidates, removed)
        *filtered_candidates* is a **new** dict with the same keys.
        *removed* is a list of ``{level, method, best_rr, reason}`` dicts.
    """
    # No targets → skip filter (AI handles open-ended R/R assessment)
    if not target_candidates:
        return safety_line_candidates, []

    removed: list[dict] = []

    def _best_rr_for_level(level: float) -> tuple[str, float | None]:
        """Return (reason, best_rr) for a candidate at *level*.

        Returns
        -------
        ("ok", best_rr)            – candidate survives
        ("stop_above_entry", None)  – risk ≤ 0
        ("no_target_achieves_1.5_rr", best_rr) – best R/R below threshold
        """
        risk = entry_price - level
        if risk <= 0:
            return "stop_above_entry", None

        best = None
        for t in target_candidates:
            reward = t["level"] - entry_price
            if reward > 0:
                rr = reward / risk
                if best is None or rr > best:
                    best = rr

        if best is not None and best >= min_rr:
            return "ok", round(best, 2)

        return "no_target_achieves_1.5_rr", round(best, 2) if best is not None else None

    def _record(level, method, best_rr, reason):
        entry = {"level": level, "method": method, "reason": reason}
        if best_rr is not None:
            entry["best_rr"] = best_rr
        removed.append(entry)

    out: dict = {}

    # --- list-type categories ---
    for key in ("weekly_swing_lows", "daily_swing_lows",
                "resistance_turned_support", "fibonacci",
                "trendline_support"):
        items = safety_line_candidates.get(key, [])
        kept = []
        for item in items:
            lvl = item.get("level")
            if lvl is None:
                kept.append(item)
                continue
            reason, best_rr = _best_rr_for_level(lvl)
            if reason == "ok":
                item_copy = dict(item)
                item_copy["best_achievable_rr"] = best_rr
                kept.append(item_copy)
            else:
                _record(lvl, key, best_rr, reason)
        out[key] = kept

    # --- chandelier ---
    ch = safety_line_candidates.get("chandelier")
    if ch is not None:
        levels = ch.get("levels", {})
        filtered_levels = {}
        for label, entry in levels.items():
            lvl = entry.get("level")
            if lvl is None:
                filtered_levels[label] = entry
                continue
            reason, best_rr = _best_rr_for_level(lvl)
            if reason == "ok":
                entry_copy = dict(entry)
                entry_copy["best_achievable_rr"] = best_rr
                filtered_levels[label] = entry_copy
            else:
                _record(lvl, f"chandelier_{label}", best_rr, reason)
        ch_copy = {k: v for k, v in ch.items() if k != "levels"}
        ch_copy["levels"] = filtered_levels
        out["chandelier"] = ch_copy if filtered_levels else None
    else:
        out["chandelier"] = None

    # --- ma_trailing ---
    ma = safety_line_candidates.get("ma_trailing")
    if ma is not None:
        ma_copy = dict(ma)
        for suffix in ("10w", "20w"):
            lvl = ma.get(f"ema_{suffix}_level")
            if lvl is None:
                continue
            reason, best_rr = _best_rr_for_level(lvl)
            if reason == "ok":
                ma_copy[f"ema_{suffix}_best_achievable_rr"] = best_rr
            else:
                _record(lvl, f"ma_trailing_{suffix}", best_rr, reason)
                ma_copy[f"ema_{suffix}_level"] = None
                ma_copy[f"ema_{suffix}_structural_level"] = None
        if ma_copy.get("ema_10w_level") is None and ma_copy.get("ema_20w_level") is None:
            out["ma_trailing"] = None
        else:
            out["ma_trailing"] = ma_copy
    else:
        out["ma_trailing"] = None

    # --- atr_based ---
    ab = safety_line_candidates.get("atr_based")
    if ab is not None:
        levels = ab.get("levels", [])
        filtered_levels = []
        for entry in levels:
            lvl = entry.get("level")
            if lvl is None:
                filtered_levels.append(entry)
                continue
            reason, best_rr = _best_rr_for_level(lvl)
            if reason == "ok":
                entry_copy = dict(entry)
                entry_copy["best_achievable_rr"] = best_rr
                filtered_levels.append(entry_copy)
            else:
                _record(lvl, f"atr_based_{entry.get('label', '')}", best_rr, reason)
        ab_copy = {k: v for k, v in ab.items() if k != "levels"}
        ab_copy["levels"] = filtered_levels
        out["atr_based"] = ab_copy
    else:
        out["atr_based"] = None

    # --- breakeven_context (pass-through — never filtered) ---
    if "breakeven_context" in safety_line_candidates:
        out["breakeven_context"] = safety_line_candidates["breakeven_context"]

    return out, removed


# ── Profit Protection Floor Filter ─────────────────────────────────────

_PROFIT_ATR_THRESHOLD = 3.0
_PROFIT_PROTECTION_PCT = 0.50


def apply_profit_protection_filter(
    safety_line_candidates: dict,
    *,
    entry_price: float | None,
    current_price: float,
    weekly_atr: float | None,
    profit_atr_threshold: float = _PROFIT_ATR_THRESHOLD,
    protection_pct: float = _PROFIT_PROTECTION_PCT,
) -> tuple[dict, list[dict], dict | None]:
    """Remove candidates that fail to protect at least 50 % of unrealised gain.

    Activates only when the user holds a position with
    ``profit_in_atr > profit_atr_threshold`` (default 3.0).

    Safety valve: if the filter removes *all* remaining candidates, the single
    candidate with the highest level is kept so the AI always has at least one
    option.

    Returns
    -------
    (filtered_candidates, removed, meta)
        *meta* is ``None`` when the filter is inactive, otherwise a dict with
        ``protection_floor`` and ``profit_in_atr``.
    """
    # --- guard: filter inactive ---
    if (
        entry_price is None
        or weekly_atr is None
        or weekly_atr <= 0
        or current_price <= entry_price
    ):
        return safety_line_candidates, [], None

    profit_in_atr = (current_price - entry_price) / weekly_atr
    if profit_in_atr <= profit_atr_threshold:
        return safety_line_candidates, [], None

    unrealised = current_price - entry_price
    protection_floor = entry_price + unrealised * protection_pct
    protection_floor = round(protection_floor, 2)

    meta = {
        "protection_floor": protection_floor,
        "profit_in_atr": round(profit_in_atr, 2),
    }

    removed: list[dict] = []

    def _below_floor(level: float | None) -> bool:
        return level is not None and level < protection_floor

    def _record(level, method):
        removed.append({
            "level": level,
            "method": method,
            "protection_floor": protection_floor,
            "reason": "below_profit_protection_floor",
        })

    # Collect all (category_path, level) pairs for safety-valve tracking.
    # category_path is a tuple so we can locate the candidate later.
    all_levels: list[tuple[str, float]] = []
    out: dict = {}

    # --- list-type categories ---
    for key in ("weekly_swing_lows", "daily_swing_lows",
                "resistance_turned_support", "fibonacci",
                "trendline_support"):
        items = safety_line_candidates.get(key, [])
        kept = []
        for item in items:
            lvl = item.get("level")
            if _below_floor(lvl):
                _record(lvl, key)
            else:
                kept.append(item)
            if lvl is not None:
                all_levels.append((key, lvl))
        out[key] = kept

    # --- chandelier ---
    ch = safety_line_candidates.get("chandelier")
    if ch is not None:
        levels = ch.get("levels", {})
        filtered_levels = {}
        for label, entry in levels.items():
            lvl = entry.get("level")
            if _below_floor(lvl):
                _record(lvl, f"chandelier_{label}")
            else:
                filtered_levels[label] = entry
            if lvl is not None:
                all_levels.append((f"chandelier_{label}", lvl))
        ch_copy = {k: v for k, v in ch.items() if k != "levels"}
        ch_copy["levels"] = filtered_levels
        out["chandelier"] = ch_copy if filtered_levels else None
    else:
        out["chandelier"] = None

    # --- ma_trailing ---
    ma = safety_line_candidates.get("ma_trailing")
    if ma is not None:
        ma_copy = dict(ma)
        for suffix in ("10w", "20w"):
            lvl = ma.get(f"ema_{suffix}_level")
            if _below_floor(lvl):
                _record(lvl, f"ma_trailing_{suffix}")
                ma_copy[f"ema_{suffix}_level"] = None
                ma_copy[f"ema_{suffix}_structural_level"] = None
                # Preserve best_achievable_rr key from previous filter if present
                if f"ema_{suffix}_best_achievable_rr" in ma_copy:
                    del ma_copy[f"ema_{suffix}_best_achievable_rr"]
            if lvl is not None:
                all_levels.append((f"ma_trailing_{suffix}", lvl))
        if ma_copy.get("ema_10w_level") is None and ma_copy.get("ema_20w_level") is None:
            out["ma_trailing"] = None
        else:
            out["ma_trailing"] = ma_copy
    else:
        out["ma_trailing"] = None

    # --- atr_based ---
    ab = safety_line_candidates.get("atr_based")
    if ab is not None:
        levels_list = ab.get("levels", [])
        filtered_levels = []
        for entry in levels_list:
            lvl = entry.get("level")
            if _below_floor(lvl):
                _record(lvl, f"atr_based_{entry.get('label', '')}")
            else:
                filtered_levels.append(entry)
            if lvl is not None:
                all_levels.append((f"atr_based_{entry.get('label', '')}", lvl))
        ab_copy = {k: v for k, v in ab.items() if k != "levels"}
        ab_copy["levels"] = filtered_levels
        out["atr_based"] = ab_copy
    else:
        out["atr_based"] = None

    # --- breakeven_context (pass-through) ---
    if "breakeven_context" in safety_line_candidates:
        out["breakeven_context"] = safety_line_candidates["breakeven_context"]

    # --- Safety valve: if ALL candidates removed, keep the highest one ---
    def _count_surviving(d: dict) -> int:
        n = 0
        for key in ("weekly_swing_lows", "daily_swing_lows",
                     "resistance_turned_support", "fibonacci",
                     "trendline_support"):
            n += len(d.get(key, []))
        ch_ = d.get("chandelier")
        if ch_ and ch_.get("levels"):
            n += len(ch_["levels"])
        ma_ = d.get("ma_trailing")
        if ma_:
            if ma_.get("ema_10w_level") is not None:
                n += 1
            if ma_.get("ema_20w_level") is not None:
                n += 1
        ab_ = d.get("atr_based")
        if ab_ and ab_.get("levels"):
            n += len(ab_["levels"])
        return n

    if removed and _count_surviving(out) == 0 and all_levels:
        # Re-insert the candidate with the highest level from the original input
        best_method, best_level = max(all_levels, key=lambda x: x[1])

        # Remove it from the removed list
        removed = [r for r in removed if not (r["level"] == best_level and r["method"] == best_method)]
        removed.append({
            "level": best_level,
            "method": best_method,
            "reason": "safety_valve_kept",
        })

        # Re-insert into the appropriate category from the original candidates
        _reinsert_candidate(out, safety_line_candidates, best_method, best_level)

    return out, removed, meta


def _reinsert_candidate(
    out: dict, original: dict, method: str, level: float,
) -> None:
    """Re-insert a single candidate (by method+level) from *original* into *out*."""
    # List-type categories
    for key in ("weekly_swing_lows", "daily_swing_lows",
                "resistance_turned_support", "fibonacci",
                "trendline_support"):
        if method == key:
            for item in original.get(key, []):
                if item.get("level") == level:
                    out[key].append(item)
                    return

    # Chandelier
    if method.startswith("chandelier_"):
        label = method[len("chandelier_"):]
        ch = original.get("chandelier")
        if ch:
            entry = ch.get("levels", {}).get(label)
            if entry and entry.get("level") == level:
                if out["chandelier"] is None:
                    out["chandelier"] = {k: v for k, v in ch.items() if k != "levels"}
                    out["chandelier"]["levels"] = {}
                out["chandelier"]["levels"][label] = entry
                return

    # MA trailing
    if method.startswith("ma_trailing_"):
        suffix = method[len("ma_trailing_"):]
        ma = original.get("ma_trailing")
        if ma:
            if ma.get(f"ema_{suffix}_level") == level:
                if out["ma_trailing"] is None:
                    out["ma_trailing"] = dict(ma)
                    # Null out the other suffix if it was also removed
                    other = "20w" if suffix == "10w" else "10w"
                    out["ma_trailing"][f"ema_{other}_level"] = None
                    out["ma_trailing"][f"ema_{other}_structural_level"] = None
                out["ma_trailing"][f"ema_{suffix}_level"] = ma[f"ema_{suffix}_level"]
                out["ma_trailing"][f"ema_{suffix}_structural_level"] = ma.get(f"ema_{suffix}_structural_level")
                return

    # ATR-based
    if method.startswith("atr_based_"):
        ab = original.get("atr_based")
        if ab:
            for entry in ab.get("levels", []):
                if entry.get("level") == level:
                    out["atr_based"]["levels"].append(entry)
                    return


# ── R/R Pre-Computation ─────────────────────────────────────────────────

STOP_CATEGORY = {
    "chandelier_2x": "trend_following",
    "chandelier_2.5x": "trend_following",
    "chandelier_3x": "trend_following",
    "weekly_swing_low": "structural",
    "resistance_turned_support": "structural",
    "daily_swing_low": "short_term",
    "ma_trailing_10w": "trend_following",
    "ma_trailing_20w": "trend_following",
    "trendline_support": "structural",
}


def compute_rr_scenarios(
    current_price: float,
    safety_line_candidates: dict,
    target_candidates: list,
) -> dict:
    """Pre-compute R/R ratios for each stop method against all targets.

    Returns pure arithmetic — no judgment about which stop is "best".

    The ``targets`` list is sorted by level ascending (nearest first).
    ``rr_vs_targets[i]`` corresponds exactly to ``targets[i]``.
    """
    # ── Build targets (only those above current price) ───────────────
    targets = []
    for t in target_candidates:
        reward = round(t["level"] - current_price, 2)
        if reward > 0:
            targets.append({
                "method": t.get("method", "unknown"),
                "level": t["level"],
                "reward_per_share": reward,
            })
    targets.sort(key=lambda x: x["level"])

    if not targets:
        return {"entry": current_price, "targets": [], "by_stop": []}

    # ── Extract stops in fixed structural priority order ─────────────
    raw_stops: list[tuple[str, float]] = []

    # 1. Chandelier (3 multipliers, skip individually if missing/invalid)
    ch = safety_line_candidates.get("chandelier")
    if ch and isinstance(ch.get("levels"), dict):
        for suffix, key in [("2x", "2x_atr"), ("2.5x", "2.5x_atr"), ("3x", "3x_atr")]:
            entry = ch["levels"].get(key)
            if entry and entry.get("level") is not None:
                raw_stops.append((f"chandelier_{suffix}", entry["level"]))

    # 2. Weekly swing lows (top 2)
    wsl = safety_line_candidates.get("weekly_swing_lows", [])
    for sl in wsl[:2]:
        raw_stops.append(("weekly_swing_low", sl["level"]))

    # 3. Resistance turned support (up to 2)
    rts = safety_line_candidates.get("resistance_turned_support", [])
    for entry in rts[:2]:
        level = entry.get("level")
        if level is not None and level < current_price:
            raw_stops.append(("resistance_turned_support", level))

    # 4. Daily swing lows (top 1)
    dsl = safety_line_candidates.get("daily_swing_lows", [])
    if dsl:
        raw_stops.append(("daily_swing_low", dsl[0]["level"]))

    # 4b. Trendline support (top 1)
    tsl = safety_line_candidates.get("trendline_support", [])
    if tsl:
        raw_stops.append(("trendline_support", tsl[0]["level"]))

    # 5. MA trailing (10w + 20w, skip individually if missing)
    ma = safety_line_candidates.get("ma_trailing")
    if ma:
        for suffix, key in [("10w", "ema_10w_level"), ("20w", "ema_20w_level")]:
            val = ma.get(key)
            if val is not None:
                raw_stops.append((f"ma_trailing_{suffix}", val))

    # ── Build by_stop entries (skip stops at or above current price) ─
    by_stop = []
    for method, level in raw_stops:
        risk = round(current_price - level, 2)
        if risk <= 0:
            continue
        rr_vs_targets = [round(t["reward_per_share"] / risk, 2) for t in targets]
        by_stop.append({
            "method": method,
            "category": STOP_CATEGORY.get(method, "unknown"),
            "level": level,
            "risk_per_share": risk,
            "rr_vs_targets": rr_vs_targets,
        })

    return {"entry": current_price, "targets": targets, "by_stop": by_stop}


# ── Current Week Enrichment ──────────────────────────────────────────────

def enrich_current_week_so_far(
    current_week: dict,
    prev_week: Optional[dict],
    weekly_atr: Optional[float],
    week_start_date: "date",
    today: "date",
) -> None:
    """Add 5 computed fields to current_week_so_far dict (mutates in place).

    Fields added:
        trading_days_elapsed    — weekdays from week_start_date to today (inclusive)
        change_from_prev_close_pct — (close - prev_close) / prev_close * 100
        gap_from_prev_close_pct    — (open  - prev_close) / prev_close * 100
        range_vs_weekly_atr_pct    — (high - low) / weekly_atr * 100
        volume_vs_prev_week_pct    — volume / prev_volume * 100

    Returns None for any field where the input is missing or zero.
    """
    from datetime import timedelta

    # 1. Trading days elapsed (count weekdays Mon-Fri inclusive)
    days = 0
    d = week_start_date
    while d <= today:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    current_week["trading_days_elapsed"] = max(days, 1)

    # 2. Change from previous week's close
    prev_close = prev_week.get("close") if prev_week else None
    if prev_close and prev_close > 0:
        current_week["change_from_prev_close_pct"] = round(
            (current_week["close"] - prev_close) / prev_close * 100, 2
        )
    else:
        current_week["change_from_prev_close_pct"] = None

    # 3. Gap from previous week's close (based on this week's open)
    if prev_close and prev_close > 0:
        current_week["gap_from_prev_close_pct"] = round(
            (current_week["open"] - prev_close) / prev_close * 100, 2
        )
    else:
        current_week["gap_from_prev_close_pct"] = None

    # 4. This week's range as % of weekly ATR
    if weekly_atr and weekly_atr > 0:
        week_range = current_week["high"] - current_week["low"]
        current_week["range_vs_weekly_atr_pct"] = round(
            week_range / weekly_atr * 100, 1
        )
    else:
        current_week["range_vs_weekly_atr_pct"] = None

    # 5. This week's volume as % of last week's volume
    prev_volume = prev_week.get("volume") if prev_week else None
    if prev_volume and prev_volume > 0:
        current_week["volume_vs_prev_week_pct"] = round(
            current_week["volume"] / prev_volume * 100, 1
        )
    else:
        current_week["volume_vs_prev_week_pct"] = None


# ── Current Week vs Setup Inference ──────────────────────────────────────

def compute_current_week_vs_setup(
    current_week: Optional[dict],
    safety_line_candidates: dict,
    swing_highs: list,
    weekly_trend: str,
    weekly_atr: float,
    current_price: float,
    completed_weekly: Optional["pd.DataFrame"] = None,
) -> dict:
    """Evaluate whether the current unfinished week has interacted with any key level.

    Core principle: status is based on CLOSE, not high/low.
    Where price IS matters more than where it's BEEN.

    Returns: {"status": str, "detail": str | None}
    Priority: first match wins (top to bottom).
    When status is "broke_resistance", includes ``breakout_volume_ratio``.
    """
    if current_week is None:
        return {"status": "neutral", "detail": None}

    close = current_week["close"]
    high = current_week["high"]
    low = current_week["low"]

    # --- Rule 1: Close below any weekly swing low → breached_support ---
    weekly_swing_lows = safety_line_candidates.get("weekly_swing_lows", [])
    for sl in weekly_swing_lows:
        if close < sl["level"]:
            return {
                "status": "breached_support",
                "detail": f"Close {close} below weekly swing low {sl['level']}",
            }

    # --- Rule 2: High above highest swing high BUT close below → failed_breakout ---
    if swing_highs:
        highest_sh = max(swing_highs, key=lambda x: x["level"])
        if high > highest_sh["level"] and close < highest_sh["level"]:
            return {
                "status": "failed_breakout",
                "detail": f"High {high} above swing high {highest_sh['level']} but close {close} below it",
            }

    # --- Rule 3: Close above highest swing high → broke_resistance ---
    if swing_highs:
        highest_sh = max(swing_highs, key=lambda x: x["level"])
        if close > highest_sh["level"]:
            breakout_volume_ratio = None
            trading_days = current_week.get("trading_days_elapsed", 0)
            current_vol = current_week.get("volume", 0)
            if (
                trading_days > 0
                and completed_weekly is not None
                and not completed_weekly.empty
                and "Volume" in completed_weekly.columns
            ):
                tail = completed_weekly.tail(4)
                avg_4w = float(tail["Volume"].mean()) if len(tail) > 0 else 0
                if avg_4w > 0:
                    normalized = current_vol * (5 / max(trading_days, 2))
                    breakout_volume_ratio = round(normalized / avg_4w, 2)
            return {
                "status": "broke_resistance",
                "detail": f"Close {close} above swing high {highest_sh['level']}",
                "breakout_volume_ratio": breakout_volume_ratio,
            }

    # --- Rule 4: Close below chandelier → below_chandelier ---
    # Use 3x_atr (widest chandelier) — if close is below even the most
    # permissive trailing stop, that's structurally significant.
    # Tighter multipliers (2x) would trigger too often mid-week.
    chandelier = safety_line_candidates.get("chandelier")
    if chandelier and isinstance(chandelier.get("levels"), dict):
        ch_3x = chandelier["levels"].get("3x_atr")
        if ch_3x and ch_3x.get("level") is not None:
            if close < ch_3x["level"]:
                return {
                    "status": "below_chandelier",
                    "detail": f"Close {close} below chandelier 3x_atr {ch_3x['level']}",
                }

    # --- Rule 5: Low below swing low BUT close recovered above → tested_support ---
    for sl in weekly_swing_lows:
        if low < sl["level"] and close >= sl["level"]:
            return {
                "status": "tested_support",
                "detail": f"Wick to {low} below swing low {sl['level']} but recovered to {close}",
            }

    # --- Rule 6: Pullback in uptrend (down > 0.5 ATR this week) ---
    change_pct = current_week.get("change_from_prev_close_pct", 0) or 0
    if weekly_atr > 0 and current_price > 0:
        half_atr_pct = (0.5 * weekly_atr / current_price) * 100

        if weekly_trend == "uptrend" and change_pct < -half_atr_pct:
            return {
                "status": "pullback_in_uptrend",
                "detail": f"Down {abs(change_pct):.1f}% this week in weekly uptrend",
            }

        # --- Rule 7: Bounce in downtrend (up > 0.5 ATR this week) ---
        if weekly_trend == "downtrend" and change_pct > half_atr_pct:
            return {
                "status": "bounce_in_downtrend",
                "detail": f"Up {change_pct:.1f}% this week in weekly downtrend",
            }

    # --- Rule 8: Nothing notable ---
    return {"status": "neutral", "detail": None}


# ── Candle Character ─────────────────────────────────────────────────────

def compute_candle_character(candle: dict, weekly_atr: Optional[float]) -> dict:
    """Add direction, body_pct, and range_vs_atr to a candle dict.

    Pure math — no judgment. Mutates in place and returns the candle.
    """
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    candle_range = h - l

    # direction
    if candle_range == 0 or abs(c - o) < 0.001 * candle_range:
        candle["direction"] = "doji"
    elif c > o:
        candle["direction"] = "up"
    else:
        candle["direction"] = "down"

    # body_pct — how decisive was this candle (0-100)
    if candle_range > 0:
        candle["body_pct"] = round(abs(c - o) / candle_range * 100)
    else:
        candle["body_pct"] = 0

    # range_vs_atr — was this a normal, quiet, or volatile week
    if weekly_atr and weekly_atr > 0:
        candle["range_vs_atr"] = round(candle_range / weekly_atr, 2)
    else:
        candle["range_vs_atr"] = None

    return candle


def compute_candle_sequence(candles: list) -> list:
    """Summarize candle character as "{direction}_{size}" strings.

    Size: "decisive" if body_pct >= 60, "moderate" if 30-59, "small" if < 30.
    Expects candles to already have 'direction' and 'body_pct' fields.
    """
    sequence = []
    for c in candles:
        direction = c.get("direction", "doji")
        body_pct = c.get("body_pct", 0)

        if body_pct >= 60:
            size = "decisive"
        elif body_pct >= 30:
            size = "moderate"
        else:
            size = "small"

        sequence.append(f"{direction}_{size}")

    return sequence


def compute_rsi_zone(rsi):
    """Classify RSI into a descriptive zone.

    Returns: "oversold" (<30), "neutral" (30-49), "strong" (50-69),
             "extended" (>=70), or None.
    """
    if rsi is None:
        return None
    if rsi < 30:
        return "oversold"
    if rsi < 50:
        return "neutral"
    if rsi < 70:
        return "strong"
    return "extended"


def compute_rsi_direction(current_rsi, prev_rsi):
    """Determine RSI momentum by comparing current vs 3-periods-ago.

    Returns: "rising" (diff > 2), "falling" (diff < -2), "flat", or None.
    """
    if current_rsi is None or prev_rsi is None:
        return None
    diff = current_rsi - prev_rsi
    if diff > 2:
        return "rising"
    if diff < -2:
        return "falling"
    return "flat"


def compute_atr_regime(current_atr, atr_10w_ago):
    """Compare current ATR(14) to ATR(14) from 10 weeks ago.

    Returns: "expanding" (>120%), "contracting" (<80%), "stable", or None.
    """
    if current_atr is None or atr_10w_ago is None or atr_10w_ago <= 0:
        return None
    ratio = current_atr / atr_10w_ago
    if ratio > 1.2:
        return "expanding"
    if ratio < 0.8:
        return "contracting"
    return "stable"


# ── Trendline detection ────────────────────────────────────────────────


def _null_trendlines() -> dict:
    """Return the all-null trendlines payload."""
    return {
        "rising_support": None,
        "falling_resistance": None,
        "trendline_status": None,
        "trendline_convergence": None,
    }


def _match_anchor_to_swing_low(
    anchor_level: float,
    anchor_date_str: str,
    swing_lows_data: list,
    weekly_atr: float,
    df_weekly: pd.DataFrame,
) -> float | None:
    """Cross-reference a trendline anchor with swing lows for bounce_volume_ratio.

    Match criteria: within 1× ATR of level AND within 2 weeks of date.
    Returns bounce_volume_ratio or None.
    """
    if not swing_lows_data or weekly_atr <= 0:
        return None
    try:
        anchor_ts = pd.Timestamp(anchor_date_str)
    except (ValueError, TypeError):
        return None

    for sl in swing_lows_data:
        sl_level = sl.get("level")
        sl_date_str = sl.get("date")
        if sl_level is None or sl_date_str is None:
            continue
        try:
            sl_ts = pd.Timestamp(sl_date_str)
        except (ValueError, TypeError):
            continue
        if abs(anchor_level - sl_level) > weekly_atr:
            continue
        if abs((anchor_ts - sl_ts).days) > 14:
            continue
        # Matched — compute bounce volume ratio
        bounce_date = sl.get("touch_dates", [None])[0] if sl.get("touch_dates") else sl_date_str
        return compute_bounce_volume_ratio(df_weekly, bounce_date)
    return None


def _compute_trendline_interaction(
    completed: pd.DataFrame,
    slope: float,
    intercept: float,
    weekly_atr: float,
    direction: str = "support",
) -> str:
    """Analyze last 6 completed candles against the projected trendline.

    Returns a priority-ordered status string:
      broken, recently_broken, breached_and_recovered, tested_and_held,
      approaching_retest, riding, distant, no_recent_interaction.
    """
    n = len(completed)
    lookback = min(6, n)
    if lookback < 2 or weekly_atr <= 0:
        return "no_recent_interaction"

    tail = completed.iloc[-lookback:]
    closes = tail["Close"].values
    lows = tail["Low"].values
    highs = tail["High"].values

    # Project trendline at each bar position
    base_idx = n - lookback
    projected = [slope * (base_idx + i) + intercept for i in range(lookback)]

    if direction == "support":
        # "wrong side" = close below projected
        wrong_side = [closes[i] < projected[i] for i in range(lookback)]
        signed_dist = [(closes[i] - projected[i]) / weekly_atr for i in range(lookback)]
        touch_dist = [(lows[i] - projected[i]) / weekly_atr for i in range(lookback)]
    else:  # resistance
        wrong_side = [closes[i] > projected[i] for i in range(lookback)]
        signed_dist = [(projected[i] - closes[i]) / weekly_atr for i in range(lookback)]
        touch_dist = [(projected[i] - highs[i]) / weekly_atr for i in range(lookback)]

    # 1. broken — last 2 closes on wrong side
    if wrong_side[-1] and wrong_side[-2]:
        return "broken"

    # 2. recently_broken — last on wrong side, previous on right
    if wrong_side[-1] and not wrong_side[-2]:
        return "recently_broken"

    # 3. breached_and_recovered — any of last 4 on wrong side, most recent on right
    if not wrong_side[-1] and any(wrong_side[-4:]):
        return "breached_and_recovered"

    # 4. tested_and_held — low/high within 0.75 ATR of line, close on right side
    if not wrong_side[-1] and abs(touch_dist[-1]) < 0.75:
        return "tested_and_held"

    # 5. approaching_retest — within 1.5 ATR now, was >2.0 ATR 4 candles ago
    if 0 < signed_dist[-1] < 1.5:
        ago_idx = max(0, lookback - 4)
        if signed_dist[ago_idx] > 2.0:
            return "approaching_retest"

    # 6. riding — 3+ of last 6 within 1.5 ATR on right side
    riding_count = sum(1 for i in range(lookback) if not wrong_side[i] and 0 <= signed_dist[i] < 1.5)
    if riding_count >= 3:
        return "riding"

    # 7. distant — more than 3.0 ATR on right side
    if signed_dist[-1] > 3.0:
        return "distant"

    # 8. fallback
    return "no_recent_interaction"


def _count_closes_wrong_side(
    completed: pd.DataFrame,
    slope: float,
    intercept: float,
    direction: str = "support",
    lookback: int = 8,
) -> int:
    """Count completed weekly candles in last N weeks closing on wrong side."""
    n = len(completed)
    lb = min(lookback, n)
    if lb == 0:
        return 0

    tail = completed.iloc[-lb:]
    closes = tail["Close"].values
    base_idx = n - lb
    count = 0
    for i in range(lb):
        projected = slope * (base_idx + i) + intercept
        if direction == "support" and closes[i] < projected:
            count += 1
        elif direction == "resistance" and closes[i] > projected:
            count += 1
    return count


def _distance_atr_n_weeks_ago(
    completed: pd.DataFrame,
    slope: float,
    intercept: float,
    weekly_atr: float,
    n: int = 4,
) -> float | None:
    """Distance from trendline N weeks ago in ATR units."""
    total = len(completed)
    if total < n + 1 or weekly_atr <= 0:
        return None
    idx = total - 1 - n
    close_then = float(completed["Close"].iloc[idx])
    projected = slope * idx + intercept
    return round(abs(close_then - projected) / weekly_atr, 2)


def detect_trendlines(
    df_weekly: pd.DataFrame,
    current_price: float,
    last_completed_close: float,
    weekly_atr: float | None,
    swing_lows_data: list | None = None,
) -> dict:
    """Detect the single most significant rising support and falling resistance
    trendlines from weekly OHLCV data using the ``trendln`` library.

    Parameters
    ----------
    df_weekly : pd.DataFrame
        Full weekly DataFrame (including current partial week as last row).
    current_price : float
        Latest price (live or last daily close).  Retained in signature for
        callers; **not used** inside this function.
    last_completed_close : float
        Close price of the last completed weekly candle.  Used for all
        distance and status computations (confirmed data only).
    weekly_atr : float | None
        14-period ATR on completed weekly candles.  ``None`` skips detection.

    Returns
    -------
    dict with keys ``rising_support``, ``falling_resistance``, ``trendline_status``.
    Each line sub-dict (or None) has: ``current_projected_level``,
    ``slope_per_week``, ``anchor_points``, ``span_weeks``,
    ``significance_score``, ``price_distance_pct``, ``price_distance_atr``.

    This function is **non-fatal**: any internal exception is caught, logged,
    and the all-null dict is returned so the rest of the analysis continues.
    """
    # current_price is retained in the signature because callers use it for
    # trendline_support candidate injection. It is NOT used inside this function.
    # All distance and status computations use last_completed_close (confirmed weekly data).
    # A future Change B will modify the filter logic to preserve broken trendlines
    # with a "below_rising_support" status instead of discarding them.

    if not _HAS_TRENDLN or weekly_atr is None or len(df_weekly) < 21:
        return _null_trendlines()

    try:
        completed = df_weekly.iloc[:-1]  # exclude current partial week
        n = len(completed)
        last_idx = n - 1
        lows = completed["Low"].values
        highs = completed["High"].values

        # ---- Collect raw candidates from trendln ----
        support_candidates = []
        resistance_candidates = []

        for method in [trendln.METHOD_NSQUREDLOGN, trendln.METHOD_NCUBED]:
            try:
                mins, maxs = trendln.calc_support_resistance(
                    lows,
                    extmethod=trendln.METHOD_NAIVE,
                    method=method,
                )
                for pts, (slope, intercept, ssr, slope_err, int_err, area_avg) in mins[2]:
                    support_candidates.append({
                        "slope": slope,
                        "intercept": intercept,
                        "anchor_indices": list(pts),
                        "slope_err": slope_err,
                    })
                break
            except Exception:
                continue

        for method in [trendln.METHOD_NSQUREDLOGN, trendln.METHOD_NCUBED]:
            try:
                mins2, maxs2 = trendln.calc_support_resistance(
                    highs,
                    extmethod=trendln.METHOD_NAIVE,
                    method=method,
                )
                for pts, (slope, intercept, ssr, slope_err, int_err, area_avg) in maxs2[2]:
                    resistance_candidates.append({
                        "slope": slope,
                        "intercept": intercept,
                        "anchor_indices": list(pts),
                        "slope_err": slope_err,
                    })
                break
            except Exception:
                continue

        # ---- Filter support: rising, ≥3 anchors, ≥12w span, ≤4× ATR (abs) ----
        best_support = None
        sup_qualified = []
        for c in support_candidates:
            anchors = c["anchor_indices"]
            n_anchors = len(anchors)
            if n_anchors < _TL_MIN_ANCHORS:
                continue
            span = anchors[-1] - anchors[0]
            if span < _TL_MIN_SPAN_WEEKS:
                continue
            if c["slope"] <= 0:
                continue
            projected = c["slope"] * last_idx + c["intercept"]
            distance = last_completed_close - projected
            if abs(distance) > _TL_SUPPORT_ATR_MULT * weekly_atr:
                continue
            score = n_anchors * span
            sup_qualified.append((score, c.get("slope_err", float("inf")), projected, distance, n_anchors, span, c))

        if sup_qualified:
            sup_qualified.sort(key=lambda x: (-x[0], x[1]))
            score, slope_err, projected, distance, n_anchors, span, c = sup_qualified[0]
            anchor_dates = [
                str(completed.index[i].date()) if i < len(completed) else None
                for i in c["anchor_indices"]
            ]
            anchor_levels = [
                round(c["slope"] * i + c["intercept"], 2)
                for i in c["anchor_indices"]
            ]
            anchors_list = []
            for idx_a, ai in enumerate(c["anchor_indices"]):
                a_date = anchor_dates[idx_a]
                a_level = anchor_levels[idx_a]
                bvr = None
                if swing_lows_data and a_date:
                    bvr = _match_anchor_to_swing_low(a_level, a_date, swing_lows_data, weekly_atr, df_weekly)
                anchors_list.append({
                    "level": a_level,
                    "date": a_date,
                    "bounce_volume_ratio": bvr,
                })
            last_anchor_idx = c["anchor_indices"][-1]
            weeks_since_last = n - 1 - last_anchor_idx
            slope_pct = round(c["slope"] / projected * 100 * 4.33, 2) if projected else None
            interaction = _compute_trendline_interaction(completed, c["slope"], c["intercept"], weekly_atr, "support")
            closes_wrong = _count_closes_wrong_side(completed, c["slope"], c["intercept"], "support")
            dist_4w = _distance_atr_n_weeks_ago(completed, c["slope"], c["intercept"], weekly_atr, 4)
            best_support = {
                "current_projected_level": round(projected, 2),
                "slope_per_week": round(c["slope"], 4),
                "slope_pct_per_month": slope_pct,
                "anchor_points": n_anchors,
                "span_weeks": span,
                "significance_score": score,
                "price_distance_pct": round(abs(last_completed_close - projected) / last_completed_close * 100, 2) if last_completed_close else 0,
                "price_distance_atr": round(abs(distance) / weekly_atr, 2),
                "anchors": anchors_list,
                "first_anchor_date": anchor_dates[0],
                "last_anchor_date": anchor_dates[-1],
                "weeks_since_last_touch": weeks_since_last,
                "fit_quality": round(1.0 / (1.0 + slope_err), 4) if slope_err < float("inf") else None,
                "trendline_interaction": interaction,
                "weekly_closes_below_line": closes_wrong,
                "distance_atr_4w_ago": dist_4w,
            }

        # ---- Filter resistance: falling, ≥3 anchors, ≥12w span, ≤2× ATR (abs) ----
        best_resistance = None
        res_qualified = []
        for c in resistance_candidates:
            anchors = c["anchor_indices"]
            n_anchors = len(anchors)
            if n_anchors < _TL_MIN_ANCHORS:
                continue
            span = anchors[-1] - anchors[0]
            if span < _TL_MIN_SPAN_WEEKS:
                continue
            if c["slope"] >= 0:
                continue
            projected = c["slope"] * last_idx + c["intercept"]
            distance = projected - last_completed_close
            if abs(distance) > _TL_RESISTANCE_ATR_MULT * weekly_atr:
                continue
            score = n_anchors * span
            res_qualified.append((score, c.get("slope_err", float("inf")), projected, distance, n_anchors, span, c))

        if res_qualified:
            res_qualified.sort(key=lambda x: (-x[0], x[1]))
            score, slope_err, projected, distance, n_anchors, span, c = res_qualified[0]
            anchor_dates = [
                str(completed.index[i].date()) if i < len(completed) else None
                for i in c["anchor_indices"]
            ]
            anchor_levels = [
                round(c["slope"] * i + c["intercept"], 2)
                for i in c["anchor_indices"]
            ]
            anchors_list = [
                {"level": anchor_levels[idx_a], "date": anchor_dates[idx_a]}
                for idx_a in range(len(c["anchor_indices"]))
            ]
            last_anchor_idx = c["anchor_indices"][-1]
            weeks_since_last = n - 1 - last_anchor_idx
            slope_pct = round(c["slope"] / projected * 100 * 4.33, 2) if projected else None
            interaction = _compute_trendline_interaction(completed, c["slope"], c["intercept"], weekly_atr, "resistance")
            closes_wrong = _count_closes_wrong_side(completed, c["slope"], c["intercept"], "resistance")
            dist_4w = _distance_atr_n_weeks_ago(completed, c["slope"], c["intercept"], weekly_atr, 4)
            best_resistance = {
                "current_projected_level": round(projected, 2),
                "slope_per_week": round(c["slope"], 4),
                "slope_pct_per_month": slope_pct,
                "anchor_points": n_anchors,
                "span_weeks": span,
                "significance_score": score,
                "price_distance_pct": round(abs(last_completed_close - projected) / last_completed_close * 100, 2) if last_completed_close else 0,
                "price_distance_atr": round(abs(distance) / weekly_atr, 2),
                "anchors": anchors_list,
                "first_anchor_date": anchor_dates[0],
                "last_anchor_date": anchor_dates[-1],
                "weeks_since_last_touch": weeks_since_last,
                "fit_quality": round(1.0 / (1.0 + slope_err), 4) if slope_err < float("inf") else None,
                "trendline_interaction": interaction,
                "weekly_closes_above_line": closes_wrong,
                "distance_atr_4w_ago": dist_4w,
            }

        # ---- Derive trendline_status ----
        if best_support is not None:
            if last_completed_close >= best_support["current_projected_level"]:
                status = "above_rising_support"
            else:
                status = "below_rising_support"
        elif best_resistance is not None:
            if last_completed_close <= best_resistance["current_projected_level"]:
                status = "below_falling_resistance"
            else:
                status = "above_falling_resistance"
        else:
            status = None

        # ---- Trendline convergence (only when both exist) ----
        convergence = None
        if best_support is not None and best_resistance is not None:
            gap_now = best_resistance["current_projected_level"] - best_support["current_projected_level"]
            gap_now_atr = round(gap_now / weekly_atr, 2)
            # Gap 8 weeks ago
            sup_slope = best_support["slope_per_week"]
            res_slope = best_resistance["slope_per_week"]
            sup_8w = best_support["current_projected_level"] - sup_slope * 8
            res_8w = best_resistance["current_projected_level"] - res_slope * 8
            gap_8w = res_8w - sup_8w
            gap_8w_atr = round(gap_8w / weekly_atr, 2)
            converging = gap_now < gap_8w
            # Estimated weeks to apex (linear extrapolation)
            slope_diff = sup_slope - res_slope  # positive if converging
            apex_weeks = round(gap_now / slope_diff, 1) if slope_diff > 0 else None
            convergence = {
                "gap_current_atr": gap_now_atr,
                "gap_8w_ago_atr": gap_8w_atr,
                "converging": converging,
                "estimated_apex_weeks": apex_weeks,
            }

        return {
            "rising_support": best_support,
            "falling_resistance": best_resistance,
            "trendline_status": status,
            "trendline_convergence": convergence,
        }

    except Exception:
        _logger.warning("trendline detection failed", exc_info=True)
        return _null_trendlines()
