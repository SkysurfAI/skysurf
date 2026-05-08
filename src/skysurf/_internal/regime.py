"""Market regime classification.

Two complementary classification systems are vendored here:

* **Weinstein-style two-signal classifier** (:func:`classify_regime`):
  uses EMA position (price above / below / at the 20-week EMA) and EMA
  direction (rising / falling / flat) to label a single moment as
  ``"bull"``, ``"bear"``, or ``"sideways"``.

* **Five-state breadth classifier** (:func:`classify_regime_breadth`,
  :func:`classify_all_weeks`): uses universe breadth (% of stocks
  trading above SMA-25) and breadth direction to label every week as
  one of ``"strong_bull"``, ``"weakening_bull"``, ``"recovering"``,
  ``"deteriorating"``, ``"bear"``. PHASE_4_V1 uses this classifier for
  the overall-market regime gate.

The two systems coexist; the Phase 4 strategy uses the breadth
classifier for entry gating and the Weinstein classifier for sector-
level regimes (which the brain consumes via
:meth:`DataProvider.get_sector_regime_for`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Minimum ATR-normalised slope to call an EMA "rising" (or, negated, "falling").
EMA_DIRECTION_THRESHOLD: float = 0.3

#: Default lookback (weeks) for price-vs-EMA consistency check.
EMA_POSITION_LOOKBACK: int = 4

#: Default lookback (weeks) for EMA slope calculation.
EMA_DIRECTION_LOOKBACK: int = 4

#: Below this breadth %, a "bull" classification is overridden to "sideways".
BREADTH_BULL_FLOOR: int = 35

#: Above this breadth %, a "bear" classification is overridden to "sideways".
BREADTH_BEAR_CEILING: int = 65


# ── Weinstein-style two-signal classifier ────────────────────────────


def compute_ema_position(
    weekly_closes: pd.Series,
    ema_series: pd.Series,
    lookback_weeks: int = EMA_POSITION_LOOKBACK,
) -> dict[str, float | int | str]:
    """Determine whether price is consistently above or below the EMA.

    Looks at the most recent ``lookback_weeks`` bars rather than just
    the latest week, to avoid flicker.
    """
    n = min(lookback_weeks, len(weekly_closes), len(ema_series))
    if n == 0:
        return {"position": "at_ema", "distance_pct": 0.0, "weeks_above": 0, "weeks_below": 0}

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
    distance_pct = (
        round((current_close - current_ema) / current_ema * 100, 1) if current_ema != 0 else 0.0
    )

    return {
        "position": position,
        "distance_pct": distance_pct,
        "weeks_above": weeks_above,
        "weeks_below": weeks_below,
    }


def compute_ema_direction(
    ema_series: pd.Series,
    atr_value: float,
    lookback_weeks: int = EMA_DIRECTION_LOOKBACK,
) -> dict[str, float | str]:
    """Determine whether the 20-week EMA is rising, falling, or flat.

    The slope threshold is normalised by ATR so it adapts to market
    volatility — the same absolute slope means different things in calm
    vs volatile markets.
    """
    if len(ema_series) < lookback_weeks + 1 or atr_value is None:
        return {"direction": "flat", "slope_pct": 0.0, "normalized_slope": 0.0}

    ema_now = float(ema_series.iloc[-1])
    ema_ago = float(ema_series.iloc[-1 - lookback_weeks])
    slope_abs = ema_now - ema_ago

    normalized_slope = round(slope_abs / atr_value, 2) if atr_value > 0 else 0.0
    slope_pct = round(slope_abs / ema_ago * 100, 2) if ema_ago != 0 else 0.0

    if normalized_slope > EMA_DIRECTION_THRESHOLD:
        direction = "rising"
    elif normalized_slope < -EMA_DIRECTION_THRESHOLD:
        direction = "falling"
    else:
        direction = "flat"

    return {"direction": direction, "slope_pct": slope_pct, "normalized_slope": normalized_slope}


def classify_regime(
    ema_position: str,
    ema_direction: str,
    breadth_200dma_pct: int | None = None,
) -> dict[str, bool | str]:
    """Classify market regime using Weinstein-style two-signal logic.

    * **Bull (Stage 2)**: price consistently above a rising 20-week EMA.
    * **Bear (Stage 4)**: price consistently below a falling 20-week EMA.
    * **Sideways (Stage 1/3)**: everything else — EMA flat, price
      tangled with EMA, or conflicting signals.

    A breadth safety override downgrades bull/bear to sideways when the
    breadth picture contradicts the structural classification.
    """
    if ema_position == "above" and ema_direction == "rising":
        raw = "bull"
    elif ema_position == "below" and ema_direction == "falling":
        raw = "bear"
    else:
        raw = "sideways"

    classification = raw
    breadth_override = False

    if breadth_200dma_pct is not None and (
        (raw == "bull" and breadth_200dma_pct < BREADTH_BULL_FLOOR)
        or (raw == "bear" and breadth_200dma_pct > BREADTH_BEAR_CEILING)
    ):
        classification = "sideways"
        breadth_override = True

    return {
        "classification": classification,
        "breadth_override": breadth_override,
        "raw_classification": raw,
    }


# ── Five-state breadth classifier ─────────────────────────────────────


def classify_regime_breadth(
    breadth_pct: float,
    breadth_delta: float,
    high_thresh: float,
    low_thresh: float,
    dir_thresh: float,
) -> str:
    """Classify a single week into one of five regimes.

    Args:
        breadth_pct: % of stocks in Stage 2 this week.
        breadth_delta: Change in ``breadth_pct`` over the direction lookback.
        high_thresh: Breadth % above which the market is in bull territory.
        low_thresh: Breadth % below which the market is in bear territory.
        dir_thresh: Minimum breadth change (pct points) to count as directional.

    Returns:
        One of ``"strong_bull"``, ``"weakening_bull"``, ``"recovering"``,
        ``"deteriorating"``, ``"bear"``. ``"deteriorating"`` is returned
        when input data is missing (conservative default).
    """
    if np.isnan(breadth_pct) or np.isnan(breadth_delta):
        return "deteriorating"

    if breadth_pct >= high_thresh:
        return "strong_bull" if breadth_delta >= dir_thresh else "weakening_bull"
    if breadth_pct >= low_thresh:
        return "recovering" if breadth_delta >= 0 else "deteriorating"
    return "bear"


def classify_all_weeks(
    breadth_df: pd.DataFrame,
    nifty_weekly: pd.DataFrame | None,
    high_thresh: float,
    low_thresh: float,
    dir_lookback: int,
    dir_thresh: float,
) -> pd.Series:
    """Classify every week in the breadth series into a regime.

    Args:
        breadth_df: DataFrame indexed by date with a ``breadth_pct`` column.
        nifty_weekly: Optional DataFrame with a ``close`` column for the
            Nifty override; if ``None``, no override is applied.
        high_thresh: Bull threshold (breadth %).
        low_thresh: Bear threshold (breadth %).
        dir_lookback: Weeks to look back for the breadth direction signal.
        dir_thresh: Minimum breadth delta (pct points) for "directional".

    Returns:
        ``pd.Series`` indexed by ``breadth_df.index`` with regime labels.
    """
    bp = breadth_df["breadth_pct"]
    delta = bp - bp.shift(dir_lookback)

    regimes: list[str] = []
    for i in range(len(bp)):
        d = delta.iloc[i]
        regimes.append(
            classify_regime_breadth(
                bp.iloc[i],
                d if not np.isnan(d) else 0.0,
                high_thresh,
                low_thresh,
                dir_thresh,
            )
        )
    result = pd.Series(regimes, index=breadth_df.index, dtype=object)

    # Optional Nifty override: if the index price action contradicts the
    # breadth-derived regime, downgrade strong/weakening bull / recovering
    # to "deteriorating", or upgrade "bear" to "deteriorating" when index
    # is rising. Symmetric, conservative.
    if nifty_weekly is None or "close" not in nifty_weekly.columns:
        return result

    ema = nifty_weekly["close"].ewm(span=20, adjust=False).mean()
    for i in range(len(result)):
        date = result.index[i]
        idx = nifty_weekly.index.searchsorted(date, side="right") - 1
        if not (0 <= idx < len(nifty_weekly)):
            continue
        nifty_close = float(nifty_weekly["close"].iloc[idx])
        nifty_ema = float(ema.iloc[idx])
        nifty_below = nifty_close < nifty_ema

        is_bullish_label = result.iloc[i] in ("strong_bull", "weakening_bull", "recovering")
        is_bear_label = result.iloc[i] == "bear"
        if not (idx >= 4 and (is_bullish_label or is_bear_label)):
            continue
        ema_slope = ema.iloc[idx] - ema.iloc[idx - 4]
        if (is_bullish_label and nifty_below and ema_slope < 0) or (
            is_bear_label and not nifty_below and ema_slope > 0
        ):
            result.iloc[i] = "deteriorating"

    return result
