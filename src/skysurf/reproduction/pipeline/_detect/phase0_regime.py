"""5-state breadth-based regime classification — pure functions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_regime_breadth(
    breadth_pct: float,
    breadth_delta: float,
    high_thresh: float,
    low_thresh: float,
    dir_thresh: float,
) -> str:
    """Classify a single week into one of 5 regimes.

    Args:
        breadth_pct: % of stocks in Stage 2 this week.
        breadth_delta: Change in breadth_pct over the direction lookback.
        high_thresh: Breadth % above which market is in bull territory.
        low_thresh: Breadth % below which market is in bear territory.
        dir_thresh: Minimum breadth change (pct points) to count as directional.

    Returns:
        One of: "strong_bull", "weakening_bull", "recovering",
                "deteriorating", "bear".
    """
    if np.isnan(breadth_pct) or np.isnan(breadth_delta):
        return "deteriorating"  # insufficient data → conservative

    if breadth_pct >= high_thresh:
        if breadth_delta >= dir_thresh:
            return "strong_bull"
        else:
            return "weakening_bull"
    elif breadth_pct >= low_thresh:
        if breadth_delta >= 0:
            return "recovering"
        else:
            return "deteriorating"
    else:
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
        breadth_df: DataFrame indexed by date with ``breadth_pct`` column.
        nifty_weekly: Optional DataFrame with ``close`` column for override logic.
            If None, no Nifty override is applied.
        high_thresh, low_thresh, dir_lookback, dir_thresh: Thresholds.

    Returns:
        pd.Series indexed by date with regime labels.
    """
    bp = breadth_df["breadth_pct"]
    delta = bp - bp.shift(dir_lookback)

    regimes = []
    for i in range(len(bp)):
        r = classify_regime_breadth(
            bp.iloc[i],
            delta.iloc[i] if not np.isnan(delta.iloc[i]) else 0.0,
            high_thresh,
            low_thresh,
            dir_thresh,
        )
        regimes.append(r)

    result = pd.Series(regimes, index=breadth_df.index, dtype=object)

    # ── Nifty override (tracked but applied) ───────────────────────
    if nifty_weekly is not None and "close" in nifty_weekly.columns:
        ema = nifty_weekly["close"].ewm(span=20, adjust=False).mean()
        for i in range(len(result)):
            date = result.index[i]
            idx = nifty_weekly.index.searchsorted(date, side="right") - 1
            if 0 <= idx < len(nifty_weekly):
                nifty_close = float(nifty_weekly["close"].iloc[idx])
                nifty_ema = float(ema.iloc[idx])
                nifty_below = nifty_close < nifty_ema

                # Override: breadth says recovering+ but Nifty below falling EMA
                if result.iloc[i] in ("strong_bull", "weakening_bull", "recovering"):
                    if nifty_below and idx >= 4:
                        ema_slope = ema.iloc[idx] - ema.iloc[idx - 4]
                        if ema_slope < 0:
                            result.iloc[i] = "deteriorating"

                # Override: breadth says bear but Nifty above rising EMA
                elif result.iloc[i] == "bear":
                    if not nifty_below and idx >= 4:
                        ema_slope = ema.iloc[idx] - ema.iloc[idx - 4]
                        if ema_slope > 0:
                            result.iloc[i] = "deteriorating"

    return result
