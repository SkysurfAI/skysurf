"""Average True Range (ATR).

ATR measures volatility as the rolling average of the true range:

    TR_i = max(high_i - low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|)
    ATR  = rolling_mean(TR, window)

The Skysurf strategy uses 14-period ATR on weekly bars to size stop
distances (`exit_atr_buffer * ATR` below the trailing moving average) and
to define the climactic-extension threshold.
"""
from __future__ import annotations

import pandas as pd
from ta.volatility import AverageTrueRange


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute Average True Range over ``window`` bars.

    Args:
        high: High prices, indexed by bar timestamp.
        low: Low prices, same index as ``high``.
        close: Close prices, same index as ``high``.
        window: Lookback in bars. Defaults to 14.

    Returns:
        ATR series aligned with the input index. Returns an all-NaN
        series of the same length when there is not enough data
        (``len(close) < window``).

    Raises:
        ValueError: If the three inputs do not share the same index or
            differ in length.
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            f"high, low, close must share length: "
            f"got {len(high)}, {len(low)}, {len(close)}"
        )
    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("high, low, close must share index")

    if len(close) < window:
        return pd.Series([float("nan")] * len(close), index=close.index, name="atr")

    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()
