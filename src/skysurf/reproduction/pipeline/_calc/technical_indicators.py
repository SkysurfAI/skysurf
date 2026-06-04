"""Technical indicator calculation utilities.

Thin wrappers around the `ta` library for consistent usage across services.
All functions accept a pandas Series of close prices and return pandas Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI interpretation for AI:
    - >70: Overbought — momentum is strong but may be unsustainable.
    - 50-70: Healthy bullish — strong upward momentum with room to run.
    - 30-50: Neutral/weak — no strong directional bias.
    - <30: Oversold — selling may be exhausted, watch for reversal.
    """
    return RSIIndicator(close=close, window=window).rsi()


def calculate_macd(
    close: pd.Series,
    window_fast: int = 12,
    window_slow: int = 26,
    window_signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and Histogram.

    Returns:
        (macd_line, signal_line, histogram)

    MACD interpretation for AI:
    - Histogram > 0: Bullish momentum (MACD above signal).
    - Histogram < 0: Bearish momentum (MACD below signal).
    - Histogram expanding: Momentum accelerating in current direction.
    - Histogram contracting: Momentum weakening, potential reversal.
    """
    indicator = MACD(
        close=close,
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_signal,
    )
    return indicator.macd(), indicator.macd_signal(), indicator.macd_diff()


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR interpretation for AI:
    - Higher ATR: Greater volatility, wider stop-loss needed.
    - Lower ATR: Less volatility, tighter stop-loss possible.
    - Use 2× ATR below price as a common trailing-stop reference.
    """
    if len(close) < window:
        return pd.Series([float("nan")] * len(close), index=close.index)
    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()


def detect_rsi_divergence(
    df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 30
) -> str:
    """Detect RSI divergence using swing lows/highs.

    Returns "bullish", "bearish", or "none".
    - Bullish: price makes lower low but RSI makes higher low.
    - Bearish: price makes higher high but RSI makes lower high.
    """
    if len(df) < lookback or len(rsi_series) < lookback:
        return "none"

    recent_price = df["Close"].iloc[-lookback:]
    recent_rsi = rsi_series.iloc[-lookback:]

    order = min(5, lookback // 4)
    if order < 2:
        return "none"

    # Bullish divergence: compare swing lows
    price_vals = recent_price.values
    rsi_vals = recent_rsi.values

    low_idx = argrelextrema(price_vals, np.less_equal, order=order)[0]
    if len(low_idx) >= 2:
        i1, i2 = low_idx[-2], low_idx[-1]
        if price_vals[i2] < price_vals[i1] and rsi_vals[i2] > rsi_vals[i1]:
            return "bullish"

    # Bearish divergence: compare swing highs
    high_idx = argrelextrema(price_vals, np.greater_equal, order=order)[0]
    if len(high_idx) >= 2:
        i1, i2 = high_idx[-2], high_idx[-1]
        if price_vals[i2] > price_vals[i1] and rsi_vals[i2] < rsi_vals[i1]:
            return "bearish"

    return "none"


def detect_macd_crossover(
    macd_line: pd.Series, signal_line: pd.Series, lookback: int = 3
) -> str | None:
    """Detect MACD crossover in the last `lookback` bars.

    Returns "bullish_crossover", "bearish_crossover", or None.
    """
    if len(macd_line) < lookback + 1 or len(signal_line) < lookback + 1:
        return None

    for i in range(-lookback, 0):
        prev_diff = float(macd_line.iloc[i - 1] - signal_line.iloc[i - 1])
        curr_diff = float(macd_line.iloc[i] - signal_line.iloc[i])
        if prev_diff <= 0 < curr_diff:
            return "bullish_crossover"
        if prev_diff >= 0 > curr_diff:
            return "bearish_crossover"

    return None
