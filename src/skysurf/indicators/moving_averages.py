"""Simple and exponential moving averages.

The Skysurf strategy uses several MA series (EMA-20 for pullback
baselines, SMA-25 for primary regime, EMA-40 for VCP baselines, SMA-27
for the initial trailing stop, etc.). All of them are computed via the
single :func:`compute_ma_series` helper below.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

MaType = Literal["SMA", "EMA"]


def compute_ma_series(close: pd.Series, ma_type: MaType, period: int) -> pd.Series:
    """Compute a simple or exponential moving average on a close series.

    Args:
        close: Close prices, indexed by bar timestamp.
        ma_type: ``"SMA"`` for simple MA, ``"EMA"`` for exponential MA.
        period: Lookback window in bars. Must be ``>= 1``.

    Returns:
        MA series aligned with ``close.index``. The first ``period - 1``
        rows of an SMA are NaN; an EMA is defined from bar 0.

    Raises:
        ValueError: If ``ma_type`` is not ``"SMA"`` or ``"EMA"``, or if
            ``period < 1``.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    if ma_type == "SMA":
        return close.rolling(period).mean()
    if ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    raise ValueError(f"ma_type must be 'SMA' or 'EMA', got {ma_type!r}")
