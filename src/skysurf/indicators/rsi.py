"""Relative Strength Index (RSI).

The Skysurf strategy does not use RSI as a hard gate in its canonical
Phase 4 configuration (``rsi_gate`` is ``None``), but RSI is exposed
because it is referenced in the cache-adapter pipeline and useful for
downstream analytics.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index over ``window`` bars.

    Args:
        close: Close prices, indexed by bar timestamp.
        window: Lookback in bars. Defaults to 14.

    Returns:
        RSI series aligned with ``close.index``. The first ``window``
        values are NaN by definition.
    """
    return RSIIndicator(close=close, window=window).rsi()
