"""MA computation, slope direction, and 18 config definitions for Test 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._calc.regime_classifier import EMA_DIRECTION_THRESHOLD

# ── MA computation ────────────────────────────────────────────────────


def compute_ma_series(close: pd.Series, ma_type: str, period: int) -> pd.Series:
    """Compute SMA or EMA on a close series.

    Args:
        close: Weekly close prices.
        ma_type: ``"SMA"`` or ``"EMA"``.
        period: Lookback window in weeks.

    Returns:
        MA series aligned with the input index.
    """
    if ma_type == "SMA":
        return close.rolling(period).mean()
    elif ma_type == "EMA":
        return close.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")


# ── Vectorised slope direction ────────────────────────────────────────


def compute_ma_slope_direction(
    ma_series: pd.Series,
    atr_series: pd.Series,
    lookback: int = 4,
    threshold: float = EMA_DIRECTION_THRESHOLD,
) -> pd.Series:
    """Classify MA slope at every bar as rising / falling / flat.

    This is the vectorised equivalent of ``compute_ema_direction`` in
    ``app.utils.regime_classifier`` (which only operates on the tail).

    Logic per bar *i*:
        slope = (ma[i] - ma[i - lookback]) / atr[i]
        > threshold  → "rising"
        < -threshold → "falling"
        else         → "flat"

    Args:
        ma_series: Pre-computed MA values (SMA or EMA).
        atr_series: Weekly ATR values (from ``calculate_atr``).
        lookback: Weeks to look back for slope (default 4).
        threshold: ATR-normalised threshold (default 0.3).

    Returns:
        pd.Series of ``"rising"`` / ``"falling"`` / ``"flat"`` strings,
        aligned with the input index.
    """
    result = pd.Series("flat", index=ma_series.index, dtype=object)

    if len(ma_series) <= lookback:
        return result

    ma_now = ma_series.iloc[lookback:]
    ma_ago = ma_series.iloc[:-lookback].values  # align by position
    atr_now = atr_series.iloc[lookback:]

    # Guard against NaN / zero ATR
    safe_atr = atr_now.replace(0, np.nan)
    slope_norm = (ma_now.values - ma_ago) / safe_atr.values

    directions = np.where(
        slope_norm > threshold,
        "rising",
        np.where(slope_norm < -threshold, "falling", "flat"),
    )
    result.iloc[lookback:] = directions
    return result


# ── Config definitions ────────────────────────────────────────────────

_SINGLE_PERIODS = [10, 15, 20, 25, 30, 40]
_MA_TYPES = ["SMA", "EMA"]


def get_single_ma_configs() -> list[dict]:
    """Return 12 single-MA config dicts."""
    configs = []
    for ma_type in _MA_TYPES:
        for period in _SINGLE_PERIODS:
            configs.append(
                {
                    "label": f"{ma_type}_{period}w",
                    "type": ma_type,
                    "period": period,
                    "config_type": "single",
                }
            )
    return configs


_TREND_TEMPLATES = [
    ("TT_S10_S30_S40", "SMA", 10, "SMA", 30, "SMA", 40),
    ("TT_S10_S20_S40", "SMA", 10, "SMA", 20, "SMA", 40),
    ("TT_S10_S25_S40", "SMA", 10, "SMA", 25, "SMA", 40),
    ("TT_E10_E30_E40", "EMA", 10, "EMA", 30, "EMA", 40),
    ("TT_E10_E20_E40", "EMA", 10, "EMA", 20, "EMA", 40),
    ("TT_E10_E25_E40", "EMA", 10, "EMA", 25, "EMA", 40),
]


def get_trend_template_configs() -> list[dict]:
    """Return 6 Trend Template config dicts."""
    configs = []
    for label, ft, fp, mt, mp, st, sp in _TREND_TEMPLATES:
        configs.append(
            {
                "label": label,
                "fast": (ft, fp),
                "mid": (mt, mp),
                "slow": (st, sp),
                "config_type": "trend_template",
            }
        )
    return configs


def get_all_ma_configs() -> list[dict]:
    """Return all 18 MA configs (12 single + 6 Trend Template)."""
    return get_single_ma_configs() + get_trend_template_configs()


# ── Pre-computation helper ────────────────────────────────────────────


def compute_all_mas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Pre-compute all unique MA series needed across 18 configs.

    Returns a dict keyed by ``"SMA_10"`` / ``"EMA_20"`` etc., each
    mapping to a pd.Series aligned with ``df.index``.
    """
    close = df["Close"]
    # Collect all unique (type, period) pairs
    needed: set[tuple[str, int]] = set()
    for cfg in get_single_ma_configs():
        needed.add((cfg["type"], cfg["period"]))
    for cfg in get_trend_template_configs():
        needed.add(cfg["fast"])
        needed.add(cfg["mid"])
        needed.add(cfg["slow"])

    result: dict[str, pd.Series] = {}
    for ma_type, period in sorted(needed):
        key = f"{ma_type}_{period}"
        result[key] = compute_ma_series(close, ma_type, period)
    return result
