"""Tests for ATR, RSI, and moving-average helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skysurf import calculate_atr, calculate_rsi, compute_ma_series


@pytest.fixture
def synthetic_close() -> pd.Series:
    """Deterministic close series — 50 weekly bars."""
    rng = np.random.default_rng(seed=0)
    idx = pd.date_range("2023-01-06", periods=50, freq="W-FRI")
    return pd.Series(
        100 + rng.standard_normal(50).cumsum(),
        index=idx,
        name="close",
    )


@pytest.fixture
def synthetic_high_low(synthetic_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    return synthetic_close + 2.0, synthetic_close - 2.0


# ── ATR ──────────────────────────────────────────────────────────────


def test_atr_returns_series_with_same_index(
    synthetic_close: pd.Series,
    synthetic_high_low: tuple[pd.Series, pd.Series],
) -> None:
    high, low = synthetic_high_low
    atr = calculate_atr(high, low, synthetic_close, window=14)
    assert isinstance(atr, pd.Series)
    assert atr.index.equals(synthetic_close.index)


def test_atr_warmup_then_finite_values(
    synthetic_close: pd.Series,
    synthetic_high_low: tuple[pd.Series, pd.Series],
) -> None:
    """During the warmup period the ta library returns 0.0; afterwards it
    returns finite, positive ATR. Both shapes are acceptable here as long
    as later bars are sensible."""
    high, low = synthetic_high_low
    atr = calculate_atr(high, low, synthetic_close, window=14)
    assert pd.notna(atr.iloc[-1])
    assert atr.iloc[-1] > 0
    # The last bar's ATR should be on the order of the daily range (~4)
    assert 0.5 < atr.iloc[-1] < 50


def test_atr_too_short_returns_all_nan(
    synthetic_close: pd.Series,
    synthetic_high_low: tuple[pd.Series, pd.Series],
) -> None:
    high, low = synthetic_high_low
    short_close = synthetic_close.iloc[:5]
    short_high = high.iloc[:5]
    short_low = low.iloc[:5]
    atr = calculate_atr(short_high, short_low, short_close, window=14)
    assert len(atr) == 5
    assert atr.isna().all()


def test_atr_mismatched_length_raises(synthetic_close: pd.Series) -> None:
    high = synthetic_close + 2.0
    low_short = synthetic_close.iloc[:-1] - 2.0
    with pytest.raises(ValueError, match="length"):
        calculate_atr(high, low_short, synthetic_close)


def test_atr_mismatched_index_raises(synthetic_close: pd.Series) -> None:
    high = synthetic_close + 2.0
    low_shifted = pd.Series(
        synthetic_close.values - 2.0,
        index=synthetic_close.index + pd.Timedelta(days=1),
    )
    with pytest.raises(ValueError, match="index"):
        calculate_atr(high, low_shifted, synthetic_close)


# ── RSI ──────────────────────────────────────────────────────────────


def test_rsi_in_zero_to_hundred_range(synthetic_close: pd.Series) -> None:
    rsi = calculate_rsi(synthetic_close, window=14)
    valid = rsi.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


# ── Moving averages ──────────────────────────────────────────────────


def test_sma_matches_rolling_mean(synthetic_close: pd.Series) -> None:
    expected = synthetic_close.rolling(20).mean()
    actual = compute_ma_series(synthetic_close, "SMA", 20)
    pd.testing.assert_series_equal(actual, expected)


def test_ema_matches_pandas_ewm(synthetic_close: pd.Series) -> None:
    expected = synthetic_close.ewm(span=20, adjust=False).mean()
    actual = compute_ma_series(synthetic_close, "EMA", 20)
    pd.testing.assert_series_equal(actual, expected)


def test_ma_unknown_type_raises(synthetic_close: pd.Series) -> None:
    with pytest.raises(ValueError, match="ma_type"):
        compute_ma_series(synthetic_close, "VWAP", 20)  # type: ignore[arg-type]


def test_ma_period_below_one_raises(synthetic_close: pd.Series) -> None:
    with pytest.raises(ValueError, match="period"):
        compute_ma_series(synthetic_close, "SMA", 0)
