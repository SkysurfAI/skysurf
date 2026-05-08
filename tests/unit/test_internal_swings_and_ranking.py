"""Tests for the internal swing-detection and ranking modules.

These are private modules (under ``skysurf._internal``), but exercising
them directly catches regressions early. Public users never import these.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from skysurf._internal.ranking import (
    DEFAULT_PRIOR,
    MIN_TOTAL_N,
    MIN_TYPE_N,
    compute_type_prior,
)
from skysurf._internal.swings import detect_swings_argrelextrema

# ── Swings ──────────────────────────────────────────────────────────


def _ohlc_from_close(close: list[float]) -> pd.DataFrame:
    """Build a minimal weekly-OHLC DataFrame from a close-price series."""
    arr = np.asarray(close, dtype=float)
    idx = pd.date_range("2023-01-06", periods=len(arr), freq="W-FRI")
    return pd.DataFrame(
        {"High": arr + 1.0, "Low": arr - 1.0, "Close": arr},
        index=idx,
    )


def test_detect_swings_returns_alternating_indices() -> None:
    """The returned highs and lows must interleave strictly in time."""
    # Synthetic zig-zag: rises, falls, rises, falls, rises
    close = [10, 12, 14, 16, 18, 16, 14, 12, 10, 12, 14, 16, 18, 20, 18, 16, 14]
    df = _ohlc_from_close(close)
    highs, lows = detect_swings_argrelextrema(df, order=2)

    merged = sorted([(i, "H") for i in highs] + [(i, "L") for i in lows])
    types = [t for _, t in merged]
    for prev, cur in pairwise(types):
        assert prev != cur, f"non-alternating sequence: {types}"


def test_detect_swings_empty_when_monotonic() -> None:
    df = _ohlc_from_close(list(range(50)))  # strictly increasing
    highs, lows = detect_swings_argrelextrema(df, order=5)
    # A monotonic series has no interior extrema.
    assert highs == []
    assert lows == []


# ── Ranking ─────────────────────────────────────────────────────────


def test_compute_type_prior_returns_default_when_history_empty() -> None:
    empty = pd.DataFrame(columns=["week_date", "entry_type", "mfe_pct", "mae_pct"])
    scores = compute_type_prior(empty, cutoff_date=pd.Timestamp("2024-06-07"))
    assert scores == {}


def test_compute_type_prior_returns_default_when_below_total_floor() -> None:
    rows = []
    for i in range(MIN_TOTAL_N - 5):  # below floor
        rows.append(
            {
                "week_date": pd.Timestamp("2023-01-06") + pd.Timedelta(weeks=i),
                "entry_type": "PULLBACK_S2" if i % 2 == 0 else "VCP_CONTINUATION",
                "mfe_pct": 10.0,
                "mae_pct": -3.0,
            }
        )
    df = pd.DataFrame(rows)
    scores = compute_type_prior(df, cutoff_date=pd.Timestamp("2025-01-01"))
    assert all(v == DEFAULT_PRIOR for v in scores.values())


def test_compute_type_prior_computes_real_ratio_when_history_sufficient() -> None:
    """With enough history, mfe/|mae| ratio is returned."""
    n = MIN_TYPE_N + 50
    rows = []
    for i in range(n):
        rows.append(
            {
                "week_date": pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=i),
                "entry_type": "PULLBACK_S2",
                "mfe_pct": 18.0,  # median = 18
                "mae_pct": -6.0,  # |median| = 6
            }
        )
    # Pad totals up past MIN_TOTAL_N with another type
    for i in range(MIN_TOTAL_N):
        rows.append(
            {
                "week_date": pd.Timestamp("2020-01-03") + pd.Timedelta(weeks=i),
                "entry_type": "VCP_CONTINUATION",
                "mfe_pct": 12.0,
                "mae_pct": -6.0,
            }
        )
    df = pd.DataFrame(rows)
    scores = compute_type_prior(df, cutoff_date=pd.Timestamp("2026-01-01"))
    assert scores["PULLBACK_S2"] == pytest.approx(18.0 / 6.0)
    assert scores["VCP_CONTINUATION"] == pytest.approx(12.0 / 6.0)


def test_compute_type_prior_strict_cutoff_excludes_equal_dates() -> None:
    """Trades whose week_date equals the cutoff are excluded (strict <)."""
    # Add enough rows past the cutoff to ensure non-default behaviour
    cutoff = pd.Timestamp("2024-06-07")
    n = MIN_TOTAL_N + 5
    rows = []
    for i in range(n):
        rows.append(
            {
                "week_date": cutoff + pd.Timedelta(weeks=i + 1),  # all after cutoff
                "entry_type": "PULLBACK_S2",
                "mfe_pct": 18.0,
                "mae_pct": -6.0,
            }
        )
    df = pd.DataFrame(rows)
    scores = compute_type_prior(df, cutoff_date=cutoff)
    # Nothing before cutoff → default for every type observed.
    assert scores["PULLBACK_S2"] == DEFAULT_PRIOR


def test_compute_type_prior_validates_required_columns() -> None:
    df = pd.DataFrame({"week_date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_type_prior(df, cutoff_date=pd.Timestamp("2024-06-07"))
