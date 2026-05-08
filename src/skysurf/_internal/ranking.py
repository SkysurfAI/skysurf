"""Dynamic per-entry-type ranking via realized MFE/MAE ratios.

The strategy ranks crowded weeks of candidates by a *type prior* — a
score derived from how each entry type has historically performed. The
score is the ratio of median favourable excursion (MFE) to the
absolute value of median adverse excursion (MAE), computed over the
trade history known *before* the current decision date.

This module reads from a DataFrame supplied by the caller (typically
:meth:`DataProvider.get_historical_trades`), not from disk. The
canonical caller is :func:`skysurf.signals.generate_weekly_signals`.

Per the canonical CF1 design (a documented residual look-ahead leak),
filtering is on entry ``week_date < cutoff``. Trades that entered
before the cutoff but exited after still contribute their realised
MFE/MAE — this is intentional and was part of the validated 1.96 MAR
result.
"""

from __future__ import annotations

import pandas as pd

#: Score returned when the prior cannot be computed (e.g., not enough
#: history to be statistically meaningful).
DEFAULT_PRIOR: float = 1.0

#: Minimum number of trades of a given entry type required before the
#: dynamic prior is computed for that type. Below this floor, the type
#: receives :data:`DEFAULT_PRIOR`.
MIN_TYPE_N: int = 20

#: Minimum total trades (across all types) required before any dynamic
#: prior is computed. Below this floor, all types receive
#: :data:`DEFAULT_PRIOR`.
MIN_TOTAL_N: int = 100

_REQUIRED_COLUMNS: tuple[str, ...] = ("week_date", "entry_type", "mfe_pct", "mae_pct")


def compute_type_prior(
    historical_trades: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    *,
    min_type_n: int = MIN_TYPE_N,
    min_total_n: int = MIN_TOTAL_N,
    default_prior: float = DEFAULT_PRIOR,
) -> dict[str, float]:
    """Compute a per-entry-type ranking score as of ``cutoff_date``.

    For each entry type, the score is::

        median(mfe_pct) / abs(median(mae_pct))

    computed over trades whose ``week_date`` is strictly less than
    ``cutoff_date``.

    Args:
        historical_trades: DataFrame with columns ``[week_date,
            entry_type, mfe_pct, mae_pct]``. Typically the return of
            :meth:`~skysurf.data.provider.DataProvider.get_historical_trades`.
        cutoff_date: Decision date. Only trades that entered strictly
            before this date contribute to the prior.
        min_type_n: Per-type minimum trade count; below this, the type
            receives ``default_prior``.
        min_total_n: Total trade count floor; below this, every type
            receives ``default_prior``.
        default_prior: Fallback score.

    Returns:
        Mapping ``{entry_type: score}``. Entry types present in
        ``historical_trades`` but with insufficient data receive
        ``default_prior``.

    Raises:
        ValueError: If the input DataFrame is missing required columns.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in historical_trades.columns]
    if missing:
        raise ValueError(
            f"historical_trades is missing required columns: {missing}. "
            f"Expected {list(_REQUIRED_COLUMNS)}."
        )

    cutoff = pd.Timestamp(cutoff_date)
    df = historical_trades.dropna(subset=["entry_type"]).copy()
    df["week_date"] = pd.to_datetime(df["week_date"])

    all_types = sorted(df["entry_type"].unique())
    sub = df[df["week_date"] < cutoff]

    if len(sub) < min_total_n:
        return dict.fromkeys(all_types, default_prior)

    scores: dict[str, float] = {}
    for entry_type in all_types:
        group = sub[sub["entry_type"] == entry_type]
        if len(group) < min_type_n:
            scores[entry_type] = default_prior
            continue
        mfe_median = float(group["mfe_pct"].median())
        mae_median_abs = abs(float(group["mae_pct"].median()))
        if mae_median_abs > 0:
            scores[entry_type] = mfe_median / mae_median_abs
        else:
            scores[entry_type] = default_prior
    return scores
