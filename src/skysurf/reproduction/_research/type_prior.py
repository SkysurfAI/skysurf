"""Dynamic type_prior — purely OOS per-entry-type MFE/MAE ratios.

Replaces the static TYPE_PRIOR dict (derived from full 2007-2026 trade_stats)
with values computed strictly from data prior to each decision point.

Note on look-ahead: spec uses week_date < cutoff (entry date), not
exit_week < cutoff. With ~12-week avg holding the residual leak is small
but non-zero; documented here for the record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skysurf.reproduction import _paths

DEFAULT_PRIOR = 1.0
MIN_TYPE_N = 20
MIN_TOTAL_N = 100

TRADE_STATS_PATH = _paths.data_dir() / "trade_stats_all.csv"


_STATS_CACHE: dict = {}


def _load_stats(trade_stats_path) -> pd.DataFrame:
    key = str(Path(trade_stats_path).resolve())
    if key not in _STATS_CACHE:
        df = pd.read_csv(trade_stats_path, low_memory=False, parse_dates=["week_date"])
        _STATS_CACHE[key] = df[["week_date", "entry_type", "mfe_pct", "mae_pct"]].dropna(subset=["entry_type"])
    return _STATS_CACHE[key]


def _per_type_ratios(sub: pd.DataFrame, all_types) -> dict:
    if len(sub) < MIN_TOTAL_N:
        return {t: DEFAULT_PRIOR for t in all_types}
    out = {}
    for t in all_types:
        g = sub[sub["entry_type"] == t]
        if len(g) < MIN_TYPE_N:
            out[t] = DEFAULT_PRIOR
            continue
        mfe_med = g["mfe_pct"].median()
        mae_med = abs(g["mae_pct"].median())
        out[t] = float(mfe_med / mae_med) if mae_med > 0 else DEFAULT_PRIOR
    return out


def compute_type_prior(trade_stats_path, cutoff_date) -> dict:
    """Per-entry-type median(mfe_pct) / abs(median(mae_pct)) over week_date < cutoff_date."""
    df = _load_stats(trade_stats_path)
    cutoff = pd.Timestamp(cutoff_date)
    all_types = sorted(df["entry_type"].unique())
    sub = df[df["week_date"] < cutoff]
    return _per_type_ratios(sub, all_types)


def compute_type_prior_for_entries(entries_df: pd.DataFrame, trade_stats_path) -> pd.DataFrame:
    """Attach a dynamic type_prior column using quarterly (13W) anchors.

    Each entry gets the snapshot from the most recent 13-week anchor at or
    before its week_date. Entries before the first anchor get DEFAULT_PRIOR.
    """
    df = _load_stats(trade_stats_path)
    all_types = sorted(df["entry_type"].unique())
    anchors = pd.date_range("2007-01-01", "2026-12-31", freq="13W-MON")
    anchors_arr = anchors.to_numpy()

    rows = []
    for i, a in enumerate(anchors):
        sub = df[df["week_date"] < a]
        snap = _per_type_ratios(sub, all_types)
        for t, v in snap.items():
            rows.append({"anchor_idx": i, "entry_type": t, "type_prior": v})
    lookup = pd.DataFrame(rows)

    out = entries_df.copy()
    out["week_date"] = pd.to_datetime(out["week_date"])
    wd = out["week_date"].to_numpy()
    idx = np.searchsorted(anchors_arr, wd, side="right") - 1
    out["_anchor_idx"] = idx

    pre_first = out["_anchor_idx"] < 0
    if pre_first.any():
        out.loc[pre_first, "_anchor_idx"] = 0  # join key placeholder; overwritten below

    merged = out.merge(lookup, how="left", left_on=["_anchor_idx", "entry_type"],
                       right_on=["anchor_idx", "entry_type"])
    merged.loc[pre_first.values, "type_prior"] = DEFAULT_PRIOR
    merged["type_prior"] = merged["type_prior"].fillna(DEFAULT_PRIOR)
    merged = merged.drop(columns=["_anchor_idx", "anchor_idx"])
    return merged


if __name__ == "__main__":
    import json
    print("Static reference (full-period in-sample):")
    print(json.dumps({"PULLBACK_S2": 2.92, "VCP_CONTINUATION": 2.19,
                      "BREAKOUT_S1_TO_S2": 2.12, "RETEST_SUPPORT": 1.61,
                      "TRENDLINE_BOUNCE": 1.21}, indent=2))
    for cutoff in ["2010-01-01", "2013-01-01", "2016-01-01", "2019-01-01",
                   "2022-01-01", "2025-01-01", "2026-04-01"]:
        snap = compute_type_prior(TRADE_STATS_PATH, cutoff)
        print(f"\nDynamic snapshot at cutoff {cutoff}:")
        for k, v in sorted(snap.items(), key=lambda kv: -kv[1]):
            print(f"  {k:25s}  {v:.3f}")
