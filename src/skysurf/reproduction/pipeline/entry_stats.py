"""Entry-stats generator — regenerates ``trade_stats_all.csv`` from the bundle.

This is the verbatim research ``val_entry_stats.py`` (Engine-A v2 per-trade entry
quality analysis), refactored only so its paths resolve from the reproduction
data bundle via :mod:`skysurf.reproduction._paths` instead of the scattered
research result folders. The simulation and analysis logic is byte-identical —
that is what makes the regenerated ``trade_stats_all.csv`` match the research
artifact (and therefore reproduce the dynamic type-prior exactly).

It simulates EVERY entry from ``entries_all.csv`` as an independent trade (no
portfolio constraints, no regime filter, no sizing), recording MFE, MAE and
related metrics, then writes:

  * ``trade_stats_all.csv``  -> the **data bundle** dir (``data_dir()``); this is
    the file :mod:`skysurf.reproduction._research.type_prior` reads to compute the
    out-of-sample dynamic type-prior.
  * ``entry_type_summary.csv`` + ``summary.json`` + ~20 analysis PNGs -> the
    **output** dir (``output_dir()/entry_stats``). These are diagnostics, not
    reproduction inputs; matplotlib is optional (``skysurf[reproduction]``).

Inputs (both from the data bundle):
  * ``entries_all.csv``        — produced by ``skysurf-build`` (the OHLCV pipeline)
  * ``stock_weekly_cache/``    — produced by ``skysurf-build``

Usage::

    skysurf-build-stats --data /path/to/skysurf-repro-data
    python -m skysurf.reproduction.pipeline.entry_stats --data ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _paths

# matplotlib is an optional extra (skysurf[reproduction]); charts are skipped if
# it is not installed. trade_stats_all.csv itself never needs matplotlib.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None
    _HAVE_MPL = False

# ── Paths (placeholders resolved at import; main() re-resolves at run time, so
# merely importing this module has no filesystem side effects) ────────────────
ENGINE_A_DIR = _paths.data_dir()
STOCK_CACHE_DIR = ENGINE_A_DIR / "stock_weekly_cache"
OUTPUT_DIR = _paths.output_dir() / "entry_stats"

# ── Exit config (constant for all entries) ────────────────────────────
EXIT_CONFIG = {
    "exit_atr_buffer": 1.0,
    "exit_gtt_field": "low",
    "max_holding_weeks": 104,
}

IS_SPLIT = "2020-01-01"


# ══════════════════════════════════════════════════════════════════════
# STOCK CACHE MANAGER
# ══════════════════════════════════════════════════════════════════════

class StockCacheManager:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._cache: dict[str, pd.DataFrame | None] = {}

    def get(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._cache:
            path = self._dir / f"{ticker}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                self._cache[ticker] = df
            else:
                self._cache[ticker] = None
        return self._cache[ticker]


# ══════════════════════════════════════════════════════════════════════
# PER-TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════════════

def simulate_single_trade(entry: pd.Series, stock_df: pd.DataFrame) -> dict:
    """Simulate a single entry → exit trade, recording MFE/MAE."""
    entry_price = float(entry["entry_price_close"])
    initial_stop = float(entry["stop_level"]) if not pd.isna(entry.get("stop_level", np.nan)) else 0
    atr_at_entry = float(entry["atr_at_entry"]) if not pd.isna(entry.get("atr_at_entry", np.nan)) else 0
    entry_date = pd.Timestamp(entry["week_date"])

    # Derive MA column for trailing stop
    ma_col = f"{str(entry['ma_type']).lower()}_{int(entry['ma_period'])}"

    # Find entry week in stock data
    if entry_date not in stock_df.index:
        # Try nearest
        loc = stock_df.index.searchsorted(entry_date, side="right") - 1
        if loc < 0 or loc >= len(stock_df):
            return _no_data_row(entry)
        entry_date = stock_df.index[loc]

    entry_iloc = stock_df.index.get_loc(entry_date)
    if isinstance(entry_iloc, slice):
        entry_iloc = entry_iloc.start

    if entry_price <= 0:
        return _no_data_row(entry)

    # Initialize tracking
    stop = initial_stop
    peak_close = entry_price
    min_low = entry_price
    mfe_week = entry_date
    exit_price = None
    exit_reason = None
    exit_week = None

    highs = stock_df["High"].values
    lows = stock_df["Low"].values
    opens = stock_df["Open"].values
    closes = stock_df["Close"].values
    n = len(stock_df)

    buffer = EXIT_CONFIG["exit_atr_buffer"]
    max_weeks = EXIT_CONFIG["max_holding_weeks"]

    for w in range(1, max_weeks + 1):
        idx = entry_iloc + w
        if idx >= n:
            # End of data
            last_idx = n - 1
            exit_price = float(closes[last_idx])
            exit_reason = "end_of_data"
            exit_week = stock_df.index[last_idx]
            break

        week_open = float(opens[idx])
        week_low = float(lows[idx])
        week_close = float(closes[idx])
        wk_date = stock_df.index[idx]

        # Update MAE (before exit check — captures exit-week Low)
        min_low = min(min_low, week_low)

        # Update trailing stop
        ma_val = stock_df.iloc[idx].get(ma_col)
        atr_val = stock_df.iloc[idx].get("atr_14")
        if ma_val is not None and atr_val is not None and not pd.isna(ma_val) and not pd.isna(atr_val):
            new_trail = float(ma_val) - buffer * float(atr_val)
            stop = max(stop, new_trail)

        # Exit check
        if week_low <= stop:
            if week_open < stop:
                exit_price = week_open
                exit_reason = "gap_down"
            else:
                exit_price = stop
                exit_reason = "trailing_stop"
            exit_week = wk_date
            break

        # Update MFE (only if surviving)
        if week_close > peak_close:
            peak_close = week_close
            mfe_week = wk_date

    else:
        # Staleness cap
        idx = min(entry_iloc + max_weeks, n - 1)
        exit_price = float(closes[idx])
        exit_reason = "staleness_cap"
        exit_week = stock_df.index[idx]

    if exit_price is None:
        return _no_data_row(entry)

    # Compute metrics
    ret_pct = (exit_price - entry_price) / entry_price * 100
    # NOTE: MAE uses min_low (weekly Low) while MFE uses peak_close (weekly Close).
    # MAE on Low = true risk (GTT stops trigger on Low).
    # MFE on Close = achievable upside (can only exit at Close, not intraweek High).
    mfe_pct = (peak_close - entry_price) / entry_price * 100
    mae_pct = (min_low - entry_price) / entry_price * 100
    mfe_mae_ratio = abs(mfe_pct / mae_pct) if mae_pct < -0.01 else 999.0
    capture = ret_pct / mfe_pct if mfe_pct > 0.01 else 0.0
    holding = max(1, ((exit_week - entry_date).days // 7) if exit_week else 1)
    t2mfe = max(1, ((mfe_week - entry_date).days // 7))

    hit_2 = 1 if atr_at_entry > 0 and peak_close >= entry_price + 2 * atr_at_entry else 0
    hit_3 = 1 if atr_at_entry > 0 and peak_close >= entry_price + 3 * atr_at_entry else 0
    hit_5 = 1 if atr_at_entry > 0 and peak_close >= entry_price + 5 * atr_at_entry else 0

    return {
        "config_id": int(entry["config_id"]),
        "ma_type": str(entry["ma_type"]),
        "ma_period": int(entry["ma_period"]),
        "ticker": str(entry["ticker"]),
        "week_date": str(entry_date.date()),
        "entry_type": str(entry["entry_type"]),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "exit_week": str(exit_week.date()) if exit_week else None,
        "exit_reason": exit_reason,
        "return_pct": round(ret_pct, 2),
        "holding_weeks": holding,
        "mfe_pct": round(mfe_pct, 2),
        "mae_pct": round(mae_pct, 2),
        "mfe_mae_ratio": round(mfe_mae_ratio, 2),
        "time_to_mfe_weeks": t2mfe,
        "capture_ratio": round(capture, 2),
        "hit_2atr": hit_2,
        "hit_3atr": hit_3,
        "hit_5atr": hit_5,
        "risk_pct_at_entry": float(entry["risk_pct_at_entry"]) if not pd.isna(entry.get("risk_pct_at_entry", np.nan)) else None,
        "rr_ratio": float(entry["rr_ratio"]) if not pd.isna(entry.get("rr_ratio", np.nan)) else None,
        "rs_13w": float(entry["rs_13w"]) if not pd.isna(entry.get("rs_13w", np.nan)) else None,
        "rsi_at_entry": float(entry["rsi_at_entry"]) if not pd.isna(entry.get("rsi_at_entry", np.nan)) else None,
        "momentum_at_entry": int(entry["momentum_at_entry"]) if not pd.isna(entry.get("momentum_at_entry", np.nan)) else None,
        "volume_ratio_at_entry": float(entry["volume_ratio_at_entry"]) if not pd.isna(entry.get("volume_ratio_at_entry", np.nan)) else None,
        "target_type": str(entry["target_type"]) if not pd.isna(entry.get("target_type", np.nan)) else None,
        "sector": str(entry["sector"]),
        "weeks_since_breakout": float(entry["weeks_since_breakout"]) if not pd.isna(entry.get("weeks_since_breakout", np.nan)) else None,
        "trendline_anchor_count": float(entry["trendline_anchor_count"]) if not pd.isna(entry.get("trendline_anchor_count", np.nan)) else None,
        "consolidation_range_atr": float(entry["consolidation_range_atr"]) if not pd.isna(entry.get("consolidation_range_atr", np.nan)) else None,
        "ath_type": str(entry["ath_type"]) if not pd.isna(entry.get("ath_type", np.nan)) else None,
        "distance_from_ma_atr": float(entry["distance_from_ma_atr"]) if not pd.isna(entry.get("distance_from_ma_atr", np.nan)) else None,
        "retest_level": float(entry["retest_level"]) if not pd.isna(entry.get("retest_level", np.nan)) else None,
        "proximity_atr": float(entry["proximity_atr"]) if not pd.isna(entry.get("proximity_atr", np.nan)) else None,
    }


def _no_data_row(entry: pd.Series) -> dict:
    """Return a row with exit_reason='no_data' and NaN metrics."""
    return {
        "config_id": int(entry["config_id"]),
        "ma_type": str(entry["ma_type"]),
        "ma_period": int(entry["ma_period"]),
        "ticker": str(entry["ticker"]),
        "week_date": str(pd.Timestamp(entry["week_date"]).date()),
        "entry_type": str(entry["entry_type"]),
        "entry_price": float(entry["entry_price_close"]) if not pd.isna(entry.get("entry_price_close", np.nan)) else None,
        "exit_price": None, "exit_week": None,
        "exit_reason": "no_data",
        "return_pct": None, "holding_weeks": None,
        "mfe_pct": None, "mae_pct": None, "mfe_mae_ratio": None,
        "time_to_mfe_weeks": None, "capture_ratio": None,
        "hit_2atr": None, "hit_3atr": None, "hit_5atr": None,
        "risk_pct_at_entry": float(entry["risk_pct_at_entry"]) if not pd.isna(entry.get("risk_pct_at_entry", np.nan)) else None,
        "rr_ratio": None, "rs_13w": None,
        "rsi_at_entry": None, "momentum_at_entry": None, "volume_ratio_at_entry": None,
        "target_type": str(entry["target_type"]) if not pd.isna(entry.get("target_type", np.nan)) else None,
        "sector": str(entry["sector"]),
        "weeks_since_breakout": None, "trendline_anchor_count": None,
        "consolidation_range_atr": None, "ath_type": None,
        "distance_from_ma_atr": None, "retest_level": None,
        "proximity_atr": None,
    }


# ══════════════════════════════════════════════════════════════════════
# BATCH SIMULATION
# ══════════════════════════════════════════════════════════════════════

def run_all_simulations(entries_df: pd.DataFrame, cache: StockCacheManager) -> pd.DataFrame:
    """Simulate all entries and return a DataFrame of trade results."""
    results = []
    n = len(entries_df)
    t0 = time.time()

    for i, (_, entry) in enumerate(entries_df.iterrows()):
        ticker = str(entry["ticker"])
        stock_df = cache.get(ticker)
        if stock_df is None:
            results.append(_no_data_row(entry))
        else:
            results.append(simulate_single_trade(entry, stock_df))

        if (i + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate if rate > 0 else 0
            m, s = divmod(int(remaining), 60)
            print(f"    {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%) | "
                  f"elapsed: {int(elapsed)}s | ETA: {m}m{s}s", flush=True)

    elapsed = time.time() - t0
    print(f"    Simulation complete: {n:,} trades in {int(elapsed)}s", flush=True)
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS + CHARTS
# ══════════════════════════════════════════════════════════════════════

def _safe_median(s):
    v = s.dropna()
    return float(v.median()) if len(v) > 0 else 0

def _safe_mean(s):
    v = s.dropna()
    return float(v.mean()) if len(v) > 0 else 0


def analysis_1_entry_type_comparison(df: pd.DataFrame):
    """The money table: per-type MFE/MAE summary."""
    print("\n  ═══ Analysis 1: Entry Type Comparison ═══")
    valid = df[df["exit_reason"] != "no_data"]
    types = sorted(valid["entry_type"].unique())

    rows = []
    for et in types:
        sub = valid[valid["entry_type"] == et]
        n = len(sub)
        rows.append({
            "entry_type": et,
            "count": n,
            "median_mfe": _safe_median(sub["mfe_pct"]),
            "mean_mfe": _safe_mean(sub["mfe_pct"]),
            "median_mae": _safe_median(sub["mae_pct"]),
            "mean_mae": _safe_mean(sub["mae_pct"]),
            "median_mfe_mae_ratio": _safe_median(sub["mfe_mae_ratio"]),
            "win_rate": round((sub["return_pct"] > 0).sum() / n * 100, 1) if n > 0 else 0,
            "mean_return": _safe_mean(sub["return_pct"]),
            "hit_2atr": round(sub["hit_2atr"].mean() * 100, 1) if n > 0 else 0,
            "hit_3atr": round(sub["hit_3atr"].mean() * 100, 1) if n > 0 else 0,
            "hit_5atr": round(sub["hit_5atr"].mean() * 100, 1) if n > 0 else 0,
            "median_time_to_mfe": _safe_median(sub["time_to_mfe_weeks"]),
            "mean_capture": _safe_mean(sub["capture_ratio"]),
            "mean_risk_pct": _safe_mean(sub["risk_pct_at_entry"]),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "entry_type_summary.csv", index=False)

    # Print table
    print(f"  {'Type':<28s} {'Count':>7s} {'MedMFE':>7s} {'MedMAE':>7s} {'MFE/MAE':>8s} {'WinR':>5s} {'2xATR':>6s} {'3xATR':>6s} {'5xATR':>6s}")
    for _, r in summary_df.iterrows():
        print(f"  {r['entry_type']:<28s} {int(r['count']):>7,} {r['median_mfe']:>6.1f}% {r['median_mae']:>6.1f}% "
              f"{r['median_mfe_mae_ratio']:>8.2f} {r['win_rate']:>4.1f}% {r['hit_2atr']:>5.1f}% {r['hit_3atr']:>5.1f}% {r['hit_5atr']:>5.1f}%")

    if not _HAVE_MPL:
        return summary_df

    # Charts
    sorted_types = summary_df.sort_values("median_mfe", ascending=False)["entry_type"].tolist()
    short = {t: t.replace("BREAKOUT_","B_").replace("CONTINUATION","CONT").replace("PULLBACK_","PB_").replace("TRENDLINE_","TL_").replace("RETEST_","RT_").replace("ATH_","ATH_") for t in types}

    # MFE boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [valid.loc[valid["entry_type"] == t, "mfe_pct"].dropna().clip(upper=500).values for t in sorted_types]
    ax.boxplot(data, tick_labels=[short[t] for t in sorted_types], vert=True, patch_artist=True,
               boxprops=dict(facecolor="lightgreen", alpha=0.6), showfliers=False)
    ax.set_title("MFE by Entry Type (sorted by median, fliers hidden)")
    ax.set_ylabel("MFE %")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "entry_type_mfe_boxplot.png", dpi=150); plt.close()

    # MAE boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [valid.loc[valid["entry_type"] == t, "mae_pct"].dropna().clip(lower=-80).values for t in sorted_types]
    ax.boxplot(data, tick_labels=[short[t] for t in sorted_types], vert=True, patch_artist=True,
               boxprops=dict(facecolor="salmon", alpha=0.6), showfliers=False)
    ax.set_title("MAE by Entry Type (sorted by median MFE, fliers hidden)")
    ax.set_ylabel("MAE %")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "entry_type_mae_boxplot.png", dpi=150); plt.close()

    # MFE/MAE ratio bar
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [summary_df.loc[summary_df["entry_type"] == t, "median_mfe_mae_ratio"].values[0] for t in sorted_types]
    ax.bar([short[t] for t in sorted_types], vals, color="steelblue", alpha=0.7)
    ax.set_title("Median MFE/MAE Ratio by Entry Type (higher = better signal)")
    ax.set_ylabel("MFE/MAE Ratio")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "entry_type_mfe_mae_ratio.png", dpi=150); plt.close()

    # Hit rate grouped bar
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_types))
    w = 0.25
    h2 = [summary_df.loc[summary_df["entry_type"] == t, "hit_2atr"].values[0] for t in sorted_types]
    h3 = [summary_df.loc[summary_df["entry_type"] == t, "hit_3atr"].values[0] for t in sorted_types]
    h5 = [summary_df.loc[summary_df["entry_type"] == t, "hit_5atr"].values[0] for t in sorted_types]
    ax.bar(x - w, h2, w, label="2×ATR", color="green", alpha=0.7)
    ax.bar(x, h3, w, label="3×ATR", color="blue", alpha=0.7)
    ax.bar(x + w, h5, w, label="5×ATR", color="purple", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([short[t] for t in sorted_types], rotation=30, ha="right")
    ax.set_ylabel("Hit Rate %"); ax.set_title("ATR Target Hit Rates by Entry Type")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "entry_type_hit_rate.png", dpi=150); plt.close()

    return summary_df


def analysis_2_ma_period(df: pd.DataFrame):
    """MA period effect heatmap."""
    print("\n  ═══ Analysis 2: MA Period Effect ═══")
    valid = df[df["exit_reason"] != "no_data"]
    types = sorted(valid["entry_type"].unique())
    periods = [20, 25, 30, 40]

    mfe_grid = np.zeros((len(types), len(periods)))
    mae_grid = np.zeros((len(types), len(periods)))
    for ti, et in enumerate(types):
        for pi, mp in enumerate(periods):
            sub = valid[(valid["entry_type"] == et) & (valid["ma_period"] == mp)]
            mfe_grid[ti, pi] = _safe_median(sub["mfe_pct"])
            mae_grid[ti, pi] = _safe_median(sub["mae_pct"])

    if not _HAVE_MPL:
        return

    short = {t: t.replace("BREAKOUT_","B_").replace("CONTINUATION","CONT").replace("PULLBACK_","PB_").replace("TRENDLINE_","TL_").replace("RETEST_","RT_") for t in types}

    # MFE heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mfe_grid, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(periods))); ax.set_xticklabels(periods)
    ax.set_yticks(range(len(types))); ax.set_yticklabels([short[t] for t in types])
    for ti in range(len(types)):
        for pi in range(len(periods)):
            ax.text(pi, ti, f"{mfe_grid[ti,pi]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("MA Period"); ax.set_title("Median MFE % by Entry Type × MA Period")
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "heatmap_ma_period_mfe.png", dpi=150); plt.close()

    # MAE heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mae_grid, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(periods))); ax.set_xticklabels(periods)
    ax.set_yticks(range(len(types))); ax.set_yticklabels([short[t] for t in types])
    for ti in range(len(types)):
        for pi in range(len(periods)):
            ax.text(pi, ti, f"{mae_grid[ti,pi]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("MA Period"); ax.set_title("Median MAE % by Entry Type × MA Period")
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "heatmap_ma_period_mae.png", dpi=150); plt.close()

    print(f"  Heatmaps saved.")


def analysis_3_ma_type(df: pd.DataFrame):
    """SMA vs EMA comparison."""
    print("\n  ═══ Analysis 3: MA Type Effect ═══")
    valid = df[df["exit_reason"] != "no_data"]
    print(f"  {'Type':<28s} {'SMA MedMFE':>10s} {'EMA MedMFE':>10s} {'Diff':>6s}")
    for et in sorted(valid["entry_type"].unique()):
        sma = valid[(valid["entry_type"] == et) & (valid["ma_type"] == "SMA")]
        ema = valid[(valid["entry_type"] == et) & (valid["ma_type"] == "EMA")]
        s_mfe = _safe_median(sma["mfe_pct"])
        e_mfe = _safe_median(ema["mfe_pct"])
        diff = e_mfe - s_mfe
        print(f"  {et:<28s} {s_mfe:>9.1f}% {e_mfe:>9.1f}% {diff:>+5.1f}%")


def analysis_4_time_to_mfe(df: pd.DataFrame):
    """Time to MFE boxplot."""
    print("\n  ═══ Analysis 4: Time to MFE ═══")
    valid = df[df["exit_reason"] != "no_data"]
    types = sorted(valid["entry_type"].unique())
    if not _HAVE_MPL:
        return
    short = {t: t.replace("BREAKOUT_","B_").replace("CONTINUATION","CONT").replace("PULLBACK_","PB_").replace("TRENDLINE_","TL_").replace("RETEST_","RT_") for t in types}

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [valid.loc[valid["entry_type"] == t, "time_to_mfe_weeks"].dropna().clip(upper=80).values for t in types]
    ax.boxplot(data, tick_labels=[short[t] for t in types], vert=True, patch_artist=True,
               boxprops=dict(facecolor="lightyellow", alpha=0.6), showfliers=False)
    ax.set_title("Time to MFE (weeks) by Entry Type")
    ax.set_ylabel("Weeks")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "time_to_mfe_by_type.png", dpi=150); plt.close()
    print(f"  Chart saved.")


def analysis_5_risk_vs_reward(df: pd.DataFrame):
    """Risk vs MFE scatter."""
    print("\n  ═══ Analysis 5: Risk vs Reward ═══")
    valid = df[df["exit_reason"] != "no_data"].dropna(subset=["risk_pct_at_entry", "mfe_pct"])
    if not _HAVE_MPL:
        return
    sample = valid.sample(min(5000, len(valid)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"PULLBACK_S2": "blue", "VCP_CONTINUATION": "green",
              "BREAKOUT_S1_TO_S2": "orange", "RETEST_SUPPORT": "red",
              "PULLBACK_STRUCTURAL": "purple", "TRENDLINE_BOUNCE": "cyan",
              "ATH_BREAKOUT": "brown"}
    for et in sorted(sample["entry_type"].unique()):
        sub = sample[sample["entry_type"] == et]
        c = colors.get(et, "gray")
        ax.scatter(sub["risk_pct_at_entry"] * 100, sub["mfe_pct"], c=c, alpha=0.4, s=15,
                   label=et.replace("BREAKOUT_","B_").replace("PULLBACK_","PB_"), edgecolors="none")
    ax.set_xlabel("Risk % at Entry"); ax.set_ylabel("MFE %")
    ax.set_title("Risk at Entry vs MFE (5K sample)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "risk_vs_mfe_scatter.png", dpi=150); plt.close()
    print(f"  Chart saved.")


def analysis_6_quality_indicators(df: pd.DataFrame):
    """Quality indicator tercile analysis."""
    print("\n  ═══ Analysis 6: Quality Indicators ═══")
    valid = df[df["exit_reason"] != "no_data"]

    indicators = {
        "RSI at Entry": ("rsi_at_entry", [(0, 40, "Low <40"), (40, 60, "Mid 40-60"), (60, 100, "High >60")]),
        "Volume Ratio": ("volume_ratio_at_entry", [(0, 0.8, "Low <0.8"), (0.8, 1.2, "Mid 0.8-1.2"), (1.2, 999, "High >1.2")]),
        "RS_13w": ("rs_13w", [(-999, 0.5, "Low <0.5"), (0.5, 1.5, "Mid 0.5-1.5"), (1.5, 999, "High >1.5")]),
    }

    # Print table (always)
    for label, (col, buckets) in indicators.items():
        print(f"  {label}:")
        for lo, hi, bname in buckets:
            sub = valid[(valid[col] >= lo) & (valid[col] < hi)]
            print(f"    {bname:<15s}: n={len(sub):>7,}  MedMFE={_safe_median(sub['mfe_pct']):>6.1f}%  MedMAE={_safe_median(sub['mae_pct']):>6.1f}%")

    if not _HAVE_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (label, (col, buckets)) in enumerate(indicators.items()):
        ax = axes[idx // 2][idx % 2]
        mfe_vals = []
        bucket_labels = []
        for lo, hi, bname in buckets:
            sub = valid[(valid[col] >= lo) & (valid[col] < hi)]
            mfe_vals.append(_safe_median(sub["mfe_pct"]))
            bucket_labels.append(f"{bname}\n(n={len(sub):,})")
        ax.bar(bucket_labels, mfe_vals, color="steelblue", alpha=0.7)
        ax.set_title(label); ax.set_ylabel("Median MFE %")
        ax.grid(True, alpha=0.3, axis="y")

    # Momentum (binary)
    ax = axes[1][1]
    for mom_val, bname in [(0, "Bearish (0)"), (1, "Bullish (1)")]:
        sub = valid[valid["momentum_at_entry"] == mom_val]
        ax.bar(f"{bname}\n(n={len(sub):,})", _safe_median(sub["mfe_pct"]), color="steelblue", alpha=0.7)
    ax.set_title("Momentum at Entry"); ax.set_ylabel("Median MFE %")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Quality Indicators vs Median MFE", fontsize=14)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "quality_indicators.png", dpi=150); plt.close()


def analysis_7_retest_specific(df: pd.DataFrame):
    """RETEST_SUPPORT specific analysis."""
    print("\n  ═══ Analysis 7: RETEST-Specific ═══")
    valid = df[(df["exit_reason"] != "no_data") & (df["entry_type"] == "RETEST_SUPPORT")]
    if len(valid) == 0:
        print("  No RETEST entries."); return

    # Timing split
    timing_buckets = [(2, 5, "2-4w"), (5, 9, "5-8w"), (9, 17, "9-16w"), (17, 999, "17w+")]
    print(f"  {'Timing':<10s} {'Count':>7s} {'MedMFE':>8s} {'MedMAE':>8s} {'MFE/MAE':>8s} {'2xATR':>6s}")
    timing_mfe = []
    timing_labels = []
    for lo, hi, label in timing_buckets:
        sub = valid[(valid["weeks_since_breakout"] >= lo) & (valid["weeks_since_breakout"] < hi)]
        ratio = _safe_median(sub["mfe_mae_ratio"])
        h2 = round(sub["hit_2atr"].mean() * 100, 1) if len(sub) > 0 else 0
        print(f"  {label:<10s} {len(sub):>7,} {_safe_median(sub['mfe_pct']):>7.1f}% {_safe_median(sub['mae_pct']):>7.1f}% {ratio:>8.2f} {h2:>5.1f}%")
        timing_mfe.append(_safe_median(sub["mfe_pct"]))
        timing_labels.append(f"{label}\n(n={len(sub):,})")

    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(timing_labels, timing_mfe, color="steelblue", alpha=0.7)
    ax.set_title("RETEST: Median MFE by Weeks Since Breakout")
    ax.set_ylabel("Median MFE %"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "retest_timing.png", dpi=150); plt.close()

    # Proximity split using proximity_atr column
    if "proximity_atr" in valid.columns:
        prox_data = valid.dropna(subset=["proximity_atr"])
        if len(prox_data) > 100:
            prox_buckets = [(0, 0.5, "<0.5 ATR"), (0.5, 1.0, "0.5-1.0"), (1.0, 1.5, "1.0-1.5"), (1.5, 999, ">1.5")]
            print(f"\n  Proximity (ATR units):")
            print(f"  {'Bucket':<12s} {'Count':>7s} {'MedMFE':>8s} {'MedMAE':>8s} {'MFE/MAE':>8s}")
            prox_mfe = []
            prox_labels = []
            for lo, hi, label in prox_buckets:
                sub = prox_data[(prox_data["proximity_atr"] >= lo) & (prox_data["proximity_atr"] < hi)]
                ratio = abs(_safe_median(sub["mfe_pct"]) / _safe_median(sub["mae_pct"])) if _safe_median(sub["mae_pct"]) < -0.01 else 999
                print(f"  {label:<12s} {len(sub):>7,} {_safe_median(sub['mfe_pct']):>7.1f}% {_safe_median(sub['mae_pct']):>7.1f}% {ratio:>8.2f}")
                prox_mfe.append(_safe_median(sub["mfe_pct"]))
                prox_labels.append(f"{label}\n(n={len(sub):,})")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(prox_labels, prox_mfe, color="steelblue", alpha=0.7)
            ax.set_title("RETEST: Median MFE by Proximity to Retest Level (ATR units)")
            ax.set_ylabel("Median MFE %"); ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout(); plt.savefig(OUTPUT_DIR / "retest_proximity.png", dpi=150); plt.close()
    else:
        print(f"  (proximity_atr column not found — skipped)")


def analysis_8_ath_specific(df: pd.DataFrame):
    """ATH_BREAKOUT specific analysis."""
    print("\n  ═══ Analysis 8: ATH-Specific ═══")
    valid = df[(df["exit_reason"] != "no_data") & (df["entry_type"] == "ATH_BREAKOUT")]
    if len(valid) == 0:
        print("  No ATH entries."); return

    buckets = [(0, 1.0, "Tight <1ATR"), (1.0, 2.0, "Med 1-2ATR"), (2.0, 999, "Loose >2ATR")]
    print(f"  {'Consolidation':<15s} {'Count':>7s} {'MedMFE':>8s} {'MedMAE':>8s} {'2xATR':>6s}")
    mfe_vals = []
    labels = []
    for lo, hi, label in buckets:
        sub = valid[(valid["consolidation_range_atr"] >= lo) & (valid["consolidation_range_atr"] < hi)]
        h2 = round(sub["hit_2atr"].mean() * 100, 1) if len(sub) > 0 else 0
        print(f"  {label:<15s} {len(sub):>7,} {_safe_median(sub['mfe_pct']):>7.1f}% {_safe_median(sub['mae_pct']):>7.1f}% {h2:>5.1f}%")
        mfe_vals.append(_safe_median(sub["mfe_pct"]))
        labels.append(f"{label}\n(n={len(sub):,})")

    # By ath_type
    for at in valid["ath_type"].dropna().unique():
        sub = valid[valid["ath_type"] == at]
        print(f"  {at}: n={len(sub):,}, MedMFE={_safe_median(sub['mfe_pct']):.1f}%, MedMAE={_safe_median(sub['mae_pct']):.1f}%")

    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, mfe_vals, color="steelblue", alpha=0.7)
    ax.set_title("ATH_BREAKOUT: Median MFE by Consolidation Tightness")
    ax.set_ylabel("Median MFE %"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "ath_consolidation_quality.png", dpi=150); plt.close()


def analysis_9_structural_specific(df: pd.DataFrame):
    """PULLBACK_STRUCTURAL specific analysis."""
    print("\n  ═══ Analysis 9: STRUCTURAL-Specific ═══")
    valid = df[(df["exit_reason"] != "no_data") & (df["entry_type"] == "PULLBACK_STRUCTURAL")]
    if len(valid) == 0:
        print("  No STRUCTURAL entries."); return

    buckets = [(1.0, 2.0, "Close 1-2ATR"), (2.0, 4.0, "Med 2-4ATR"), (4.0, 999, "Far >4ATR")]
    mfe_vals = []
    labels = []
    print(f"  {'MA Distance':<15s} {'Count':>7s} {'MedMFE':>8s} {'MedMAE':>8s}")
    for lo, hi, label in buckets:
        sub = valid[(valid["distance_from_ma_atr"] >= lo) & (valid["distance_from_ma_atr"] < hi)]
        print(f"  {label:<15s} {len(sub):>7,} {_safe_median(sub['mfe_pct']):>7.1f}% {_safe_median(sub['mae_pct']):>7.1f}%")
        mfe_vals.append(_safe_median(sub["mfe_pct"]))
        labels.append(f"{label}\n(n={len(sub):,})")

    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, mfe_vals, color="steelblue", alpha=0.7)
    ax.set_title("PULLBACK_STRUCTURAL: Median MFE by Distance from MA")
    ax.set_ylabel("Median MFE %"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "structural_ma_distance.png", dpi=150); plt.close()


def analysis_10_seasonality(df: pd.DataFrame):
    """Seasonality: MFE by calendar month."""
    print("\n  ═══ Analysis 10: Seasonality ═══")
    valid = df[df["exit_reason"] != "no_data"].copy()
    valid["month"] = pd.to_datetime(valid["week_date"]).dt.month
    monthly = valid.groupby("month")["mfe_pct"].median()

    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly.index, monthly.values, color="steelblue", alpha=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Median MFE by Entry Month (all types)")
    ax.set_ylabel("Median MFE %"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "seasonality.png", dpi=150); plt.close()
    print(f"  Chart saved.")


def analysis_11_is_vs_oos(df: pd.DataFrame):
    """IS vs OOS comparison."""
    print("\n  ═══ Analysis 11: IS vs OOS ═══")
    valid = df[df["exit_reason"] != "no_data"].copy()
    valid["period"] = np.where(pd.to_datetime(valid["week_date"]) >= IS_SPLIT, "IS", "OOS")
    types = sorted(valid["entry_type"].unique())

    print(f"  {'Type':<28s} {'OOS MedMFE':>10s} {'IS MedMFE':>10s} {'OOS n':>7s} {'IS n':>7s}")
    oos_vals = []
    is_vals = []
    for et in types:
        oos = valid[(valid["entry_type"] == et) & (valid["period"] == "OOS")]
        is_ = valid[(valid["entry_type"] == et) & (valid["period"] == "IS")]
        o_mfe = _safe_median(oos["mfe_pct"])
        i_mfe = _safe_median(is_["mfe_pct"])
        print(f"  {et:<28s} {o_mfe:>9.1f}% {i_mfe:>9.1f}% {len(oos):>7,} {len(is_):>7,}")
        oos_vals.append(o_mfe)
        is_vals.append(i_mfe)

    if not _HAVE_MPL:
        return

    short = [t.replace("BREAKOUT_","B_").replace("CONTINUATION","CONT").replace("PULLBACK_","PB_").replace("TRENDLINE_","TL_").replace("RETEST_","RT_") for t in types]
    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.2, oos_vals, 0.35, label="OOS (pre-2020)", color="steelblue", alpha=0.7)
    ax.bar(x + 0.2, is_vals, 0.35, label="IS (2020+)", color="coral", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_ylabel("Median MFE %"); ax.set_title("IS vs OOS: Median MFE by Entry Type")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "is_vs_oos.png", dpi=150); plt.close()


def analysis_12_yearly_mfe(df: pd.DataFrame):
    """Year-by-year MFE lines."""
    print("\n  ═══ Analysis 12: Yearly MFE ═══")
    valid = df[df["exit_reason"] != "no_data"].copy()
    valid["year"] = pd.to_datetime(valid["week_date"]).dt.year
    types = sorted(valid["entry_type"].unique())
    years = sorted(valid["year"].unique())

    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"PULLBACK_S2": "blue", "VCP_CONTINUATION": "green",
              "BREAKOUT_S1_TO_S2": "orange", "RETEST_SUPPORT": "red",
              "PULLBACK_STRUCTURAL": "purple", "TRENDLINE_BOUNCE": "cyan",
              "ATH_BREAKOUT": "brown"}
    for et in types:
        sub = valid[valid["entry_type"] == et]
        yearly = sub.groupby("year")["mfe_pct"].median()
        ax.plot(yearly.index, yearly.values, marker="o", markersize=3, linewidth=1.2,
                color=colors.get(et, "gray"),
                label=et.replace("BREAKOUT_","B_").replace("PULLBACK_","PB_").replace("CONTINUATION","CONT"))
    ax.set_xlabel("Year"); ax.set_ylabel("Median MFE %")
    ax.set_title("Median MFE by Year and Entry Type")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "yearly_mfe.png", dpi=150); plt.close()
    print(f"  Chart saved.")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY JSON
# ══════════════════════════════════════════════════════════════════════

def build_summary(df: pd.DataFrame, entry_summary: pd.DataFrame) -> dict:
    valid = df[df["exit_reason"] != "no_data"]

    by_type = {}
    for _, r in entry_summary.iterrows():
        by_type[r["entry_type"]] = {
            "count": int(r["count"]),
            "median_mfe": round(r["median_mfe"], 2),
            "median_mae": round(r["median_mae"], 2),
            "mfe_mae_ratio": round(r["median_mfe_mae_ratio"], 2),
            "win_rate": round(r["win_rate"], 1),
            "hit_2atr": round(r["hit_2atr"], 1),
            "hit_3atr": round(r["hit_3atr"], 1),
            "hit_5atr": round(r["hit_5atr"], 1),
            "median_time_to_mfe": round(r["median_time_to_mfe"], 1),
            "mean_capture": round(r["mean_capture"], 2),
        }

    by_ma_period = {}
    for mp in [20, 25, 30, 40]:
        sub = valid[valid["ma_period"] == mp]
        by_ma_period[str(mp)] = {
            "count": len(sub),
            "median_mfe": round(_safe_median(sub["mfe_pct"]), 2),
            "median_mae": round(_safe_median(sub["mae_pct"]), 2),
        }

    by_ma_type = {}
    for mt in ["SMA", "EMA"]:
        sub = valid[valid["ma_type"] == mt]
        by_ma_type[mt] = {
            "count": len(sub),
            "median_mfe": round(_safe_median(sub["mfe_pct"]), 2),
        }

    # Top configs
    top = valid.groupby(["ma_type", "ma_period", "entry_type"]).agg(
        median_mfe=("mfe_pct", "median"),
        mfe_mae_ratio=("mfe_mae_ratio", "median"),
        count=("mfe_pct", "size"),
    ).reset_index().sort_values("mfe_mae_ratio", ascending=False).head(10)

    top_configs = []
    for _, r in top.iterrows():
        top_configs.append({
            "config": f"{r['ma_type']}_{int(r['ma_period'])} + {r['entry_type']}",
            "median_mfe": round(r["median_mfe"], 2),
            "mfe_mae_ratio": round(r["mfe_mae_ratio"], 2),
            "count": int(r["count"]),
        })

    return {
        "total_trades_simulated": len(df),
        "valid_trades": len(valid),
        "no_data_trades": int((df["exit_reason"] == "no_data").sum()),
        "by_entry_type": by_type,
        "by_ma_period": by_ma_period,
        "by_ma_type": by_ma_type,
        "top_configs": top_configs,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def build_entry_stats(data_dir: Path | str | None = None,
                      output_dir: Path | str | None = None) -> Path:
    """Regenerate ``trade_stats_all.csv`` (and diagnostics) from the data bundle.

    Args:
        data_dir: bundle directory holding ``entries_all.csv`` + ``stock_weekly_cache/``.
            If ``None``, resolves from :func:`skysurf.reproduction._paths.data_dir`.
        output_dir: where charts/summary land. If ``None``, resolves from
            :func:`skysurf.reproduction._paths.output_dir` (``/entry_stats``).

    Returns the path to the written ``trade_stats_all.csv``.
    """
    global ENGINE_A_DIR, STOCK_CACHE_DIR, OUTPUT_DIR
    if data_dir is not None:
        _paths.set_data_dir(data_dir)
    ENGINE_A_DIR = _paths.require_data_dir()
    STOCK_CACHE_DIR = ENGINE_A_DIR / "stock_weekly_cache"
    OUTPUT_DIR = (Path(output_dir).expanduser().resolve()
                  if output_dir is not None else _paths.output_dir() / "entry_stats")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 70)
    print("VAL-ENTRY-STATS: Per-Trade Entry Quality Analysis")
    print("=" * 70)
    print(f"  Input: {ENGINE_A_DIR}")
    print(f"  Output (trade_stats_all.csv): {ENGINE_A_DIR}")
    print(f"  Output (charts/summary): {OUTPUT_DIR}")
    print(f"  Charts: {'on' if _HAVE_MPL else 'OFF (install skysurf[reproduction])'}")
    print(f"  Exit config: buffer={EXIT_CONFIG['exit_atr_buffer']}, "
          f"trigger={EXIT_CONFIG['exit_gtt_field']}, max_weeks={EXIT_CONFIG['max_holding_weeks']}")
    print()

    # Load entries
    print("[Step 1] Loading entries ...", flush=True)
    entries_df = pd.read_csv(ENGINE_A_DIR / "entries_all.csv")
    print(f"  Loaded {len(entries_df):,} entries")

    # Initialize cache
    cache = StockCacheManager(STOCK_CACHE_DIR)

    # Run simulations
    print("\n[Step 2] Simulating all trades ...", flush=True)
    trade_df = run_all_simulations(entries_df, cache)

    # Save raw results -> data bundle (consumed by the dynamic type-prior)
    trade_stats_path = ENGINE_A_DIR / "trade_stats_all.csv"
    trade_df.to_csv(trade_stats_path, index=False)
    print(f"\n  Saved trade_stats_all.csv ({len(trade_df):,} rows) -> {trade_stats_path}")

    # Verify row count
    assert len(trade_df) == len(entries_df), \
        f"Row count mismatch: {len(trade_df)} vs {len(entries_df)}"
    no_data = (trade_df["exit_reason"] == "no_data").sum()
    print(f"  no_data entries: {no_data:,}")

    # Run analyses
    print("\n[Step 3] Running analyses ...", flush=True)
    entry_summary = analysis_1_entry_type_comparison(trade_df)
    analysis_2_ma_period(trade_df)
    analysis_3_ma_type(trade_df)
    analysis_4_time_to_mfe(trade_df)
    analysis_5_risk_vs_reward(trade_df)
    analysis_6_quality_indicators(trade_df)
    analysis_7_retest_specific(trade_df)
    analysis_8_ath_specific(trade_df)
    analysis_9_structural_specific(trade_df)
    analysis_10_seasonality(trade_df)
    analysis_11_is_vs_oos(trade_df)
    analysis_12_yearly_mfe(trade_df)

    # Save summary
    summary = build_summary(trade_df, entry_summary)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary.json")

    # Print top configs
    print(f"\n  ═══ Top 10 Configs by MFE/MAE Ratio ═══")
    for tc in summary["top_configs"]:
        print(f"    {tc['config']:<40s}  MFE/MAE={tc['mfe_mae_ratio']:.2f}  MedMFE={tc['median_mfe']:.1f}%  n={tc['count']:,}")

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\n{'='*70}")
    n_charts = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"VAL-ENTRY-STATS complete. {m}m{s}s | {len(trade_df):,} trades | {n_charts} charts")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
    return trade_stats_path


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="skysurf-build-stats",
        description=(
            "Regenerate trade_stats_all.csv (per-trade MFE/MAE entry stats, the "
            "dynamic type-prior input) from the reproduction data bundle's "
            "entries_all.csv + stock_weekly_cache/."
        ),
    )
    p.add_argument("--data", metavar="DIR", default=None,
                   help="Data-bundle dir with entries_all.csv + stock_weekly_cache/. "
                        "Defaults to $SKYSURF_REPRO_DATA, then ./skysurf-repro-data.")
    p.add_argument("--out", metavar="DIR", default=None,
                   help="Where charts/summary land. Defaults to "
                        "$SKYSURF_REPRO_OUTPUT/entry_stats.")
    args = p.parse_args(argv)
    try:
        path = build_entry_stats(data_dir=args.data, output_dir=args.out)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\ntrade_stats_all.csv written to: {path}")
    return 0


# Backwards-compatible alias matching the research script's entrypoint name.
main = _cli


if __name__ == "__main__":
    raise SystemExit(_cli())
