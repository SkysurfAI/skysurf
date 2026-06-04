"""VAL-ENGINE-A v2: Data Pipeline + Entry Detection Grid (8-config, 7-detector).

Pre-computes all entries across 8 indicator configs (MA type x MA period) with
per-detector swing orders and 4 new entry types (RETEST_SUPPORT, PULLBACK_STRUCTURAL,
TRENDLINE_BOUNCE, ATH_BREAKOUT). Saves everything to CSV so that VAL-ENGINE-B
can run cheap portfolio simulations.

Refactored for the open-source reproduction: reads raw daily OHLCV from files
(``stocks_daily`` / ``benchmark_daily``) instead of a SQL database, and writes
all intermediates into the reproduction data bundle.

Usage:
    skysurf-build --ohlcv /path/to/raw_ohlcv --out /path/to/skysurf-repro-data
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from skysurf.reproduction import _paths

# ── Verbatim research closure (indicators / regime / detectors) ───────
from ._calc.skysurf_calc import resample_weekly, detect_trendlines
from ._calc.technical_indicators import calculate_atr, calculate_rsi
from ._calc.regime_classifier import (
    compute_ema_position,
    compute_ema_direction,
    classify_regime,
)
from ._detect.swing_methods import method_a_argrelextrema
from ._detect.ma_configs import (
    compute_ma_series,
    compute_ma_slope_direction,
)
from ._detect.stage_classifier import classify_stages_single_ma
from ._detect.breakout_detector import detect_breakouts
from ._detect.pullback_detector import detect_pullbacks
from ._detect.vcp_detector import detect_vcp_continuations
from ._detect.entry_quality import compute_entry_quality
from ._detect.phase0_regime import classify_all_weeks

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

START_DATE = "2007-01-01"
# END_DATE is overridable via env var so the V1-refresh cron (and the
# /api/v1/brain/_validate_v1_refresh_run admin endpoint) can extend the
# window weekly without editing source. Default keeps research-script
# invocations stable.
END_DATE = os.getenv("END_DATE", "2026-03-21")
MARKET_CAP_THRESHOLD_CR = 1500
NIFTY_REFERENCE_LEVEL = 23000
EXCLUDE_SME = True
MIN_WEEKLY_BARS = 52

# Entry detection defaults (same as v1)
BREAKOUT_CEILING = "ma_plus_atr"
BREAKOUT_BASE_MIN = 4
PULLBACK_DEPTH = "pb_swing_low"
PULLBACK_CONFIRM = "close_above_ma"
PULLBACK_MIN_S2 = 4
VCP_CONSOL_MIN = 3
VCP_CONTRACTION = "vcp_any"
VCP_VOLUME = "vol_declining"

# Regime classification thresholds
REGIME_HIGH = 60.0
REGIME_LOW = 35.0
REGIME_DIR_LOOKBACK = 8
REGIME_DIR_THRESH = 5.0

# Output paths — the builder writes every intermediate (caches, regimes,
# entries_all.csv, ...) into the reproduction data-bundle directory, the same
# directory skysurf-reproduce later reads from. These are placeholders resolved
# at import; main() re-resolves and creates them at run time, so merely
# importing this module has no filesystem side effects.
OUTPUT_DIR = _paths.data_dir()
CACHE_DIR = OUTPUT_DIR / "stock_weekly_cache"
SWING_CACHE_DIR = OUTPUT_DIR / "swing_cache"

# ══════════════════════════════════════════════════════════════════════
# CONFIG GRID (8 configs: 2 MA types x 4 periods)
# ══════════════════════════════════════════════════════════════════════

INDICATOR_CONFIGS: list[dict] = []
_config_id = 0
for _ma_type in ["SMA", "EMA"]:
    for _ma_period in [20, 25, 30, 40]:
        INDICATOR_CONFIGS.append({
            "config_id": _config_id,
            "ma_type": _ma_type,
            "ma_period": _ma_period,
        })
        _config_id += 1

# Per-detector swing orders (all start at 8/5 for baseline)
DETECTOR_SWING = {
    "breakout":             {"major": 8, "minor": 5},
    "pullback_ma":          {"major": 8, "minor": 5},
    "vcp":                  {"major": 8, "minor": 5},
    "retest":               {"major": 8, "minor": 5},
    "pullback_structural":  {"major": 8, "minor": 5},
    "ath_breakout":         {"major": 8, "minor": 5},
    "trendline":            None,
}

# Compute swings at ALL orders for analysis cache
ALL_SWING_ORDERS = [3, 5, 6, 7, 8, 10]

# All unique (ma_type, ma_period) combos
ALL_MA_COMBOS = sorted({(cfg["ma_type"], cfg["ma_period"]) for cfg in INDICATOR_CONFIGS})


def ma_key(ma_type: str, ma_period: int) -> str:
    return f"{ma_type.lower()}_{ma_period}"


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_quarter(dt) -> str:
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"


def count_consecutive_stage2(stages_series: pd.Series, idx: int) -> int:
    count = 0
    vals = stages_series.values
    i = idx
    while i >= 0 and vals[i] == "stage2":
        count += 1
        i -= 1
    return count


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ══════════════════════════════════════════════════════════════════════
# NEW DETECTORS
# ══════════════════════════════════════════════════════════════════════

def detect_retest_support(
    df: pd.DataFrame,
    stages_series: pd.Series,
    major_swing_highs: list[int],
    atr_series: pd.Series,
    min_gap: int = 4,
) -> list[dict]:
    """Detect retest of broken swing high levels as support."""
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    opens = df["Open"].values
    n = len(df)

    used_levels: set[float] = set()
    last_entry_idx = -min_gap - 1

    for i in range(2, n):
        if stages_series.iloc[i] != "stage2":
            continue
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr <= 0:
            continue
        if i - last_entry_idx < min_gap:
            continue

        close_i = closes[i]
        low_i = lows[i]
        open_i = opens[i]

        for sh_idx in major_swing_highs:
            sh_level = float(highs[sh_idx])
            if sh_level >= close_i:
                continue  # not broken yet
            if sh_level in used_levels:
                continue

            # Find first break above this level
            break_week = None
            for b in range(sh_idx + 1, i):
                if closes[b] > sh_level:
                    break_week = b
                    break
            if break_week is None or break_week >= i - 1:
                continue  # no break or too recent

            # Price pulling back toward level
            if low_i > sh_level * (1 + 1.5 * atr / sh_level):
                continue  # too far above
            if close_i <= sh_level:
                continue  # broken below — not a bounce
            if close_i <= open_i and close_i <= sh_level * 1.005:
                continue  # bearish candle right at level

            # Post-breakout high for pullback depth
            post_break_high = float(np.max(highs[break_week:i]))
            pullback_depth_pct = round((post_break_high - low_i) / post_break_high * 100, 2) if post_break_high > 0 else 0

            # Volume ratio
            vol_start = max(0, i - 20)
            trailing_vol = df["Volume"].values[vol_start:i]
            avg_vol = float(np.mean(trailing_vol)) if len(trailing_vol) > 0 else 1.0
            bounce_vol = round(float(df["Volume"].values[i]) / avg_vol, 2) if avg_vol > 0 else 0

            used_levels.add(sh_level)
            last_entry_idx = i

            events.append({
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": round(close_i, 2),
                "entry_type": "RETEST_SUPPORT",
                "retest_level": round(sh_level, 2),
                "weeks_since_breakout": i - break_week,
                "pullback_depth_pct": pullback_depth_pct,
                "bounce_volume_ratio": bounce_vol,
                "stop_atr_level": round(sh_level - 1.0 * atr, 2),
            })
            break  # one entry per week

    return events


def detect_pullback_structural(
    df: pd.DataFrame,
    stages_series: pd.Series,
    major_swing_highs: list[int],
    major_swing_lows: list[int],
    ma_series: pd.Series,
    atr_series: pd.Series,
    min_s2: int = 4,
    min_gap: int = 4,
) -> list[dict]:
    """Detect pullbacks to structural support (not MA)."""
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    used_levels: set[float] = set()
    last_entry_idx = -min_gap - 1

    # Build consecutive s2 counter
    s2_count = np.zeros(n, dtype=int)
    for i in range(n):
        if stages_series.iloc[i] == "stage2":
            s2_count[i] = (s2_count[i - 1] + 1) if i > 0 else 1

    # Gather candidate support levels: old swing highs (turned support) + swing lows
    for i in range(2, n):
        if stages_series.iloc[i] != "stage2":
            continue
        if s2_count[i] < min_s2:
            continue
        if i - last_entry_idx < min_gap:
            continue

        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr <= 0:
            continue
        close_i = closes[i]
        low_i = lows[i]
        ma_val = float(ma_series.iloc[i]) if not pd.isna(ma_series.iloc[i]) else close_i

        candidates = []
        # Old swing highs below close AND below MA (resistance turned support)
        for sh_idx in major_swing_highs:
            if sh_idx >= i:
                continue
            sh_level = float(highs[sh_idx])
            if sh_level < close_i and sh_level < ma_val:
                candidates.append((sh_level, "swing_high_turned_support", sh_idx))
        # Swing lows below close
        for sl_idx in major_swing_lows:
            if sl_idx >= i:
                continue
            sl_level = float(lows[sl_idx])
            if sl_level < close_i:
                candidates.append((sl_level, "swing_low_support", sl_idx))

        # Sort by proximity to current price (closest first)
        candidates.sort(key=lambda x: abs(close_i - x[0]))

        for level, support_type, origin_idx in candidates:
            if level in used_levels:
                continue
            # Price near level
            if low_i > level * (1 + 1.5 * atr / level):
                continue  # too far above
            if close_i <= level:
                continue  # broken below
            # Level NOT near MA (must be structural, not MA pullback)
            if abs(level - ma_val) <= 1.0 * atr:
                continue

            distance_from_ma_atr = round(abs(level - ma_val) / atr, 2) if atr > 0 else 0
            # Level significance: weeks since the swing was created
            level_age = i - origin_idx

            used_levels.add(level)
            last_entry_idx = i

            events.append({
                "week_idx": i,
                "date": str(df.index[i].date()),
                "close": round(close_i, 2),
                "entry_type": "PULLBACK_STRUCTURAL",
                "support_level": round(level, 2),
                "support_type": support_type,
                "distance_from_ma_atr": distance_from_ma_atr,
                "level_significance_weeks": level_age,
                "stop_atr_level": round(level - 1.0 * atr, 2),
            })
            break  # one per week

    return events


def precompute_trendlines(
    df: pd.DataFrame,
    atr_series: pd.Series,
    swing_lows_data: list[dict] | None,
    interval: int = 26,
) -> list[dict | None]:
    """Precompute trendline data ONCE per stock at fixed intervals.

    Returns a list of length len(df), where each element is the trendline
    result dict (or None) valid at that week. Expensive trendln calls happen
    only every `interval` weeks.
    """
    n = len(df)
    closes = df["Close"].values
    tl_cache: list[dict | None] = [None] * n

    if n < 26:
        return tl_cache

    last_result = None
    last_week = -interval - 1

    # Cap lookback to 104 weeks (2 years) to keep trendln fast
    MAX_TL_LOOKBACK = 104

    for i in range(26, n):
        atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr_val <= 0:
            tl_cache[i] = last_result
            continue

        if i - last_week >= interval:
            start_idx = max(0, i + 1 - MAX_TL_LOOKBACK)
            tl_result = detect_trendlines(
                df.iloc[start_idx:i + 1],
                float(closes[i]),
                float(closes[i - 1]) if i > 0 else float(closes[i]),
                atr_val,
                swing_lows_data,
            )
            last_result = tl_result
            last_week = i

        tl_cache[i] = last_result

    return tl_cache


def detect_trendline_bounce(
    df: pd.DataFrame,
    stages_series: pd.Series,
    atr_series: pd.Series,
    tl_cache: list[dict | None],
    min_gap: int = 4,
) -> list[dict]:
    """Detect bounces off rising support trendlines using precomputed cache."""
    events = []
    closes = df["Close"].values
    lows = df["Low"].values
    n = len(df)

    if n < 26:
        return events

    last_entry_idx = -min_gap - 1

    for i in range(26, n):
        if stages_series.iloc[i] != "stage2":
            continue
        if i - last_entry_idx < min_gap:
            continue

        atr_val = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr_val <= 0:
            continue

        cached_tl = tl_cache[i]
        if cached_tl is None:
            continue

        rising = cached_tl.get("rising_support")
        if rising is None:
            continue

        projected = rising.get("current_projected_level")
        if projected is None or projected <= 0:
            continue

        slope = rising.get("slope_per_week", 0) or 0
        # Approximate projection (cache may be from a few weeks ago)
        projected_at_i = projected + slope * 2  # rough forward projection

        close_i = closes[i]
        low_i = lows[i]

        if low_i > projected_at_i * (1 + 1.0 * atr_val / projected_at_i):
            continue
        if close_i <= projected_at_i * 0.97:
            continue

        span = rising.get("span_weeks", 0) or 0
        anchors = rising.get("anchor_points", 0)
        anchor_count = len(anchors) if isinstance(anchors, (list, tuple)) else int(anchors or 0)
        slope_pct_month = round(slope / projected_at_i * 100 * 4.33, 2) if projected_at_i > 0 else 0
        dist_atr = round((low_i - projected_at_i) / atr_val, 2) if atr_val > 0 else 0

        last_entry_idx = i
        events.append({
            "week_idx": i,
            "date": str(df.index[i].date()),
            "close": round(close_i, 2),
            "entry_type": "TRENDLINE_BOUNCE",
            "trendline_slope_pct_month": slope_pct_month,
            "trendline_anchor_count": anchor_count,
            "trendline_age_weeks": span,
            "distance_from_trendline_atr": dist_atr,
            "stop_atr_level": round(projected_at_i - 1.0 * atr_val, 2),
        })

    return events


def detect_ath_breakout(
    df: pd.DataFrame,
    stages_series: pd.Series,
    atr_series: pd.Series,
    major_swing_highs: list[int],
    min_gap: int = 4,
) -> list[dict]:
    """Detect all-time or 52-week high breakouts from tight consolidation.

    Fixed v2: Uses Close (not High) for ATH comparison, requires tight 3-week
    consolidation immediately before breakout, volume confirmation, ATR-based stop.
    """
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values
    n = len(df)

    last_entry_idx = -min_gap - 1

    for i in range(8, n):
        if stages_series.iloc[i] != "stage2":
            continue
        if i - last_entry_idx < min_gap:
            continue

        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
        if atr <= 0:
            continue

        close_i = closes[i]

        # FIX: Use Close (not High) for ATH comparison — breakout confirmed by weekly close
        all_prior_close_max = float(np.max(closes[:i])) if i > 0 else 0
        start_52 = max(0, i - 52)
        close_52w_max = float(np.max(closes[start_52:i])) if start_52 < i else 0

        if close_i <= all_prior_close_max and close_i <= close_52w_max:
            continue  # close didn't exceed prior ATH close

        ath_type = "all_time" if close_i > all_prior_close_max else "52_week"

        # Check for overhead resistance (no major swing highs above)
        has_overhead = any(
            highs[sh] > close_i for sh in major_swing_highs if sh > i
        )
        if has_overhead:
            continue

        # FIX: Tight consolidation in the 3 weeks immediately before breakout
        # Total range of [i-3, i-2, i-1] must be < 1.0 × ATR
        if i < 3:
            continue
        consol_start = i - 3
        consol_end = i - 1
        consol_high = float(np.max(highs[consol_start:consol_end + 1]))
        consol_low = float(np.min(lows[consol_start:consol_end + 1]))
        consol_range = consol_high - consol_low
        consol_range_atr = round(consol_range / atr, 2) if atr > 0 else 0

        if consol_range >= 1.0 * atr:
            continue  # consolidation not tight enough

        # FIX: Volume confirmation — breakout week must have above-average volume
        vol_start = max(0, i - 20)
        trailing_vol = volumes[vol_start:i]
        avg_vol = float(np.mean(trailing_vol)) if len(trailing_vol) > 0 else 1.0
        breakout_vol_ratio = round(float(volumes[i]) / avg_vol, 2) if avg_vol > 0 else 0

        if avg_vol > 0 and float(volumes[i]) <= 1.3 * avg_vol:
            continue  # insufficient volume on breakout

        # FIX: ATR-based stop instead of distant consolidation low
        stop_level = round(close_i - 2.0 * atr, 2)

        last_entry_idx = i
        events.append({
            "week_idx": i,
            "date": str(df.index[i].date()),
            "close": round(close_i, 2),
            "entry_type": "ATH_BREAKOUT",
            "consolidation_weeks": 3,
            "consolidation_range_atr": consol_range_atr,
            "consolidation_low": round(consol_low, 2),
            "breakout_volume_ratio": breakout_vol_ratio,
            "ath_type": ath_type,
            "stop_atr_level": stop_level,
        })

    return events


# ══════════════════════════════════════════════════════════════════════
# STEP 1-3: DATA LOADING, UNIVERSE, BENCHMARK (same as v1)
# ══════════════════════════════════════════════════════════════════════

def _read_ohlcv_table(ohlcv_dir: Path, name: str) -> pd.DataFrame:
    """Read ``<name>.parquet`` (preferred) or ``<name>.csv`` from the OHLCV input dir."""
    pq = ohlcv_dir / f"{name}.parquet"
    csv = ohlcv_dir / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"Raw OHLCV input not found: expected '{pq.name}' or '{csv.name}' in {ohlcv_dir}. "
        "See docs/reproduction-data-bundle.md for the raw daily-OHLCV schema."
    )


def step1_load_data(ohlcv_dir: Path):
    """Load raw daily OHLCV from files (replaces the original SQL reads).

    Expects ``stocks_daily`` and ``benchmark_daily`` (``.parquet`` or ``.csv``)
    under ``ohlcv_dir``. Schema in docs/reproduction-data-bundle.md.
    """
    print("=" * 70)
    print("STEP 1: Loading raw daily OHLCV (stocks_daily, benchmark_daily)")
    print("=" * 70)

    stock_df = _read_ohlcv_table(ohlcv_dir, "stocks_daily")
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    stock_df = stock_df[
        (stock_df["date"] >= START_DATE) & (stock_df["date"] <= END_DATE)
    ].sort_values(["ticker", "date"]).reset_index(drop=True)

    if EXCLUDE_SME and "has_sme_history" in stock_df.columns:
        sme_tickers = set(stock_df.loc[stock_df["has_sme_history"] == True, "ticker"].unique())
        stock_df = stock_df[~stock_df["ticker"].isin(sme_tickers)]
        print(f"  SME-excluded tickers: {len(sme_tickers)}")

    print(f"  Total rows: {len(stock_df):,}")
    print(f"  Unique tickers: {stock_df['ticker'].nunique()}")
    print(f"  Date range: {stock_df['date'].min()} to {stock_df['date'].max()}")

    bench_df = _read_ohlcv_table(ohlcv_dir, "benchmark_daily")
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df[
        (bench_df["date"] >= START_DATE) & (bench_df["date"] <= END_DATE)
    ].sort_values("date").reset_index(drop=True)

    print(f"  Benchmark rows: {len(bench_df):,}")
    print()
    return stock_df, bench_df


def step2_quarterly_universe(stock_df: pd.DataFrame, bench_df: pd.DataFrame):
    print("=" * 70)
    print("STEP 2: Building quarterly universe (Nifty-scaled market cap)")
    print("=" * 70)

    bench_daily = bench_df.copy()
    bench_daily["date"] = pd.to_datetime(bench_daily["date"])
    bench_daily = bench_daily.sort_values("date")
    bench_daily["quarter"] = bench_daily["date"].apply(get_quarter)
    quarter_first = bench_daily.groupby("quarter").first().reset_index()

    ticker_meta = stock_df.groupby("ticker").agg({
        "market_cap": "max",
        "primary_sector": "first",
        "has_demerger": "first",
        "demerger_no_trade_until": "first",
    }).reset_index()

    universe_rows = []
    for _, qrow in quarter_first.iterrows():
        qstr = qrow["quarter"]
        nifty_at_q = float(qrow["close"])
        threshold_inr = MARKET_CAP_THRESHOLD_CR * 1e7 * (nifty_at_q / NIFTY_REFERENCE_LEVEL)
        qualifying = ticker_meta[ticker_meta["market_cap"] >= threshold_inr]
        for _, trow in qualifying.iterrows():
            universe_rows.append({
                "quarter": qstr,
                "ticker": trow["ticker"],
                "market_cap": trow["market_cap"],
                "primary_sector": trow["primary_sector"],
                "threshold_inr": threshold_inr,
                "nifty_at_quarter": nifty_at_q,
            })

    universe_df = pd.DataFrame(universe_rows)
    universe_df.to_csv(OUTPUT_DIR / "universe_quarterly.csv", index=False)

    q_counts = universe_df.groupby("quarter")["ticker"].nunique()
    print(f"  Qualifying tickers per quarter: min={q_counts.min()}, max={q_counts.max()}, mean={q_counts.mean():.0f}")

    qualifying_by_quarter: dict[str, set[str]] = {}
    for qstr, grp in universe_df.groupby("quarter"):
        qualifying_by_quarter[qstr] = set(grp["ticker"])

    ticker_metadata: dict[str, dict] = {}
    for _, row in ticker_meta.iterrows():
        ticker_metadata[row["ticker"]] = {
            "has_demerger": row["has_demerger"],
            "demerger_no_trade_until": row["demerger_no_trade_until"],
            "primary_sector": row["primary_sector"],
            "market_cap": row["market_cap"],
        }

    print()
    return qualifying_by_quarter, ticker_metadata, universe_df


def step3_benchmark(bench_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 3: Processing Nifty benchmark (weekly resample + indicators)")
    print("=" * 70)

    nifty_daily = bench_df.copy()
    nifty_daily["date"] = pd.to_datetime(nifty_daily["date"])
    nifty_daily = nifty_daily.set_index("date").sort_index()
    nifty_daily = nifty_daily.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    nifty_weekly = resample_weekly(nifty_daily)
    nifty_weekly["atr_14"] = calculate_atr(
        nifty_weekly["High"], nifty_weekly["Low"], nifty_weekly["Close"], window=14
    )
    for period in [20, 25, 30, 40]:
        nifty_weekly[f"sma_{period}"] = nifty_weekly["Close"].rolling(period).mean()
        nifty_weekly[f"ema_{period}"] = nifty_weekly["Close"].ewm(span=period, adjust=False).mean()

    save_cols = ["Open", "High", "Low", "Close", "Volume", "atr_14",
                 "sma_20", "sma_25", "sma_30", "sma_40",
                 "ema_20", "ema_25", "ema_30", "ema_40"]
    nifty_out = nifty_weekly[save_cols].copy()
    nifty_out.index.name = "date"
    nifty_out.to_csv(OUTPUT_DIR / "nifty_weekly.csv")

    print(f"  Nifty weekly bars: {len(nifty_weekly)}")
    print(f"  Date range: {nifty_weekly.index[0].date()} to {nifty_weekly.index[-1].date()}")
    print()
    return nifty_weekly


# ══════════════════════════════════════════════════════════════════════
# STEP 4: PER-STOCK LOOP + ENTRY DETECTION
# ══════════════════════════════════════════════════════════════════════

def _build_entry_row(
    cfg: dict, ticker: str, week_date, entry_type: str,
    evt: dict, next_open: float,
    rs_13w: pd.Series, atr_series: pd.Series,
    sector: str, weeks_in_stage2: int, idx: int,
    ma_series: pd.Series,
    detector_name: str,
    rsi_series: pd.Series | None = None,
    momentum_series: pd.Series | None = None,
    vol_ratio_series: pd.Series | None = None,
) -> dict:
    """Build a flat dict row for entries_all.csv."""
    swing_info = DETECTOR_SWING.get(detector_name, {"major": 8, "minor": 5})
    det_major = swing_info["major"] if swing_info else None
    det_minor = swing_info["minor"] if swing_info else None

    row = {
        # Config
        "config_id": cfg["config_id"],
        "ma_type": cfg["ma_type"],
        "ma_period": cfg["ma_period"],
        # Detector swing info
        "detector_swing_major": det_major,
        "detector_swing_minor": det_minor,
        # Core entry
        "ticker": ticker,
        "week_date": str(week_date.date()),
        "entry_type": entry_type,
        "entry_price_close": evt.get("close"),
        "next_week_open": next_open,
        # Quality from compute_entry_quality (may be absent for new types)
        "stop_level": evt.get("stop_level") or evt.get("stop_atr_level"),
        "target_level": evt.get("target_level"),
        "rr_ratio": evt.get("rr_ratio"),
        "volume_ratio": evt.get("volume_ratio"),
        # Context
        "rs_13w": float(rs_13w.iloc[idx]) if idx < len(rs_13w) and not pd.isna(rs_13w.iloc[idx]) else None,
        "sector": sector,
        "atr_at_entry": float(atr_series.iloc[idx]) if idx < len(atr_series) and not pd.isna(atr_series.iloc[idx]) else None,
        "ma_value": float(ma_series.iloc[idx]) if idx < len(ma_series) and not pd.isna(ma_series.iloc[idx]) else None,
        # Filter columns
        "weeks_in_stage2": weeks_in_stage2,
        # New context columns
        "rsi_at_entry": float(rsi_series.iloc[idx]) if rsi_series is not None and idx < len(rsi_series) and not pd.isna(rsi_series.iloc[idx]) else None,
        "momentum_at_entry": int(momentum_series.iloc[idx]) if momentum_series is not None and idx < len(momentum_series) and not pd.isna(momentum_series.iloc[idx]) else None,
        "volume_ratio_at_entry": float(vol_ratio_series.iloc[idx]) if vol_ratio_series is not None and idx < len(vol_ratio_series) and not pd.isna(vol_ratio_series.iloc[idx]) else None,
        # Breakout-specific
        "base_duration": evt.get("base_duration"),
        "base_start_idx": evt.get("base_start_idx"),
        "ceiling": evt.get("ceiling"),
        "ceiling_def": evt.get("ceiling_def"),
        # Pullback-specific
        "depth_def": evt.get("depth_def"),
        "confirmation": evt.get("confirmation"),
        "consecutive_s2_weeks": evt.get("consecutive_s2_weeks"),
        # VCP-specific
        "consolidation_start_idx": evt.get("consolidation_start_idx"),
        "consolidation_weeks": evt.get("consolidation_weeks"),
        "consolidation_ceiling": evt.get("consolidation_ceiling"),
        "consolidation_low": evt.get("consolidation_low"),
        "contraction_count": evt.get("contraction_count"),
        "contraction_depths": str(evt.get("contraction_depths")) if evt.get("contraction_depths") is not None else None,
        "volume_slope": evt.get("volume_slope"),
        "breakout_volume_ratio": evt.get("breakout_volume_ratio"),
        # Retest-specific
        "retest_level": evt.get("retest_level"),
        "weeks_since_breakout": evt.get("weeks_since_breakout"),
        "pullback_depth_pct": evt.get("pullback_depth_pct"),
        "bounce_volume_ratio": evt.get("bounce_volume_ratio"),
        # Structural pullback-specific
        "support_level": evt.get("support_level"),
        "support_type": evt.get("support_type"),
        "distance_from_ma_atr": evt.get("distance_from_ma_atr"),
        "level_significance_weeks": evt.get("level_significance_weeks"),
        # Trendline-specific
        "trendline_slope_pct_month": evt.get("trendline_slope_pct_month"),
        "trendline_anchor_count": evt.get("trendline_anchor_count"),
        "trendline_age_weeks": evt.get("trendline_age_weeks"),
        "distance_from_trendline_atr": evt.get("distance_from_trendline_atr"),
        # ATH-specific
        "consolidation_range_atr": evt.get("consolidation_range_atr"),
        "ath_type": evt.get("ath_type"),
        # Computed target/risk columns
        "target_type": (
            "synthetic" if (
                entry_type == "ATH_BREAKOUT"
                or (evt.get("close") is not None
                    and evt.get("target_level") is not None
                    and evt.get("target_level") == round(evt["close"] * 1.20, 2))
            ) else "structural"
        ) if evt.get("close") is not None else None,
        "risk_pct_at_entry": None,
        "measured_move_target": None,
    }

    # Compute proximity_atr for RETEST and STRUCTURAL entries
    atr_val = row.get("atr_at_entry")
    close = evt.get("close")
    if entry_type == "RETEST_SUPPORT" and evt.get("retest_level") and atr_val and atr_val > 0 and close:
        row["proximity_atr"] = round(abs(close - float(evt["retest_level"])) / atr_val, 2)
    elif entry_type == "PULLBACK_STRUCTURAL" and evt.get("support_level") and atr_val and atr_val > 0 and close:
        row["proximity_atr"] = round(abs(close - float(evt["support_level"])) / atr_val, 2)
    else:
        row["proximity_atr"] = None

    # Compute risk_pct_at_entry
    stop = row["stop_level"]
    if close and stop and close > 0:
        row["risk_pct_at_entry"] = round((close - stop) / close, 4)

    # Measured move for breakout/ATH entries
    if entry_type.startswith("BREAKOUT") and evt.get("ceiling") is not None and stop is not None:
        row["measured_move_target"] = round(float(evt["ceiling"]) + (float(evt["ceiling"]) - stop), 2)
    elif entry_type == "ATH_BREAKOUT" and evt.get("consolidation_low") is not None:
        ceiling_est = close if close else 0
        row["measured_move_target"] = round(ceiling_est + (ceiling_est - float(evt["consolidation_low"])), 2)

    return row


def _process_entry_events(events, weekly, cfg, ticker, week_dates_set,
                          has_demerger, demerger_until, qualifying_by_quarter,
                          stages, rs_13w, atr_series, sector, ma_s,
                          major_highs, minor_lows, all_entries,
                          detector_name, entry_type_override=None,
                          rsi_series=None, momentum_series=None,
                          vol_ratio_series=None,
                          use_entry_quality=True,
                          base_start_from_evt=False):
    """Common logic to filter and build entry rows from detector events."""
    for evt in events:
        if use_entry_quality:
            if base_start_from_evt:
                evt = compute_entry_quality(
                    weekly, evt, major_highs, minor_lows, atr_series,
                    base_start_idx=evt.get("base_start_idx"),
                )
            else:
                evt = compute_entry_quality(
                    weekly, evt, major_highs, minor_lows, atr_series,
                    base_start_idx=None,
                )

        idx = evt["week_idx"]
        week_date = weekly.index[idx]

        if has_demerger and demerger_until is not None and demerger_until > week_date:
            continue
        entry_q = get_quarter(week_date)
        if entry_q in qualifying_by_quarter and ticker not in qualifying_by_quarter[entry_q]:
            continue

        w_s2 = count_consecutive_stage2(stages, idx)
        entry_type = entry_type_override or evt.get("entry_type", "UNKNOWN")

        # Breakout subtype
        if entry_type == "BREAKOUT_S1S2":
            entry_type = "BREAKOUT_S1_TO_S2" if w_s2 < 4 else "BREAKOUT_S2_CONTINUATION"

        next_open = float(weekly.iloc[idx + 1]["Open"]) if idx + 1 < len(weekly) else np.nan

        all_entries.append(_build_entry_row(
            cfg, ticker, week_date, entry_type, evt, next_open,
            rs_13w, atr_series, sector, w_s2, idx, ma_s,
            detector_name, rsi_series, momentum_series, vol_ratio_series,
        ))


def process_ticker_entries(
    ticker: str,
    daily_raw: pd.DataFrame,
    nifty_close_weekly: pd.Series,
    qualifying_by_quarter: dict[str, set[str]],
    ticker_metadata: dict[str, dict],
    all_entries: list[dict],
    stages_by_ticker: dict[str, dict[str, pd.Series]],
    cache_dir: Path = CACHE_DIR,
    swing_cache_dir: Path = SWING_CACHE_DIR,
    write_cache: bool = True,
) -> bool:
    """Per-ticker indicator computation + entry detection.

    Mechanical extraction of step4_per_stock_loop's inner per-ticker block.
    Mutates `all_entries` and `stages_by_ticker` in place (same semantics
    as the original inline code).

    Returns True if processed, False if skipped (insufficient weekly bars).
    """
    # ── Prepare daily -> weekly ───────────────────────────────
    daily = daily_raw.copy()
    daily = daily.set_index("date").sort_index()
    daily = daily.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    weekly = resample_weekly(daily)
    if len(weekly) < MIN_WEEKLY_BARS:
        return False

    # ── Compute ATR (once) ────────────────────────────────────
    atr_series = calculate_atr(weekly["High"], weekly["Low"], weekly["Close"], window=14)

    # ── Compute ALL MA variants (once: 8 combos) ─────────────
    ma_dict: dict[str, pd.Series] = {}
    for mt, mp in ALL_MA_COMBOS:
        k = ma_key(mt, mp)
        ma_dict[k] = compute_ma_series(weekly["Close"], mt, mp)

    # ── RS_13w (once) ─────────────────────────────────────────
    stock_ret = weekly["Close"] / weekly["Close"].shift(13) - 1
    nifty_aligned = nifty_close_weekly.reindex(weekly.index, method="nearest")
    nifty_ret = nifty_aligned / nifty_aligned.shift(13) - 1
    rs_13w = stock_ret / nifty_ret.replace(0, np.nan)

    # ── NEW: RSI_14 (once) ────────────────────────────────────
    rsi_series = calculate_rsi(weekly["Close"], window=14)

    # ── NEW: Momentum flag (once) ─────────────────────────────
    momentum_series = (weekly["Close"] > weekly["Close"].shift(1)).astype(int)

    # ── NEW: Volume ratio 20w (once) ──────────────────────────
    vol_20w = weekly["Volume"].rolling(20).mean()
    vol_ratio_series = weekly["Volume"] / vol_20w.replace(0, np.nan)

    # ── Save to cache (with new columns) ──────────────────────
    if write_cache:
        cache_df = weekly[["Open", "High", "Low", "Close", "Volume"]].copy()
        cache_df["atr_14"] = atr_series
        cache_df["rs_13w"] = rs_13w
        cache_df["rsi_14"] = rsi_series
        cache_df["momentum"] = momentum_series
        cache_df["volume_ratio_20w"] = vol_ratio_series
        for k, ma_s in ma_dict.items():
            cache_df[k] = ma_s
        cache_df.to_csv(cache_dir / f"{ticker}.csv")

    # ── Compute swings at ALL orders ──────────────────────────
    swing_cache: dict[int, tuple[list[int], list[int]]] = {}
    swing_json: dict[str, dict] = {}
    for order in ALL_SWING_ORDERS:
        h, l = method_a_argrelextrema(weekly, order=order)
        swing_cache[order] = (h, l)
        swing_json[f"order_{order}"] = {"highs": [int(x) for x in h], "lows": [int(x) for x in l]}

    # Save swing cache as JSON
    if write_cache:
        with open(swing_cache_dir / f"{ticker}.json", "w") as f:
            json.dump(swing_json, f)

    # ── Compute direction + stages (once per MA combo) ────────
    direction_cache: dict[str, pd.Series] = {}
    stages_cache: dict[str, pd.Series] = {}
    for mt, mp in ALL_MA_COMBOS:
        k = ma_key(mt, mp)
        direction = compute_ma_slope_direction(ma_dict[k], atr_series)
        direction_cache[k] = direction
        stages = classify_stages_single_ma(weekly, ma_dict[k], direction)
        stages_cache[k] = stages
        stages_by_ticker[k][ticker] = stages

    # ── Get ticker metadata ───────────────────────────────────
    meta = ticker_metadata.get(ticker, {})
    has_demerger = meta.get("has_demerger", False)
    demerger_until = meta.get("demerger_no_trade_until")
    if demerger_until is not None:
        demerger_until = pd.Timestamp(demerger_until)
    sector = meta.get("primary_sector", "UNKNOWN")

    # Build swing_lows_data for trendline detector (list of dicts)
    tl_swing_order = 5  # use order 5 for trendline swing lows
    tl_lows = swing_cache.get(tl_swing_order, ([], []))[1]
    swing_lows_data = []
    for sl_idx in tl_lows:
        if sl_idx < len(weekly):
            swing_lows_data.append({
                "level": float(weekly["Low"].iloc[sl_idx]),
                "date": str(weekly.index[sl_idx].date()),
            })

    # ── Precompute trendlines ONCE per stock (expensive) ───────
    # Use interval=52 (yearly) to keep runtime manageable
    tl_cache = precompute_trendlines(weekly, atr_series, swing_lows_data, interval=52)

    # ── Entry detection for each of 8 configs ─────────────────
    for cfg in INDICATOR_CONFIGS:
        k = ma_key(cfg["ma_type"], cfg["ma_period"])
        stages = stages_cache[k]
        ma_s = ma_dict[k]

        # Get swings per detector
        bo_sw = DETECTOR_SWING["breakout"]
        bo_major_h, bo_major_l = swing_cache[bo_sw["major"]]
        bo_minor_h, bo_minor_l = swing_cache[bo_sw["minor"]]

        pb_sw = DETECTOR_SWING["pullback_ma"]
        pb_minor_h, pb_minor_l = swing_cache[pb_sw["minor"]]

        vcp_sw = DETECTOR_SWING["vcp"]
        vcp_minor_h, vcp_minor_l = swing_cache[vcp_sw["minor"]]

        rt_sw = DETECTOR_SWING["retest"]
        rt_major_h, rt_major_l = swing_cache[rt_sw["major"]]

        ps_sw = DETECTOR_SWING["pullback_structural"]
        ps_major_h, ps_major_l = swing_cache[ps_sw["major"]]

        ath_sw = DETECTOR_SWING["ath_breakout"]
        ath_major_h, ath_major_l = swing_cache[ath_sw["major"]]

        common_args = dict(
            weekly=weekly, cfg=cfg, ticker=ticker,
            week_dates_set=set(), has_demerger=has_demerger,
            demerger_until=demerger_until,
            qualifying_by_quarter=qualifying_by_quarter,
            stages=stages, rs_13w=rs_13w, atr_series=atr_series,
            sector=sector, ma_s=ma_s,
            major_highs=bo_major_h, minor_lows=bo_minor_l,
            all_entries=all_entries,
            rsi_series=rsi_series, momentum_series=momentum_series,
            vol_ratio_series=vol_ratio_series,
        )

        # 1. Breakouts
        bo_events = detect_breakouts(
            weekly, stages, BREAKOUT_BASE_MIN, BREAKOUT_CEILING,
            swing_high_indices=bo_major_h,
            ma_series=ma_s, atr_series=atr_series,
        )
        for evt in bo_events:
            evt["entry_type"] = "BREAKOUT_S1S2"
        _process_entry_events(
            bo_events, detector_name="breakout",
            entry_type_override=None, use_entry_quality=True,
            base_start_from_evt=True, **common_args,
        )

        # 2. Pullbacks (MA)
        pb_events = detect_pullbacks(
            weekly, stages, ma_s, atr_series, pb_minor_l,
            PULLBACK_MIN_S2, PULLBACK_DEPTH, PULLBACK_CONFIRM,
        )
        _process_entry_events(
            pb_events, detector_name="pullback_ma",
            entry_type_override="PULLBACK_S2", use_entry_quality=True,
            **common_args,
        )

        # 3. VCP
        vcp_events = detect_vcp_continuations(
            weekly, stages, atr_series, vcp_minor_h, vcp_minor_l,
            VCP_CONSOL_MIN, VCP_CONTRACTION, VCP_VOLUME,
        )
        _process_entry_events(
            vcp_events, detector_name="vcp",
            entry_type_override="VCP_CONTINUATION", use_entry_quality=True,
            **common_args,
        )

        # 4. Retest Support (NEW)
        retest_events = detect_retest_support(
            weekly, stages, rt_major_h, atr_series,
        )
        _process_entry_events(
            retest_events, detector_name="retest",
            entry_type_override="RETEST_SUPPORT", use_entry_quality=False,
            **common_args,
        )

        # 5. Pullback Structural (NEW)
        ps_events = detect_pullback_structural(
            weekly, stages, ps_major_h, ps_major_l, ma_s, atr_series,
        )
        _process_entry_events(
            ps_events, detector_name="pullback_structural",
            entry_type_override="PULLBACK_STRUCTURAL", use_entry_quality=False,
            **common_args,
        )

        # 6. Trendline Bounce (NEW) — uses precomputed tl_cache
        tl_events = detect_trendline_bounce(
            weekly, stages, atr_series, tl_cache,
        )
        _process_entry_events(
            tl_events, detector_name="trendline",
            entry_type_override="TRENDLINE_BOUNCE", use_entry_quality=False,
            **common_args,
        )

        # 7. ATH Breakout (NEW)
        ath_events = detect_ath_breakout(
            weekly, stages, atr_series, ath_major_h,
        )
        _process_entry_events(
            ath_events, detector_name="ath_breakout",
            entry_type_override="ATH_BREAKOUT", use_entry_quality=False,
            **common_args,
        )

    return True


def step4_per_stock_loop(
    stock_df: pd.DataFrame,
    nifty_weekly: pd.DataFrame,
    qualifying_by_quarter: dict[str, set[str]],
    ticker_metadata: dict[str, dict],
) -> tuple[list[dict], dict[str, dict[str, pd.Series]]]:
    print("=" * 70)
    print("STEP 4: Per-stock indicator computation + entry detection (v2)")
    print("=" * 70)

    print(f"\n  Indicator Config Grid ({len(INDICATOR_CONFIGS)} configs):")
    print(f"  {'ID':>3} {'MA Type':<6} {'Period':>6}")
    for cfg in INDICATOR_CONFIGS:
        print(f"  {cfg['config_id']:>3} {cfg['ma_type']:<6} {cfg['ma_period']:>6}")
    print(f"\n  Per-detector swing orders:")
    for det, sw in DETECTOR_SWING.items():
        print(f"    {det:<22s}: {sw}")
    print()

    all_qualifying = set()
    for tickers in qualifying_by_quarter.values():
        all_qualifying |= tickers
    all_qualifying = sorted(all_qualifying)
    print(f"  Total unique qualifying tickers: {len(all_qualifying)}")

    stock_df_copy = stock_df.copy()
    stock_df_copy["date"] = pd.to_datetime(stock_df_copy["date"])
    stock_grouped = {ticker: grp for ticker, grp in stock_df_copy.groupby("ticker")}

    nifty_close_weekly = nifty_weekly["Close"]

    all_entries: list[dict] = []
    stages_by_ticker: dict[str, dict[str, pd.Series]] = {
        ma_key(mt, mp): {} for mt, mp in ALL_MA_COMBOS
    }

    skipped_insufficient = 0
    processed = 0
    t_start = time.time()

    for ticker_idx, ticker in enumerate(all_qualifying):
        if ticker not in stock_grouped:
            continue

        ok = process_ticker_entries(
            ticker=ticker,
            daily_raw=stock_grouped[ticker],
            nifty_close_weekly=nifty_close_weekly,
            qualifying_by_quarter=qualifying_by_quarter,
            ticker_metadata=ticker_metadata,
            all_entries=all_entries,
            stages_by_ticker=stages_by_ticker,
            cache_dir=CACHE_DIR,
            swing_cache_dir=SWING_CACHE_DIR,
            write_cache=True,
        )
        if not ok:
            skipped_insufficient += 1
            continue

        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            remaining = (len(all_qualifying) - ticker_idx - 1) / rate if rate > 0 else 0
            print(f"  Processed {processed} stocks ({ticker_idx+1}/{len(all_qualifying)}), "
                  f"entries so far: {len(all_entries):,}, "
                  f"elapsed: {fmt_time(elapsed)}, ETA: {fmt_time(remaining)}",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Completed: {processed} stocks in {fmt_time(elapsed)}")
    print(f"  Skipped (< {MIN_WEEKLY_BARS} weekly bars): {skipped_insufficient}")
    print(f"  Total entries: {len(all_entries):,}")
    print()

    return all_entries, stages_by_ticker


# ══════════════════════════════════════════════════════════════════════
# STEP 5: REGIME CLASSIFICATION (same as v1)
# ══════════════════════════════════════════════════════════════════════

def step5_regime(
    nifty_weekly: pd.DataFrame,
    stages_by_ticker: dict[str, dict[str, pd.Series]],
    qualifying_by_quarter: dict[str, set[str]],
):
    print("=" * 70)
    print("STEP 5: Regime classification (breadth 5-state + Weinstein 2-signal)")
    print("=" * 70)

    all_dates: set = set()
    for k in stages_by_ticker:
        for stages in stages_by_ticker[k].values():
            all_dates.update(stages.index.tolist())
    all_dates_sorted = sorted(all_dates)

    breadth_data: dict[str, pd.DataFrame] = {}
    regime_breadth: dict[str, pd.Series] = {}

    nifty_lc = nifty_weekly[["Close"]].rename(columns={"Close": "close"})

    for mk in sorted(stages_by_ticker.keys()):
        ticker_stages = stages_by_ticker[mk]
        breadth_values = []
        breadth_dates = []

        for week in all_dates_sorted:
            qstr = get_quarter(week)
            qualifying = qualifying_by_quarter.get(qstr, set())
            if not qualifying:
                continue
            in_stage2 = 0
            total_with_data = 0
            for tkr in qualifying:
                stages = ticker_stages.get(tkr)
                if stages is None:
                    continue
                loc = stages.index.searchsorted(week, side="right") - 1
                if loc < 0 or loc >= len(stages):
                    continue
                val = stages.iloc[loc]
                if pd.isna(val):
                    continue
                total_with_data += 1
                if val == "stage2":
                    in_stage2 += 1

            if total_with_data > 0:
                breadth_values.append(round(in_stage2 / total_with_data * 100, 2))
            else:
                breadth_values.append(np.nan)
            breadth_dates.append(week)

        b_df = pd.DataFrame(
            {"breadth_pct": breadth_values},
            index=pd.DatetimeIndex(breadth_dates, name="date"),
        ).dropna()

        breadth_data[mk] = b_df
        regime_series = classify_all_weeks(
            b_df, nifty_lc, REGIME_HIGH, REGIME_LOW, REGIME_DIR_LOOKBACK, REGIME_DIR_THRESH,
        )
        regime_breadth[mk] = regime_series

    # Weinstein 2-signal
    ema_20 = nifty_weekly["ema_20"]
    weinstein_regimes = []
    for i in range(len(nifty_weekly)):
        week_date = nifty_weekly.index[i]
        closes_up_to = nifty_weekly["Close"].iloc[:i + 1]
        ema_up_to = ema_20.iloc[:i + 1]
        atr_val = float(nifty_weekly["atr_14"].iloc[i]) if not pd.isna(nifty_weekly["atr_14"].iloc[i]) else 0.0

        pos = compute_ema_position(closes_up_to, ema_up_to)
        dirn = compute_ema_direction(ema_up_to, atr_val)

        breadth_200_proxy = None
        if "sma_25" in breadth_data:
            b_df_25 = breadth_data["sma_25"]
            b_loc = b_df_25.index.searchsorted(week_date, side="right") - 1
            if 0 <= b_loc < len(b_df_25):
                breadth_200_proxy = int(b_df_25["breadth_pct"].iloc[b_loc])

        regime_result = classify_regime(pos["position"], dirn["direction"], breadth_200_proxy)
        weinstein_regimes.append({
            "week_date": week_date,
            "regime_weinstein": regime_result["classification"],
        })

    weinstein_df = pd.DataFrame(weinstein_regimes).set_index("week_date")

    # Save breadth_weekly.csv
    breadth_combined = pd.DataFrame(index=pd.DatetimeIndex(all_dates_sorted, name="date"))
    for mk, b_df in breadth_data.items():
        breadth_combined[f"breadth_pct_{mk}"] = b_df["breadth_pct"]
    breadth_combined.dropna(how="all").to_csv(OUTPUT_DIR / "breadth_weekly.csv")

    # Save regime_weekly.csv
    regime_combined = pd.DataFrame(index=nifty_weekly.index)
    regime_combined.index.name = "week_date"
    for mk, b_df in breadth_data.items():
        regime_combined[f"breadth_pct_{mk}"] = b_df["breadth_pct"].reindex(regime_combined.index)
    for mk, r_series in regime_breadth.items():
        regime_combined[f"regime_breadth_{mk}"] = r_series.reindex(regime_combined.index)
    regime_combined["regime_weinstein"] = weinstein_df["regime_weinstein"].reindex(regime_combined.index)
    regime_combined["nifty_close"] = nifty_weekly["Close"]
    regime_combined["nifty_ema20"] = nifty_weekly["ema_20"]
    regime_combined["nifty_atr"] = nifty_weekly["atr_14"]
    regime_combined.to_csv(OUTPUT_DIR / "regime_weekly.csv")

    if "regime_weinstein" in regime_combined.columns:
        w_counts = regime_combined["regime_weinstein"].value_counts()
        print(f"  Weinstein regime distribution:")
        for regime, cnt in w_counts.items():
            print(f"    {regime}: {cnt} weeks ({cnt/len(regime_combined)*100:.1f}%)")
    if "regime_breadth_sma_25" in regime_combined.columns:
        b_counts = regime_combined["regime_breadth_sma_25"].value_counts()
        print(f"  Breadth (SMA_25) regime distribution:")
        for regime, cnt in b_counts.items():
            print(f"    {regime}: {cnt} weeks ({cnt/len(regime_combined)*100:.1f}%)")
    print()


# ══════════════════════════════════════════════════════════════════════
# STEP 6: SAVE ENTRIES + CONFIG ECHO
# ══════════════════════════════════════════════════════════════════════

def step6_save(all_entries: list[dict]):
    print("=" * 70)
    print("STEP 6: Saving entries_all.csv and config_echo.json")
    print("=" * 70)

    entries_df = pd.DataFrame(all_entries)
    entries_df.to_csv(OUTPUT_DIR / "entries_all.csv", index=False)

    print(f"  Total entries: {len(entries_df):,}")

    if not entries_df.empty:
        per_config = entries_df.groupby("config_id").size()
        print(f"  Entries per config: min={per_config.min()}, max={per_config.max()}, mean={per_config.mean():.0f}")
        print(f"  Entries by type:")
        for et, cnt in entries_df["entry_type"].value_counts().items():
            print(f"    {et}: {cnt:,}")

        entries_df["year"] = pd.to_datetime(entries_df["week_date"]).dt.year
        print(f"  Entries per year (config_id=0):")
        cfg0 = entries_df[entries_df["config_id"] == 0]
        if not cfg0.empty:
            for yr, cnt in cfg0.groupby("year").size().items():
                print(f"    {yr}: {cnt}")

    config_echo = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "market_cap_threshold_cr": MARKET_CAP_THRESHOLD_CR,
        "nifty_reference_level": NIFTY_REFERENCE_LEVEL,
        "exclude_sme": EXCLUDE_SME,
        "min_weekly_bars": MIN_WEEKLY_BARS,
        "total_indicator_configs": len(INDICATOR_CONFIGS),
        "indicator_configs": INDICATOR_CONFIGS,
        "detector_swing_orders": DETECTOR_SWING,
        "all_swing_orders_cached": ALL_SWING_ORDERS,
        "entry_detection_defaults": {
            "breakout_ceiling": BREAKOUT_CEILING,
            "breakout_base_min": BREAKOUT_BASE_MIN,
            "pullback_depth": PULLBACK_DEPTH,
            "pullback_confirm": PULLBACK_CONFIRM,
            "pullback_min_s2": PULLBACK_MIN_S2,
            "vcp_consol_min": VCP_CONSOL_MIN,
            "vcp_contraction": VCP_CONTRACTION,
            "vcp_volume": VCP_VOLUME,
        },
        "regime_thresholds": {
            "high": REGIME_HIGH,
            "low": REGIME_LOW,
            "dir_lookback": REGIME_DIR_LOOKBACK,
            "dir_thresh": REGIME_DIR_THRESH,
        },
    }
    with open(OUTPUT_DIR / "config_echo.json", "w") as f:
        json.dump(config_echo, f, indent=2)

    print(f"  Saved config_echo.json")
    print()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(ohlcv_dir: Path | str | None = None):
    """Build all reproduction intermediates from raw daily OHLCV files.

    Args:
        ohlcv_dir: directory holding ``stocks_daily`` + ``benchmark_daily``
            (``.parquet``/``.csv``). If ``None``, resolves from
            :func:`skysurf.reproduction._paths.ohlcv_dir`.

    Writes the stock weekly cache, regimes, universe, nifty_weekly, and
    ``entries_all.csv`` into the reproduction data bundle (``OUTPUT_DIR``).
    """
    t0 = time.time()
    ohlcv_dir = Path(ohlcv_dir) if ohlcv_dir is not None else _paths.ohlcv_dir()

    # Re-resolve output paths from the (possibly just-set) data dir, so the
    # module works regardless of when it was first imported.
    global OUTPUT_DIR, CACHE_DIR, SWING_CACHE_DIR
    OUTPUT_DIR = _paths.data_dir()
    CACHE_DIR = OUTPUT_DIR / "stock_weekly_cache"
    SWING_CACHE_DIR = OUTPUT_DIR / "swing_cache"
    for _d in (OUTPUT_DIR, CACHE_DIR, SWING_CACHE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VAL-ENGINE-A v2: Data Pipeline + Entry Detection Grid")
    print("=" * 70)
    print(f"  Raw OHLCV input:  {ohlcv_dir}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Config: {len(INDICATOR_CONFIGS)} indicator configs, 7 detectors")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print()

    stock_df, bench_df = step1_load_data(ohlcv_dir)
    qualifying_by_quarter, ticker_metadata, universe_df = step2_quarterly_universe(stock_df, bench_df)
    nifty_weekly = step3_benchmark(bench_df)
    all_entries, stages_by_ticker = step4_per_stock_loop(
        stock_df, nifty_weekly, qualifying_by_quarter, ticker_metadata,
    )
    step5_regime(nifty_weekly, stages_by_ticker, qualifying_by_quarter)
    step6_save(all_entries)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"VAL-ENGINE-A v2 complete. Total time: {fmt_time(elapsed)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    return OUTPUT_DIR


if __name__ == "__main__":
    main()
