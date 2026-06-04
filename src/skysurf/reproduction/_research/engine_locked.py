"""VAL-ENGINE-B: Portfolio Simulation Engine.

Reads pre-computed entries from Engine A CSVs, simulates a week-by-week
portfolio with trailing stops, position sizing, and regime filters.

Self-contained: no database access, no app imports. All data from Engine A CSVs.

Usage:
    python scripts/v2_validation/val_engine_b.py
    python scripts/v2_validation/val_engine_b.py --sensitivity
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skysurf.reproduction import _paths


# matplotlib is imported lazily (charts are optional). See engine.py for notes.
class _LazyPlt:
    _plt = None

    def __getattr__(self, name):  # pragma: no cover - charts are optional
        if _LazyPlt._plt is None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            _LazyPlt._plt = _plt
        return getattr(_LazyPlt._plt, name)


plt = _LazyPlt()

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ENGINE_A_DIR = _paths.data_dir()
STOCK_CACHE_DIR = ENGINE_A_DIR / "stock_weekly_cache"
OUTPUT_DIR = _paths.output_dir() / "val_engine_b_results"

# ── Default Configuration ─────────────────────────────────────────────

CONFIG = {
    # Indicator config selection (filters entries_all.csv)
    "ma_type": "SMA",
    "ma_period": 25,
    "swing_order_major": 8,

    # Entry filters
    "base_minimum_weeks": 16,
    "entry_types": ["BREAKOUT", "PULLBACK", "VCP"],  # prefix match
    "rr_floor": 0.3,              # applied only to structural target_type
    "max_risk_pct": None,         # applied only to synthetic target_type
    "max_risk_pct_all": None,     # if set, applied to ALL entries
    "rs_gate": None,
    "volume_gate": None,
    "sector_gate": False,
    "entry_price_method": "week_close",  # "week_close" or "next_week_open"

    # Regime
    "regime_type": "breadth_5state",
    "entry_filter": "aggressive",

    # Exit
    "exit_type": "sma_trail",
    "exit_ma_type": "SMA",
    "exit_ma_period": 25,
    "exit_atr_buffer": 1.0,
    "exit_gtt_field": "low",  # "low" or "close"

    # Sizing
    "risk_pct": 0.01,
    "sizing_on": "total_equity",  # "total_equity" or "capital"
    "tier_rr_thresholds": [1.5, 2.0],
    "concentration_caps": [0.08, 0.15, 0.25],   # starter / half / full
    "tier_multipliers": [0.5, 0.75, 1.0],       # starter / half / full

    # Constraints
    "sector_limit": 3,
    "max_positions": 30,

    # Costs
    "slippage_pct": 0.001,
    "brokerage_pct": 0.0011,

    # Ranking
    "ranking_method": "rs_13w",

    # Period
    "starting_capital": 500_000,
    "start_date": "2007-01-01",
    "end_date": "2026-03-21",
    "is_split_date": "2020-01-01",

    # Metrics
    "risk_free_annual": 0.06,

    # Walk-forward
    "force_close_at_end": False,

    # Triple-stack exit tightening (all OFF by default)
    "triple_stack_enabled": False,
    "stall_tighten_week": 10,
    "stall_tighten_threshold": 0.05,   # 5% as fraction
    "stall_tighten_ma": "sma_20",
    "extension_atr_mult": 3.0,
    "extension_tighten_ma": "sma_15",
    "climactic_vol_threshold": 2.5,
    "climactic_tighten_ma": "sma_15",
}

TIER_NAMES = ["starter", "half", "full"]


# ── Derived column names ──────────────────────────────────────────────

def _derive_columns(cfg: dict) -> dict:
    """Derive regime_col, breadth_col, exit_ma_col from config."""
    mt = cfg["ma_type"].lower()
    mp = cfg["ma_period"]
    if cfg["regime_type"] == "breadth_5state":
        regime_col = f"regime_breadth_{mt}_{mp}"
        breadth_col = f"breadth_pct_{mt}_{mp}"
    else:
        regime_col = "regime_weinstein"
        breadth_col = f"breadth_pct_{mt}_{mp}"
    exit_ma_col = f"{cfg['exit_ma_type'].lower()}_{cfg['exit_ma_period']}"
    return {"regime_col": regime_col, "breadth_col": breadth_col, "exit_ma_col": exit_ma_col}


# ── Position dataclass ────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    sector: str
    entry_date: Any  # pd.Timestamp
    entry_price: float
    qty: int
    cost: float
    stop_level: float
    tier: str
    peak_close: float
    entry_type: str
    target_type: str
    risk_pct_at_entry: float
    rs_13w_at_entry: float
    regime_at_entry: str
    rr_at_entry: float
    current_close: float
    stop_at_entry: float  # original stop for verification
    min_low: float = 0.0       # tracks worst intraweek Low (for MAE)
    mfe_week: Any = None       # week when peak_close was last updated (for time_to_mfe)
    tighten_state: str | None = None   # None | "stall" | "climactic" (triple-stack)
    prev_close: float = 0.0            # previous week's close (for climactic vol check)
    # Per-position exit config (copied from active cfg at creation)
    exit_ma_col: str = "sma_25"
    exit_atr_buffer: float = 1.0
    pos_triple_stack_enabled: bool = False
    pos_stall_tighten_week: int = 10
    pos_stall_tighten_threshold: float = 0.05
    pos_stall_tighten_ma: str = "sma_20"
    pos_extension_atr_mult: float = 3.0
    pos_extension_tighten_ma: str = "sma_15"
    pos_climactic_vol_threshold: float = 2.5
    pos_climactic_tighten_ma: str = "sma_15"
    config_id: str = "default"


# ── Stock Cache Manager ───────────────────────────────────────────────

class StockCacheManager:
    """Lazy-loading cache for stock weekly CSV files."""

    def __init__(self, cache_dir: Path, extra_sma_periods: list[int] | None = None):
        self._dir = cache_dir
        self._cache: dict[str, pd.DataFrame | None] = {}
        self._extra_sma = extra_sma_periods or []

    def get(self, ticker: str) -> pd.DataFrame | None:
        if ticker in self._cache:
            return self._cache[ticker]
        fpath = self._dir / f"{ticker}.csv"
        if not fpath.exists():
            self._cache[ticker] = None
            return None
        df = pd.read_csv(fpath, parse_dates=["date"])
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        for p in self._extra_sma:
            col = f"sma_{p}"
            if col not in df.columns:
                df[col] = df["Close"].rolling(p, min_periods=1).mean()
        self._cache[ticker] = df
        return df


# ── Data Loading ──────────────────────────────────────────────────────

def load_and_filter_entries(cfg: dict) -> pd.DataFrame:
    """Load entries_all.csv and apply all entry filters."""
    path = ENGINE_A_DIR / "entries_all.csv"
    print(f"  Loading entries from {path} ...")
    df = pd.read_csv(path)
    total = len(df)
    print(f"    Total entries in file: {total:,}")

    # 1. Indicator config filter
    mask = (
        (df["ma_type"] == cfg["ma_type"]) &
        (df["ma_period"] == cfg["ma_period"]) &
        (df["swing_order_major"] == cfg["swing_order_major"])
    )
    df = df[mask].copy()
    print(f"    After config filter (MA={cfg['ma_type']}_{cfg['ma_period']}, swing={cfg['swing_order_major']}): {len(df):,}")

    # 2. Base minimum weeks
    df = df[df["weeks_in_stage2"] >= cfg["base_minimum_weeks"]]
    print(f"    After base_minimum_weeks >= {cfg['base_minimum_weeks']}: {len(df):,}")

    # 3. Entry type prefix match
    type_mask = pd.Series(False, index=df.index)
    for prefix in cfg["entry_types"]:
        type_mask |= df["entry_type"].str.startswith(prefix)
    df = df[type_mask]
    print(f"    After entry_type filter {cfg['entry_types']}: {len(df):,}")

    # 4. Target-type-aware filtering
    structural_mask = df["target_type"] == "structural"
    synthetic_mask = df["target_type"] == "synthetic"

    # Structural: apply rr_floor
    if cfg["rr_floor"] is not None:
        drop_structural = structural_mask & (df["rr_ratio"] < cfg["rr_floor"])
        df = df[~drop_structural]
        print(f"    After rr_floor >= {cfg['rr_floor']} (structural only): {len(df):,}")

    # Synthetic: apply max_risk_pct if set
    if cfg["max_risk_pct"] is not None:
        synthetic_mask = df["target_type"] == "synthetic"  # recompute after prior filter
        drop_synthetic = synthetic_mask & (df["risk_pct_at_entry"] >= cfg["max_risk_pct"])
        df = df[~drop_synthetic]
        print(f"    After max_risk_pct < {cfg['max_risk_pct']} (synthetic only): {len(df):,}")

    # Universal max_risk_pct_all
    if cfg["max_risk_pct_all"] is not None:
        df = df[df["risk_pct_at_entry"] < cfg["max_risk_pct_all"]]
        print(f"    After max_risk_pct_all < {cfg['max_risk_pct_all']} (all): {len(df):,}")

    # 5. RS gate
    if cfg["rs_gate"] is not None:
        df = df[df["rs_13w"] >= cfg["rs_gate"]]
        print(f"    After rs_gate >= {cfg['rs_gate']}: {len(df):,}")

    # 6. Volume gate
    if cfg["volume_gate"] is not None:
        df = df[df["volume_ratio"] >= cfg["volume_gate"]]
        print(f"    After volume_gate >= {cfg['volume_gate']}: {len(df):,}")

    # 7. Entry price selection
    if cfg["entry_price_method"] == "next_week_open":
        df = df.dropna(subset=["next_week_open"])
        df["entry_price"] = df["next_week_open"]
        print(f"    Using next_week_open (dropped NaN): {len(df):,}")
    else:
        df["entry_price"] = df["entry_price_close"]

    # 8. Parse dates, sort
    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.sort_values("week_date").reset_index(drop=True)

    # 9. Date range filter
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    df = df[(df["week_date"] >= start) & (df["week_date"] <= end)]
    print(f"    After date range [{cfg['start_date']}, {cfg['end_date']}]: {len(df):,}")

    print(f"    Final filtered entries: {len(df):,}")
    # Stash counts for run_metadata
    df.attrs["entries_before_filter"] = total
    df.attrs["entries_after_filter"] = len(df)
    return df


def load_regime(cfg: dict) -> pd.DataFrame:
    """Load regime_weekly.csv with parsed dates as index."""
    path = ENGINE_A_DIR / "regime_weekly.csv"
    df = pd.read_csv(path)
    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.set_index("week_date")
    return df


def load_nifty() -> pd.DataFrame:
    """Load nifty_weekly.csv."""
    path = ENGINE_A_DIR / "nifty_weekly.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# ── Regime Filter ─────────────────────────────────────────────────────

def is_entry_allowed(regime: str, breadth_pct: float | None, cfg: dict) -> bool:
    """Check if entries are allowed under the current regime."""
    rt = cfg["regime_type"]
    ef = cfg["entry_filter"]
    bp = breadth_pct if breadth_pct is not None else 0.0

    if rt == "breadth_5state":
        if ef == "aggressive":
            if regime == "bear":
                return False
            if regime in ("recovering", "deteriorating"):
                return bp > 65
            return True  # strong_bull, weakening_bull
        elif ef == "moderate":
            return regime in ("strong_bull", "weakening_bull")
        elif ef == "conservative":
            return regime == "strong_bull"
    elif rt == "weinstein_2signal":
        if ef == "aggressive":
            if regime == "bear":
                return False
            if regime == "sideways":
                return bp > 65
            return True
        elif ef == "moderate":
            return regime == "bull"
        elif ef == "conservative":
            return regime == "bull" and bp > 50
    return False


# ── Position Sizing ───────────────────────────────────────────────────

def compute_sizing(
    entry_price: float,
    stop_level: float,
    rr_ratio: float,
    equity_base: float,
    cash: float,
    cfg: dict,
) -> dict | None:
    """Compute position size. Returns {qty, tier, entry_cost, binding} or None."""
    sl_distance = entry_price - stop_level
    if sl_distance <= 0:
        return None

    # Determine starting tier index
    thresholds = cfg["tier_rr_thresholds"]
    if rr_ratio >= thresholds[1]:
        tier_idx = 2  # full
    elif rr_ratio >= thresholds[0]:
        tier_idx = 1  # half
    else:
        tier_idx = 0  # starter

    # Try tier_idx, then downgrade
    for try_idx in range(tier_idx, -1, -1):
        multiplier = cfg["tier_multipliers"][try_idx]
        cap = cfg["concentration_caps"][try_idx]

        risk_amount = equity_base * cfg["risk_pct"] * multiplier
        risk_qty = math.floor(risk_amount / sl_distance)
        cap_qty = math.floor(equity_base * cap / entry_price)
        qty = min(risk_qty, cap_qty)

        if qty <= 0:
            continue

        binding = "risk" if risk_qty <= cap_qty else "cap"
        fees = cfg["slippage_pct"] + cfg["brokerage_pct"]
        entry_cost = qty * entry_price * (1 + fees)

        if entry_cost <= cash:
            return {
                "qty": qty,
                "tier": TIER_NAMES[try_idx],
                "entry_cost": entry_cost,
                "binding": binding,
            }
    return None


# ── Trade Record Builder ──────────────────────────────────────────────

def _build_trade(pos: Position, exit_price: float, exit_date, exit_reason: str,
                 proceeds: float) -> dict:
    """Build a closed trade record dict."""
    ret_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
    pnl = proceeds - pos.cost
    days = (exit_date - pos.entry_date).days
    holding_weeks = max(1, round(days / 7))
    peak_unreal = (pos.peak_close - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
    give_back = peak_unreal - ret_pct

    # NOTE: MAE uses min_low (weekly Low) while MFE uses peak_close (weekly Close).
    # This asymmetry is intentional and matches real trading:
    # - MAE: GTT stops trigger on intraweek Low, so the worst drawdown the position
    #   actually experiences is the Low, not the Close. MAE on Low = true risk exposure.
    # - MFE: You can only capture gains at Close (or next Open), not at the intraweek
    #   High. MFE on Close = achievable upside, not theoretical High.
    mae_pct = ((pos.min_low - pos.entry_price) / pos.entry_price) * 100 if pos.entry_price > 0 else 0
    mfe_pct = peak_unreal * 100  # identical to peak_unrealized_pct by construction
    time_to_mfe_weeks = max(1, (pos.mfe_week - pos.entry_date).days // 7) if pos.mfe_week is not None else 1
    capture_ratio = round((ret_pct * 100) / mfe_pct, 2) if mfe_pct > 0 else 0.0

    return {
        "ticker": pos.ticker,
        "sector": pos.sector,
        "entry_date": str(pos.entry_date.date()),
        "entry_price": round(pos.entry_price, 2),
        "exit_date": str(exit_date.date()),
        "exit_price": round(exit_price, 2),
        "qty": pos.qty,
        "tier": pos.tier,
        "stop_level_at_entry": round(pos.stop_at_entry, 2),
        "stop_level_at_exit": round(pos.stop_level, 2),
        "return_pct": round(ret_pct * 100, 2),
        "pnl": round(pnl, 2),
        "holding_weeks": holding_weeks,
        "peak_unrealized_pct": round(peak_unreal * 100, 2),
        "mfe_pct": round(mfe_pct, 2),
        "mae_pct": round(mae_pct, 2),
        "time_to_mfe_weeks": time_to_mfe_weeks,
        "capture_ratio": capture_ratio,
        "give_back_pp": round(give_back * 100, 2),
        "rs_13w_at_entry": round(pos.rs_13w_at_entry, 4) if not np.isnan(pos.rs_13w_at_entry) else None,
        "regime_at_entry": pos.regime_at_entry,
        "regime_at_exit": "",  # placeholder, set by caller
        "exit_reason": exit_reason,
        "entry_type": pos.entry_type,
        "target_type": pos.target_type,
        "risk_pct_at_entry": round(pos.risk_pct_at_entry, 4) if not np.isnan(pos.risk_pct_at_entry) else None,
    }


# ── Main Simulation ──────────────────────────────────────────────────

def run_simulation(
    cfg: dict,
    quiet: bool = False,
    entries_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    nifty_df: pd.DataFrame | None = None,
    stock_cache: "StockCacheManager | None" = None,
    initial_positions: list | None = None,
    initial_cash: float | None = None,
) -> dict:
    """Run the portfolio simulation. Returns {trades, equity_curve, open_positions}.

    Args:
        cfg: Configuration dict.
        quiet: Suppress progress output.
        entries_df: Pre-filtered entries DataFrame. If None, loads from CSV.
            Must have 'entry_price' and 'week_date' columns if provided.
        regime_df: Pre-loaded regime DataFrame. If None, loads from CSV.
        nifty_df: Pre-loaded Nifty DataFrame. If None, loads from CSV.
        stock_cache: Pre-loaded StockCacheManager. If None, creates new one.
        initial_positions: Pre-existing positions to carry over (for walk-forward).
        initial_cash: Starting cash (overrides cfg["starting_capital"] when set).
    """
    cols = _derive_columns(cfg)
    regime_col = cols["regime_col"]
    breadth_col = cols["breadth_col"]
    exit_ma_col = cols["exit_ma_col"]

    # Load data
    if entries_df is None:
        if not quiet:
            print("\n[Step 1] Loading and filtering entries ...")
        entries_df = load_and_filter_entries(cfg)
    else:
        if not quiet:
            print(f"\n[Step 1] Using pre-loaded entries ({len(entries_df):,} rows) ...")
        if "entry_price" not in entries_df.columns:
            raise ValueError("Pre-loaded entries_df must have an 'entry_price' column")
        if not pd.api.types.is_datetime64_any_dtype(entries_df["week_date"]):
            entries_df = entries_df.copy()
            entries_df["week_date"] = pd.to_datetime(entries_df["week_date"])
        entries_df.attrs.setdefault("entries_before_filter", len(entries_df))
        entries_df.attrs.setdefault("entries_after_filter", len(entries_df))

    if regime_df is None:
        if not quiet:
            print("\n[Step 2] Loading regime data ...")
        regime_df = load_regime(cfg)
    else:
        if not quiet:
            print("\n[Step 2] Using pre-loaded regime data ...")
    if not quiet:
        print(f"    Regime column: {regime_col}")
        print(f"    Breadth column: {breadth_col}")
        print(f"    Exit MA column: {exit_ma_col}")
        if regime_col not in regime_df.columns:
            print(f"    WARNING: {regime_col} not in regime_weekly.csv columns!")
        if breadth_col not in regime_df.columns:
            print(f"    WARNING: {breadth_col} not in regime_weekly.csv columns!")

    if nifty_df is None:
        if not quiet:
            print("\n[Step 3] Loading Nifty data ...")
        nifty_df = load_nifty()
    else:
        if not quiet:
            print("\n[Step 3] Using pre-loaded Nifty data ...")

    if stock_cache is not None:
        cache = stock_cache
        if not quiet:
            print("\n[Step 4] Using pre-loaded stock cache ...")
    else:
        if not quiet:
            print("\n[Step 4] Initializing stock cache ...")
        if cfg.get("triple_stack_enabled"):
            extra = set()
            for key in ["stall_tighten_ma", "extension_tighten_ma", "climactic_tighten_ma"]:
                ma_name = cfg.get(key, "")
                if ma_name.startswith("sma_"):
                    period = int(ma_name.split("_")[1])
                    extra.add(period)
            cache = StockCacheManager(STOCK_CACHE_DIR, extra_sma_periods=sorted(extra))
        else:
            cache = StockCacheManager(STOCK_CACHE_DIR)

    # Pre-load stock caches for all tickers in entries
    unique_tickers = entries_df["ticker"].unique()
    if not quiet:
        print(f"    Unique tickers in filtered entries: {len(unique_tickers)}")

    # Group entries by week_date for O(1) lookup
    entries_by_week: dict[pd.Timestamp, pd.DataFrame] = {}
    for wk, grp in entries_df.groupby("week_date"):
        entries_by_week[wk] = grp

    # Use actual week dates from regime_weekly.csv as simulation timeline
    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    all_weeks = regime_df.index[(regime_df.index >= start) & (regime_df.index <= end)]
    all_weeks = all_weeks.sort_values()

    if not quiet:
        print(f"\n[Step 5] Running simulation: {len(all_weeks)} weeks, "
              f"{cfg['start_date']} to {cfg['end_date']}")
        print(f"    Starting capital: ₹{cfg['starting_capital']:,.0f}")

    # Initialize (support carry-over from previous walk-forward segment)
    if initial_cash is not None:
        cash = float(initial_cash)
    else:
        cash = float(cfg["starting_capital"])

    if initial_positions is not None:
        positions: list[Position] = list(initial_positions)
        initial_invested = sum(p.qty * p.current_close for p in positions)
        start_equity = cash + initial_invested
    else:
        positions: list[Position] = []
        start_equity = cash

    closed_trades: list[dict] = []
    equity_curve: list[dict] = []
    running_peak_equity = start_equity
    prev_equity = start_equity
    fees = cfg["slippage_pct"] + cfg["brokerage_pct"]

    diag = defaultdict(int)
    t0 = time.time()

    for week_num, wk in enumerate(all_weeks):
        # ── 6a. Mark-to-market ────────────────────────────────────
        for pos in positions:
            sdf = cache.get(pos.ticker)
            if sdf is not None and wk in sdf.index:
                pos.current_close = float(sdf.loc[wk, "Close"])

        # ── 6b. Process exits (BEFORE entries — Rule 1) ──────────
        surviving: list[Position] = []
        for pos in positions:
            sdf = cache.get(pos.ticker)
            if sdf is None or wk not in sdf.index:
                surviving.append(pos)
                continue

            row = sdf.loc[wk]
            week_low = float(row["Low"])
            week_open = float(row["Open"])
            week_close = float(row["Close"])

            # Compute new trailing stop (Rule 3: ratchets up only)
            # Uses per-position exit config (supports multi-config carry-over)
            atr_val = row.get("atr_14")

            # Triple-stack exit tightening: determine which MA to use
            use_ma_col = pos.exit_ma_col
            if pos.pos_triple_stack_enabled and atr_val is not None and not pd.isna(atr_val):
                atr_f = float(atr_val)
                holding_w = max(1, round((wk - pos.entry_date).days / 7))

                # (a) Stall tightening: permanent once triggered
                if pos.tighten_state == "stall":
                    use_ma_col = pos.pos_stall_tighten_ma
                if (pos.tighten_state is None
                        and holding_w >= pos.pos_stall_tighten_week):
                    curr_ret = (week_close - pos.entry_price) / pos.entry_price
                    if curr_ret < pos.pos_stall_tighten_threshold:
                        pos.tighten_state = "stall"
                        use_ma_col = pos.pos_stall_tighten_ma
                        diag["triple_stall"] += 1

                # (b) Extension tightening: reversible, overrides stall
                baseline_ma = row.get(pos.exit_ma_col)
                if (baseline_ma is not None and not pd.isna(baseline_ma)
                        and week_close > float(baseline_ma) + pos.pos_extension_atr_mult * atr_f):
                    use_ma_col = pos.pos_extension_tighten_ma
                    diag["triple_extension"] += 1

                # (c) Climactic volume: permanent, highest priority
                if pos.tighten_state == "climactic":
                    use_ma_col = pos.pos_climactic_tighten_ma
                if (pos.tighten_state != "climactic"
                        and pos.prev_close > 0
                        and week_close < pos.prev_close):
                    vol_ratio = row.get("volume_ratio_20w")
                    if (vol_ratio is not None and not pd.isna(vol_ratio)
                            and float(vol_ratio) > pos.pos_climactic_vol_threshold):
                        pos.tighten_state = "climactic"
                        use_ma_col = pos.pos_climactic_tighten_ma
                        diag["triple_climactic"] += 1

                # Update prev_close for next week's climactic check
                pos.prev_close = week_close

            ma_val = row.get(use_ma_col)
            if ma_val is not None and atr_val is not None and not pd.isna(ma_val) and not pd.isna(atr_val):
                new_trail = float(ma_val) - pos.exit_atr_buffer * float(atr_val)
                pos.stop_level = max(pos.stop_level, new_trail)

            # Update MAE tracking (always, even on exit week)
            pos.min_low = min(pos.min_low, week_low)

            # GTT trigger check (Rule 5)
            if cfg["exit_gtt_field"] == "low":
                trigger_val = week_low
            else:
                trigger_val = week_close

            if trigger_val <= pos.stop_level:
                # Gap-down handling (Rule 4)
                if week_open < pos.stop_level:
                    exit_price = week_open
                    exit_reason = "gap_down"
                else:
                    exit_price = pos.stop_level
                    exit_reason = "gtt_stop"

                # Transaction costs on exit (Rule 12)
                proceeds = pos.qty * exit_price * (1 - fees)
                cash += proceeds
                trade = _build_trade(pos, exit_price, wk, exit_reason, proceeds)
                # Set regime at exit from regime_df
                exit_regime = "unknown"
                if wk in regime_df.index and regime_col in regime_df.columns:
                    val = regime_df.loc[wk, regime_col]
                    if not pd.isna(val):
                        exit_regime = str(val)
                trade["regime_at_exit"] = exit_regime
                closed_trades.append(trade)
                diag["exits"] += 1
            else:
                # Update peak close + MFE week (Rule 10)
                if week_close > pos.peak_close:
                    pos.peak_close = week_close
                    pos.mfe_week = wk
                pos.current_close = week_close
                surviving.append(pos)

        positions = surviving

        # ── 6c. Compute equity base (Rule 2) ─────────────────────
        invested_value = sum(p.qty * p.current_close for p in positions)
        total_equity = cash + invested_value

        if cfg["sizing_on"] == "total_equity":
            equity_base = total_equity
        else:
            equity_base = cash

        # ── 6d. Process entries ──────────────────────────────────
        week_entries = entries_by_week.get(wk)
        if week_entries is not None and len(week_entries) > 0:
            # Look up regime
            regime = "unknown"
            breadth_pct = 0.0
            if wk in regime_df.index:
                r = regime_df.loc[wk]
                if regime_col in r.index:
                    regime = str(r[regime_col])
                if breadth_col in r.index:
                    breadth_pct = float(r[breadth_col]) if not pd.isna(r[breadth_col]) else 0.0

            if is_entry_allowed(regime, breadth_pct, cfg):
                # Build exclusion sets
                held_tickers = {p.ticker for p in positions}
                sector_counts = Counter(p.sector for p in positions)

                # Filter candidates
                candidates = week_entries.copy()

                # Skip already held (Rule 8)
                candidates = candidates[~candidates["ticker"].isin(held_tickers)]

                # Skip sector limit (Rule 9) — filter per-candidate in loop below

                # Rank by rs_13w descending (Rule 7)
                candidates = candidates.sort_values("rs_13w", ascending=False, na_position="last")

                for _, entry_row in candidates.iterrows():
                    if len(positions) >= cfg["max_positions"]:
                        diag["max_pos"] += 1
                        break

                    sector = str(entry_row.get("sector", "UNKNOWN"))
                    if sector_counts.get(sector, 0) >= cfg["sector_limit"]:
                        diag["sector_limit"] += 1
                        continue

                    ep = float(entry_row["entry_price"])
                    sl = float(entry_row["stop_level"])
                    rr = float(entry_row["rr_ratio"]) if not pd.isna(entry_row["rr_ratio"]) else 0

                    sizing = compute_sizing(ep, sl, rr, equity_base, cash, cfg)
                    if sizing is None:
                        diag["sizing_fail"] += 1
                        continue

                    # Create position
                    rs_val = float(entry_row["rs_13w"]) if not pd.isna(entry_row["rs_13w"]) else 0
                    risk_val = float(entry_row["risk_pct_at_entry"]) if not pd.isna(entry_row["risk_pct_at_entry"]) else 0

                    pos = Position(
                        ticker=str(entry_row["ticker"]),
                        sector=sector,
                        entry_date=wk,
                        entry_price=ep,
                        qty=sizing["qty"],
                        cost=sizing["entry_cost"],
                        stop_level=sl,
                        tier=sizing["tier"],
                        peak_close=ep,
                        entry_type=str(entry_row["entry_type"]),
                        target_type=str(entry_row["target_type"]),
                        risk_pct_at_entry=risk_val,
                        rs_13w_at_entry=rs_val,
                        regime_at_entry=regime,
                        rr_at_entry=rr,
                        current_close=ep,
                        stop_at_entry=sl,
                        min_low=ep,
                        mfe_week=wk,
                        # Per-position exit config
                        exit_ma_col=exit_ma_col,
                        exit_atr_buffer=cfg["exit_atr_buffer"],
                        pos_triple_stack_enabled=cfg.get("triple_stack_enabled", False),
                        pos_stall_tighten_week=cfg.get("stall_tighten_week", 10),
                        pos_stall_tighten_threshold=cfg.get("stall_tighten_threshold", 0.05),
                        pos_stall_tighten_ma=cfg.get("stall_tighten_ma", "sma_20"),
                        pos_extension_atr_mult=cfg.get("extension_atr_mult", 3.0),
                        pos_extension_tighten_ma=cfg.get("extension_tighten_ma", "sma_15"),
                        pos_climactic_vol_threshold=cfg.get("climactic_vol_threshold", 2.5),
                        pos_climactic_tighten_ma=cfg.get("climactic_tighten_ma", "sma_15"),
                        config_id=cfg.get("config_id", "default"),
                    )
                    cash -= sizing["entry_cost"]
                    positions.append(pos)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    diag["entries"] += 1
            else:
                diag["regime_blocked"] += len(week_entries)

        # ── 6e. Update running peak and record snapshot ──────────
        invested_value = sum(p.qty * p.current_close for p in positions)
        total_equity = cash + invested_value
        running_peak_equity = max(running_peak_equity, total_equity)
        dd_pct = (total_equity - running_peak_equity) / running_peak_equity if running_peak_equity > 0 else 0
        weekly_ret = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        prev_equity = total_equity

        nifty_close = None
        if wk in nifty_df.index:
            nifty_close = float(nifty_df.loc[wk, "Close"])

        regime_snap = "unknown"
        breadth_snap = 0.0
        if wk in regime_df.index:
            r = regime_df.loc[wk]
            if regime_col in r.index:
                regime_snap = str(r[regime_col])
            if breadth_col in r.index:
                breadth_snap = float(r[breadth_col]) if not pd.isna(r[breadth_col]) else 0.0

        equity_curve.append({
            "date": str(wk.date()),
            "cash": round(cash, 2),
            "invested_value": round(invested_value, 2),
            "total_equity": round(total_equity, 2),
            "num_positions": len(positions),
            "regime": regime_snap,
            "weekly_return_pct": round(weekly_ret * 100, 4),
            "drawdown_pct": round(dd_pct * 100, 4),
            "nifty_close": round(nifty_close, 2) if nifty_close else None,
            "breadth_pct": round(breadth_snap, 2),
        })

        # Progress
        if not quiet and (week_num + 1) % 200 == 0:
            elapsed = time.time() - t0
            pct = (week_num + 1) / len(all_weeks) * 100
            print(f"    Week {week_num+1}/{len(all_weeks)} ({pct:.0f}%) | "
                  f"Equity: ₹{total_equity:,.0f} | Positions: {len(positions)} | "
                  f"Trades: {len(closed_trades)} | {elapsed:.0f}s")

    # ── End of simulation ─────────────────────────────────────────
    # Rule 13: positions still open stay open, NOT closed as trades
    # UNLESS force_close_at_end is set (for walk-forward windows)
    if cfg.get("force_close_at_end") and positions:
        for pos in positions:
            exit_price = pos.current_close
            proceeds = pos.qty * exit_price * (1 - fees)
            cash += proceeds
            trade = _build_trade(pos, exit_price, wk, "force_close", proceeds)
            # Use last known regime
            exit_regime = "unknown"
            if wk in regime_df.index and regime_col in regime_df.columns:
                val = regime_df.loc[wk, regime_col]
                if not pd.isna(val):
                    exit_regime = str(val)
            trade["regime_at_exit"] = exit_regime
            closed_trades.append(trade)
            diag["force_closed"] += 1
        positions = []

    elapsed = time.time() - t0
    final_invested = sum(p.qty * p.current_close for p in positions)
    final_equity = cash + final_invested

    if not quiet:
        print(f"\n  Simulation complete in {elapsed:.1f}s")
        print(f"  Final equity: ₹{final_equity:,.0f} | Cash: ₹{cash:,.0f} | Invested: ₹{final_invested:,.0f}")
        print(f"  Closed trades: {len(closed_trades)} | Open positions: {len(positions)}")
        print(f"  Diagnostics: {dict(diag)}")

    return {
        "trades": closed_trades,
        "equity_curve": equity_curve,
        "open_positions": positions,
        "diag": dict(diag),
        "entries_before_filter": entries_df.attrs.get("entries_before_filter", 0),
        "entries_after_filter": entries_df.attrs.get("entries_after_filter", 0),
    }


# ── Metrics Computation ──────────────────────────────────────────────

def compute_metrics(result: dict, nifty_df: pd.DataFrame, cfg: dict) -> dict:
    """Compute performance metrics for full, OOS, and IS periods."""
    trades = result["trades"]
    eq_curve = result["equity_curve"]
    open_pos = result["open_positions"]

    is_split = pd.Timestamp(cfg["is_split_date"])
    rf_weekly = cfg["risk_free_annual"] / 52

    def _slice_metrics(eq_slice: list[dict], trades_slice: list[dict], label: str) -> dict:
        if not eq_slice or len(eq_slice) < 2:
            return {"label": label, "error": "insufficient data"}

        equities = [s["total_equity"] for s in eq_slice]
        start_eq = equities[0]
        end_eq = equities[-1]
        weeks = len(equities)
        years = weeks / 52.0

        # CAGR
        if start_eq > 0 and years > 0:
            cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        # Weekly returns
        wk_rets = [(equities[i] - equities[i-1]) / equities[i-1]
                    for i in range(1, len(equities)) if equities[i-1] > 0]

        # Sharpe
        if len(wk_rets) > 1:
            avg_wk = np.mean(wk_rets)
            std_wk = np.std(wk_rets, ddof=1)
            sharpe = (avg_wk - rf_weekly) / std_wk * np.sqrt(52) if std_wk > 0 else 0
            down_rets = [r for r in wk_rets if r < 0]
            down_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else 1
            sortino = (avg_wk - rf_weekly) / down_std * np.sqrt(52) if down_std > 0 else 0
        else:
            sharpe = sortino = 0

        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for v in equities:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

        # MAR
        mar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        n_trades = len(trades_slice)
        winners = [t for t in trades_slice if t["return_pct"] > 0]
        losers = [t for t in trades_slice if t["return_pct"] <= 0]
        win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

        sum_wins = sum(t["pnl"] for t in winners)
        sum_losses = abs(sum(t["pnl"] for t in losers))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else (10.0 if sum_wins > 0 else 0)

        avg_holding = np.mean([t["holding_weeks"] for t in trades_slice]) if trades_slice else 0
        avg_give_back = np.mean([t["give_back_pp"] for t in trades_slice]) if trades_slice else 0
        avg_winner = np.mean([t["return_pct"] for t in winners]) if winners else 0
        avg_loser = np.mean([t["return_pct"] for t in losers]) if losers else 0

        # Max positions
        max_pos = max(s["num_positions"] for s in eq_slice) if eq_slice else 0

        # Nifty CAGR
        nifty_vals = [s["nifty_close"] for s in eq_slice if s["nifty_close"] is not None]
        if len(nifty_vals) >= 2 and nifty_vals[0] > 0 and years > 0:
            nifty_cagr = ((nifty_vals[-1] / nifty_vals[0]) ** (1 / years) - 1) * 100
        else:
            nifty_cagr = 0

        alpha = cagr - nifty_cagr

        # Pct weeks invested
        invested_weeks = sum(1 for s in eq_slice if s["invested_value"] > 0)
        pct_invested = invested_weeks / len(eq_slice) * 100 if eq_slice else 0

        return {
            "label": label,
            "start_equity": round(start_eq),
            "end_equity": round(end_eq),
            "cagr_pct": round(cagr, 2),
            "max_dd_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "mar": round(mar, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_holding_weeks": round(avg_holding, 1),
            "avg_give_back_pp": round(avg_give_back, 1),
            "avg_winner_pct": round(avg_winner, 1),
            "avg_loser_pct": round(avg_loser, 1),
            "max_simultaneous_positions": max_pos,
            "nifty_cagr_pct": round(nifty_cagr, 2),
            "alpha_pct": round(alpha, 2),
            "pct_weeks_invested": round(pct_invested, 1),
            "avg_mfe_pct": round(np.mean([t["mfe_pct"] for t in trades_slice]), 1) if trades_slice else 0,
            "avg_mae_pct": round(np.mean([t["mae_pct"] for t in trades_slice]), 1) if trades_slice else 0,
            "avg_capture_ratio": round(np.mean([t["capture_ratio"] for t in trades_slice]), 2) if trades_slice else 0,
            "avg_time_to_mfe_weeks": round(np.mean([t["time_to_mfe_weeks"] for t in trades_slice]), 1) if trades_slice else 0,
        }

    def _by_group(trades_slice: list[dict], group_key: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trades_slice:
            groups[t.get(group_key, "unknown")].append(t)
        result = {}
        for gname, gtrades in sorted(groups.items()):
            n = len(gtrades)
            wins = [t for t in gtrades if t["return_pct"] > 0]
            result[gname] = {
                "trade_count": n,
                "win_rate_pct": round(len(wins) / n * 100, 1) if n > 0 else 0,
                "avg_return_pct": round(np.mean([t["return_pct"] for t in gtrades]), 2) if n > 0 else 0,
                "avg_give_back_pp": round(np.mean([t["give_back_pp"] for t in gtrades]), 2) if n > 0 else 0,
                "avg_holding_weeks": round(np.mean([t["holding_weeks"] for t in gtrades]), 1) if n > 0 else 0,
                "avg_mfe_pct": round(np.mean([t["mfe_pct"] for t in gtrades]), 1) if n > 0 else 0,
                "avg_mae_pct": round(np.mean([t["mae_pct"] for t in gtrades]), 1) if n > 0 else 0,
                "avg_capture_ratio": round(np.mean([t["capture_ratio"] for t in gtrades]), 2) if n > 0 else 0,
                "avg_time_to_mfe_weeks": round(np.mean([t["time_to_mfe_weeks"] for t in gtrades]), 1) if n > 0 else 0,
            }
            if group_key == "entry_type":
                rpae = [t["risk_pct_at_entry"] for t in gtrades if t["risk_pct_at_entry"] is not None]
                result[gname]["avg_risk_pct_at_entry"] = round(np.mean(rpae), 4) if rpae else None
        return result

    # Split equity curve and trades by period
    eq_dates = [pd.Timestamp(s["date"]) for s in eq_curve]

    full_eq = eq_curve
    full_trades = trades

    is_eq = [s for s, d in zip(eq_curve, eq_dates) if d < is_split]
    is_trades = [t for t in trades if pd.Timestamp(t["exit_date"]) < is_split]

    oos_eq = [s for s, d in zip(eq_curve, eq_dates) if d >= is_split]
    oos_trades = [t for t in trades if pd.Timestamp(t["exit_date"]) >= is_split]

    # Yearly breakdown
    yearly = {}
    for t in trades:
        yr = pd.Timestamp(t["exit_date"]).year
        if yr not in yearly:
            yearly[yr] = []
        yearly[yr].append(t)

    yearly_metrics = {}
    for yr in sorted(yearly.keys()):
        yr_trades = yearly[yr]
        yr_eq = [s for s, d in zip(eq_curve, eq_dates) if d.year == yr]
        wins = [t for t in yr_trades if t["return_pct"] > 0]
        n = len(yr_trades)

        # Nifty return for year
        nifty_yr = [s["nifty_close"] for s in yr_eq if s["nifty_close"] is not None]
        nifty_ret = ((nifty_yr[-1] / nifty_yr[0]) - 1) * 100 if len(nifty_yr) >= 2 and nifty_yr[0] > 0 else 0

        # Strategy return for year
        eq_vals = [s["total_equity"] for s in yr_eq]
        strat_ret = ((eq_vals[-1] / eq_vals[0]) - 1) * 100 if len(eq_vals) >= 2 and eq_vals[0] > 0 else 0

        # Max DD for year
        peak = eq_vals[0] if eq_vals else 0
        max_dd_yr = 0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak > 0 else 0
            if dd < max_dd_yr:
                max_dd_yr = dd

        yearly_metrics[str(yr)] = {
            "strategy_return_pct": round(strat_ret, 2),
            "nifty_return_pct": round(nifty_ret, 2),
            "max_dd_pct": round(max_dd_yr, 2),
            "trades": n,
            "win_rate_pct": round(len(wins) / n * 100, 1) if n > 0 else 0,
        }

    metrics = {
        "full": _slice_metrics(full_eq, full_trades, "full"),
        "oos": _slice_metrics(oos_eq, oos_trades, "oos"),
        "is": _slice_metrics(is_eq, is_trades, "is"),
        "yearly": yearly_metrics,
        "open_positions_at_end": len(open_pos),
    }

    # by_target_type and by_entry_type for each slice
    for key, tslice in [("full", full_trades), ("oos", oos_trades), ("is", is_trades)]:
        metrics[key]["by_target_type"] = _by_group(tslice, "target_type")
        metrics[key]["by_entry_type"] = _by_group(tslice, "entry_type")

    return metrics


# ── Output Writers ────────────────────────────────────────────────────

def write_outputs(result: dict, metrics: dict, cfg: dict, out_dir: Path):
    """Write all output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # trades.csv
    if result["trades"]:
        pd.DataFrame(result["trades"]).to_csv(out_dir / "trades.csv", index=False)

    # equity_curve.csv
    pd.DataFrame(result["equity_curve"]).to_csv(out_dir / "equity_curve.csv", index=False)

    # summary.json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # config_echo.json
    with open(out_dir / "config_echo.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # run_metadata.json
    from datetime import datetime
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "data_files": {
            "entries_file": str(ENGINE_A_DIR / "entries_all.csv"),
            "regime_file": str(ENGINE_A_DIR / "regime_weekly.csv"),
            "stock_cache_dir": str(STOCK_CACHE_DIR),
        },
        "entries_before_filter": result.get("entries_before_filter", 0),
        "entries_after_filter": result.get("entries_after_filter", 0),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def generate_charts(result: dict, metrics: dict, cfg: dict, out_dir: Path):
    """Generate baseline charts as PNG files."""
    trades = result.get("trades", [])
    eq_curve = result.get("equity_curve", [])
    if not trades or not eq_curve:
        print("  No trades/equity data — skipping charts.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df_trades = pd.DataFrame(trades)
    df_eq = pd.DataFrame(eq_curve)
    df_eq["date"] = pd.to_datetime(df_eq["date"])
    is_split = pd.Timestamp(cfg["is_split_date"])

    # ── Chart 1: Equity Curve ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_eq["date"], df_eq["total_equity"], color="blue", linewidth=1.2, label="Strategy")
    # Scale Nifty to start at same value
    nifty = df_eq[["date", "nifty_close"]].dropna(subset=["nifty_close"])
    if len(nifty) > 1:
        scale = df_eq["total_equity"].iloc[0] / nifty["nifty_close"].iloc[0]
        ax.plot(nifty["date"], nifty["nifty_close"] * scale, color="red", linewidth=1, alpha=0.7,
                label=f"Nifty 50 (scaled)")
    ax.axvline(is_split, color="gray", linestyle="--", alpha=0.6, label="IS/OOS split")
    ax.set_title(f"Strategy vs Nifty 50 — {cfg['start_date']} to {cfg['end_date']}")
    ax.set_ylabel("Value (₹)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    # ── Chart 2: Drawdown ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df_eq["date"], df_eq["drawdown_pct"], 0, color="red", alpha=0.3)
    ax.plot(df_eq["date"], df_eq["drawdown_pct"], color="red", linewidth=0.8)
    worst_idx = df_eq["drawdown_pct"].idxmin()
    worst_date = df_eq.loc[worst_idx, "date"]
    worst_dd = df_eq.loc[worst_idx, "drawdown_pct"]
    ax.annotate(f"{worst_dd:.1f}%\n{worst_date.strftime('%Y-%m-%d')}",
                xy=(worst_date, worst_dd), fontsize=8, color="darkred",
                xytext=(20, -15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="darkred"))
    ax.axvline(is_split, color="gray", linestyle="--", alpha=0.6)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close()

    # ── Chart 3: MAE Distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    mae = df_trades["mae_pct"].dropna()
    ax.hist(mae, bins=30, color="salmon", edgecolor="darkred", alpha=0.7)
    med = mae.median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median: {med:.1f}%")
    ax.set_title(f"MAE Distribution (all trades) — median: {med:.1f}%")
    ax.set_xlabel("MAE %")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mae_distribution.png", dpi=150)
    plt.close()

    # ── Chart 4: MFE Distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    mfe = df_trades["mfe_pct"].dropna()
    ax.hist(mfe, bins=30, color="lightgreen", edgecolor="darkgreen", alpha=0.7)
    med = mfe.median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median: {med:.1f}%")
    ax.set_title(f"MFE Distribution (all trades) — median: {med:.1f}%")
    ax.set_xlabel("MFE %")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mfe_distribution.png", dpi=150)
    plt.close()

    # ── Chart 5: MFE vs Return ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    types = df_trades["entry_type"].unique()
    colors = {"PULLBACK_S2": "blue", "VCP_CONTINUATION": "green",
              "BREAKOUT_S1_TO_S2": "orange", "BREAKOUT_S2_CONTINUATION": "red"}
    for et in sorted(types):
        sub = df_trades[df_trades["entry_type"] == et]
        c = colors.get(et, "gray")
        ax.scatter(sub["mfe_pct"], sub["return_pct"], c=c, alpha=0.6, s=30, label=et, edgecolors="none")
    # Perfect capture diagonal
    lim = max(df_trades["mfe_pct"].max(), df_trades["return_pct"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="Perfect capture")
    ax.set_title("MFE vs Actual Return — gap = exit inefficiency")
    ax.set_xlabel("MFE %")
    ax.set_ylabel("Return %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mfe_vs_return.png", dpi=150)
    plt.close()

    # ── Chart 6: MAE by Entry Type ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    etypes = sorted(df_trades["entry_type"].unique())
    data = [df_trades.loc[df_trades["entry_type"] == et, "mae_pct"].dropna().values for et in etypes]
    short_labels = [e.replace("BREAKOUT_", "B_").replace("CONTINUATION", "CONT")
                    .replace("PULLBACK_", "PB_") for e in etypes]
    ax.boxplot(data, tick_labels=short_labels, vert=True, patch_artist=True,
               boxprops=dict(facecolor="salmon", alpha=0.6))
    ax.set_title("MAE by Entry Type")
    ax.set_ylabel("MAE %")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_entry_type.png", dpi=150)
    plt.close()

    # ── Chart 7: MFE by Entry Type ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df_trades.loc[df_trades["entry_type"] == et, "mfe_pct"].dropna().values for et in etypes]
    ax.boxplot(data, tick_labels=short_labels, vert=True, patch_artist=True,
               boxprops=dict(facecolor="lightgreen", alpha=0.6))
    ax.set_title("MFE by Entry Type")
    ax.set_ylabel("MFE %")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "mfe_by_entry_type.png", dpi=150)
    plt.close()

    print(f"  Charts saved to {out_dir}")


def _print_summary(metrics: dict, label: str = "BASELINE"):
    """Print summary table to console."""
    print(f"\n{'='*70}")
    print(f"  {label} RESULTS")
    print(f"{'='*70}")
    for period in ["full", "oos", "is"]:
        m = metrics.get(period, {})
        if "error" in m:
            print(f"\n  {period.upper()}: {m['error']}")
            continue
        print(f"\n  {period.upper()} ({m.get('label', '')})")
        print(f"    CAGR: {m['cagr_pct']:.1f}%  |  Max DD: {m['max_dd_pct']:.1f}%  |  "
              f"Sharpe: {m['sharpe']:.2f}  |  Sortino: {m['sortino']:.2f}  |  MAR: {m['mar']:.2f}")
        print(f"    Trades: {m['total_trades']}  |  Win Rate: {m['win_rate_pct']:.1f}%  |  "
              f"PF: {m['profit_factor']:.2f}  |  Avg Hold: {m['avg_holding_weeks']:.1f}w")
        print(f"    Avg Winner: {m['avg_winner_pct']:.1f}%  |  Avg Loser: {m['avg_loser_pct']:.1f}%  |  "
              f"Avg Give-back: {m['avg_give_back_pp']:.1f}pp")
        print(f"    MFE: {m.get('avg_mfe_pct', 0):.1f}%  |  MAE: {m.get('avg_mae_pct', 0):.1f}%  |  "
              f"Capture: {m.get('avg_capture_ratio', 0):.2f}  |  "
              f"Time-to-MFE: {m.get('avg_time_to_mfe_weeks', 0):.1f}w")
        print(f"    Nifty CAGR: {m['nifty_cagr_pct']:.1f}%  |  Alpha: {m['alpha_pct']:.1f}%  |  "
              f"Max Positions: {m['max_simultaneous_positions']}  |  "
              f"Invested: {m['pct_weeks_invested']:.0f}%")

    # Yearly
    print(f"\n  YEARLY BREAKDOWN")
    print(f"    {'Year':>4s}  {'Strat':>7s}  {'Nifty':>7s}  {'MaxDD':>7s}  {'Trades':>6s}  {'WinRate':>7s}")
    for yr, ym in sorted(metrics.get("yearly", {}).items()):
        print(f"    {yr:>4s}  {ym['strategy_return_pct']:>6.1f}%  {ym['nifty_return_pct']:>6.1f}%  "
              f"{ym['max_dd_pct']:>6.1f}%  {ym['trades']:>6d}  {ym['win_rate_pct']:>6.1f}%")

    # by_target_type for IS
    is_m = metrics.get("is", {})
    tt = is_m.get("by_target_type", {})
    if tt:
        print(f"\n  IS BY TARGET TYPE")
        for ttype, tdata in sorted(tt.items()):
            print(f"    {ttype}: {tdata['trade_count']} trades, "
                  f"WR {tdata['win_rate_pct']:.1f}%, "
                  f"Avg Ret {tdata['avg_return_pct']:.1f}%, "
                  f"Give-back {tdata['avg_give_back_pp']:.1f}pp, "
                  f"MFE {tdata.get('avg_mfe_pct', 0):.1f}%, "
                  f"MAE {tdata.get('avg_mae_pct', 0):.1f}%, "
                  f"Capture {tdata.get('avg_capture_ratio', 0):.2f}")

    # by_entry_type for IS
    et = is_m.get("by_entry_type", {})
    if et:
        print(f"\n  IS BY ENTRY TYPE")
        for etype, edata in sorted(et.items()):
            print(f"    {etype}: {edata['trade_count']} trades, "
                  f"WR {edata['win_rate_pct']:.1f}%, "
                  f"Avg Ret {edata['avg_return_pct']:.1f}%, "
                  f"Risk {edata.get('avg_risk_pct_at_entry', 'N/A')}, "
                  f"MFE {edata.get('avg_mfe_pct', 0):.1f}%, "
                  f"MAE {edata.get('avg_mae_pct', 0):.1f}%")


# ── Sensitivity Runner ────────────────────────────────────────────────

def run_sensitivity(baseline_cfg: dict, baseline_metrics: dict):
    """Run sensitivity checks and print comparison."""
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    base_is = baseline_metrics["is"]
    comparisons = []

    def _run_variant(label, overrides, subfolder):
        cfg = copy.deepcopy(baseline_cfg)
        cfg.update(overrides)
        print(f"\n  Running: {label} ...")
        result = run_simulation(cfg, quiet=True)
        nifty_df = load_nifty()
        metrics = compute_metrics(result, nifty_df, cfg)
        out_dir = OUTPUT_DIR / subfolder
        write_outputs(result, metrics, cfg, out_dir)

        is_m = metrics["is"]
        comparisons.append({
            "label": label,
            "is_cagr": is_m["cagr_pct"],
            "is_max_dd": is_m["max_dd_pct"],
            "is_sharpe": is_m["sharpe"],
            "is_trades": is_m["total_trades"],
            "delta_cagr": is_m["cagr_pct"] - base_is["cagr_pct"],
        })
        print(f"    IS CAGR: {is_m['cagr_pct']:.1f}% (Δ {is_m['cagr_pct'] - base_is['cagr_pct']:+.1f}pp)")

    # 1. Entry timing
    _run_variant("next_week_open", {"entry_price_method": "next_week_open"},
                 "sensitivity_next_week_open")

    # 2. Capital scaling
    for cap in [500_000, 1_000_000, 5_000_000]:
        _run_variant(f"capital_{cap//1000}K", {"starting_capital": cap},
                     f"sensitivity_capital_{cap}")

    # 3. Slippage
    for slip in [0.001, 0.003, 0.005]:
        _run_variant(f"slippage_{slip}", {"slippage_pct": slip},
                     f"sensitivity_slippage_{slip}")

    # Print comparison table
    print(f"\n  {'='*70}")
    print(f"  SENSITIVITY COMPARISON (IS period)")
    print(f"  {'Label':<25s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Trades':>6s}  {'ΔCAGR':>7s}")
    print(f"  {'-'*60}")
    print(f"  {'BASELINE':<25s}  {base_is['cagr_pct']:>6.1f}%  {base_is['max_dd_pct']:>6.1f}%  "
          f"{base_is['sharpe']:>7.2f}  {base_is['total_trades']:>6d}  {'—':>7s}")
    for c in comparisons:
        print(f"  {c['label']:<25s}  {c['is_cagr']:>6.1f}%  {c['is_max_dd']:>6.1f}%  "
              f"{c['is_sharpe']:>7.2f}  {c['is_trades']:>6d}  {c['delta_cagr']:>+6.1f}pp")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    run_sens = "--sensitivity" in sys.argv

    print("=" * 70)
    print("VAL-ENGINE-B: Portfolio Simulation Engine")
    print("=" * 70)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Config: MA={CONFIG['ma_type']}_{CONFIG['ma_period']}, "
          f"swing={CONFIG['swing_order_major']}, "
          f"base_min={CONFIG['base_minimum_weeks']}, "
          f"risk={CONFIG['risk_pct']}, "
          f"sizing={CONFIG['sizing_on']}")
    print(f"  Regime: {CONFIG['regime_type']} / {CONFIG['entry_filter']}")
    print(f"  Exit: {CONFIG['exit_ma_type']}_{CONFIG['exit_ma_period']} trail, "
          f"buffer={CONFIG['exit_atr_buffer']} ATR, trigger={CONFIG['exit_gtt_field']}")
    print(f"  Capital: ₹{CONFIG['starting_capital']:,.0f}")
    print(f"  Period: {CONFIG['start_date']} to {CONFIG['end_date']}, "
          f"IS split: {CONFIG['is_split_date']}")

    # Baseline run
    result = run_simulation(CONFIG)
    nifty_df = load_nifty()
    metrics = compute_metrics(result, nifty_df, CONFIG)
    write_outputs(result, metrics, CONFIG, OUTPUT_DIR)
    generate_charts(result, metrics, CONFIG, OUTPUT_DIR)
    _print_summary(metrics, "BASELINE")

    if run_sens:
        run_sensitivity(CONFIG, metrics)

    print(f"\n{'='*70}")
    print(f"VAL-ENGINE-B complete. Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
