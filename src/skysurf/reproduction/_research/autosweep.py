"""Autosweep v1: Parameter sweep runner for Engine B.

Loads entries once from Engine A v2, filters per config, calls Engine B
many times, logs results to TSV. Each run ~0.6s.

Usage:
    python scripts/v2_validation/autosweep.py --test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
from skysurf.reproduction import _paths

SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_A_V2_DIR = _paths.data_dir()
ENGINE_B_PATH = SCRIPT_DIR / "engine_locked.py"
OUTPUT_DIR = _paths.output_dir() / "autosweep_results"

# ── Mixed-MA mapping (from VAL-ENTRY-STATS optimal MA analysis) ──────
OPTIMAL_MAS = {
    "PULLBACK_S2":        ("EMA", 20),   # MFE/MAE=4.57
    "BREAKOUT_S1_TO_S2":  ("SMA", 40),   # MFE/MAE=2.51
    "VCP_CONTINUATION":   ("EMA", 40),   # MFE/MAE=2.06
    "RETEST_SUPPORT":     ("SMA", 40),   # MFE/MAE=1.89
    "TRENDLINE_BOUNCE":   ("EMA", 40),   # MFE/MAE=1.73
}

ACTIVE_ENTRY_TYPES = list(OPTIMAL_MAS.keys())

# ── Test config ───────────────────────────────────────────────────────
TEST_CONFIG = {
    "regime_filter": "breadth_5state_aggressive",
    "exit_ma_type": "SMA",
    "exit_ma_period": 25,
    "base_min_brk": 0,
    "base_min_pb": 4,
    "base_min_vcp": 4,
    "base_min_retest": 0,
    "base_min_trendline": 0,
    "rr_floor": 0.3,
    "rs_gate": None,
    "rsi_gate": None,
    "volume_gate": None,
    "max_risk_pct": None,
    "overlap_priority": "tightest_stop",
    "retest_max_weeks": None,
    "retest_min_proximity_atr": None,
    "entry_price_method": "week_close",
    "starting_capital": 500000,
    "risk_pct": 0.01,
    "sizing_on": "total_equity",
    "sector_limit": 3,
    "max_positions": 30,
    "slippage_pct": 0.001,
    "brokerage_pct": 0.0011,
}


# ── Sweep grid specification ──────────────────────────────────────────

SWEEP_SPEC = {
    "regime_filter": ["breadth_5state_aggressive", "weinstein_2signal"],
    "base_min_brk": [0, 2, 4],
    "base_min_pb": [4, 8, 12],
    "base_min_vcp": [4, 8, 12],
    "base_min_retest": [0, 4, 8],
    "base_min_trendline": [0, 4, 8],
    "rr_floor": [0.0, 0.2, 0.3, 0.5],
    "rs_gate": [None, 0.5, 1.0, 1.5],
    "rsi_gate": [None, 50, 60],
    "volume_gate": [None, 1.2, 1.5],
    "max_risk_pct": [0.08, 0.10, 0.15, None],
    "overlap_priority": ["tightest_stop", "highest_mfe"],
    "retest_max_weeks": [8, 12, 16, None],
    "retest_min_proximity_atr": [None, 0.5],
}

# Stage 1: fix less-important dimensions to reduce grid
STAGE1_OVERRIDES = {
    "overlap_priority": ["tightest_stop"],
    "volume_gate": [None],
    "max_risk_pct": [None],
    "retest_min_proximity_atr": [None],
}

# Fixed params not in the sweep grid (same for all configs)
FIXED_PARAMS = {
    "exit_ma_type": "SMA",
    "exit_ma_period": 25,
    "entry_price_method": "week_close",
    "starting_capital": 500000,
    "risk_pct": 0.01,
    "sizing_on": "total_equity",
    "sector_limit": 3,
    "max_positions": 30,
    "slippage_pct": 0.001,
    "brokerage_pct": 0.0011,
}


# ══════════════════════════════════════════════════════════════════════
# GRID GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_grid(spec: dict, fixed_params: dict) -> list[dict]:
    """Cartesian product of all sweep dimensions + fixed params."""
    keys = sorted(spec.keys())
    values = [spec[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params.update(fixed_params)
        grid.append(params)
    return grid


def apply_stage_overrides(spec: dict, overrides: dict) -> dict:
    """Return new spec with overrides applied."""
    return {**spec, **overrides}


def load_completed_run_ids(results_path: Path) -> set[str]:
    """Read existing results.tsv, return set of completed run_ids."""
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path, sep="\t")
        return set(df["run_id"].astype(str).tolist())
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════
# BEST TRACKER
# ══════════════════════════════════════════════════════════════════════

class BestTracker:
    """Track the best IS MAR during a sweep."""

    def __init__(self):
        self.best_mar = -999.0
        self.best_run_id = None
        self.staircase: list[tuple] = []

    def check(self, index: int, row: dict) -> bool:
        """Returns True if this is a new best."""
        if (row.get("status") == "pass"
                and row.get("passes_fitness")
                and row.get("is_mar", -999) > self.best_mar):
            self.best_mar = row["is_mar"]
            self.best_run_id = row["run_id"]
            self.staircase.append((
                index, row["run_id"], row["is_mar"],
                row["is_cagr"], row["is_trades"],
                row.get("complexity_score", 0),
            ))
            return True
        return False

    def print_staircase(self):
        if not self.staircase:
            print("\nNo configs passed fitness.")
            return
        print(f"\n{'='*60}")
        print("BEST CONFIGS (staircase — progression of new bests)")
        print(f"{'='*60}")
        for i, (idx, rid, mar, cagr, trades, cx) in enumerate(self.staircase, 1):
            print(f"  #{i:3d}  [config {idx:>5d}]  MAR={mar:.2f}  "
                  f"CAGR={cagr:.1f}%  trades={trades}  complexity={cx}")


# ══════════════════════════════════════════════════════════════════════
# SWEEP RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_sweep(
    grid: list[dict],
    all_entries: pd.DataFrame,
    nifty_df: pd.DataFrame,
    engine_b,
    results_path: Path,
    output_dir: Path,
    resume: bool = False,
    regime_df: "pd.DataFrame | None" = None,
    stock_cache=None,
):
    """Run all configs in grid, logging results to TSV."""
    completed = load_completed_run_ids(results_path) if resume else set()
    skipped = 0
    tracker = BestTracker()

    total = len(grid)
    start_time = time.time()
    completed_count = 0
    pass_count = 0
    fail_count = 0
    crash_count = 0

    print(f"\nGrid size: {total:,} configs")
    if resume and completed:
        remaining = sum(1 for g in grid if make_run_id(g) not in completed)
        print(f"Resuming: {len(completed)} already completed, {remaining} remaining")
    est_seconds = total * 1.5
    print(f"Estimated runtime: ~{est_seconds/3600:.1f}h (at 1.5s/run)\n")

    try:
        for i, sweep_params in enumerate(grid, 1):
            run_id = make_run_id(sweep_params)

            # Skip if already completed (resume mode)
            if run_id in completed:
                skipped += 1
                continue

            row = run_one_config(sweep_params, all_entries, nifty_df, engine_b,
                                regime_df=regime_df, stock_cache=stock_cache)

            # Track counts
            completed_count += 1
            if row["status"] == "pass":
                pass_count += 1
            elif row["status"] == "fail_fitness":
                fail_count += 1
            elif row["status"] == "crash":
                crash_count += 1

            # Check for new best
            is_new_best = tracker.check(i, row)

            # Progress line (every config for first 20, then every 50)
            if completed_count <= 20 or completed_count % 50 == 0 or is_new_best:
                if row["status"] == "crash":
                    print(f"[{i}/{total}] CRASH: {row.get('error', 'unknown')[:60]} "
                          f"({row['elapsed_sec']}s)")
                else:
                    star = " ★ NEW BEST" if is_new_best else ""
                    sym = "✓" if row["passes_fitness"] else "✗"
                    elapsed_total = time.time() - start_time
                    rate = completed_count / elapsed_total if elapsed_total > 0 else 0
                    eta_s = (total - i) / rate if rate > 0 else 0
                    print(f"[{i}/{total}] MAR={row['is_mar']:.2f} "
                          f"cagr={row['is_cagr']:.1f}% dd={row['is_max_dd']:.1f}% "
                          f"trades={row['is_trades']} cx={row['complexity_score']} "
                          f"{sym} ({row['elapsed_sec']}s) "
                          f"[ETA {eta_s/3600:.1f}h]{star}",
                          flush=True)

            # Write results (atomic per row)
            append_to_results_tsv(row, results_path)

            # Write metadata only for passing configs
            if row.get("passes_fitness"):
                cfg = build_engine_b_config(sweep_params)
                write_run_metadata(run_id, sweep_params, cfg, row, output_dir)

    except KeyboardInterrupt:
        print(f"\n\n--- INTERRUPTED after {completed_count} configs ---")

    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"Total in grid:  {total:,}")
    print(f"Completed:      {completed_count:,}"
          + (f" (+{skipped} resumed)" if skipped else ""))
    print(f"  Pass:         {pass_count:,} "
          f"({100*pass_count/max(completed_count,1):.1f}%)")
    print(f"  Fail fitness: {fail_count:,} "
          f"({100*fail_count/max(completed_count,1):.1f}%)")
    print(f"  Crash:        {crash_count:,} "
          f"({100*crash_count/max(completed_count,1):.1f}%)")
    print(f"Runtime:        {elapsed_total/3600:.1f}h "
          f"({elapsed_total/max(completed_count,1):.1f}s avg)")

    # Top 10 by IS MAR
    if results_path.exists():
        df = pd.read_csv(results_path, sep="\t")
        passed = df[df["passes_fitness"] == True].sort_values("is_mar", ascending=False)

        if len(passed) > 0:
            cols = ["run_id", "is_mar", "is_cagr", "is_max_dd", "is_trades",
                    "complexity_score", "regime_filter", "oos_mar", "oos_trades"]
            avail_cols = [c for c in cols if c in passed.columns]

            print(f"\n{'='*60}")
            print(f"TOP 10 BY IS MAR (fitness=pass)")
            print(f"{'='*60}")
            print(passed.head(10)[avail_cols].to_string(index=False))

            simple = passed[passed["complexity_score"] <= 2]
            if len(simple) > 0:
                print(f"\n{'='*60}")
                print(f"TOP 10 BY IS MAR (fitness=pass, complexity ≤ 2)")
                print(f"{'='*60}")
                print(simple.head(10)[avail_cols].to_string(index=False))

    # Staircase
    tracker.print_staircase()


# ══════════════════════════════════════════════════════════════════════
# ENGINE B LOADER
# ══════════════════════════════════════════════════════════════════════

def load_engine_b():
    """Load val_engine_b.py as a module via importlib."""
    if not ENGINE_B_PATH.exists():
        print(f"ERROR: {ENGINE_B_PATH} not found")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("val_engine_b", ENGINE_B_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass can resolve
    sys.modules["val_engine_b"] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════
# ENTRY LOADING (mixed-MA)
# ══════════════════════════════════════════════════════════════════════

def load_all_entries() -> pd.DataFrame:
    """Load entries from Engine A v2, selecting optimal MA per type."""
    path = ENGINE_A_V2_DIR / "entries_all.csv"
    print(f"Loading entries from {path.name}...")
    raw = pd.read_csv(path, low_memory=False)

    chunks = []
    for entry_type, (ma_type, ma_period) in OPTIMAL_MAS.items():
        mask = (
            (raw["entry_type"] == entry_type) &
            (raw["ma_type"] == ma_type) &
            (raw["ma_period"] == ma_period)
        )
        chunk = raw[mask].copy()
        chunks.append(chunk)
        print(f"  {entry_type} from {ma_type}_{ma_period}: {len(chunk):,} entries")

    combined = pd.concat(chunks, ignore_index=True)
    combined["week_date"] = pd.to_datetime(combined["week_date"])
    combined = combined.sort_values(["week_date", "ticker"]).reset_index(drop=True)
    print(f"  Total: {len(combined):,} entries")
    return combined


# ══════════════════════════════════════════════════════════════════════
# ENTRY FILTERING
# ══════════════════════════════════════════════════════════════════════

def filter_entries(all_entries: pd.DataFrame, sp: dict, out_diag: dict | None = None) -> pd.DataFrame:
    """Apply sweep filters to the master entries DataFrame.

    Sets the 'entry_price' column required by Engine B.

    Q-DETECT.1: when ``out_diag`` is provided (non-None), per-sub-filter
    survivor counts are written into it as ``sub_*`` keys. If a sub-filter
    produces a transition from non-zero to zero, the pre-filter rejected
    rows' relevant column values are sampled into ``sub_<name>_rejected_sample``
    to diagnose the failure mode (e.g., values vs threshold). Default
    ``None`` keeps the function byte-equivalent for legacy callers
    (research walk-forward, MAR 1.96 reproduction).
    """
    df = all_entries.copy()
    n0 = len(df)
    if out_diag is not None:
        out_diag["sub_in"] = n0

    def _record(label: str, prev_df: pd.DataFrame, after_df: pd.DataFrame,
                sample_cols: list[str] | None = None,
                threshold: object = None) -> None:
        if out_diag is None:
            return
        out_diag[f"sub_after_{label}"] = len(after_df)
        if len(after_df) == 0 and len(prev_df) > 0 and sample_cols:
            try:
                rej_sample = []
                for _, row in prev_df.head(5).iterrows():
                    rec = {c: (None if pd.isna(row.get(c)) else row[c]) for c in sample_cols if c in prev_df.columns}
                    rec = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in rec.items() if v is not None}
                    rej_sample.append(rec)
                out_diag[f"sub_{label}_rejected_sample"] = rej_sample
                if threshold is not None:
                    out_diag[f"sub_{label}_threshold"] = threshold
            except Exception:
                pass

    # 1. Per-type base_minimum (weeks_in_stage2)
    base_min_map = {
        "BREAKOUT_S1_TO_S2": sp["base_min_brk"],
        "PULLBACK_S2":       sp["base_min_pb"],
        "VCP_CONTINUATION":  sp["base_min_vcp"],
        "RETEST_SUPPORT":    sp["base_min_retest"],
        "TRENDLINE_BOUNCE":  sp["base_min_trendline"],
    }
    keep = pd.Series(True, index=df.index)
    for etype, min_weeks in base_min_map.items():
        if min_weeks > 0:
            is_type = df["entry_type"] == etype
            passes = df["weeks_in_stage2"] >= min_weeks
            keep &= (~is_type | passes)
    prev = df
    df = df[keep]
    _record("base_min", prev, df, ["ticker", "entry_type", "weeks_in_stage2"], threshold=base_min_map)

    # 2. R/R floor (structural targets only)
    if sp["rr_floor"] and sp["rr_floor"] > 0:
        drop = (df["target_type"] == "structural") & (df["rr_ratio"] < sp["rr_floor"])
        prev = df
        df = df[~drop]
        _record("rr_floor", prev, df, ["ticker", "entry_type", "target_type", "rr_ratio"], threshold=sp["rr_floor"])

    # 3. Quality gates
    if sp["rs_gate"] is not None:
        prev = df
        df = df[df["rs_13w"].fillna(-999) >= sp["rs_gate"]]
        _record("rs_gate", prev, df, ["ticker", "entry_type", "rs_13w"], threshold=sp["rs_gate"])

    if sp["rsi_gate"] is not None:
        prev = df
        df = df[df["rsi_at_entry"].fillna(-999) >= sp["rsi_gate"]]
        _record("rsi_gate", prev, df, ["ticker", "entry_type", "rsi_at_entry"], threshold=sp["rsi_gate"])

    if sp["volume_gate"] is not None:
        prev = df
        df = df[df["volume_ratio_at_entry"].fillna(-999) >= sp["volume_gate"]]
        _record("volume_gate", prev, df, ["ticker", "entry_type", "volume_ratio_at_entry"], threshold=sp["volume_gate"])

    if sp["max_risk_pct"] is not None:
        prev = df
        df = df[df["risk_pct_at_entry"].fillna(999) <= sp["max_risk_pct"]]
        _record("max_risk_pct", prev, df, ["ticker", "entry_type", "risk_pct_at_entry"], threshold=sp["max_risk_pct"])

    # 4. RETEST-specific filters
    if sp["retest_max_weeks"] is not None:
        is_retest = df["entry_type"] == "RETEST_SUPPORT"
        too_old = df["weeks_since_breakout"].fillna(0) > sp["retest_max_weeks"]
        prev = df
        df = df[~(is_retest & too_old)]
        _record("retest_max_weeks", prev, df, ["ticker", "weeks_since_breakout"], threshold=sp["retest_max_weeks"])

    if sp["retest_min_proximity_atr"] is not None:
        is_retest = df["entry_type"] == "RETEST_SUPPORT"
        too_close = df["proximity_atr"].fillna(0) < sp["retest_min_proximity_atr"]
        prev = df
        df = df[~(is_retest & too_close)]
        _record("retest_min_proximity_atr", prev, df, ["ticker", "proximity_atr"], threshold=sp["retest_min_proximity_atr"])

    # 5. Overlap dedup: same (ticker, week_date) → keep tightest stop
    prev = df
    df = df.sort_values("risk_pct_at_entry", na_position="last")
    df = df.drop_duplicates(subset=["ticker", "week_date"], keep="first")
    _record("dedup", prev, df, None)

    # 6. Set entry_price column (required by Engine B)
    if sp["entry_price_method"] == "next_week_open":
        prev = df
        df = df.dropna(subset=["next_week_open"])
        df["entry_price"] = df["next_week_open"]
        _record("next_week_open_dropna", prev, df, ["ticker", "next_week_open"])
    else:
        df["entry_price"] = df["entry_price_close"]

    # 7. Date range — lower bound only (defensive; no pre-2007 OHLCV data anyway).
    # Q-DATE.1: removed hardcoded upper bound `<= "2026-03-21"`. That date is the
    # research data-freeze marker (val_engine_a_v2's END_DATE default); it killed
    # every brain signal whose as_of_date was past 2026-03-21 in production. The
    # upper bound was redundant for both call paths:
    #   - Research's batch sweep: input `entries_all_lagged.csv` is itself
    #     date-bounded by val_engine_a_v2's END_DATE env var, so the filter
    #     would have been a no-op even before this change (input never has
    #     post-freeze rows).
    #   - Brain's per-week flow: `_signals_from_raw_entries` pins
    #     `week_date == as_of_date` BEFORE calling filter_entries, so every
    #     row reaching this point already has a single, caller-controlled
    #     date — no upper bound needed for safety.
    # Removal is byte-equivalent for reproduction (input never contained
    # post-freeze rows). If a future caller needs an upper bound, pass it via
    # `sp.get("end_date")` and gate explicitly here; do not re-hardcode.
    prev = df
    df = df[df["week_date"] >= "2007-01-01"]
    _record("date_range", prev, df, ["ticker", "week_date"], threshold=">= 2007-01-01")

    df = df.sort_values("week_date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════
# CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════

def _parse_regime_type(s: str) -> str:
    return "breadth_5state" if s.startswith("breadth") else "weinstein_2signal"


def _parse_entry_filter(s: str) -> str:
    for f in ("aggressive", "moderate", "conservative"):
        if f in s:
            return f
    return "aggressive"


def build_engine_b_config(sp: dict) -> dict:
    """Translate sweep params into Engine B's config dict."""
    return {
        # MA for regime derivation + exit trailing stop
        "ma_type": sp.get("exit_ma_type", "SMA"),
        "ma_period": sp.get("exit_ma_period", 25),
        "swing_order_major": 8,  # not used when entries pre-loaded, but required

        # Zero all entry filters — autosweep handles filtering
        "base_minimum_weeks": 0,
        "entry_types": ["BREAKOUT", "PULLBACK", "VCP", "RETEST", "TRENDLINE"],
        "rr_floor": 0,
        "max_risk_pct": None,
        "max_risk_pct_all": None,
        "rs_gate": None,
        "volume_gate": None,
        "sector_gate": False,
        "entry_price_method": sp.get("entry_price_method", "week_close"),

        # Regime
        "regime_type": _parse_regime_type(sp["regime_filter"]),
        "entry_filter": _parse_entry_filter(sp["regime_filter"]),

        # Exit
        "exit_type": "sma_trail",
        "exit_ma_type": sp.get("exit_ma_type", "SMA"),
        "exit_ma_period": sp.get("exit_ma_period", 25),
        "exit_atr_buffer": 1.0,
        "exit_gtt_field": "low",

        # Sizing
        "risk_pct": sp.get("risk_pct", 0.01),
        "sizing_on": sp.get("sizing_on", "total_equity"),
        "tier_rr_thresholds": [1.5, 2.0],
        "concentration_caps": [0.08, 0.15, 0.25],
        "tier_multipliers": [0.5, 0.75, 1.0],

        # Constraints
        "sector_limit": sp.get("sector_limit", 3),
        "max_positions": sp.get("max_positions", 30),

        # Costs
        "slippage_pct": sp.get("slippage_pct", 0.001),
        "brokerage_pct": sp.get("brokerage_pct", 0.0011),

        # Ranking
        "ranking_method": "rs_13w",

        # Period
        "starting_capital": sp.get("starting_capital", 500_000),
        "start_date": "2007-01-01",
        "end_date": "2026-03-21",
        "is_split_date": "2020-01-01",

        # Metrics
        "risk_free_annual": 0.06,

        # Triple-stack exit tightening
        "triple_stack_enabled": sp.get("triple_stack_enabled", False),
        "stall_tighten_week": 10,
        "stall_tighten_threshold": 0.05,
        "stall_tighten_ma": "sma_20",
        "extension_atr_mult": 3.0,
        "extension_tighten_ma": "sma_15",
        "climactic_vol_threshold": 2.5,
        "climactic_tighten_ma": "sma_15",
    }


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def compute_complexity_score(sp: dict) -> int:
    """Count active (non-None, non-default) filter params."""
    score = 0
    if sp.get("rs_gate") is not None:
        score += 1
    if sp.get("rsi_gate") is not None:
        score += 1
    if sp.get("volume_gate") is not None:
        score += 1
    if sp.get("max_risk_pct") is not None:
        score += 1
    if sp.get("retest_max_weeks") is not None:
        score += 1
    if sp.get("retest_min_proximity_atr") is not None:
        score += 1
    return score


def make_run_id(sp: dict) -> str:
    """Deterministic hash of sweep params for resume capability."""
    config_str = json.dumps(sp, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════
# CORE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_one_config(
    sp: dict,
    all_entries: pd.DataFrame,
    nifty_df: pd.DataFrame,
    engine_b,
    regime_df: "pd.DataFrame | None" = None,
    stock_cache=None,
) -> dict:
    """Run one sweep config end-to-end. Returns flat result dict."""
    run_id = make_run_id(sp)
    n_before = len(all_entries)

    # Crash-safe wrapper
    try:
        filtered = filter_entries(all_entries, sp)
        n_after = len(filtered)

        if n_after == 0:
            return _empty_row(run_id, sp, n_before, 0, "fail_no_entries")

        cfg = build_engine_b_config(sp)

        t0 = time.time()
        result = engine_b.run_simulation(
            cfg, quiet=True, entries_df=filtered,
            regime_df=regime_df, nifty_df=nifty_df, stock_cache=stock_cache,
        )
        metrics = engine_b.compute_metrics(result, nifty_df, cfg)
        elapsed = time.time() - t0

        is_m = metrics.get("is", {})
        oos_m = metrics.get("oos", {})
        full_m = metrics.get("full", {})

        # Fitness check
        passes = (
            is_m.get("total_trades", 0) >= 30
            and is_m.get("cagr_pct", 0) > 0
            and is_m.get("max_dd_pct", -100) > -40
            and is_m.get("profit_factor", 0) > 1.0
        )
        status = "pass" if passes else "fail_fitness"

        return {
            "run_id": run_id,
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 2),
            "status": status,
            # Sweep params
            "regime_filter": sp["regime_filter"],
            "base_min_brk": sp["base_min_brk"],
            "base_min_pb": sp["base_min_pb"],
            "base_min_vcp": sp["base_min_vcp"],
            "base_min_retest": sp.get("base_min_retest", 0),
            "base_min_trendline": sp.get("base_min_trendline", 0),
            "rr_floor": sp["rr_floor"],
            "rs_gate": sp.get("rs_gate"),
            "rsi_gate": sp.get("rsi_gate"),
            "volume_gate": sp.get("volume_gate"),
            "max_risk_pct": sp.get("max_risk_pct"),
            "overlap_priority": sp.get("overlap_priority", "tightest_stop"),
            "retest_max_weeks": sp.get("retest_max_weeks"),
            "retest_min_proximity_atr": sp.get("retest_min_proximity_atr"),
            "triple_stack_enabled": sp.get("triple_stack_enabled", False),
            "complexity_score": compute_complexity_score(sp),
            "entries_before_filters": n_before,
            "entries_after_filters": n_after,
            # IS metrics
            "is_cagr": is_m.get("cagr_pct", 0),
            "is_max_dd": is_m.get("max_dd_pct", 0),
            "is_sharpe": is_m.get("sharpe", 0),
            "is_sortino": is_m.get("sortino", 0),
            "is_mar": is_m.get("mar", 0),
            "is_trades": is_m.get("total_trades", 0),
            "is_win_rate": is_m.get("win_rate_pct", 0),
            "is_profit_factor": is_m.get("profit_factor", 0),
            "is_avg_hold_weeks": is_m.get("avg_holding_weeks", 0),
            "is_pct_invested": is_m.get("pct_weeks_invested", 0),
            "is_capture_ratio": is_m.get("avg_capture_ratio", 0),
            # OOS metrics
            "oos_cagr": oos_m.get("cagr_pct", 0),
            "oos_max_dd": oos_m.get("max_dd_pct", 0),
            "oos_sharpe": oos_m.get("sharpe", 0),
            "oos_mar": oos_m.get("mar", 0),
            "oos_trades": oos_m.get("total_trades", 0),
            # Full metrics
            "full_cagr": full_m.get("cagr_pct", 0),
            "full_max_dd": full_m.get("max_dd_pct", 0),
            "full_sharpe": full_m.get("sharpe", 0),
            "full_mar": full_m.get("mar", 0),
            "full_trades": full_m.get("total_trades", 0),
            # Benchmark
            "nifty_cagr_is": is_m.get("nifty_cagr_pct", 0),
            "nifty_cagr_oos": oos_m.get("nifty_cagr_pct", 0),
            "nifty_cagr_full": full_m.get("nifty_cagr_pct", 0),
            "is_excess_return": is_m.get("alpha_pct", 0),
            "passes_fitness": passes,
        }

    except Exception as e:
        return _empty_row(run_id, sp, n_before, 0, "crash", str(e))


def _empty_row(run_id, sp, n_before, n_after, status, error=""):
    """Build a result row with zeroed metrics."""
    metric_zeros = {
        "is_cagr": 0, "is_max_dd": 0, "is_sharpe": 0, "is_sortino": 0, "is_mar": 0,
        "is_trades": 0, "is_win_rate": 0, "is_profit_factor": 0,
        "is_avg_hold_weeks": 0, "is_pct_invested": 0, "is_capture_ratio": 0,
        "oos_cagr": 0, "oos_max_dd": 0, "oos_sharpe": 0, "oos_mar": 0, "oos_trades": 0,
        "full_cagr": 0, "full_max_dd": 0, "full_sharpe": 0, "full_mar": 0, "full_trades": 0,
        "nifty_cagr_is": 0, "nifty_cagr_oos": 0, "nifty_cagr_full": 0,
        "is_excess_return": 0, "passes_fitness": False,
    }
    return {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "elapsed_sec": 0,
        "status": status,
        "error": error,
        "regime_filter": sp["regime_filter"],
        "base_min_brk": sp["base_min_brk"],
        "base_min_pb": sp["base_min_pb"],
        "base_min_vcp": sp["base_min_vcp"],
        "base_min_retest": sp.get("base_min_retest", 0),
        "base_min_trendline": sp.get("base_min_trendline", 0),
        "rr_floor": sp["rr_floor"],
        "rs_gate": sp.get("rs_gate"),
        "rsi_gate": sp.get("rsi_gate"),
        "volume_gate": sp.get("volume_gate"),
        "max_risk_pct": sp.get("max_risk_pct"),
        "overlap_priority": sp.get("overlap_priority", "tightest_stop"),
        "retest_max_weeks": sp.get("retest_max_weeks"),
        "retest_min_proximity_atr": sp.get("retest_min_proximity_atr"),
        "complexity_score": compute_complexity_score(sp),
        "entries_before_filters": n_before,
        "entries_after_filters": n_after,
        **metric_zeros,
    }


# ══════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════

def append_to_results_tsv(row: dict, path: Path):
    """Append one row to the TSV. Creates file with header if needed."""
    # Exclude 'error' key from TSV (it's in metadata JSON)
    row_clean = {k: v for k, v in row.items() if k != "error"}
    df = pd.DataFrame([row_clean])

    if not path.exists():
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, sep="\t", index=False, mode="a", header=False)


def write_run_metadata(run_id: str, sp: dict, cfg: dict, result: dict, out_dir: Path):
    """Write per-run metadata JSON."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Git status
    git_status = ""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain", "scripts/v2_validation/"],
            capture_output=True, text=True, timeout=5,
        )
        git_status = r.stdout.strip()
    except Exception:
        git_status = "unknown"

    metadata = {
        "run_id": run_id,
        "timestamp": result.get("timestamp", ""),
        "sweep_params": sp,
        "engine_b_config": cfg,
        "result_summary": {k: v for k, v in result.items()
                          if k not in ("sweep_params", "engine_b_config")},
        "git_status": git_status,
    }

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def _preload_caches(engine_b, all_entries, triple_stack: bool = False):
    """Pre-load regime_df, nifty_df, and stock cache once for all configs."""
    print("Pre-loading shared data (regime, nifty, stock cache)...")
    t0 = time.time()

    nifty_df = engine_b.load_nifty()
    print(f"  Nifty: {len(nifty_df)} weeks")

    regime_df = engine_b.load_regime(engine_b.CONFIG)
    print(f"  Regime: {len(regime_df)} weeks")

    # Create one StockCacheManager, pre-warm for all tickers in entries
    extra_sma = [15] if triple_stack else []
    stock_cache = engine_b.StockCacheManager(engine_b.STOCK_CACHE_DIR, extra_sma_periods=extra_sma)
    tickers = all_entries["ticker"].unique()
    for ticker in tickers:
        stock_cache.get(ticker)
    print(f"  Stock cache: {len(tickers)} tickers pre-loaded")

    elapsed = time.time() - t0
    print(f"  Pre-load complete in {elapsed:.1f}s\n")
    return nifty_df, regime_df, stock_cache


def _parse_filter_value(val_str: str):
    """Parse a --filter value string to the correct Python type."""
    if val_str == "None":
        return None
    try:
        val = float(val_str)
        if val == int(val):
            return int(val)
        return val
    except ValueError:
        return val_str


def merge_shard_tsvs(output_dir: Path):
    """Merge all results_shard*.tsv into results.tsv. Verify no duplicate run_ids."""
    shard_files = sorted(output_dir.glob("results_shard*.tsv"))
    if not shard_files:
        print("No shard files found to merge.")
        return

    dfs = []
    for sf in shard_files:
        df = pd.read_csv(sf, sep="\t")
        print(f"  {sf.name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    n_dupes = merged["run_id"].duplicated().sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate run_ids found — removing duplicates")
        merged = merged.drop_duplicates(subset="run_id", keep="first")

    out_path = output_dir / "results.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Merged: {len(merged)} rows → {out_path}")
    print(f"  From {len(shard_files)} shard files")


def main():
    parser = argparse.ArgumentParser(description="Autosweep: VAL-ENTRY-SWEEP runner")
    parser.add_argument("--test", action="store_true", help="Run single test config")
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 (reduced grid)")
    parser.add_argument("--full", action="store_true", help="Run full grid (all dimensions)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed run_ids")
    parser.add_argument("--filter", type=str, default=None,
                        help="Key=value filter, e.g. 'regime_filter=weinstein_2signal'")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard N/M, e.g. '1/4' for first of 4 shards")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shard TSVs into results.tsv")
    parser.add_argument("--triple-stack", action="store_true",
                        help="Add triple_stack ON/OFF dimension (doubles grid size)")
    args = parser.parse_args()

    print("=== AUTOSWEEP: VAL-ENTRY-SWEEP ===\n")

    # ── Merge mode (no simulation, just combine files) ────────
    if args.merge:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merge_shard_tsvs(OUTPUT_DIR)
        return

    # Load Engine B
    print("Loading Engine B from val_engine_b.py...")
    engine_b = load_engine_b()

    # Load entries (mixed-MA)
    all_entries = load_all_entries()

    # Pre-load shared data (regime, nifty, stock cache)
    nifty_df, regime_df, stock_cache = _preload_caches(engine_b, all_entries, triple_stack=args.triple_stack)

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        # ── Test config ──────────────────────────────────────────
        results_path = OUTPUT_DIR / "results.tsv"

        test_variants = [False]
        if args.triple_stack:
            test_variants = [False, True]

        for tri_idx, tri_enabled in enumerate(test_variants):
            test_sp = {**TEST_CONFIG, "triple_stack_enabled": tri_enabled}
            tri_label = "TriStack ON" if tri_enabled else "Plain"

            print(f"\nTest config ({tri_label}):")
            for k, v in test_sp.items():
                print(f"  {k}={v}")
            print(f"  complexity_score={compute_complexity_score(test_sp)}")

            filtered = filter_entries(all_entries, test_sp)
            print(f"\nEntries after filters: {len(filtered):,} (from {len(all_entries):,})")

            print("\nRunning simulation...")
            row = run_one_config(test_sp, all_entries, nifty_df, engine_b,
                                 regime_df=regime_df, stock_cache=stock_cache)

            n_total = tri_idx + 1
            n_tests = len(test_variants)
            if row["status"] == "crash":
                print(f"\n[{n_total}/{n_tests}] {tri_label} CRASH: {row.get('error', 'unknown')}")
            else:
                sym = "✓" if row["passes_fitness"] else "✗"
                print(f"\n[{n_total}/{n_tests}] {tri_label} is_mar={row['is_mar']:.2f} "
                      f"cagr={row['is_cagr']:.1f}% dd={row['is_max_dd']:.1f}% "
                      f"trades={row['is_trades']} triple_stack={tri_enabled} "
                      f"{sym} {row['status'].upper()} ({row['elapsed_sec']}s)")

            append_to_results_tsv(row, results_path)
            cfg = build_engine_b_config(test_sp)
            write_run_metadata(row["run_id"], test_sp, cfg, row, OUTPUT_DIR)

        print(f"\nResults → {results_path}")

    elif args.stage1 or args.full:
        # ── Sweep mode ───────────────────────────────────────────
        if args.stage1:
            spec = apply_stage_overrides(SWEEP_SPEC, STAGE1_OVERRIDES)
            print("Mode: STAGE 1 (reduced grid)")
        else:
            spec = SWEEP_SPEC
            print("Mode: FULL GRID")

        grid = generate_grid(spec, FIXED_PARAMS)

        # Apply --filter if provided
        if args.filter:
            key, val_str = args.filter.split("=", 1)
            val = _parse_filter_value(val_str)
            grid = [g for g in grid if g.get(key) == val]
            print(f"Filter: {key}={val} → {len(grid):,} configs")

        # Inject triple-stack dimension (before sharding for even distribution)
        if args.triple_stack:
            expanded = []
            for g in grid:
                expanded.append({**g, "triple_stack_enabled": False})
                expanded.append({**g, "triple_stack_enabled": True})
            grid = expanded
            print(f"Triple-stack: ON/OFF dimension → {len(grid):,} configs")
        else:
            grid = [{**g, "triple_stack_enabled": False} for g in grid]

        # Apply --shard if provided
        if args.shard:
            shard_num, shard_total = map(int, args.shard.split("/"))
            grid = [g for i, g in enumerate(grid) if i % shard_total == shard_num - 1]
            print(f"Shard: {shard_num}/{shard_total} → {len(grid):,} configs")

        # Determine output file (per-shard or single)
        if args.shard:
            shard_num, _ = map(int, args.shard.split("/"))
            results_path = OUTPUT_DIR / f"results_shard{shard_num}.tsv"
        else:
            results_path = OUTPUT_DIR / "results.tsv"

        # Print spec
        print(f"\nSweep dimensions:")
        for k in sorted(spec.keys()):
            print(f"  {k}: {spec[k]}")
        print(f"\nFixed params:")
        for k, v in sorted(FIXED_PARAMS.items()):
            print(f"  {k}: {v}")

        # Run sweep
        run_sweep(grid, all_entries, nifty_df, engine_b,
                  results_path, OUTPUT_DIR, resume=args.resume,
                  regime_df=regime_df, stock_cache=stock_cache)

    else:
        # ── No mode: print usage + grid sizes ────────────────────
        s1_spec = apply_stage_overrides(SWEEP_SPEC, STAGE1_OVERRIDES)
        s1_grid = generate_grid(s1_spec, FIXED_PARAMS)
        full_grid = generate_grid(SWEEP_SPEC, FIXED_PARAMS)

        print("No mode specified. Options:")
        print("  --test       Run single test config")
        print("  --stage1     Run Stage 1 (reduced grid)")
        print("  --full       Run full grid")
        print("  --shard N/M  Run shard N of M (e.g., --shard 1/4)")
        print("  --merge      Merge shard TSVs into results.tsv")
        print("  --resume     Skip completed configs")
        print("  --filter k=v Subset grid by dimension")
        print(f"\n  Stage 1 grid: {len(s1_grid):,} configs "
              f"(~{len(s1_grid)*1.5/3600:.1f}h at 1.5s/run)")
        print(f"  Full grid:    {len(full_grid):,} configs "
              f"(~{len(full_grid)*1.5/3600:.1f}h at 1.5s/run)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial results saved.")
        sys.exit(1)
