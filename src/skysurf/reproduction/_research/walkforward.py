#!/usr/bin/env python3
"""Phase 4 Continuous Walk-Forward — positions + cash carry across 6 segments.

Produces a single realistic equity curve from 2010-2026.
3 configs: Baseline, Phase4 Best, Phase4 + Time Stop.

Usage:
    cd <project_root>
    source venv/bin/activate
    python scripts/v2_validation/run_phase4_walkforward_continuous.py
"""
from __future__ import annotations

import copy
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from skysurf.reproduction import _paths
from skysurf.reproduction._research import autosweep as auto
from skysurf.reproduction._research import engine as engine_exp
from skysurf.reproduction._research import engine_locked
from skysurf.reproduction._research.type_prior import (
    DEFAULT_PRIOR,
    TRADE_STATS_PATH,
    compute_type_prior,
)

BASE_DIR = Path(__file__).resolve().parent
# All input CSVs live in a single flat reproduction data bundle.
V2 = _paths.data_dir()

# OPTIMAL_MAS used to select per-type lagged entries (matches generate_phase4_report_data_v2.py)
OPTIMAL_MAS = {
    "PULLBACK_S2": ("EMA", 20), "BREAKOUT_S1_TO_S2": ("SMA", 40),
    "VCP_CONTINUATION": ("EMA", 40), "RETEST_SUPPORT": ("SMA", 40),
    "TRENDLINE_BOUNCE": ("EMA", 40),
}

# ── Segments (from val_walk_forward_continuous.py) ───────────────────
SEGMENTS = [
    {"id": 0, "apply_from": "2010-01-01", "apply_until": "2012-12-31"},
    {"id": 1, "apply_from": "2013-01-01", "apply_until": "2015-12-31"},
    {"id": 2, "apply_from": "2016-01-01", "apply_until": "2018-12-31"},
    {"id": 3, "apply_from": "2019-01-01", "apply_until": "2021-12-31"},
    {"id": 4, "apply_from": "2022-01-01", "apply_until": "2024-12-31"},
    {"id": 5, "apply_from": "2025-01-01", "apply_until": "2026-03-21"},
]

# ── Tier 2 Winner Config ─────────────────────────────────────────────
SP_BASELINE = {
    "regime_filter": "breadth_5state_aggressive", "base_min_brk": 0, "base_min_pb": 12,
    "base_min_vcp": 4, "base_min_retest": 8, "base_min_trendline": 8, "rr_floor": 0.5,
    "rs_gate": 0.5, "rsi_gate": None, "retest_max_weeks": 12, "overlap_priority": "tightest_stop",
    "volume_gate": None, "max_risk_pct": None, "retest_min_proximity_atr": None,
    "entry_price_method": "week_close", "starting_capital": 500000, "risk_pct": 0.01,
    "sizing_on": "total_equity", "sector_limit": 3, "max_positions": 30,
    "slippage_pct": 0.001, "brokerage_pct": 0.0011, "triple_stack_enabled": True,
}
# Phase 4 honest stack uses the same SP but with rr_floor=0 (no structural-target gate).
SP_PHASE4 = {**SP_BASELINE, "rr_floor": 0.0, "sector_limit": 5}
EXIT_OV = {"exit_ma_period": 27, "exit_atr_buffer": 0.75}
TRISTACK_OV = {
    "stall_tighten_week": 12, "stall_tighten_threshold": 0.07, "stall_tighten_ma": "sma_20",
    "extension_atr_mult": 3.5, "extension_tighten_ma": "sma_15",
    "climactic_vol_threshold": 2.0, "climactic_tighten_ma": "sma_15",
}


def build_baseline_cfg(start, end, force_close=False):
    cfg = auto.build_engine_b_config(SP_BASELINE)
    cfg["ma_period"] = 25
    cfg.update(EXIT_OV)
    cfg.update(TRISTACK_OV)
    cfg["triple_stack_enabled"] = True
    cfg["start_date"] = start
    cfg["end_date"] = end
    cfg["force_close_at_end"] = force_close
    cfg["starting_capital"] = 500000
    cfg["is_split_date"] = "2099-01-01"
    return cfg


def build_phase4_cfg(start, end, with_time_stop=False, force_close=False,
                     ranking_method="type_prior", cost_model="ZERODHA"):
    cfg = auto.build_engine_b_config(SP_PHASE4)
    cfg["ma_period"] = 25
    cfg.update(EXIT_OV)
    cfg.update(TRISTACK_OV)
    cfg["triple_stack_enabled"] = True
    cfg["start_date"] = start
    cfg["end_date"] = end
    cfg["force_close_at_end"] = force_close
    cfg["starting_capital"] = 500000
    cfg["is_split_date"] = "2099-01-01"
    # Phase 4
    cfg["regime_combination"] = "OVERALL_OR_SECTOR"
    cfg["exit_gtt_field"] = "close"
    cfg["sector_rs_filter"] = "TOP"
    cfg["regime_type"] = "breadth_5state"
    cfg["entry_filter"] = "aggressive"
    cfg["overall_breadth_threshold"] = 60
    cfg["ranking_method"] = ranking_method
    cfg["sector_limit"] = 5
    cfg["cost_model"] = cost_model
    cfg["stop_method"] = "CURRENT"
    cfg["volume_min_ratio"] = None
    cfg["pyramid_enabled"] = False
    cfg["partial_profit_enabled"] = True
    cfg["partial_profit_trigger_pct"] = 30.0
    cfg["partial_profit_sell_pct"] = 50.0
    cfg["partial_profit_move_stop"] = "HALF_BACK"
    cfg["trail_tighten_enabled"] = True
    cfg["trail_tighten_trigger_pct"] = 50.0
    cfg["trail_tighten_ma_type"] = "SMA"
    cfg["trail_tighten_ma_period"] = 20
    cfg["trail_tighten_atr_buffer"] = 0.75
    if with_time_stop:
        cfg["time_stop_enabled"] = True
        cfg["time_stop_weeks"] = 4
        cfg["time_stop_min_return_pct"] = 5.0
        cfg["time_stop_action"] = "RANGE_LOW_025"
    else:
        cfg["time_stop_enabled"] = False
    return cfg


def load_lagged_entries():
    """Load lagged entries (look-ahead-fixed) and apply Phase 4 filter (rr_floor=0)."""
    raw = pd.read_csv(V2 / "entries_all_lagged.csv", low_memory=False)
    raw["week_date"] = pd.to_datetime(raw["week_date"])
    chunks = []
    for et, (mt, mp) in OPTIMAL_MAS.items():
        chunks.append(raw[(raw["entry_type"] == et) & (raw["ma_type"] == mt) & (raw["ma_period"] == mp)])
    combined = pd.concat(chunks, ignore_index=True).sort_values(["week_date", "ticker"]).reset_index(drop=True)
    filt = auto.filter_entries(combined, SP_PHASE4)
    filt = filt.copy()
    if "entry_price" not in filt.columns:
        filt["entry_price"] = filt["entry_price_close"]
    return filt


def compute_curve_metrics(equity_curves, total_trades):
    """Compute MAR/CAGR/MaxDD from a stitched equity curve list."""
    if not equity_curves:
        return {"cagr": 0, "maxdd": 0, "mar": 0}
    start_eq = equity_curves[0]["total_equity"]
    end_eq = equity_curves[-1]["total_equity"]
    weeks = len(equity_curves)
    years = weeks / 52.0
    if start_eq > 0 and years > 0:
        cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    else:
        cagr = 0
    peak = start_eq
    max_dd = 0
    for snap in equity_curves:
        v = snap["total_equity"]
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
    mar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"cagr": round(cagr, 2), "maxdd": round(max_dd, 2), "mar": round(mar, 2),
            "start_eq": round(start_eq), "end_eq": round(end_eq), "total_trades": total_trades}


def run_continuous(config_name, config_builder, engine, filt_entries,
                   regime_df, nifty_df, cache, extra_kwargs=None,
                   type_prior_mode="none", static_type_prior=None):
    """Run continuous walk-forward with carry-over for one config.

    type_prior_mode:
      - "none"     : leave entries untouched (engine ranks on cfg["ranking_method"])
      - "static"   : map entry_type → static_type_prior dict once
      - "dynamic"  : per-segment, recompute type_prior from trade_stats with cutoff = seg.apply_from
    """
    positions = []
    cash = 500_000.0
    equity_curves = []
    seg_results = []
    total_trades = 0

    for seg in SEGMENTS:
        start = seg["apply_from"]
        end = seg["apply_until"]
        is_last = seg["id"] == len(SEGMENTS) - 1

        # Filter entries to segment window
        window_entries = filt_entries[
            (filt_entries["week_date"] >= start) & (filt_entries["week_date"] <= end)
        ].copy()

        # Attach type_prior per CF1 spec
        if type_prior_mode == "dynamic":
            type_scores = compute_type_prior(TRADE_STATS_PATH, start)
            window_entries["type_prior"] = window_entries["entry_type"].map(type_scores).fillna(DEFAULT_PRIOR)
        elif type_prior_mode == "static" and static_type_prior is not None:
            window_entries["type_prior"] = window_entries["entry_type"].map(static_type_prior).fillna(DEFAULT_PRIOR)

        cfg = config_builder(start, end, force_close=False)
        cfg["config_id"] = f"seg_{seg['id']}"

        # Run with carry-over
        kwargs = dict(
            cfg=cfg, quiet=True, entries_df=window_entries,
            regime_df=regime_df, nifty_df=nifty_df, stock_cache=cache,
            initial_positions=positions, initial_cash=cash,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        result = engine.run_simulation(**kwargs)
        eq = result["equity_curve"]

        # Stitch equity curves
        if equity_curves and eq:
            prev_final = equity_curves[-1]["total_equity"]
            next_start = eq[0]["total_equity"]
            gap_pct = abs(next_start - prev_final) / prev_final * 100 if prev_final > 0 else 0

            last_date = equity_curves[-1]["date"]
            first_date = eq[0]["date"]
            if last_date == first_date:
                equity_curves.extend(eq[1:])
            else:
                equity_curves.extend(eq)

            if gap_pct > 0.1:
                print(f"    Boundary seg{seg['id']-1}→{seg['id']}: gap={gap_pct:.4f}%")
        else:
            equity_curves.extend(eq)

        # Carry over
        positions = result["open_positions"]
        cash = eq[-1]["cash"] if eq else cash

        # Segment metrics
        seg_trades = len(result["trades"])
        total_trades += seg_trades

        if eq:
            seg_start_eq = eq[0]["total_equity"]
            seg_end_eq = eq[-1]["total_equity"]
            seg_weeks = len(eq)
            seg_years = seg_weeks / 52.0
            seg_cagr = ((seg_end_eq / seg_start_eq) ** (1 / seg_years) - 1) * 100 if seg_start_eq > 0 and seg_years > 0 else 0
            seg_peak = seg_start_eq
            seg_dd = 0
            for snap in eq:
                v = snap["total_equity"]
                if v > seg_peak:
                    seg_peak = v
                dd = (v - seg_peak) / seg_peak * 100 if seg_peak > 0 else 0
                if dd < seg_dd:
                    seg_dd = dd
            seg_mar = seg_cagr / abs(seg_dd) if seg_dd != 0 else 0
        else:
            seg_cagr = seg_dd = seg_mar = 0

        wr = 0
        if seg_trades > 0:
            wr = sum(1 for t in result["trades"] if t["return_pct"] > 0) / seg_trades * 100

        seg_results.append({
            "seg": seg["id"], "period": f"{start[:4]}-{end[:4]}",
            "trades": seg_trades, "cagr": round(seg_cagr, 1), "maxdd": round(seg_dd, 1),
            "mar": round(seg_mar, 2), "wr": round(wr, 1),
            "open_carry": len(positions),
        })

        print(f"    Seg{seg['id']} ({start[:4]}-{end[:4]}): "
              f"MAR={seg_mar:5.2f} CAGR={seg_cagr:5.1f}% MDD={seg_dd:5.1f}% "
              f"trades={seg_trades:3d} carry={len(positions)}")

    full = compute_curve_metrics(equity_curves, total_trades)
    return full, seg_results, equity_curves


def main():
    t_start = time.time()
    print("=" * 110)
    print("PHASE 4 CONTINUOUS WALK-FORWARD (carry-over)")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"3 configs × 6 segments, 2010-2026")
    print("=" * 110)

    print("\n=== LOADING DATA ===")
    all_entries = auto.load_all_entries()
    filt_baseline = auto.filter_entries(all_entries, SP_BASELINE)
    if "entry_price" not in filt_baseline.columns:
        filt_baseline = filt_baseline.copy()
        filt_baseline["entry_price"] = filt_baseline["entry_price_close"]
    print(f"  Baseline entries (unlagged, rr_floor=0.5): {len(filt_baseline):,}")

    filt_lagged = load_lagged_entries()
    print(f"  Phase4 entries  (lagged,  rr_floor=0.0): {len(filt_lagged):,}")

    regime_df = pd.read_csv(V2 / "regime_weekly.csv", parse_dates=["week_date"]).set_index("week_date").sort_index()
    nifty_locked = engine_locked.load_nifty()
    nifty_exp = engine_exp.load_nifty()
    cache_locked = engine_locked.StockCacheManager(V2 / "stock_weekly_cache", extra_sma_periods=[15, 27])
    cache_exp = engine_exp.StockCacheManager(V2 / "stock_weekly_cache", extra_sma_periods=[15, 27])

    sr = engine_exp.load_sector_regime(V2 / "sector_regime_weekly.csv")
    cr = engine_exp.load_captier_regime(V2 / "captier_regime_weekly.csv")
    tm = engine_exp.load_ticker_metadata(V2 / "ticker_metadata.csv")
    rs_q = engine_exp.load_sector_rs_quartile(V2 / "sector_regime_weekly.csv")
    print("  All data loaded.")

    exp_kwargs = dict(
        sector_regime_lookup=sr, captier_regime_lookup=cr,
        ticker_metadata_lookup=tm, sector_rs_lookup=rs_q,
    )

    # ── Config 1: Baseline ───────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("CONFIG 1: BASELINE (locked engine, Tier 2 winner)")
    print("=" * 110)
    full1, segs1, eq1 = run_continuous(
        "Baseline", lambda s, e, force_close=False: build_baseline_cfg(s, e, force_close),
        engine_locked, filt_baseline, regime_df, nifty_locked, cache_locked,
    )
    print(f"\n  FULL: MAR={full1['mar']:.2f} CAGR={full1['cagr']:.1f}% MDD={full1['maxdd']:.1f}% trades={full1['total_trades']}")

    if abs(full1["mar"] - 1.32) > 0.3:
        print(f"  *** WARNING: Expected ~MAR 1.32, got {full1['mar']:.2f}")
        print("  Proceeding but noting the discrepancy.")

    # ── Config 2: Phase4 Best (lagged entries, dynamic type_prior) ───
    print(f"\n{'=' * 110}")
    print("CONFIG 2: PHASE4 BEST (PP + TT, dynamic type_prior, no time stop)")
    print("=" * 110)
    full2, segs2, eq2 = run_continuous(
        "Phase4 Best",
        lambda s, e, force_close=False: build_phase4_cfg(s, e, with_time_stop=False, force_close=force_close),
        engine_exp, filt_lagged, regime_df, nifty_exp, cache_exp, exp_kwargs,
        type_prior_mode="dynamic",
    )
    print(f"\n  FULL: MAR={full2['mar']:.2f} CAGR={full2['cagr']:.1f}% MDD={full2['maxdd']:.1f}% trades={full2['total_trades']}")

    # ── Config 3: Phase4 + Time Stop ─────────────────────────────────
    print(f"\n{'=' * 110}")
    print("CONFIG 3: PHASE4 + STRUCTURAL TIME STOP (dynamic type_prior)")
    print("=" * 110)
    full3, segs3, eq3 = run_continuous(
        "Phase4+TS",
        lambda s, e, force_close=False: build_phase4_cfg(s, e, with_time_stop=True, force_close=force_close),
        engine_exp, filt_lagged, regime_df, nifty_exp, cache_exp, exp_kwargs,
        type_prior_mode="dynamic",
    )
    print(f"\n  FULL: MAR={full3['mar']:.2f} CAGR={full3['cagr']:.1f}% MDD={full3['maxdd']:.1f}% trades={full3['total_trades']}")

    # ══════════════════════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print("PER-SEGMENT COMPARISON")
    print("=" * 110)

    print(f"\n  {'Seg':>4s} {'Period':>10s}  │ {'MAR':>5s} {'CAGR':>6s} {'MDD':>6s} {'Trd':>4s}  │ "
          f"{'MAR':>5s} {'CAGR':>6s} {'MDD':>6s} {'Trd':>4s}  │ {'MAR':>5s} {'CAGR':>6s} {'MDD':>6s} {'Trd':>4s}")
    print(f"  {'':>4s} {'':>10s}  │ {'--- Baseline ---':^25s}  │ {'--- Phase4 Best ---':^25s}  │ {'--- Phase4 + TS ---':^25s}")
    print("  " + "-" * 100)

    for i in range(len(SEGMENTS)):
        s1, s2, s3 = segs1[i], segs2[i], segs3[i]
        print(f"  {s1['seg']:>4d} {s1['period']:>10s}  │ "
              f"{s1['mar']:5.2f} {s1['cagr']:5.1f}% {s1['maxdd']:5.1f}% {s1['trades']:4d}  │ "
              f"{s2['mar']:5.2f} {s2['cagr']:5.1f}% {s2['maxdd']:5.1f}% {s2['trades']:4d}  │ "
              f"{s3['mar']:5.2f} {s3['cagr']:5.1f}% {s3['maxdd']:5.1f}% {s3['trades']:4d}")

    # ── Full continuous metrics ───────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("FULL CONTINUOUS WALK-FORWARD METRICS (2010-2026)")
    print("=" * 110)

    print(f"\n  {'Metric':20s}  {'Baseline':>12s}  {'Phase4 Best':>12s}  {'Phase4 + TS':>12s}")
    print("  " + "-" * 60)
    for label, k in [("CAGR", "cagr"), ("MaxDD", "maxdd"), ("MAR", "mar"),
                      ("Total Trades", "total_trades"), ("Start Equity", "start_eq"), ("End Equity", "end_eq")]:
        v1 = full1[k]
        v2 = full2[k]
        v3 = full3[k]
        if k in ("cagr", "maxdd"):
            print(f"  {label:20s}  {v1:>11.1f}%  {v2:>11.1f}%  {v3:>11.1f}%")
        elif k == "mar":
            print(f"  {label:20s}  {v1:>12.2f}  {v2:>12.2f}  {v3:>12.2f}")
        else:
            print(f"  {label:20s}  {v1:>12,}  {v2:>12,}  {v3:>12,}")

    # ── PASS/FAIL ────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("PASS/FAIL ASSESSMENT")
    print("=" * 110)

    for label, segs, full in [("Phase4 Best", segs2, full2), ("Phase4 + TS", segs3, full3)]:
        neg_cagr = sum(1 for s in segs if s["cagr"] < 0)
        deep_neg = sum(1 for s in segs if s["cagr"] < -5)
        bad_dd = sum(1 for s in segs if s["maxdd"] < -30)
        deep_dd = sum(1 for s in segs if s["maxdd"] < -25)
        mar_above_05 = sum(1 for s in segs if s["mar"] > 0.5)
        low_trades = sum(1 for s in segs if 0 < s["trades"] < 15)

        passed = True
        if deep_neg >= 2:
            passed = False
        if deep_dd >= 2:
            passed = False
        if mar_above_05 < 4:
            passed = False

        print(f"\n  {label}:")
        print(f"    Negative CAGR segments: {neg_cagr}/6 (deep < -5%: {deep_neg})")
        print(f"    DD > -30%: {bad_dd}/6 (deep > -25%: {deep_dd})")
        print(f"    MAR > 0.5: {mar_above_05}/6 (need ≥ 4)")
        print(f"    Low trade segments: {low_trades}/6")
        print(f"    Continuous MAR: {full['mar']:.2f}")
        print(f"    Verdict: {'*** PASS ***' if passed else '*** FAIL ***'}")

    # ── Per-segment improvement ──────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("PER-SEGMENT IMPROVEMENT vs BASELINE")
    print("=" * 110)

    for label, segs in [("Phase4 Best", segs2), ("Phase4 + TS", segs3)]:
        improved = degraded = 0
        print(f"\n  {label}:")
        print(f"  {'Seg':>4s} {'Period':>10s}  {'Base MAR':>8s}  {'New MAR':>7s}  {'Delta':>7s}")
        print("  " + "-" * 42)
        for i in range(len(SEGMENTS)):
            b_mar = segs1[i]["mar"]
            n_mar = segs[i]["mar"]
            delta = n_mar - b_mar
            if delta > 0.05:
                improved += 1
            elif delta < -0.05:
                degraded += 1
            print(f"  {segs1[i]['seg']:>4d} {segs1[i]['period']:>10s}  {b_mar:>8.2f}  {n_mar:>7.2f}  {delta:>+6.2f}")
        print(f"  Improved: {improved}, Degraded: {degraded}")

    # ── The honest number ────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("THE HONEST CONTINUOUS WALK-FORWARD MAR")
    print("=" * 110)
    print(f"\n  Baseline (locked, Tier 2 winner): MAR = {full1['mar']:.2f}")
    print(f"  Phase4 Best (PP + TT):            MAR = {full2['mar']:.2f}")
    print(f"  Phase4 Best + Time Stop:          MAR = {full3['mar']:.2f}")

    total_time = time.time() - t_start
    print(f"\n{'=' * 110}")
    print(f"DONE. Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'=' * 110}")


# ── Public reproduction entry point ───────────────────────────────────
# Published targets for each canonical config, and the tolerances within
# which an independent run is considered an exact reproduction.
EXPECTED = {
    "phase4_best": {"mar": 1.96, "cagr": 24.1, "maxdd": -12.3, "trades": 604},
    "phase4_time_stop": {"mar": None, "cagr": None, "maxdd": None, "trades": None},
    "baseline": {"mar": 1.32, "cagr": None, "maxdd": None, "trades": None},
}
TOLERANCES = {"mar": 0.01, "cagr": 0.2, "maxdd": 0.2, "trades": 3}


def _load_inputs():
    """Load every input the experimental walk-forward needs from the bundle."""
    regime_df = (
        pd.read_csv(V2 / "regime_weekly.csv", parse_dates=["week_date"])
        .set_index("week_date")
        .sort_index()
    )
    nifty_exp = engine_exp.load_nifty()
    cache_exp = engine_exp.StockCacheManager(V2 / "stock_weekly_cache", extra_sma_periods=[15, 27])
    exp_kwargs = dict(
        sector_regime_lookup=engine_exp.load_sector_regime(V2 / "sector_regime_weekly.csv"),
        captier_regime_lookup=engine_exp.load_captier_regime(V2 / "captier_regime_weekly.csv"),
        ticker_metadata_lookup=engine_exp.load_ticker_metadata(V2 / "ticker_metadata.csv"),
        sector_rs_lookup=engine_exp.load_sector_rs_quartile(V2 / "sector_regime_weekly.csv"),
    )
    return regime_df, nifty_exp, cache_exp, exp_kwargs


def reproduce(config: str = "phase4_best", verbose: bool = True) -> dict:
    """Run a canonical Phase-4 continuous walk-forward and return its metrics.

    Args:
        config: one of
          - ``"phase4_best"``      → the published headline
            (MAR 1.96 / CAGR 24.1% / MaxDD -12.3% / 604 trades)
          - ``"phase4_time_stop"`` → Phase 4 + structural time stop
          - ``"baseline"``         → Tier-2 locked baseline (MAR ~1.32)
        verbose: print a formatted achieved-vs-expected summary table.

    Returns:
        dict with keys ``config``, ``metrics`` (full MAR/CAGR/MaxDD/trades/
        equity), ``per_segment``, ``expected``, ``deltas``, and ``passed``
        (``True``/``False`` against tolerances, or ``None`` if no published
        target exists for the config).

    The data bundle directory must be set before calling this — either via
    ``skysurf.reproduction.reproduce(data_dir=...)`` or the
    ``SKYSURF_REPRO_DATA`` environment variable.
    """
    if config not in EXPECTED:
        raise ValueError(f"unknown config {config!r}; choose from {sorted(EXPECTED)}")

    regime_df, nifty_exp, cache_exp, exp_kwargs = _load_inputs()

    if config == "baseline":
        filt = auto.filter_entries(auto.load_all_entries(), SP_BASELINE)
        if "entry_price" not in filt.columns:
            filt = filt.copy()
            filt["entry_price"] = filt["entry_price_close"]
        full, segs, _eq = run_continuous(
            "Baseline",
            lambda s, e, force_close=False: build_baseline_cfg(s, e, force_close),
            engine_locked,
            filt,
            regime_df,
            engine_locked.load_nifty(),
            engine_locked.StockCacheManager(V2 / "stock_weekly_cache", extra_sma_periods=[15, 27]),
        )
    else:
        with_ts = config == "phase4_time_stop"
        filt = load_lagged_entries()
        full, segs, _eq = run_continuous(
            "Phase4+TS" if with_ts else "Phase4 Best",
            lambda s, e, force_close=False: build_phase4_cfg(
                s, e, with_time_stop=with_ts, force_close=force_close
            ),
            engine_exp,
            filt,
            regime_df,
            nifty_exp,
            cache_exp,
            exp_kwargs,
            type_prior_mode="dynamic",
        )

    achieved = {
        "mar": full["mar"],
        "cagr": full["cagr"],
        "maxdd": full["maxdd"],
        "trades": full["total_trades"],
    }
    expected = EXPECTED[config]
    deltas = {
        k: (achieved[k] - expected[k]) if expected[k] is not None else None for k in achieved
    }
    passed = None
    if all(expected[k] is not None for k in achieved):
        passed = all(abs(deltas[k]) <= TOLERANCES[k] for k in achieved)

    if verbose:
        _print_reproduction_summary(config, achieved, expected, deltas, passed, full)

    return {
        "config": config,
        "metrics": full,
        "per_segment": segs,
        "expected": expected,
        "deltas": deltas,
        "passed": passed,
    }


def _print_reproduction_summary(config, achieved, expected, deltas, passed, full):
    """Pretty-print the achieved-vs-expected reproduction table."""
    print("=" * 72)
    print(f"SKYSURF REPRODUCTION — {config}")
    print("=" * 72)
    print(f"  {'Metric':14s} {'Achieved':>12s} {'Expected':>12s} {'Delta':>10s}")
    print("  " + "-" * 50)
    rows = [
        ("mar", "MAR", "{:.2f}"),
        ("cagr", "CAGR %", "{:.2f}"),
        ("maxdd", "MaxDD %", "{:.2f}"),
        ("trades", "Trades", "{:d}"),
    ]
    for k, label, fmt in rows:
        a_s = fmt.format(achieved[k])
        e_s = fmt.format(expected[k]) if expected[k] is not None else "—"
        if deltas[k] is None:
            d_s = "—"
        elif k == "trades":
            d_s = f"{deltas[k]:+d}"
        else:
            d_s = f"{deltas[k]:+.2f}"
        print(f"  {label:14s} {a_s:>12s} {e_s:>12s} {d_s:>10s}")
    print(f"\n  Start equity: Rs {full['start_eq']:,}    End equity: Rs {full['end_eq']:,}")
    if passed is True:
        print("\n  RESULT: PASS — reproduces the published number within tolerance.")
    elif passed is False:
        print("\n  RESULT: FAIL — outside tolerance. Check the data bundle version.")
    print("=" * 72)


if __name__ == "__main__":
    main()
