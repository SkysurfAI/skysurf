#!/usr/bin/env python3
"""Rebuild entries with look-ahead-free target_level and rr_ratio.

The original entries_all.csv uses argrelextrema(order=8) which confirms
swing highs using 8 future bars. This script recomputes target_level
using only swing highs that are FULLY confirmed before entry time.

Target = nearest confirmed swing high ABOVE entry price (overhead resistance).
If none exists, fallback to entry × 1.20.

Output: val_engine_a_v2_results/entries_all_lagged.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from skysurf.reproduction import _paths

# Reads entries_all.csv + the weekly cache from the data bundle, writes
# entries_all_lagged.csv next to them.
V2 = _paths.data_dir()

SWING_ORDER = 8  # same as strategy uses


def compute_lagged_target(weekly_highs, weekly_close, entry_idx, entry_close, stop_level, swing_order=SWING_ORDER):
    """Compute target and rr_ratio using only confirmed-before-entry swing highs.

    A swing high at index S is confirmed when S + swing_order <= current index.
    This means at entry_idx E, only swings where S + swing_order <= E are known.

    Target = nearest confirmed swing high ABOVE entry_close (overhead resistance).
    """
    # Find all swing highs on the full series
    if len(weekly_highs) < swing_order * 2 + 1:
        return None, None

    raw_highs = argrelextrema(weekly_highs, np.greater, order=swing_order)[0]

    # Filter to only those confirmed before entry
    # A swing at index S is confirmed at S + swing_order
    confirmed = [s for s in raw_highs if s + swing_order <= entry_idx]

    # Find nearest confirmed swing high ABOVE entry price
    target = None
    best_dist = float("inf")
    for s in confirmed:
        level = float(weekly_highs[s])
        if level > entry_close:
            dist = level - entry_close
            if dist < best_dist:
                best_dist = dist
                target = level

    if target is None:
        target = entry_close * 1.20  # fallback

    # Compute rr_ratio
    risk = entry_close - stop_level
    reward = target - entry_close
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    return round(target, 2), rr


def main():
    t0 = time.time()
    # Re-resolve from the current data dir (robust to import order).
    global V2
    V2 = _paths.data_dir()
    print("=" * 90)
    print("REBUILD ENTRIES — Look-Ahead-Free Targets")
    print("=" * 90)

    # Load entries
    entries = pd.read_csv(V2 / "entries_all.csv", low_memory=False)
    entries["week_date"] = pd.to_datetime(entries["week_date"])
    print(f"\n  Loaded {len(entries):,} entries")

    # Load stock weekly caches
    print("  Loading stock caches...")
    cache_dir = V2 / "stock_weekly_cache"
    stock_data = {}
    tickers = entries["ticker"].unique()
    for ticker in tickers:
        path = cache_dir / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            stock_data[ticker] = df
    print(f"  Loaded {len(stock_data)}/{len(tickers)} stock caches")

    # Recompute target_level and rr_ratio
    print("  Recomputing targets (lagged)...")
    new_targets = np.full(len(entries), np.nan)
    new_rr = np.full(len(entries), np.nan)
    orig_rr = entries["rr_ratio"].values.copy()

    processed = 0
    changed = 0
    fallback_count = 0

    for i, (_, row) in enumerate(entries.iterrows()):
        ticker = row["ticker"]
        wk = row["week_date"]
        close = row["entry_price_close"]
        stop = row["stop_level"]

        sdf = stock_data.get(ticker)
        if sdf is None or wk not in sdf.index:
            continue

        entry_idx = sdf.index.get_loc(wk)
        highs = sdf["High"].values

        target, rr = compute_lagged_target(highs, sdf["Close"].values, entry_idx, close, stop)
        if target is not None:
            new_targets[i] = target
            new_rr[i] = rr
            processed += 1
            if pd.notna(orig_rr[i]) and abs(rr - orig_rr[i]) > 0.01:
                changed += 1
            if abs(target - close * 1.20) < 0.01:
                fallback_count += 1

        if (i + 1) % 100000 == 0:
            print(f"    {i+1:,}/{len(entries):,} processed...")

    entries["target_level_original"] = entries["target_level"]
    entries["rr_ratio_original"] = entries["rr_ratio"]
    entries["target_level"] = new_targets
    entries["rr_ratio"] = new_rr

    print(f"\n  Processed: {processed:,}")
    print(f"  Changed rr_ratio: {changed:,}")
    print(f"  Fallback (no overhead resistance): {fallback_count:,}")

    # Save
    out_path = V2 / "entries_all_lagged.csv"
    entries.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("ANALYSIS")
    print("=" * 90)

    # RR ratio coverage
    orig_has_rr = entries["rr_ratio_original"].notna().sum()
    new_has_rr = entries["rr_ratio"].notna().sum()
    print(f"\n  RR ratio coverage:")
    print(f"    Original: {orig_has_rr:,}/{len(entries):,} ({orig_has_rr/len(entries)*100:.1f}%)")
    print(f"    Lagged:   {new_has_rr:,}/{len(entries):,} ({new_has_rr/len(entries)*100:.1f}%)")

    # Correlation
    both = entries[entries["rr_ratio_original"].notna() & entries["rr_ratio"].notna()]
    if len(both) > 100:
        corr = both["rr_ratio_original"].corr(both["rr_ratio"])
        print(f"\n  Correlation (original vs lagged, where both exist): {corr:.3f}")

    # Distribution comparison
    print(f"\n  RR ratio distribution (entries that have both):")
    print(f"    {'':15s}  {'Original':>10s}  {'Lagged':>10s}")
    print(f"    {'Mean':15s}  {both['rr_ratio_original'].mean():>10.2f}  {both['rr_ratio'].mean():>10.2f}")
    print(f"    {'Median':15s}  {both['rr_ratio_original'].median():>10.2f}  {both['rr_ratio'].median():>10.2f}")
    print(f"    {'Std':15s}  {both['rr_ratio_original'].std():>10.2f}  {both['rr_ratio'].std():>10.2f}")

    # How many entries had their rr_ratio INCREASE vs DECREASE?
    both_c = both.copy()
    both_c["delta"] = both_c["rr_ratio"] - both_c["rr_ratio_original"]
    increased = (both_c["delta"] > 0.1).sum()
    decreased = (both_c["delta"] < -0.1).sum()
    unchanged = len(both_c) - increased - decreased
    print(f"\n  RR ratio changes (>0.1 threshold):")
    print(f"    Increased: {increased:,}")
    print(f"    Decreased: {decreased:,}")
    print(f"    Unchanged: {unchanged:,}")

    # 5 example entries showing original vs lagged
    print(f"\n  5 example entries (largest change):")
    examples = both_c.nlargest(5, "delta", keep="first") if (both_c["delta"] > 0).any() else both_c.head(5)
    worst = both_c.nsmallest(3, "delta")
    examples = pd.concat([worst, examples.head(2)])
    print(f"  {'Ticker':18s} {'Date':12s} {'Close':>7s} {'OrigTarget':>10s} {'LagTarget':>10s} {'OrigRR':>7s} {'LagRR':>7s} {'Delta':>7s}")
    for _, r in examples.iterrows():
        print(f"  {r['ticker']:18s} {str(r['week_date'])[:10]:12s} {r['entry_price_close']:>7.1f} "
              f"{r['target_level_original']:>10.1f} {r['target_level']:>10.1f} "
              f"{r['rr_ratio_original']:>7.2f} {r['rr_ratio']:>7.2f} {r['delta']:>+6.2f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"Done. {elapsed:.1f}s")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
