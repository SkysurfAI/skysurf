# Reproducibility

This page is the honest answer to "can I reproduce the headline 24.13% CAGR myself?".

## What you can reproduce

Given your own historical OHLCV data (which you must source separately — see below), you can:

* **Run `generate_weekly_signals` for any historical week** and see what the strategy would have signalled.
* **Run `manage_positions` on a pretend held-position list** and see what stops, partials, or exits the strategy would have called.
* **Compute the exact same trade entries, exits, sizes, and stops** as our internal validation, on your data.

These are deterministic. Given the same inputs, the brain returns byte-identical outputs. That's enforced by the contract test suite.

## What you cannot reproduce from this repo alone

The walk-forward harness — the wrapper that ran the brain over 16 years of weekly data, stitched segments, and computed MAR / CAGR / max-drawdown — is **not** part of this open-source release. It's research code, not production code, and we kept it private.

So you cannot from a fresh clone:

* Reproduce the 24.13% CAGR / 1.96 MAR / −12.31% max-drawdown / 604-trade headline numbers.
* Run a continuous walk-forward backtest of arbitrary date ranges.
* Run the CF1 (dynamic type-prior) or CF2 (Zerodha cost) comparison passes.

If you need an end-to-end backtest, you have two paths:

1. **Build your own walk-forward harness.** It's not complicated — loop over weeks, call `generate_weekly_signals`, simulate fills, call `manage_positions`, simulate stops. Maybe 200 lines of Python. We may publish a reference implementation in a future release.
2. **Trust the published number** and run forward-test on paper for a quarter or two before risking capital. This is what we recommend even after a successful backtest reproduction.

## Why this split

Two reasons.

**Operational moat.** Skysurf the managed service charges for running the strategy live: data pipelines, broker integration, GTT order plumbing, monitoring, compliance. Those are the hard parts and they're closed-source. The strategy logic — the brain — is genuinely the easy part to reproduce given OHLCV data, so we made it open. Withholding the research harness slows down anyone who wants to run a competing managed service without changing the experience for researchers.

**Research vs production.** The walk-forward harness was tightly coupled to one specific dataset shape (the keen-hellman research worktree's CSV cache, with point-in-time mappings, segment boundaries, etc.). Open-sourcing it would either ship a thousand lines of research-specific code that nobody can run without our data, or require a substantial generalisation pass. Neither is a good use of time relative to publishing the production brain.

## Data caveats — read these

Several caveats apply to any backtest you run, regardless of the harness:

* **Survivorship bias.** Universe selection (Nifty 500 / market-cap floor) typically uses the *current* membership applied historically. This inflates strategy CAGR by an estimated 2–3 pp. Both Skysurf's published 24.13% and any number you reproduce on your own data will share this bias unless you're using point-in-time membership rosters.
* **Static sector and cap-tier mapping.** Sector and market-cap-tier mappings are typically a current snapshot applied historically. The Skysurf research dataset uses 2026 mappings throughout 2010–2026. This is a known limitation.
* **No dividends in the price index.** OHLCV used by the strategy is split-adjusted but not dividend-adjusted, by design — the strategy doesn't capture dividends, and the Nifty benchmark you compare against shouldn't either. Use the price index (not the TRI) for an apples-to-apples comparison.
* **Costs.** PHASE_4_V1 includes the Zerodha equity-delivery cost model (regulatory + slippage + DP charge). If your broker is more or less expensive, set `cost_model="FLAT"` and tune `slippage_pct` / `brokerage_pct` accordingly. CAGR will move ~0.4 pp per 0.1% per-side cost change.
* **Slippage.** Default slippage of 0.1% per side is reasonable for liquid stocks but optimistic for thinly-traded names. Filter your universe by ADTV before running production.

## Forward-test before going live

Even with a successful backtest reproduction on your own data, run forward on paper for a quarter or two before risking real capital. Backtests find fits that don't generalise; forward-tests don't. The validated 24.13% number includes a known small look-ahead leak (research call CF1, documented in `docs/strategy.md`) plus the survivorship bias mentioned above. Our internal estimate is that forward returns will be 2–4 pp below backtest, on average.

If your forward results are within a few pp of expectations, the strategy is working as designed. If they're radically different in either direction, that's a signal worth investigating.
