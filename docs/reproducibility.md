# Reproducibility

This page is the honest answer to "can I reproduce the headline 24.1% CAGR myself?".

**Short answer: yes.** As of v0.3.0 the walk-forward harness ships in this
package (`skysurf.reproduction`). Bring the data bundle and you reproduce the
published numbers bit-for-bit:

| Config        | MAR  | CAGR  | MaxDD   | Trades |
|---------------|------|-------|---------|--------|
| Phase4 Best   | 1.96 | 24.1% | −12.3%  | 604    |

## TL;DR

```bash
pip install skysurf                       # harness ships in the base install
# download the reproduction data bundle (~450 MB), then:
skysurf-reproduce --data /path/to/skysurf-repro-data
```

or from Python:

```python
import skysurf.reproduction as repro

result = repro.reproduce(data_dir="/path/to/skysurf-repro-data")
assert result["passed"]                   # within published tolerance
print(result["metrics"]["mar"])           # 1.96
```

The data bundle is distributed separately from the code (it's ~450 MB of
pre-computed weekly caches and trade statistics). See
[`reproduction-data-bundle.md`](./reproduction-data-bundle.md) for its exact
layout, schema, and how to obtain or rebuild it.

## What you can reproduce

* **The exact headline metrics** — MAR, CAGR, max-drawdown, and trade count of
  the continuous 2010–2026 walk-forward, via `skysurf.reproduction`.
* **`generate_weekly_signals` for any historical week** — the strategy's signal
  for that week on your data.
* **`manage_positions` on a held-position list** — the stops, partials, and
  exits the strategy would call.
* **The exact trade entries, exits, sizes, and stops** of our internal
  validation, on the bundled data.

These are deterministic. Given identical inputs, the engine returns
byte-identical outputs — enforced by the contract test suite, and (for the full
backtest) by the `phase4_best` tolerance check inside `reproduce()`.

## How the reproduction is wired

`skysurf.reproduction` is the **unmodified research engine** — the same
walk-forward driver, portfolio simulator, and dynamic type-prior that produced
the published numbers. The only change from the research worktree is that all
data paths resolve from a single flat *data bundle directory* instead of the
original scattered result folders. The simulation logic is byte-identical, which
is what makes the reproduction exact rather than approximate.

Configs available to `reproduce(config=...)`:

* `"phase4_best"` — the published headline (default).
* `"phase4_time_stop"` — Phase 4 plus a structural time stop.
* `"baseline"` — the Tier-2 locked baseline (MAR ≈ 1.32).

Charts (equity curve, drawdown, MAE/MFE distributions) are optional. Install
`skysurf[reproduction]` to enable matplotlib output; the numbers themselves run
on the base install.

## Data caveats — read these

Several caveats apply to the backtest regardless of who runs it:

* **Survivorship bias.** Universe selection (Nifty 500 / market-cap floor) uses
  *current* membership applied historically. This inflates strategy CAGR by an
  estimated 2–3 pp. The published 24.1% shares this bias unless point-in-time
  membership rosters are used.
* **Static sector and cap-tier mapping.** Sector and market-cap-tier mappings
  are a current snapshot applied historically (2026 mappings throughout
  2010–2026). A known limitation.
* **No dividends in the price index.** OHLCV is split-adjusted but not
  dividend-adjusted, by design — the strategy doesn't capture dividends, and the
  Nifty benchmark shouldn't either. Compare against the price index, not the TRI.
* **Costs.** Phase 4 includes the Zerodha equity-delivery cost model (regulatory
  + slippage + DP charge). For a different broker, set `cost_model="FLAT"` and
  tune `slippage_pct` / `brokerage_pct`. CAGR moves ~0.4 pp per 0.1% per-side
  cost change.
* **Slippage.** The default 0.1% per side is reasonable for liquid stocks but
  optimistic for thin names. Filter your universe by ADTV before going live.

## Forward-test before going live

Even with an exact backtest reproduction, run forward on paper for a quarter or
two before risking real capital. Backtests find fits that don't generalise;
forward-tests don't. The 24.1% number includes a known small look-ahead leak
(research call CF1, documented in `docs/strategy.md`) plus the survivorship bias
above. Our internal estimate is that forward returns will be 2–4 pp below
backtest, on average.

If your forward results are within a few pp of expectations, the strategy is
working as designed. If they're radically different in either direction, that's
a signal worth investigating.
