# Strategy reference

This page describes every parameter, every detector, every exit rule in the canonical Phase 4 strategy. It's a reference manual, not a research paper — for the validation methodology and the why-not-other-things, see the maintainers' own writeups.

The configuration in this document is what `PHASE_4_V1` (in [src/skysurf/brain/config.py](../src/skysurf/brain/config.py)) ships with, and what produces the validated 1.96 MAR / 24.13% CAGR / −12.31% MaxDD / 604 trades.

## At a glance

| | |
|---|---|
| Bar frequency | Weekly (Friday close) |
| Universe | Indian equities ≥ ₹1,500 Cr market cap, ≥ ₹2 Cr 20-day ADTV |
| Per-trade risk | 1% of total equity |
| Position cap | 30 total, 5 per sector |
| Cost model | Zerodha equity delivery (regulatory + flat DP charge) |
| Validation window | 2010-01-01 → 2026-03-21 (16 years 3 months) |

## Decision flow per week

Every Friday after the cash-market close:

1. **Regime gate.** Block all entries unless either (a) the overall market breadth (% of universe above SMA-25) exceeds 60, or (b) the candidate's sector is in a Weinstein bull regime.
2. **Sector RS filter.** Drop candidates whose sector is not in the top quartile by 13-week relative strength.
3. **Stock RS gate.** Drop candidates whose own 13-week RS is below 0.5.
4. **Run the five entry detectors** (below). Each candidate carries a stop derived from its detection rule.
5. **Per-(ticker, week) deduplication.** If multiple detectors fire on the same ticker in the same week, keep the one with the **tightest stop** (lowest `risk_pct_at_entry`).
6. **Cross-candidate ranking.** Sort by dynamic type-prior score (per-segment, computed from completed trades to date).
7. **Position sizing.** 1% of equity at risk × tier multiplier (0.5 / 0.75 / 1.0 by R:R).
8. **Constraints.** Max 5 positions per sector, max 30 total positions.

For each held position, the exit pipeline runs in parallel:

1. **Maintain initial trailing stop.** SMA-27 minus 0.75×ATR, ratchets up only.
2. **Triple-stack tightening.** Three independent triggers move the trailing-MA baseline to a faster MA:
   * Stall: week ≥ 12 and current return < 7% → SMA-20.
   * Extension: close > MA + 3.5×ATR → SMA-15 (transient).
   * Climactic: volume ratio > 2.0 and close < previous close → SMA-15.
3. **Partial profit at +30%.** Sell 50% of the position; move stop to halfway between entry and current peak (HALF_BACK).
4. **Progressive trail at +50%.** Switch the trailing MA to SMA-20 minus 0.75×ATR.
5. **Exit on weekly close ≤ stop.** Place a GTT order at the stop level on Monday morning.

## Universe filters

| Parameter | Value | Notes |
|---|---|---|
| `market_cap_floor_inr` | ₹1,500 Cr | Absolute INR (not Nifty-relative). Production live filter. |
| `adtv_floor_inr` | ₹2 Cr | 20-day average daily traded value. |

## Regime gate

Five-state breadth-based classifier on the overall market (SMA-25 of the universe), combined with Weinstein-style sector regimes.

| Parameter | Value |
|---|---|
| `regime_combination` | `OVERALL_OR_SECTOR` |
| `overall_breadth_threshold` | 60 (%) |
| `sector_rs_filter` | `TOP` (top quartile only) |
| `regime_type` | `breadth_5state` |
| `entry_filter` | `aggressive` |

The `OVERALL_OR_SECTOR` combination admits a candidate if **either** the overall regime is permissive **or** the sector regime is bullish. This is more aggressive than `AND` — it lets sector strength rescue an overall-flat market.

## Stock-level filters

| Parameter | Value |
|---|---|
| `rs_gate` | 0.5 (hard floor on 13-week RS) |
| `rsi_gate` | none |
| `volume_gate` | none |
| `rr_floor` | 0 (no risk/reward filter) |

## Entry detectors

Five detectors run in parallel on every eligible stock. Each has its own stage definition (the MA the price must be above) and its own minimum-stage-2-weeks.

| Detector | MA | Min weeks in stage 2 |
|---|---|---|
| `PULLBACK_S2` | EMA-20 | 12 |
| `BREAKOUT_S1_TO_S2` | SMA-40 | 0 (breakouts are fresh by definition) |
| `VCP_CONTINUATION` | EMA-40 | 4 |
| `RETEST_SUPPORT` | SMA-40 | 8 |
| `TRENDLINE_BOUNCE` | EMA-40 | 8 |

Two detectors are documented but **not** run in PHASE_4_V1: `ATH_BREAKOUT` and `PULLBACK_STRUCTURAL`. They're listed in `SUSPENDED_ENTRY_TYPES` so the orchestrator drops them.

### Per-detector specifics

| Parameter | Value |
|---|---|
| `breakout_base_min_weeks` | 4 (minimum non-stage-2 base period) |
| `breakout_ceiling_def` | `ma_plus_atr` (MA + 1×ATR, **not** prior swing-high) |
| `pullback_min_stage2` | 4 |
| `pullback_depth_def` | `pb_swing_low` (most-recent minor swing low — not MA proximity) |
| `pullback_confirmation` | `close_above_ma` |
| `vcp_consol_min_weeks` | 3 |
| `vcp_contraction_req` | `vcp_any` (accepts any contraction count) |
| `vcp_volume_req` | `vol_declining` |
| `breakout_subtype_split_weeks` | 4 |

### Swing detection

| Parameter | Value |
|---|---|
| `swing_order` | 8 (argrelextrema order — major swings) |
| `swing_order_minor` | 5 (PULLBACK / VCP minor swings) |
| `swing_lag_weeks` | 8 (a swing at index S becomes available at S+8) |
| `retest_proximity_atr` | 1.5 |
| `retest_max_weeks` | 12 |
| `entry_price_method` | `week_close` (Friday close) |

## Overlap and ranking

When multiple detectors fire on the same ticker in the same week:

| Stage | Rule |
|---|---|
| Per-(ticker, week) | `tightest_stop` (lowest `risk_pct_at_entry`) wins |
| Cross-candidate | `type_prior` descending |

Type-prior is computed dynamically from the trade history available before the cutoff date. Behaviour:

| Parameter | Value | Effect |
|---|---|---|
| `type_prior_mode` | `dynamic` | Per-segment OOS, no in-sample leak |
| `type_prior_min_type_n` | 20 | Below: this type uses `type_prior_default` |
| `type_prior_min_total_n` | 100 | Below: all types use `type_prior_default` |
| `type_prior_default` | 1.0 | Fallback |

The static reference (`static_type_prior_reference`) is for audit only — it shows full-period MFE/MAE-derived priors but is not used by ranking.

## Sizing

| Parameter | Value |
|---|---|
| `risk_pct` | 0.01 (1% per trade) |
| `sizing_on` | `total_equity` |
| `tier_rr_thresholds` | (1.5, 2.0) |
| `tier_multipliers` | (0.5, 0.75, 1.0) — starter / half / full |
| `concentration_caps` | (0.08, 0.15, 0.25) — max equity fraction per tier |

A candidate with R:R below 1.5 sizes at 0.5× the risk multiplier (a "starter"). Between 1.5 and 2.0 it's a "half"; above 2.0 it's a "full".

## Portfolio constraints

| Parameter | Value |
|---|---|
| `sector_limit` | 5 |
| `max_positions` | 30 |
| `pyramid_enabled` | False |
| `time_stop_enabled` | False |

## Initial trailing stop

| Parameter | Value |
|---|---|
| `exit_type` | `sma_trail` |
| `exit_ma_type` | SMA |
| `exit_ma_period` | 27 |
| `exit_atr_buffer` | 0.75 |
| `exit_gtt_field` | `close` (trigger on weekly close ≤ stop) |
| `stop_method` | `CURRENT` |

## Triple-stack tightening

Three independent triggers move the trailing-MA baseline to a faster MA. Only the first two persist on `Position` state:

| Trigger | Condition | Effect |
|---|---|---|
| Stall | `week ≥ 12` and `current_return < 7%` | Switch baseline to SMA-20 |
| Extension | `close > baseline_MA + 3.5 × ATR` | Switch baseline to SMA-15 (transient — applied per-week, not persisted) |
| Climactic | `volume_ratio > 2.0` and `close < prev_close` | Switch baseline to SMA-15 |

## Partial profit

| Parameter | Value |
|---|---|
| `partial_profit_trigger_pct` | 30 |
| `partial_profit_sell_pct` | 50 |
| `partial_profit_move_stop` | `HALF_BACK` |

`HALF_BACK` moves the stop to the midpoint between entry and current peak.

## Progressive trail

| Parameter | Value |
|---|---|
| `trail_tighten_trigger_pct` | 50 (unrealized gain) |
| `trail_tighten_ma_type` | SMA |
| `trail_tighten_ma_period` | 20 |
| `trail_tighten_atr_buffer` | 0.75 |

Once unrealized gain crosses 50%, the trailing stop switches from SMA-27 to SMA-20 (tighter), still with 0.75×ATR buffer.

## Costs

| Parameter | Value |
|---|---|
| `cost_model` | `ZERODHA` |
| `zerodha_buy_cost_pct` | 0.0011874 |
| `zerodha_sell_cost_pct` | 0.0010374 |
| `zerodha_dp_charge_inr` | ₹15.93 |
| `slippage_pct` | 0.001 (0.1%) |
| `brokerage_pct` | 0 (Zerodha equity delivery is free) |

A trade with a partial profit incurs **two** DP charges (one on the partial sell, one on the final exit).

## Known caveats baked into the validated number

* **Survivorship bias.** Universe membership is a 2026 snapshot applied historically. Estimated CAGR inflation: 2–3 pp.
* **Static sector / cap-tier mapping.** Same — 2026 snapshot.
* **CF1 residual leak.** The dynamic type-prior filters historical trades by entry `week_date < cutoff`. Trades that entered before the cutoff but exited after still contribute MFE/MAE — small look-ahead leak, accepted at validation.
* **Universe difference.** Backtest uses Nifty-scaled cap floor; production uses absolute INR cap floor. Accepted.

See [reproducibility.md](reproducibility.md) for the full discussion.
