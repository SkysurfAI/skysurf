"""Strategy configuration — every tunable parameter, in one immutable object.

This module is the single source of truth for strategy parameters; no
strategy magic number should appear in any other file. The frozen
dataclass means the configuration cannot be mutated by accident at
runtime.

:data:`PHASE_4_V1` is the canonical locked configuration that reproduces
the validated walk-forward result on the research dataset:

* MAR = 1.96
* CAGR = 24.13%
* Max drawdown = −12.31%
* 604 trades over 2010-01-01 → 2026-03-21

Any deviation from :data:`PHASE_4_V1` is a strategy change, not a bug
fix, and must be re-validated on the user's own data before going live.

Field names match the equivalent CONFIG-dict keys in the original
research code so that the porting diff stays readable. See
:doc:`/docs/strategy.md` for the public-facing description.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Entry types that exist historically but are intentionally NOT used by
#: the Phase 4 production configuration. Surfaced as a constant so the
#: orchestrator can drop them without referencing a magic list elsewhere.
SUSPENDED_ENTRY_TYPES: tuple[str, ...] = ("ATH_BREAKOUT", "PULLBACK_STRUCTURAL")


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy configuration.

    Fields are grouped by functional area. Every parameter the brain
    consumes is here; downstream modules read the config but never mutate
    it. To change behaviour, build a new ``StrategyConfig`` (or use
    :func:`dataclasses.replace`) — never edit ``PHASE_4_V1`` in place.
    """

    # ── Universe filters ─────────────────────────────────────────────
    market_cap_floor_inr: float
    """Minimum market cap (absolute INR). Production filter."""

    adtv_floor_inr: float
    """Minimum 20-day average daily traded value (absolute INR)."""

    # ── Regime gate ──────────────────────────────────────────────────
    regime_combination: str
    """Multi-regime combination mode. PHASE_4_V1 = ``"OVERALL_OR_SECTOR"``."""

    overall_breadth_threshold: int
    """Minimum % of universe above SMA-25 for overall bullish state."""

    sector_rs_filter: str
    """Quartile filter on sector RS. PHASE_4_V1 = ``"TOP"`` (top 25%)."""

    regime_type: str
    """Overall regime classification scheme. PHASE_4_V1 = ``"breadth_5state"``."""

    entry_filter: str
    """Strictness of regime gate. PHASE_4_V1 = ``"aggressive"``."""

    # ── Stock-level filters ──────────────────────────────────────────
    rs_gate: float | None
    """Hard floor on stock 13-week relative strength. PHASE_4_V1 = 0.5."""

    rsi_gate: float | None
    """Optional RSI gate. PHASE_4_V1 = ``None`` (not enforced)."""

    volume_gate: float | None
    """Optional minimum 20-week volume ratio. PHASE_4_V1 = ``None``."""

    volume_min_ratio: float | None
    """Optional engine-side minimum volume ratio. PHASE_4_V1 = ``None``."""

    rr_floor: float
    """Minimum risk/reward ratio. PHASE_4_V1 = 0 (no filter)."""

    # ── Entry detection ──────────────────────────────────────────────
    entry_mas_per_type: dict[str, tuple[str, int]]
    """Per-entry-type ``(MA type, MA period)`` pair."""

    base_min_weeks_per_type: dict[str, int]
    """Per-entry-type minimum weeks-in-stage-2 required to fire."""

    swing_order: int
    """argrelextrema order for major swing detection. PHASE_4_V1 = 8."""

    swing_order_minor: int
    """Minor swing order; used by PULLBACK and VCP. PHASE_4_V1 = 5."""

    swing_lag_weeks: int
    """A swing at index ``S`` becomes available at ``S + lag_weeks``."""

    retest_proximity_atr: float
    """RETEST_SUPPORT proximity to retest level, in ATR units."""

    retest_max_weeks: int
    """RETEST_SUPPORT max weeks since breakout."""

    entry_price_method: str
    """When the entry price is sampled. PHASE_4_V1 = ``"week_close"``."""

    suspended_entry_types: tuple[str, ...]
    """Entry types historically present but not used by Phase 4."""

    # ── Per-detector parameter constants ─────────────────────────────
    breakout_base_min_weeks: int
    """Minimum non-stage-2 weeks before breakout. Research = 4."""

    breakout_ceiling_def: str
    """Breakout-ceiling formula. Research = ``"ma_plus_atr"``."""

    pullback_min_stage2: int
    """Minimum consecutive stage-2 weeks before pullback. Research = 4."""

    pullback_depth_def: str
    """Pullback depth-zone definition. Research = ``"pb_swing_low"``."""

    pullback_confirmation: str
    """Pullback confirmation rule. Research = ``"close_above_ma"``."""

    vcp_consol_min_weeks: int
    """Minimum consolidation duration for VCP. Research = 3."""

    vcp_contraction_req: str
    """VCP contraction requirement. Research = ``"vcp_any"``."""

    vcp_volume_req: str
    """VCP volume requirement. Research = ``"vol_declining"``."""

    breakout_subtype_split_weeks: int
    """Threshold (weeks_in_stage2) for splitting breakout subtypes. Research = 4."""

    # ── Overlap & ranking ────────────────────────────────────────────
    overlap_priority: str
    """Per-(ticker, week) collision dedup rule. PHASE_4_V1 = ``"tightest_stop"``."""

    ranking_method: str
    """Cross-candidate sort key. PHASE_4_V1 = ``"type_prior"``."""

    type_prior_mode: str
    """How type_prior is sourced. PHASE_4_V1 = ``"dynamic"`` (per-segment OOS)."""

    type_prior_min_type_n: int
    """Min trades-of-type pre-cutoff to compute dynamic prior."""

    type_prior_min_total_n: int
    """Min total trades pre-cutoff. Below → all types use ``type_prior_default``."""

    type_prior_default: float
    """Fallback prior when sample sizes too small."""

    static_type_prior_reference: dict[str, float]
    """Audit-only full-period reference. NOT used by dynamic ranking."""

    # ── Sizing ───────────────────────────────────────────────────────
    risk_pct: float
    """Per-trade risk as fraction of equity. PHASE_4_V1 = 0.01 (1%)."""

    sizing_on: str
    """``"total_equity"`` or ``"capital"``. PHASE_4_V1 = ``"total_equity"``."""

    tier_rr_thresholds: tuple[float, float]
    """RR breakpoints for ``(starter→half, half→full)`` tier assignment."""

    tier_multipliers: tuple[float, float, float]
    """Risk multipliers for ``(starter, half, full)`` tiers."""

    concentration_caps: tuple[float, float, float]
    """Per-position max-equity-fraction for ``(starter, half, full)``."""

    # ── Portfolio constraints ────────────────────────────────────────
    sector_limit: int
    """Maximum concurrent positions per sector. PHASE_4_V1 = 5."""

    max_positions: int
    """Maximum concurrent total positions. PHASE_4_V1 = 30."""

    pyramid_enabled: bool
    """Whether pyramiding adds are allowed. PHASE_4_V1 = ``False``."""

    time_stop_enabled: bool
    """Whether time-stop forced exit is enabled. PHASE_4_V1 = ``False``."""

    # ── Initial trail ────────────────────────────────────────────────
    exit_type: str
    """Stop methodology. PHASE_4_V1 = ``"sma_trail"``."""

    exit_ma_type: str
    """Initial trail MA type. PHASE_4_V1 = ``"SMA"``."""

    exit_ma_period: int
    """Initial trail MA period. PHASE_4_V1 = 27."""

    exit_atr_buffer: float
    """ATR multiplier subtracted from MA for stop. PHASE_4_V1 = 0.75."""

    exit_gtt_field: str
    """OHLC field that triggers GTT exit. PHASE_4_V1 = ``"close"``."""

    stop_method: str
    """Stop methodology variant. PHASE_4_V1 = ``"CURRENT"``."""

    # ── Triple-stack tightening ──────────────────────────────────────
    triple_stack_enabled: bool
    """Master switch for stall / extension / climactic triggers."""

    stall_tighten_week: int
    """Week threshold for stall trigger. PHASE_4_V1 = 12."""

    stall_tighten_threshold: float
    """Return threshold for stall trigger. PHASE_4_V1 = 0.07 (7%)."""

    stall_tighten_ma: str
    """MA to switch to on stall. PHASE_4_V1 = ``"sma_20"``."""

    extension_atr_mult: float
    """ATR multiplier for extension trigger. PHASE_4_V1 = 3.5."""

    extension_tighten_ma: str
    """MA to switch to on extension. PHASE_4_V1 = ``"sma_15"``."""

    climactic_vol_threshold: float
    """Volume-ratio threshold for climactic trigger. PHASE_4_V1 = 2.0."""

    climactic_tighten_ma: str
    """MA to switch to on climactic. PHASE_4_V1 = ``"sma_15"``."""

    # ── Partial profit ───────────────────────────────────────────────
    partial_profit_enabled: bool
    """Master switch. PHASE_4_V1 = ``True``."""

    partial_profit_trigger_pct: float
    """Unrealized-gain threshold (%). PHASE_4_V1 = 30."""

    partial_profit_sell_pct: float
    """Fraction of position to sell. PHASE_4_V1 = 50."""

    partial_profit_move_stop: str
    """Post-partial stop action. PHASE_4_V1 = ``"HALF_BACK"``."""

    # ── Progressive trail ────────────────────────────────────────────
    trail_tighten_enabled: bool
    """Master switch. PHASE_4_V1 = ``True``."""

    trail_tighten_trigger_pct: float
    """Unrealized-gain threshold (%) to switch trail. PHASE_4_V1 = 50."""

    trail_tighten_ma_type: str
    """Switched-trail MA type. PHASE_4_V1 = ``"SMA"``."""

    trail_tighten_ma_period: int
    """Switched-trail MA period. PHASE_4_V1 = 20."""

    trail_tighten_atr_buffer: float
    """ATR multiplier for switched trail. PHASE_4_V1 = 0.75."""

    # ── Costs ────────────────────────────────────────────────────────
    cost_model: str
    """Cost model identifier. PHASE_4_V1 = ``"ZERODHA"``."""

    zerodha_buy_cost_pct: float
    """Buy-side regulatory %. PHASE_4_V1 = 0.0011874."""

    zerodha_sell_cost_pct: float
    """Sell-side regulatory %. PHASE_4_V1 = 0.0010374."""

    zerodha_dp_charge_inr: float
    """Flat DP charge per sell transaction per scrip. PHASE_4_V1 = 15.93 INR."""

    slippage_pct: float
    """Per-side market-impact slippage. PHASE_4_V1 = 0.001 (0.1%)."""

    brokerage_pct: float
    """Per-side brokerage (FLAT model). PHASE_4_V1 = 0 (Zerodha delivery is free)."""

    # ── Walk-forward (validation only) ───────────────────────────────
    is_split_date: str
    """Sentinel for in-sample / OOS split. PHASE_4_V1 = ``"2099-01-01"``."""

    starting_capital: float
    """Validation starting capital (INR). PHASE_4_V1 = 500_000."""

    risk_free_annual: float
    """Risk-free rate for Sharpe / Sortino. PHASE_4_V1 = 0.06."""


#: One crore in absolute INR. Used to convert the human-friendly
#: market-cap and ADTV thresholds into raw rupees.
_INR_CRORE = 1e7


PHASE_4_V1 = StrategyConfig(
    # Universe filters
    market_cap_floor_inr=1_500 * _INR_CRORE,  # ₹1,500 Cr
    adtv_floor_inr=2 * _INR_CRORE,  # ₹2 Cr 20-day avg
    # Regime gate
    regime_combination="OVERALL_OR_SECTOR",
    overall_breadth_threshold=60,
    sector_rs_filter="TOP",
    regime_type="breadth_5state",
    entry_filter="aggressive",
    # Stock-level filters
    rs_gate=0.5,
    rsi_gate=None,
    volume_gate=None,
    volume_min_ratio=None,
    rr_floor=0,
    # Entry detection
    entry_mas_per_type={
        "PULLBACK_S2": ("EMA", 20),
        "BREAKOUT_S1_TO_S2": ("SMA", 40),
        "VCP_CONTINUATION": ("EMA", 40),
        "RETEST_SUPPORT": ("SMA", 40),
        "TRENDLINE_BOUNCE": ("EMA", 40),
    },
    base_min_weeks_per_type={
        "PULLBACK_S2": 12,
        "BREAKOUT_S1_TO_S2": 0,
        "VCP_CONTINUATION": 4,
        "RETEST_SUPPORT": 8,
        "TRENDLINE_BOUNCE": 8,
    },
    swing_order=8,
    swing_order_minor=5,
    swing_lag_weeks=8,
    retest_proximity_atr=1.5,
    retest_max_weeks=12,
    entry_price_method="week_close",
    suspended_entry_types=SUSPENDED_ENTRY_TYPES,
    # Per-detector parameter constants
    breakout_base_min_weeks=4,
    breakout_ceiling_def="ma_plus_atr",
    pullback_min_stage2=4,
    pullback_depth_def="pb_swing_low",
    pullback_confirmation="close_above_ma",
    vcp_consol_min_weeks=3,
    vcp_contraction_req="vcp_any",
    vcp_volume_req="vol_declining",
    breakout_subtype_split_weeks=4,
    # Overlap & ranking
    overlap_priority="tightest_stop",
    ranking_method="type_prior",
    type_prior_mode="dynamic",
    type_prior_min_type_n=20,
    type_prior_min_total_n=100,
    type_prior_default=1.0,
    static_type_prior_reference={
        "PULLBACK_S2": 2.92,
        "VCP_CONTINUATION": 2.19,
        "BREAKOUT_S1_TO_S2": 2.12,
        "RETEST_SUPPORT": 1.61,
        "TRENDLINE_BOUNCE": 1.21,
    },
    # Sizing
    risk_pct=0.01,
    sizing_on="total_equity",
    tier_rr_thresholds=(1.5, 2.0),
    tier_multipliers=(0.5, 0.75, 1.0),
    concentration_caps=(0.08, 0.15, 0.25),
    # Portfolio constraints
    sector_limit=5,
    max_positions=30,
    pyramid_enabled=False,
    time_stop_enabled=False,
    # Initial trail
    exit_type="sma_trail",
    exit_ma_type="SMA",
    exit_ma_period=27,
    exit_atr_buffer=0.75,
    exit_gtt_field="close",
    stop_method="CURRENT",
    # Triple-stack tightening
    triple_stack_enabled=True,
    stall_tighten_week=12,
    stall_tighten_threshold=0.07,
    stall_tighten_ma="sma_20",
    extension_atr_mult=3.5,
    extension_tighten_ma="sma_15",
    climactic_vol_threshold=2.0,
    climactic_tighten_ma="sma_15",
    # Partial profit
    partial_profit_enabled=True,
    partial_profit_trigger_pct=30,
    partial_profit_sell_pct=50,
    partial_profit_move_stop="HALF_BACK",
    # Progressive trail
    trail_tighten_enabled=True,
    trail_tighten_trigger_pct=50,
    trail_tighten_ma_type="SMA",
    trail_tighten_ma_period=20,
    trail_tighten_atr_buffer=0.75,
    # Costs
    cost_model="ZERODHA",
    zerodha_buy_cost_pct=0.0011874,
    zerodha_sell_cost_pct=0.0010374,
    zerodha_dp_charge_inr=15.93,
    slippage_pct=0.001,
    brokerage_pct=0,
    # Walk-forward
    is_split_date="2099-01-01",
    starting_capital=500_000,
    risk_free_annual=0.06,
)
"""Canonical Phase 4 locked configuration. Reproduces MAR 1.96 / CAGR 24.13% /
MaxDD −12.31% / 604 trades on the validation walk-forward window."""
