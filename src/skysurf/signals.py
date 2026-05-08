"""Weekly signal generation — public entry point for new-position recommendations.

Given a :class:`~skysurf.data.provider.DataProvider`, a Friday-close
``as_of_date``, the current portfolio of held positions, and the total
equity, returns a ranked list of :class:`EntrySignal` objects: the new
positions the strategy recommends opening at the next week's open.

The library is stateless. The caller maintains the canonical store of
held positions and passes them on every call.

End-to-end pipeline
--------------------

1. Pull the eligible universe from the provider.
2. For each ticker, call
   :func:`~skysurf._internal.detection.find_entries_for_ticker` to
   compute indicators + run the five PHASE_4_V1 detectors.
3. Filter to entries firing on ``as_of_date``, dedup per
   ``(ticker, week_date)`` keeping the tightest stop, apply the
   ``rs_gate``, and rank by dynamic type-prior (computed from the
   provider's historical-trade history).
4. Build the cache adapter and a one-row regime DataFrame for the
   evaluation week.
5. Translate held positions to the engine's internal format.
6. Call :func:`~skysurf._internal.engine.select_entries_for_week` —
   applies the regime gate, sector RS filter, sector / total caps,
   sizing, and constructs new internal positions.
7. Translate the newly-opened positions to public
   :class:`EntrySignal` objects with diagnostic context.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date

import pandas as pd

from skysurf._internal.detection import find_entries_for_ticker
from skysurf._internal.engine import select_entries_for_week
from skysurf._internal.ranking import compute_type_prior
from skysurf._internal.translation import public_to_internal
from skysurf.config import PHASE_4_V1, StrategyConfig
from skysurf.data.provider import DataProvider
from skysurf.types import EntrySignal, Position

_LOG = logging.getLogger(__name__)

#: Lookback (weeks) used for cache-adapter and detection data fetches.
#: Must give the longest indicator (typically SMA-40) enough warmup
#: plus a buffer for swing detection.
_DETECTION_LOOKBACK_WEEKS: int = 300


def generate_weekly_signals(
    provider: DataProvider,
    as_of_date: date | pd.Timestamp,
    current_positions: list[Position],
    total_equity: float,
    config: StrategyConfig = PHASE_4_V1,
) -> list[EntrySignal]:
    """Return ranked entry signals for the week ending ``as_of_date``.

    Args:
        provider: Concrete :class:`DataProvider` supplying weekly
            OHLCV, regime / sector lookups, the eligible universe, and
            historical-trade history (for dynamic type-prior).
        as_of_date: Friday close. The signals returned are valid for
            the following week's open.
        current_positions: Currently-held positions. Used to apply
            per-sector and total-position constraints; not mutated.
        total_equity: Current portfolio equity (cash + invested).
        config: Strategy configuration. Defaults to the canonical
            :data:`PHASE_4_V1` lock.

    Returns:
        Ranked list of :class:`EntrySignal` instances, highest-priority
        first. The list respects ``config.sector_limit`` and
        ``config.max_positions`` after counting the held positions.
        Returns ``[]`` when the regime gate blocks, no candidates pass
        filters, or portfolio constraints saturate.
    """
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(weeks=_DETECTION_LOOKBACK_WEEKS)

    # ── Universe + Nifty ─────────────────────────────────────────
    universe = provider.get_universe(as_of)
    if universe.empty:
        _LOG.info("generate_weekly_signals: empty universe at %s", as_of)
        return []

    nifty_weekly = provider.get_nifty_weekly(window_start, as_of)
    if nifty_weekly.empty:
        _LOG.warning("generate_weekly_signals: no Nifty data at %s", as_of)
        return []
    nifty_close = nifty_weekly["Close"]

    # ── Per-ticker detection ─────────────────────────────────────
    sector_by_ticker: dict[str, str] = dict(
        zip(universe["ticker"], universe["sector"], strict=False)
    )
    all_events: list[dict[str, object]] = []
    for ticker in universe["ticker"]:
        weekly_map = provider.get_weekly_ohlcv([ticker], window_start, as_of)
        weekly = weekly_map.get(ticker)
        if weekly is None or weekly.empty:
            continue
        sector = sector_by_ticker.get(ticker, "UNKNOWN")
        events = find_entries_for_ticker(
            ticker=ticker,
            weekly_ohlcv=weekly,
            nifty_close_weekly=nifty_close,
            sector=sector,
            config=config,
        )
        all_events.extend(events)

    if not all_events:
        return []

    df = pd.DataFrame(all_events)
    df["week_date"] = pd.to_datetime(df["week_date"])

    # ── Filter to this week ──────────────────────────────────────
    week_entries = df[df["week_date"] == as_of].copy()
    if week_entries.empty:
        return []

    # ── Apply the canonical Phase 4 filter chain ─────────────────
    week_entries = _apply_phase_4_filters(week_entries, config)
    if week_entries.empty:
        return []

    # ── Dynamic type-prior ranking ───────────────────────────────
    historical_trades = provider.get_historical_trades(before_date=as_of)
    if historical_trades is None or historical_trades.empty:
        # Fallback to default prior across all types when no history.
        type_scores: dict[str, float] = {}
    else:
        type_scores = compute_type_prior(
            historical_trades,
            cutoff_date=as_of,
            min_type_n=config.type_prior_min_type_n,
            min_total_n=config.type_prior_min_total_n,
            default_prior=config.type_prior_default,
        )
    week_entries["type_prior"] = (
        week_entries["entry_type"].map(type_scores).fillna(config.type_prior_default)
    )
    # Engine ranks by config.ranking_method; we set "type_prior" as the column.
    week_entries = week_entries.sort_values("type_prior", ascending=False).reset_index(drop=True)

    # ── One-row regime DataFrame for this Friday ─────────────────
    # ``select_entries_for_week`` reads regime + breadth from this
    # frame; no cache adapter is needed here because the candidate
    # entries already carry every per-bar indicator they require.
    snap = provider.get_overall_regime_snapshot(as_of)
    regime_col = _regime_col(config)
    breadth_col = _breadth_col(config)
    if snap is None:
        regime_df = pd.DataFrame({regime_col: ["unknown"], breadth_col: [0.0]}, index=[as_of])
    else:
        regime_df = pd.DataFrame(
            {regime_col: [snap["regime"]], breadth_col: [float(snap["breadth_pct"])]},
            index=[as_of],
        )

    # ── Translate held positions ─────────────────────────────────
    internal_held = [public_to_internal(p, config) for p in current_positions]
    n_held = len(internal_held)

    # Cash = equity minus invested.
    invested = sum(p.qty * p.current_close for p in current_positions)
    cash = float(total_equity) - float(invested)

    cfg_dict = _config_to_dict(config)
    diag: Counter[str] = Counter()
    new_positions, _new_cash = select_entries_for_week(
        wk=as_of,
        positions=internal_held,
        cash=cash,
        equity_base=float(total_equity),
        entries_by_week={as_of: week_entries},
        cfg=cfg_dict,
        regime_df=regime_df,
        regime_col=regime_col,
        breadth_col=breadth_col,
        exit_ma_col=f"{config.exit_ma_type.lower()}_{config.exit_ma_period}",
        diag=diag,
        sector_regime_lookup=getattr(provider, "sector_regime_lookup", None),
        captier_regime_lookup=getattr(provider, "captier_regime_lookup", None),
        ticker_metadata_lookup=getattr(provider, "ticker_metadata", None),
        sector_rs_lookup=getattr(provider, "sector_rs_lookup", None),
    )

    appended = new_positions[n_held:]
    if not appended:
        return []

    # ── Translate to EntrySignal ─────────────────────────────────
    by_ticker = week_entries.set_index("ticker")
    signals: list[EntrySignal] = []
    for rpos in appended:
        target_level = 0.0
        type_prior_score = config.type_prior_default
        if rpos.ticker in by_ticker.index:
            row = by_ticker.loc[rpos.ticker]
            target_val = row.get("target_level")
            if pd.notna(target_val):
                target_level = float(target_val)
            tp_val = row.get("type_prior")
            if pd.notna(tp_val):
                type_prior_score = float(tp_val)
        signals.append(
            EntrySignal(
                ticker=rpos.ticker,
                sector=rpos.sector,
                entry_type=rpos.entry_type,
                entry_price=float(rpos.entry_price),
                initial_stop=float(rpos.stop_at_entry),
                qty=int(rpos.qty),
                tier=rpos.tier,
                entry_cost=float(rpos.cost),
                type_prior_score=type_prior_score,
                rs_13w=float(rpos.rs_13w_at_entry),
                target_level=target_level,
                rr_at_entry=float(rpos.rr_at_entry),
                diagnostic={
                    "regime_at_entry": rpos.regime_at_entry,
                    "stop_level": float(rpos.stop_level),
                    "diag": dict(diag),
                },
            )
        )
    return signals


def _apply_phase_4_filters(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Apply the canonical Phase 4 entry-filter chain to a candidate DataFrame.

    Filters in order:

    1. Per-type ``base_min_weeks_per_type`` floor on ``weeks_in_stage2``.
    2. ``rr_floor`` floor on ``rr_ratio`` for structural targets.
    3. ``rs_gate`` floor on ``rs_13w``.
    4. ``rsi_gate`` floor on ``rsi_at_entry``.
    5. ``volume_gate`` floor on ``volume_ratio_at_entry``.
    6. ``retest_max_weeks`` cap on ``weeks_since_breakout``.
    7. Per-(ticker, week_date) dedup keeping the tightest stop.
    """
    out = df.copy()

    # 1. Base-min per type.
    keep = pd.Series(True, index=out.index)
    for entry_type, min_weeks in config.base_min_weeks_per_type.items():
        if min_weeks > 0 and "weeks_in_stage2" in out.columns:
            is_type = out["entry_type"] == entry_type
            passes = out["weeks_in_stage2"] >= min_weeks
            keep &= ~is_type | passes
    out = out[keep]

    # 2. R/R floor on structural targets.
    if config.rr_floor and config.rr_floor > 0 and "rr_ratio" in out.columns:
        is_structural = out["target_type"] == "structural"
        out = out[~(is_structural & (out["rr_ratio"] < config.rr_floor))]

    # 3. RS gate.
    if config.rs_gate is not None and "rs_13w" in out.columns:
        out = out[out["rs_13w"].fillna(-999) >= config.rs_gate]

    # 4. RSI gate.
    if config.rsi_gate is not None and "rsi_at_entry" in out.columns:
        out = out[out["rsi_at_entry"].fillna(-999) >= config.rsi_gate]

    # 5. Volume gate.
    if config.volume_gate is not None and "volume_ratio_at_entry" in out.columns:
        out = out[out["volume_ratio_at_entry"].fillna(-999) >= config.volume_gate]

    # 6. Retest max-weeks-since-breakout.
    if "weeks_since_breakout" in out.columns:
        is_retest = out["entry_type"] == "RETEST_SUPPORT"
        too_old = out["weeks_since_breakout"].fillna(0) > config.retest_max_weeks
        out = out[~(is_retest & too_old)]

    # 7. Per-(ticker, week_date) dedup — keep tightest stop.
    if not out.empty and "risk_pct_at_entry" in out.columns:
        out = out.sort_values("risk_pct_at_entry", na_position="last")
        out = out.drop_duplicates(subset=["ticker", "week_date"], keep="first")

    return out.reset_index(drop=True)


def _regime_col(config: StrategyConfig) -> str:
    """Column name the engine reads regime values from."""
    if config.regime_type == "breadth_5state":
        # SMA-25 is the canonical breadth MA used by PHASE_4_V1.
        return "regime_breadth_sma_25"
    return "regime_weinstein"


def _breadth_col(config: StrategyConfig) -> str:
    """Column name the engine reads breadth percentages from."""
    return "breadth_pct_sma_25"


def _config_to_dict(config: StrategyConfig) -> dict[str, object]:
    """Translate a :class:`StrategyConfig` into the engine's cfg dict.

    The engine's per-week functions expect a ``Mapping`` keyed by the
    same field names as the dataclass. Returning a plain dict makes the
    keys mutable / iterable in the canonical form the engine expects.
    """
    cfg: dict[str, object] = {
        "regime_type": config.regime_type,
        "entry_filter": config.entry_filter,
        "overall_breadth_threshold": config.overall_breadth_threshold,
        "regime_combination": config.regime_combination,
        "sector_rs_filter": config.sector_rs_filter,
        "ranking_method": config.ranking_method,
        "rs_gate": config.rs_gate,
        "rsi_gate": config.rsi_gate,
        "volume_gate": config.volume_gate,
        "volume_min_ratio": config.volume_min_ratio,
        "rr_floor": config.rr_floor,
        "risk_pct": config.risk_pct,
        "sizing_on": config.sizing_on,
        "tier_rr_thresholds": config.tier_rr_thresholds,
        "tier_multipliers": config.tier_multipliers,
        "concentration_caps": config.concentration_caps,
        "sector_limit": config.sector_limit,
        "max_positions": config.max_positions,
        "pyramid_enabled": config.pyramid_enabled,
        "time_stop_enabled": config.time_stop_enabled,
        "exit_type": config.exit_type,
        "exit_ma_type": config.exit_ma_type,
        "exit_ma_period": config.exit_ma_period,
        "exit_atr_buffer": config.exit_atr_buffer,
        "exit_gtt_field": config.exit_gtt_field,
        "stop_method": config.stop_method,
        "triple_stack_enabled": config.triple_stack_enabled,
        "stall_tighten_week": config.stall_tighten_week,
        "stall_tighten_threshold": config.stall_tighten_threshold,
        "stall_tighten_ma": config.stall_tighten_ma,
        "extension_atr_mult": config.extension_atr_mult,
        "extension_tighten_ma": config.extension_tighten_ma,
        "climactic_vol_threshold": config.climactic_vol_threshold,
        "climactic_tighten_ma": config.climactic_tighten_ma,
        "partial_profit_enabled": config.partial_profit_enabled,
        "partial_profit_trigger_pct": config.partial_profit_trigger_pct,
        "partial_profit_sell_pct": config.partial_profit_sell_pct,
        "partial_profit_move_stop": config.partial_profit_move_stop,
        "trail_tighten_enabled": config.trail_tighten_enabled,
        "trail_tighten_trigger_pct": config.trail_tighten_trigger_pct,
        "trail_tighten_ma_type": config.trail_tighten_ma_type,
        "trail_tighten_ma_period": config.trail_tighten_ma_period,
        "trail_tighten_atr_buffer": config.trail_tighten_atr_buffer,
        "cost_model": config.cost_model,
        "zerodha_buy_cost_pct": config.zerodha_buy_cost_pct,
        "zerodha_sell_cost_pct": config.zerodha_sell_cost_pct,
        "zerodha_dp_charge_inr": config.zerodha_dp_charge_inr,
        "slippage_pct": config.slippage_pct,
        "brokerage_pct": config.brokerage_pct,
        "config_id": "skysurf-public",
    }
    return cfg
