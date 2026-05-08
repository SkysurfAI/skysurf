"""Position management — public entry point for exit-side decisions.

Given a :class:`~skysurf.data.provider.DataProvider`, a Friday-close
``as_of_date``, and the list of currently-held positions, returns a
list of :class:`PositionAction` objects describing what (if anything)
to do with each position this week: hold, ratchet the stop, take
partial profit, switch to a tighter trailing MA, or fully exit.

The library emits decisions; the caller places the broker orders. The
library has no knowledge of whether orders fill.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date

import pandas as pd

from skysurf._internal.adapter import DataProviderCacheAdapter
from skysurf._internal.engine import _Position, decide_exits_for_week
from skysurf._internal.translation import public_to_internal
from skysurf.config import PHASE_4_V1, StrategyConfig
from skysurf.data.provider import DataProvider
from skysurf.types import ActionType, Position, PositionAction, TrailMode, TripleStackTighten

_LOG = logging.getLogger(__name__)

#: Lookback (weeks) used by the cache adapter — long enough for the
#: longest indicator (typically SMA-40) plus a buffer.
_EXIT_LOOKBACK_WEEKS: int = 300


def manage_positions(
    provider: DataProvider,
    as_of_date: date | pd.Timestamp,
    current_positions: list[Position],
    config: StrategyConfig = PHASE_4_V1,
) -> list[PositionAction]:
    """Return the action to take on each held position this week.

    Args:
        provider: Concrete :class:`DataProvider` supplying weekly OHLCV
            and regime data.
        as_of_date: Friday close. The actions returned are valid for
            the following week's open.
        current_positions: Currently-held positions. Inputs are not
            mutated; new state is expressed via :class:`PositionAction`
            objects.
        config: Strategy configuration. Defaults to the canonical
            :data:`PHASE_4_V1` lock.

    Returns:
        One :class:`PositionAction` per input position, in the same
        order. Possible action types are documented on
        :class:`~skysurf.types.ActionType`.
    """
    as_of = pd.Timestamp(as_of_date)
    if not current_positions:
        return []

    window_start = as_of - pd.Timedelta(weeks=_EXIT_LOOKBACK_WEEKS)
    nifty_weekly = provider.get_nifty_weekly(window_start, as_of)
    if nifty_weekly.empty:
        _LOG.warning("manage_positions: no Nifty data; carrying all positions forward")
        return [_hold_action(p, "no_nifty_data") for p in current_positions]
    nifty_close = nifty_weekly["Close"]

    # Cache adapter populated with every MA the engine could touch.
    ma_periods: set[tuple[str, int]] = set(config.entry_mas_per_type.values())
    extra_smas = sorted(
        {config.exit_ma_period}
        | {int(config.stall_tighten_ma.split("_")[-1])}
        | {int(config.extension_tighten_ma.split("_")[-1])}
        | {int(config.climactic_tighten_ma.split("_")[-1])}
        | {config.trail_tighten_ma_period}
    )
    adapter = DataProviderCacheAdapter(
        provider=provider,
        nifty_close_weekly=nifty_close,
        window_start=window_start,
        window_end=as_of,
        ma_periods=ma_periods,
        extra_sma_periods=extra_smas,
    )

    # One-row regime DataFrame for the exit week's regime label.
    snap = provider.get_overall_regime_snapshot(as_of)
    regime_col = (
        "regime_breadth_sma_25" if config.regime_type == "breadth_5state" else "regime_weinstein"
    )
    if snap is None:
        regime_df = pd.DataFrame({regime_col: ["unknown"]}, index=[as_of])
    else:
        regime_df = pd.DataFrame({regime_col: [snap["regime"]]}, index=[as_of])

    # Snapshot inputs so we can diff and emit PositionAction objects.
    inputs_by_ticker: dict[str, Position] = {p.ticker: p for p in current_positions}
    internal_in: list[_Position] = [public_to_internal(p, config) for p in current_positions]
    inputs_internal_by_ticker: dict[str, _Position] = {p.ticker: p for p in internal_in}

    cfg_dict = _config_to_dict(config)
    closed_trades: list[dict[str, object]] = []
    diag: Counter[str] = Counter()

    survivors, _cash = decide_exits_for_week(
        wk=as_of,
        positions=internal_in,
        cash=0.0,
        cfg=cfg_dict,
        cache=adapter,
        regime_df=regime_df,
        regime_col=regime_col,
        closed_trades=closed_trades,
        diag=diag,
    )

    survivors_by_ticker: dict[str, _Position] = {p.ticker: p for p in survivors}
    closed_by_ticker: dict[str, dict[str, object]] = {str(t["ticker"]): t for t in closed_trades}

    actions: list[PositionAction] = []
    for ticker, original_public in inputs_by_ticker.items():
        original_internal = inputs_internal_by_ticker[ticker]
        if ticker in closed_by_ticker:
            trade = closed_by_ticker[ticker]
            actions.append(
                PositionAction(
                    ticker=ticker,
                    action_type=ActionType.EXIT_FULL,
                    reason=str(trade.get("exit_reason", "exit")),
                    sell_qty=int(original_public.qty),
                    sell_price=float(trade.get("exit_price", 0.0)),  # type: ignore[arg-type]
                )
            )
            continue

        if ticker not in survivors_by_ticker:
            actions.append(_hold_action(original_public, "no_data_this_week"))
            continue

        actions.append(_diff_to_action(original_internal, survivors_by_ticker[ticker]))

    return actions


def _hold_action(position: Position, reason: str) -> PositionAction:
    return PositionAction(
        ticker=position.ticker,
        action_type=ActionType.HOLD,
        reason=reason,
    )


def _diff_to_action(before: _Position, after: _Position) -> PositionAction:
    """Compute the public action that explains the engine's update."""
    # Partial sell taken this week.
    if after.partial_taken and not before.partial_taken:
        return PositionAction(
            ticker=after.ticker,
            action_type=ActionType.PARTIAL_SELL,
            reason="partial_profit",
            sell_qty=int(after.partial_sale_qty),
            sell_price=float(after.partial_sale_price),
            new_stop_level=float(after.stop_level),
        )

    # Trail mode flipped to PROGRESSIVE.
    if after.trail_tightened and not before.trail_tightened:
        return PositionAction(
            ticker=after.ticker,
            action_type=ActionType.SWITCH_TRAIL_MA,
            reason="progressive_trail",
            new_trail_mode=TrailMode.PROGRESSIVE,
            new_stop_level=float(after.stop_level),
        )

    # Triple-stack state changed (None → stall / climactic).
    if after.tighten_state != before.tighten_state and after.tighten_state is not None:
        return PositionAction(
            ticker=after.ticker,
            action_type=ActionType.SWITCH_TRAIL_MA,
            reason=f"triple_stack_{after.tighten_state}",
            new_triple_stack_state=(
                TripleStackTighten.STALL
                if after.tighten_state == "stall"
                else TripleStackTighten.CLIMACTIC
            ),
            new_stop_level=float(after.stop_level),
        )

    # Plain stop ratchet.
    if after.stop_level > before.stop_level + 1e-6:
        return PositionAction(
            ticker=after.ticker,
            action_type=ActionType.UPDATE_STOP,
            reason="trail_ratchet",
            new_stop_level=float(after.stop_level),
        )

    return PositionAction(
        ticker=after.ticker,
        action_type=ActionType.HOLD,
        reason="no_change",
    )


def _config_to_dict(config: StrategyConfig) -> dict[str, object]:
    """Translate :class:`StrategyConfig` into the engine's cfg dict."""
    return {
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
