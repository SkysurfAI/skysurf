"""Per-week entry-selection and exit-decision engine.

This is the heart of the strategy: given a week's candidate entries and
the currently held positions, decide which entries to take and what to
do with each held position.

Two entry points:

* :func:`select_entries_for_week` — given a candidate-entry DataFrame
  and the held portfolio, returns the updated portfolio (held + newly
  opened) plus updated cash. Applies regime gates, sector RS filter,
  per-position sizing, and concentration / sector / total-position
  constraints.
* :func:`decide_exits_for_week` — for each held position, evaluates
  the trailing stop, triple-stack tightening, partial-profit trigger,
  progressive trail, and time-stop. Closes positions whose week-close
  trips the stop (or whose forced-exit condition fires); ratchets
  stops on the survivors.

Both functions delegate to the cost helpers in
:mod:`skysurf.costs.zerodha`. Logic is byte-equivalent to the research
:mod:`val_engine_b_experimental` per-week functions used to validate
the published 1.96 MAR / 24.13% CAGR / −12.31% MaxDD.

Internal types:

* :class:`_Position` — engine-internal position record (matches the
  research :class:`Position`). The :func:`skysurf.signals` /
  :func:`skysurf.positions` public APIs translate between
  :class:`skysurf.types.Position` and this internal shape.
* :class:`_StockCacheManager` — base class consumed by the per-week
  exit evaluator. The cache adapter
  (:class:`skysurf._internal.adapter.DataProviderCacheAdapter`)
  subclasses it to read from the user's
  :class:`~skysurf.data.provider.DataProvider`.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from skysurf.costs.zerodha import buy_cost as _buy_cost
from skysurf.costs.zerodha import sell_proceeds as _sell_proceeds

_LOG = logging.getLogger(__name__)

#: Tier names indexed in the same order as ``cfg.tier_multipliers``.
_TIER_NAMES: tuple[str, str, str] = ("starter", "half", "full")

#: Captier multipliers applied to the structural-stop ATR buffer when
#: ``stop_atr_by_captier`` is enabled. PHASE_4_V1 does not enable this.
_CAPTIER_BUFFER_MULT: dict[str, float] = {
    "LARGE": 0.75,
    "MID": 1.0,
    "SMALL": 1.25,
    "MICRO": 1.25,
}


# ── Internal Position ────────────────────────────────────────────────


@dataclass
class _Position:
    """Engine-internal position record.

    Mirrors the research :class:`val_engine_b_experimental.Position`
    field-for-field. The public-API
    :class:`skysurf.types.Position` is a slimmer caller-facing record;
    :mod:`skysurf._internal.translation` converts between the two.
    """

    ticker: str
    sector: str
    entry_date: pd.Timestamp
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
    stop_at_entry: float
    # Tracking
    min_low: float = 0.0
    mfe_week: pd.Timestamp | None = None
    tighten_state: str | None = None  # None | "stall" | "climactic"
    prev_close: float = 0.0
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
    # Exit mechanism tracking
    time_stop_triggered: bool = False
    partial_taken: bool = False
    partial_sale_proceeds: float = 0.0
    partial_sale_qty: int = 0
    partial_sale_price: float = 0.0
    partial_sale_date: pd.Timestamp | None = None
    trail_tightened: bool = False


# ── StockCacheManager base class ─────────────────────────────────────


class _StockCacheManager:
    """Base interface for per-ticker indicator-augmented OHLCV access.

    The exit evaluator calls :meth:`get` once per held position per
    week. It expects a DataFrame indexed by week timestamp with at least
    these columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
    ``atr_14``, ``volume_ratio_20w``, plus the configured trail-MA
    columns (``sma_27``, ``sma_20``, ``sma_15``, etc., depending on
    config).

    Subclasses override :meth:`get` to fetch from their data source.
    The :class:`~skysurf._internal.adapter.DataProviderCacheAdapter`
    subclass bridges to the public
    :class:`~skysurf.data.provider.DataProvider`.
    """

    def get(self, ticker: str) -> pd.DataFrame | None:
        """Return the indicator-augmented weekly DataFrame for ``ticker``.

        Returns ``None`` when no data is available (the evaluator then
        carries the position forward unchanged for that week).
        """
        raise NotImplementedError


# ── Regime gates ──────────────────────────────────────────────────────


def is_entry_allowed(
    regime: str,
    breadth_pct: float | None,
    cfg: Mapping[str, Any],
) -> bool:
    """Decide whether the overall regime gate allows entries this week."""
    rt = cfg["regime_type"]
    ef = cfg["entry_filter"]
    bp = breadth_pct if breadth_pct is not None else 0.0
    bp_thresh = cfg.get("overall_breadth_threshold", 65)

    if rt == "breadth_5state":
        if ef == "aggressive":
            if regime == "bear":
                return False
            if regime in ("recovering", "deteriorating"):
                return bool(bp > bp_thresh)
            return True  # strong_bull / weakening_bull
        if ef == "moderate":
            return regime in ("strong_bull", "weakening_bull")
        if ef == "conservative":
            return regime == "strong_bull"
    elif rt == "weinstein_2signal":
        if ef == "aggressive":
            if regime == "bear":
                return False
            if regime == "sideways":
                return bool(bp > bp_thresh)
            return True
        if ef == "moderate":
            return regime == "bull"
        if ef == "conservative":
            return regime == "bull" and bp > 50
    return False


def passes_sector_rs_filter(quartile: str | None, cfg: Mapping[str, Any]) -> bool:
    """Decide whether a sector's relative-strength quartile passes the gate."""
    filt = cfg.get("sector_rs_filter", "NONE")
    if filt == "NONE":
        return True
    if quartile is None:
        return True  # don't block unclassifiable entries
    if filt == "TOP":
        return quartile == "TOP"
    if filt == "TOP_UPPER":
        return quartile in ("TOP", "UPPER")
    if filt == "NOT_BOTTOM":
        return quartile != "BOTTOM"
    return True


def is_entry_allowed_multi_regime(
    overall_allowed: bool,
    sector_regime: str | None,
    captier_regime: str | None,
    sector_has_data: bool,
    cfg: Mapping[str, Any],
) -> bool:
    """Apply the multi-dimensional regime combination logic.

    Combines overall, sector, and cap-tier regimes per
    ``cfg["regime_combination"]``. PHASE_4_V1 uses
    ``"OVERALL_OR_SECTOR"``.
    """
    combo = cfg.get("regime_combination", "OVERALL_ONLY")
    if combo == "OVERALL_ONLY":
        return overall_allowed

    overall_bull = overall_allowed
    sector_bull: bool | None = (
        (sector_regime == "bull") if sector_has_data and sector_regime else None
    )
    captier_bull = (captier_regime == "bull") if captier_regime else False

    if combo == "SECTOR_ONLY":
        return overall_bull if sector_bull is None else sector_bull
    if combo == "CAPTIER_ONLY":
        return captier_bull
    if combo == "OVERALL_OR_SECTOR":
        return overall_bull if sector_bull is None else (overall_bull or sector_bull)
    if combo == "OVERALL_OR_CAPTIER":
        return overall_bull or captier_bull
    if combo == "SECTOR_AND_CAPTIER":
        return (
            (overall_bull and captier_bull)
            if sector_bull is None
            else (sector_bull and captier_bull)
        )
    if combo == "OVERALL_OR_SECTOR_AND_CAPTIER":
        if sector_bull is None:
            return overall_bull or captier_bull
        return overall_bull or (sector_bull and captier_bull)
    if combo == "ANY_TWO_OF_THREE":
        score = int(overall_bull)
        score += int(sector_bull) if sector_bull is not None else 0
        score += int(captier_bull)
        return score >= 2

    return overall_allowed


# ── Position sizing ──────────────────────────────────────────────────


def compute_sizing(
    entry_price: float,
    stop_level: float,
    rr_ratio: float,
    equity_base: float,
    cash: float,
    cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Compute position size for a candidate entry.

    Returns ``{qty, tier, entry_cost, binding}`` or ``None`` if the
    entry can't be sized (insufficient cash at any tier, or
    ``entry_price <= stop_level``).
    """
    sl_distance = entry_price - stop_level
    if sl_distance <= 0:
        return None

    thresholds = cfg["tier_rr_thresholds"]
    if rr_ratio >= thresholds[1]:
        tier_idx = 2
    elif rr_ratio >= thresholds[0]:
        tier_idx = 1
    else:
        tier_idx = 0

    tier_multipliers = cfg["tier_multipliers"]
    concentration_caps = cfg["concentration_caps"]

    for try_idx in range(tier_idx, -1, -1):
        multiplier = tier_multipliers[try_idx]
        cap = concentration_caps[try_idx]

        risk_amount = equity_base * cfg["risk_pct"] * multiplier
        risk_qty = math.floor(risk_amount / sl_distance)
        cap_qty = math.floor(equity_base * cap / entry_price)
        qty = min(risk_qty, cap_qty)
        if qty <= 0:
            continue

        entry_cost = _buy_cost(qty, entry_price, cfg)
        if entry_cost > cash:
            continue

        return {
            "qty": qty,
            "tier": _TIER_NAMES[try_idx],
            "entry_cost": entry_cost,
            "binding": "risk" if risk_qty <= cap_qty else "cap",
        }
    return None


# ── Trade-record builder ─────────────────────────────────────────────


def _build_trade(
    pos: _Position,
    exit_price: float,
    exit_date: pd.Timestamp,
    exit_reason: str,
    proceeds: float,
) -> dict[str, Any]:
    """Build a closed-trade record dict from a position + exit."""
    ret_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
    pnl = proceeds - pos.cost
    days = (exit_date - pos.entry_date).days
    holding_weeks = max(1, round(days / 7))
    peak_unreal = (pos.peak_close - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
    give_back = peak_unreal - ret_pct

    # MAE on Low (worst intraweek), MFE on Close (achievable upside).
    mae_pct = (
        ((pos.min_low - pos.entry_price) / pos.entry_price) * 100 if pos.entry_price > 0 else 0
    )
    mfe_pct = peak_unreal * 100
    time_to_mfe_weeks = (
        max(1, (pos.mfe_week - pos.entry_date).days // 7) if pos.mfe_week is not None else 1
    )
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
        "rs_13w_at_entry": (
            round(pos.rs_13w_at_entry, 4) if not np.isnan(pos.rs_13w_at_entry) else None
        ),
        "regime_at_entry": pos.regime_at_entry,
        "regime_at_exit": "",  # set by caller
        "exit_reason": exit_reason,
        "entry_type": pos.entry_type,
        "target_type": pos.target_type,
        "risk_pct_at_entry": (
            round(pos.risk_pct_at_entry, 4) if not np.isnan(pos.risk_pct_at_entry) else None
        ),
        "cost": round(pos.cost, 2),
    }


# ── Exit evaluation ──────────────────────────────────────────────────


def decide_exits_for_week(
    wk: pd.Timestamp,
    positions: list[_Position],
    cash: float,
    cfg: Mapping[str, Any],
    cache: _StockCacheManager,
    regime_df: pd.DataFrame,
    regime_col: str,
    closed_trades: list[dict[str, Any]],
    diag: Counter[str],
) -> tuple[list[_Position], float]:
    """Per-week exit pipeline.

    Walks every held position, ratcheting trailing stops, applying
    triple-stack tightening, taking partial profits, and closing
    positions whose stop trips. Returns the surviving positions and the
    updated cash balance. ``closed_trades`` and ``diag`` are mutated in
    place.
    """
    surviving: list[_Position] = []
    for pos in positions:
        sdf = cache.get(pos.ticker)
        if sdf is None or wk not in sdf.index:
            surviving.append(pos)
            continue

        row = sdf.loc[wk]
        week_low = float(row["Low"])
        week_open = float(row["Open"])
        week_close = float(row["Close"])
        atr_val = row.get("atr_14")

        # ── Triple-stack tightening: pick the trailing-MA column ──
        use_ma_col = pos.exit_ma_col
        if pos.pos_triple_stack_enabled and atr_val is not None and not pd.isna(atr_val):
            atr_f = float(atr_val)
            holding_w = max(1, round((wk - pos.entry_date).days / 7))

            # (a) Stall — permanent once triggered.
            if pos.tighten_state == "stall":
                use_ma_col = pos.pos_stall_tighten_ma
            if pos.tighten_state is None and holding_w >= pos.pos_stall_tighten_week:
                curr_ret = (week_close - pos.entry_price) / pos.entry_price
                if curr_ret < pos.pos_stall_tighten_threshold:
                    pos.tighten_state = "stall"
                    use_ma_col = pos.pos_stall_tighten_ma
                    diag["triple_stall"] += 1

            # (b) Extension — transient (overrides stall this week only).
            baseline_ma = row.get(pos.exit_ma_col)
            if (
                baseline_ma is not None
                and not pd.isna(baseline_ma)
                and week_close > float(baseline_ma) + pos.pos_extension_atr_mult * atr_f
            ):
                use_ma_col = pos.pos_extension_tighten_ma
                diag["triple_extension"] += 1

            # (c) Climactic — permanent, highest priority.
            if pos.tighten_state == "climactic":
                use_ma_col = pos.pos_climactic_tighten_ma
            if (
                pos.tighten_state != "climactic"
                and pos.prev_close > 0
                and week_close < pos.prev_close
            ):
                vol_ratio = row.get("volume_ratio_20w")
                if (
                    vol_ratio is not None
                    and not pd.isna(vol_ratio)
                    and float(vol_ratio) > pos.pos_climactic_vol_threshold
                ):
                    pos.tighten_state = "climactic"
                    use_ma_col = pos.pos_climactic_tighten_ma
                    diag["triple_climactic"] += 1

            pos.prev_close = week_close

        # ── Ratchet trailing stop ──
        ma_val = row.get(use_ma_col)
        if (
            ma_val is not None
            and atr_val is not None
            and not pd.isna(ma_val)
            and not pd.isna(atr_val)
        ):
            new_trail = float(ma_val) - pos.exit_atr_buffer * float(atr_val)
            pos.stop_level = max(pos.stop_level, new_trail)

        # ── Progressive trail tightening ──
        if cfg.get("trail_tighten_enabled") and atr_val is not None and not pd.isna(atr_val):
            unrealized = (
                (week_close - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            )
            if unrealized >= cfg["trail_tighten_trigger_pct"] and not pos.trail_tightened:
                pos.trail_tightened = True
                diag["trail_tighten_triggered"] += 1
            if pos.trail_tightened:
                tt_col = f"{cfg['trail_tighten_ma_type'].lower()}_{cfg['trail_tighten_ma_period']}"
                tt_ma = row.get(tt_col)
                if tt_ma is not None and not pd.isna(tt_ma):
                    tt_trail = float(tt_ma) - cfg["trail_tighten_atr_buffer"] * float(atr_val)
                    pos.stop_level = max(pos.stop_level, tt_trail)

        # MAE tracking (always — even on exit week).
        pos.min_low = min(pos.min_low, week_low)

        # ── Optional time stop ──
        forced_exit = False
        forced_exit_reason = ""
        if cfg.get("time_stop_enabled"):
            hold_ts = max(1, round((wk - pos.entry_date).days / 7))
            unrealized_ts = (
                (week_close - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            )
            if (
                hold_ts >= cfg["time_stop_weeks"]
                and unrealized_ts < cfg["time_stop_min_return_pct"]
                and not pos.time_stop_triggered
            ):
                action = cfg["time_stop_action"]
                if action == "EXIT":
                    forced_exit = True
                    forced_exit_reason = "time_stop_exit"
                    diag["time_stop_exits"] += 1
                elif action == "TIGHTEN":
                    tightened = pos.entry_price * (1 - cfg["time_stop_tighten_pct"] / 100)
                    pos.stop_level = max(pos.stop_level, tightened)
                    pos.time_stop_triggered = True
                    diag["time_stop_tightens"] += 1

        # ── Partial profit ──
        if cfg.get("partial_profit_enabled") and not forced_exit:
            unrealized_pp = (
                (week_close - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            )
            if unrealized_pp >= cfg["partial_profit_trigger_pct"] and not pos.partial_taken:
                sell_qty = math.floor(pos.qty * cfg["partial_profit_sell_pct"] / 100)
                if 0 < sell_qty < pos.qty:
                    pp_proceeds = _sell_proceeds(sell_qty, week_close, cfg)
                    cash += pp_proceeds
                    pos.partial_sale_proceeds = pp_proceeds
                    pos.partial_sale_qty = sell_qty
                    pos.partial_sale_price = week_close
                    pos.partial_sale_date = wk
                    pos.qty -= sell_qty
                    pos.partial_taken = True
                    move = cfg.get("partial_profit_move_stop", "NONE")
                    if move == "BREAKEVEN":
                        pos.stop_level = max(pos.stop_level, pos.entry_price)
                    elif move == "HALF_BACK":
                        half_back = pos.entry_price + (pos.peak_close - pos.entry_price) / 2
                        pos.stop_level = max(pos.stop_level, half_back)
                    diag["partial_profit_taken"] += 1

        # ── GTT trigger or forced exit ──
        if forced_exit:
            cash, trade = _close_position(
                pos, week_close, wk, forced_exit_reason, cfg, regime_df, regime_col, cash
            )
            closed_trades.append(trade)
            diag["exits"] += 1
        else:
            trigger_val = week_low if cfg["exit_gtt_field"] == "low" else week_close
            if trigger_val <= pos.stop_level:
                if week_open < pos.stop_level:
                    exit_price = week_open
                    exit_reason = "gap_down"
                else:
                    exit_price = pos.stop_level
                    exit_reason = "gtt_stop"
                cash, trade = _close_position(
                    pos, exit_price, wk, exit_reason, cfg, regime_df, regime_col, cash
                )
                closed_trades.append(trade)
                diag["exits"] += 1
            else:
                if week_close > pos.peak_close:
                    pos.peak_close = week_close
                    pos.mfe_week = wk
                pos.current_close = week_close
                surviving.append(pos)

    return surviving, cash


def _close_position(
    pos: _Position,
    exit_price: float,
    wk: pd.Timestamp,
    exit_reason: str,
    cfg: Mapping[str, Any],
    regime_df: pd.DataFrame,
    regime_col: str,
    cash: float,
) -> tuple[float, dict[str, Any]]:
    """Common close-position bookkeeping; returns updated cash and trade record."""
    rem_proceeds = _sell_proceeds(pos.qty, exit_price, cfg)
    cash += rem_proceeds
    proceeds = rem_proceeds + pos.partial_sale_proceeds
    trade = _build_trade(pos, exit_price, wk, exit_reason, proceeds)

    exit_regime = "unknown"
    if wk in regime_df.index and regime_col in regime_df.columns:
        val = regime_df.loc[wk, regime_col]
        if not pd.isna(val):
            exit_regime = str(val)
    trade["regime_at_exit"] = exit_regime
    trade["partial_sale_qty"] = pos.partial_sale_qty
    trade["partial_sale_price"] = (
        round(pos.partial_sale_price, 2) if pos.partial_sale_price > 0 else None
    )
    trade["partial_sale_proceeds"] = (
        round(pos.partial_sale_proceeds, 2) if pos.partial_sale_proceeds > 0 else None
    )
    return cash, trade


# ── Entry selection ──────────────────────────────────────────────────


def select_entries_for_week(
    wk: pd.Timestamp,
    positions: list[_Position],
    cash: float,
    equity_base: float,
    entries_by_week: Mapping[pd.Timestamp, pd.DataFrame],
    cfg: Mapping[str, Any],
    regime_df: pd.DataFrame,
    regime_col: str,
    breadth_col: str,
    exit_ma_col: str,
    diag: Counter[str],
    *,
    sector_regime_lookup: dict[tuple[str, str], str] | None = None,
    captier_regime_lookup: dict[tuple[str, str], str] | None = None,
    ticker_metadata_lookup: dict[str, dict[str, Any]] | None = None,
    sector_rs_lookup: dict[tuple[str, str], str] | None = None,
) -> tuple[list[_Position], float]:
    """Per-week entry pipeline.

    Filters candidates through the regime gate, sector RS filter,
    sector + total-position constraints, then sizes and opens new
    positions. Returns the updated portfolio (held + new) and updated
    cash. ``diag`` is mutated in place.
    """
    week_entries = entries_by_week.get(wk)
    if week_entries is None or len(week_entries) == 0:
        return positions, cash

    regime, breadth_pct = _lookup_regime(regime_df, regime_col, breadth_col, wk)
    overall_allowed = is_entry_allowed(regime, breadth_pct, cfg)
    use_multi = cfg.get("regime_combination", "OVERALL_ONLY") != "OVERALL_ONLY"

    # Fast path: OVERALL_ONLY blocks everything if regime says no.
    if not use_multi and not overall_allowed:
        diag["regime_blocked"] += len(week_entries)
        return positions, cash

    held_tickers = {p.ticker for p in positions}
    sector_counts = Counter(p.sector for p in positions)

    candidates = week_entries[~week_entries["ticker"].isin(held_tickers)].sort_values(
        cfg.get("ranking_method", "rs_13w"), ascending=False, na_position="last"
    )

    sr_filt = cfg.get("sector_rs_filter", "NONE")
    sr_lookup = sector_rs_lookup or {}
    tm_lookup = ticker_metadata_lookup or {}
    sec_regime_lookup = sector_regime_lookup or {}
    cap_regime_lookup = captier_regime_lookup or {}
    wk_str = str(wk)[:10]

    for _, entry_row in candidates.iterrows():
        if len(positions) >= cfg["max_positions"]:
            diag["max_pos"] += 1
            break

        ticker = str(entry_row["ticker"])
        sector = str(entry_row.get("sector", "UNKNOWN"))

        # Multi-regime gate (per-ticker).
        if use_multi:
            meta = tm_lookup.get(ticker)
            if meta is None:
                diag["regime_blocked"] += 1
                continue
            idx_sym = meta.get("index_symbol")
            cap_tier = meta.get("cap_tier", "MICRO")
            has_data = meta.get("sector_has_index_data", False)
            s_regime = (
                sec_regime_lookup.get((wk_str, idx_sym), "sideways") if idx_sym else "sideways"
            )
            c_regime = cap_regime_lookup.get((wk_str, cap_tier), "sideways")
            if not is_entry_allowed_multi_regime(
                overall_allowed, s_regime, c_regime, has_data, cfg
            ):
                diag["regime_blocked"] += 1
                continue

        # Sector RS quartile filter.
        if sr_filt != "NONE":
            meta = tm_lookup.get(ticker, {})
            idx_sym = meta.get("index_symbol")
            quartile = sr_lookup.get((wk_str, idx_sym)) if idx_sym else None
            if not passes_sector_rs_filter(quartile, cfg):
                diag["rs_filter_blocked"] = diag.get("rs_filter_blocked", 0) + 1
                continue

        # Volume gate.
        if cfg.get("volume_min_ratio") is not None:
            vr = entry_row.get("volume_ratio")
            if pd.notna(vr) and float(vr) < cfg["volume_min_ratio"]:
                diag["volume_blocked"] = diag.get("volume_blocked", 0) + 1
                continue

        # Sector concentration.
        if sector_counts.get(sector, 0) >= cfg["sector_limit"]:
            diag["sector_limit"] += 1
            continue

        ep = float(entry_row["entry_price"])
        sl = float(entry_row["stop_level"])
        # Optional structural stop override for breakouts.
        if cfg.get("stop_method") == "STRUCTURAL":
            entry_type = str(entry_row["entry_type"])
            if entry_type.startswith("BREAKOUT"):
                ceiling = entry_row.get("ceiling")
                atr = entry_row.get("atr_at_entry")
                if pd.notna(ceiling) and pd.notna(atr) and float(atr) > 0:
                    buf = cfg.get("stop_atr_buffer", 1.0)
                    if cfg.get("stop_atr_by_captier"):
                        cap_tier = tm_lookup.get(ticker, {}).get("cap_tier", "MID")
                        buf *= _CAPTIER_BUFFER_MULT.get(cap_tier, 1.0)
                    structural = float(ceiling) - buf * float(atr)
                    if 0 < structural < ep:
                        sl = structural

        rr = float(entry_row["rr_ratio"]) if not pd.isna(entry_row["rr_ratio"]) else 0
        sizing = compute_sizing(ep, sl, rr, equity_base, cash, cfg)
        if sizing is None:
            diag["sizing_fail"] += 1
            continue

        rs_val = float(entry_row["rs_13w"]) if not pd.isna(entry_row["rs_13w"]) else 0
        risk_val = (
            float(entry_row["risk_pct_at_entry"])
            if not pd.isna(entry_row["risk_pct_at_entry"])
            else 0
        )
        target_type = str(entry_row.get("target_type", "structural"))
        pos = _Position(
            ticker=ticker,
            sector=sector,
            entry_date=wk,
            entry_price=ep,
            qty=sizing["qty"],
            cost=sizing["entry_cost"],
            stop_level=sl,
            tier=sizing["tier"],
            peak_close=ep,
            entry_type=str(entry_row["entry_type"]),
            target_type=target_type,
            risk_pct_at_entry=risk_val,
            rs_13w_at_entry=rs_val,
            regime_at_entry=regime,
            rr_at_entry=rr,
            current_close=ep,
            stop_at_entry=sl,
            min_low=ep,
            mfe_week=wk,
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

    return positions, cash


def _lookup_regime(
    regime_df: pd.DataFrame,
    regime_col: str,
    breadth_col: str,
    wk: pd.Timestamp,
) -> tuple[str, float]:
    """Read overall regime and breadth from ``regime_df`` for week ``wk``."""
    regime = "unknown"
    breadth_pct = 0.0
    if wk in regime_df.index:
        r = regime_df.loc[wk]
        if regime_col in r.index:
            regime = str(r[regime_col])
        if breadth_col in r.index:
            breadth_pct = float(r[breadth_col]) if not pd.isna(r[breadth_col]) else 0.0
    return regime, breadth_pct
