"""Per-ticker entry detection orchestrator.

Given a single ticker's weekly OHLCV plus the Nifty close series and
the strategy config, this module computes every indicator the
detectors need (ATR, MAs per detector, RS, swings at orders 5 and 8,
stages, trendline cache) and runs the five PHASE_4_V1 entry detectors,
returning a list of enriched entry events.

The wider research codebase computes 8 MA configurations × 5 detectors
in a grid; the OSS package only needs the single MA per detector that
:data:`PHASE_4_V1.entry_mas_per_type` specifies, so we compute strictly
what's required.

Public surface (private):

* :func:`find_entries_for_ticker` — the main per-ticker entry point.

The result of this module feeds directly into
:func:`skysurf._internal.engine.select_entries_for_week` (after dedup
+ ranking by type-prior).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from skysurf._internal.detectors import (
    compute_entry_quality,
    detect_breakouts,
    detect_pullbacks,
    detect_retest_support,
    detect_trendline_bounce,
    detect_vcp_continuations,
    precompute_trendlines,
)
from skysurf._internal.stages import (
    classify_stages_single_ma,
    compute_ma_slope_direction,
    count_consecutive_stage2,
)
from skysurf._internal.swings import detect_swings_argrelextrema
from skysurf.config import StrategyConfig
from skysurf.indicators import calculate_atr, calculate_rsi, compute_ma_series

_LOG = logging.getLogger(__name__)

#: Minimum weekly bars required to attempt detection (one year of data).
MIN_WEEKLY_BARS: int = 52

#: Per-detector swing-order assignment, mirroring research's
#: ``DETECTOR_SWING`` constants.
_DETECTOR_SWING_MAJOR: int = 8
_DETECTOR_SWING_MINOR: int = 5


def find_entries_for_ticker(
    ticker: str,
    weekly_ohlcv: pd.DataFrame,
    nifty_close_weekly: pd.Series,
    sector: str,
    config: StrategyConfig,
) -> list[dict[str, Any]]:
    """Detect every PHASE_4_V1 entry signal for one ticker.

    Args:
        ticker: Ticker symbol.
        weekly_ohlcv: Weekly OHLCV DataFrame indexed by Friday close,
            with columns ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume``. Must have at least :data:`MIN_WEEKLY_BARS` rows.
        nifty_close_weekly: Nifty 50 weekly close series (used for the
            relative-strength computation).
        sector: Sector label for this ticker.
        config: Strategy configuration; the per-detector MA pairs come
            from ``config.entry_mas_per_type``.

    Returns:
        List of entry-event dicts in the format produced by
        :func:`skysurf._internal.detectors.compute_entry_quality`.
        Returns ``[]`` when there isn't enough history.
    """
    if len(weekly_ohlcv) < MIN_WEEKLY_BARS:
        return []

    weekly = weekly_ohlcv.copy()
    n = len(weekly)

    # ── Indicators ───────────────────────────────────────────────
    atr_series = calculate_atr(weekly["High"], weekly["Low"], weekly["Close"], window=14)

    # Compute exactly the (ma_type, period) combos needed by
    # config.entry_mas_per_type. Each detector consumes its own MA.
    # The cast keeps mypy happy: config.entry_mas_per_type values are
    # ``tuple[str, int]`` but compute_ma_series wants Literal["SMA","EMA"].
    needed_ma_combos: set[tuple[str, int]] = set(config.entry_mas_per_type.values())
    ma_dict: dict[tuple[str, int], pd.Series] = {}
    for mt, mp in needed_ma_combos:
        if mt not in ("SMA", "EMA"):
            raise ValueError(f"entry_mas_per_type contains invalid MA type: {mt!r}")
        ma_dict[(mt, mp)] = compute_ma_series(weekly["Close"], mt, mp)  # type: ignore[arg-type]

    # Relative strength vs Nifty (13 weeks).
    stock_ret = weekly["Close"] / weekly["Close"].shift(13) - 1
    nifty_aligned = nifty_close_weekly.reindex(weekly.index, method="nearest")
    nifty_ret = nifty_aligned / nifty_aligned.shift(13) - 1
    rs_13w = stock_ret / nifty_ret.replace(0, np.nan)

    # RSI (used for downstream filters; not by all detectors).
    rsi_series = calculate_rsi(weekly["Close"], window=14)

    # Volume ratio (20-week).
    vol_20w = weekly["Volume"].rolling(20).mean()
    vol_ratio_series = weekly["Volume"] / vol_20w.replace(0, np.nan)

    # Stage classification per MA combo we computed.
    stages_per_ma: dict[tuple[str, int], pd.Series] = {}
    for combo, ma in ma_dict.items():
        direction = compute_ma_slope_direction(ma, atr_series)
        stages_per_ma[combo] = classify_stages_single_ma(weekly, ma, direction)

    # Swings at order 5 (minor) and order 8 (major). Lows are used for
    # pullback / VCP / trendline detection; highs for retest / breakouts.
    major_h, _major_l = detect_swings_argrelextrema(weekly, order=_DETECTOR_SWING_MAJOR)
    minor_h, minor_l = detect_swings_argrelextrema(weekly, order=_DETECTOR_SWING_MINOR)

    # Trendline cache (interval=52 to keep runtime reasonable).
    swing_lows_data: list[dict[str, Any]] = [
        {
            "level": float(weekly["Low"].iloc[idx]),
            "date": str(weekly.index[idx].date()),
        }
        for idx in minor_l
        if idx < n
    ]
    tl_cache = precompute_trendlines(weekly, atr_series, swing_lows_data, interval=52)

    # ── Per-detector dispatch ────────────────────────────────────
    events: list[dict[str, Any]] = []

    for entry_type, ma_combo in config.entry_mas_per_type.items():
        if entry_type in config.suspended_entry_types:
            continue

        stages = stages_per_ma[ma_combo]
        ma = ma_dict[ma_combo]
        # NB: config.base_min_weeks_per_type[entry_type] is consumed inside
        # the engine's filter chain (filter_phase_4_entries), not by the
        # detector itself — the detector receives its own stage-2 minimum.

        if entry_type == "BREAKOUT_S1_TO_S2":
            raw = detect_breakouts(
                weekly,
                stages,
                base_min_weeks=config.breakout_base_min_weeks,
                ceiling_def=config.breakout_ceiling_def,
                swing_high_indices=major_h,
                ma_series=ma,
                atr_series=atr_series,
            )
            for evt in raw:
                evt = compute_entry_quality(
                    weekly,
                    evt,
                    major_h,
                    minor_l,
                    atr_series,
                    base_start_idx=evt.get("base_start_idx"),
                )
                _enrich_event(
                    evt,
                    weekly,
                    ticker,
                    sector,
                    ma,
                    atr_series,
                    rs_13w,
                    rsi_series,
                    vol_ratio_series,
                    stages,
                )
                # Subtype split: PHASE_4_V1 keeps S1_TO_S2 only.
                weeks_in_s2 = count_consecutive_stage2(stages, evt["week_idx"])
                evt["weeks_in_stage2"] = weeks_in_s2
                if weeks_in_s2 < config.breakout_subtype_split_weeks:
                    evt["entry_type"] = "BREAKOUT_S1_TO_S2"
                else:
                    evt["entry_type"] = "BREAKOUT_S2_CONTINUATION"
                if evt["entry_type"] == "BREAKOUT_S1_TO_S2":
                    events.append(evt)

        elif entry_type == "PULLBACK_S2":
            raw = detect_pullbacks(
                weekly,
                stages,
                ma,
                atr_series,
                minor_l,
                min_stage2_duration=config.pullback_min_stage2,
                depth_def=config.pullback_depth_def,
                confirmation=config.pullback_confirmation,
            )
            for evt in raw:
                evt = compute_entry_quality(weekly, evt, major_h, minor_l, atr_series)
                _enrich_event(
                    evt,
                    weekly,
                    ticker,
                    sector,
                    ma,
                    atr_series,
                    rs_13w,
                    rsi_series,
                    vol_ratio_series,
                    stages,
                )
                events.append(evt)

        elif entry_type == "VCP_CONTINUATION":
            raw = detect_vcp_continuations(
                weekly,
                stages,
                atr_series,
                minor_h,
                minor_l,
                consol_min_weeks=config.vcp_consol_min_weeks,
                contraction_req=config.vcp_contraction_req,
                volume_req=config.vcp_volume_req,
            )
            for evt in raw:
                evt = compute_entry_quality(weekly, evt, major_h, minor_l, atr_series)
                _enrich_event(
                    evt,
                    weekly,
                    ticker,
                    sector,
                    ma,
                    atr_series,
                    rs_13w,
                    rsi_series,
                    vol_ratio_series,
                    stages,
                )
                events.append(evt)

        elif entry_type == "RETEST_SUPPORT":
            raw = detect_retest_support(weekly, stages, major_h, atr_series)
            for evt in raw:
                # The detector itself sets stop_atr_level; compute_entry_quality
                # fills target/rr but should NOT overwrite the stop.
                evt = compute_entry_quality(weekly, evt, major_h, minor_l, atr_series)
                _enrich_event(
                    evt,
                    weekly,
                    ticker,
                    sector,
                    ma,
                    atr_series,
                    rs_13w,
                    rsi_series,
                    vol_ratio_series,
                    stages,
                )
                events.append(evt)

        elif entry_type == "TRENDLINE_BOUNCE":
            raw = detect_trendline_bounce(weekly, stages, atr_series, tl_cache)
            for evt in raw:
                evt = compute_entry_quality(weekly, evt, major_h, minor_l, atr_series)
                _enrich_event(
                    evt,
                    weekly,
                    ticker,
                    sector,
                    ma,
                    atr_series,
                    rs_13w,
                    rsi_series,
                    vol_ratio_series,
                    stages,
                )
                events.append(evt)

    return events


def _enrich_event(
    evt: dict[str, Any],
    weekly: pd.DataFrame,
    ticker: str,
    sector: str,
    ma_series: pd.Series,
    atr_series: pd.Series,
    rs_13w: pd.Series,
    rsi_series: pd.Series,
    vol_ratio_series: pd.Series,
    stages_series: pd.Series,
) -> None:
    """Attach context fields onto an entry event in place.

    Adds: ``ticker``, ``sector``, ``week_date``, ``next_week_open``,
    ``rs_13w``, ``rsi_at_entry``, ``volume_ratio``, ``atr_at_entry``,
    ``ma_value``, ``weeks_in_stage2``, ``risk_pct_at_entry``,
    ``target_type``, ``proximity_atr`` (for retest), ``entry_price`` (=
    close at entry; the engine reads either ``entry_price`` or
    ``entry_price_close`` interchangeably).
    """
    idx = evt["week_idx"]
    week_date = weekly.index[idx]
    close = evt["close"]
    stop = evt.get("stop_level")

    evt["ticker"] = ticker
    evt["sector"] = sector
    evt["week_date"] = week_date
    evt["entry_price"] = close
    evt["entry_price_close"] = close
    evt["next_week_open"] = (
        float(weekly.iloc[idx + 1]["Open"]) if idx + 1 < len(weekly) else float("nan")
    )

    evt["rs_13w"] = (
        float(rs_13w.iloc[idx]) if idx < len(rs_13w) and not pd.isna(rs_13w.iloc[idx]) else None
    )
    evt["rsi_at_entry"] = (
        float(rsi_series.iloc[idx])
        if idx < len(rsi_series) and not pd.isna(rsi_series.iloc[idx])
        else None
    )
    if "volume_ratio" not in evt:
        evt["volume_ratio"] = (
            float(vol_ratio_series.iloc[idx])
            if idx < len(vol_ratio_series) and not pd.isna(vol_ratio_series.iloc[idx])
            else None
        )
    evt["volume_ratio_at_entry"] = evt.get("volume_ratio")
    evt["atr_at_entry"] = (
        float(atr_series.iloc[idx])
        if idx < len(atr_series) and not pd.isna(atr_series.iloc[idx])
        else None
    )
    evt["ma_value"] = (
        float(ma_series.iloc[idx])
        if idx < len(ma_series) and not pd.isna(ma_series.iloc[idx])
        else None
    )

    weeks_in_s2 = count_consecutive_stage2(stages_series, idx)
    evt["weeks_in_stage2"] = weeks_in_s2

    # Risk %.
    if stop is not None and close > 0:
        evt["risk_pct_at_entry"] = round((close - stop) / close, 4)
    else:
        evt["risk_pct_at_entry"] = None

    # Target type.
    target_level = evt.get("target_level")
    if target_level is not None and target_level == round(close * 1.20, 2):
        evt["target_type"] = "synthetic"
    else:
        evt["target_type"] = "structural"

    # Retest proximity (in ATR units).
    if (
        evt.get("entry_type") == "RETEST_SUPPORT"
        and evt.get("retest_level") is not None
        and evt.get("atr_at_entry") is not None
        and evt["atr_at_entry"] > 0
    ):
        evt["proximity_atr"] = round(
            abs(close - float(evt["retest_level"])) / float(evt["atr_at_entry"]), 2
        )
    else:
        evt["proximity_atr"] = None
