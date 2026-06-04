"""VCP continuation entry detection for Test 4."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_vcp_continuations(
    df: pd.DataFrame,
    stages_series: pd.Series,
    atr_series: pd.Series,
    minor_swing_highs: list[int],
    minor_swing_lows: list[int],
    consol_min_weeks: int,
    contraction_req: str,
    volume_req: str,
    min_stage2_duration: int = 8,
    min_gap_weeks: int = 4,
) -> list[dict]:
    """Detect VCP continuation entries within established Stage 2 trends.

    A VCP continuation is a consolidation within Stage 2 that shows
    contracting price ranges and (optionally) declining volume, followed
    by a breakout above the consolidation ceiling.

    Consolidation starts when the stock stops making new Stage 2 highs
    (tracked as the peak of the entire Stage 2 run, not a rolling window).

    Args:
        df: Weekly OHLCV DataFrame.
        stages_series: Per-bar stage labels.
        atr_series: 14-week ATR series.
        minor_swing_highs: Indices of A_order5 swing highs.
        minor_swing_lows: Indices of A_order5 swing lows.
        consol_min_weeks: Minimum consolidation duration.
        contraction_req: Contraction requirement.
        volume_req: Volume requirement.
        min_stage2_duration: Min weeks in Stage 2 before VCP.
        min_gap_weeks: Min weeks between VCP entries.

    Returns:
        List of event dicts with VCP-specific fields.
    """
    stages = stages_series.values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    atr = atr_series.values
    n = len(df)

    # Build consecutive Stage 2 counter + track Stage 2 peak
    consec = np.zeros(n, dtype=int)
    s2_peak = np.full(n, np.nan)
    for i in range(n):
        if stages[i] == "stage2":
            consec[i] = (consec[i - 1] + 1) if i > 0 else 1
            if i == 0 or consec[i] == 1:
                s2_peak[i] = float(highs[i])
            else:
                s2_peak[i] = max(s2_peak[i - 1], float(highs[i]))
        else:
            consec[i] = 0

    events: list[dict] = []
    last_entry_idx = -min_gap_weeks - 1
    i = 0

    while i < n:
        # Find bars deep enough in Stage 2
        if consec[i] < min_stage2_duration or pd.isna(s2_peak[i]):
            i += 1
            continue

        peak = s2_peak[i]

        # Check if consolidation starts here (no new Stage 2 high)
        if float(highs[i]) >= peak:
            i += 1
            continue

        # Count consecutive bars below the Stage 2 peak
        consol_start = i
        consol_end = i
        while consol_end < n and stages[consol_end] == "stage2":
            if float(highs[consol_end]) >= peak:
                # Made a new high — this is a breakout, not consolidation
                break
            consol_end += 1

        consol_len = consol_end - consol_start
        if consol_len < consol_min_weeks:
            i = consol_end
            continue

        # Check if the bar at consol_end is a breakout
        if consol_end >= n:
            break
        if stages[consol_end] != "stage2":
            # Consolidation ended without breakout (left Stage 2)
            i = consol_end
            continue
        if float(highs[consol_end]) < peak:
            i = consol_end
            continue

        # We have a breakout at consol_end: High >= peak
        breakout_idx = consol_end
        breakout_close = float(closes[breakout_idx])
        consol_ceiling = float(np.max(highs[consol_start:consol_end]))
        consol_low = float(np.min(lows[consol_start:consol_end]))

        # Check Close > ceiling for confirmation
        if breakout_close <= consol_ceiling:
            i = breakout_idx + 1
            continue

        # ── Contraction check ───────────────────────────────────
        # Find minor swings within consolidation window
        sh_in = [j for j in minor_swing_highs if consol_start <= j < consol_end]
        sl_in = [j for j in minor_swing_lows if consol_start <= j < consol_end]

        # Build contraction pairs (chronological high-low pairs)
        pairs = _build_contraction_pairs(df, sh_in, sl_in)
        depths = [(float(highs[h]) - float(lows[l])) / float(highs[h])
                  for h, l in pairs] if pairs else []

        contraction_count = len(depths)
        contraction_ok = _check_contraction_req(
            contraction_req, depths, df, breakout_idx, atr
        )
        if not contraction_ok:
            i = breakout_idx + 1
            continue

        # ── Volume check ────────────────────────────────────────
        consol_vols = volumes[consol_start:consol_end].astype(float)
        breakout_vol = float(volumes[breakout_idx])
        vol_slope = _volume_slope(consol_vols)
        consol_avg_vol = float(np.mean(consol_vols)) if len(consol_vols) > 0 else 1.0
        breakout_vol_ratio = breakout_vol / consol_avg_vol if consol_avg_vol > 0 else 0.0

        volume_ok = _check_volume_req(
            volume_req, vol_slope, breakout_vol_ratio
        )
        if not volume_ok:
            i = breakout_idx + 1
            continue

        # ── Gap check ───────────────────────────────────────────
        if breakout_idx - last_entry_idx < min_gap_weeks:
            i = breakout_idx + 1
            continue

        # ── Record entry ────────────────────────────────────────
        events.append({
            "week_idx": breakout_idx,
            "date": str(df.index[breakout_idx].date()),
            "close": round(breakout_close, 2),
            "entry_type": "VCP_CONTINUATION",
            "consolidation_start_idx": consol_start,
            "consolidation_weeks": consol_len,
            "consolidation_ceiling": round(consol_ceiling, 2),
            "consolidation_low": round(consol_low, 2),
            "contraction_count": contraction_count,
            "contraction_depths": [round(d, 4) for d in depths],
            "volume_slope": round(vol_slope, 4) if not np.isnan(vol_slope) else None,
            "breakout_volume_ratio": round(breakout_vol_ratio, 2),
        })
        last_entry_idx = breakout_idx
        # Continue scanning from the breakout bar
        i = breakout_idx + 1
        continue

        i += 1  # noqa — unreachable but defensive

    return events


# ── Helpers ────────────────────────────────────────────────────────────


def _build_contraction_pairs(
    df: pd.DataFrame,
    swing_high_indices: list[int],
    swing_low_indices: list[int],
) -> list[tuple[int, int]]:
    """Build chronological (high_idx, low_idx) pairs from minor swings."""
    # Merge and sort by index
    all_swings = [(idx, "H") for idx in swing_high_indices] + \
                 [(idx, "L") for idx in swing_low_indices]
    all_swings.sort(key=lambda x: x[0])

    pairs = []
    i = 0
    while i < len(all_swings) - 1:
        if all_swings[i][1] == "H" and all_swings[i + 1][1] == "L":
            pairs.append((all_swings[i][0], all_swings[i + 1][0]))
            i += 2
        else:
            i += 1
    return pairs


def _check_contraction_req(
    req: str,
    depths: list[float],
    df: pd.DataFrame,
    breakout_idx: int,
    atr: np.ndarray,
) -> bool:
    """Validate contraction requirement."""
    if req == "vcp_any":
        return True
    if req == "vcp_2plus":
        if len(depths) < 2:
            return False
        return all(depths[i] > depths[i + 1] for i in range(len(depths) - 1))
    if req == "vcp_tight_final":
        if len(depths) < 2:
            return False
        if not all(depths[i] > depths[i + 1] for i in range(len(depths) - 1)):
            return False
        # Final week range < 1.5 × ATR
        final_idx = breakout_idx - 1
        if final_idx < 0 or final_idx >= len(df) or pd.isna(atr[final_idx]):
            return False
        final_range = float(df["High"].iloc[final_idx]) - float(df["Low"].iloc[final_idx])
        return final_range < 1.5 * float(atr[final_idx])
    return True


def _check_volume_req(req: str, vol_slope: float, breakout_vol_ratio: float) -> bool:
    """Validate volume requirement."""
    if req == "vol_any":
        return True
    if req == "vol_declining":
        return vol_slope < 0
    if req == "vol_dry_breakout":
        return vol_slope < 0 and breakout_vol_ratio > 1.5
    return True


def _volume_slope(volumes: np.ndarray) -> float:
    """Linear regression slope of volume series."""
    if len(volumes) < 2:
        return 0.0
    x = np.arange(len(volumes), dtype=float)
    try:
        slope = float(np.polyfit(x, volumes, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return slope
