"""Weinstein stage classification + vectorised MA slope direction.

The strategy uses Stan Weinstein's stage analysis to classify every
weekly bar of every stock into one of four stages:

* **Stage 1 (base / accumulation)** — flat MA, price tangled with MA
* **Stage 2 (advance)** — close above a rising MA
* **Stage 3 (top / distribution)** — flat MA after recent Stage 2
* **Stage 4 (decline)** — close below a falling MA

A stock can cycle through stages non-sequentially (Stage 1 → Stage 2
→ Stage 1 → Stage 2) — the strategy detects pullbacks and breakouts
within Stage 2 specifically, so accurate classification matters.

The classifier needs a per-bar slope-direction signal for the MA,
which is computed by :func:`compute_ma_slope_direction` (the vectorised
counterpart to :func:`skysurf._internal.regime.compute_ema_direction`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skysurf._internal.regime import EMA_DIRECTION_THRESHOLD


def compute_ma_slope_direction(
    ma_series: pd.Series,
    atr_series: pd.Series,
    lookback: int = 4,
    threshold: float = EMA_DIRECTION_THRESHOLD,
) -> pd.Series:
    """Classify the MA slope at every bar as rising / falling / flat.

    Vectorised counterpart to
    :func:`skysurf._internal.regime.compute_ema_direction`, which only
    operates on the tail.

    Per-bar logic::

        slope_norm = (ma[i] - ma[i - lookback]) / atr[i]
        > threshold  → "rising"
        < -threshold → "falling"
        else         → "flat"

    Args:
        ma_series: Pre-computed MA values (SMA or EMA).
        atr_series: Weekly ATR values (from
            :func:`skysurf.indicators.calculate_atr`).
        lookback: Weeks to look back for the slope. Defaults to 4.
        threshold: ATR-normalised threshold. Defaults to
            :data:`EMA_DIRECTION_THRESHOLD` (0.3).

    Returns:
        ``pd.Series`` of ``"rising"`` / ``"falling"`` / ``"flat"``,
        aligned with ``ma_series.index``. Bars where the slope cannot
        be computed (insufficient lookback or NaN ATR) are ``"flat"``.
    """
    result = pd.Series("flat", index=ma_series.index, dtype=object)

    if len(ma_series) <= lookback:
        return result

    ma_now = ma_series.iloc[lookback:]
    ma_ago = ma_series.iloc[:-lookback].to_numpy()
    atr_now = atr_series.iloc[lookback:]

    safe_atr = atr_now.replace(0, np.nan)
    slope_norm = (ma_now.to_numpy() - ma_ago) / safe_atr.to_numpy()

    directions = np.where(
        slope_norm > threshold,
        "rising",
        np.where(slope_norm < -threshold, "falling", "flat"),
    )
    # numpy's tolist() narrows ndarray[Any] to list[str], which pandas iloc
    # accepts directly; .values assignment trips a strict mypy overload check.
    result.iloc[lookback:] = directions.tolist()
    return result


def classify_stages_single_ma(
    df: pd.DataFrame,
    ma_series: pd.Series,
    direction_series: pd.Series,
) -> pd.Series:
    """Classify each weekly bar into one of four Weinstein stages.

    Stage rules (evaluated in order):

    * **Stage 2 (Advance)** — ``Close > MA`` and ``direction == "rising"``.
    * **Stage 4 (Decline)** — ``Close < MA`` and ``direction == "falling"``.
    * **Stage 3 (Top / Distribution)** — ``direction == "flat"`` and the
      bar was in Stage 2 within the last 8 bars.
    * **Stage 1 (Base / Accumulation)** — ``direction == "flat"`` and
      not Stage 3.
    * **Ambiguous** (e.g., close above MA but slope falling) — carry
      forward the previous stage.

    Stages are *not* forced sequential. A stock can cycle Stage 1 →
    Stage 2 → Stage 1 → Stage 2 (failed breakout, re-base, successful
    breakout).

    Args:
        df: Weekly OHLCV DataFrame; must contain a ``Close`` column.
        ma_series: Pre-computed MA values, aligned with ``df.index``.
        direction_series: Per-bar slope direction (one of ``"rising"``,
            ``"falling"``, ``"flat"``), typically from
            :func:`compute_ma_slope_direction`.

    Returns:
        ``pd.Series`` of stage labels (``"stage1"``, ``"stage2"``,
        ``"stage3"``, ``"stage4"``), index-aligned with ``df``.
    """
    n = len(df)
    stages: list[str] = ["stage1"] * n
    close = df["Close"].to_numpy()
    ma = ma_series.to_numpy()
    dirs = direction_series.to_numpy()

    for i in range(n):
        # MA not yet computed (insufficient warm-up): treat as base.
        if pd.isna(ma[i]):
            stages[i] = "stage1"
            continue

        d = dirs[i]
        if close[i] > ma[i] and d == "rising":
            stages[i] = "stage2"
        elif close[i] < ma[i] and d == "falling":
            stages[i] = "stage4"
        elif d == "flat":
            lookback_start = max(0, i - 8)
            was_stage2 = any(stages[j] == "stage2" for j in range(lookback_start, i))
            stages[i] = "stage3" if was_stage2 else "stage1"
        else:
            # Ambiguous (close above MA but slope falling, or close
            # below MA but slope rising). Carry forward.
            stages[i] = stages[i - 1] if i > 0 else "stage1"

    return pd.Series(stages, index=df.index, dtype=object)


def count_consecutive_stage2(stages_series: pd.Series, idx: int) -> int:
    """Return the run-length of consecutive Stage 2 bars ending at ``idx``.

    Counts bars at ``idx``, ``idx - 1``, ``idx - 2``, … as long as each
    is labelled ``"stage2"``. Returns 0 if ``stages_series.iloc[idx]``
    is not Stage 2.

    Args:
        stages_series: Output of :func:`classify_stages_single_ma`.
        idx: Positional index into ``stages_series``.

    Returns:
        Run length (≥ 0).
    """
    if idx < 0 or idx >= len(stages_series):
        return 0

    count = 0
    for i in range(idx, -1, -1):
        if stages_series.iloc[i] == "stage2":
            count += 1
        else:
            break
    return count
