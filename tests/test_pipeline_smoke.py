"""Smoke tests for skysurf.reproduction.pipeline (the OHLCV->entries builder).

The import test runs everywhere. The end-to-end test generates a tiny synthetic
OHLCV set and runs the full pipeline (no DB, no real data), asserting it emits
entries_all.csv + entries_all_lagged.csv with the expected schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_build_subpackage_imports():
    import skysurf.reproduction.pipeline as build
    from skysurf.reproduction.pipeline import builder, lagged  # noqa: F401
    from skysurf.reproduction.pipeline._calc import (  # noqa: F401
        regime_classifier,
        skysurf_calc,
        technical_indicators,
    )
    from skysurf.reproduction.pipeline._detect import (  # noqa: F401
        breakout_detector,
        entry_quality,
        ma_configs,
        phase0_regime,
        pullback_detector,
        stage_classifier,
        swing_methods,
        vcp_detector,
    )

    assert callable(build.build)
    assert callable(build.build_lagged)


def _write_synthetic_ohlcv(ohlcv_dir):
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2008-01-01", "2024-12-31")

    def walk(n, start, drift, vol):
        c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        h = c * (1 + np.abs(rng.normal(0, 0.01, n)))
        low = c * (1 - np.abs(rng.normal(0, 0.01, n)))
        o = np.concatenate([[c[0]], c[:-1]])
        return o, h, low, c, rng.integers(1e5, 5e6, n)

    o, h, low, c, v = walk(len(days), 4000, 0.0004, 0.01)
    pd.DataFrame({"date": days, "open": o, "high": h, "low": low, "close": c, "volume": v}).to_csv(
        ohlcv_dir / "benchmark_daily.csv", index=False
    )

    secs = ["NIFTY IT", "NIFTY BANK", "NIFTY AUTO", "NIFTY PHARMA", "NIFTY FMCG"]
    rows = []
    for i in range(6):
        o, h, low, c, v = walk(len(days), 100 + i * 40, 0.0006, 0.02)
        rows.append(
            pd.DataFrame(
                {
                    "ticker": f"TEST{i}.NS", "date": days, "open": o, "high": h, "low": low,
                    "close": c, "volume": v, "primary_sector": secs[i % len(secs)],
                    "market_cap": 5e11, "has_demerger": False,
                    "demerger_no_trade_until": pd.NaT, "has_sme_history": False,
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(ohlcv_dir / "stocks_daily.csv", index=False)


def test_build_end_to_end_on_synthetic_ohlcv(tmp_path):
    import skysurf.reproduction.pipeline as build

    ohlcv = tmp_path / "ohlcv"
    ohlcv.mkdir()
    out = tmp_path / "bundle"
    _write_synthetic_ohlcv(ohlcv)

    build.build(ohlcv_dir=str(ohlcv), data_dir=str(out), lagged=True)

    for name in (
        "entries_all.csv", "entries_all_lagged.csv", "nifty_weekly.csv",
        "regime_weekly.csv", "universe_quarterly.csv",
    ):
        assert (out / name).exists(), f"builder did not produce {name}"

    entries = pd.read_csv(out / "entries_all.csv")
    assert len(entries) > 0
    for col in ("ticker", "week_date", "entry_type", "ma_type", "ma_period"):
        assert col in entries.columns
    assert (out / "stock_weekly_cache" / "TEST0.NS.csv").exists()
