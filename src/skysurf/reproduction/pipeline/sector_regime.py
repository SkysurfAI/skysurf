"""Sector / cap-tier regime + ticker-metadata builder (OHLCV -> bundle CSVs).

Open-sourced from the research ``compute_regime_dimensions.py`` +
``add_sector_rs.py``, refactored only so the data comes from **files** instead
of SQL. The regime classification (Weinstein 2-signal on the 20-week EMA) and
the 13-week relative-strength ranking are byte-identical to the research code.

It closes the last gap in the ``skysurf-build`` chain: the three bundle inputs
``skysurf-reproduce`` still needed supplied directly —
``sector_regime_weekly.csv``, ``captier_regime_weekly.csv``,
``ticker_metadata.csv`` (plus the diagnostic ``sector_breadth_weekly.csv``) —
now regenerate from raw OHLCV.

Extra raw input (beyond ``stocks_daily`` / ``benchmark_daily``):

    sector_daily.parquet / .csv     # per-index daily OHLCV for the NIFTY sector
                                     # and cap-tier indices (and NIFTY 50, used
                                     # as the RS benchmark)
        columns: index_symbol, trade_date, open, high, low, close, volume

Stock sector/market-cap come from ``stocks_daily`` (``primary_sector`` +
``market_cap`` columns); the per-sector breadth reads ``stock_weekly_cache/``
from the data bundle (so run ``skysurf-build`` first).

Usage::

    skysurf-build-sectors --ohlcv /path/to/raw_ohlcv --data /path/to/skysurf-repro-data
    python -m skysurf.reproduction.pipeline.sector_regime --ohlcv ... --data ...
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from .. import _paths
from ._calc.skysurf_calc import resample_weekly

# ── Paths (placeholders resolved at import; re-resolved in build_*()) ─────────
OUTPUT_DIR = _paths.data_dir()
STOCK_CACHE_DIR = OUTPUT_DIR / "stock_weekly_cache"

# ── Constants (match regime_classifier.py exactly) ───────────────────────
EMA_SPAN = 20
ATR_WINDOW = 14
EMA_DIRECTION_THRESHOLD = 0.3
EMA_POSITION_LOOKBACK = 4
EMA_DIRECTION_LOOKBACK = 4
MIN_WEEKLY_ROWS = 52  # need ≥1 year of weekly data before regime is meaningful
RS_LOOKBACK = 13  # weeks

# Sector index start-date cutoff: must start 2007 or earlier for Tier A
TIER_A_CUTOFF = pd.Timestamp("2007-07-01")

# ── primary_sector → index_symbol mapping ────────────────────────────────
# Sourced from app/services/sector_resolution_service.py INDEX_SYMBOL_MAP
FULL_NAME_TO_DB_SYMBOL = {
    "NIFTY 500": "NIFTY 500",
    "NIFTY CAPITAL MARKETS": "NIFTY CAPITAL MKT",
    "NIFTY MIDCAP 150": "NIFTY MIDCAP 150",
    "NIFTY MIDSMALL FINANCIAL SERVICES": "NIFTY MS FIN SERV",
    "NIFTY SMALLCAP 250": "NIFTY SMLCAP 250",
    "NIFTY CHEMICALS": "NIFTY CHEMICALS",
    "NIFTY ENERGY": "NIFTY ENERGY",
    "NIFTY INDIA MANUFACTURING": "NIFTY INDIA MFG",
    "NIFTY NEXT 50": "NIFTY NEXT 50",
    "NIFTY HEALTHCARE INDEX": "NIFTY HEALTHCARE",
    "NIFTY MIDSMALL HEALTHCARE": "NIFTY MIDSML HLTH",
    "NIFTY PHARMA": "NIFTY PHARMA",
    "NIFTY CORE HOUSING": "NIFTY COREHOUSING",
    "NIFTY COMMODITIES": "NIFTY COMMODITIES",
    "NIFTY 50": "NIFTY 50",
    "NIFTY METAL": "NIFTY METAL",
    "NIFTY INFRASTRUCTURE": "NIFTY INFRA",
    "NIFTY TRANSPORTATION & LOGISTICS": "NIFTY TRANS LOGIS",
    "NIFTY INDIA CONSUMPTION": "NIFTY CONSUMPTION",
    "NIFTY OIL & GAS": "NIFTY OIL AND GAS",
    "NIFTY MIDSMALL IT & TELECOM": "NIFTY MS IT TELCM",
    "NIFTY CONSUMER DURABLES": "NIFTY CONSR DURBL",
    "NIFTY REALTY": "NIFTY REALTY",
    "NIFTY EV & NEW AGE AUTOMOTIVE": "NIFTY EV",
    "NIFTY AUTO": "NIFTY AUTO",
    "NIFTY INDIA DEFENCE": "NIFTY IND DEFENCE",
    "NIFTY MICROCAP 250": "NIFTY MICROCAP250",
    "NIFTY BANK": "NIFTY BANK",
    "NIFTY FINANCIAL SERVICES": "NIFTY FIN SERVICE",
    "NIFTY PRIVATE BANK": "NIFTY PVT BANK",
    "NIFTY PSU BANK": "NIFTY PSU BANK",
    "NIFTY CPSE": "NIFTY CPSE",
    "NIFTY PSE": "NIFTY PSE",
    "NIFTY FMCG": "NIFTY FMCG",
    "NIFTY IT": "NIFTY IT",
    "NIFTY MEDIA": "NIFTY MEDIA",
}

DB_SYMBOL_TO_FULL_NAME = {v: k for k, v in FULL_NAME_TO_DB_SYMBOL.items()}

CAP_TIER_INDICES = {
    "NIFTY 50": "LARGE",
    "NIFTY NEXT 50": "LARGE100",
    "NIFTY MIDCAP 150": "MID",
    "NIFTY SMLCAP 250": "SMALL",
    "NIFTY MICROCAP250": "MICRO",
}


# ══════════════════════════════════════════════════════════════════════════
# RAW-OHLCV READERS (replace the SQL reads; identical row semantics)
# ══════════════════════════════════════════════════════════════════════════

def _load_sector_daily(ohlcv_dir: Path) -> pd.DataFrame:
    """Read ``sector_daily`` (.parquet preferred, .csv accepted) from ohlcv_dir."""
    pq = ohlcv_dir / "sector_daily.parquet"
    csv = ohlcv_dir / "sector_daily.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"Sector index OHLCV not found: expected 'sector_daily.parquet' or "
            f"'sector_daily.csv' in {ohlcv_dir}. Columns: index_symbol, trade_date, "
            "open, high, low, close, volume."
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _index_daily(sector_daily: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Equivalent of: SELECT trade_date as date, OHLCV FROM sector_historical_data
    WHERE index_symbol = :sym ORDER BY trade_date."""
    sub = sector_daily[sector_daily["index_symbol"] == sym].copy()
    if sub.empty:
        return sub
    sub = sub.rename(columns={"trade_date": "date"})
    return sub[["date", "open", "high", "low", "close", "volume"]].sort_values("date")


def _load_stocks_meta(ohlcv_dir: Path) -> pd.DataFrame:
    """Read ``stocks_daily`` ticker/primary_sector/market_cap from ohlcv_dir."""
    pq = ohlcv_dir / "stocks_daily.parquet"
    csv = ohlcv_dir / "stocks_daily.csv"
    if pq.exists():
        return pd.read_parquet(pq, columns=["ticker", "primary_sector", "market_cap"])
    if csv.exists():
        return pd.read_csv(csv, usecols=["ticker", "primary_sector", "market_cap"])
    raise FileNotFoundError(
        f"stocks_daily not found in {ohlcv_dir} (needed for primary_sector + market_cap)."
    )


# ══════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION — replicated from app/utils/regime_classifier.py
# ══════════════════════════════════════════════════════════════════════════

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  window: int = ATR_WINDOW) -> pd.Series:
    """ATR via ta library — same as app/utils/technical_indicators.py."""
    if len(close) < window:
        return pd.Series([float("nan")] * len(close), index=close.index)
    return AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()


def compute_ema_position(
    weekly_closes: pd.Series,
    ema_series: pd.Series,
    lookback_weeks: int = EMA_POSITION_LOOKBACK,
) -> str:
    """Return 'above', 'below', or 'at_ema' — 4-week consistency check."""
    n = min(lookback_weeks, len(weekly_closes), len(ema_series))
    if n == 0:
        return "at_ema"
    closes_tail = weekly_closes.iloc[-n:]
    ema_tail = ema_series.iloc[-n:]
    weeks_above = int((closes_tail > ema_tail).sum())
    threshold_above = n * 0.75
    threshold_below = n * 0.25
    if weeks_above >= threshold_above:
        return "above"
    elif weeks_above <= threshold_below:
        return "below"
    return "at_ema"


def compute_ema_direction(
    ema_series: pd.Series,
    atr_value: float,
    lookback_weeks: int = EMA_DIRECTION_LOOKBACK,
) -> tuple[str, float]:
    """Return (direction, normalized_slope) — ATR-normalized EMA slope."""
    if len(ema_series) < lookback_weeks + 1 or atr_value is None:
        return "flat", 0.0
    ema_now = float(ema_series.iloc[-1])
    ema_ago = float(ema_series.iloc[-1 - lookback_weeks])
    slope_abs = ema_now - ema_ago
    if atr_value > 0:
        normalized_slope = round(slope_abs / atr_value, 2)
    else:
        normalized_slope = 0.0
    if normalized_slope > EMA_DIRECTION_THRESHOLD:
        return "rising", normalized_slope
    elif normalized_slope < -EMA_DIRECTION_THRESHOLD:
        return "falling", normalized_slope
    return "flat", normalized_slope


def classify_regime(ema_position: str, ema_direction: str) -> str:
    """Weinstein 2-signal classification (no breadth override for sector-level)."""
    if ema_position == "above" and ema_direction == "rising":
        return "bull"
    elif ema_position == "below" and ema_direction == "falling":
        return "bear"
    return "sideways"


def compute_index_regime(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Given daily OHLCV DataFrame, compute weekly Weinstein regime."""
    df = daily_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    weekly = resample_weekly(df)
    if len(weekly) < MIN_WEEKLY_ROWS:
        return pd.DataFrame()

    weekly["ema_20"] = weekly["Close"].ewm(span=EMA_SPAN, adjust=False).mean()
    weekly["atr_14"] = calculate_atr(weekly["High"], weekly["Low"], weekly["Close"], ATR_WINDOW)

    regimes = []
    for i in range(len(weekly)):
        closes_up_to = weekly["Close"].iloc[:i + 1]
        ema_up_to = weekly["ema_20"].iloc[:i + 1]
        atr_val = float(weekly["atr_14"].iloc[i]) if not pd.isna(weekly["atr_14"].iloc[i]) else 0.0

        pos = compute_ema_position(closes_up_to, ema_up_to)
        dirn, norm_slope = compute_ema_direction(ema_up_to, atr_val)
        regime = classify_regime(pos, dirn)

        regimes.append({
            "close": float(weekly["Close"].iloc[i]),
            "ema_20": float(weekly["ema_20"].iloc[i]),
            "atr_14": float(weekly["atr_14"].iloc[i]) if not pd.isna(weekly["atr_14"].iloc[i]) else np.nan,
            "ema_direction": norm_slope,
            "regime": regime,
        })

    result = pd.DataFrame(regimes, index=weekly.index)
    result.index.name = "week_date"
    return result


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT 1: sector_regime_weekly.csv
# ══════════════════════════════════════════════════════════════════════════

def build_sector_regime(sector_daily: pd.DataFrame, primary_sectors: list[str]) -> pd.DataFrame:
    print("=" * 70)
    print("OUTPUT 1: Sector regime weekly")
    print("=" * 70)

    print(f"  Primary sectors in stocks_daily: {len(primary_sectors)}")

    # Determine which sectors have index data and their start dates
    nifty = sector_daily[sector_daily["index_symbol"].str.startswith("NIFTY")]
    starts = nifty.groupby("index_symbol")["trade_date"].agg(["min", "count"])
    sector_index_info = {sym: {"start_date": row["min"], "rows": int(row["count"])}
                         for sym, row in starts.iterrows()}

    # Classify into tiers
    tier_a, tier_b, tier_c = [], [], []
    for ps in primary_sectors:
        db_sym = FULL_NAME_TO_DB_SYMBOL.get(ps)
        if not db_sym or db_sym not in sector_index_info:
            tier_c.append((ps, db_sym, None))
            continue
        start = pd.Timestamp(sector_index_info[db_sym]["start_date"])
        if start <= TIER_A_CUTOFF:
            tier_a.append((ps, db_sym, start))
        elif start < pd.Timestamp("2024-01-01"):
            tier_b.append((ps, db_sym, start))
        else:
            tier_c.append((ps, db_sym, start))

    print(f"  Tier A (full coverage): {len(tier_a)} sectors")
    print(f"  Tier B (partial): {len(tier_b)} sectors")
    print(f"  Tier C (skipped, post-2024): {len(tier_c)} sectors")
    for ps, sym, start in tier_c:
        print(f"    SKIP: {ps} ({sym}, start={start})")

    all_frames = []
    for ps, db_sym, start in tier_a + tier_b:
        df = _index_daily(sector_daily, db_sym)
        if df.empty:
            print(f"    WARN: No data for {db_sym}")
            continue
        df = df.set_index("date").sort_index()

        regime_df = compute_index_regime(df)
        if regime_df.empty:
            print(f"    WARN: Insufficient data for {db_sym} ({len(df)} daily rows)")
            continue

        regime_df["index_symbol"] = db_sym
        regime_df["primary_sector"] = ps
        all_frames.append(regime_df)
        print(f"    {ps:40s} ({db_sym:20s}) — {len(regime_df)} weeks, {start.date()}")

    result = pd.concat(all_frames).reset_index()
    result = result[["week_date", "index_symbol", "primary_sector", "close",
                      "ema_20", "atr_14", "ema_direction", "regime"]]
    result = result.sort_values(["index_symbol", "week_date"])
    print(f"\n  Sector regime rows: {len(result):,}, weeks: {result['week_date'].nunique()}, "
          f"sectors: {result['index_symbol'].nunique()}")
    print()
    return result


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT 2: captier_regime_weekly.csv
# ══════════════════════════════════════════════════════════════════════════

def build_captier_regime(sector_daily: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("=" * 70)
    print("OUTPUT 2: Cap-tier regime weekly")
    print("=" * 70)

    all_frames = []
    for db_sym, cap_tier in CAP_TIER_INDICES.items():
        df = _index_daily(sector_daily, db_sym)
        if df.empty:
            print(f"    WARN: No data for {db_sym}")
            continue
        df = df.set_index("date").sort_index()

        regime_df = compute_index_regime(df)
        if regime_df.empty:
            print(f"    WARN: Insufficient data for {db_sym}")
            continue

        regime_df["index_symbol"] = db_sym
        regime_df["cap_tier"] = cap_tier
        all_frames.append(regime_df)
        start = df.index[0].date()
        print(f"    {cap_tier:10s} ({db_sym:20s}) — {len(regime_df)} weeks, from {start}")

    result = pd.concat(all_frames).reset_index()
    result = result[["week_date", "index_symbol", "cap_tier", "close",
                      "ema_20", "atr_14", "ema_direction", "regime"]]
    result = result.sort_values(["index_symbol", "week_date"])

    out_path = out_dir / "captier_regime_weekly.csv"
    result.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Total rows: {len(result):,}, weeks: {result['week_date'].nunique()}")
    print()
    return result


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT 3: sector_breadth_weekly.csv (diagnostic; bottom-up from cache)
# ══════════════════════════════════════════════════════════════════════════

def build_sector_breadth(ticker_sector_map: dict[str, str], cache_dir: Path,
                         out_dir: Path) -> pd.DataFrame:
    print("=" * 70)
    print("OUTPUT 3: Sector breadth weekly (bottom-up from stock_weekly_cache)")
    print("=" * 70)

    sector_weekly_data: dict[str, list[tuple[str, pd.Timestamp, float, float]]] = {}
    files = sorted(cache_dir.glob("*.csv"))
    loaded = 0
    skipped_no_sector = 0

    for f in files:
        ticker = f.stem
        sector = ticker_sector_map.get(ticker)
        if sector is None:
            skipped_no_sector += 1
            continue
        try:
            df = pd.read_csv(f, usecols=["date", "Close", "sma_40"], parse_dates=["date"])
        except (ValueError, KeyError):
            continue
        df = df.dropna(subset=["Close"])
        for _, row in df.iterrows():
            if sector not in sector_weekly_data:
                sector_weekly_data[sector] = []
            sector_weekly_data[sector].append((
                ticker, row["date"], float(row["Close"]),
                float(row["sma_40"]) if not pd.isna(row["sma_40"]) else np.nan,
            ))
        loaded += 1

    print(f"  Loaded {loaded} stock files, skipped {skipped_no_sector} (no sector mapping)")

    breadth_rows = []
    all_data: dict[pd.Timestamp, dict] = {}
    for sector, records in sector_weekly_data.items():
        by_week: dict[pd.Timestamp, list[tuple[float, float]]] = {}
        for ticker, wdate, close, sma40 in records:
            by_week.setdefault(wdate, []).append((close, sma40))

        for wdate in sorted(by_week.keys()):
            pairs = by_week[wdate]
            stocks_in_sector = len(pairs)
            stocks_above_sma40 = sum(1 for close, sma40 in pairs
                                     if not np.isnan(sma40) and close > sma40)
            stocks_with_sma40 = sum(1 for _, sma40 in pairs if not np.isnan(sma40))
            pct = round(stocks_above_sma40 / stocks_with_sma40 * 100, 2) if stocks_with_sma40 > 0 else np.nan

            breadth_rows.append({
                "week_date": wdate, "sector": sector,
                "stocks_in_sector": stocks_in_sector,
                "stocks_with_sma40": stocks_with_sma40,
                "stocks_above_sma40": stocks_above_sma40,
                "sector_breadth_pct": pct,
                "sufficient_stocks": stocks_with_sma40 >= 10,
            })
            if wdate not in all_data:
                all_data[wdate] = {"above": 0, "total": 0, "count": 0}
            all_data[wdate]["above"] += stocks_above_sma40
            all_data[wdate]["total"] += stocks_with_sma40
            all_data[wdate]["count"] += stocks_in_sector

    for wdate in sorted(all_data.keys()):
        d = all_data[wdate]
        pct = round(d["above"] / d["total"] * 100, 2) if d["total"] > 0 else np.nan
        breadth_rows.append({
            "week_date": wdate, "sector": "OVERALL",
            "stocks_in_sector": d["count"], "stocks_with_sma40": d["total"],
            "stocks_above_sma40": d["above"], "sector_breadth_pct": pct,
            "sufficient_stocks": True,
        })

    result = pd.DataFrame(breadth_rows).sort_values(["sector", "week_date"])
    out_path = out_dir / "sector_breadth_weekly.csv"
    result.to_csv(out_path, index=False)
    n_sectors = result[result["sector"] != "OVERALL"]["sector"].nunique()
    print(f"  Sectors with breadth: {n_sectors}, weeks: {result['week_date'].nunique()}")
    print(f"  Saved: {out_path}\n")
    return result


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT 4: ticker_metadata.csv
# ══════════════════════════════════════════════════════════════════════════

def build_ticker_metadata(stocks_meta: pd.DataFrame,
                          sector_index_starts: dict[str, pd.Timestamp],
                          out_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    print("=" * 70)
    print("OUTPUT 4: Ticker metadata")
    print("=" * 70)

    # Equivalent of: SELECT DISTINCT ticker, primary_sector, market_cap
    #                FROM backtest_data WHERE market_cap IS NOT NULL
    df = (stocks_meta.dropna(subset=["market_cap"])[["ticker", "primary_sector", "market_cap"]]
          .drop_duplicates())

    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    def assign_cap_tier(rank):
        if rank <= 100:
            return "LARGE"
        elif rank <= 250:
            return "MID"
        elif rank <= 500:
            return "SMALL"
        return "MICRO"

    df["cap_tier"] = df["rank"].apply(assign_cap_tier)
    df["index_symbol"] = df["primary_sector"].map(FULL_NAME_TO_DB_SYMBOL)
    df["sector_has_index_data"] = df["index_symbol"].apply(
        lambda s: s in sector_index_starts if pd.notna(s) else False)
    df["sector_data_start_date"] = df["index_symbol"].apply(
        lambda s: sector_index_starts.get(s, pd.NaT) if pd.notna(s) else pd.NaT)

    result = df[["ticker", "primary_sector", "index_symbol", "market_cap",
                  "cap_tier", "sector_has_index_data", "sector_data_start_date"]]

    out_path = out_dir / "ticker_metadata.csv"
    result.to_csv(out_path, index=False)
    ticker_sector_map = dict(zip(df["ticker"], df["primary_sector"]))

    print(f"  Total tickers: {len(result)}")
    for tier, cnt in result["cap_tier"].value_counts().sort_index().items():
        print(f"    {tier:10s}: {cnt}")
    print(f"  Sectors with index data: {result['sector_has_index_data'].sum()}/{len(result)} tickers")
    print(f"  Saved: {out_path}\n")
    return result, ticker_sector_map


# ══════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH (from add_sector_rs.py)
# ══════════════════════════════════════════════════════════════════════════

def _load_nifty50_weekly_close(sector_daily: pd.DataFrame) -> pd.Series:
    """Nifty 50 daily from sector OHLCV, resampled to weekly close (RS benchmark)."""
    df = _index_daily(sector_daily, "NIFTY 50")
    df = df.set_index("date").sort_index().rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    weekly = resample_weekly(df)
    return weekly["Close"]


def compute_sector_rs(sector_df: pd.DataFrame, nifty_weekly_close: pd.Series) -> pd.DataFrame:
    """Add the 5 RS columns to the sector_regime DataFrame (verbatim logic)."""
    nifty_close = nifty_weekly_close.sort_index()
    nifty_ret_13w = (nifty_close / nifty_close.shift(RS_LOOKBACK) - 1) * 100
    nifty_ret_map = {str(dt.date()): val for dt, val in nifty_ret_13w.items()}

    sector_df = sector_df.copy()
    sector_df["week_date"] = pd.to_datetime(sector_df["week_date"])
    sector_df["week_date_str"] = sector_df["week_date"].astype(str).str[:10]

    sector_returns = []
    for idx_sym, grp in sector_df.groupby("index_symbol"):
        grp = grp.sort_values("week_date").copy()
        closes = grp["close"].values
        ret_13w = np.full(len(closes), np.nan)
        for i in range(RS_LOOKBACK, len(closes)):
            if closes[i - RS_LOOKBACK] > 0:
                ret_13w[i] = (closes[i] / closes[i - RS_LOOKBACK] - 1) * 100
        grp["sector_return_13w"] = ret_13w
        sector_returns.append(grp)
    sector_df = pd.concat(sector_returns)

    sector_df["nifty50_return_13w"] = sector_df["week_date_str"].map(nifty_ret_map)
    sector_df["sector_rs_vs_nifty"] = sector_df["sector_return_13w"] - sector_df["nifty50_return_13w"]

    ranks, quartiles = [], []
    for wk, grp in sector_df.groupby("week_date"):
        valid = grp["sector_rs_vs_nifty"].notna()
        n_valid = valid.sum()
        if n_valid == 0:
            for idx in grp.index:
                ranks.append((idx, np.nan)); quartiles.append((idx, np.nan))
            continue
        rs_vals = grp.loc[valid, "sector_rs_vs_nifty"]
        ranked = rs_vals.rank(ascending=False, method="min").astype(int)
        q1 = math.ceil(n_valid * 0.25)
        q2 = math.ceil(n_valid * 0.50)
        q3 = math.ceil(n_valid * 0.75)
        for idx in grp.index:
            if idx in ranked.index:
                r = int(ranked.loc[idx])
                ranks.append((idx, r))
                if r <= q1:
                    quartiles.append((idx, "TOP"))
                elif r <= q2:
                    quartiles.append((idx, "UPPER"))
                elif r <= q3:
                    quartiles.append((idx, "LOWER"))
                else:
                    quartiles.append((idx, "BOTTOM"))
            else:
                ranks.append((idx, np.nan)); quartiles.append((idx, np.nan))

    sector_df["sector_rs_rank"] = pd.Series(dict(ranks), name="sector_rs_rank")
    sector_df["sector_rs_quartile"] = pd.Series(dict(quartiles), name="sector_rs_quartile")

    for col in ["sector_return_13w", "nifty50_return_13w", "sector_rs_vs_nifty"]:
        sector_df[col] = sector_df[col].round(2)

    sector_df = sector_df.drop(columns=["week_date_str"])
    sector_df = sector_df.sort_values(["index_symbol", "week_date"])
    return sector_df


# ══════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════

def build_sector_dimensions(ohlcv_dir: Path | str | None = None,
                            data_dir: Path | str | None = None) -> Path:
    """Regenerate the sector/cap-tier/ticker-metadata bundle CSVs from raw OHLCV.

    Args:
        ohlcv_dir: dir holding ``sector_daily`` + ``stocks_daily``
            (``.parquet``/``.csv``). Defaults to ``_paths.ohlcv_dir()``.
        data_dir: bundle dir (output target + ``stock_weekly_cache/`` source).
            Defaults to ``_paths.data_dir()``.

    Returns the bundle directory the CSVs were written to.
    """
    global OUTPUT_DIR, STOCK_CACHE_DIR
    if data_dir is not None:
        _paths.set_data_dir(data_dir)
    if ohlcv_dir is not None:
        _paths.set_ohlcv_dir(ohlcv_dir)
    OUTPUT_DIR = _paths.require_data_dir()
    STOCK_CACHE_DIR = OUTPUT_DIR / "stock_weekly_cache"
    ohlcv = _paths.ohlcv_dir()

    t0 = time.time()
    print("=" * 70)
    print("COMPUTE REGIME DIMENSIONS — Phase 4 Data Build (file-based)")
    print("=" * 70)
    print(f"  OHLCV input: {ohlcv}")
    print(f"  Bundle out:  {OUTPUT_DIR}")

    sector_daily = _load_sector_daily(ohlcv)
    stocks_meta = _load_stocks_meta(ohlcv)

    # Sector index start dates (NIFTY* only), used for ticker metadata.
    nifty = sector_daily[sector_daily["index_symbol"].str.startswith("NIFTY")]
    sector_index_starts = {sym: pd.Timestamp(d) for sym, d in
                           nifty.groupby("index_symbol")["trade_date"].min().items()}

    primary_sectors = sorted(stocks_meta["primary_sector"].dropna().unique())

    # OUTPUT 4 first (provides the ticker→sector map for breadth).
    ticker_meta, ticker_sector_map = build_ticker_metadata(stocks_meta, sector_index_starts, OUTPUT_DIR)

    # OUTPUT 1 (+ RS augmentation), then save.
    sector_regime = build_sector_regime(sector_daily, primary_sectors)
    print("  Adding 13-week relative strength columns ...")
    nifty50_weekly = _load_nifty50_weekly_close(sector_daily)
    sector_regime = compute_sector_rs(sector_regime, nifty50_weekly)
    sector_path = OUTPUT_DIR / "sector_regime_weekly.csv"
    sector_regime.to_csv(sector_path, index=False)
    rs_valid = sector_regime["sector_rs_vs_nifty"].notna().sum()
    print(f"  Saved: {sector_path} ({len(sector_regime):,} rows, "
          f"RS coverage {rs_valid/len(sector_regime)*100:.1f}%)\n")

    # OUTPUT 2 + 3.
    build_captier_regime(sector_daily, OUTPUT_DIR)
    build_sector_breadth(ticker_sector_map, STOCK_CACHE_DIR, OUTPUT_DIR)

    print("=" * 70)
    print(f"Done. Total time: {time.time() - t0:.1f}s")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    return OUTPUT_DIR


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="skysurf-build-sectors",
        description="Regenerate sector_regime_weekly.csv, captier_regime_weekly.csv, "
                    "ticker_metadata.csv (+ sector_breadth_weekly.csv) from raw sector "
                    "index OHLCV (sector_daily) + stocks_daily.",
    )
    p.add_argument("--ohlcv", metavar="DIR", default=None,
                   help="Dir with sector_daily + stocks_daily (.parquet/.csv). "
                        "Defaults to $SKYSURF_REPRO_OHLCV, then ./skysurf-repro-ohlcv.")
    p.add_argument("--data", metavar="DIR", default=None,
                   help="Bundle dir (output + stock_weekly_cache source). "
                        "Defaults to $SKYSURF_REPRO_DATA, then ./skysurf-repro-data.")
    args = p.parse_args(argv)
    try:
        out = build_sector_dimensions(ohlcv_dir=args.ohlcv, data_dir=args.data)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nSector dimension CSVs written to: {out}")
    return 0


main = _cli


if __name__ == "__main__":
    raise SystemExit(_cli())
