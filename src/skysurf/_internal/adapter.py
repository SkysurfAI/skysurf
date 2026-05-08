"""DataProviderCacheAdapter — feed the engine from a live DataProvider.

The engine's exit-evaluation pipeline reads per-ticker weekly OHLCV
plus indicators (ATR, RS, MAs, volume ratio) through
:class:`~skysurf._internal.engine._StockCacheManager`. The base class
just declares the interface; this adapter implements it by pulling raw
weekly OHLCV from the user's
:class:`~skysurf.data.provider.DataProvider` and computing the
indicators on-demand.

The result of :meth:`DataProviderCacheAdapter.get` is a DataFrame
indexed by weekly date with these columns::

    Open, High, Low, Close, Volume,
    atr_14, rs_13w, rsi_14, momentum, volume_ratio_20w,
    sma_<period>, ema_<period>  (one column per MA needed)

Per-ticker results are cached for the lifetime of the adapter; misses
(no data from the provider) are cached as ``None`` so re-queries are
free.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from skysurf._internal.detection import MIN_WEEKLY_BARS
from skysurf._internal.engine import _StockCacheManager
from skysurf.data.provider import DataProvider
from skysurf.indicators import calculate_atr, calculate_rsi, compute_ma_series

_LOG = logging.getLogger(__name__)


class DataProviderCacheAdapter(_StockCacheManager):
    """Bridge between a public :class:`DataProvider` and the engine cache.

    Subclasses :class:`~skysurf._internal.engine._StockCacheManager` so
    the engine code that calls ``cache.get(ticker)`` can stay agnostic
    to where data comes from.

    Args:
        provider: Concrete DataProvider supplying weekly OHLCV.
        nifty_close_weekly: Nifty 50 weekly close series, used for the
            relative-strength computation.
        window_start: Earliest week to fetch — must give the longest
            indicator (typically SMA-40) enough warmup.
        window_end: Latest week to fetch (typically the as-of date).
        ma_periods: Per-ticker (ma_type, period) pairs to compute.
            Should cover every MA the active config needs (entry MAs,
            exit MA, triple-stack tightening MAs, progressive trail MA).
        extra_sma_periods: Additional plain-SMA periods to compute on
            top of ``ma_periods`` (kept for API parity with the
            research StockCacheManager which had this knob).
    """

    def __init__(
        self,
        provider: DataProvider,
        nifty_close_weekly: pd.Series,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        ma_periods: Iterable[tuple[str, int]] | None = None,
        extra_sma_periods: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._nifty = nifty_close_weekly
        self._window_start = pd.Timestamp(window_start)
        self._window_end = pd.Timestamp(window_end)
        self._ma_periods: list[tuple[str, int]] = list(ma_periods or [])
        self._extra_sma: list[int] = list(extra_sma_periods or [])
        self._cache: dict[str, pd.DataFrame | None] = {}

    def get(self, ticker: str) -> pd.DataFrame | None:
        if ticker in self._cache:
            return self._cache[ticker]
        df = self._build_for_ticker(ticker)
        self._cache[ticker] = df
        return df

    def _build_for_ticker(self, ticker: str) -> pd.DataFrame | None:
        weekly_map = self._provider.get_weekly_ohlcv([ticker], self._window_start, self._window_end)
        weekly = weekly_map.get(ticker)
        if weekly is None or weekly.empty:
            return None
        if len(weekly) < MIN_WEEKLY_BARS:
            _LOG.debug(
                "DataProviderCacheAdapter: %s has %d bars (< %d), skipping",
                ticker,
                len(weekly),
                MIN_WEEKLY_BARS,
            )
            return None

        out = weekly[["Open", "High", "Low", "Close", "Volume"]].copy()

        # ATR(14)
        out["atr_14"] = calculate_atr(out["High"], out["Low"], out["Close"], window=14)

        # rs_13w vs Nifty
        nifty_aligned = self._nifty.reindex(out.index, method="nearest")
        stock_ret = out["Close"] / out["Close"].shift(13) - 1
        nifty_ret = nifty_aligned / nifty_aligned.shift(13) - 1
        out["rs_13w"] = stock_ret / nifty_ret.replace(0, pd.NA)

        # RSI, momentum, 20-week volume ratio
        out["rsi_14"] = calculate_rsi(out["Close"], window=14)
        out["momentum"] = (out["Close"] > out["Close"].shift(1)).astype(int)
        vol_20w = out["Volume"].rolling(20).mean()
        out["volume_ratio_20w"] = out["Volume"] / vol_20w.replace(0, pd.NA)

        # MA series — one column per requested (ma_type, period).
        for ma_type, period in self._ma_periods:
            if ma_type not in ("SMA", "EMA"):
                raise ValueError(f"ma_periods contains invalid MA type: {ma_type!r}")
            col = f"{ma_type.lower()}_{period}"
            out[col] = compute_ma_series(out["Close"], ma_type, period)  # type: ignore[arg-type]

        # Extra plain-SMA columns (e.g., sma_15, sma_27 for trail tightening).
        for p in self._extra_sma:
            col = f"sma_{p}"
            if col not in out.columns:
                out[col] = out["Close"].rolling(p, min_periods=1).mean()

        return out


__all__ = ["DataProviderCacheAdapter"]
