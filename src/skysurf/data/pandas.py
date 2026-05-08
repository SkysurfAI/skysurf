"""PandasDataProvider — wrap user-supplied DataFrames.

Use this when you already have all required data as
:class:`pandas.DataFrame` objects in memory and just need to satisfy the
:class:`~skysurf.data.provider.DataProvider` contract.

Functionally identical to :class:`~skysurf.data.provider.InMemoryDataProvider`,
but exposed under a more discoverable name and with an explicit
constructor that documents which DataFrames are required versus
optional.
"""

from __future__ import annotations

from dataclasses import dataclass

from skysurf.data.provider import InMemoryDataProvider


@dataclass
class PandasDataProvider(InMemoryDataProvider):
    """Bring-your-own-DataFrame DataProvider.

    The required fields (positional or keyword) are:

    * ``weekly_ohlcv``: ``dict[ticker, DataFrame[Open, High, Low, Close, Volume]]``
    * ``daily_ohlcv``: same shape, daily-indexed.
    * ``nifty_weekly``: ``DataFrame[Close, ...]`` indexed by weekly date.
    * ``sector_indices_weekly``: ``dict[sector_name, DataFrame[Close, ...]]``.
    * ``universe``: ``DataFrame[ticker, sector, market_cap, adtv_20d]``.
    * ``historical_trades``: ``DataFrame[ticker, week_date, entry_type, mfe_pct, mae_pct]``.

    Optional regime / sector lookup fields are inherited from
    :class:`~skysurf.data.provider.InMemoryDataProvider`. See
    ``docs/data-schema.md`` for the canonical schema.

    Example::

        from skysurf.data.pandas import PandasDataProvider

        provider = PandasDataProvider(
            weekly_ohlcv={"RELIANCE.NS": weekly_df, ...},
            daily_ohlcv={"RELIANCE.NS": daily_df, ...},
            nifty_weekly=nifty_df,
            sector_indices_weekly={"NIFTY ENERGY": energy_df, ...},
            universe=universe_df,
            historical_trades=pd.DataFrame(columns=[
                "ticker", "week_date", "entry_type", "mfe_pct", "mae_pct",
            ]),
        )
    """
