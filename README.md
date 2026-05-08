# Skysurf

A production trading-signal library for Indian equities, implementing the validated **Phase 4** swing-trading strategy.

```bash
pip install skysurf
```

```python
from datetime import date

from skysurf import (
    InMemoryDataProvider, PHASE_4_V1, generate_weekly_signals,
)

# 1. Bring your own data — wire any DataProvider:
provider = InMemoryDataProvider(...)   # or CsvDataProvider, ParquetDataProvider, ...

# 2. Generate signals for the week:
signals = generate_weekly_signals(
    provider=provider,
    as_of_date=date(2024, 6, 7),
    current_positions=[],
    total_equity=500_000.0,
    config=PHASE_4_V1,
)

for s in signals:
    print(f"{s.ticker}  entry={s.entry_price:.2f}  stop={s.stop_level:.2f}  qty={s.qty}")
```

That's the whole production loop. Wire it into a Friday-evening cron, place GTT orders against the signals, and you're done. **The 5-minute quickstart is in [`docs/guide.md`](docs/guide.md).**

## What is this

Skysurf is the open-source core of a quantitative swing-trading strategy that has been validated on 16 years of Indian equity data (2010–2026). It uses weekly bars, breadth-based market regime gating, sector relative-strength filtering, five complementary entry detectors (pullback, breakout, VCP continuation, retest, trendline), dynamic per-segment ranking, and a triple-stack tightening exit framework with partial profit-taking.

The validated walk-forward numbers (with Zerodha-equivalent costs):

| Metric | Value |
|---|---|
| MAR | 1.96 |
| CAGR | 24.13% |
| Max drawdown | −12.31% |
| Trades | 604 |
| Window | 2010-01-01 → 2026-03-21 |

The full strategy specification, including every parameter, is in [`docs/strategy.md`](docs/strategy.md).

## What this library does — and doesn't

**Does**

- Generates entry signals (which stocks to buy this week, at what price, with what stop).
- Manages exits for open positions (which to sell, at what stop level, with partial profit-taking).
- Computes the Zerodha equity-delivery cost model.
- Connects to your data via plug-in `DataProvider` adapters: in-memory pandas, a directory of CSVs, a directory of Parquet, or any SQL DB via SQLAlchemy.

**Doesn't**

- Place orders. You wire your broker; we generate the signals.
- Provide market data. You bring it. NSE owns the licensed OHLCV data and we cannot redistribute it.
- Run the research walk-forward. The validation harness that produces the headline 24.13% CAGR is intentionally not open-sourced. You can verify the strategy on your own data; you cannot reproduce our numbers from a fresh clone alone. See [`docs/reproducibility.md`](docs/reproducibility.md) for what you *can* do.

## Connectors

Pick the one that matches your data:

| Connector | Use when... |
|---|---|
| `InMemoryDataProvider` | You have everything as Python dicts/DataFrames in memory. Best for tests and demos. |
| `PandasDataProvider` | You already have OHLCV / regime / metadata as pandas DataFrames. |
| `CsvDataProvider` | You have a directory of CSVs in the documented schema. |
| `ParquetDataProvider` | Same, but Parquet. Recommended for production: faster, smaller, typed. |
| `SQLAlchemyDataProvider` | Your data is in Postgres / MySQL / SQLite / any SQL DB. |

Detail and a decision tree in [`docs/connectors.md`](docs/connectors.md). Schema reference for all connectors is in [`docs/data-schema.md`](docs/data-schema.md).

## Documentation

- [User guide](docs/guide.md) — install, first signal, tests, production wiring
- [Strategy reference](docs/strategy.md) — every parameter, every detector, every exit rule
- [Data schema](docs/data-schema.md) — what columns, what dtypes
- [Connectors](docs/connectors.md) — pick one, or write your own
- [Reproducibility](docs/reproducibility.md) — what you can verify, what you can't

## License & trademark

Source code is licensed under [Apache 2.0](LICENSE). The "Skysurf" name and logo are trademarks; see [TRADEMARK.md](TRADEMARK.md).

The Apache license covers the code only. If you fork this for a commercial managed service, please rename the fork.

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style, and the quality gates we run.

## Acknowledgements

The Phase 4 strategy was developed and validated over an extended research program on Indian equities. The published numbers above were reproduced exactly by an independent code-review pass (CC-VERIFIED, April 2026).

Skysurf is **not** investment advice. Past performance does not indicate future results. Run your own validation on your own data before risking capital.
