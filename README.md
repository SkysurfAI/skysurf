# Skysurf

**A production trading-signal library for Indian equities, implementing the validated Phase 4 swing-trading strategy.**

[![CI](https://github.com/SkysurfAI/skysurf/actions/workflows/ci.yml/badge.svg)](https://github.com/SkysurfAI/skysurf/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/SkysurfAI/skysurf)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Type-checked: mypy --strict](https://img.shields.io/badge/typed-mypy%20--strict-blue)](https://mypy.readthedocs.io/)

Validated over 16 years of weekly Indian-equity data (2010–2026):

| MAR | CAGR | Max drawdown | Trades |
|---|---|---|---|
| **1.96** | **24.13%** | **−12.31%** | **604** |

---

## Install

```bash
pip install skysurf
```

Verify the install:

```bash
python -m skysurf doctor
```

That runs a synthetic-data hello-world end-to-end and reports any issues.

## Quick start

```python
from datetime import date

from skysurf import generate_weekly_signals, manage_positions, ParquetDataProvider

# 1. Point at your data (CSV, Parquet, SQL, or in-memory).
provider = ParquetDataProvider("/path/to/data")

# 2. Generate entry signals for the week.
signals = generate_weekly_signals(
    provider=provider,
    as_of_date=date(2024, 6, 7),
    current_positions=[],
    total_equity=500_000.0,
)

for s in signals:
    print(f"{s.ticker:14s}  entry={s.entry_price:>8.2f}  "
          f"stop={s.initial_stop:>8.2f}  qty={s.qty}  ({s.entry_type})")

# 3. Manage exits for held positions.
actions = manage_positions(
    provider=provider,
    as_of_date=date(2024, 6, 7),
    current_positions=current_positions,
)
```

That's the whole production loop. Wire it into a Friday-evening cron, place GTT orders against the signals, and you're done. The full 5-minute walkthrough is in [`docs/guide.md`](docs/guide.md).

## Why use this

- **Validated.** 1.96 MAR / 24.13% CAGR over 16 years on Indian equities, with realistic Zerodha equity-delivery costs included.
- **Production-quality.** Type-annotated end to end, `mypy --strict` clean, 186+ tests passing on Python 3.11 / 3.12 / 3.13.
- **Plug-and-play data.** Connectors for in-memory pandas, CSV directories, Parquet directories, and any SQL DB. Or write your own — the `DataProvider` interface is small.
- **Stateless.** The library doesn't write files, doesn't talk to brokers, doesn't keep secrets. It emits decisions; you place the orders.
- **No magic numbers.** Every strategy parameter lives in one immutable `StrategyConfig`. The canonical `PHASE_4_V1` constant is the configuration that produced the validated numbers above.
- **Apache 2.0 licensed.** Use it for research, paper trading, or your own production system. The trademark policy in [`TRADEMARK.md`](TRADEMARK.md) keeps the name controlled.

## Connectors

Pick the one that matches your data:

| Connector | When to use |
|---|---|
| `InMemoryDataProvider` | You have data as Python dicts/DataFrames already. Best for tests. |
| `PandasDataProvider` | Same as `InMemory` but a friendlier name for BYO-DataFrame setups. |
| `CsvDataProvider` | A directory of CSVs in the canonical schema. |
| `ParquetDataProvider` | Same, but Parquet. **Recommended for production** — faster reads, smaller on disk, dtypes preserved. |
| `SQLAlchemyDataProvider` | Postgres, MySQL, SQLite, or any other SQL DB SQLAlchemy supports. |

Decision tree and the full `DataProvider` interface in [`docs/connectors.md`](docs/connectors.md). Canonical column names and dtypes for every connector in [`docs/data-schema.md`](docs/data-schema.md).

## What this library does — and doesn't

**Does**

- Generates ranked entry signals (which stocks to buy this week, at what price, with what stop, sized per portfolio risk).
- Manages exits for open positions: trailing stops, partial profit-taking, triple-stack tightening, time stops.
- Computes the Zerodha equity-delivery cost model (regulatory + slippage + DP charge).
- Reads from any data source via the plug-in connectors above.

**Doesn't**

- Place orders. You wire the broker.
- Provide market data. You bring it. NSE owns the licensed OHLCV data and we cannot redistribute it.
- Run a multi-year backtest harness. The walk-forward harness that produced the headline 24.13% CAGR is intentionally not open-sourced. You can verify the strategy on your own data; you cannot reproduce our exact numbers from a fresh clone alone. See [`docs/reproducibility.md`](docs/reproducibility.md) for the rationale and what you *can* do.

## How the strategy works

Every Friday after market close:

1. **Filter the universe** by market cap (≥₹1,500 Cr) and 20-day ADTV (≥₹2 Cr).
2. **Apply the regime gate.** Block all entries unless either (a) the overall market breadth (% of universe above SMA-25) exceeds 60, or (b) the candidate's sector is in a Weinstein bull regime.
3. **Run five entry detectors** on each ticker:
   - **PULLBACK_S2** — pullback to support within Stage 2.
   - **BREAKOUT_S1_TO_S2** — base-to-Stage-2 transition with close clearing the ceiling.
   - **VCP_CONTINUATION** — Volatility Contraction Pattern broken upward.
   - **RETEST_SUPPORT** — broken swing high retested as support.
   - **TRENDLINE_BOUNCE** — bounce off a rising support trendline.
4. **Filter** by RS gate (13-week RS ≥ 0.5), per-detector base-min-weeks, retest-max-weeks.
5. **Dedupe** per `(ticker, week)` keeping the tightest stop.
6. **Rank** by dynamic type-prior (computed from your historical-trade history).
7. **Size** each entry: 1% of equity at risk × tier multiplier (0.5 / 0.75 / 1.0).
8. **Apply portfolio constraints**: max 5 positions per sector, max 30 total.

For each held position:

1. **Ratchet the trailing stop** (SMA-27 minus 0.75×ATR by default).
2. **Triple-stack tightening** kicks in if the position stalls (week ≥ 12, return < 7%) or shows a climactic-volume reversal.
3. **Take partial profit** at +30% gain, sell 50%, move stop to halfway between entry and current peak.
4. **Switch to a tighter trail** (SMA-20) at +50% unrealized gain.
5. **Exit** when the weekly close trips the stop.

Full strategy reference, every parameter explained, in [`docs/strategy.md`](docs/strategy.md).

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/guide.md`](docs/guide.md) | End-to-end user guide: install → first signal → tests → production wiring |
| [`docs/strategy.md`](docs/strategy.md) | Every Phase 4 parameter, every detector, every exit rule |
| [`docs/data-schema.md`](docs/data-schema.md) | Canonical column names, dtypes, file layouts |
| [`docs/connectors.md`](docs/connectors.md) | Pick a connector, or write your own |
| [`docs/reproducibility.md`](docs/reproducibility.md) | What you can verify, what you can't |

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the setup, style, and quality gates (`ruff`, `mypy --strict`, `pytest --cov`).

## License & trademark

Source code is licensed under [Apache 2.0](LICENSE). The "Skysurf" name and logo are trademarks; see [`TRADEMARK.md`](TRADEMARK.md). The Apache license covers the code only — if you fork for a commercial managed service, please rename the fork.

## Disclaimer

Skysurf is **not** investment advice. Past performance does not indicate future results. Run your own validation on your own data before risking capital, and consult a SEBI-registered investment adviser if you are advising others.
