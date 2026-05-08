# Examples

Five worked examples demonstrating Skysurf's connectors and the end-to-end signal-generation flow.

| File | What it shows | Setup needed |
|---|---|---|
| [`01_hello_world.py`](01_hello_world.py) | End-to-end pipeline on synthetic data | None — runs on a fresh clone |
| [`02_csv_quickstart.py`](02_csv_quickstart.py) | Read OHLCV from a directory of CSVs | `SKYSURF_DATA_DIR` env var |
| [`03_parquet_quickstart.py`](03_parquet_quickstart.py) | Same, but Parquet | `SKYSURF_DATA_DIR`, `pip install skysurf[parquet]` |
| [`04_sqlalchemy_quickstart.py`](04_sqlalchemy_quickstart.py) | Read OHLCV from any SQL DB | `DATABASE_URL`, `pip install skysurf[sqlalchemy]` |
| [`05_production_loop.py`](05_production_loop.py) | Cron-ready weekly loop with broker stubs | Implement `build_provider()` / `load_held_positions()` / etc. |

## Read first

Start with [`01_hello_world.py`](01_hello_world.py) — it requires no setup and proves the install works:

```bash
python examples/01_hello_world.py
```

Then pick the connector example that matches where your data lives.

## Where to source the data

Skysurf does not ship Indian-equity OHLCV (NSE owns the licensed data and we cannot redistribute). Recommended sources:

- [Zerodha Kite Connect](https://kite.trade) — paid
- [Truedata](https://truedata.in) — paid
- [GDFL](https://gdfl.com) — paid
- NSE itself for index data

Avoid `yfinance` for production: its Indian-equity corporate-action adjustment is unreliable.

See [`docs/data-schema.md`](../docs/data-schema.md) for the canonical column names and shapes every connector expects.
