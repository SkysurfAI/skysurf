# User guide

End-to-end walkthrough: install → first signal → run the tests → wire into a weekly cron.

This doc assumes Python 3.11+. The instructions are POSIX (macOS / Linux); Windows works the same with PowerShell-flavoured paths.

## 1. Install

```bash
pip install skysurf
```

If you use Parquet or SQL connectors:

```bash
pip install "skysurf[parquet]"
pip install "skysurf[sqlalchemy]"
pip install "skysurf[all]"
```

To verify:

```python
import skysurf
print(skysurf.__version__)        # 0.1.0
print(skysurf.PHASE_4_V1.exit_ma_period)  # 27
```

## 2. Bring your data

Skysurf needs six things, all schema-documented in [data-schema.md](data-schema.md):

1. Weekly OHLCV per ticker
2. Daily OHLCV per ticker (for ADTV)
3. Nifty 50 weekly bars
4. Per-sector weekly bars (one DataFrame per sector you care about)
5. The eligible universe (one row per ticker with sector, market cap, ADTV)
6. Historical strategy trades (for dynamic ranking — empty is OK on day one)

Optionally you can also provide pre-computed regime data; the brain falls back to "block" if absent.

If your data is already in pandas DataFrames, use `PandasDataProvider`. If it's in CSV, Parquet, or any SQL DB, use the matching connector. See [connectors.md](connectors.md) for the decision tree.

> **Where to source the data**: Skysurf does not ship Indian-equity OHLCV (NSE owns the licensed data and we cannot redistribute). Recommended sources:
> * Zerodha Kite Connect (paid, https://kite.trade)
> * Truedata (paid, https://truedata.in)
> * GDFL (paid, https://gdfl.com)
> * NSE itself for index data
>
> **Avoid yfinance for production**: its corporate-action adjustment is unreliable on Indian symbols.

## 3. First signal

Once you have data wired, generating signals for a week is one call:

```python
from datetime import date

from skysurf import generate_weekly_signals, PHASE_4_V1

signals = generate_weekly_signals(
    provider=my_provider,
    as_of_date=date(2024, 6, 7),         # a Friday
    current_positions=[],                 # held positions, if any
    total_equity=500_000.0,               # current portfolio equity
    config=PHASE_4_V1,
)

for s in signals:
    print(f"{s.ticker:14s}  entry={s.entry_price:>8.2f}  "
          f"stop={s.initial_stop:>8.2f}  qty={s.qty:>4d}  "
          f"({s.entry_type})")
```

`signals` is a ranked list of `EntrySignal` instances. Higher-ranked signals come first. Act on as many as your portfolio constraints allow — the brain has already applied the per-sector and total-position caps.

> **Note**: in the v0.1.0 release, the engine that powers `generate_weekly_signals` is being vendored from the production research code. Until that vendoring lands, the function is a placeholder. Track [ROADMAP.md](../ROADMAP.md) for status.

## 4. Manage exits

For each held position you maintain externally, ask the brain what to do this week:

```python
from skysurf import manage_positions, PHASE_4_V1

actions = manage_positions(
    provider=my_provider,
    as_of_date=date(2024, 6, 7),
    current_positions=my_held_positions,  # list[Position]
    config=PHASE_4_V1,
)

for action in actions:
    match action.action_type:
        case "HOLD":
            pass
        case "UPDATE_STOP":
            update_gtt_stop(action.ticker, action.new_stop_level)
        case "PARTIAL_SELL":
            sell(action.ticker, action.sell_qty, action.sell_price)
            update_gtt_stop(action.ticker, action.new_stop_level)
        case "EXIT_FULL":
            sell(action.ticker, action.sell_qty, action.sell_price)
        case "SWITCH_TRAIL_MA":
            update_gtt_stop(action.ticker, action.new_stop_level)
```

The brain itself does not place orders. You wire the broker.

## 5. Run the tests

Skysurf ships a comprehensive test suite that runs on synthetic data — no setup, no external dependencies:

```bash
git clone https://github.com/<org>/skysurf.git
cd skysurf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expect ~140 tests, all passing in under a second. If you implement a custom `DataProvider`, copy the contract tests in `tests/unit/test_in_memory_provider.py` and `tests/unit/test_filesystem_providers.py` against your subclass — they're the easiest way to catch schema drift.

## 6. Wire into a weekly cron

Production setup, in 30 lines:

```python
# /opt/skysurf/run_weekly.py
import logging
from datetime import date

from skysurf import (
    generate_weekly_signals, manage_positions, PHASE_4_V1,
)

from my_app.data import build_provider
from my_app.broker import place_orders, current_positions, total_equity

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("skysurf-runner")

def main():
    today = date.today()
    if today.weekday() != 4:  # 4 = Friday
        log.warning("Not Friday; skipping run.")
        return

    provider = build_provider()
    held = current_positions()
    equity = total_equity()

    actions = manage_positions(provider, today, held, config=PHASE_4_V1)
    log.info("manage_positions: %d actions", len(actions))
    place_orders(actions)

    signals = generate_weekly_signals(provider, today, held, equity, config=PHASE_4_V1)
    log.info("generate_weekly_signals: %d signals", len(signals))
    place_orders(signals[:PHASE_4_V1.max_positions - len(held)])

if __name__ == "__main__":
    main()
```

Run via cron at, say, 16:05 IST on Fridays after the cash market closes:

```cron
5 16 * * 5  /opt/skysurf/.venv/bin/python /opt/skysurf/run_weekly.py >> /var/log/skysurf.log 2>&1
```

Things you'll want to add in real life:
* **Idempotency**: if the cron retries, don't double-place orders. Check broker order book first.
* **Alerting**: send signals to Slack or email so a human reviews them before placing.
* **Pre-check**: confirm your data pipeline ran successfully and the latest week's bars exist before calling the brain.
* **Rate limits**: most Indian brokers throttle order placement; insert delays between submissions.

## 7. Customising the strategy

`PHASE_4_V1` is the canonical locked configuration that produces the published numbers. Don't edit it; derive a new config:

```python
from dataclasses import replace
from skysurf import PHASE_4_V1

# More conservative: 0.5% risk per trade, max 20 positions
my_config = replace(PHASE_4_V1, risk_pct=0.005, max_positions=20)
signals = generate_weekly_signals(..., config=my_config)
```

Every parameter is documented in [strategy.md](strategy.md). Re-validate on your data before going live with anything other than `PHASE_4_V1`.

## Troubleshooting

* **`ModuleNotFoundError: pyarrow`** — install with `pip install "skysurf[parquet]"`.
* **`ModuleNotFoundError: sqlalchemy`** — install with `pip install "skysurf[sqlalchemy]"`.
* **Empty signals on a week with obvious setups** — likely the regime gate is blocking. Check `provider.get_overall_regime_snapshot(week_date)` returns the expected `regime` and `breadth_pct`.
* **`ValueError: high, low, close must share index`** in indicator calls — your weekly OHLCV DataFrame's columns are misaligned. Make sure they're slices from a single DataFrame, not built independently.
* **Tests fail on `pytest`** — make sure you ran `pip install -e ".[dev]"` to pick up dev tooling. The bare install excludes pytest.

## Where to go next

* [strategy.md](strategy.md) — every Phase 4 parameter explained
* [reproducibility.md](reproducibility.md) — what you can and cannot reproduce, and the data caveats
* [connectors.md](connectors.md) — pick your connector, write your own
* [data-schema.md](data-schema.md) — exact column names and dtypes

If you get stuck, [open an issue](https://github.com/<org>/skysurf/issues) — include the Python version, your connector, and a minimal reproduction.
