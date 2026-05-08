# Roadmap

## v0.2.0 — current

**Engine fully landed.** ✅ `generate_weekly_signals` and `manage_positions` run end-to-end.

* `StrategyConfig` and the canonical `PHASE_4_V1` lock.
* Public types: `Position`, `EntrySignal`, `PositionAction`, `ActionType`, `TrailMode`, `TripleStackTighten`, `ValidationResult`.
* `DataProvider` abstract base + five reference connectors: `InMemoryDataProvider`, `PandasDataProvider`, `CsvDataProvider`, `ParquetDataProvider`, `SQLAlchemyDataProvider`.
* `skysurf._internal.*` — vendored strategy engine (swings, regime, stages, detectors, detection, ranking, engine, adapter, translation).
* Zerodha equity-delivery cost model.
* Indicators: ATR, RSI, moving averages.
* `python -m skysurf doctor` smoke check.
* Five worked examples (`examples/01_hello_world.py` through `05_production_loop.py`).
* Integration smoke tests + 186 unit tests, all passing on Python 3.11 / 3.12 / 3.13.
* CI: ruff + mypy `--strict` + pytest matrix.
* `py.typed` (PEP 561) marker.
* Documentation set: README, user guide, strategy reference, data schema, connectors guide, reproducibility notes.
* GitHub issue templates.

## v0.3.0 — next

Things on deck (in no particular order):

* **Behaviour-parity check** against the internal `skysurf_brain` package on one historical week with identical inputs (one-time bring-up assertion).
* **More example datasets** — sample Parquet pack with synthetic but realistically-shaped data so users can run examples 02–04 without bringing their own data.
* **Better docstring coverage** in `_internal/engine.py` — the largest module currently has function-level docstrings but some helpers could use more detail.
* **Logging redesign** — the engine currently uses `print` for diagnostic counters in some paths; route through `logging` consistently.

## v1.0.0 — API stability promise

Once v0.2.0 has been used in earnest by a few people and the rough edges are smoothed, we'll promise to keep the public API stable across minor versions. Until then, breaking changes may land in any release.

## Out of scope

* **Re-publishing the research walk-forward harness.** By design — see [reproducibility.md](docs/reproducibility.md).
* **Reference Kite / Truedata / Dhan adapters.** Each broker SDK has its own version cadence and auth dance; embedding them in core would couple users' upgrades to ours. May publish a separate `skysurf-kite` companion package later.
* **Live (sub-weekly) signal generation.** Skysurf operates on weekly bars — that's the strategy.
* **Order placement.** The library emits signals; you wire the broker.
