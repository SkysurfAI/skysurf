# Roadmap

This file tracks where the open-source build is, and what's coming. The plan is in the maintainers' working notes.

## v0.1.0 — current

**Foundation: shipped and tested.**

* `StrategyConfig` and the canonical `PHASE_4_V1` lock.
* Public types: `Position`, `EntrySignal`, `PositionAction`, `ActionType`, `TrailMode`, `TripleStackTighten`, `ValidationResult`.
* `DataProvider` abstract base + five reference connectors:
  * `InMemoryDataProvider`
  * `PandasDataProvider`
  * `CsvDataProvider`
  * `ParquetDataProvider`
  * `SQLAlchemyDataProvider`
* Zerodha equity-delivery cost model (`buy_cost`, `sell_proceeds`, `buy_cost_factor`).
* Indicators: ATR, RSI, moving averages (SMA / EMA).
* Documentation: `README.md`, `docs/guide.md`, `docs/strategy.md`, `docs/data-schema.md`, `docs/connectors.md`, `docs/reproducibility.md`.
* CI: lint (ruff), type-check (mypy --strict), tests (pytest --cov) on Python 3.11 / 3.12 / 3.13.
* 136 unit tests, all passing on synthetic data, no external setup.

## v0.2.0 — next milestone (engine vendoring)

**`generate_weekly_signals` and `manage_positions` become functional end-to-end.**

The brain currently exports the two functions as a public surface but they're not wired to the strategy engine yet. The engine — entry detectors, per-week evaluator, sizing, exits — is being vendored from the production research repo into this package. Until that lands:

* `generate_weekly_signals(...)` raises `NotImplementedError` if called.
* `manage_positions(...)` likewise.

Vendoring scope (~3500 LOC across three modules):

* `_detect.py` — entry detection (subset of `val_engine_a_v2.process_ticker_entries`)
* `_evaluate.py` — per-week sizing, ranking, and exit evaluation (subset of `val_engine_b_experimental`)
* `_ranking.py` — dynamic type-prior helper (subset of `compute_dynamic_type_prior`)

After vendoring lands:

* `signals.py` and `positions.py` get rewired to call the vendored modules instead of the research shim.
* Five worked examples become runnable: `01_hello_world.py` (synthetic data), `02_csv_quickstart.py`, `03_parquet_quickstart.py`, `04_sqlalchemy_quickstart.py`, `05_production_loop.py`.
* Integration smoke test in `tests/integration/` validates end-to-end signal generation against synthetic data.
* Behaviour-parity check vs. the internal production brain (one historical week, identical inputs → identical signals) — one-time bring-up assertion.

ETA: separate focused work session. The engine vendoring is the largest single piece of work in this build and is best done with full focus rather than rushed.

## v1.0.0 — public-API stability promise

After v0.2.0 ships and we've collected feedback, we promise to keep the public API stable across minor versions. Until then, breaking changes may land in any release.

## Out of scope

* **Re-publishing the research walk-forward harness.** Out of scope by design — see [reproducibility.md](docs/reproducibility.md) for the rationale.
* **Reference Kite / Truedata / Dhan adapters.** Each broker SDK has its own auth dance and version cadence; embedding them in core would couple users' upgrade timelines to ours. We may publish a separate `skysurf-kite` companion package later.
* **Live (sub-weekly) signal generation.** Skysurf operates on weekly bars — that's the strategy.
* **Order placement.** The brain emits signals; you wire the broker.
