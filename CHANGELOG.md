# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API is unstable until version `1.0.0`.

## [Unreleased]

## [0.2.0] - 2026-05-08

### Added

- **Strategy engine fully landed.** `generate_weekly_signals` and `manage_positions` now run end-to-end; previously raised `NotImplementedError`.
- New private subpackage `skysurf._internal` containing the vendored engine modules:
  - `swings.py` — swing high / low detection (`detect_swings_argrelextrema`).
  - `regime.py` — Weinstein 2-signal classifier and breadth-based 5-state classifier.
  - `stages.py` — Weinstein stage 1/2/3/4 classifier and MA slope-direction.
  - `detectors.py` — five entry detectors: BREAKOUT, PULLBACK, VCP, RETEST, TRENDLINE.
  - `detection.py` — per-ticker orchestrator (`find_entries_for_ticker`).
  - `engine.py` — per-week selection (`select_entries_for_week`) and exit (`decide_exits_for_week`) functions, sizing, position dataclass, cache base class.
  - `ranking.py` — dynamic type-prior helper.
  - `adapter.py` — `DataProviderCacheAdapter` bridging the public DataProvider to the engine cache.
  - `translation.py` — public `Position` ↔ engine-internal `_Position` conversion.
- `python -m skysurf doctor` — installation health check that validates required and optional dependencies, then runs a synthetic-data hello-world end-to-end.
- `py.typed` (PEP 561) marker so callers' type-checkers see the type hints we ship.
- GitHub issue templates (bug report, feature request).
- Five runnable example scripts in `examples/`.
- Integration smoke tests in `tests/integration/test_signals_smoke.py`.
- Optional `[trendlines]` extra: install `skysurf[trendlines]` to enable the `TRENDLINE_BOUNCE` detector (gracefully no-ops without it).

### Changed

- README rewritten with badges, headline numbers up top, and a clean five-minute quickstart modeled on polars / fastapi / ruff.

## [0.1.0] - 2026-05-08

### Added

- Initial public release (foundation only — engine pending).
- `generate_weekly_signals` and `manage_positions` placeholders raising `NotImplementedError`.
- `StrategyConfig` dataclass with the canonical `PHASE_4_V1` configuration.
- `DataProvider` abstract base class and five reference implementations: `InMemoryDataProvider`, `PandasDataProvider`, `CsvDataProvider`, `ParquetDataProvider`, `SQLAlchemyDataProvider`.
- Zerodha equity-delivery cost model.
- ATR, RSI, and moving-average indicators.
- Initial documentation set: user guide, strategy reference, data schema, connectors guide, reproducibility notes.
