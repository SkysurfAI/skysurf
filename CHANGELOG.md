# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The public API is unstable until version `1.0.0`.

## [Unreleased]

## [0.1.0] - 2026-05-08

### Added

- Initial public release.
- `generate_weekly_signals` and `manage_positions` public APIs.
- `StrategyConfig` dataclass with the canonical `PHASE_4_V1` configuration locked to MAR 1.96 / CAGR 24.13% / MaxDD −12.31% / 604 trades on the validation walk-forward window (2010-01-01 to 2026-03-21).
- `DataProvider` abstract base class and five reference implementations: `InMemoryDataProvider`, `PandasDataProvider`, `CsvDataProvider`, `ParquetDataProvider`, `SQLAlchemyDataProvider`.
- Zerodha equity-delivery cost model.
- ATR, RSI, and moving-average indicator implementations.
- User guide, strategy reference, data schema reference, connectors guide, and reproducibility notes.
- Five worked examples covering hello-world, CSV, Parquet, SQLAlchemy, and a production weekly loop.
