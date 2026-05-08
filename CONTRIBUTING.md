# Contributing to Skysurf

Thanks for considering a contribution. This document covers the basics; for substantive design changes please open an issue first to discuss before sending a PR.

## Development setup

```bash
git clone https://github.com/SkysurfAI/skysurf.git
cd skysurf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

## Quality gates

Every PR runs the following in CI; please run them locally before pushing.

```bash
ruff check .
ruff format --check .
mypy src/skysurf
pytest
```

We aim to keep coverage at or above 80%. New public functions should have unit tests.

## Style

- Code is formatted by `ruff format` (Black-compatible).
- Public functions, classes, and modules carry Google-style docstrings.
- Type annotations are required on every public surface; prefer `from __future__ import annotations` everywhere.
- `print()` is for examples and the CLI. The library uses `logging.getLogger(__name__)`.
- Avoid `Any` unless you can justify it.
- Avoid `# type: ignore` without a comment explaining why.
- No commented-out code in committed PRs.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Examples:

```
feat(data): add SQLAlchemyDataProvider
fix(brain): correct ATR buffer when stop equals close
docs(guide): add weekly cron snippet
```

## DCO

By contributing, you certify that the contribution complies with the [Developer Certificate of Origin](https://developercertificate.org/). Sign your commits with `git commit -s`.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. Email the maintainers privately first.
