"""Verbatim Phase-4 research code (engine, walk-forward driver, type-prior).

These modules are the *unmodified* research scripts that produced the published
Skysurf backtest, with one surgical change: data paths resolve from the
reproduction data bundle (see :mod:`skysurf.reproduction._paths`) instead of the
original in-repo result directories. The simulation logic is byte-for-byte the
original, which is what makes reproduction exact.

Treat this as private API — use :func:`skysurf.reproduction.reproduce`.
"""
