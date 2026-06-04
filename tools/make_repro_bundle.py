#!/usr/bin/env python3
"""Assemble a reproduction data bundle from a Skysurf research checkout.

Flattens the scattered research result directories into the single-directory
bundle layout that `skysurf.reproduction` expects (see
docs/reproduction-data-bundle.md), then writes manifest.json + SHA256SUMS.

Usage:
    python tools/make_repro_bundle.py \
        --src /path/to/scripts/v2_validation \
        --out ./skysurf-repro-data \
        [--symlink] [--version 1.0.0]

By default files are copied. Use --symlink for a fast local bundle (not portable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# (relative-to-src path, bundle filename)
FILES = [
    ("val_engine_a_v2_results/regime_weekly.csv", "regime_weekly.csv"),
    ("val_engine_a_v2_results/nifty_weekly.csv", "nifty_weekly.csv"),
    ("val_engine_a_v2_results/sector_regime_weekly.csv", "sector_regime_weekly.csv"),
    ("val_engine_a_v2_results/captier_regime_weekly.csv", "captier_regime_weekly.csv"),
    ("val_engine_a_v2_results/ticker_metadata.csv", "ticker_metadata.csv"),
    ("val_engine_a_v2_results/entries_all.csv", "entries_all.csv"),
    ("val_engine_a_v2_results/entries_all_lagged.csv", "entries_all_lagged.csv"),
    ("val_entry_stats_results/trade_stats_all.csv", "trade_stats_all.csv"),
]
CACHE_DIR = ("val_engine_a_v2_results/stock_weekly_cache", "stock_weekly_cache")


def _place(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="path to scripts/v2_validation")
    ap.add_argument("--out", required=True, type=Path, help="output bundle directory")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copy (local only)")
    ap.add_argument("--version", default="1.0.0", help="bundle_version for manifest.json")
    args = ap.parse_args(argv)

    src: Path = args.src
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    missing = [rel for rel, _ in FILES if not (src / rel).exists()]
    if not (src / CACHE_DIR[0]).is_dir():
        missing.append(CACHE_DIR[0])
    if missing:
        print("ERROR: missing required inputs under --src:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    # Flat files
    for rel, name in FILES:
        _place(src / rel, out / name, args.symlink)
        print(f"  + {name}")

    # Per-ticker cache
    cache_src = src / CACHE_DIR[0]
    cache_dst = out / CACHE_DIR[1]
    cache_dst.mkdir(parents=True, exist_ok=True)
    n_cache = 0
    for f in sorted(cache_src.glob("*.csv")):
        _place(f, cache_dst / f.name, args.symlink)
        n_cache += 1
    print(f"  + stock_weekly_cache/ ({n_cache} files)")

    # Manifest
    manifest = {
        "bundle_version": args.version,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": n_cache,
        "files": sorted([name for _, name in FILES] + [CACHE_DIR[1] + "/"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("  + manifest.json")

    # Checksums (skip when symlinking — paths aren't portable)
    if not args.symlink:
        lines = []
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                lines.append(f"{_sha256(p)}  {p.relative_to(out)}")
        (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        print(f"  + SHA256SUMS ({len(lines)} entries)")

    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nBundle ready at {out}  (~{size_mb:.0f} MB)")
    print("Verify with:  cd", out, "&& sha256sum -c SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
