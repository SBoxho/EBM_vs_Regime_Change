"""Parse the downloaded zips into cleaned grids and 1h bars under data/interim (never committed).

Outputs, per symbol:
    data/interim/grid_15m/<SYM>.parquet          15m grid with present/valid flags
    data/interim/bars_1h/<SYM>.parquet           1h bars aggregated from 15m (valid iff 4/4 sub-bars)
    data/interim/archive_1h/<SYM>.parquet        the archive's own 1h grid (cross-check only)
    data/interim/file_units/<SYM>_<iv>.csv       per-file timestamp unit (ms/us) and row count
    data/interim/crosscheck/<SYM>.csv            aggregated-vs-archive 1h comparison
    data/interim/build_info.json                 manifest hash and package versions

Usage:
    python scripts/build_bars.py [--symbols BTCUSDT ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

from ebmregime.config import load_config, repo_path
from ebmregime.data.bars import aggregate, apply_archive_flags, crosscheck, to_grid
from ebmregime.data.download import manifest_hash, verify_against_manifest
from ebmregime.data.parse import load_symbol


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="data")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--skip-verify", action="store_true", help="skip re-hashing raw files")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    raw, interim = repo_path(cfg["paths"]["raw"]), repo_path(cfg["paths"]["interim"])
    manifest = repo_path(cfg["paths"]["manifest"])
    base, model, xcheck = (cfg["intervals"][k] for k in ("base", "model", "crosscheck"))

    if not args.skip_verify:
        bad = verify_against_manifest(manifest, raw)
        if bad:
            print(f"ERROR: {len(bad)} raw files missing or not matching the manifest, e.g. {bad[:3]}",
                  file=sys.stderr)
            return 1
        print("raw files verified against manifest")

    for sub in ("grid_15m", "bars_1h", "archive_1h", "file_units", "crosscheck"):
        (interim / sub).mkdir(parents=True, exist_ok=True)

    for sym in args.symbols or list(cfg["symbols"]):
        kl, units = load_symbol(list((raw / base / sym).glob("*.zip")), base)
        units.to_csv(interim / "file_units" / f"{sym}_{base}.csv", index=False)
        grid = to_grid(kl, base)
        grid.to_parquet(interim / "grid_15m" / f"{sym}.parquet")
        bars = aggregate(grid, base, model)

        kl1, units1 = load_symbol(list((raw / xcheck / sym).glob("*.zip")), xcheck)
        units1.to_csv(interim / "file_units" / f"{sym}_{xcheck}.csv", index=False)
        arch = to_grid(kl1, xcheck)
        arch.to_parquet(interim / "archive_1h" / f"{sym}.parquet")
        xc = crosscheck(bars, arch)
        xc.to_csv(interim / "crosscheck" / f"{sym}.csv", index=False)
        bars = apply_archive_flags(bars, arch)
        bars.to_parquet(interim / "bars_1h" / f"{sym}.parquet")
        print(f"{sym}: {len(grid)} {base} slots, {int(bars['valid'].sum())}/{len(bars)} valid {model} bars; "
              f"crosscheck over {int(xc['compared'].iloc[0])} bars: price-conflict hours (now invalid) "
              f"{int(bars['price_conflict'].sum())}, volume-suspect hours {int(bars['volume_suspect'].sum())}")

    info = {"manifest_hash": manifest_hash(manifest),
            "packages": {p: metadata.version(p) for p in ("pandas", "numpy", "pyarrow")}}
    (interim / "build_info.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
