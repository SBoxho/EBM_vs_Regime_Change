"""Download and verify Binance Vision spot monthly klines into data/raw (never committed).

Usage:
    python scripts/download_data.py                 # everything in configs/data.yaml
    python scripts/download_data.py --symbols BTCUSDT --intervals 15m
    python scripts/download_data.py --dry-run       # list what would be fetched

The script can be re-run at any time. Files already on disk are re-verified against the
published SHA-256 and are not downloaded again. data/manifest.csv records every file.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ebmregime.config import load_config, repo_path
from ebmregime.data.download import (ManifestRow, fetch_verified, make_fetcher, manifest_hash,
                                     now_utc, write_manifest)
from ebmregime.data.listing import klines_prefix, list_prefix, month_of, monthly_zips


def expected_months(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="data")
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--intervals", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    arc = cfg["archive"]
    symbols = args.symbols or list(cfg["symbols"])
    intervals = args.intervals or sorted({cfg["intervals"]["base"], cfg["intervals"]["crosscheck"]})
    raw_root = repo_path(cfg["paths"]["raw"])
    fetch = make_fetcher(arc["timeout_s"], arc["retries"])

    jobs = []
    for sym in symbols:
        for iv in intervals:
            objs = monthly_zips(list_prefix(arc["list_url"], klines_prefix(arc["market"], sym, iv), fetch),
                                cfg["last_month"])
            if not objs:
                print(f"ERROR {sym} {iv}: no files listed", file=sys.stderr)
                return 2
            months = [month_of(o.filename) for o in objs]
            gap = sorted(set(expected_months(months[0], months[-1])) - set(months))
            print(f"{sym} {iv}: {len(objs)} files {months[0]}..{months[-1]}, "
                  f"{sum(o.size for o in objs) / 1e6:.1f} MB" + (f", MISSING MONTHS {gap}" if gap else ""))
            jobs += [(sym, iv, o) for o in objs]
    if args.dry_run:
        return 0

    rows, errors, n_new = [], [], 0

    def run(sym, iv, o):
        url = f"{arc['base_url']}/{o.key}"
        sha, new = fetch_verified(url, raw_root / iv / sym / o.filename, fetch)
        return ManifestRow(sym, iv, o.filename, url, sha, o.size, o.last_modified,
                           now_utc() if new else ""), new

    with ThreadPoolExecutor(max_workers=arc["workers"]) as pool:
        futs = {pool.submit(run, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                row, new = f.result()
                rows.append(row)
                n_new += new
            except Exception as e:  # keep going; report at the end
                errors.append((futs[f][2].filename, repr(e)))
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files checked, {n_new} downloaded, {len(errors)} errors", flush=True)

    manifest = repo_path(cfg["paths"]["manifest"])
    write_manifest(manifest, rows)
    print(f"manifest: {manifest} ({len(rows)} rows updated), hash {manifest_hash(manifest)[:16]}")
    for name, err in errors:
        print(f"ERROR {name}: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
