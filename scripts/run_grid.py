"""Run (or dry-run) a fit grid: one cached EBM fit per job, resumable (PLAN §12, M2_SPEC §5).

Usage:
    python scripts/run_grid.py configs/grid_slice.yaml [--workers 16] [--limit N] [--dry-run]
                               [--cache-root DIR] [--progress-every 50]

The cache root defaults to $EBMREGIME_FIT_CACHE, else data/cache/fits. Killing the run loses only
the fits in flight; rerunning the same command skips every finished fit.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from ebmregime.config import load_config
from ebmregime.runner import cache
from ebmregime.runner.jobs import expand_grid, fit_code_files, fit_code_hash, load_grid, prereg_hash
from ebmregime.runner.run import plan_jobs, run_jobs


def dry_run(jobs, root) -> int:
    keyed, finished = plan_jobs(jobs, root)
    total, cached = Counter(), Counter()
    for k, j in keyed:
        g = (j.data, j.target, j.protocol)
        total[g] += 1
        cached[g] += k in finished
    root_p = cache.cache_root(root)
    print(f"cache root   {root_p}")
    print(f"prereg hash  {prereg_hash()}")
    print(f"code hash    {fit_code_hash()}  ({len(fit_code_files())} files)")
    for sym in sorted({j.symbol for j in jobs}):
        from ebmregime import panel

        print(f"data hash    {sym} {panel.data_hash(sym)}")
    print(f"\n{'data':<14}{'target':<8}{'protocol':<10}{'jobs':>7}{'cached':>8}")
    for g in sorted(total):
        print(f"{g[0]:<14}{g[1]:<8}{g[2]:<10}{total[g]:>7}{cached[g]:>8}")
    n_cached = sum(cached.values())
    print(f"{'total':<32}{len(keyed):>7}{n_cached:>8}")
    print(f"\nto run: {len(keyed) - n_cached}; stale tmp files present: {len(cache.tmp_files(root))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grid", help="grid config, e.g. configs/grid_slice.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="run at most N uncached jobs (dispatch order)")
    ap.add_argument("--dry-run", action="store_true", help="print job counts and cache status only")
    ap.add_argument("--cache-root", default=None, help=f"default ${cache.ENV_VAR} or {cache.DEFAULT_ROOT}")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--max-errors", type=int, default=20)
    args = ap.parse_args(argv)

    jobs = expand_grid(load_grid(args.grid), load_config("prereg"))
    if args.dry_run:
        return dry_run(jobs, args.cache_root)
    summary = run_jobs(jobs, n_workers=args.workers, root=args.cache_root, limit=args.limit,
                       progress_every=args.progress_every, max_errors=args.max_errors)
    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
