"""D-67 (2): surrogate block-length sensitivity (lead-owned).

The perm generator of :mod:`surfloor_perm` (unchanged) with mean block 720 h or 2,160 h,
surrogates 0-8, targets dir and vol, seed-only fits in a separate cache per block length.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import panel as P
from . import surfloor as S
from . import surfloor_perm as Q
from .config import REPO_ROOT as ROOT
from .config import load_config
from .splits import stable_seed

BLOCKS = (720, 2160)
N_SUR = 9
TARGETS = ("dir", "vol")
SYMBOL, H = S.SYMBOL, S.H


def cache_root(mb: int):
    return ROOT / "data" / "cache" / f"fits_perm_mb{int(mb)}"


@lru_cache(maxsize=4)
def block_panel(symbol: str, index: int, mb: int) -> pd.DataFrame:
    from .stability.surrogate import surrogate_bounds

    pr = load_config("prereg")
    start, end = surrogate_bounds(symbol, pr)
    seed = stable_seed(int(pr["surrogates"]["seed"]), symbol, "perm_surrogate_mb", int(mb), int(index))
    return Q.make_perm_surrogate(P.load_panel(symbol), start, end, float(mb), seed)


def _run(job, mb: int):
    from .runner import execute

    execute._surrogate_panel = lambda symbol, index: block_panel(symbol, int(index), mb)
    if job.data == "real":
        raise ValueError("block path is for surrogate jobs only")
    return execute.run_fit_job(job)


def run_job_720(job):
    return _run(job, 720)


def run_job_2160(job):
    return _run(job, 2160)


JOB_FN = {720: run_job_720, 2160: run_job_2160}


@lru_cache(maxsize=4)
def code_hash(mb: int) -> str:
    h = hashlib.sha256(Q.code_hash().encode())
    h.update((ROOT / "src/ebmregime/surfloor_block.py").read_bytes().replace(b"\r\n", b"\n"))
    h.update(f"mb={int(mb)}".encode())
    return h.hexdigest()


def key_fn_for(mb: int):
    from .runner.jobs import job_key

    return lambda job: job_key(job, code=code_hash(mb))


def jobs(prereg=None):
    return S.seed_jobs(S.datasets(N_SUR), TARGETS, prereg)


def cmd_fit(workers: int):
    from .runner.run import run_jobs

    out = {}
    for mb in BLOCKS:
        js = jobs()
        print(f"mean block {mb} h: {len(js)} jobs, cache {cache_root(mb)}", flush=True)
        out[mb] = run_jobs(js, n_workers=workers, job_fn=JOB_FN[mb], key_fn=key_fn_for(mb), root=cache_root(mb))
        print(json.dumps(out[mb], default=str), flush=True)
    return out


# --------------------------------------------------------------------------- pairs

@lru_cache(maxsize=64)
def _fit(key: str, mb: int):
    from .runner import cache

    return cache.read(key, cache_root(mb))


def pair_task(mb, data, target, tau_a, tau_b, lag, keys_a, keys_b, prereg) -> list[dict]:
    from .explain import cell_shapes_many
    from .models.ebm import eval_terms_from_export
    from .runner import compare as C
    from .stability import floor as F

    pnl = block_panel(SYMBOL, int(data.split(":")[1]), mb)
    ma = C._fit_mask(pnl, tau_a, target, H, prereg)
    mb_ = C._fit_mask(pnl, tau_b, target, H, prereg)
    pos = C._xref([ma, mb_], prereg, SYMBOL, data, target, tau_a, tau_b, "pair")
    X = pnl.iloc[pos][P.feature_names()]
    grid = C._cells(X, prereg)

    def shapes(keys):
        return cell_shapes_many(np.stack([eval_terms_from_export(_fit(k, mb)[0], X) for k in keys]), grid)

    Sa, Sb = shapes(list(keys_a)), shapes(list(keys_b))
    return [{"mb": mb, "data": data, "target": target, "tau_a": tau_a, "tau_b": tau_b, "lag": lag, "feature": name,
             "g": F.cross_mean_d2(Sa[j], Sb[j], grid.weights[j]),
             "s2_seed_a": 0.5 * F.pair_mean_d2(Sa[j], grid.weights[j]),
             "s2_seed_b": 0.5 * F.pair_mean_d2(Sb[j], grid.weights[j])}
            for j, name in enumerate(P.feature_names())]


def cmd_pairs(workers: int) -> pd.DataFrame:
    from joblib import Parallel, delayed

    from .runner import cache
    from .runner.compare import group_jobs

    prereg = load_config("prereg")
    lags = list(prereg["stability"]["lags_months"])
    tasks = []
    for mb in BLOCKS:
        js = jobs(prereg)
        kf = key_fn_for(mb)
        keys = {j: kf(j) for j in js}
        done = cache.finished_keys(cache_root(mb))
        missing = [k for k in keys.values() if k not in done]
        if missing:
            raise RuntimeError(f"mb {mb}: {len(missing)} fits missing")
        for (data, _, target), g in group_jobs(js, keys).items():
            taus = list(g)
            for lag in lags:
                for i in range(len(taus) - lag):
                    a, b = taus[i], taus[i + lag]
                    tasks.append(delayed(pair_task)(mb, data, target, a, b, lag, g[a]["seed"], g[b]["seed"], prereg))
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", batch_size=8, verbose=2)(tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(S.OUT / "block_pairs.parquet")
    print(f"{len(tasks)} pairs in {time.time() - t0:.0f} s", flush=True)
    return df


# --------------------------------------------------------------------------- summary

def summarise(pf: pd.DataFrame | None = None) -> pd.DataFrame:
    """D-67 (2): real dir and vol in the pool real + 9 surrogates, per block length and lag."""
    pf = pd.read_parquet(S.OUT / "block_pairs.parquet") if pf is None else pf
    real = pd.read_parquet(S.OUT / "real_pairs.parquet")
    pool = ["real"] + S.datasets(N_SUR)
    rows = []
    for mb in BLOCKS:
        tot = S.totals(pd.concat([pf[pf.mb == mb].drop(columns="mb"), real], ignore_index=True))
        for target in TARGETS:
            for lag in sorted(tot.lag.unique()):
                w = S._wide(tot, target, int(lag), pool)
                st = S.pooled_stats(w)
                cal = S.pooled_stats(w[pool[1:]])
                rows.append({"mean_block": mb, **S._real_row(st, target, int(lag), False),
                             "cal_median_M": float(cal.M.median()), "cal_M_min": float(cal.M.min()),
                             "cal_M_max": float(cal.M.max())})
    out = pd.DataFrame(rows)
    out.to_csv(S.OUT / "block_sensitivity.csv", index=False)
    print(out[["mean_block", "target", "lag", "M_real", "U_real", "rank_from_top", "p_upper", "M_null_max",
               "cal_median_M", "cal_M_min", "cal_M_max"]].round(3).to_string())
    return out
