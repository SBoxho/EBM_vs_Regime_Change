"""Leakage-free block-permutation surrogate and its fit/pair path (D-60; lead-owned).

The frozen with-replacement surrogate leaks training rows into the validation tail (D-59). Here the
source range is cut into consecutive blocks (geometric lengths, mean ``mean_block`` rows), the
blocks are put in random order without replacement and laid on the original timestamps; every source
row appears exactly once. The last H rows of every block (except the block that ends the range) get
NaN ``y_*_h{H}`` targets, so no label window reaches into another block.

Fits run through the unchanged :func:`runner.execute.run_fit_job` with the surrogate panel hook
replaced by :func:`perm_panel`, into a separate cache root with an extended code hash, so the frozen
runner cache is untouched. Data labels stay ``"surrogate:<i>"`` inside this path; the cache root and
output names (``perm_*``) tell the generators apart.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config
from .splits import stable_seed

SYMBOL, H = "BTCUSDT", 8
CACHE = ROOT / "data" / "cache" / "fits_perm"
_TCOL = re.compile(r"^y_[a-z]+_h(\d+)$")


def perm_rows(n: int, mean_block: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(src, end_dist)``: source position of each surrogate row, and for each surrogate row the
    number of source rows left in its block after it (0 = last row of its block). The block that ends
    at source position n gets ``end_dist = n`` (never purged)."""
    n = int(n)
    if n < 1 or not mean_block >= 1:
        raise ValueError("need n >= 1 and mean_block >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x9E37]))
    p = 1.0 / float(mean_block)
    starts = [0]
    while True:
        nxt = starts[-1] + int(rng.geometric(p))
        if nxt >= n:
            break
        starts.append(nxt)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.append(starts[1:], n)
    order = rng.permutation(len(starts))
    src = np.concatenate([np.arange(starts[b], ends[b]) for b in order])
    end_dist = np.concatenate([(n if ends[b] == n else 0) + (ends[b] - 1 - np.arange(starts[b], ends[b]))
                               for b in order])
    assert len(src) == n and np.array_equal(np.sort(src), np.arange(n))
    return src, end_dist


def target_horizons(columns) -> dict[str, int]:
    return {c: int(m.group(1)) for c in columns if (m := _TCOL.match(str(c)))}


def make_perm_surrogate(panel: pd.DataFrame, start, end, mean_block: float, seed: int,
                        purge: dict[str, int] | None = None) -> pd.DataFrame:
    """Block-permutation surrogate of the rows of ``panel`` in ``[start, end)`` (D-60)."""
    from .stability.surrogate import _range_rows

    rows = _range_rows(panel, start, end)
    sub = panel.iloc[rows]
    src, end_dist = perm_rows(len(sub), mean_block, seed)
    out = sub.iloc[src].copy()
    out.index = sub.index.copy()
    purge = target_horizons(out.columns) if purge is None else purge
    for col, h in purge.items():
        v = out[col].to_numpy(copy=True)
        v[end_dist < int(h)] = np.nan
        out[col] = v
    return out


def perm_seed(symbol: str, index: int, prereg=None) -> int:
    pr = prereg or load_config("prereg")
    return stable_seed(int(pr["surrogates"]["seed"]), symbol, "perm_surrogate", int(index))


@lru_cache(maxsize=4)
def perm_panel(symbol: str, index: int) -> pd.DataFrame:
    from .stability.surrogate import surrogate_bounds

    pr = load_config("prereg")
    start, end = surrogate_bounds(symbol, pr)
    return make_perm_surrogate(P.load_panel(symbol), start, end, float(pr["surrogates"]["mean_block_hours"]),
                               perm_seed(symbol, index, pr))


def _patched_surrogate_panel(symbol, index):
    return perm_panel(symbol, int(index))


def run_perm_job(job):
    """The runner's fit function with the surrogate hook pointing at :func:`perm_panel`."""
    from .runner import execute

    execute._surrogate_panel = _patched_surrogate_panel
    if job.data == "real":
        raise ValueError("perm path is for surrogate jobs only")
    return execute.run_fit_job(job)


@lru_cache(maxsize=1)
def code_hash() -> str:
    from .runner.jobs import fit_code_hash

    h = hashlib.sha256(fit_code_hash().encode())
    h.update((ROOT / "src/ebmregime/surfloor_perm.py").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def key_fn(job) -> str:
    from .runner.jobs import job_key

    return job_key(job, code=code_hash())


def cmd_fit(workers: int, n_sur: int = 20):
    import json

    from .runner.run import run_jobs
    from .surfloor import datasets, seed_jobs

    jobs = seed_jobs(datasets(n_sur))
    print(f"{len(jobs)} seed-only perm-surrogate jobs, cache {CACHE}", flush=True)
    s = run_jobs(jobs, n_workers=workers, job_fn=run_perm_job, key_fn=key_fn, root=CACHE)
    print(json.dumps(s, default=str), flush=True)
    return s


# --------------------------------------------------------------------------- pairs

@lru_cache(maxsize=64)
def _fit(key: str):
    from .runner import cache

    return cache.read(key, CACHE)


def _shapes(keys, X, grid):
    from .explain import cell_shapes_many
    from .models.ebm import eval_terms_from_export

    return cell_shapes_many(np.stack([eval_terms_from_export(_fit(k)[0], X) for k in keys]), grid)


def pair_perm_task(data, target, tau_a, tau_b, lag, keys_a, keys_b, prereg) -> list[dict]:
    from .runner import compare as C
    from .stability import floor as F

    pnl = perm_panel(SYMBOL, int(data.split(":")[1]))
    ma = C._fit_mask(pnl, tau_a, target, H, prereg)
    mb = C._fit_mask(pnl, tau_b, target, H, prereg)
    pos = C._xref([ma, mb], prereg, SYMBOL, data, target, tau_a, tau_b, "pair")
    X = pnl.iloc[pos][P.feature_names()]
    grid = C._cells(X, prereg)
    Sa, Sb = _shapes(list(keys_a), X, grid), _shapes(list(keys_b), X, grid)
    return [{"data": data, "target": target, "tau_a": tau_a, "tau_b": tau_b, "lag": lag, "feature": name,
             "g": F.cross_mean_d2(Sa[j], Sb[j], grid.weights[j]),
             "s2_seed_a": 0.5 * F.pair_mean_d2(Sa[j], grid.weights[j]),
             "s2_seed_b": 0.5 * F.pair_mean_d2(Sb[j], grid.weights[j])}
            for j, name in enumerate(P.feature_names())]


def cmd_pairs(workers: int, n_sur: int = 20):
    from joblib import Parallel, delayed

    from .runner import cache
    from .runner.compare import group_jobs
    from .surfloor import OUT, TARGETS, datasets, seed_jobs

    prereg = load_config("prereg")
    lags = list(prereg["stability"]["lags_months"])
    jobs = seed_jobs(datasets(n_sur), TARGETS, prereg)
    keys = {j: key_fn(j) for j in jobs}
    done = cache.finished_keys(CACHE)
    missing = [k for k in keys.values() if k not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} perm fits missing")
    groups = group_jobs(jobs, keys)
    tasks = []
    for (data, _, target), g in groups.items():
        taus = list(g)
        for lag in lags:
            for i in range(len(taus) - lag):
                a, b = taus[i], taus[i + lag]
                tasks.append(delayed(pair_perm_task)(data, target, a, b, lag, g[a]["seed"], g[b]["seed"], prereg))
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", batch_size=8, verbose=2)(tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    real = pd.read_parquet(OUT / "control_pairs.parquet")
    real = real[(real.data == "real") & (real.target == "shuf")]          # unchanged real-data shuf pairs
    df = pd.concat([df, real], ignore_index=True)
    df.to_parquet(OUT / "perm_control_pairs.parquet")
    print(f"{len(tasks)} perm pairs in {time.time() - t0:.0f} s", flush=True)
    return df


def leakage(gen: str = "perm", n_sur: int = 20, every: int = 1) -> pd.DataFrame:
    """D-59/D-60 diagnostic: for each surrogate window, the share of validation-tail rows that are a
    copy of a fitted training row (``dup``) or share label hours with one (``near``: fewer than H
    source rows apart). Fitted training rows exclude the rows purged before the tail, as in the fit.
    ``gen`` is "perm" (D-60) or "replace" (the frozen generator)."""
    from .splits import retrain_schedule
    from .splits.bags import tail_start
    from .stability.surrogate import _range_rows, surrogate_bounds, surrogate_panel, surrogate_rows

    pr = load_config("prereg")
    real = P.load_panel(SYMBOL)
    start, end = surrogate_bounds(SYMBOL, pr)
    rows = _range_rows(real, start, end)
    idx = real.index[rows]
    mb = float(pr["surrogates"]["mean_block_hours"])
    tail_frac = float(pr["validation"]["tail_frac"])
    taus = [str(t.date()) for t in retrain_schedule(pr["windows"]["first_tau"][SYMBOL], pr["windows"]["last_tau"])][::every]
    out = []
    for s in range(n_sur):
        if gen == "perm":
            src, _ = perm_rows(len(rows), mb, perm_seed(SYMBOL, s, pr))
            sp = perm_panel(SYMBOL, s)
        else:
            src = surrogate_rows(len(rows), mb, stable_seed(int(pr["surrogates"]["seed"]), SYMBOL, "surrogate", s))
            sp = surrogate_panel(SYMBOL, s)
        for tau in taus:
            for target in ("dir", "shuf", "vol"):
                tr = P.train_rows(sp, tau, target, H, int(pr["windows"]["train_months"]))
                vs = tail_start(sp.index[tr], tail_frac)             # as the fit: on the unpurged train rows
                fr = P.fit_rows(sp, tau, target, H, int(pr["windows"]["train_months"]), tail_frac)
                pos = np.flatnonzero(fr)
                ts = sp.index[pos]
                tail = np.asarray(ts >= vs)
                s_src = src[idx.get_indexer(ts)]
                trs = np.sort(s_src[~tail])
                k = np.searchsorted(trs, s_src[tail])
                lo = trs[np.clip(k - 1, 0, len(trs) - 1)]
                hi = trs[np.clip(k, 0, len(trs) - 1)]
                dist = np.minimum(np.abs(s_src[tail] - lo), np.abs(s_src[tail] - hi))
                out.append({"gen": gen, "s": s, "tau": tau, "target": target, "dup": float(np.mean(dist == 0)),
                            "near": float(np.mean(dist < H))})
    return pd.DataFrame(out)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = ["make_perm_surrogate", "perm_panel", "perm_rows", "run_perm_job", "key_fn", "cmd_fit", "cmd_pairs",
           "leakage", "CACHE"]
