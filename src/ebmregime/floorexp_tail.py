"""Exploratory "B-tail" floor (D-57; not eligible for adoption): bootstrap floor B with a resampled
validation tail. Tests whether B's under-dispersion (D-56) comes from the tail shared by all replicates.

Training rows as in candidate B (:func:`floorexp.bootstrap_bags`). Validation-tail rows are repeated
according to stationary-bootstrap counts (mean block 168 rows) drawn over the tail; every copy is -1
in all bags; tail rows drawn 0 times are left out. 10 fits per window on the three control arms.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config
from .floorexp import ARMS, H, MEAN_BLOCK, N_BOOT, SYMBOL, _panel, bootstrap_bags
from .splits import stable_seed

OUT = ROOT / "results" / "floor_exp"


def code_tag() -> str:
    h = hashlib.sha256()
    for rel in ("src/ebmregime/floorexp_tail.py", "src/ebmregime/floorexp.py", "src/ebmregime/models/ebm.py",
                "src/ebmregime/splits/bags.py", "src/ebmregime/stability/surrogate.py"):
        h.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def fit_path(data, target, tau, r):
    return ROOT / "data" / "cache" / "floor_exp_tail" / code_tag() / f"{data.replace(':', '-')}_{target}" / tau / f"{r}.npz"


def tail_rows_bags(ts, bags, seed):
    """(rep, bags_rep): row positions with tail rows repeated by bootstrap counts, and their bags."""
    from .stability.surrogate import surrogate_rows

    tail = np.flatnonzero((bags == -1).all(axis=1))
    draws = surrogate_rows(len(tail), MEAN_BLOCK, seed)
    mult = np.ones(len(ts), dtype=np.int64)
    mult[tail] = np.bincount(draws, minlength=len(tail))
    rep = np.repeat(np.arange(len(ts)), mult)
    return rep, bags[rep]


def fit_task(data, target, tau, r, prereg):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .models.ebm import export_fit, fit_ebm, kind_for_target
    from .runner.cache import write_atomic

    path = fit_path(data, target, tau, r)
    if path.exists():
        return None
    v = prereg["validation"]
    base = int(prereg["replicates"]["base_seed"])
    pnl = _panel(data)
    tr = P.train_rows(pnl, tau, target, H, int(prereg["windows"]["train_months"]))
    pos = np.flatnonzero(tr)
    ts = pnl.index[pos]
    seed = stable_seed(base, "floorexp_tail", SYMBOL, tau, target, data, r)
    bags = bootstrap_bags(ts, H, int(v["outer_bags"]), tail_frac=float(v["tail_frac"]), block_hours=int(v["block_hours"]),
                          bag_train_frac=float(v["bag_train_frac"]), mean_block=MEAN_BLOCK, seed=seed)
    rep, bags_rep = tail_rows_bags(ts, bags, seed ^ 0x7A11)
    rows = pos[rep]
    X = pnl[P.feature_names()].iloc[rows]
    y = pnl[P.target_column(target, H)].to_numpy()[rows]
    kind = kind_for_target(target)
    if kind == "classifier":
        y = y.astype(np.int64)
    model, info = fit_ebm(X, y, bags_rep, kind, seed)
    arrays = dict(export_fit(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, arrays, {"data": data, "target": target, "tau": tau, "r": r, "seed": seed,
                                "n_rows_fit": info["n_rows_fit"], "n_tail_copies": int((bags_rep == -1).all(axis=1).sum())})
    return info["fit_seconds"]


def eval_pair(data, target, ta, tb, ga, gb, prereg):
    from .floorexp_eval import _shapes_from_files
    from .runner import compare as Cm
    from .stability.floor import cross_mean_d2, pair_mean_d2

    pnl = Cm._panel(SYMBOL, data)
    ma, mb = (Cm._fit_mask(pnl, t, target, H, prereg) for t in (ta, tb))
    pos = Cm._xref([ma, mb], prereg, SYMBOL, data, target, ta, tb, "pair")
    X = pnl.iloc[pos][P.feature_names()]
    grid = Cm._cells(X, prereg)
    w = grid.weights
    S = {t: Cm._shapes_for(g["seed"], X, grid)[0] for t, g in ((ta, ga), (tb, gb))}
    d2c = sum(cross_mean_d2(S[ta][j], S[tb][j], w[j]) for j in range(len(w)))
    fl = 0.0
    for t in (ta, tb):
        B = _shapes_from_files([fit_path(data, target, t, r) for r in range(N_BOOT)], X, grid)
        fl += sum(0.5 * pair_mean_d2(B[j], w[j]) for j in range(len(w)))
    return {"data": data, "target": target, "tau_a": ta, "tau_b": tb, "d2_cross": d2c, "floor_Btail": fl,
            "DR_Btail": d2c / fl}


def cmd_tail(workers: int):
    from joblib import Parallel, delayed

    from .floorexp import tasks_for
    from .runner.compare import group_jobs
    from .runner.jobs import expand_grid
    from .stats import median_ci

    prereg = load_config("prereg")
    taus, _ = tasks_for(prereg)
    todo = [(d, t, tau, r) for d, t in sorted(ARMS, key=lambda a: a[1] != "vol") for tau in taus for r in range(N_BOOT)
            if not fit_path(d, t, tau, r).exists()]
    print(f"B-tail: {len(todo)} fits to run, tag {code_tag()}", flush=True)
    t0 = time.time()
    Parallel(n_jobs=workers, backend="loky", batch_size=1, verbose=5)(delayed(fit_task)(*x, prereg) for x in todo)
    print(f"fits done in {(time.time() - t0) / 60:.1f} min", flush=True)
    groups = group_jobs(expand_grid(load_config("configs/grid_slice.yaml"), prereg))
    lag = int(prereg["stability"]["headline_lag_months"])
    jobs = [(d, t, taus[i], taus[i + lag], groups[(d, SYMBOL, t)][taus[i]], groups[(d, SYMBOL, t)][taus[i + lag]])
            for d, t in ARMS for i in range(len(taus) - lag)]
    df = pd.DataFrame(Parallel(n_jobs=workers, backend="loky", batch_size=2)(delayed(eval_pair)(*j, prereg) for j in jobs))
    df.to_csv(OUT / "lag12_pairs_Btail.csv", index=False)
    rows = []
    for (d, t), s in df.groupby(["data", "target"], sort=False):
        ci = median_ci(s.sort_values("tau_a").DR_Btail.values, delta=lag)
        rows.append({"arm": f"{d} {t}", "floor": "B-tail", "n": ci["n"], "median": ci["est"], "lo": ci["lo"], "hi": ci["hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "g0_Btail.csv", index=False)
    print(out.round(3).to_string())
    return out
