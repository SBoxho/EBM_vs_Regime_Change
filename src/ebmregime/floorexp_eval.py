"""Evaluation of the floor-fix experiment (D-55): G0 per candidate floor on the control arms."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config
from .floorexp import ARMS, H, N_BOOT, N_HALF_PART, N_QUART_PART, SYMBOL, fit_path, tasks_for

OUT = ROOT / "results" / "floor_exp"



def _shapes_from_files(paths, X, grid):
    from ebmregime.explain import cell_shapes_many
    from ebmregime.models.ebm import eval_terms_from_export
    from ebmregime.runner.cache import read_path
    C = np.stack([eval_terms_from_export(read_path(p)[0], X) for p in paths])
    return cell_shapes_many(C, grid)


def eval_pair(data, target, ta, tb, ga, gb, prereg):
    from ebmregime.runner import compare as Cm
    from ebmregime.stability.floor import cross_mean_d2, pair_mean_d2

    pnl = Cm._panel(SYMBOL, data)
    ma, mb = (Cm._fit_mask(pnl, t, target, H, prereg) for t in (ta, tb))
    pos = Cm._xref([ma, mb], prereg, SYMBOL, data, target, ta, tb, "pair")       # same X_ref as the compare stage
    X = pnl.iloc[pos][P.feature_names()]
    grid = Cm._cells(X, prereg)
    J = len(grid.weights)
    S, Hh = {}, {}
    for t, g in ((ta, ga), (tb, gb)):
        S[t], _ = Cm._shapes_for(g["seed"], X, grid)
        Hh[t], _ = Cm._shapes_for(g["half"], X, grid)
    out = {"tau_a": ta, "tau_b": tb}
    d2c = np.array([cross_mean_d2(S[ta][j], S[tb][j], grid.weights[j]) for j in range(J)])
    fl = {}
    for t, tag in ((ta, "a"), (tb, "b")):
        w = grid.weights
        boot = _shapes_from_files([fit_path(data, target, t, "boot", r, 0) for r in range(N_BOOT)], X, grid)
        halves = {p: _shapes_from_files([fit_path(data, target, t, "half", p, k) for k in range(2)], X, grid)
                  for p in range(N_HALF_PART)}
        quarts = {p: _shapes_from_files([fit_path(data, target, t, "quart", p, k) for k in range(4)], X, grid)
                  for p in range(N_QUART_PART)}
        s_seed = np.array([0.5 * pair_mean_d2(S[t][j], w[j]) for j in range(J)])
        s_iii = np.array([0.5 * pair_mean_d2(Hh[t][j], w[j]) for j in range(J)])
        s_boot = np.array([0.5 * pair_mean_d2(boot[j], w[j]) for j in range(J)])
        v2 = np.array([np.mean([0.5 * pair_mean_d2(halves[p][j], w[j]) for p in halves]) for j in range(J)])
        v4 = np.array([np.mean([0.5 * pair_mean_d2(quarts[p][j], w[j]) for p in quarts]) for j in range(J)])
        beta = float(np.log2(v4.sum() / v2.sum()))
        s_R = v2 * 2.0 ** (-beta)
        a2_full = sum(float(np.mean(np.sum(w[j] * S[t][j] ** 2, axis=1))) for j in range(J))
        a2_boot = sum(float(np.mean(np.sum(w[j] * boot[j] ** 2, axis=1))) for j in range(J))
        fl[tag] = dict(seed=s_seed, iii=s_iii, B=s_boot, R=s_R)
        out.update({f"beta_{tag}": beta, f"V2_{tag}": float(v2.sum()), f"V4_{tag}": float(v4.sum()),
                    f"A2_full_{tag}": a2_full, f"A2_boot_{tag}": a2_boot,
                    f"s2_seed_{tag}": float(s_seed.sum())})
    out["d2_cross"] = float(d2c.sum())
    for c in ("iii", "B", "R"):
        out[f"floor_{c}_a"] = float(fl["a"][c].sum())
        out[f"floor_{c}_b"] = float(fl["b"][c].sum())
        out[f"DR_{c}"] = float(d2c.sum() / (fl["a"][c].sum() + fl["b"][c].sum()))
    return out


def eval_job(data, target, a, b, ga, gb, prereg):
    return {"data": data, "target": target, **eval_pair(data, target, a, b, ga, gb, prereg)}


def cmd_eval(workers):
    from ebmregime.runner.compare import group_jobs
    from ebmregime.runner.jobs import expand_grid
    from ebmregime.stats import median_ci

    prereg = load_config("prereg")
    taus, tasks = tasks_for(prereg)
    missing = [t for t in tasks if not fit_path(*t).exists()]
    if missing:
        sys.exit(f"{len(missing)} experiment fits missing")
    groups = group_jobs(expand_grid(load_config("configs/grid_slice.yaml"), prereg))
    lag = int(prereg["stability"]["headline_lag_months"])
    jobs = []
    for data, target in ARMS:
        g = groups[(data, SYMBOL, target)]
        for i in range(len(taus) - lag):
            a, b = taus[i], taus[i + lag]
            jobs.append((data, target, a, b, g[a], g[b]))
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=workers, backend="loky", batch_size=2)(delayed(eval_job)(*j, prereg) for j in jobs)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lag12_pairs.csv", index=False)
    band = prereg["hypotheses"]["g0"]["pass_ci_within"]
    summ = []
    for (data, target), s in df.groupby(["data", "target"], sort=False):
        s = s.sort_values("tau_a")
        for c in ("iii", "B", "R"):
            ci = median_ci(s[f"DR_{c}"].values, delta=lag)
            summ.append({"arm": f"{data} {target}", "floor": c, "n": ci["n"], "median": ci["est"], "lo": ci["lo"],
                         "hi": ci["hi"], "pass": bool(ci["lo"] >= band[0] and ci["hi"] <= band[1])})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "g0_by_floor.csv", index=False)
    print(summ.round(3).to_string())
    diag = df.groupby(["data", "target"]).agg(beta=("beta_a", "median"), A2_full=("A2_full_a", "median"),
                                              A2_boot=("A2_boot_a", "median"), V2=("V2_a", "median"),
                                              V4=("V4_a", "median"), seed=("s2_seed_a", "median"),
                                              floor_B=("floor_B_a", "median"), floor_R=("floor_R_a", "median"),
                                              floor_iii=("floor_iii_a", "median"), d2_cross=("d2_cross", "median"))
    print(diag.to_string(float_format=lambda x: f"{x:.5f}"))
    (OUT / "summary.json").write_text(json.dumps({"g0": summ.to_dict("records"),
                                                  "diag": diag.reset_index().to_dict("records")}, indent=1))


