"""Second stage: per-window and per-pair explanation metrics from cached fits (PLAN §6, §12; lead-owned).

Inputs are the fits written by :mod:`ebmregime.runner.run` for a grid config. Nothing is refitted.

* **Window stage** (one row per data variant, target and tau): X_ref = the window's own fit rows
  (D-50); cells; shapes of all 15 fits; per feature sigma2_seed, sigma2_tot, sigma2_samp, amplitude and
  SNR (mean over the seed-only fits of A^2 / sigma2_tot); seed share; test-month skill of the
  seed-ensemble mean (D-50): R2_OOS against the training climatology for vol, log-loss skill against
  the training base rate for dir and shuf; seed-mean shapes for the gallery.
* **Pair stage** (one row per data variant, target, lag and t): X_ref = union of both windows' fit
  rows; cells; shapes of the 2 x 15 fits; :func:`ebmregime.stability.floor.drift_pair`; prediction
  distance of the seed-only fits (cross mean of mean-square eta differences) and its half-sample
  floor.

Outputs (parquet) in ``out_dir``: ``windows.parquet``, ``window_features.parquet``,
``pairs.parquet``, ``pair_features.parquet``, ``shapes.parquet``, ``fits.parquet``.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .. import panel as P
from ..config import load_config
from ..splits import overlap_fraction, stable_seed


# ---------------------------------------------------------------------------------------------
# grouping of jobs


def group_jobs(jobs, keys: dict | None = None) -> dict:
    """{(data, symbol, target): {tau: {"seed": [key...], "half": [key...]}}}, replicates sorted by r.

    Keys are computed once here (in the parent) with :func:`runner.jobs.job_key`, so workers never
    recompute code or data hashes; pass ``keys`` (job -> key) to reuse precomputed ones.
    """
    from .jobs import job_key

    g: dict = defaultdict(lambda: defaultdict(lambda: {"seed": [], "half": []}))
    for j in sorted(jobs, key=lambda j: j.r):
        k = keys[j] if keys is not None else job_key(j)
        g[(j.data, j.symbol, j.target)][j.tau][j.protocol].append(k)
    return {k: dict(sorted(v.items())) for k, v in g.items()}


# ---------------------------------------------------------------------------------------------
# worker-side helpers (cached per process)


@lru_cache(maxsize=4)
def _panel(symbol: str, data: str) -> pd.DataFrame:
    if data == "real":
        return P.load_panel(symbol)
    from ..stability.surrogate import surrogate_panel

    kind, idx = data.split(":")
    assert kind == "surrogate"
    return surrogate_panel(symbol, int(idx))


@lru_cache(maxsize=64)
def _fit(key: str) -> tuple[dict, dict]:
    from . import cache

    return cache.read(key)


def _keys(group: dict, protocol: str) -> list[str]:
    return list(group[protocol])


def _shapes_for(keys: list[str], X: pd.DataFrame, grid):
    """Per feature j: (R, K_j) re-centred shapes; plus (R, n) eta."""
    from ..explain import cell_shapes_many
    from ..models.ebm import eval_terms_from_export

    C, etas = [], []
    for k in keys:
        arrays, _ = _fit(k)
        c = eval_terms_from_export(arrays, X)
        C.append(c)
        etas.append(float(arrays["intercept"][0]) + c.sum(axis=1))
    C = np.stack(C)                                   # (R, n, J)
    return cell_shapes_many(C, grid), np.stack(etas)


def _xref(masks: list[np.ndarray], prereg: dict, *seed_parts) -> np.ndarray:
    from ..explain import make_xref

    xr = prereg["explain"]["x_ref"]
    return make_xref(masks, cap=int(xr["cap_rows"]), seed=stable_seed(int(xr["seed"]), *seed_parts))


def _cells(X: pd.DataFrame, prereg: dict):
    from ..explain import make_cells

    c = prereg["explain"]["cells"]
    return make_cells(X, n_cells=int(c["n"]), min_rows=int(c["min_rows"]), nominal=P.nominal_features())


def _fit_mask(pnl, tau, target, h, prereg):
    return P.fit_rows(pnl, tau, target, h, int(prereg["windows"]["train_months"]),
                      float(prereg["validation"]["tail_frac"]))


def _mean_pair_sq(E: np.ndarray) -> float:
    """Mean over distinct row pairs of mean_i (E_r,i - E_s,i)^2 (E: (R, n))."""
    R = len(E)
    v = E.var(axis=0, ddof=1).mean()                  # identity: mean pair sq diff = 2 * var(ddof=1)
    return float(2.0 * v) if R > 1 else np.nan


# ---------------------------------------------------------------------------------------------
# window stage


def window_task(data, symbol, target, tau, group, h, prereg) -> dict:
    from ..stability import floor as F
    from ..stability import metrics as M

    pnl = _panel(symbol, data)
    fm = _fit_mask(pnl, tau, target, h, prereg)
    pos = _xref([fm], prereg, symbol, data, target, tau, "window")
    X = pnl.iloc[pos][P.feature_names()]
    grid = _cells(X, prereg)
    keys_s, keys_h = _keys(group, "seed"), _keys(group, "half")
    S, _ = _shapes_for(keys_s, X, grid)
    Hh, _ = _shapes_for(keys_h, X, grid)
    feats = P.feature_names()
    frows, shape_rows = [], []
    for j, name in enumerate(feats):
        w = grid.weights[j]
        s2 = F.sigma2(S[j], Hh[j], w)
        A2 = M.amplitude(S[j], w) ** 2                # (R_seed,)
        mean_shape = S[j].mean(axis=0)
        frows.append({"feature": name, "s2_seed": s2["seed"], "s2_tot": s2["tot"], "s2_samp": s2["samp"],
                      "A2_mean": float(np.mean(A2)), "A_seedmean": float(M.amplitude(mean_shape, w)),
                      "snr": float(np.mean(A2) / s2["tot"]) if s2["tot"] > 0 else np.nan,
                      "n_cells": len(w), "fallback": bool(grid.fallback_used[j]),
                      "importance": float(M.importance(mean_shape, w))})
        sd = Hh[j].std(axis=0, ddof=1)
        for k in range(len(w)):
            shape_rows.append({"feature": name, "cell": k, "x_rep": float(grid.x_rep[j][k]), "w": float(w[k]),
                               "f_seed_mean": float(mean_shape[k]), "f_half_sd": float(sd[k])})
    # skill of the seed-ensemble mean on the test month
    preds, ys, ytr, ts = [], None, [], None
    meta_rows = []
    for k in keys_s + keys_h:
        arrays, meta = _fit(k)
        fi = meta.get("fit_info", {})
        meta_rows.append({"key": k, "protocol": meta["job"]["protocol"], "r": meta["job"]["r"],
                          "fit_seconds": fi.get("fit_seconds"), "n_rows_fit": fi.get("n_rows_fit"),
                          "best_iteration_median": float(np.median(arrays["best_iteration"])),
                          "y_train_mean": meta.get("y_train_mean")})
    for k in keys_s:
        arrays, meta = _fit(k)
        preds.append(arrays["test_pred"].astype(float))
        ys, ts = arrays["test_y"].astype(float), arrays["test_ts"]
        ytr.append(float(meta["y_train_mean"]))
    p = np.mean(preds, axis=0)
    clim = float(np.mean(ytr))
    if target == "vol":
        skill = 1.0 - np.sum((ys - p) ** 2) / np.sum((ys - clim) ** 2)
        per_fit = [1.0 - np.sum((ys - q) ** 2) / np.sum((ys - clim) ** 2) for q in preds]
    else:
        eps = 1e-12
        ll = lambda q: -np.mean(ys * np.log(np.clip(q, eps, 1)) + (1 - ys) * np.log(np.clip(1 - q, eps, 1)))  # noqa: E731
        base = ll(np.full_like(ys, clim))
        skill = 1.0 - ll(p) / base
        per_fit = [1.0 - ll(q) / base for q in preds]
    S2s = sum(r["s2_seed"] for r in frows)
    S2t = sum(r["s2_tot"] for r in frows)
    row = {"data": data, "symbol": symbol, "target": target, "tau": tau, "n_ref": len(pos), "n_test": len(ys),
           "skill": float(skill), "skill_per_fit_mean": float(np.mean(per_fit)), "climatology": clim,
           "seed_share": S2s / S2t if S2t > 0 else np.nan, "s2_seed_sum": S2s, "s2_tot_sum": S2t}
    head = {"data": data, "symbol": symbol, "target": target, "tau": tau}
    return {"window": row, "features": [{**head, **r} for r in frows],
            "shapes": [{**head, **r} for r in shape_rows], "fits": [{**head, **r} for r in meta_rows]}


# ---------------------------------------------------------------------------------------------
# pair stage


def pair_task(data, symbol, target, tau_a, tau_b, lag, ga, gb, h, prereg) -> dict:
    from ..stability.floor import drift_pair

    pnl = _panel(symbol, data)
    ma = _fit_mask(pnl, tau_a, target, h, prereg)
    mb = _fit_mask(pnl, tau_b, target, h, prereg)
    pos = _xref([ma, mb], prereg, symbol, data, target, tau_a, tau_b, "pair")
    X = pnl.iloc[pos][P.feature_names()]
    grid = _cells(X, prereg)
    Sa, Ea = _shapes_for(_keys(ga, "seed"), X, grid)
    Ha, EHa = _shapes_for(_keys(ga, "half"), X, grid)
    Sb, Eb = _shapes_for(_keys(gb, "seed"), X, grid)
    Hb, EHb = _shapes_for(_keys(gb, "half"), X, grid)
    rho = overlap_fraction(tau_a, tau_b, int(prereg["windows"]["train_months"]))
    nominal = set(P.nominal_features())
    x_rep = [None if n in nominal else grid.x_rep[j] for j, n in enumerate(P.feature_names())]
    res = drift_pair(_pack(Sa, Ha), _pack(Sb, Hb), grid.weights, rho, x_rep=x_rep)
    # prediction distance: cross mean over seed pairs, and the half-sample floor of each window
    d2p = float(np.mean([np.mean((ea - eb) ** 2) for ea in Ea for eb in Eb]))
    fp_a, fp_b = 0.5 * _mean_pair_sq(EHa), 0.5 * _mean_pair_sq(EHb)
    head = {"data": data, "symbol": symbol, "target": target, "tau_a": tau_a, "tau_b": tau_b, "lag": lag}
    row = {**head, "overlap": rho, "n_ref": len(pos), "d2_pred": d2p, "s2_pred_tot_a": fp_a,
           "s2_pred_tot_b": fp_b, "DR_pred": d2p / (fp_a + fp_b) if fp_a + fp_b > 0 else np.nan}
    feats = []
    for key, val in res.items():
        if np.ndim(val) == 0:
            row[key] = float(val)
    for j, name in enumerate(P.feature_names()):
        r = {**head, "feature": name}
        for key, val in res.items():
            if np.ndim(val) == 1 and len(val) == len(P.feature_names()):
                r[key] = float(val[j])
        feats.append(r)
    return {"pair": row, "features": feats}


def _pack(S, H):
    """Shapes for one window in the layout drift_pair expects."""
    return {"seed": S, "half": H}


# ---------------------------------------------------------------------------------------------
# driver


def run_compare(grid_path: str | Path, out_dir: str | Path, n_workers: int = 16, lags=None,
                verbose: bool = True) -> dict:
    from .jobs import expand_grid

    prereg = load_config("prereg")
    grid_cfg = load_config(grid_path)
    jobs = expand_grid(grid_cfg, prereg)
    groups = group_jobs(jobs)
    from . import cache
    done = cache.finished_keys()
    missing = [k for g in groups.values() for d in g.values() for p in d.values() for k in p if k not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} fits of the grid are not in the cache (first: {missing[0]})")
    h = int(grid_cfg.get("horizon", prereg["targets"]["horizons"]["primary"]))
    lags = list(lags or prereg["stability"]["lags_months"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wtasks, ptasks = [], []
    for (data, symbol, target), g in groups.items():
        taus = list(g)
        for tau in taus:
            wtasks.append(delayed(window_task)(data, symbol, target, tau, g[tau], h, prereg))
        for lag in lags:
            for i in range(len(taus) - lag):
                a, b = taus[i], taus[i + lag]
                ptasks.append(delayed(pair_task)(data, symbol, target, a, b, lag, g[a], g[b], h, prereg))
    t0 = time.time()
    par = Parallel(n_jobs=n_workers, backend="loky", batch_size=4, verbose=0)
    wres = par(wtasks)
    t1 = time.time()
    pres = par(ptasks)
    t2 = time.time()
    pd.DataFrame([r["window"] for r in wres]).to_parquet(out / "windows.parquet")
    pd.DataFrame([x for r in wres for x in r["features"]]).to_parquet(out / "window_features.parquet")
    pd.DataFrame([x for r in wres for x in r["shapes"]]).to_parquet(out / "shapes.parquet")
    pd.DataFrame([x for r in wres for x in r["fits"]]).to_parquet(out / "fits.parquet")
    pd.DataFrame([r["pair"] for r in pres]).to_parquet(out / "pairs.parquet")
    pd.DataFrame([x for r in pres for x in r["features"]]).to_parquet(out / "pair_features.parquet")
    info = {"grid": str(grid_path), "n_windows": len(wtasks), "n_pairs": len(ptasks),
            "seconds_windows": t1 - t0, "seconds_pairs": t2 - t1, "lags": lags}
    (out / "compare_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if verbose:
        print(json.dumps(info))
    return info
