"""D-57 check (c): power of the surrogate-referenced floor on a synthetic planted break (lead-owned).

The stationary synthetic process of ``scripts/verify_floor_synthetic.py`` (same feature
construction, seed 7), lengthened to [2019-01-01, 2026-01-01), vol target only, with a break at
2022-07-01: the ``rv_level_pct`` effect flips sign (+0.3x -> -0.3x) and the ``rv_ratio_4h`` effect
halves (0.4 tanh -> 0.2 tanh). 9 surrogates of the synthetic panel come from
:func:`stability.surrogate.make_surrogate` (mean block 252 h). Per data set: 5 seed-only fits per
window (prereg settings), 72 windows (tau 2020-01 .. 2025-12). Pair distances at lags 12 and 24 use
the same X_ref, cells and shapes as the slice.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config
from .splits import replicate_plan, retrain_schedule, stable_seed, time_ordered_bags, train_mask

H = 8
START, END, BREAK = "2019-01-01", "2026-01-01", "2022-07-01"
FIRST_TAU, LAST_TAU = "2020-01-01", "2025-12-01"
N_SUR = 9
LAGS = (12, 24)
STRONG, MODERATE = "rv_level_pct", "rv_ratio_4h"
OUT = ROOT / "results" / "surfloor"


def code_tag() -> str:
    h = hashlib.sha256()
    for rel in ("src/ebmregime/surfloor_power.py", "src/ebmregime/models/ebm.py", "src/ebmregime/splits/bags.py",
                "src/ebmregime/stability/surrogate.py"):
        h.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def make_break_process(seed: int = 7) -> pd.DataFrame:
    ts = pd.date_range(START, END, freq="h", tz="UTC", inclusive="left")
    n = len(ts)
    rng = np.random.default_rng(seed)
    names = P.feature_names()
    phis = [0.0, 0.5, 0.9, 0.97, 0.99, 0.0, 0.8, 0.95, 0.0, 0.6, 0.9, 0.98, None, None, 0.7, 0.3]
    X = {}
    for name, phi in zip(names, phis):
        if name == "hour":
            X[name] = ts.hour.to_numpy().astype(np.int64)
        elif name == "dow":
            X[name] = ts.day_of_week.to_numpy().astype(np.int64)
        else:
            e = rng.standard_normal(n)
            x = np.empty(n)
            x[0] = e[0] / np.sqrt(1 - phi**2)
            for i in range(1, n):
                x[i] = phi * x[i - 1] + e[i]
            X[name] = x * np.sqrt(1 - phi**2)
    df = pd.DataFrame(X, index=pd.DatetimeIndex(ts, name="ts"))
    u = rng.standard_normal(n)
    noise = np.empty(n)
    noise[0] = u[0]
    phi_n = 1 - 1 / H
    for i in range(1, n):
        noise[i] = phi_n * noise[i - 1] + np.sqrt(1 - phi_n**2) * u[i]
    post = np.asarray(ts >= pd.Timestamp(BREAK, tz="UTC"))
    a_mod = np.where(post, 0.2, 0.4)
    a_str = np.where(post, -0.3, 0.3)
    signal = (a_mod * np.tanh(df[MODERATE]) + a_str * df[STRONG] - 0.2 * np.abs(df["ret_1h"])
              + 0.3 * np.cos(2 * np.pi * df["hour"] / 24) + 0.2 * (df["dow"] >= 5))
    df["y_vol"] = signal + 0.8 * noise
    return df


GEN = "replace"          # "replace" (frozen generator, D-57) or "perm" (D-60); set by cmd_power


def data_sets() -> list[str]:
    tag = "sur" if GEN == "replace" else "perm"
    return ["synth"] + [f"{tag}:{i}" for i in range(N_SUR)]


@lru_cache(maxsize=2)
def panel_of(data: str) -> pd.DataFrame:
    base = make_break_process()
    if data == "synth":
        return base
    pr = load_config("prereg")
    kind, i = data.split(":")
    mb = float(pr["surrogates"]["mean_block_hours"])
    if kind == "perm":
        from .surfloor_perm import make_perm_surrogate

        seed = stable_seed(int(pr["surrogates"]["seed"]), "SYNTH_BREAK", "perm_surrogate", int(i))
        return make_perm_surrogate(base, START, END, mb, seed, purge={"y_vol": H})
    from .stability.surrogate import make_surrogate

    seed = stable_seed(int(pr["surrogates"]["seed"]), "SYNTH_BREAK", "surrogate", int(i))
    return make_surrogate(base, START, END, mb, seed)


def taus():
    return retrain_schedule(FIRST_TAU, LAST_TAU)


def fit_path(data, tau, r):
    return ROOT / "data" / "cache" / "surfloor_power" / code_tag() / data.replace(":", "-") / f"{tau.date()}_{r}.npz"


def fit_task(data, tau, rep, prereg):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .models.ebm import export_fit, fit_ebm
    from .runner.cache import write_atomic

    path = fit_path(data, tau, rep["r"])
    if path.exists():
        return None
    v = prereg["validation"]
    df = panel_of(data)
    tm = train_mask(df.index, tau, 12, H, df["y_vol"].notna().to_numpy())
    ts_tr = df.index[tm]
    X = df.loc[tm, P.feature_names()]
    y = df.loc[tm, "y_vol"].to_numpy()
    bags = time_ordered_bags(ts_tr, H, v["outer_bags"], v["tail_frac"], v["block_hours"], v["bag_train_frac"],
                             seed=rep["seed"])
    model, info = fit_ebm(X, y, bags, "regressor", rep["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, dict(export_fit(model)), {"data": data, "tau": str(tau.date()), "r": rep["r"],
                                                 "fit_seconds": info["fit_seconds"]})
    return info["fit_seconds"]


def _fit_rows(df, tau, tail_frac):
    tm = train_mask(df.index, tau, 12, H, df["y_vol"].notna().to_numpy())
    pos = np.flatnonzero(tm)
    keep = ~P.purged_before_tail(df.index[pos], H, tail_frac)
    out = np.zeros(len(df), bool)
    out[pos[keep]] = True
    return out


def pair_task(data, a, b, lag, prereg):
    from .explain import cell_shapes_many, make_cells, make_xref
    from .models.ebm import eval_terms_from_export
    from .runner.cache import read_path
    from .stability import floor as F

    df = panel_of(data)
    tail = float(prereg["validation"]["tail_frac"])
    xr, c = prereg["explain"]["x_ref"], prereg["explain"]["cells"]
    pos = make_xref([_fit_rows(df, a, tail), _fit_rows(df, b, tail)], cap=int(xr["cap_rows"]),
                    seed=stable_seed(int(xr["seed"]), "SYNTH_BREAK", data, str(a.date()), str(b.date()), "pair"))
    X = df.iloc[pos][P.feature_names()]
    grid = make_cells(X, n_cells=int(c["n"]), min_rows=int(c["min_rows"]), nominal=P.nominal_features())
    sh = {}
    for w in (a, b):
        C = np.stack([eval_terms_from_export(read_path(fit_path(data, w, r))[0], X) for r in range(5)])
        sh[w] = cell_shapes_many(C, grid)
    return [{"data": data, "tau_a": str(a.date()), "tau_b": str(b.date()), "lag": lag, "feature": nm,
             "g": F.cross_mean_d2(sh[a][j], sh[b][j], grid.weights[j]),
             "s2_seed_a": 0.5 * F.pair_mean_d2(sh[a][j], grid.weights[j]),
             "s2_seed_b": 0.5 * F.pair_mean_d2(sh[b][j], grid.weights[j])}
            for j, nm in enumerate(P.feature_names())]


def cmd_power(workers: int, gen: str = "replace"):
    global GEN
    GEN = gen
    from joblib import Parallel, delayed

    prereg = load_config("prereg")
    T = taus()
    reps = lambda tau: [r for r in replicate_plan("SYNTH", tau, "vol", prereg["replicates"]) if r["protocol"] == "seed"]  # noqa: E731
    todo = [(d, tau, rep) for d in data_sets() for tau in T for rep in reps(tau) if not fit_path(d, tau, rep["r"]).exists()]
    print(f"power: {len(todo)} fits to run, tag {code_tag()}", flush=True)
    t0 = time.time()
    Parallel(n_jobs=workers, backend="loky", batch_size=1, verbose=5)(delayed(fit_task)(*t, prereg) for t in todo)
    print(f"fits done in {(time.time() - t0) / 60:.1f} min", flush=True)
    ptasks = [delayed(pair_task)(d, T[i], T[i + lag], lag, prereg) for d in data_sets() for lag in LAGS
              for i in range(len(T) - lag)]
    res = Parallel(n_jobs=workers, backend="loky", batch_size=4, verbose=2)(ptasks)
    pf = pd.DataFrame([r for rr in res for r in rr])
    OUT.mkdir(parents=True, exist_ok=True)
    pf.to_parquet(OUT / f"{_pfx()}power_pairs.parquet")

    return summarise(pf)


def _pfx() -> str:
    return "" if GEN == "replace" else "perm_"


def summarise(pf: pd.DataFrame) -> dict:
    """D-58 (c): medians of DR_sur over pair sets, against the leave-one-out null of the 9 surrogates."""
    from .stats import median_ci

    brk = pd.Timestamp(BREAK)
    pf = pf.copy()
    ta, tb = pd.to_datetime(pf.tau_a), pd.to_datetime(pf.tau_b)
    pf["side"] = np.where((ta <= brk) & (tb - pd.DateOffset(months=12) >= brk), "opposite",
                          np.where((tb <= brk) | (ta - pd.DateOffset(months=12) >= brk), "same", "mixed"))
    sur = [d for d in data_sets() if d != "synth"]
    tot = pf.groupby(["lag", "tau_a", "side", "data"]).g.sum().unstack("data")
    feat = pf.pivot_table(index=["lag", "tau_a", "side", "feature"], columns="data", values="g")

    def ratio(w, d):
        refs = [r for r in sur if r != d]
        return w[d] / w[refs].mean(axis=1)

    rows = []
    for side, lags in (("opposite", (12, 24)), ("same", (12,))):
        for nm in ["TOTAL"] + P.feature_names():
            w = tot if nm == "TOTAL" else feat.xs(nm, level="feature")
            w = w[w.index.get_level_values("side").isin([side]) & w.index.get_level_values("lag").isin(lags)]
            w = w.sort_index(level="tau_a")
            x = ratio(w, "synth").values
            null = np.array([np.median(ratio(w, d).values) for d in sur])
            ci = median_ci(x, delta=12)
            rows.append({"side": side, "feature": nm, "n": len(x), "m": float(np.median(x)),
                         "null_max": float(null.max()), "null_median": float(np.median(null)),
                         "null_min": float(null.min()), "boot_lo": ci["lo"], "boot_hi": ci["hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{_pfx()}check_c_power.csv", index=False)
    opp = out[out.side == "opposite"].set_index("feature")
    st, to = opp.loc[STRONG], opp.loc["TOTAL"]
    gate = {"strong_m": float(st.m), "strong_null_max": float(st.null_max), "total_m": float(to.m),
            "total_null_max": float(to.null_max), "moderate_m": float(opp.loc[MODERATE, "m"]),
            "moderate_null_max": float(opp.loc[MODERATE, "null_max"])}
    gate["pass"] = bool(st.m > max(1.25, st.null_max) and to.m > max(1.0, to.null_max))
    print(out.round(3).to_string())
    print("gate (c):", json.dumps(gate), "PASS" if gate["pass"] else "FAIL")
    (OUT / f"{_pfx()}check_c.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    return gate
