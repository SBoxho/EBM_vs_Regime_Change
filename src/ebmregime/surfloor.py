"""Surrogate-referenced noise floor (D-57; lead-owned, outside the runner's fit-code hash).

* :func:`seed_jobs`: the seed-only fits (protocol i) of the real data and of surrogates 0..N-1,
  run through the normal runner cache (``data/cache/fits``).
* :func:`pair_seed_task`: per pair of windows, the per-feature cross-window distance
  ``g_j = d2_cross_j`` and the seed floors, computed with the slice comparison stage's own helpers
  (pair X_ref, cells, re-centred shapes of the seed-only fits), so the numbers are those of
  :func:`runner.compare.pair_task`.
* :func:`dr_sur`: DR_sur of one data set against a set of reference surrogates (D-57).
* :func:`check_calibration`, :func:`check_negative_control`: checks (a) and (b) of D-57.

Real ``dir`` and ``vol`` pairs are only computed with ``allow_real_signal=True``, which the CLI
gates on checks (a), (b) and (c) having passed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config

SYMBOL, H = "BTCUSDT", 8
N_SUR = 20
TARGETS = ("dir", "vol", "shuf")
OUT = ROOT / "results" / "surfloor"


def datasets(n_sur: int = N_SUR) -> list[str]:
    return [f"surrogate:{i}" for i in range(n_sur)]


def seed_jobs(data_list, targets=TARGETS, prereg=None):
    """Seed-only jobs (prereg replicate seeds) for each data set and target, all 83 windows."""
    from .runner.jobs import expand_grid

    prereg = prereg or load_config("prereg")
    cfg = {"model": "ebm", "arm": "primary", "symbols": [SYMBOL], "horizon": H, "taus": "all",
           "data": [{"source": d, "targets": list(targets)} for d in data_list]}
    return [j for j in expand_grid(cfg, prereg) if j.protocol == "seed"]


def cmd_fit(workers: int, n_sur: int = N_SUR):
    from .runner.run import run_jobs

    jobs = seed_jobs(datasets(n_sur))
    print(f"{len(jobs)} seed-only surrogate jobs", flush=True)
    s = run_jobs(jobs, n_workers=workers)
    print(json.dumps(s, default=str), flush=True)
    return s


# --------------------------------------------------------------------------- pair distances

def pair_seed_task(data, target, tau_a, tau_b, lag, keys_a, keys_b, prereg) -> list[dict]:
    from .runner import compare as C
    from .stability import floor as F

    pnl = C._panel(SYMBOL, data)
    ma = C._fit_mask(pnl, tau_a, target, H, prereg)
    mb = C._fit_mask(pnl, tau_b, target, H, prereg)
    pos = C._xref([ma, mb], prereg, SYMBOL, data, target, tau_a, tau_b, "pair")
    X = pnl.iloc[pos][P.feature_names()]
    grid = C._cells(X, prereg)
    Sa, _ = C._shapes_for(list(keys_a), X, grid)
    Sb, _ = C._shapes_for(list(keys_b), X, grid)
    rows = []
    for j, name in enumerate(P.feature_names()):
        w = grid.weights[j]
        rows.append({"data": data, "target": target, "tau_a": tau_a, "tau_b": tau_b, "lag": lag,
                     "feature": name, "g": F.cross_mean_d2(Sa[j], Sb[j], w),
                     "s2_seed_a": 0.5 * F.pair_mean_d2(Sa[j], w), "s2_seed_b": 0.5 * F.pair_mean_d2(Sb[j], w)})
    return rows


def control_arms(n_sur: int = N_SUR) -> list[tuple[str, str]]:
    """Every surrogate x (dir, vol, shuf), plus real shuf. Never real dir or vol."""
    return [(d, t) for d in datasets(n_sur) for t in TARGETS] + [("real", "shuf")]


def cmd_pairs(arms, workers: int, out_name: str, lags=None, allow_real_signal=False):
    from joblib import Parallel, delayed

    from .runner import cache
    from .runner.compare import group_jobs

    if not allow_real_signal and any(d == "real" and t in ("dir", "vol") for d, t in arms):
        raise RuntimeError("real dir/vol drift is withheld until D-57 checks (a)-(c) pass")
    prereg = load_config("prereg")
    lags = list(lags or prereg["stability"]["lags_months"])
    jobs = [j for d, t in arms for j in seed_jobs([d], [t], prereg)]
    groups = group_jobs(jobs)
    done = cache.finished_keys()
    missing = [k for g in groups.values() for d in g.values() for k in d["seed"] if k not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} seed fits missing from the cache")
    tasks = []
    for (data, _, target), g in groups.items():
        taus = list(g)
        for lag in lags:
            for i in range(len(taus) - lag):
                a, b = taus[i], taus[i + lag]
                tasks.append(delayed(pair_seed_task)(data, target, a, b, lag, g[a]["seed"], g[b]["seed"], prereg))
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", batch_size=8, verbose=2)(tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{out_name}.parquet")
    print(f"{len(tasks)} pairs in {time.time() - t0:.0f} s -> {OUT / out_name}.parquet", flush=True)
    return df


# --------------------------------------------------------------------------- DR_sur

def totals(pf: pd.DataFrame) -> pd.DataFrame:
    """Per (data, target, lag, tau_a, tau_b): G = sum_j g_j and S = sum_j seed floors."""
    t = pf.assign(S=pf.s2_seed_a + pf.s2_seed_b)
    return t.groupby(["data", "target", "lag", "tau_a", "tau_b"], as_index=False)[["g", "S"]].sum().rename(
        columns={"g": "G"})


def dr_sur(tot: pd.DataFrame, data: str, target: str, refs, lag: int, adj: bool = False) -> pd.Series:
    """Per pair (index tau_a): DR_sur of ``data`` against the mean over ``refs`` (D-57)."""
    sub = tot[(tot.target == target) & (tot.lag == lag)]
    val = (sub.G - sub.S) if adj else sub.G
    wide = sub.assign(v=val).pivot(index="tau_a", columns="data", values="v").sort_index()
    ref = wide[list(refs)].mean(axis=1)
    return (wide[data] / ref).rename("DR_sur")


def dr_sur_feature(pf: pd.DataFrame, data: str, target: str, refs, lag: int) -> pd.DataFrame:
    """Per pair x feature DR_sur,j (columns = features, index = tau_a)."""
    sub = pf[(pf.target == target) & (pf.lag == lag)]
    wide = sub.pivot_table(index=["tau_a", "feature"], columns="data", values="g")
    r = wide[data] / wide[list(refs)].mean(axis=1)
    return r.unstack("feature").sort_index()


def _ci(x, lag, prereg):
    from .stats import median_ci

    return median_ci(np.asarray(x, float), delta=lag, level=float(prereg["statistics"]["bootstrap"]["ci_level"]))


def loo_medians(tot: pd.DataFrame, target: str, lag: int, n_sur: int = N_SUR, adj: bool = False) -> pd.Series:
    """Null distribution of m (D-58): per surrogate s, the median over pairs of DR_sur of s
    against the other surrogates."""
    sur = datasets(n_sur)
    return pd.Series({s: float(np.median(dr_sur(tot, s, target, [r for r in sur if r != s], lag, adj=adj).values))
                      for s in sur}, name="m")


def check_calibration(tot: pd.DataFrame, prereg=None, n_sur: int = N_SUR, with_ci: bool = True) -> dict:
    """D-58 (a): for dir and vol at the headline lag, the median of the leave-one-out m_s lies in
    [0.9, 1.11]. Every lag is tabulated, with the (reported only) prereg bootstrap CI per m_s."""
    prereg = prereg or load_config("prereg")
    sur = datasets(n_sur)
    lag12 = int(prereg["stability"]["headline_lag_months"])
    rows = []
    for target in ("dir", "vol"):
        for lag in sorted(tot.lag.unique()):
            for s in sur:
                x = dr_sur(tot, s, target, [r for r in sur if r != s], int(lag))
                ci = _ci(x.values, int(lag), prereg) if with_ci else {"lo": np.nan, "hi": np.nan}
                rows.append({"target": target, "lag": int(lag), "held_out": s, "m": float(np.median(x.values)),
                             "boot_lo": ci["lo"], "boot_hi": ci["hi"],
                             "boot_covers_1": bool(ci["lo"] <= 1 <= ci["hi"])})
    df = pd.DataFrame(rows)
    gate = {}
    for target in ("dir", "vol"):
        d = df[(df.target == target) & (df.lag == lag12)]
        med = float(d["m"].median())
        gate[target] = {"median_m": med, "q025": float(d.m.quantile(0.025)), "q975": float(d.m.quantile(0.975)),
                        "boot_n_cover": int(d.boot_covers_1.sum()), "pass": bool(0.9 <= med <= 1.11)}
    return {"table": df, "gate": gate, "pass": all(g["pass"] for g in gate.values())}


def check_negative_control(tot: pd.DataFrame, prereg=None, n_sur: int = N_SUR, with_ci: bool = True) -> dict:
    """D-58 (b): real shuf against shuf on all surrogates. At the headline lag: m_real within
    [min, max] of the leave-one-out null m_s; m_real in the G0 band; the null's 2.5-97.5% quantiles
    inside the G0 band. Every lag is tabulated (plain and seed-adjusted), with the prereg bootstrap CI."""
    prereg = prereg or load_config("prereg")
    band = prereg["hypotheses"]["g0"]["pass_ci_within"]
    lag12 = int(prereg["stability"]["headline_lag_months"])
    rows = []
    for lag in sorted(tot.lag.unique()):
        for adj in (False, True):
            x = dr_sur(tot, "real", "shuf", datasets(n_sur), int(lag), adj=adj)
            null = loo_medians(tot, "shuf", int(lag), n_sur, adj=adj)
            ci = _ci(x.values, int(lag), prereg) if with_ci else {"n": len(x), "lo": np.nan, "hi": np.nan}
            m = float(np.median(x.values))
            rows.append({"lag": int(lag), "adj": adj, "n": len(x), "m_real": m, "null_min": float(null.min()),
                         "null_q025": float(null.quantile(0.025)), "null_median": float(null.median()),
                         "null_q975": float(null.quantile(0.975)), "null_max": float(null.max()),
                         "rank_from_top": int(1 + (null >= m).sum()), "boot_lo": ci["lo"], "boot_hi": ci["hi"]})
    df = pd.DataFrame(rows)
    g = df[(df.lag == lag12) & (~df.adj)].iloc[0]
    c1 = bool(g.null_min <= g.m_real <= g.null_max)
    c2 = bool(band[0] <= g.m_real <= band[1])
    c3 = bool(band[0] <= g.null_q025 and g.null_q975 <= band[1])
    gate = {"m_real": float(g.m_real), "null_min": float(g.null_min), "null_max": float(g.null_max),
            "null_q025": float(g.null_q025), "null_q975": float(g.null_q975), "within_null": c1, "in_band": c2,
            "null_precise": c3, "boot_lo": float(g.boot_lo), "boot_hi": float(g.boot_hi), "pass": c1 and c2 and c3}
    return {"table": df, "gate": gate, "pass": gate["pass"]}


def rank_p(tot: pd.DataFrame, data: str, target: str, lag: int, n_sur: int = N_SUR) -> float:
    """One-sided rank p of T_data = median over pairs of G among the surrogates' T (D-57)."""
    sub = tot[(tot.target == target) & (tot.lag == lag)]
    T = sub.groupby("data").G.median()
    t_obs = T[data]
    ts = T[[s for s in datasets(n_sur) if s != data]]
    return float((1 + (ts >= t_obs).sum()) / (1 + len(ts)))


def cmd_check(name: str = "control_pairs"):
    prereg = load_config("prereg")
    pf = pd.read_parquet(OUT / f"{name}.parquet")
    tot = totals(pf)
    cal = check_calibration(tot, prereg)
    neg = check_negative_control(tot, prereg)
    cal["table"].to_csv(OUT / "check_a_calibration.csv", index=False)
    neg["table"].to_csv(OUT / "check_b_negative_control.csv", index=False)
    summ = cal["table"].groupby(["target", "lag"]).agg(median_m=("m", "median"), q025=("m", lambda v: v.quantile(.025)),
                                                        q975=("m", lambda v: v.quantile(.975)),
                                                        boot_cover=("boot_covers_1", "sum"))
    print("(a) calibration, leave one surrogate out (m = median DR_sur over pairs)\n", summ.round(3).to_string())
    print("gate (a):", json.dumps(cal["gate"]), "PASS" if cal["pass"] else "FAIL")
    print("(b) real shuf vs 20 surrogates\n", neg["table"].round(3).to_string())
    print("gate (b):", json.dumps(neg["gate"]), "PASS" if neg["pass"] else "FAIL")
    out = {"a": cal["gate"], "a_pass": cal["pass"], "b": neg["gate"], "b_pass": neg["pass"],
           "rank_p_shuf_12": rank_p(tot, "real", "shuf", 12)}
    (OUT / "checks_ab.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- real targets (gated)

def gates_passed() -> dict:
    a = json.loads((OUT / "checks_ab.json").read_text()) if (OUT / "checks_ab.json").exists() else {}
    c = json.loads((OUT / "check_c.json").read_text()) if (OUT / "check_c.json").exists() else {}
    return {"a": bool(a.get("a_pass")), "b": bool(a.get("b_pass")), "c": bool(c.get("pass"))}


def real_summary(tot: pd.DataFrame, pf: pd.DataFrame, prereg=None, n_sur: int = N_SUR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per target and lag: m_real, the leave-one-out null, one-sided rank p and the prereg bootstrap CI;
    per feature at the headline lag: median DR_sur,j with its own leave-one-out null."""
    prereg = prereg or load_config("prereg")
    sur = datasets(n_sur)
    lag12 = int(prereg["stability"]["headline_lag_months"])
    rows, frows = [], []
    for target in ("dir", "vol"):
        for lag in sorted(tot.lag.unique()):
            for adj in (False, True):
                x = dr_sur(tot, "real", target, sur, int(lag), adj=adj)
                null = loo_medians(tot, target, int(lag), n_sur, adj=adj)
                m = float(np.median(x.values))
                ci = _ci(x.values, int(lag), prereg)
                rows.append({"target": target, "lag": int(lag), "adj": adj, "n": len(x), "m_real": m,
                             "null_median": float(null.median()), "null_q975": float(null.quantile(0.975)),
                             "null_max": float(null.max()), "p_rank": float((1 + (null >= m).sum()) / (1 + len(null))),
                             "boot_lo": ci["lo"], "boot_hi": ci["hi"]})
        R = dr_sur_feature(pf, "real", target, sur, lag12)
        nulls = {s: dr_sur_feature(pf, s, target, [r for r in sur if r != s], lag12).median() for s in sur}
        N = pd.DataFrame(nulls).T                                  # surrogate x feature
        for f in R.columns:
            m = float(R[f].median())
            frows.append({"target": target, "lag": lag12, "feature": f, "m_real": m,
                          "null_median": float(N[f].median()), "null_max": float(N[f].max()),
                          "p_rank": float((1 + (N[f] >= m).sum()) / (1 + len(N)))})
    return pd.DataFrame(rows), pd.DataFrame(frows)


def cmd_real(workers: int):
    g = gates_passed()
    if not all(g.values()):
        raise SystemExit(f"D-57/D-58 checks not all passed {g}; real dir/vol drift stays withheld")
    pr = cmd_pairs([("real", "dir"), ("real", "vol")], workers, "real_pairs", allow_real_signal=True)
    ctrl = pd.read_parquet(OUT / "control_pairs.parquet")
    pf = pd.concat([ctrl[ctrl.target.isin(["dir", "vol"])], pr], ignore_index=True)
    tot = totals(pf)
    summ, feat = real_summary(tot, pf)
    summ.to_csv(OUT / "real_dr_sur.csv", index=False)
    feat.to_csv(OUT / "real_dr_sur_features_lag12.csv", index=False)
    print(summ.round(3).to_string())
    print(feat.sort_values(["target", "m_real"], ascending=[True, False]).round(3).to_string())
