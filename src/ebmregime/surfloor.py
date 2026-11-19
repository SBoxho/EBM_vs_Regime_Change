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


# --------------------------------------------------------------------------- D-62: pooled rank / median ratio

def pooled_stats(wide: pd.DataFrame) -> pd.DataFrame:
    """D-62: ``wide`` is pairs x data sets (one pool). Each member D is referenced to the other k-1:
    u_D(p) = mid-rank fraction of G_D(p) among the others, r_D(p) = G_D(p) / median of the others.
    Returns per member U = mean_p u_D(p) and M = median_p r_D(p)."""
    wide = wide.dropna()
    V = wide.to_numpy(float)
    k = V.shape[1]
    if k < 2 or len(V) == 0:
        raise ValueError("pool needs >= 2 members and >= 1 complete pair")
    out = {}
    for i, d in enumerate(wide.columns):
        others = np.delete(V, i, axis=1)
        x = V[:, [i]]
        u = ((others < x).sum(1) + 0.5 * (others == x).sum(1)) / (k - 1)
        r = V[:, i] / np.median(others, axis=1)
        out[d] = {"U": float(u.mean()), "M": float(np.median(r)), "n": len(V)}
    return pd.DataFrame(out).T.astype({"U": float, "M": float, "n": int})


def _wide(tot: pd.DataFrame, target: str, lag: int, datas, adj: bool = False) -> pd.DataFrame:
    sub = tot[(tot.target == target) & (tot.lag == lag) & tot.data.isin(list(datas))]
    val = (sub.G - sub.S) if adj else sub.G
    return sub.assign(v=val).pivot(index="tau_a", columns="data", values="v")[list(datas)].sort_index()


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q))


def check_calibration_rank(tot: pd.DataFrame, prereg=None, n_sur: int = N_SUR) -> dict:
    """D-62 (a) and (d): pool = the surrogates. (a) median of M_s at the headline lag in [0.9, 1.11];
    (d) precision: 2.5-97.5% quantiles of M_s inside the G0 band (per target, not a stop rule)."""
    prereg = prereg or load_config("prereg")
    band = prereg["hypotheses"]["g0"]["pass_ci_within"]
    lag12 = int(prereg["stability"]["headline_lag_months"])
    sur = datasets(n_sur)
    rows = []
    for target in ("dir", "vol", "shuf"):
        for lag in sorted(tot.lag.unique()):
            for adj in (False, True):
                st = pooled_stats(_wide(tot, target, int(lag), sur, adj))
                for s, r in st.iterrows():
                    rows.append({"target": target, "lag": int(lag), "adj": adj, "data": s, **r.to_dict()})
    df = pd.DataFrame(rows)
    gate, prec = {}, {}
    for target in ("dir", "vol", "shuf"):
        d = df[(df.target == target) & (df.lag == lag12) & (~df.adj)]
        med, lo, hi = float(d.M.median()), _q(d.M, .025), _q(d.M, .975)
        prec[target] = {"q025": lo, "q975": hi, "precise": bool(band[0] <= lo and hi <= band[1])}
        if target != "shuf":
            gate[target] = {"median_M": med, "q025": lo, "q975": hi, "pass": bool(0.9 <= med <= 1.11)}
    return {"table": df, "gate": gate, "precision": prec, "pass": all(g["pass"] for g in gate.values())}


def check_negative_control_rank(tot: pd.DataFrame, prereg=None, n_sur: int = N_SUR) -> dict:
    """D-62 (b): real shuf in the pool of real + surrogates. At the headline lag: U_real within
    [min, max] of the surrogates' U, and M_real in the G0 band. Every lag tabulated, plain and adjusted."""
    prereg = prereg or load_config("prereg")
    band = prereg["hypotheses"]["g0"]["pass_ci_within"]
    lag12 = int(prereg["stability"]["headline_lag_months"])
    pool = ["real"] + datasets(n_sur)
    rows = []
    for lag in sorted(tot.lag.unique()):
        for adj in (False, True):
            rows.append(_real_row(pooled_stats(_wide(tot, "shuf", int(lag), pool, adj)), "shuf", int(lag), adj))
    df = pd.DataFrame(rows)
    g = df[(df.lag == lag12) & (~df.adj)].iloc[0]
    c1 = bool(g.U_null_min <= g.U_real <= g.U_null_max)
    c2 = bool(band[0] <= g.M_real <= band[1])
    gate = {k: (v.item() if hasattr(v, "item") else v) for k, v in g.items() if k not in ("target", "adj")}
    gate.update({"U_within_null": c1, "M_in_band": c2, "pass": c1 and c2})
    return {"table": df, "gate": gate, "pass": gate["pass"]}


def _real_row(st: pd.DataFrame, target: str, lag: int, adj: bool, data: str = "real") -> dict:
    nul = st.drop(index=data)
    u, m = float(st.loc[data, "U"]), float(st.loc[data, "M"])
    return {"target": target, "lag": lag, "adj": adj, "n": int(st.loc[data, "n"]), "U_real": u, "M_real": m,
            "U_null_min": float(nul.U.min()), "U_null_max": float(nul.U.max()),
            "M_null_min": float(nul.M.min()), "M_null_q025": _q(nul.M, .025), "M_null_median": float(nul.M.median()),
            "M_null_q975": _q(nul.M, .975), "M_null_max": float(nul.M.max()),
            "rank_from_top": int((st.U >= u).sum()), "p_upper": float((st.U >= u).sum() / len(st))}


def cmd_check_rank(name: str = "perm_control_pairs", prefix: str = "perm_rank_"):
    prereg = load_config("prereg")
    tot = totals(pd.read_parquet(OUT / f"{name}.parquet"))
    cal = check_calibration_rank(tot, prereg)
    neg = check_negative_control_rank(tot, prereg)
    cal["table"].to_csv(OUT / f"{prefix}check_a_calibration.csv", index=False)
    neg["table"].to_csv(OUT / f"{prefix}check_b_negative_control.csv", index=False)
    t = cal["table"][~cal["table"].adj]
    summ = t.groupby(["target", "lag"]).agg(median_M=("M", "median"), q025=("M", lambda v: v.quantile(.025)),
                                            q975=("M", lambda v: v.quantile(.975)), U_min=("U", "min"),
                                            U_max=("U", "max"))
    print("(a)/(d) surrogate pool: M = median over pairs of G / median(other surrogates)\n", summ.round(3).to_string())
    print("gate (a):", json.dumps(cal["gate"]), "PASS" if cal["pass"] else "FAIL")
    print("precision (d):", json.dumps(cal["precision"]))
    print("(b) real shuf in the 21-member pool\n", neg["table"].round(3).to_string())
    print("gate (b):", json.dumps(neg["gate"]), "PASS" if neg["pass"] else "FAIL")
    out = {"a": cal["gate"], "a_pass": cal["pass"], "b": neg["gate"], "b_pass": neg["pass"],
           "precision": cal["precision"]}
    (OUT / f"{prefix}checks_ab.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def cmd_check(name: str = "control_pairs", prefix: str = ""):
    prereg = load_config("prereg")
    pf = pd.read_parquet(OUT / f"{name}.parquet")
    tot = totals(pf)
    cal = check_calibration(tot, prereg)
    neg = check_negative_control(tot, prereg)
    cal["table"].to_csv(OUT / f"{prefix}check_a_calibration.csv", index=False)
    neg["table"].to_csv(OUT / f"{prefix}check_b_negative_control.csv", index=False)
    summ = cal["table"].groupby(["target", "lag"]).agg(median_m=("m", "median"), q025=("m", lambda v: v.quantile(.025)),
                                                        q975=("m", lambda v: v.quantile(.975)),
                                                        boot_cover=("boot_covers_1", "sum"))
    print("(a) calibration, leave one surrogate out (m = median DR_sur over pairs)\n", summ.round(3).to_string())
    print("gate (a):", json.dumps(cal["gate"]), "PASS" if cal["pass"] else "FAIL")
    print("(b) real shuf vs 20 surrogates\n", neg["table"].round(3).to_string())
    print("gate (b):", json.dumps(neg["gate"]), "PASS" if neg["pass"] else "FAIL")
    out = {"a": cal["gate"], "a_pass": cal["pass"], "b": neg["gate"], "b_pass": neg["pass"],
           "rank_p_shuf_12": rank_p(tot, "real", "shuf", 12)}
    (OUT / f"{prefix}checks_ab.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- real targets (gated)

def gates_passed(prefix: str = "") -> dict:
    a = json.loads((OUT / f"{prefix}checks_ab.json").read_text()) if (OUT / f"{prefix}checks_ab.json").exists() else {}
    c = json.loads((OUT / f"{prefix}check_c.json").read_text()) if (OUT / f"{prefix}check_c.json").exists() else {}
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


# --------------------------------------------------------------------------- D-62 (c), from the stored power pairs

def summarise_power_rank(pf: pd.DataFrame | None = None) -> dict:
    """D-62 (c): pool = synthetic panel + its 9 perm surrogates; U and M of each member against the
    other 9, on the opposite-side (lags 12, 24) and same-side (lag 12) pairs. Reads
    ``perm_power_pairs.parquet``; no fits (kept outside ``surfloor_power`` so its fit-cache tag is unchanged)."""
    from . import surfloor_power as Q

    pf = pd.read_parquet(OUT / "perm_power_pairs.parquet") if pf is None else pf
    brk = pd.Timestamp(Q.BREAK)
    ta, tb = pd.to_datetime(pf.tau_a), pd.to_datetime(pf.tau_b)
    pf = pf.assign(side=np.where((ta <= brk) & (tb - pd.DateOffset(months=12) >= brk), "opposite",
                                 np.where((tb <= brk) | (ta - pd.DateOffset(months=12) >= brk), "same", "mixed")))
    pool = ["synth"] + [f"perm:{i}" for i in range(Q.N_SUR)]
    tot = pf.groupby(["lag", "tau_a", "side", "data"]).g.sum().unstack("data")
    feat = pf.pivot_table(index=["lag", "tau_a", "side", "feature"], columns="data", values="g")
    rows = []
    for side, lags in (("opposite", (12, 24)), ("same", (12,))):
        for nm in ["TOTAL"] + P.feature_names():
            w = tot if nm == "TOTAL" else feat.xs(nm, level="feature")
            w = w[w.index.get_level_values("side").isin([side]) & w.index.get_level_values("lag").isin(lags)]
            st = pooled_stats(w[pool])
            nul = st.drop(index="synth")
            u = float(st.loc["synth", "U"])
            rows.append({"side": side, "feature": nm, "n": int(st.loc["synth", "n"]), "U": u,
                         "U_rank": int((st.U >= u).sum()), "U_null_max": float(nul.U.max()),
                         "M": float(st.loc["synth", "M"]), "M_null_min": float(nul.M.min()),
                         "M_null_median": float(nul.M.median()), "M_null_max": float(nul.M.max())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "perm_rank_check_c_power.csv", index=False)
    opp = out[out.side == "opposite"].set_index("feature")
    st, to, mo = opp.loc[Q.STRONG], opp.loc["TOTAL"], opp.loc[Q.MODERATE]
    gate = {"strong_U_rank": int(st.U_rank), "strong_M": float(st.M), "strong_M_null_max": float(st.M_null_max),
            "total_U_rank": int(to.U_rank), "total_M": float(to.M), "total_M_null_max": float(to.M_null_max),
            "moderate_U_rank": int(mo.U_rank), "moderate_M": float(mo.M),
            "moderate_M_null_max": float(mo.M_null_max)}
    gate["pass"] = bool(st.U_rank == 1 and to.U_rank == 1 and st.M > max(1.25, st.M_null_max)
                        and to.M > max(1.0, to.M_null_max))
    print(out.round(3).to_string())
    print("gate (c):", json.dumps(gate), "PASS" if gate["pass"] else "FAIL")
    (OUT / "perm_rank_check_c.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    return gate


# --------------------------------------------------------------------------- D-62 real targets (gated)

def gates_passed_rank() -> dict:
    ab, c = OUT / "perm_rank_checks_ab.json", OUT / "perm_rank_check_c.json"
    a = json.loads(ab.read_text()) if ab.exists() else {}
    cc = json.loads(c.read_text()) if c.exists() else {}
    return {"a": bool(a.get("a_pass")), "b": bool(a.get("b_pass")), "c": bool(cc.get("pass"))}


def real_summary_rank(tot: pd.DataFrame, pf: pd.DataFrame, precision: dict, prereg=None,
                      n_sur: int = N_SUR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-62: real dir and vol in the 21-member pool (real + perm surrogates). Per target, lag and
    adjustment: U, M, the surrogates' range and the exact rank p; per feature at the headline lag."""
    prereg = prereg or load_config("prereg")
    lag12 = int(prereg["stability"]["headline_lag_months"])
    pool = ["real"] + datasets(n_sur)
    rows, frows = [], []
    for target in ("dir", "vol"):
        status = "precise" if precision[target]["precise"] else "exploratory (not precise, D-62 (d))"
        for lag in sorted(tot.lag.unique()):
            for adj in (False, True):
                r = _real_row(pooled_stats(_wide(tot, target, int(lag), pool, adj)), target, int(lag), adj)
                rows.append({**r, "status": status})
        sub = pf[(pf.target == target) & (pf.lag == lag12) & pf.data.isin(pool)]
        for f, d in sub.groupby("feature"):
            w = d.pivot(index="tau_a", columns="data", values="g")[pool]
            frows.append({**_real_row(pooled_stats(w), target, lag12, False), "feature": f, "status": status})
    return pd.DataFrame(rows), pd.DataFrame(frows)


def cmd_real_rank(workers: int):
    g = gates_passed_rank()
    if not all(g.values()):
        raise SystemExit(f"D-62 gates not all passed {g}; real dir/vol drift stays withheld")
    prec = json.loads((OUT / "perm_rank_checks_ab.json").read_text())["precision"]
    pr = cmd_pairs([("real", "dir"), ("real", "vol")], workers, "real_pairs", allow_real_signal=True)
    ctrl = pd.read_parquet(OUT / "perm_control_pairs.parquet")
    pf = pd.concat([ctrl[ctrl.target.isin(["dir", "vol"])], pr], ignore_index=True)
    summ, feat = real_summary_rank(totals(pf), pf, prec)
    summ.to_csv(OUT / "perm_rank_real.csv", index=False)
    feat.to_csv(OUT / "perm_rank_real_features_lag12.csv", index=False)
    print(summ.round(3).to_string())
    print(feat.sort_values(["target", "M_real"], ascending=[True, False]).round(3).to_string())
