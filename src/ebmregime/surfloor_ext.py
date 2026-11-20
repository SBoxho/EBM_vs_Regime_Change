"""D-65: perm surrogates 20-119 for vol only, and the 121-member pooled rank test (lead-owned).

Reuses :mod:`surfloor_perm` unchanged (same generator, seed stream, cache and keys), so the
D-61 fits of surrogates 0-19 stay valid and are pooled with the new ones.
"""

from __future__ import annotations

import json
import time

import pandas as pd

from . import surfloor as S
from . import surfloor_perm as Q
from .config import load_config

LO, HI = 20, 120
TARGET = "vol"


def ext_datasets(lo: int = LO, hi: int = HI) -> list[str]:
    return [f"surrogate:{i}" for i in range(lo, hi)]


def ext_jobs(lo: int = LO, hi: int = HI, prereg=None):
    return S.seed_jobs(ext_datasets(lo, hi), [TARGET], prereg)


def cmd_fit(workers: int, lo: int = LO, hi: int = HI):
    from .runner.run import run_jobs

    jobs = ext_jobs(lo, hi)
    print(f"{len(jobs)} seed-only perm vol jobs (surrogates {lo}-{hi - 1}), cache {Q.CACHE}", flush=True)
    s = run_jobs(jobs, n_workers=workers, job_fn=Q.run_perm_job, key_fn=Q.key_fn, root=Q.CACHE)
    print(json.dumps(s, default=str), flush=True)
    return s


def cmd_pairs(workers: int, lo: int = LO, hi: int = HI) -> pd.DataFrame:
    from joblib import Parallel, delayed

    from .runner import cache
    from .runner.compare import group_jobs

    prereg = load_config("prereg")
    lags = list(prereg["stability"]["lags_months"])
    jobs = ext_jobs(lo, hi, prereg)
    keys = {j: Q.key_fn(j) for j in jobs}
    done = cache.finished_keys(Q.CACHE)
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
                tasks.append(delayed(Q.pair_perm_task)(data, target, a, b, lag, g[a]["seed"], g[b]["seed"], prereg))
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", batch_size=8, verbose=2)(tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(S.OUT / "perm_ext_vol_pairs.parquet")
    print(f"{len(tasks)} pairs in {time.time() - t0:.0f} s", flush=True)
    return df


def pooled_vol_pairs() -> pd.DataFrame:
    base = pd.read_parquet(S.OUT / "perm_control_pairs.parquet")
    base = base[(base.target == TARGET) & base.data.isin(S.datasets(LO))]
    ext = pd.read_parquet(S.OUT / "perm_ext_vol_pairs.parquet")
    real = pd.read_parquet(S.OUT / "real_pairs.parquet")
    real = real[real.target == TARGET]
    return pd.concat([base, ext, real], ignore_index=True)


def summarise(pf: pd.DataFrame | None = None, prereg=None) -> dict:
    """D-65: (a) and (d) on the 120 surrogates; real vol in the 121-member pool, every lag and per feature."""
    prereg = prereg or load_config("prereg")
    pf = pooled_vol_pairs() if pf is None else pf
    band = prereg["hypotheses"]["g0"]["pass_ci_within"]
    lag12 = int(prereg["stability"]["headline_lag_months"])
    sur = S.datasets(HI)
    pool = ["real"] + sur
    tot = S.totals(pf)
    cal = S.pooled_stats(S._wide(tot, TARGET, lag12, sur))
    med, lo, hi = float(cal.M.median()), S._q(cal.M, .025), S._q(cal.M, .975)
    gate_a = {"median_M": med, "q025": lo, "q975": hi, "pass": bool(0.9 <= med <= 1.11)}
    prec = {"q025": lo, "q975": hi, "precise": bool(band[0] <= lo and hi <= band[1])}
    rows = [S._real_row(S.pooled_stats(S._wide(tot, TARGET, int(lag), pool, adj)), TARGET, int(lag), adj)
            for lag in sorted(tot.lag.unique()) for adj in (False, True)]
    summ = pd.DataFrame(rows)
    sub = pf[(pf.lag == lag12) & (pf.target == TARGET)]
    feat = pd.DataFrame([{**S._real_row(S.pooled_stats(d.pivot(index="tau_a", columns="data", values="g")[pool]),
                                        TARGET, lag12, False), "feature": f} for f, d in sub.groupby("feature")])
    summ.to_csv(S.OUT / "perm_ext_real_vol.csv", index=False)
    feat.to_csv(S.OUT / "perm_ext_real_vol_features_lag12.csv", index=False)
    out = {"a": gate_a, "precision": prec, "n_pool": len(pool),
           "lag12": summ[(summ.lag == lag12) & (~summ.adj)].iloc[0].to_dict()}
    (S.OUT / "perm_ext_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("(a) vol, 120 surrogates:", json.dumps(gate_a), "PASS" if gate_a["pass"] else "FAIL")
    print("(d) precision:", json.dumps(prec))
    print(summ.round(4).to_string())
    print(feat.sort_values("M_real", ascending=False).round(3).to_string())
    return out
