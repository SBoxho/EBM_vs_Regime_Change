"""D-67 (1): H1-dir signal condition under the surrogate floor (lead-owned).

SNR_D,j(tau) = A2_D,j(tau) / sigma2_sur,j(tau). A2 is the mean squared amplitude of the 5 seed-only
shapes on the window's own X_ref and cells (as in the slice window stage); sigma2_sur is half the mean
lag-12 cross-window distance g of the same target's perm surrogates over the pairs containing tau.
Pool: real dir, real shuf, shuf on perm surrogates 0-19; real dir "carries signal" if its median
SNR is the largest of the 22.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import panel as P
from . import surfloor as S
from . import surfloor_perm as Q
from .config import load_config

SLICE = S.ROOT / "results" / "slice" / "compare"
LAG = 12


def _window_a2(pnl, data, target, tau, keys, reader, prereg) -> list[dict]:
    from .explain import cell_shapes_many
    from .models.ebm import eval_terms_from_export
    from .runner import compare as C
    from .stability import metrics as M

    fm = C._fit_mask(pnl, tau, target, S.H, prereg)
    pos = C._xref([fm], prereg, S.SYMBOL, data, target, tau, "window")
    X = pnl.iloc[pos][P.feature_names()]
    grid = C._cells(X, prereg)
    sh = cell_shapes_many(np.stack([eval_terms_from_export(reader(k)[0], X) for k in keys]), grid)
    return [{"data": data, "target": target, "tau": str(pd.Timestamp(tau).date()), "feature": nm,
             "A2_mean": float(np.mean(M.amplitude(sh[j], grid.weights[j]) ** 2))}
            for j, nm in enumerate(P.feature_names())]


def perm_window_task(data, target, tau, keys, prereg):
    return _window_a2(Q.perm_panel(S.SYMBOL, int(data.split(":")[1])), data, target, tau, keys, Q._fit, prereg)


def real_window_task(target, tau, keys, prereg):
    from .runner import compare as C

    return _window_a2(P.load_panel(S.SYMBOL), "real", target, tau, keys, C._fit, prereg)


def surrogate_a2(workers: int, n_sur: int = S.N_SUR) -> pd.DataFrame:
    from joblib import Parallel, delayed

    from .runner.compare import group_jobs

    prereg = load_config("prereg")
    jobs = S.seed_jobs(S.datasets(n_sur), ["shuf"], prereg)
    keys = {j: Q.key_fn(j) for j in jobs}
    tasks = [delayed(perm_window_task)(data, target, tau, g[tau]["seed"], prereg)
             for (data, _, target), g in group_jobs(jobs, keys).items() for tau in g]
    res = Parallel(n_jobs=workers, backend="loky", batch_size=4, verbose=2)(tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(S.OUT / "snr_surrogate_shuf_a2.parquet")
    return df


def floors(pf: pd.DataFrame, target: str, members, exclude: str | None = None) -> pd.DataFrame:
    """sigma2_sur,j(tau): half the mean of g over ``members`` (minus ``exclude``) and the lag-12
    pairs containing tau. Index (tau, feature)."""
    sub = pf[(pf.target == target) & (pf.lag == LAG) & pf.data.isin([m for m in members if m != exclude])]
    long = pd.concat([sub.rename(columns={"tau_a": "tau"})[["tau", "feature", "g"]],
                      sub.rename(columns={"tau_b": "tau"})[["tau", "feature", "g"]]])
    long["tau"] = pd.to_datetime(long.tau).dt.strftime("%Y-%m-%d")
    return (0.5 * long.groupby(["tau", "feature"]).g.mean()).rename("s2_sur")


def snr_table(a2: pd.DataFrame, pf: pd.DataFrame, n_sur: int = S.N_SUR) -> pd.DataFrame:
    """Per member, tau and feature: SNR. a2 has columns data, target, tau, feature, A2_mean."""
    sur = S.datasets(n_sur)
    out = []
    for (data, target), d in a2.groupby(["data", "target"]):
        fl = floors(pf, target, sur, exclude=data if data != "real" else None)
        m = d.join(fl, on=["tau", "feature"])
        m["snr"] = m.A2_mean / m.s2_sur
        m["member"] = f"real {target}" if data == "real" else f"{data} {target}"
        out.append(m)
    return pd.concat(out, ignore_index=True)


def decide(t: pd.DataFrame) -> dict:
    Sd = t.groupby("member").snr.median()
    noise = Sd.drop(index="real dir")
    s_dir = float(Sd["real dir"])
    per = t.groupby(["member", "feature"]).snr.median().unstack("feature")
    feat = pd.DataFrame({"real_dir": per.loc["real dir"], "noise_max": per.drop(index="real dir").max(),
                         "noise_median": per.drop(index="real dir").median()})
    feat["above_noise_max"] = feat.real_dir > feat.noise_max
    shuf_rank = int((noise >= noise["real shuf"]).sum())
    out = {"S_real_dir": s_dir, "S_noise_median": float(noise.median()), "S_noise_max": float(noise.max()),
           "ratio_to_noise_median": s_dir / float(noise.median()), "rank_from_top": int((Sd >= s_dir).sum()),
           "pool": len(Sd), "p_upper": float((Sd >= s_dir).sum() / len(Sd)),
           "signal": bool(s_dir > noise.max()), "real_shuf_rank_among_noise": shuf_rank,
           "share_snr_gt4": {m: float((d.snr > 4).mean()) for m, d in t.groupby("member")
                             if m in ("real dir", "real shuf", "real vol")},
           "n_features_above_noise_max": int(feat.above_noise_max.sum())}
    return out, feat


def cmd_snr(workers: int):
    pf = pd.read_parquet(S.OUT / "perm_control_pairs.parquet")
    wf = pd.read_parquet(SLICE / "window_features.parquet")
    real = wf[(wf.data == "real") & wf.target.isin(["dir", "shuf", "vol"])][["data", "target", "tau", "feature",
                                                                              "A2_mean"]]
    real = real.assign(tau=pd.to_datetime(real.tau).dt.strftime("%Y-%m-%d"))
    p_sur = S.OUT / "snr_surrogate_shuf_a2.parquet"
    sa = pd.read_parquet(p_sur) if p_sur.exists() else surrogate_a2(workers)
    t = snr_table(pd.concat([real, sa], ignore_index=True), pf)
    t.to_parquet(S.OUT / "snr_table.parquet")
    pool = t[t.member != "real vol"]
    out, feat = decide(pool)
    out["S_real_vol"] = float(t[t.member == "real vol"].snr.median())
    feat.to_csv(S.OUT / "snr_dir_features.csv")
    (S.OUT / "snr_dir_decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(feat.round(3).to_string())
    print(json.dumps(out, indent=1))
    return out
