"""M2 slice evaluation against the go/no-go criteria (PLAN §13 M2, §2 G0, §9.3; D-50; lead-owned).

All functions take the parquet outputs of :func:`ebmregime.runner.compare.run_compare` as frames.
Thresholds come from ``configs/prereg.yaml`` (``hypotheses.g0``, ``m2_go_no_go``, ``statistics``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..stats import median_ci


def _prereg(prereg=None):
    return prereg or load_config("prereg")


def _series(pairs: pd.DataFrame, data: str, target: str, lag: int, col: str = "DR_tot") -> pd.Series:
    s = pairs[(pairs.data == data) & (pairs.target == target) & (pairs.lag == lag)].sort_values("tau_a")
    return s.set_index("tau_a")[col]


def _inside(ci: dict, band) -> bool:
    return bool(np.isfinite(ci["lo"]) and ci["lo"] >= band[0] and ci["hi"] <= band[1])


def g0(pairs: pd.DataFrame, prereg=None, symbol_note: str = "BTCUSDT", surrogate: str = "surrogate:0") -> dict:
    """G0 on the shuffled control and the surrogate(s), plus the pre-specified DR* fallback."""
    pr = _prereg(prereg)
    g = pr["hypotheses"]["g0"]
    lag = int(pr["stability"]["headline_lag_months"])
    band = g["pass_ci_within"]
    lvl = float(g["ci_level"])
    out = {"lag": lag, "band": band, "arms": {}}
    shuf = _series(pairs, "real", "shuf", lag)
    out["arms"]["shuffled"] = median_ci(shuf.values, delta=lag, level=lvl)
    for tgt in ("dir", "vol"):
        s = _series(pairs, surrogate, tgt, lag)
        if len(s):
            out["arms"][f"surrogate_{tgt}"] = median_ci(s.values, delta=lag, level=lvl)
    out["pass_shuffled"] = _inside(out["arms"]["shuffled"], band)
    out["pass_surrogates"] = all(_inside(v, band) for k, v in out["arms"].items() if k.startswith("surrogate"))
    out["pass"] = out["pass_shuffled"] and out["pass_surrogates"]
    # fallback: DR* = DR / median surrogate DR at the same lag (shuffled uses the dir surrogate; D-50)
    star = {}
    sur_dir = _series(pairs, surrogate, "dir", lag)
    if len(sur_dir):
        m = float(np.median(sur_dir))
        star["shuffled"] = median_ci((shuf / m).values, delta=lag, level=lvl)
        for tgt in ("dir", "vol"):
            s = _series(pairs, surrogate, tgt, lag)
            if len(s):
                star[f"surrogate_{tgt}"] = median_ci((s / float(np.median(s))).values, delta=lag, level=lvl)
        out["surrogate_median_DR"] = {t: float(np.median(_series(pairs, surrogate, t, lag)))
                                      for t in ("dir", "vol") if len(_series(pairs, surrogate, t, lag))}
    out["dr_star"] = star
    out["fallback_pass"] = bool(star) and all(_inside(v, band) for v in star.values())
    out["criterion2"] = out["pass_shuffled"] or out["fallback_pass"]
    return out


def gamma(pairs: pd.DataFrame, data: str, target: str, col: str = "DR_adj_tot", prereg=None) -> pd.DataFrame:
    """Drift variogram gamma(lag) = median over t of ``col`` with stationary-bootstrap 95% CIs."""
    pr = _prereg(prereg)
    rows = []
    for lag in pr["stability"]["lags_months"]:
        s = _series(pairs, data, target, int(lag), col)
        if len(s):
            ci = median_ci(s.values, delta=int(lag))
            rows.append({"lag": int(lag), "n": ci["n"], "median": ci["est"], "lo": ci["lo"], "hi": ci["hi"],
                         "block": ci["block"]})
    return pd.DataFrame(rows)


def positive_control(windows: pd.DataFrame, window_features: pd.DataFrame, prereg=None,
                     data: str = "real") -> dict:
    """Criterion 3: vol-target median SNR > 4 for >= 3 features, and OOS R^2 > 0 in >= 80% of months."""
    pr = _prereg(prereg)
    pc = pr["m2_go_no_go"]["positive_control"]
    wf = window_features[(window_features.data == data) & (window_features.target == pc["target"])]
    snr = wf.groupby("feature")["snr"].median().sort_values(ascending=False)
    n_feat = int((snr > float(pc["median_snr_above"])).sum())
    w = windows[(windows.data == data) & (windows.target == pc["target"])]
    share = float((w["skill"] > 0).mean())
    return {"median_snr": snr, "n_features_snr_above": n_feat, "oos_r2_positive_share": share,
            "oos_r2_median": float(w["skill"].median()), "n_months": int(len(w)),
            "pass_snr": n_feat >= int(pc["min_features"]),
            "pass_r2": share >= float(pc["oos_r2_positive_month_share"]),
            "pass": n_feat >= int(pc["min_features"]) and share >= float(pc["oos_r2_positive_month_share"])}


def floor_table(windows: pd.DataFrame, window_features: pd.DataFrame, data: str = "real") -> pd.DataFrame:
    """Per target and feature: median sigma2_seed, sigma2_tot, seed share, amplitude, SNR over windows."""
    wf = window_features[window_features.data == data].copy()
    wf["seed_share_j"] = wf["s2_seed"] / wf["s2_tot"]
    return (wf.groupby(["target", "feature"])
              .agg(s2_seed=("s2_seed", "median"), s2_tot=("s2_tot", "median"),
                   seed_share=("seed_share_j", "median"), A=("A_seedmean", "median"),
                   snr=("snr", "median"), snr_gt4_share=("snr", lambda s: float((s > 4).mean())))
              .reset_index())
