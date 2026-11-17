"""Lead's end-to-end check of the noise floor on a stationary synthetic process (M2 verification).

A stationary hourly process with the 16 panel feature names is generated (AR(1) features of
different persistence, true hour/dow, a learnable regression target with AR(1) noise that mimics
overlapping 8-hour labels, and a near-noise binary target). The full M2 pipeline is applied with
real EBM fits (prereg settings, 14 bags): rolling 12-month windows, time-ordered bags with purge,
5 seed-only + 10 half-sample fits, pair X_ref, 50 cells, re-centred shapes. Because the process is
stationary, the drift ratio at zero overlap (lag 12) must be about 1 (PLAN §6.4, §9.3).

DR is computed twice: with ``ebmregime.stability.floor.drift_pair`` and with the lead's naive
reference ``tests/m2_reference.py``; the two must agree to 1e-9.

Usage: python scripts/verify_floor_synthetic.py [--workers 16] [--out results/verify]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from ebmregime import panel as P  # noqa: E402
from ebmregime.config import load_config  # noqa: E402
from ebmregime.splits import half_sample, replicate_plan, retrain_schedule, time_ordered_bags, train_mask  # noqa: E402

H = 8
START, END = "2019-01-01", "2024-01-01"
FIRST_TAU, LAST_TAU = "2020-01-01", "2023-12-01"


def make_process(seed: int = 7) -> pd.DataFrame:
    ts = pd.date_range(START, END, freq="h", tz="UTC", inclusive="left")
    n = len(ts)
    rng = np.random.default_rng(seed)
    names = P.feature_names()
    X = {}
    phis = [0.0, 0.5, 0.9, 0.97, 0.99, 0.0, 0.8, 0.95, 0.0, 0.6, 0.9, 0.98, None, None, 0.7, 0.3]
    for name, phi in zip(names, phis):
        if name == "hour":
            X[name] = ts.hour.to_numpy().astype(np.int64)
        elif name == "dow":
            X[name] = ts.day_of_week.to_numpy().astype(np.int64)
        else:
            e = rng.standard_normal(n)
            x = np.empty(n)
            x[0] = e[0] / np.sqrt(1 - phi**2) if phi < 1 else e[0]
            for i in range(1, n):
                x[i] = phi * x[i - 1] + e[i]
            X[name] = x * np.sqrt(1 - phi**2)          # unit variance
    df = pd.DataFrame(X, index=pd.DatetimeIndex(ts, name="ts"))
    # AR(1) noise with phi = 1 - 1/H: autocorrelation like overlapping H-bar labels
    u = rng.standard_normal(n)
    noise = np.empty(n)
    noise[0] = u[0]
    phi_n = 1 - 1 / H
    for i in range(1, n):
        noise[i] = phi_n * noise[i - 1] + np.sqrt(1 - phi_n**2) * u[i]
    hour_eff = 0.3 * np.cos(2 * np.pi * df["hour"] / 24)
    wkend = 0.2 * (df["dow"] >= 5)
    signal = (0.4 * np.tanh(df["rv_ratio_4h"]) + 0.3 * df["rv_level_pct"] - 0.2 * np.abs(df["ret_1h"])
              + hour_eff + wkend)
    df["y_vol"] = signal + 0.8 * noise
    p = 1 / (1 + np.exp(-(0.08 * df["ret_4h"] + 0.05 * np.sin(df["qv_z_1h"]))))
    df["y_dir"] = (rng.random(n) < p).astype(float)
    return df


def fit_one(df, tau, target, rep, prereg):
    from ebmregime.models.ebm import export_fit, fit_ebm

    v = prereg["validation"]
    tm = train_mask(df.index, tau, 12, H)
    ts_tr = df.index[tm]
    Xtr = df.loc[tm, P.feature_names()]
    ytr = df.loc[tm, f"y_{target}"].to_numpy()
    if target == "dir":
        ytr = ytr.astype(np.int64)                     # float 0/1 labels would become string classes
    if rep["protocol"] == "half":
        m, vs = half_sample(ts_tr, H, v["tail_frac"], v["block_hours"], rep["sample_seed"])
        bags = time_ordered_bags(ts_tr[m], H, v["outer_bags"], v["tail_frac"], v["block_hours"],
                                 v["bag_train_frac"], seed=rep["seed"], val_start=vs, block_origin=ts_tr[0])
        Xtr, ytr = Xtr[m], ytr[m]
    else:
        bags = time_ordered_bags(ts_tr, H, v["outer_bags"], v["tail_frac"], v["block_hours"],
                                 v["bag_train_frac"], seed=rep["seed"])
    kind = "regressor" if target == "vol" else "classifier"
    model, info = fit_ebm(Xtr, ytr, bags, kind, rep["seed"])
    return (str(tau.date()), target, rep["protocol"], rep["r"]), export_fit(model), info["fit_seconds"]


def fit_rows_mask(df, tau, tail_frac):
    tm = train_mask(df.index, tau, 12, H)
    pos = np.flatnonzero(tm)
    keep = ~P.purged_before_tail(df.index[pos], H, tail_frac)
    out = np.zeros(len(df), bool)
    out[pos[keep]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/verify")
    ap.add_argument("--targets", default="vol,dir")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    prereg = load_config("prereg")
    df = make_process()
    taus = retrain_schedule(FIRST_TAU, LAST_TAU)
    targets = args.targets.split(",")
    fits = {}
    for tgt in targets:                                   # one pickle per target, so a failure loses less
        fits_path = out / f"synthetic_fits_{tgt}.pkl"
        if fits_path.exists():
            fits.update(pickle.loads(fits_path.read_bytes()))
            continue
        tasks = [delayed(fit_one)(df, tau, tgt, rep, prereg)
                 for tau in taus for rep in replicate_plan("SYNTH", tau, tgt, prereg["replicates"])]
        t0 = time.time()
        res = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(tasks)
        print(f"{tgt}: {len(tasks)} fits in {time.time() - t0:.0f} s wall; "
              f"mean fit {np.mean([r[2] for r in res]):.2f} s", flush=True)
        got = {r[0]: r[1] for r in res}
        fits_path.write_bytes(pickle.dumps(got))
        fits.update(got)

    import m2_reference as REF
    from ebmregime.explain import cell_shapes_many, make_cells, make_xref
    from ebmregime.models.ebm import eval_terms_from_export
    from ebmregime.splits import overlap_fraction
    from ebmregime.stability.floor import drift_pair

    tail = prereg["validation"]["tail_frac"]
    rows = []
    for tgt in targets:
        for lag in (1, 3, 12):
            for i in range(len(taus) - lag):
                a, b = taus[i], taus[i + lag]
                pos = make_xref([fit_rows_mask(df, a, tail), fit_rows_mask(df, b, tail)], 20000, 1)
                X = df.iloc[pos][P.feature_names()]
                grid = make_cells(X)
                sh = {}
                for w in (a, b):
                    for prot, R in (("seed", 5), ("half", 10)):
                        C = np.stack([eval_terms_from_export(fits[(str(w.date()), tgt, prot, r)], X) for r in range(R)])
                        sh[(w, prot)] = cell_shapes_many(C, grid)
                rho = overlap_fraction(a, b, 12)
                xr = [None if nm in P.nominal_features() else grid.x_rep[j] for j, nm in enumerate(P.feature_names())]
                res = drift_pair({"seed": sh[(a, "seed")], "half": sh[(a, "half")]},
                                 {"seed": sh[(b, "seed")], "half": sh[(b, "half")]}, grid.weights, rho, x_rep=xr)
                ref = REF.drift(sh[(a, "seed")], sh[(a, "half")], sh[(b, "seed")], sh[(b, "half")],
                                grid.weights, rho)
                diff = max(abs(float(res[k]) - float(ref[k])) for k in ("DR_tot", "DR_adj_tot", "seed_share_a"))
                diff = max(diff, float(np.nanmax(np.abs(np.asarray(res["DR"]) - ref["DR"]))))
                rows.append({"target": tgt, "lag": lag, "tau_a": a, "DR_tot": float(res["DR_tot"]),
                             "DR_adj_tot": float(res["DR_adj_tot"]), "ref_DR_tot": ref["DR_tot"],
                             "max_abs_diff_vs_ref": diff, "seed_share": float(res["seed_share_a"]),
                             "snr_max": float(np.max(res["snr_a"]))})
    r = pd.DataFrame(rows)
    r.to_csv(out / "synthetic_floor.csv", index=False)
    summ = r.groupby(["target", "lag"]).agg(n=("DR_tot", "size"), DR_med=("DR_tot", "median"),
                                             DRadj_med=("DR_adj_tot", "median"),
                                             seed_share_med=("seed_share", "median"),
                                             snr_max_med=("snr_max", "median"),
                                             max_diff=("max_abs_diff_vs_ref", "max"))
    print(summ.to_string())
    (out / "synthetic_floor_summary.json").write_text(summ.reset_index().to_json(orient="records", indent=1))


if __name__ == "__main__":
    main()
