"""Time EBM / XGBoost / LogisticRegression fits on synthetic market-like data.

Purpose: measure per-fit wall-clock (and approximate CPU time / peak RSS) on the
local machine so the cost of a rolling-window study can be extrapolated.

Synthetic data only: 16 continuous AR(1) features (phi from 0.30 to 0.97), a
near-noise binary "direction" target (~51/49, oracle AUC ~0.535) and a stronger
regression "forward volatility" target (oracle R^2 ~0.30). Models are trained on
the first n rows and evaluated on the following --n-test rows.

Example:
    python time_fits.py --out results.json
    python time_fits.py --out quick.json --ns 8760 17520 --sections ebm_cls lr
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from importlib import metadata

import numpy as np
from scipy.signal import lfilter

ALL_SECTIONS = ["ebm_cls", "ebm_reg", "explain", "parallel", "xgb", "lr", "determinism"]

# EBM configurations. "default" leaves every interpret default untouched.
EBM_CONFIGS = {
    "default": {},
    "no_inter": {"interactions": 0},
    "reduced": {"outer_bags": 4, "interactions": 10, "max_bins": 256, "n_jobs": -1},
    "reduced_1t": {"outer_bags": 4, "interactions": 10, "max_bins": 256, "n_jobs": 1},
    "default_1t": {"n_jobs": 1},
    "no_inter_1t": {"interactions": 0, "n_jobs": 1},
}


# --------------------------------------------------------------------------- data
def make_data(n: int, n_features: int = 16, seed: int = 0, dir_signal: float = 0.08,
              vol_r2: float = 0.30, burn_in: int = 1000):
    """Return X (n, p), y_dir (n,) int8, y_vol (n,), and the noiseless signals
    f_dir, f_vol (used only to report oracle AUC / R^2)."""
    rng = np.random.default_rng(seed)
    phis = np.linspace(0.30, 0.97, n_features)
    eps = rng.standard_normal((n + burn_in, n_features))
    X = np.empty_like(eps)
    for j, phi in enumerate(phis):  # stationary AR(1) with unit variance
        X[:, j] = lfilter([np.sqrt(1.0 - phi**2)], [1.0, -phi], eps[:, j])
    X = X[burn_in:]

    def _z(v):
        return (v - v.mean()) / v.std()

    # Direction: weak, partly non-linear signal plus Gaussian noise; 51/49 balance.
    f_dir = _z(0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.5 * np.tanh(X[:, 2] * X[:, 3])
               + 0.3 * (X[:, 4] ** 2 - 1.0))
    latent = dir_signal * f_dir + rng.standard_normal(n)
    y_dir = (latent > np.quantile(latent, 0.49)).astype(np.int8)

    # Forward (log) volatility: stronger signal, target R^2 ~ vol_r2.
    f_vol = _z(0.8 * X[:, 5] + 0.5 * X[:, 6] + 0.4 * np.abs(X[:, 7])
               + 0.3 * X[:, 5] * X[:, 8] + 0.3 * X[:, 15])
    y_vol = np.sqrt(vol_r2) * f_vol + np.sqrt(1.0 - vol_r2) * rng.standard_normal(n)
    return X, y_dir, y_vol, f_dir, f_vol


# ----------------------------------------------------------------------- measuring
def _tree_cpu_and_rss(proc):
    cpu, rss = 0.0, 0
    for p in [proc] + proc.children(recursive=True):
        try:
            t = p.cpu_times()
            cpu += t.user + t.system
            rss += p.memory_info().rss
        except Exception:  # process vanished between listing and query
            pass
    return cpu, rss


class Meter:
    """Wall time, approximate CPU seconds and peak RSS of this process tree
    (includes loky worker processes). RSS is sampled every `interval` s."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        try:
            import psutil
            self.proc = psutil.Process()
        except ImportError:
            self.proc = None

    @contextmanager
    def measure(self):
        res = {"wall_s": None, "cpu_s": None, "peak_rss_mb": None, "peak_delta_mb": None}
        stop = threading.Event()
        peak = [0]
        if self.proc is not None:
            cpu0, rss0 = _tree_cpu_and_rss(self.proc)
            peak[0] = rss0

            def _sample():
                while not stop.is_set():
                    peak[0] = max(peak[0], _tree_cpu_and_rss(self.proc)[1])
                    stop.wait(self.interval)

            th = threading.Thread(target=_sample, daemon=True)
            th.start()
        t0 = time.perf_counter()
        try:
            yield res
        finally:
            res["wall_s"] = time.perf_counter() - t0
            if self.proc is not None:
                stop.set()
                th.join()
                cpu1, rss1 = _tree_cpu_and_rss(self.proc)
                peak[0] = max(peak[0], rss1)
                res["cpu_s"] = cpu1 - cpu0
                res["peak_rss_mb"] = peak[0] / 2**20
                res["peak_delta_mb"] = (peak[0] - rss0) / 2**20


def timed(fn, meter: Meter, repeats: int, warmup_max_s: float, budget_s: float):
    """Run fn(i) repeatedly (i = run index, usable as a seed offset so repeats
    sample different random workloads). The first run is a discarded warm-up if
    it took less than warmup_max_s; otherwise it counts as a measurement. Stops
    once the total time spent exceeds budget_s. Returns (summary, last output)."""
    runs, out, spent = [], None, 0.0
    with meter.measure() as m:
        out = fn(0)
    spent += m["wall_s"]
    warmed = m["wall_s"] < warmup_max_s
    if not warmed:
        runs.append(m)
    while len(runs) < repeats and spent + (runs[-1]["wall_s"] if runs else m["wall_s"]) <= budget_s:
        with meter.measure() as m:
            out = fn(len(runs) + 1)
        spent += m["wall_s"]
        runs.append(m)
    walls = [r["wall_s"] for r in runs]
    cpus = [r["cpu_s"] for r in runs if r["cpu_s"] is not None]
    peaks = [r["peak_delta_mb"] for r in runs if r["peak_delta_mb"] is not None]
    return {
        "wall_median_s": statistics.median(walls),
        "wall_all_s": walls,
        "cpu_median_s": statistics.median(cpus) if cpus else None,
        "peak_delta_mb_max": max(peaks) if peaks else None,
        "n_measured": len(walls),
        "warmup_discarded": warmed,
    }, out


# ------------------------------------------------------------------------- models
def ebm_cls(params, seed=42):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=seed, **params)


def ebm_reg(params, seed=42):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(random_state=seed, **params)


def _fit_ebm_single_thread(X, y, params, seed):
    """Top-level so joblib/loky can pickle it. Returns wall seconds of the fit."""
    p = dict(params, n_jobs=1)
    t0 = time.perf_counter()
    ebm_cls(p, seed).fit(X, y)
    return time.perf_counter() - t0


def _noop(i):
    return i


# ----------------------------------------------------------------------- sections
def section_ebm_cls(args, data, meter, results):
    from sklearn.metrics import roc_auc_score
    for n in args.ns:
        X, y = data["X"][:n], data["y_dir"][:n]
        Xt, yt = data["X"][n:n + args.n_test], data["y_dir"][n:n + args.n_test]
        for cfg in args.cls_configs:
            params = EBM_CONFIGS[cfg]
            summ, model = timed(lambda i: ebm_cls(params, seed=42 + i).fit(X, y), meter,
                                args.repeats, args.warmup_max_s, args.budget_s)
            summ.update(section="ebm_cls", model="EBM-C", config=cfg, n=n,
                        n_terms=len(model.term_features_),
                        test_auc=float(roc_auc_score(yt, model.predict_proba(Xt)[:, 1])))
            results.append(summ)
            _print_row(summ)


def section_ebm_reg(args, data, meter, results):
    from sklearn.metrics import r2_score
    n = args.n_par
    X, y = data["X"][:n], data["y_vol"][:n]
    Xt, yt = data["X"][n:n + args.n_test], data["y_vol"][n:n + args.n_test]
    for cfg in ("reduced", "reduced_1t"):
        params = EBM_CONFIGS[cfg]
        summ, model = timed(lambda i: ebm_reg(params, seed=42 + i).fit(X, y), meter,
                            args.repeats, args.warmup_max_s, args.budget_s)
        summ.update(section="ebm_reg", model="EBM-R", config=cfg, n=n,
                    n_terms=len(model.term_features_),
                    test_r2=float(r2_score(yt, model.predict(Xt))))
        results.append(summ)
        _print_row(summ)


def section_explain(args, data, meter, results):
    n = args.n_par
    X, y = data["X"][:n], data["y_dir"][:n]
    Xt = data["X"][n:n + args.n_test]
    model = ebm_cls(EBM_CONFIGS["reduced"]).fit(X, y)
    ops = {
        "term_importances": lambda i: model.term_importances(),
        "explain_global": lambda i: model.explain_global(),
        f"predict_proba_{len(Xt)}": lambda i: model.predict_proba(Xt),
        f"explain_local_{min(len(Xt), 500)}": lambda i: model.explain_local(Xt[:500]),
    }
    for name, fn in ops.items():
        summ, _ = timed(fn, meter, max(args.repeats, 5), 1e9, args.budget_s)
        summ.update(section="explain", model="EBM-C", config="reduced", n=n, op=name)
        results.append(summ)
        _print_row(summ)


def section_parallel(args, data, meter, results):
    """Throughput: sequential fits with n_jobs=-1 vs P concurrent n_jobs=1 fits."""
    from joblib import Parallel, delayed
    n = args.n_par
    X, y = data["X"][:n], data["y_dir"][:n]
    base = {k: v for k, v in EBM_CONFIGS[args.par_config].items() if k != "n_jobs"}
    cfg = args.par_config.removesuffix("_1t")

    # Strategy A: sequential, each fit parallel over outer bags (n_jobs=-1).
    ebm_cls(dict(base, n_jobs=-1)).fit(X[:2000], y[:2000])  # warm the loky pool
    with meter.measure() as m:
        for i in range(args.par_seq_fits):
            ebm_cls(dict(base, n_jobs=-1), seed=100 + i).fit(X, y)
    rec = {"section": "parallel", "model": "EBM-C", "config": cfg, "n": n,
           "strategy": "sequential_njobs-1", "workers": 1, "fits": args.par_seq_fits,
           "wall_total_s": m["wall_s"], "cpu_total_s": m["cpu_s"],
           "peak_delta_mb": m["peak_delta_mb"],
           "fits_per_hour": 3600.0 * args.par_seq_fits / m["wall_s"]}
    results.append(rec)
    _print_par(rec)

    # Strategy B: P workers, each running single-threaded fits.
    for P in args.par_workers:
        k = P * args.par_fits_per_worker
        Parallel(n_jobs=P, backend="loky")(delayed(_noop)(i) for i in range(P))  # spawn
        with meter.measure() as m:
            per_fit = Parallel(n_jobs=P, backend="loky")(
                delayed(_fit_ebm_single_thread)(X, y, base, 200 + i) for i in range(k))
        rec = {"section": "parallel", "model": "EBM-C", "config": cfg + "_1t", "n": n,
               "strategy": "concurrent_njobs1", "workers": P, "fits": k,
               "wall_total_s": m["wall_s"], "cpu_total_s": m["cpu_s"],
               "peak_delta_mb": m["peak_delta_mb"],
               "per_fit_median_s": statistics.median(per_fit),
               "per_fit_max_s": max(per_fit),
               "fits_per_hour": 3600.0 * k / m["wall_s"]}
        results.append(rec)
        _print_par(rec)


def section_xgb(args, data, meter, results):
    import xgboost as xgb
    n = args.n_par
    X, y, yv = data["X"][:n], data["y_dir"][:n], data["y_vol"][:n]
    n_val = int(0.15 * n)  # temporal validation split: last 15% of the train window
    Xtr, ytr, Xva, yva = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    common = dict(tree_method="hist", n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=0)

    def fit_cls(n_jobs, es, seed=0):
        m = xgb.XGBClassifier(**dict(common, random_state=seed), n_jobs=n_jobs,
                              eval_metric="logloss", early_stopping_rounds=30 if es else None)
        return m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    full_model = None
    for n_jobs in (-1, 1):
        for es in (True, False):
            summ, model = timed(lambda i: fit_cls(n_jobs, es, i), meter,
                                args.repeats, args.warmup_max_s, args.budget_s)
            summ.update(section="xgb", model="XGB-C", n=n, n_jobs=n_jobs,
                        config=f"hist300_d4_{'es30' if es else 'noES'}_nj{n_jobs}",
                        n_trees=int(model.best_iteration + 1) if es else 300)
            results.append(summ)
            _print_row(summ)
            if not es and n_jobs == -1:
                full_model = model

    def fit_reg(seed):
        m = xgb.XGBRegressor(**dict(common, random_state=seed), n_jobs=-1,
                             early_stopping_rounds=30)
        return m.fit(X[:-n_val], yv[:-n_val], eval_set=[(Xva, yv[-n_val:])], verbose=False)

    summ, model = timed(fit_reg, meter, args.repeats, args.warmup_max_s, args.budget_s)
    summ.update(section="xgb", model="XGB-R", n=n, config="hist300_d4_es30_nj-1",
                n_trees=int(model.best_iteration + 1))
    results.append(summ)
    _print_row(summ)

    # TreeSHAP on a reference sample, using the full 300-tree model (upper bound).
    ref = X[-args.shap_ref:]
    booster = full_model.get_booster()
    dref = xgb.DMatrix(ref)
    for nthread in (-1, 1):
        booster.set_param({"nthread": nthread if nthread > 0 else os.cpu_count()})
        summ, contrib = timed(lambda i: booster.predict(dref, pred_contribs=True), meter,
                              args.repeats, args.warmup_max_s, args.budget_s)
        summ.update(section="xgb", model="XGB-C", n=len(ref), op="pred_contribs",
                    config=f"300trees_nthread{nthread}", n_trees=300)
        results.append(summ)
        _print_row(summ)
    booster.set_param({"nthread": os.cpu_count()})
    try:
        import shap
        expl = shap.TreeExplainer(full_model)
        summ, sv = timed(lambda i: expl.shap_values(ref), meter,
                         args.repeats, args.warmup_max_s, args.budget_s)
        sv = np.asarray(sv)
        summ.update(section="xgb", model="XGB-C", n=len(ref), op="shap.TreeExplainer",
                    config="300trees", n_trees=300,
                    max_abs_diff_vs_pred_contribs=float(np.max(np.abs(sv - contrib[:, :-1]))))
        results.append(summ)
        _print_row(summ)
    except ImportError as e:
        results.append({"section": "xgb", "op": "shap.TreeExplainer", "error": repr(e)})
        print(f"shap unavailable: {e!r}")


def section_lr(args, data, meter, results):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n = args.n_par
    X, y = data["X"][:n], data["y_dir"][:n]

    def fit(i):
        model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
        return model.fit(X, y)
    summ, _ = timed(fit, meter, max(args.repeats, 5), 1e9, args.budget_s)
    summ.update(section="lr", model="LR", n=n, config="lbfgs_standardised")
    results.append(summ)
    _print_row(summ)


def section_determinism(args, data, meter, results):
    """Fit config 'reduced' twice with n_jobs=-1 and twice with n_jobs=1 (same seed)."""
    n = args.det_n
    X, y = data["X"][:n], data["y_dir"][:n]
    base = {k: v for k, v in EBM_CONFIGS["reduced"].items() if k != "n_jobs"}
    models = {}
    for nj in (-1, 1):
        for rep in (0, 1):
            models[(nj, rep)] = ebm_cls(dict(base, n_jobs=nj), seed=42).fit(X, y)
    models["other_seed"] = ebm_cls(dict(base, n_jobs=-1), seed=43).fit(X, y)

    def same(a, b):
        if a.term_features_ != b.term_features_:
            return False
        return all(np.array_equal(s, t) for s, t in zip(a.term_scores_, b.term_scores_)) \
            and np.array_equal(np.atleast_1d(a.intercept_), np.atleast_1d(b.intercept_))

    def maxdiff(a, b):
        if a.term_features_ != b.term_features_:
            return None
        return float(max(np.max(np.abs(s - t)) for s, t in zip(a.term_scores_, b.term_scores_)))

    rec = {"section": "determinism", "model": "EBM-C", "config": "reduced", "n": n,
           "njobs-1_rep_identical": same(models[(-1, 0)], models[(-1, 1)]),
           "njobs1_rep_identical": same(models[(1, 0)], models[(1, 1)]),
           "njobs-1_vs_njobs1_identical": same(models[(-1, 0)], models[(1, 0)]),
           "njobs-1_vs_njobs1_maxabsdiff": maxdiff(models[(-1, 0)], models[(1, 0)]),
           "seed42_vs_seed43_identical": same(models[(-1, 0)], models["other_seed"]),
           "seed42_vs_seed43_same_interaction_set":
               models[(-1, 0)].term_features_ == models["other_seed"].term_features_}
    results.append(rec)
    print("determinism:", json.dumps({k: v for k, v in rec.items() if k != "section"}))


# ------------------------------------------------------------------------ output
def _print_row(r):
    label = r.get("op") or r.get("config")
    cpu = f"{r['cpu_median_s']:.1f}" if r.get("cpu_median_s") is not None else "-"
    mem = f"{r['peak_delta_mb_max']:.0f}" if r.get("peak_delta_mb_max") is not None else "-"
    extra = {k: r[k] for k in ("test_auc", "test_r2", "n_trees", "n_terms") if k in r}
    extra = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in extra.items())
    print(f"{r['section']:<12}{r.get('model', ''):<7}{label:<28}{r['n']:>7}"
          f"{r['wall_median_s']:>10.3f}{cpu:>9}{mem:>8}{r['n_measured']:>4}  {extra}",
          flush=True)


def _print_par(r):
    print(f"{r['section']:<12}{r['config']:<12}{r['strategy']:<22} workers={r['workers']:<3}"
          f" fits={r['fits']:<4} wall={r['wall_total_s']:.1f}s cpu={r['cpu_total_s'] or 0:.0f}s "
          f"fits/h={r['fits_per_hour']:.0f}"
          + (f" per_fit_median={r['per_fit_median_s']:.1f}s" if "per_fit_median_s" in r else ""),
          flush=True)


def machine_info():
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_freq_mhz"] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        info["load_cpu_percent_1s"] = psutil.cpu_percent(interval=1.0)
    except ImportError:
        pass
    pkgs = {}
    for p in ("interpret-core", "interpret", "xgboost", "shap", "scikit-learn", "numpy",
              "pandas", "scipy", "joblib", "psutil"):
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    info["packages"] = pkgs
    return info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="path of the JSON results file")
    ap.add_argument("--sections", nargs="+", default=ALL_SECTIONS, choices=ALL_SECTIONS)
    ap.add_argument("--ns", nargs="+", type=int, default=[8760, 17520, 35040, 70080],
                    help="training sizes for the EBM classifier grid")
    ap.add_argument("--cls-configs", nargs="+",
                    default=["default", "no_inter", "reduced", "reduced_1t"],
                    choices=list(EBM_CONFIGS))
    ap.add_argument("--n-par", type=int, default=35040,
                    help="training size for regressor/explain/parallel/xgb/lr sections")
    ap.add_argument("--n-test", type=int, default=2880, help="test rows (1 month of 15m)")
    ap.add_argument("--det-n", type=int, default=17520, help="training size for determinism")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-max-s", type=float, default=15.0,
                    help="first run is a discarded warm-up only if faster than this")
    ap.add_argument("--budget-s", type=float, default=300.0,
                    help="stop repeating a measurement once this much time is spent")
    ap.add_argument("--par-workers", nargs="+", type=int, default=[8, 16],
                    help="concurrent single-threaded fit counts to test")
    ap.add_argument("--par-config", default="reduced", choices=list(EBM_CONFIGS),
                    help="EBM config for the parallel section (n_jobs is overridden)")
    ap.add_argument("--par-fits-per-worker", type=int, default=2)
    ap.add_argument("--par-seq-fits", type=int, default=3)
    ap.add_argument("--shap-ref", type=int, default=5000, help="TreeSHAP reference rows")
    ap.add_argument("--dir-signal", type=float, default=0.08)
    ap.add_argument("--vol-r2", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n_total = max(args.ns + [args.n_par, args.det_n]) + args.n_test
    X, y_dir, y_vol, f_dir, f_vol = make_data(n_total, seed=args.seed,
                                              dir_signal=args.dir_signal, vol_r2=args.vol_r2)
    from sklearn.metrics import r2_score, roc_auc_score
    data_info = {"rows": n_total, "features": X.shape[1], "pos_rate": float(y_dir.mean()),
                 "oracle_auc_dir": float(roc_auc_score(y_dir, f_dir)),
                 "oracle_r2_vol": float(r2_score(y_vol, np.sqrt(args.vol_r2) * f_vol)),
                 "lag1_autocorr_min_max": [float(np.corrcoef(X[1:, j], X[:-1, j])[0, 1])
                                           for j in (0, X.shape[1] - 1)]}
    info = machine_info()
    print(json.dumps({"machine": info, "data": data_info}, indent=1), flush=True)
    print(f"{'section':<12}{'model':<7}{'config/op':<28}{'n':>7}{'wall_med':>10}"
          f"{'cpu_s':>9}{'peakMB':>8}{'reps':>4}", flush=True)

    data = {"X": X, "y_dir": y_dir, "y_vol": y_vol}
    meter, results = Meter(), []
    t_start = time.perf_counter()
    funcs = {"ebm_cls": section_ebm_cls, "ebm_reg": section_ebm_reg,
             "explain": section_explain, "parallel": section_parallel, "xgb": section_xgb,
             "lr": section_lr, "determinism": section_determinism}
    out = {"args": vars(args), "machine": info, "data": data_info,
           "ebm_configs": EBM_CONFIGS, "results": results}
    for name in args.sections:
        funcs[name](args, data, meter, results)
        out["elapsed_s"] = time.perf_counter() - t_start
        with open(args.out, "w") as fh:  # rewrite after each section (partial results)
            json.dump(out, fh, indent=1, default=str)
    print(f"total elapsed {time.perf_counter() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
