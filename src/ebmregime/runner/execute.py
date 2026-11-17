"""One fit job: data, bags, EBM fit, and what is stored (M2_SPEC §5 "Fit job").

``run_fit_job(job) -> (arrays, meta)`` is a pure function of the job (plus the frozen configs, the
panel and the fit code, all of which are in the cache key). It does not write anything: the
runner's worker wrapper (:func:`ebmregime.runner.run.execute_one`) adds the key and writes the npz.

``ebmregime.models.ebm`` and ``ebmregime.stability.surrogate`` are imported lazily (inside
``_model_api`` and ``_surrogate_panel``) so that this module imports without them; tests replace
those two hooks to check the plumbing without fitting.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import subprocess
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import panel as P
from ..config import REPO_ROOT, load_config
from ..splits import half_sample, time_ordered_bags
from .jobs import Job

# prereg names the targets in words; jobs use the panel's short names
_TARGET_SHORT = {"direction": "dir", "dir": "dir", "vol": "vol", "shuffled": "shuf", "shuf": "shuf"}


@lru_cache(maxsize=1)
def _prereg() -> dict:
    return load_config("prereg")


def model_kind(target: str, prereg: dict | None = None) -> str:
    """``"classifier"`` for dir/shuf, ``"regressor"`` for vol (``prereg.models.ebm``)."""
    m = (prereg or _prereg())["models"]["ebm"]
    if target in {_TARGET_SHORT[t] for t in m["classifier_for"]}:
        return "classifier"
    if target in {_TARGET_SHORT[t] for t in m["regressor_for"]}:
        return "regressor"
    raise ValueError(f"no EBM kind configured for target {target!r}")


def _model_api():
    """``(fit_ebm, export_fit)`` from ``ebmregime.models.ebm`` (lazy import)."""
    from ..models.ebm import export_fit, fit_ebm

    return fit_ebm, export_fit


def _surrogate_panel(symbol: str, index: int) -> pd.DataFrame:
    from ..stability.surrogate import surrogate_panel

    return surrogate_panel(symbol, index)


def job_panel(job: Job) -> pd.DataFrame:
    """The real panel, or surrogate ``i`` of the symbol for ``data="surrogate:<i>"``."""
    if job.data == "real":
        return P.load_panel(job.symbol)
    return _surrogate_panel(job.symbol, job.surrogate_index)


def fit_inputs(job: Job, panel: pd.DataFrame, prereg: dict | None = None) -> dict:
    """Rows and bags of one fit (step 2 of the spec).

    Returns ``rows`` (int64 positions into ``panel`` of the rows given to ``fit_ebm``, in time
    order), ``bags`` (int8 ``(len(rows), outer_bags)``), ``train_rows`` (positions of the full
    window's train rows) and ``val_start``.
    """
    prereg = prereg or _prereg()
    v = prereg["validation"]
    h = int(job.horizon)
    tr = P.train_rows(panel, job.tau, job.target, h, int(prereg["windows"]["train_months"]))
    pos = np.flatnonzero(tr).astype(np.int64)
    if len(pos) == 0:
        raise ValueError(f"no training rows for {job}")
    ts_tr = panel.index[pos]
    common = dict(n_bags=int(v["outer_bags"]), tail_frac=float(v["tail_frac"]),
                  block_hours=int(v["block_hours"]), bag_train_frac=float(v["bag_train_frac"]))
    if job.protocol == "half":
        mask, vs = half_sample(ts_tr, h, float(v["tail_frac"]), int(v["block_hours"]), int(job.sample_seed))
        rows = pos[mask]
        bags = time_ordered_bags(ts_tr[mask], h, seed=int(job.seed), val_start=vs, block_origin=ts_tr[0],
                                 **common)
    elif job.protocol == "seed":
        rows = pos
        bags = time_ordered_bags(ts_tr, h, seed=int(job.seed), **common)
        vs = None
    else:  # pragma: no cover - Job validates the protocol
        raise ValueError(job.protocol)
    return {"rows": rows, "bags": bags, "train_rows": pos, "val_start": vs}


@lru_cache(maxsize=1)
def _git_info() -> dict:
    """HEAD SHA and dirty flag of the repository (informational; not part of the key)."""
    def git(*args):
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return r.stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except Exception:
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def provenance() -> dict:
    """Provenance fields stored in every fit's meta (not part of the key)."""
    import interpret

    return {
        **_git_info(),
        "interpret_version": str(interpret.__version__),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(idx).as_unit("ns").asi8, dtype=np.int64)


def run_fit_job(job: Job) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one EBM and return ``(arrays, meta)`` (M2_SPEC §5 steps 1-5).

    arrays: ``export_fit(model)`` + ``test_ts`` (int64 ns), ``test_pred`` (float32: probability of
    class 1 for classifiers, prediction for regressors), ``test_y`` (float32), ``best_iteration``,
    ``bag_counts``. meta: job fields, kind, fit_info scalars, ``y_train_mean`` (over the fitted rows),
    ``n_test``, window details, provenance (git, interpret version, host, UTC time).
    """
    if job.model != "ebm" or job.arm != "primary":
        raise NotImplementedError(f"run_fit_job handles model='ebm', arm='primary' only, got {job.model}/{job.arm}")
    prereg = _prereg()
    fit_ebm, export_fit = _model_api()

    panel = job_panel(job)                                                   # 1
    inp = fit_inputs(job, panel, prereg)                                     # 2
    rows, bags = inp["rows"], inp["bags"]
    names = P.feature_names()
    ycol = P.target_column(job.target, job.horizon)
    X = panel[names].iloc[rows]
    y = panel[ycol].to_numpy()[rows]

    kind = model_kind(job.target, prereg)                                    # 3
    if kind == "classifier":
        # panel labels are float64 0.0/1.0 (NaN-able); interpret would turn float labels into the
        # string classes '0.0'/'1.0', so pass int64 0/1 (usable rows have no NaN target)
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError(f"{ycol}: classifier labels must be 0/1 on the train rows")
        y = y.astype(np.int64)
    model, fit_info = fit_ebm(X, y, bags, kind, int(job.seed))

    fitted = (bags != 0).any(axis=1)                                         # rows fit_ebm keeps (D-45)
    if "n_rows_fit" in fit_info and int(fit_info["n_rows_fit"]) != int(fitted.sum()):
        raise AssertionError(f"fit_ebm fitted {fit_info['n_rows_fit']} rows, expected {int(fitted.sum())}")

    te = np.flatnonzero(P.test_rows(panel, job.tau, job.target, job.horizon))   # 4
    X_te = panel[names].iloc[te]
    if len(te) == 0:
        pred = np.zeros(0)
    elif kind == "classifier":
        pred = model.predict_proba(X_te)[:, 1]
    else:
        pred = model.predict(X_te)

    arrays = dict(export_fit(model))
    arrays["test_ts"] = _ns(panel.index[te])
    arrays["test_pred"] = np.asarray(pred, dtype=np.float32)
    arrays["test_y"] = np.asarray(panel[ycol].to_numpy()[te], dtype=np.float32)
    if "best_iteration" in fit_info:
        arrays["best_iteration"] = np.asarray(fit_info["best_iteration"], dtype=np.int32)
    if "bag_counts" in fit_info:
        arrays["bag_counts"] = np.asarray(fit_info["bag_counts"], dtype=np.int32)

    scalars = {k: v for k, v in fit_info.items()
               if np.ndim(v) == 0 and isinstance(v, (int, float, bool, str, np.generic))}
    ts_rows = panel.index[rows]
    meta = {                                                                 # 5
        "job": job.as_dict(),
        "kind": kind,
        "fit_info": scalars,
        "y_train_mean": float(np.mean(y[fitted])),
        "n_test": int(len(te)),
        "n_train_rows_window": int(len(inp["train_rows"])),
        "n_rows_passed": int(len(rows)),
        "first_row_ts": ts_rows[0].isoformat(),
        "last_row_ts": ts_rows[-1].isoformat(),
        "val_start": None if inp["val_start"] is None else pd.Timestamp(inp["val_start"]).isoformat(),
        **provenance(),
    }
    return arrays, meta


__all__ = ["fit_inputs", "job_panel", "model_kind", "provenance", "run_fit_job"]
