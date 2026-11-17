"""Primary EBM arm: construction, fitting, export and export-based evaluation (M2_SPEC §1).

* ``make_ebm``: interpret 0.7.8 defaults with the overrides of ``prereg.models.ebm.overrides``
  (``interactions=0``, ``n_jobs=1``), an explicit ``random_state`` and the panel's feature names and
  types (D-11). The interpret version and every default the study relies on are checked against
  ``prereg.models.ebm`` at construction.
* ``fit_ebm``: fits on the 16 panel features with time-ordered ``bags`` of shape
  ``(n_rows, outer_bags)`` (D-13), after dropping the rows that are 0 in every bag (D-45: their
  labels would otherwise move the intercept).
* ``export_fit``: the stored fit as plain numpy arrays (nothing is pickled): intercept, float32 term
  scores, float64 continuous cuts, int64 nominal categories.
* ``eval_terms_from_export`` / ``predict_from_export``: the D-46 lookup on an export, reproducing
  ``eval_terms`` / ``predict_proba`` / ``predict`` without the model object.

interpret 0.7.8 facts used here (pinned in ``tests/test_models_ebm.py`` and
``tests/test_interpret_api.py``):

* continuous term ``j``: ``bins_[j][0]`` is the float64 cut array; scores are
  ``[missing, len(cuts) + 1 regular bins, unseen]`` and the score index is
  ``np.digitize(x, cuts) + 1`` (``right=False``), NaN -> 0; +-inf fall into the end regular bins;
* nominal term ``j`` (``feature_types="nominal"`` on an int column): ``bins_[j][0]`` is a ``dict``
  mapping the category **as a string** (``str(int)``, e.g. ``"7"``) to its score index ``1..n_cats``,
  ordered numerically; scores are ``[missing, n_cats levels, unseen]``. At prediction time an int
  or an integral float (``7.0``) both find level ``"7"``; NaN -> 0; any other value (an unseen
  level, a non-integral float) -> the last (unseen) index, whose score is 0 after training;
* ``intercept_`` is a length-1 array for the classifier but a Python float for the regressor;
* ``best_iteration_`` is an int64 array of shape ``(1, outer_bags)`` (stage 0 = main effects).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import load_config
from ..panel import feature_names, feature_types

KINDS = ("classifier", "regressor")
_TARGET_ALIASES = {"dir": "direction", "shuf": "shuffled", "vol": "vol"}


# --------------------------------------------------------------------------- configuration


def _ebm_cfg(cfg_models: dict | None) -> dict:
    """The ``models.ebm`` block; ``cfg_models`` is the ``models`` block of ``configs/prereg.yaml``."""
    models = load_config("prereg")["models"] if cfg_models is None else cfg_models
    return models["ebm"]


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _estimator_class(kind: str):
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    _check_kind(kind)
    return ExplainableBoostingClassifier if kind == "classifier" else ExplainableBoostingRegressor


def kind_for_target(target: str, cfg_models: dict | None = None) -> str:
    """``"classifier"`` for the direction and shuffled targets, ``"regressor"`` for vol.

    ``target`` is a panel target code (``dir``, ``vol``, ``shuf``) or a prereg name
    (``direction``, ``vol``, ``shuffled``); the mapping is read from ``prereg.models.ebm``.
    """
    cfg = _ebm_cfg(cfg_models)
    name = _TARGET_ALIASES.get(target, target)
    if name in cfg["classifier_for"]:
        return "classifier"
    if name in cfg["regressor_for"]:
        return "regressor"
    raise ValueError(f"no model kind configured for target {target!r}")


def _expected_default(value, kind: str):
    return value[kind] if isinstance(value, dict) else value


# --------------------------------------------------------------------------- construction


def make_ebm(kind: str, random_state: int, cfg_models: dict | None = None, *,
             outer_bags: int | None = None):
    """Unfitted primary EBM (D-11).

    ``outer_bags`` is a **test-only** override (small synthetic fits); ``None`` (the default, and the
    only value production code may use) keeps the pre-registered interpret default of 14. Every other
    pinned default is checked whatever ``outer_bags`` is.

    Raises ``RuntimeError`` if the installed interpret version or any default in
    ``prereg.models.ebm.defaults_relied_on`` differs from the pre-registration.
    """
    import interpret

    cfg = _ebm_cfg(cfg_models)
    if interpret.__version__ != str(cfg["interpret_version"]):
        raise RuntimeError(f"interpret {interpret.__version__} installed, prereg pins "
                           f"{cfg['interpret_version']}")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError(f"random_state must be an int, got {random_state!r}")
    cls = _estimator_class(kind)
    overrides = cfg["overrides"]
    params = dict(
        interactions=overrides["interactions"],
        n_jobs=overrides["n_jobs"],
        random_state=int(random_state),
        feature_names=feature_names(),
        feature_types=feature_types(),
    )
    if outer_bags is not None:
        params["outer_bags"] = int(outer_bags)
    est = cls(**params)

    # the pre-registered values must be interpret's defaults, and the estimator must use them
    default_params = cls().get_params()
    got = est.get_params()
    for key, value in cfg["defaults_relied_on"].items():
        want = _expected_default(value, kind)
        if key not in default_params or default_params[key] != want:
            raise RuntimeError(f"interpret default {key}={default_params.get(key)!r} for {kind}, "
                               f"prereg relies on {want!r}")
        if key == "outer_bags" and outer_bags is not None:
            continue
        if got[key] != want:
            raise RuntimeError(f"{kind} param {key}={got[key]!r}, prereg requires {want!r}")
    for key in ("interactions", "n_jobs", "random_state", "feature_names", "feature_types"):
        if got[key] != params[key]:
            raise RuntimeError(f"{kind} param {key}={got[key]!r}, expected {params[key]!r}")
    return est


# --------------------------------------------------------------------------- fitting


def _check_columns(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with the panel feature columns")
    names = feature_names()
    if list(X.columns) != names:
        raise ValueError(f"X columns must be the {len(names)} panel features in frozen order; "
                         f"got {list(X.columns)}")


def fit_ebm(X: pd.DataFrame, y: np.ndarray, bags: np.ndarray, kind: str, random_state: int,
            cfg_models: dict | None = None, *, outer_bags: int | None = None):
    """Fit the primary EBM; returns ``(model, fit_info)``.

    ``bags``: integer array of shape ``(n_rows, outer_bags)`` (values -1 validation, 0 out of bag,
    +1 or a replication count in-bag; must fit int8). Rows that are 0 in every bag are dropped before
    ``fit`` (D-45). ``outer_bags`` is the test-only override of ``make_ebm``.

    ``fit_info``: ``n_rows_in``, ``n_rows_dropped``, ``n_rows_fit``, ``fit_mask`` (bool
    ``(n_rows_in,)``, the rows passed to ``fit``), ``bag_counts`` (int32 ``(outer_bags, 3)``: rows with
    value -1, 0 and > 0 per bag among the fitted rows), ``best_iteration`` (int32, shape of
    ``best_iteration_``, ``(1, outer_bags)``), ``fit_seconds``.
    """
    cfg = _ebm_cfg(cfg_models)
    if cfg.get("drop_rows_zero_in_every_bag") is not True:
        raise RuntimeError("prereg.models.ebm.drop_rows_zero_in_every_bag must be true (D-45)")
    _check_columns(X)
    model = make_ebm(kind, random_state, cfg_models, outer_bags=outer_bags)

    n = len(X)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError(f"y must be 1-d with {n} rows, got shape {y.shape}")
    bags = np.asarray(bags)
    if bags.ndim != 2 or bags.shape != (n, model.outer_bags):
        raise ValueError(f"bags must have shape ({n}, {model.outer_bags}), got {bags.shape}")
    if not np.issubdtype(bags.dtype, np.integer):
        raise TypeError(f"bags must be an integer array, got {bags.dtype}")
    if bags.min() < np.iinfo(np.int8).min or bags.max() > np.iinfo(np.int8).max:
        raise ValueError("bags values must fit int8")
    bags = bags.astype(np.int8, copy=False)

    keep = ~(bags == 0).all(axis=1)                               # D-45
    Xf, yf, bf = X.iloc[keep], y[keep], np.ascontiguousarray(bags[keep])
    if not ((bf > 0).any(axis=0).all() and (bf < 0).any(axis=0).all()):
        raise ValueError("every bag needs in-bag (> 0) and validation (-1) rows (D-13)")
    if kind == "classifier" and not np.isin(yf, (0, 1)).all():
        raise ValueError("classifier labels must be 0/1")
    if kind == "regressor" and not np.isfinite(yf.astype(np.float64)).all():
        raise ValueError("regressor targets must be finite")

    t0 = time.perf_counter()
    model.fit(Xf, yf, bags=bf)
    fit_seconds = time.perf_counter() - t0

    if kind == "classifier" and (len(model.classes_) != 2 or model.classes_[1] != 1):
        raise ValueError(f"expected classes [0, 1], got {model.classes_}")
    bag_counts = np.stack([(bf < 0).sum(axis=0), (bf == 0).sum(axis=0), (bf > 0).sum(axis=0)],
                          axis=1).astype(np.int32)
    fit_info = {
        "n_rows_in": int(n),
        "n_rows_dropped": int(n - keep.sum()),
        "n_rows_fit": int(keep.sum()),
        "fit_mask": keep,
        "bag_counts": bag_counts,
        "best_iteration": np.asarray(model.best_iteration_).astype(np.int32),
        "fit_seconds": float(fit_seconds),
    }
    return model, fit_info


# --------------------------------------------------------------------------- export


def _nominal_categories(mapping: dict, j: int) -> np.ndarray:
    """int64 categories in score-index order (index 1..n_cats) from interpret's str -> index dict."""
    n_cats = len(mapping)
    idx = np.array(list(mapping.values()), dtype=np.int64)
    if sorted(idx.tolist()) != list(range(1, n_cats + 1)):
        raise ValueError(f"feature {j}: nominal indices are not 1..{n_cats}: {mapping}")
    cats = np.empty(n_cats, dtype=np.int64)
    for key, i in mapping.items():
        try:
            v = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"feature {j}: nominal category {key!r} is not an integer") from None
        if str(v) != str(key):
            raise ValueError(f"feature {j}: nominal category {key!r} is not a canonical integer")
        cats[i - 1] = v
    return cats


def export_fit(model) -> dict[str, np.ndarray]:
    """The stored fit as numpy arrays (no objects; loadable with ``allow_pickle=False``)."""
    n_feat = len(model.feature_names_in_)
    term_features = [tuple(t) for t in model.term_features_]
    if term_features != [(j,) for j in range(n_feat)]:
        raise ValueError(f"expected main effects only, in feature order; got {term_features}")
    out: dict[str, np.ndarray] = {
        "intercept": np.asarray(model.intercept_, dtype=np.float64).reshape(1),
        "term_features": np.arange(n_feat, dtype=np.int32),
    }
    for j in range(n_feat):
        scores = np.asarray(model.term_scores_[j])
        if scores.ndim != 1:
            raise ValueError(f"feature {j}: expected 1-d term scores, got shape {scores.shape}")
        ftype = model.feature_types_in_[j]
        bins = model.bins_[j][0]
        if ftype == "continuous":
            cuts = np.asarray(bins, dtype=np.float64)
            if len(scores) != len(cuts) + 3:
                raise ValueError(f"feature {j}: {len(scores)} scores for {len(cuts)} cuts")
            out[f"cuts_{j}"] = cuts
        elif ftype == "nominal":
            cats = _nominal_categories(bins, j)
            if len(scores) != len(cats) + 2:
                raise ValueError(f"feature {j}: {len(scores)} scores for {len(cats)} categories")
            out[f"cats_{j}"] = cats
        else:
            raise ValueError(f"feature {j}: unsupported feature type {ftype!r}")
        out[f"scores_{j}"] = scores.astype(np.float32)
    return out


# --------------------------------------------------------------------------- evaluation


def _score_index_continuous(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, cuts, right=False) + 1
    idx[np.isnan(x)] = 0
    return idx


def _score_index_nominal(x: np.ndarray, cats: np.ndarray, n_scores: int) -> np.ndarray:
    order = np.argsort(cats, kind="stable")
    levels = cats[order].astype(np.float64)
    pos = np.minimum(np.searchsorted(levels, x), len(levels) - 1)
    hit = levels[pos] == x                                       # False for NaN and unknown values
    idx = np.where(hit, order[pos] + 1, n_scores - 1)            # unknown -> unseen (last)
    idx[np.isnan(x)] = 0
    return idx


def eval_terms_from_export(export: dict, X: pd.DataFrame) -> np.ndarray:
    """Per-term contributions ``(n, J)`` float64 from an export (D-46 lookup).

    Continuous: score index ``np.digitize(x, cuts) + 1`` (``right=False``), NaN -> 0. Nominal:
    category -> its index, NaN -> 0, any other value -> the last (unseen) index.
    """
    _check_columns(X)
    types = feature_types()
    n_feat = len(export["term_features"])
    if n_feat != len(types) or not np.array_equal(export["term_features"], np.arange(n_feat)):
        raise ValueError("export term_features must be 0..J-1 for the panel features")
    out = np.empty((len(X), n_feat), dtype=np.float64)
    for j in range(n_feat):
        scores = np.asarray(export[f"scores_{j}"])
        x = X.iloc[:, j].to_numpy(dtype=np.float64, na_value=np.nan)
        if types[j] == "nominal":
            if f"cats_{j}" not in export:
                raise ValueError(f"feature {j} is nominal but the export has no cats_{j}")
            idx = _score_index_nominal(x, np.asarray(export[f"cats_{j}"]), len(scores))
        else:
            if f"cuts_{j}" not in export:
                raise ValueError(f"feature {j} is continuous but the export has no cuts_{j}")
            idx = _score_index_continuous(x, np.asarray(export[f"cuts_{j}"], dtype=np.float64))
        out[:, j] = scores[idx]
    return out


def predict_from_export(export: dict, X: pd.DataFrame, kind: str) -> np.ndarray:
    """eta = intercept + row sum of terms; classifier -> P(y = 1) = 1 / (1 + exp(-eta)), regressor -> eta."""
    _check_kind(kind)
    eta = float(np.asarray(export["intercept"]).reshape(-1)[0]) + eval_terms_from_export(export, X).sum(axis=1)
    return expit(eta) if kind == "classifier" else eta
