"""Primary EBM wrapper (M2_SPEC §1; D-11, D-13, D-45, D-46). Synthetic data only.

n = 1500 hourly rows with the 16 panel features (two heavily tied continuous features; ``hour`` and
``dow`` int64 nominal), time-ordered bags from ``splits.time_ordered_bags``. Fits use the test-only
``outer_bags`` override (3 bags): none of the properties tested depends on the bag count, and the
pre-registered 14 is checked on the unfitted estimator.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config
from ebmregime.models import (
    eval_terms_from_export,
    export_fit,
    fit_ebm,
    kind_for_target,
    make_ebm,
    predict_from_export,
)
from ebmregime.panel import feature_names, feature_types
from ebmregime.splits import time_ordered_bags

N, H, OUTER, SEED = 1500, 8, 3, 20261108
PREREG = load_config("prereg")
EBM_CFG = PREREG["models"]["ebm"]
VAL = {k: PREREG["validation"][k] for k in ("tail_frac", "block_hours", "bag_train_frac")}
NAMES = feature_names()
TYPES = feature_types()
CONT = [j for j, t in enumerate(TYPES) if t == "continuous"]
NOM = [j for j, t in enumerate(TYPES) if t == "nominal"]
TIED = ["rv_level_pct", "pos_range_24h"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=N, freq="h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(N, len(NAMES))), columns=NAMES)
    X["rv_level_pct"] = np.round(rng.random(N), 1)              # 11 tied levels
    X["pos_range_24h"] = np.round(rng.random(N) * 4) / 4        # 5 tied levels
    X["hour"] = np.asarray(ts.hour, dtype=np.int64)
    X["dow"] = np.asarray(ts.day_of_week, dtype=np.int64)
    logit = (0.6 * X["ret_1h"] - 0.4 * np.sin(2 * X["ret_4h"]) + 0.5 * np.sin(2 * np.pi * X["hour"] / 24)
             + 0.5 * X["rv_level_pct"]).to_numpy()
    y_clf = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    y_reg = (0.5 * X["ret_1h"] + np.cos(X["dow"]) + 0.3 * X["pos_range_24h"]).to_numpy() + rng.normal(size=N)
    bags = time_ordered_bags(ts, H, OUTER, seed=1, **VAL)
    return {"X": X, "ts": ts, "bags": bags, "classifier": y_clf, "regressor": y_reg}


@pytest.fixture(scope="module", params=["classifier", "regressor"])
def fitted(request, data):
    kind = request.param
    model, info = fit_ebm(data["X"], data[kind], data["bags"], kind, SEED, outer_bags=OUTER)
    return kind, model, info


def _intercept(model) -> float:
    return float(np.asarray(model.intercept_).reshape(-1)[0])


def _max_term_diff(a, b) -> float:
    return max(float(np.abs(a.term_scores_[j] - b.term_scores_[j]).max()) for j in range(len(NAMES)))


# --------------------------------------------------------------------------- make_ebm


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
def test_make_ebm_params_match_prereg(kind):
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

    est = make_ebm(kind, 12345)
    cls = ExplainableBoostingClassifier if kind == "classifier" else ExplainableBoostingRegressor
    assert type(est) is cls
    p = est.get_params()
    assert p["outer_bags"] == 14 == PREREG["validation"]["outer_bags"]
    assert p["interactions"] == 0 == EBM_CFG["overrides"]["interactions"]
    assert p["n_jobs"] == 1 == EBM_CFG["overrides"]["n_jobs"]
    assert p["random_state"] == 12345
    assert p["feature_names"] == NAMES and len(NAMES) == 16
    assert p["feature_types"] == TYPES
    assert [NAMES[j] for j in NOM] == ["hour", "dow"]
    for key, value in EBM_CFG["defaults_relied_on"].items():
        assert p[key] == (value[kind] if isinstance(value, dict) else value), key
    # everything not overridden is interpret's default
    default = cls().get_params()
    overridden = {"interactions", "n_jobs", "random_state", "feature_names", "feature_types"}
    assert {k: v for k, v in p.items() if k not in overridden} == \
        {k: v for k, v in default.items() if k not in overridden}


def test_make_ebm_checks_version_and_defaults():
    import interpret

    assert interpret.__version__ == EBM_CFG["interpret_version"]
    bad = copy.deepcopy(PREREG["models"])
    bad["ebm"]["interpret_version"] = "0.0.0"
    with pytest.raises(RuntimeError, match="interpret"):
        make_ebm("classifier", 1, bad)
    bad = copy.deepcopy(PREREG["models"])
    bad["ebm"]["defaults_relied_on"]["learning_rate"]["regressor"] = 0.01
    make_ebm("classifier", 1, bad)                               # classifier value still right
    with pytest.raises(RuntimeError, match="learning_rate"):
        make_ebm("regressor", 1, bad)
    with pytest.raises(ValueError, match="kind"):
        make_ebm("ranker", 1)
    with pytest.raises(TypeError):
        make_ebm("classifier", None)


def test_make_ebm_outer_bags_override_is_test_only_and_explicit():
    est = make_ebm("classifier", 1, outer_bags=OUTER)
    assert est.outer_bags == OUTER
    assert est.max_bins == EBM_CFG["defaults_relied_on"]["max_bins"]
    assert make_ebm("classifier", 1).outer_bags == 14


def test_kind_for_target():
    assert kind_for_target("dir") == kind_for_target("shuf") == "classifier"
    assert kind_for_target("direction") == kind_for_target("shuffled") == "classifier"
    assert kind_for_target("vol") == "regressor"
    with pytest.raises(ValueError):
        kind_for_target("price")


# --------------------------------------------------------------------------- fit_ebm


def test_fit_info(fitted, data):
    kind, model, info = fitted
    bags = data["bags"]
    purged = (bags == 0).all(axis=1)
    assert purged.sum() == H                                      # the h-row purge on a gap-free grid
    assert info["n_rows_in"] == N
    assert info["n_rows_dropped"] == H
    assert info["n_rows_fit"] == N - H
    assert np.array_equal(info["fit_mask"], ~purged)
    bc = info["bag_counts"]
    assert bc.dtype == np.int32 and bc.shape == (OUTER, 3)
    assert np.all(bc.sum(axis=1) == N - H)
    kept = bags[~purged]
    assert np.array_equal(bc[:, 0], (kept == -1).sum(axis=0))
    assert np.array_equal(bc[:, 1], (kept == 0).sum(axis=0))
    assert np.array_equal(bc[:, 2], (kept == 1).sum(axis=0))
    bi = info["best_iteration"]
    assert bi.dtype == np.int32 and bi.shape == (1, OUTER)       # [stage, bag]; stage 0 = mains
    assert np.array_equal(bi, np.asarray(model.best_iteration_))
    assert np.all(bi > 0) and np.all(bi < model.max_rounds)
    assert info["fit_seconds"] > 0
    assert model.outer_bags == OUTER and model.interactions == 0


def test_d45_rows_zero_in_every_bag_are_dropped(fitted, data):
    """Labels of rows that are 0 in every bag must not reach the fit (they would move the intercept)."""
    kind, model, _ = fitted
    X, y, bags = data["X"], data[kind], data["bags"]
    purged = (bags == 0).all(axis=1)
    keep = ~purged
    y_alt = np.where(purged, 1 - y, y) if kind == "classifier" else np.where(purged, y + 50.0, y)

    # fit_ebm equals a plain fit on the pre-dropped rows, bit for bit
    pre = make_ebm(kind, SEED, outer_bags=OUTER).fit(X[keep], y[keep], bags=bags[keep])
    assert _intercept(model) == _intercept(pre)
    assert _max_term_diff(model, pre) == 0.0

    # so the purged rows' labels have no effect on fit_ebm ...
    alt, _ = fit_ebm(X, y_alt, bags, kind, SEED, outer_bags=OUTER)
    assert _intercept(alt) == _intercept(model)
    assert _max_term_diff(alt, model) == 0.0

    # ... whereas when they are passed to interpret, their labels move the intercept (shapes unchanged)
    leak = make_ebm(kind, SEED, outer_bags=OUTER).fit(X, y, bags=bags)
    leak_alt = make_ebm(kind, SEED, outer_bags=OUTER).fit(X, y_alt, bags=bags)
    assert _max_term_diff(leak, leak_alt) == 0.0
    assert abs(_intercept(leak) - _intercept(leak_alt)) > 1e-4
    assert _intercept(leak) != _intercept(model)                 # and the fit differs from fit_ebm's


def test_fit_ebm_input_validation(data):
    X, y, bags = data["X"], data["classifier"], data["bags"]
    with pytest.raises(ValueError, match="frozen order"):
        fit_ebm(X[NAMES[::-1]], y, bags, "classifier", SEED, outer_bags=OUTER)
    with pytest.raises(TypeError):
        fit_ebm(X.to_numpy(), y, bags, "classifier", SEED, outer_bags=OUTER)
    with pytest.raises(ValueError, match="bags must have shape"):
        fit_ebm(X, y, np.ascontiguousarray(bags.T), "classifier", SEED, outer_bags=OUTER)
    with pytest.raises(ValueError, match="bags must have shape"):
        fit_ebm(X, y, bags, "classifier", SEED)                  # 3 bags but prereg outer_bags = 14
    with pytest.raises(ValueError, match="validation"):
        fit_ebm(X, y, np.abs(bags), "classifier", SEED, outer_bags=OUTER)
    with pytest.raises(ValueError, match="1-d"):
        fit_ebm(X, y[:-1], bags, "classifier", SEED, outer_bags=OUTER)
    with pytest.raises(ValueError, match="0/1"):
        fit_ebm(X, y + 1, bags, "classifier", SEED, outer_bags=OUTER)


# --------------------------------------------------------------------------- export


def test_export_contents(fitted, tmp_path):
    kind, model, _ = fitted
    exp = export_fit(model)
    expected_keys = {"intercept", "term_features"} | {f"scores_{j}" for j in range(16)} \
        | {f"cuts_{j}" for j in CONT} | {f"cats_{j}" for j in NOM}
    assert set(exp) == expected_keys
    assert exp["intercept"].dtype == np.float64 and exp["intercept"].shape == (1,)
    assert exp["intercept"][0] == _intercept(model)
    assert exp["term_features"].dtype == np.int32 and np.array_equal(exp["term_features"], np.arange(16))
    for j in range(16):
        s = exp[f"scores_{j}"]
        assert s.dtype == np.float32
        assert np.array_equal(s, model.term_scores_[j].astype(np.float32))
    for j in CONT:
        assert exp[f"cuts_{j}"].dtype == np.float64
        assert np.array_equal(exp[f"cuts_{j}"], model.bins_[j][0])
        assert len(exp[f"scores_{j}"]) == len(exp[f"cuts_{j}"]) + 3
    for name in TIED:
        assert len(exp[f"cuts_{NAMES.index(name)}"]) < 11            # ties -> few cuts
    # interpret keys nominal levels by str(int), numerically ordered; the export maps them back
    hour, dow = NOM
    assert all(isinstance(k, str) for k in model.bins_[hour][0])
    assert exp[f"cats_{hour}"].dtype == np.int64
    assert np.array_equal(exp[f"cats_{hour}"], np.arange(24))
    assert np.array_equal(exp[f"cats_{dow}"], np.arange(7))
    assert len(exp[f"scores_{hour}"]) == 24 + 2
    for j in NOM:
        for i, c in enumerate(exp[f"cats_{j}"], start=1):
            assert model.bins_[j][0][str(c)] == i
    # stored without pickling
    path = tmp_path / f"{kind}.npz"
    np.savez(path, **exp)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == expected_keys
        for k in expected_keys:
            assert np.array_equal(z[k], exp[k]) and z[k].dtype == exp[k].dtype


def _probe_frame(model, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rows with values on cuts, beyond the range, +-inf, NaN and unseen nominal levels."""
    Xe = X.iloc[:400].copy().reset_index(drop=True)
    on_cut = {}
    for j in CONT:
        cuts = model.bins_[j][0]
        k = min(len(cuts), 250)
        Xe.iloc[:k, j] = cuts[:k]
        on_cut[j] = k
    Xe.iloc[300, CONT] = 1e9
    Xe.iloc[301, CONT] = -1e9
    Xe.iloc[302, CONT] = np.inf
    Xe.iloc[303, CONT] = -np.inf
    Xe.iloc[304, CONT] = np.nan
    hour, dow = NOM
    Xe.iloc[305, hour] = 99                                       # unseen levels
    Xe.iloc[306, dow] = 7
    Xe.iloc[307, dow] = -1
    assert Xe["hour"].dtype == np.int64 and Xe["dow"].dtype == np.int64
    return Xe, on_cut


def _float32_model(model):
    m = copy.deepcopy(model)
    m.term_scores_ = [np.asarray(s).astype(np.float32).astype(np.float64) for s in model.term_scores_]
    return m


def test_eval_terms_from_export_matches_model(fitted, data):
    _, model, _ = fitted
    exp = export_fit(model)
    for frame in (data["X"], _probe_frame(model, data["X"])[0]):
        mine = eval_terms_from_export(exp, frame)
        assert mine.dtype == np.float64 and mine.shape == (len(frame), 16)
        assert np.array_equal(mine, _float32_model(model).eval_terms(frame))      # exact
        assert np.abs(mine - model.eval_terms(frame)).max() <= 1e-6


def test_eval_terms_lookup_semantics(fitted, data):
    """The D-46 conventions, checked index by index on the export."""
    _, model, _ = fitted
    exp = export_fit(model)
    Xe, on_cut = _probe_frame(model, data["X"])
    E = eval_terms_from_export(exp, Xe)
    for j in CONT:
        s, cuts = exp[f"scores_{j}"], exp[f"cuts_{j}"]
        k = on_cut[j]
        assert np.array_equal(E[:k, j], s[np.arange(k) + 2])      # value on cut i -> bin above (index i + 2)
        assert E[300, j] == E[302, j] == s[-2]                    # beyond range / +inf -> last regular bin
        assert E[301, j] == E[303, j] == s[1]                     # -> first regular bin
        assert E[304, j] == s[0] == 0.0                           # NaN -> missing
        assert len(cuts) + 3 == len(s)
    hour, dow = NOM
    assert E[305, hour] == exp[f"scores_{hour}"][-1] == 0.0       # unseen -> last, empty after training
    assert E[306, dow] == E[307, dow] == exp[f"scores_{dow}"][-1]
    i_hour = int(Xe.iloc[0, hour])
    assert E[0, hour] == exp[f"scores_{hour}"][i_hour + 1]        # level h has index h + 1 here


def test_nominal_float_and_nan_values(fitted, data):
    """Nominal columns given as floats: integral values find their level, NaN -> missing, others -> unseen."""
    _, model, _ = fitted
    exp = export_fit(model)
    Xf = data["X"].iloc[:8].copy().reset_index(drop=True)
    Xf["hour"] = np.array([0, 7, 23, np.nan, 7.5, 24, -3, 12], dtype=np.float64)
    Xf["dow"] = np.array([np.nan, 0, 6, 3, 2.5, 7, 1, 5], dtype=np.float64)
    mine = eval_terms_from_export(exp, Xf)
    assert np.array_equal(mine, _float32_model(model).eval_terms(Xf))
    hour, dow = NOM
    s_h, s_d = exp[f"scores_{hour}"], exp[f"scores_{dow}"]
    assert mine[3, hour] == 0.0 and mine[0, dow] == 0.0
    assert mine[1, hour] == s_h[8] and mine[2, dow] == s_d[7]
    Xi = Xf.iloc[1:3].copy()                                      # same levels as int64
    Xi["hour"] = np.array([7, 23], dtype=np.int64)
    Xi["dow"] = np.array([0, 6], dtype=np.int64)
    assert np.array_equal(eval_terms_from_export(exp, Xi), mine[1:3])
    # pandas nullable integers with missing values
    Xn = data["X"].iloc[:3].copy()
    Xn["hour"] = pd.array([5, None, 6], dtype="Int64")
    En = eval_terms_from_export(exp, Xn)
    assert En[1, hour] == 0.0 and En[0, hour] == s_h[6] and En[2, hour] == s_h[7]


def test_predict_from_export_matches_model(fitted, data):
    kind, model, _ = fitted
    exp = export_fit(model)
    for frame in (data["X"], _probe_frame(model, data["X"])[0]):
        p = predict_from_export(exp, frame, kind)
        ref = model.predict_proba(frame)[:, 1] if kind == "classifier" else model.predict(frame)
        assert p.shape == (len(frame),)
        assert np.abs(p - ref).max() <= 1e-6


def test_eval_rejects_wrong_columns(fitted, data):
    _, model, _ = fitted
    exp = export_fit(model)
    with pytest.raises(ValueError, match="frozen order"):
        eval_terms_from_export(exp, data["X"][NAMES[::-1]])
    with pytest.raises(ValueError, match="kind"):
        predict_from_export(exp, data["X"], "ranker")
