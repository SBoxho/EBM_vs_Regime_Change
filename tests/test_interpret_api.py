"""Pins of interpret 0.7.8 behaviour the study depends on (D-11, D-12, D-13, D-15, D-17, D-30).

Any interpret version bump must pass this file. Synthetic data only: n = 3000 hourly rows,
6 continuous features (feature 2 heavily tied: rounded to integers; feature 5 rounded to 0.5),
weak-signal binary target. Small ``outer_bags`` where the property does not depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ebmregime.splits import stationary_bootstrap_counts, time_ordered_bags

N, P, H_LABEL, OUTER = 3000, 6, 8, 4
SEED = 20261108
VAL = dict(tail_frac=0.15, block_hours=168, bag_train_frac=0.85)

slow = pytest.mark.slow  # a few seconds each; kept in the default run


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, P))
    X[:, 2] = np.round(X[:, 2])            # heavy ties
    X[:, 5] = np.round(X[:, 5] * 2) / 2
    logit = 0.4 * X[:, 0] - 0.3 * np.sin(2 * X[:, 1]) + 0.2 * X[:, 2]
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    ts = pd.date_range("2024-01-01", periods=N, freq="h", tz="UTC")
    return X, y, ts


@pytest.fixture(scope="module")
def bags(data):
    return time_ordered_bags(data[2], H_LABEL, OUTER, seed=1, **VAL)


def ebm(**kw):
    from interpret.glassbox import ExplainableBoostingClassifier

    params = dict(outer_bags=OUTER, interactions=0, random_state=SEED, n_jobs=1)
    params.update(kw)
    return ExplainableBoostingClassifier(**params)


@pytest.fixture(scope="module")
def model(data, bags):
    X, y, _ = data
    return ebm().fit(X, y, bags=bags)


def max_term_diff(a, b, n_terms=P):
    return max(float(np.abs(a.term_scores_[j] - b.term_scores_[j]).max()) for j in range(n_terms))


def test_version_pinned():
    import interpret

    assert interpret.__version__ == "0.7.8"


# 1. D-13: bags shape (n_samples, outer_bags); the transpose is rejected.
def test_bags_shape(data, bags, model):
    X, y, _ = data
    assert bags.shape == (N, OUTER)
    assert len(model.term_scores_) == P
    with pytest.raises(ValueError, match=f"bags has {N} bags and self.outer_bags is {OUTER} bags"):
        ebm().fit(X, y, bags=np.ascontiguousarray(bags.T))


# 2. D-12: main effects are fitted first and frozen; pairs do not change them.
@slow
def test_mains_identical_with_interactions(data, bags, model):
    X, y, _ = data
    m3 = ebm(interactions=3).fit(X, y, bags=bags)
    assert len(m3.term_features_) == P + 3
    for j in range(P):
        assert np.array_equal(model.term_scores_[j], m3.term_scores_[j])
    # only the intercept moves, slightly
    assert abs(float(model.intercept_[0] - m3.intercept_[0])) < 1e-2


# 3. D-15: each main term is centred under its own bin weights, and those weights count
#    every row passed to fit (validation and purged / unbagged rows included).
def test_terms_centred_under_own_bin_weights(model):
    for j in range(P):
        w, s = model.bin_weights_[j], model.term_scores_[j]
        assert abs(float((w * s).sum() / w.sum())) < 1e-10
        assert w.sum() == pytest.approx(N)


# 4. D-15: layout of a continuous term: [missing, cuts+1 regular bins, unseen].
def test_term_layout(model):
    for j in range(P):
        assert model.feature_types_in_[j] == "continuous"
        cuts = model.bins_[j][0]
        s = model.term_scores_[j]
        assert len(s) == len(cuts) + 3
        assert s[0] == 0.0 and s[-1] == 0.0                  # no NaN in the data, unseen bin empty
        assert model.bin_weights_[j][0] == 0 and model.bin_weights_[j][-1] == 0
    assert len(model.bins_[2][0]) < 15                          # the tied feature has few cuts


# 5. Lookup convention: score index = np.digitize(x, cuts) + 1 (right=False), i.e. bins are
#    [c_{k-1}, c_k) and a value exactly on a cut goes to the bin ABOVE it. NaN -> index 0;
#    values beyond the data range fall in the end regular bins, never the "unseen" slot.
def test_eval_terms_lookup_convention(data, model):
    X = data[0]
    Xe = X[:200].copy()
    for j in range(P):
        cuts = model.bins_[j][0]
        k = min(len(cuts), 100)
        Xe[:k, j] = cuts[:k]                                    # values exactly on cut points
    Xe[150, :] = 1e9
    Xe[151, :] = -1e9
    E = model.eval_terms(Xe)
    for j in range(P):
        cuts, s = model.bins_[j][0], model.term_scores_[j]
        ok = s[np.digitize(Xe[:, j], cuts) + 1]
        assert np.array_equal(E[:, j], ok)
        alt = s[np.digitize(Xe[:, j], cuts, right=True) + 1]
        assert not np.array_equal(E[:, j], alt)                # the test discriminates the two
        assert E[150, j] == s[-2] and E[151, j] == s[1]
        # and on the training data itself
        assert np.array_equal(model.eval_terms(X)[:, j], s[np.digitize(X[:, j], cuts) + 1])
    Xn = X[:3].copy()
    Xn[0, :] = np.nan
    assert np.all(model.eval_terms(Xn)[0] == 0.0)


# 6. D-11: same seed and bags -> bit-identical fits for n_jobs = 1 and 2.
@slow
def test_deterministic_across_n_jobs(data, bags, model):
    X, y, _ = data
    m2 = ebm(n_jobs=2).fit(X, y, bags=bags)
    for j in range(P):
        assert np.array_equal(model.term_scores_[j], m2.term_scores_[j])
    assert np.array_equal(model.intercept_, m2.intercept_)


# 7. The model is exactly additive on the logit scale.
def test_predict_proba_is_sigmoid_of_terms(data, model):
    X = data[0]
    p = model.predict_proba(X)[:, 1]
    q = 1 / (1 + np.exp(-(model.eval_terms(X).sum(axis=1) + model.intercept_[0])))
    assert np.abs(p - q).max() < 1e-12


# 8. D-17 protocol (ii): replication counts > 1 in bags are accepted and change the fit
#    relative to the 0/1 bag with the same support. Observed on 0.7.8: they act like integer
#    sample weights (close to, but not bit-identical with, the same counts given as
#    sample_weight; max |diff| ~1e-3 with min_samples_leaf=1, ~2e-2 with the default 4).
@slow
def test_bag_replication_counts(data):
    X, y, ts = data
    counts = np.column_stack(
        [stationary_bootstrap_counts(ts, H_LABEL, 0.15, 24, seed=s) for s in range(OUTER)]
    )
    assert counts.max() > 1
    binary = np.where(counts < 0, -1, (counts > 0).astype(np.int8)).astype(np.int8)
    mc = ebm().fit(X, y, bags=counts)
    mb = ebm().fit(X, y, bags=binary)
    assert max_term_diff(mc, mb) > 1e-3
    # weighting interpretation: one bag with counts ~ the 0/1 bag plus sample_weight = counts
    c0 = counts[:, :1]
    w = np.where(c0[:, 0] > 0, c0[:, 0], 1).astype(float)
    m_c = ebm(outer_bags=1, min_samples_leaf=1).fit(X, y, bags=c0)
    m_w = ebm(outer_bags=1, min_samples_leaf=1).fit(X, y, bags=binary[:, :1], sample_weight=w)
    m_b = ebm(outer_bags=1, min_samples_leaf=1).fit(X, y, bags=binary[:, :1])
    assert max_term_diff(m_c, m_w) < 0.1 * max_term_diff(m_c, m_b)
    # bag values must fit in int8
    with pytest.raises(ValueError, match="outside the valid range"):
        ebm(outer_bags=1).fit(X, y, bags=np.where(c0 > 0, 200, c0.astype(np.int16)))


# 9. D-13: with a -1 tail, early stopping is active and stops well before max_rounds.
def test_early_stopping_with_time_ordered_bags(model):
    bi = np.asarray(model.best_iteration_)
    assert bi.shape == (1, OUTER)                               # [stage, bag]; stage 0 = mains
    assert np.all(bi > 0) and np.all(bi < model.max_rounds)


@slow
def test_labels_of_rows_excluded_from_every_bag(data, bags):
    """Added by the lead (verification of the purge, D-45).

    Rows that are 0 in EVERY bag (the purge before the validation tail) must not influence a fit.
    In interpret 0.7.8 the shape functions ignore their labels, but the intercept does NOT: it
    uses the labels of every row passed to ``fit``, for the classifier and the regressor alike.
    The model wrapper must therefore drop rows that are 0 in every bag before calling ``fit``.
    """
    from interpret.glassbox import ExplainableBoostingRegressor

    X, y, _ = data
    purged = (bags == 0).all(axis=1)
    assert purged.sum() == H_LABEL  # an h-row purge on a gap-free grid
    clf_a = ebm().fit(X, y, bags=bags)
    clf_b = ebm().fit(X, np.where(purged, 1 - y, y), bags=bags)
    assert max_term_diff(clf_a, clf_b) == 0.0
    assert abs(float(clf_a.intercept_[0]) - float(clf_b.intercept_[0])) > 1e-4   # leak via intercept
    keep0 = ~purged
    clf_c = ebm().fit(X[keep0], y[keep0], bags=bags[keep0])
    clf_d = ebm().fit(X[keep0], np.where(purged, 1 - y, y)[keep0], bags=bags[keep0])
    assert float(clf_c.intercept_[0]) == float(clf_d.intercept_[0]) and max_term_diff(clf_c, clf_d) == 0.0

    yr = X[:, 0] + np.random.default_rng(1).normal(size=len(y))
    reg = dict(outer_bags=OUTER, interactions=0, random_state=SEED, n_jobs=1)
    reg_a = ExplainableBoostingRegressor(**reg).fit(X, yr, bags=bags)
    reg_b = ExplainableBoostingRegressor(**reg).fit(X, np.where(purged, yr + 50.0, yr), bags=bags)
    assert max_term_diff(reg_a, reg_b) == 0.0
    assert abs(float(reg_a.intercept_) - float(reg_b.intercept_)) > 1e-3   # the leak being pinned
    # The remedy: the same fit without the purged rows equals the fit with them, intercept aside.
    keep = ~purged
    reg_c = ExplainableBoostingRegressor(**reg).fit(X[keep], yr[keep], bags=bags[keep])
    reg_d = ExplainableBoostingRegressor(**reg).fit(X[keep], np.where(purged, yr + 50.0, yr)[keep], bags=bags[keep])
    assert float(reg_c.intercept_) == float(reg_d.intercept_)
