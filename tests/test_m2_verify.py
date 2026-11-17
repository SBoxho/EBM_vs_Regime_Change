"""Lead's independent verification of the M2 explain + stability chain (synthetic, known answers).

Checks the implementers' code end to end (hand-built EBM exports -> D-46 lookup -> pooled cells ->
re-centred shapes -> metrics) against analytic answers and against the naive reference
implementation in ``tests/m2_reference.py``, which was written from PLAN §6 without reading the
implementers' code. The stationary-process check with real EBM fits (DR ~ 1) is the script
``scripts/verify_floor_synthetic.py`` (minutes of compute, so not in the default suite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import m2_reference as REF
from ebmregime import panel as P
from ebmregime.explain import cell_shapes, cell_shapes_many, make_cells
from ebmregime.models.ebm import eval_terms_from_export
from ebmregime.stability import floor as F
from ebmregime.stability import metrics as M

NAMES = P.feature_names()
NOMINAL = set(P.nominal_features())
N = 6000


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(11)
    d = {}
    for j, n in enumerate(NAMES):
        if n == "hour":
            d[n] = rng.integers(0, 24, N)
        elif n == "dow":
            d[n] = rng.integers(0, 7, N)
        elif j == 6:
            d[n] = np.round(rng.random(N), 2)          # heavy ties, like rv_level_pct
        else:
            d[n] = rng.standard_normal(N) * (1 + j)
    return pd.DataFrame(d)[NAMES]


def export_from_functions(funcs: dict, cuts_by_feature: dict) -> dict:
    """Hand-built export: continuous feature j is a step function with the given cuts, scores from funcs[j]
    evaluated at each bin's midpoint (end bins at cut -/+ 1); nominal features score = funcs[j](level)."""
    ex = {"intercept": np.array([0.25]), "term_features": np.arange(len(NAMES), dtype=np.int32)}
    for j, n in enumerate(NAMES):
        f = funcs.get(j, lambda x: 0.0 * x)
        if n in NOMINAL:
            cats = np.arange(24 if n == "hour" else 7, dtype=np.int64)
            ex[f"cats_{j}"] = cats
            ex[f"scores_{j}"] = np.concatenate(([0.0], f(cats.astype(float)), [0.0])).astype(np.float32)
        else:
            c = np.asarray(cuts_by_feature.get(j, [0.0]), dtype=np.float64)
            mids = np.concatenate(([c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]))
            ex[f"cuts_{j}"] = c
            ex[f"scores_{j}"] = np.concatenate(([0.0], f(mids), [0.0])).astype(np.float32)
    return ex


# 1. identical shapes -> 0 distance, unit correlation, full sign agreement, DR numerator 0
def test_identical_shapes_give_zero(X):
    grid = make_cells(X)
    ex = export_from_functions({0: np.tanh, 13: np.sin}, {0: np.linspace(-2, 2, 41)})
    s = cell_shapes(eval_terms_from_export(ex, X), grid)
    for j in (0, 13):
        w = grid.weights[j]
        assert M.d2(s[j], s[j], w) == 0.0
        assert M.shape_corr(s[j], s[j], w) == pytest.approx(1.0)
        assert M.sign_agreement(s[j], s[j], w, np.zeros_like(w))[0] == 1.0
        assert M.nd2(s[j], s[j], w) == 0.0


# 2. sign flip: d2 = 4 A^2, rho = -1, SA = 0, nd2 = 4; MA = 0 for a monotone continuous shape
def test_sign_flip(X):
    grid = make_cells(X)
    cuts = {0: np.linspace(-2, 2, 81)}
    a = cell_shapes(eval_terms_from_export(export_from_functions({0: np.tanh}, cuts), X), grid)[0]
    b = cell_shapes(eval_terms_from_export(export_from_functions({0: lambda x: -np.tanh(x)}, cuts), X), grid)[0]
    w = grid.weights[0]
    A = M.amplitude(a, w)
    assert M.d2(a, b, w) == pytest.approx(4 * A**2, rel=1e-6)
    assert M.shape_corr(a, b, w) == pytest.approx(-1.0, abs=1e-6)
    assert M.nd2(a, b, w) == pytest.approx(4.0, rel=1e-6)
    assert M.sign_agreement(a, b, w, np.zeros_like(w))[0] == 0.0
    assert M.monotonicity_agreement(grid.x_rep[0], a, b) == 0.0
    assert REF.d2(a, b, w) == pytest.approx(M.d2(a, b, w), rel=1e-12)


# 3. density weighting: differences where there is no data are invisible; a difference confined to
#    the top cell (mass ω_K) gives exactly δ² ω_K (1 − ω_K) after re-centring
def test_density_weighting_ignores_empty_tails(X):
    grid = make_cells(X)
    j = 1
    x = X.iloc[:, j].to_numpy()
    base = {j: np.tanh}
    far = {j: lambda v: np.tanh(v) + 10.0 * (v > x.max() + 1)}     # differs only beyond the data
    cuts = {j: np.concatenate((np.linspace(-6, 6, 61), [x.max() + 1]))}
    a = cell_shapes(eval_terms_from_export(export_from_functions(base, cuts), X), grid)[j]
    b = cell_shapes(eval_terms_from_export(export_from_functions(far, cuts), X), grid)[j]
    assert M.d2(a, b, grid.weights[j]) == pytest.approx(0.0, abs=1e-12)

    # sparse tail: exact known answer, using the contribution matrix directly
    top = grid.edges[j][-1]
    delta = 3.0
    C = np.zeros((N, len(NAMES)))
    C[:, j] = np.tanh(x)
    C2 = C.copy()
    C2[:, j] += delta * (x >= top)
    sa, sb = cell_shapes(C, grid)[j], cell_shapes(C2, grid)[j]
    wK = grid.weights[j][-1]
    assert M.d2(sa, sb, grid.weights[j]) == pytest.approx(delta**2 * wK * (1 - wK), rel=1e-9)
    assert wK == pytest.approx(1 / 50, abs=0.002)                  # the tail gets ~2% weight, no more


# 4. invariance to bin edges: equal functions with different internal cuts, and a constant offset from
#    centring under a different density, give identical shapes
def test_invariance_to_bin_edges(X):
    grid = make_cells(X)
    j = 2
    step = lambda v: np.where(v > 0.5, 1.0, np.where(v > -1.0, -0.3, 0.7))  # noqa: E731
    exA = export_from_functions({j: step}, {j: np.array([-1.0, 0.5])})
    exB = export_from_functions({j: lambda v: step(v) + 0.42},
                                {j: np.array([-3.0, -1.0, -0.2, 0.5, 0.9, 4.0])})   # redundant cuts + offset
    a = cell_shapes(eval_terms_from_export(exA, X), grid)[j]
    b = cell_shapes(eval_terms_from_export(exB, X), grid)[j]
    np.testing.assert_allclose(a, b, atol=1e-6)                     # float32 scores
    assert M.d2(a, b, grid.weights[j]) < 1e-12


# 5. drift_pair agrees with the naive reference on random inputs (all keys the reference computes)
@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_drift_pair_matches_reference(overlap):
    rng = np.random.default_rng(5)
    Ks = [50, 37, 24, 7, 12]
    weights = [rng.dirichlet(np.ones(k) * 5) for k in Ks]

    def shapes(R, shift):
        out = []
        for k, w in zip(Ks, weights):
            g = np.sin(np.linspace(0, 3, k)) + shift
            F_ = g + 0.3 * rng.standard_normal((R, k))
            out.append(F_ - (F_ @ w)[:, None])
        return out

    sa, ha, sb, hb = shapes(5, 0), shapes(10, 0), shapes(5, 0.1), shapes(10, 0.1)
    res = F.drift_pair({"seed": sa, "half": ha}, {"seed": sb, "half": hb}, weights, overlap)
    ref = REF.drift(sa, ha, sb, hb, weights, overlap)
    for key in ("d2_cross", "s2_seed_a", "s2_tot_a", "s2_samp_a", "s2_seed_b", "s2_tot_b", "DR", "DR_adj",
                "snr_a", "snr_b", "A_a", "A_b", "rho", "nd2"):
        np.testing.assert_allclose(res[key], ref[key], rtol=1e-9, atol=1e-12, err_msg=key)
    for key in ("DR_tot", "DR_adj_tot", "seed_share_a", "seed_share_b", "wtau", "jacc3", "tv"):
        assert float(res[key]) == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key
    tau = REF.cell_tau(ha[0], hb[0])
    sa_m, sb_m = sa[0].mean(0), sb[0].mean(0)
    SA_ref, cov_ref = REF.sign_agreement(sa_m, sb_m, weights[0], tau)
    assert float(res["SA"][0]) == pytest.approx(SA_ref)
    assert float(res["SA_cov"][0]) == pytest.approx(cov_ref)


# 6. cell_shapes_many equals cell means computed by hand
def test_cell_means_by_hand(X):
    grid = make_cells(X)
    rng = np.random.default_rng(2)
    C = rng.standard_normal((3, N, len(NAMES)))
    S = cell_shapes_many(C, grid)
    for j in (0, 6, 12):
        cell, w = grid.cell_of_row[j], grid.weights[j]
        for r in range(3):
            m = np.array([C[r, cell == k, j].mean() for k in range(len(w))])
            np.testing.assert_allclose(S[j][r], m - (w * m).sum(), atol=1e-12)
