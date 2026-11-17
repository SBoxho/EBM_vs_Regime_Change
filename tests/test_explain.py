"""Reference sample, pooled cells and cell-averaged shapes (PLAN §6.1; D-15, D-46, D-50). Synthetic data only."""

from __future__ import annotations

import inspect
import sys
import types

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config
from ebmregime.explain import (
    CellGrid,
    cell_shapes,
    cell_shapes_many,
    ebm_contrib,
    eta,
    explain_settings,
    make_cells,
    make_xref,
)
from ebmregime.explain.cells import _merge_small


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def _brute_shape(x_cell: np.ndarray, c: np.ndarray, w: np.ndarray) -> np.ndarray:
    k = w.size
    f = np.array([c[x_cell == i].mean() for i in range(k)])
    return f - (w * f).sum()


# --------------------------------------------------------------------------------------------- config


def test_defaults_match_prereg():
    pre = load_config("prereg")
    e = pre["explain"]
    s = explain_settings()
    sig = inspect.signature(make_cells).parameters
    assert sig["n_cells"].default == e["cells"]["n"] == s["n_cells"] == 50
    assert sig["min_rows"].default == e["cells"]["min_rows"] == s["min_rows"] == 20
    assert e["cells"]["kind"] == "pooled_equal_probability" and e["cells"]["merge_ties"] is True
    assert e["x_ref"]["source"] == "union_of_window_rows"
    assert s["cap_rows"] == e["x_ref"]["cap_rows"] == 20000 and s["seed"] == e["x_ref"]["seed"]
    assert e["recentre"] == "pooled_density"
    fb = [f for f in pre["fallbacks"] if "cell merging" in f]
    assert len(fb) == 1 and "< 10 cells" in fb[0] and "G = 20" in fb[0]
    assert sig["fallback_cells"].default == s["fallback_cells"] == 20
    assert sig["min_cells"].default == s["min_cells"] == 10


# ---------------------------------------------------------------------------------------------- X_ref


def test_xref_union_without_cap():
    a = np.zeros(100, bool)
    b = np.zeros(100, bool)
    a[10:40] = True
    b[30:60] = True                              # overlap 30..39 counts once
    pos = make_xref([a, b], cap=1000, seed=1)
    assert pos.dtype == np.int64
    np.testing.assert_array_equal(pos, np.arange(10, 60))


def test_xref_cap_is_sorted_uniform_subsample_without_replacement():
    rng = np.random.default_rng(0)
    masks = [rng.random(50_000) < 0.4, rng.random(50_000) < 0.4]
    union = np.flatnonzero(masks[0] | masks[1])
    pos = make_xref(masks, cap=5000, seed=7)
    assert pos.size == 5000
    assert np.all(np.diff(pos) > 0)              # sorted, no repeats
    assert np.isin(pos, union).all()
    np.testing.assert_array_equal(pos, make_xref(masks, cap=5000, seed=7))      # deterministic
    assert not np.array_equal(pos, make_xref(masks, cap=5000, seed=8))
    # exactly the documented draw
    pick = np.random.default_rng(7).choice(union.size, size=5000, replace=False)
    np.testing.assert_array_equal(pos, np.sort(union[pick]))
    # uniform: the subsample spreads evenly over the union
    frac = np.searchsorted(pos, np.quantile(union, [0.25, 0.5, 0.75])) / pos.size
    np.testing.assert_allclose(frac, [0.25, 0.5, 0.75], atol=0.03)


def test_xref_rejects_bad_masks():
    with pytest.raises(ValueError):
        make_xref([], cap=10, seed=0)
    with pytest.raises(ValueError):
        make_xref([np.zeros(5, bool), np.zeros(6, bool)], cap=10, seed=0)
    with pytest.raises(TypeError):
        make_xref([np.zeros(5, int)], cap=10, seed=0)


# ---------------------------------------------------------------------------------------------- cells


def test_equal_probability_cells_on_continuous_data():
    rng = np.random.default_rng(1)
    X = _frame(a=rng.standard_normal(20_000), b=rng.exponential(size=20_000))
    g = make_cells(X, nominal=[])
    assert isinstance(g, CellGrid)
    assert g.features == ("a", "b") and g.n_ref == 20_000
    for j in range(2):
        assert g.n_cells[j] == 50 and not g.fallback_used[j] and not g.is_nominal[j]
        np.testing.assert_allclose(g.weights[j], 1 / 50, atol=1e-4)
        assert g.weights[j].sum() == pytest.approx(1.0)
        assert g.edges[j].size == 49 and np.all(np.diff(g.edges[j]) > 0)
        np.testing.assert_array_equal(g.cell_of_row[j], np.searchsorted(g.edges[j], X.iloc[:, j], side="right"))
        # edges are the pooled k/50 quantiles
        np.testing.assert_allclose(g.edges[j], np.quantile(X.iloc[:, j], np.arange(1, 50) / 50))
        # x_rep is the cell median and lies inside the cell
        x = X.iloc[:, j].to_numpy()
        med = [np.median(x[g.cell_of_row[j] == k]) for k in range(50)]
        np.testing.assert_allclose(g.x_rep[j], med)
    # arrays are read-only
    with pytest.raises(ValueError):
        g.weights[0][0] = 1.0


def test_ties_are_merged():
    rng = np.random.default_rng(2)
    n = 20_000
    x = np.concatenate([rng.uniform(-1, -0.01, int(0.2 * n)), np.zeros(int(0.6 * n)),
                        rng.uniform(0.01, 1, int(0.2 * n))])
    rng.shuffle(x)
    g = make_cells(_frame(x=x), nominal=[])
    e = g.edges[0]
    assert np.all(np.diff(e) > 0)                # duplicate edges removed
    zero_cells = np.unique(g.cell_of_row[0][x == 0])
    assert zero_cells.size == 1                  # the mass point is one cell
    k0 = zero_cells[0]
    assert g.weights[0][k0] >= 0.6 and g.x_rep[0][k0] == 0.0
    # about 10 cells of 1/50 on each side, plus the tie cell
    assert 19 <= g.n_cells[0] <= 23 and not g.fallback_used[0]
    assert (np.bincount(g.cell_of_row[0]) >= 20).all()


def test_tie_at_minimum_leaves_no_empty_cell():
    rng = np.random.default_rng(3)
    x = np.concatenate([np.zeros(6000), rng.uniform(0.1, 1, 4000)])
    g = make_cells(_frame(x=x), nominal=[])
    cnt = np.bincount(g.cell_of_row[0], minlength=g.n_cells[0])
    assert (cnt >= 20).all()
    assert g.cell_of_row[0][x == 0].max() == 0 and g.weights[0][0] == pytest.approx(0.6)


def test_merge_small_rule():
    # the small cell goes to its smaller neighbour
    assert _merge_small(np.array([30, 5, 25, 40]), 20).tolist() == [0, 2]      # cells [30, 30, 40]
    # smallest first: 3 joins 5 (smaller neighbour), then 8 joins 30
    assert _merge_small(np.array([30, 5, 3, 40]), 20).tolist() == [2]          # cells [38, 40]
    # end cells have one neighbour; ties go left
    assert _merge_small(np.array([4, 50, 50]), 20).tolist() == [1]
    assert _merge_small(np.array([50, 50, 4]), 20).tolist() == [0]
    assert _merge_small(np.array([30, 10, 30]), 20).tolist() == [1]
    # all small: ends at one cell at most
    assert _merge_small(np.array([1, 1, 1]), 20).tolist() == []
    # empty cells merged even with min_rows = 0
    assert _merge_small(np.array([0, 30, 30]), 0).tolist() == [1]


def test_min_rows_merge_on_small_sample():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(600)                  # 50 cells of 12 rows: all below 20
    g = make_cells(_frame(x=x), nominal=[])
    cnt = np.bincount(g.cell_of_row[0], minlength=g.n_cells[0])
    assert (cnt >= 20).all()
    assert 10 <= g.n_cells[0] < 50 and not g.fallback_used[0]
    q = np.quantile(x, np.arange(1, 50) / 50)
    assert np.isin(g.edges[0], q).all()          # merging only removes edges
    np.testing.assert_allclose(g.weights[0], cnt / 600)


def test_fallback_to_20_cells():
    rng = np.random.default_rng(5)
    x = rng.permutation(np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 1000))   # 5 levels: < 10 cells at G = 50
    g = make_cells(_frame(x=x), nominal=[])
    assert g.fallback_used[0]
    assert g.n_cells[0] == 5                      # still < 10: kept, flag stays set
    np.testing.assert_allclose(g.weights[0], 0.2)
    np.testing.assert_array_equal(g.x_rep[0], [1, 2, 3, 4, 5])
    g20 = make_cells(_frame(x=x), n_cells=20, nominal=[], min_cells=0)
    np.testing.assert_array_equal(g.edges[0], g20.edges[0])
    assert not g20.fallback_used[0]
    # a continuous companion is not affected
    g2 = make_cells(_frame(x=x, y=rng.standard_normal(5000)), nominal=[])
    assert g2.fallback_used == (True, False) and g2.n_cells[1] == 50


def test_nominal_levels_weighted_by_pooled_frequency():
    rng = np.random.default_rng(6)
    hours = np.array([h for h in range(24) if h not in (3, 17)])
    p = rng.random(hours.size)
    p /= p.sum()
    hour = rng.choice(hours, size=10_000, p=p).astype(np.int64)
    dow = rng.integers(0, 7, 10_000).astype(np.int64)
    X = _frame(ret=rng.standard_normal(10_000), hour=hour, dow=dow)
    g = make_cells(X)                            # default nominal = panel.nominal_features() (hour, dow)
    assert g.is_nominal == (False, True, True) and g.fallback_used == (False, False, False)
    present = np.unique(hour)
    assert g.n_cells[1] == present.size and 3 not in present and 17 not in present
    np.testing.assert_array_equal(g.edges[1], present)
    np.testing.assert_array_equal(g.x_rep[1], present.astype(float))
    np.testing.assert_array_equal(g.edges[1][g.cell_of_row[1]], hour)
    np.testing.assert_allclose(g.weights[1], [np.mean(hour == h) for h in present])
    assert g.n_cells[2] == 7 and g.weights[2].sum() == pytest.approx(1.0)
    # nominal levels are never merged even if rare
    X2 = _frame(hour=np.r_[np.zeros(500, np.int64), [5]])
    g2 = make_cells(X2)
    assert g2.n_cells == (2,) and g2.weights[0][1] == pytest.approx(1 / 501)


def test_cells_reject_missing_values():
    with pytest.raises(ValueError):
        make_cells(_frame(x=[1.0, np.nan, 2.0]), nominal=[])
    with pytest.raises(ValueError):
        make_cells(_frame(x=[]), nominal=[])


# --------------------------------------------------------------------------------------------- shapes


@pytest.fixture(scope="module")
def grid_data():
    rng = np.random.default_rng(10)
    n = 5000
    X = _frame(a=rng.standard_normal(n), b=rng.uniform(0, 1, n), hour=rng.integers(0, 24, n).astype(np.int64),
               c=np.round(rng.standard_normal(n), 1))
    return X, make_cells(X, nominal=["hour"])


def test_recentred_shapes_have_zero_weighted_mean(grid_data):
    X, g = grid_data
    rng = np.random.default_rng(11)
    C = rng.standard_normal((len(X), 4)) * 3 + 7
    shapes = cell_shapes(C, g)
    assert len(shapes) == 4
    for j, s in enumerate(shapes):
        assert s.shape == (g.n_cells[j],)
        assert abs(np.dot(g.weights[j], s)) < 1e-12
        np.testing.assert_allclose(s, _brute_shape(g.cell_of_row[j], C[:, j], g.weights[j]), atol=1e-12)


def test_shape_of_function_of_x_is_analytic_cell_mean():
    n = 20_000
    x = (np.arange(n) + 0.5) / n                   # deterministic uniform grid: cell k = rows 400k..400k+399
    X = _frame(x=x)
    g = make_cells(X, nominal=[])
    assert g.n_cells == (50,)
    np.testing.assert_array_equal(np.bincount(g.cell_of_row[0]), np.full(50, 400))
    k = np.arange(50)
    (s,) = cell_shapes(x[:, None], g)            # c = x: cell mean (k + 0.5)/50, pooled mean 1/2
    np.testing.assert_allclose(s, (k + 0.5) / 50 - 0.5, atol=1e-12)
    (s2,) = cell_shapes((2 * x - 3)[:, None], g)  # affine: shape scales, constant drops
    np.testing.assert_allclose(s2, 2 * ((k + 0.5) / 50 - 0.5), atol=1e-12)
    np.testing.assert_allclose(g.x_rep[0], (k + 0.5) / 50, atol=1e-12)


def test_step_functions_with_different_bin_edges_give_identical_shapes(grid_data):
    X, g = grid_data
    a = X["a"].to_numpy()

    def lookup(x, cuts, scores):                 # D-46: scores[digitize(x, cuts) + 1], index 0 = missing
        return scores[np.digitize(x, cuts, right=False) + 1]

    # same function: −1 below 0, +2 on [0, 1), 0.5 from 1, with redundant internal cuts in the second
    f1 = lookup(a, np.array([0.0, 1.0]), np.array([np.nan, -1.0, 2.0, 0.5, 0.0]))
    f2 = lookup(a, np.array([-0.7, -0.2, 0.0, 0.3, 0.6, 1.0, 1.4]),
                np.array([np.nan, -1, -1, -1, 2, 2, 2, 0.5, 0.5, 0.0]))
    np.testing.assert_array_equal(f1, f2)
    C1 = np.zeros((len(X), 4))
    C2 = np.zeros((len(X), 4))
    C1[:, 0] = f1
    C2[:, 0] = f2 - 0.37                         # each fit centred under its own density: constant offset
    s1 = cell_shapes(C1, g)
    s2 = cell_shapes(C2, g)
    np.testing.assert_allclose(s1[0], s2[0], atol=1e-12)
    for j in range(1, 4):
        np.testing.assert_array_equal(s1[j], 0.0)


def test_constant_shift_removed_by_recentring(grid_data):
    X, g = grid_data
    rng = np.random.default_rng(12)
    C = rng.standard_normal((len(X), 4))
    # centre column 1 under a different density (only the rows with b > 0.7): a pure constant shift
    shifted = C.copy()
    sel = X["b"].to_numpy() > 0.7
    shifted -= shifted[sel].mean(axis=0)
    shifted[:, 3] += 123.0
    for s, t in zip(cell_shapes(C, g), cell_shapes(shifted, g)):
        np.testing.assert_allclose(s, t, atol=1e-10)


def test_many_matches_single_and_accepts_float32(grid_data):
    X, g = grid_data
    rng = np.random.default_rng(13)
    C = rng.standard_normal((3, len(X), 4)).astype(np.float32)
    many = cell_shapes_many(C, g)
    assert [m.shape for m in many] == [(3, k) for k in g.n_cells]
    for r in range(3):
        single = cell_shapes(C[r], g)
        for j in range(4):
            assert many[j].dtype == np.float64
            np.testing.assert_allclose(many[j][r], single[j], atol=1e-12)
            np.testing.assert_allclose(single[j], _brute_shape(g.cell_of_row[j], C[r, :, j].astype(float),
                                                               g.weights[j]), atol=1e-6)


def test_shapes_reject_wrong_shape(grid_data):
    X, g = grid_data
    with pytest.raises(ValueError):
        cell_shapes(np.zeros((len(X), 3)), g)
    with pytest.raises(ValueError):
        cell_shapes(np.zeros((len(X) - 1, 4)), g)
    with pytest.raises(ValueError):
        cell_shapes_many(np.zeros((len(X), 4)), g)


# ------------------------------------------------------------------------------------- contributions


def test_eta_is_intercept_plus_row_sum():
    export = {"intercept": np.array([-0.25])}
    C = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(eta(export, C), -0.25 + C.sum(axis=1))
    stack = np.ones((2, 3, 4))
    np.testing.assert_allclose(eta(export, stack), np.full((2, 3), 3.75))


def test_ebm_contrib_delegates_to_models(monkeypatch):
    calls = []

    def fake_eval(export, X):
        calls.append((export, X))
        return np.ones((len(X), 2), dtype=np.float32)

    mod = types.ModuleType("ebmregime.models.ebm")
    mod.eval_terms_from_export = fake_eval
    if "ebmregime.models" not in sys.modules:
        monkeypatch.setitem(sys.modules, "ebmregime.models", types.ModuleType("ebmregime.models"))
    monkeypatch.setitem(sys.modules, "ebmregime.models.ebm", mod)
    export = {"intercept": np.array([0.0])}
    X = _frame(a=[1.0, 2.0, 3.0], b=[0.0, 0.0, 1.0])
    out = ebm_contrib(export, X)
    assert out.dtype == np.float64 and out.shape == (3, 2)
    assert calls[0][0] is export and calls[0][1] is X
