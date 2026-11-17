"""Stationary surrogates of the panel (PLAN §9.3, D-50). Synthetic data except one skipped-if-absent test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ebmregime.panel as panel_mod
from ebmregime.config import load_config
from ebmregime.splits import stable_seed
from ebmregime.stability import surrogate as sg
from ebmregime.stability.surrogate import (
    make_surrogate,
    surrogate_bounds,
    surrogate_panel,
    surrogate_rows,
)


def synth_panel(n: int = 2000, t0: str = "2020-01-01", seed: int = 0) -> pd.DataFrame:
    """Hourly UTC panel with the dtypes of the real one (float, int64, bool, nullable Int8)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(t0, periods=n, freq="h", tz="UTC", name="ts")
    ok = rng.random(n) > 0.1
    y = rng.normal(size=n)
    y[~ok] = np.nan
    reg = pd.array(rng.integers(0, 2, size=n), dtype="Int8")
    reg[rng.random(n) < 0.05] = pd.NA
    return pd.DataFrame({
        "row_id": np.arange(n, dtype=np.int64),
        "x": np.sin(np.arange(n) / 50.0) + rng.normal(scale=0.1, size=n),
        "hour": ts.hour.to_numpy().astype(np.int64),
        "features_ok": ok,
        "y_dir_h8": y,
        "r1_vol_2y": reg,
        "valid": rng.random(n) > 0.2,
    }, index=ts)


def run_lengths(pos: np.ndarray, n: int) -> np.ndarray:
    """Lengths of maximal runs of consecutive (mod n) source positions."""
    brk = np.flatnonzero(pos[1:] != (pos[:-1] + 1) % n) + 1
    edges = np.concatenate([[0], brk, [len(pos)]])
    return np.diff(edges)


# ---- surrogate_rows -------------------------------------------------------------------------


def test_rows_deterministic_and_seed_sensitive():
    a = surrogate_rows(5000, 252, 123)
    b = surrogate_rows(5000, 252, 123)
    c = surrogate_rows(5000, 252, 124)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rows_bounds_and_dtype():
    for n, L, s in [(1, 1, 0), (1, 252, 1), (7, 3, 2), (10_000, 252, 3), (10_000, 1, 4)]:
        pos = surrogate_rows(n, L, s)
        assert pos.dtype == np.int64 and pos.shape == (n,)
        assert pos.min() >= 0 and pos.max() < n
    with pytest.raises(ValueError):
        surrogate_rows(0, 10, 0)
    with pytest.raises(ValueError):
        surrogate_rows(10, 0.5, 0)
    with pytest.raises(ValueError):
        surrogate_rows(10, float("nan"), 0)


def test_rows_within_block_steps_are_consecutive_mod_n():
    # mean_block = 1: every draw starts a new block -> iid uniform positions, almost no runs.
    pos = surrogate_rows(20_000, 1, 5)
    assert run_lengths(pos, 20_000).mean() < 1.01


@pytest.mark.parametrize("L", [10, 50, 252])
def test_mean_run_length_matches_mean_block(L):
    n = 400_000
    lens = run_lengths(surrogate_rows(n, L, 11), n)
    # Each run is a block (blocks merge only if a new start lands exactly on the next position).
    assert abs(lens.mean() - L) < 0.08 * L
    # Geometric: P(length = 1) = 1/L, and SD ≈ sqrt(L(L-1)).
    assert abs(lens.std() - np.sqrt(L * (L - 1))) < 0.12 * L


def test_circular_wrap_single_block_is_rotation():
    n = 50
    rotated = 0
    for s in range(20):
        pos = surrogate_rows(n, float("inf"), s)
        np.testing.assert_array_equal(pos, (pos[0] + np.arange(n)) % n)
        rotated += pos[0] > 0
    assert rotated > 0  # the block wraps from n-1 to 0 in those cases


def test_circular_wrap_occurs_inside_long_blocks():
    n, L = 1000, 200
    wraps = 0
    for s in range(50):
        pos = surrogate_rows(n, L, s)
        wraps += int(np.sum((pos[:-1] == n - 1) & (pos[1:] == 0)))
    assert wraps > 0


def test_marginal_uniform_in_expectation_including_edges():
    # Circular wrap + uniform starts: every source position has expected count 1, including the
    # ends of the range (a non-circular block bootstrap would underweight them).
    n, L, R = 200, 10, 3000
    counts = np.zeros(n)
    for s in range(R):
        counts += np.bincount(surrogate_rows(n, L, s), minlength=n)
    mean = counts / R
    assert np.abs(mean - 1).max() < 0.35
    assert abs(mean[:10].mean() - 1) < 0.15 and abs(mean[-10:].mean() - 1) < 0.15


# ---- make_surrogate -------------------------------------------------------------------------


def test_make_surrogate_index_is_original_grid_and_range():
    p = synth_panel(3000)
    start, end = p.index[100], p.index[2500]
    out = make_surrogate(p, start, end, 50, 7)
    expected = p.index[(p.index >= start) & (p.index < end)]
    pd.testing.assert_index_equal(out.index, expected)
    assert out.index.name == "ts" and str(out.index.tz) == "UTC"
    assert len(out) == 2400
    assert out["row_id"].between(100, 2499).all()  # only rows from inside the range


def test_make_surrogate_rows_move_jointly_and_dtypes_preserved():
    p = synth_panel(3000)
    start, end = p.index[100], p.index[2500]
    out = make_surrogate(p, start, end, 50, 7)
    pd.testing.assert_series_equal(out.dtypes, p.dtypes)
    sub = p.iloc[100:2500]
    pos = surrogate_rows(len(sub), 50, 7)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), sub.iloc[pos].reset_index(drop=True))
    # Permutation with repetition: each output row is the source row named by its row_id, all
    # columns jointly (NaN / NA masks included).
    src = p.set_index("row_id").loc[out["row_id"].to_numpy()].reset_index()
    pd.testing.assert_frame_equal(out.reset_index(drop=True)[src.columns], src)
    assert out["row_id"].duplicated().any()  # with repetition
    assert (out.loc[~out["features_ok"], "y_dir_h8"].isna()).all()  # masked rows stay masked


def test_make_surrogate_deterministic_and_does_not_mutate():
    p = synth_panel(1500)
    before = p.copy()
    a = make_surrogate(p, "2020-01-02", "2020-02-15", 252, 3)
    b = make_surrogate(p, pd.Timestamp("2020-01-02", tz="UTC"), "2020-02-15T00:00Z", 252, 3)
    c = make_surrogate(p, "2020-01-02", "2020-02-15", 252, 4)
    pd.testing.assert_frame_equal(a, b)  # naive bounds are taken as UTC
    assert not a.equals(c)
    pd.testing.assert_frame_equal(p, before)


def test_make_surrogate_marginal_preserved_in_expectation():
    p = synth_panel(1000)
    sub = p.loc["2020-01-05":"2020-02-05"]
    start, end = sub.index[0], sub.index[-1] + pd.Timedelta(hours=1)
    means = [make_surrogate(p, start, end, 24, s)["x"].mean() for s in range(400)]
    se = np.std(means) / np.sqrt(len(means))
    assert abs(np.mean(means) - sub["x"].mean()) < 4 * se + 1e-12


def test_make_surrogate_rejects_bad_input():
    p = synth_panel(100)
    with pytest.raises(ValueError):
        make_surrogate(p, "2021-01-01", "2021-02-01", 10, 0)  # no rows in range
    with pytest.raises(ValueError):
        make_surrogate(p, "2020-01-03", "2020-01-02", 10, 0)  # end before start
    with pytest.raises(ValueError):
        make_surrogate(p.iloc[::-1], "2020-01-01", "2020-01-03", 10, 0)  # not sorted


# ---- surrogate_panel ------------------------------------------------------------------------


def test_prereg_settings():
    cfg = load_config("prereg")
    assert cfg["surrogates"]["mean_block_hours"] == 252
    start, end = surrogate_bounds("BTCUSDT", cfg)
    assert start == pd.Timestamp("2018-12-01", tz="UTC")
    assert end == pd.Timestamp("2026-11-01", tz="UTC")
    s2, e2 = surrogate_bounds("SOLUSDT", cfg)
    assert s2 == pd.Timestamp("2021-03-01", tz="UTC") and e2 == end


@pytest.fixture
def synth_symbol(monkeypatch):
    p = synth_panel(24 * 120, t0="2020-01-01")
    monkeypatch.setattr(panel_mod, "load_panel", lambda sym: p if sym == "SYNTH" else None)
    cfg = {
        "windows": {"train_months": 1, "first_tau": {"SYNTH": "2020-02-01"}, "last_tau": "2020-03-01"},
        "surrogates": {"mean_block_hours": 30, "seed": 99},
    }
    sg._surrogate_cached.cache_clear()
    yield p, cfg
    sg._surrogate_cached.cache_clear()


def test_surrogate_panel_uses_prereg_bounds_seed_and_cache(synth_symbol):
    p, cfg = synth_symbol
    out = surrogate_panel("SYNTH", 0, cfg)
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-04-01", tz="UTC")
    expected = make_surrogate(p, start, end, 30, stable_seed(99, "SYNTH", "surrogate", 0))
    pd.testing.assert_frame_equal(out, expected)
    assert out.index[0] == start and out.index[-1] == end - pd.Timedelta(hours=1)
    assert surrogate_panel("SYNTH", 0, cfg) is out  # cached per process
    other = surrogate_panel("SYNTH", 1, cfg)
    assert not other.equals(out)
    pd.testing.assert_index_equal(other.index, out.index)
    with pytest.raises(ValueError):
        surrogate_panel("SYNTH", -1, cfg)


def test_real_btc_surrogate_shape_and_dtypes():
    if not panel_mod.panel_path("BTCUSDT").exists():
        pytest.skip("BTCUSDT panel not built")
    real = panel_mod.load_panel("BTCUSDT")
    out = surrogate_panel("BTCUSDT", 0)
    start, end = surrogate_bounds("BTCUSDT")
    sub = real[(real.index >= start) & (real.index < end)]
    pd.testing.assert_index_equal(out.index, sub.index)
    pd.testing.assert_series_equal(out.dtypes, real.dtypes)
    assert surrogate_panel("BTCUSDT", 0) is out
