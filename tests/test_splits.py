"""Windows, purge, time-ordered bags, half-samples and seeds (PLAN §6.4, §7; D-13, D-17, D-19, D-34).

Synthetic timestamps only.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config
from ebmregime.splits import (
    half_sample,
    label_time,
    overlap_fraction,
    replicate_plan,
    retrain_schedule,
    stable_seed,
    stationary_bootstrap_counts,
    tail_start,
    time_ordered_bags,
    train_mask,
    xgb_split,
)
from ebmregime.splits import test_mask as window_test_mask

H = pd.Timedelta(hours=1)
VAL = dict(tail_frac=0.15, block_hours=168, bag_train_frac=0.85)


def hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")


@pytest.fixture(scope="module")
def prereg():
    return load_config("prereg")


@pytest.fixture(scope="module")
def grid():
    return hourly("2018-12-01", "2026-11-01")


@pytest.fixture(scope="module")
def window_ts(grid):
    """Training rows of the window at tau = 2021-01-01, h = 8 (gap-free)."""
    return grid[train_mask(grid, "2021-01-01", 12, 8)]


@pytest.fixture(scope="module")
def gapped_ts(window_ts):
    """Same window with 10% of rows dropped at random and a 2-week hole."""
    rng = np.random.default_rng(7)
    keep = rng.random(len(window_ts)) > 0.10
    hole = (window_ts >= "2020-05-03") & (window_ts < "2020-05-17")
    return window_ts[keep & ~np.asarray(hole)]


# ---------------------------------------------------------------- windows


def test_schedule_counts_from_config(prereg):
    w = prereg["windows"]
    counts = {}
    for sym, first in w["first_tau"].items():
        taus = retrain_schedule(first, w["last_tau"], w["step_months"])
        counts[sym] = len(taus)
        assert taus[0] == pd.Timestamp(first, tz="UTC")
        assert taus[-1] == pd.Timestamp("2026-10-01", tz="UTC")
        assert all(t.day == 1 and t == t.normalize() and str(t.tz) == "UTC" for t in taus)
        assert all((b.year * 12 + b.month) - (a.year * 12 + a.month) == 1 for a, b in zip(taus, taus[1:]))
    assert counts == {"BTCUSDT": 83, "ETHUSDT": 83, "XRPUSDT": 83, "SOLUSDT": 56}
    assert w["train_months"] == 12


def test_schedule_rejects_non_month_start():
    with pytest.raises(ValueError):
        retrain_schedule("2019-12-02", "2020-01-01")


def test_label_time():
    ts = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 15:00"], tz="UTC")
    assert list(label_time(ts, 8)) == list(pd.DatetimeIndex(["2020-01-01 09:00", "2020-01-02 00:00"], tz="UTC"))


def test_train_mask_hand_case(grid):
    m = train_mask(grid, "2020-01-01", 12, 8)
    sel = grid[m]
    assert sel[0] == pd.Timestamp("2019-01-01", tz="UTC")
    # last row: label known at ts + 9h <= 2020-01-01 00:00
    assert sel[-1] == pd.Timestamp("2019-12-31 15:00", tz="UTC")
    assert len(sel) == 365 * 24 - 8   # rows 00:00 .. 15:00 on 12-31: an h-row purge
    assert np.all(np.diff(np.flatnonzero(m)) == 1)


def test_train_mask_random_properties(grid):
    rng = np.random.default_rng(0)
    taus = retrain_schedule("2019-12-01", "2026-10-01")
    for _ in range(12):
        tau = taus[rng.integers(len(taus))]
        h = int(rng.choice([1, 8, 24]))
        m = train_mask(grid, tau, 12, h)
        start = tau - pd.DateOffset(months=12)
        lt = label_time(grid, h)
        assert np.all(lt[m] <= tau)
        assert np.all(grid[m] >= start)
        # and nothing that satisfies both conditions is left out
        expected = np.asarray((grid >= start) & (lt <= tau))
        assert np.array_equal(m, expected)
        assert m.sum() == ((tau - start) // H) - h   # exactly h rows purged


def test_train_mask_usable(grid):
    usable = np.ones(len(grid), dtype=bool)
    usable[::3] = False
    m = train_mask(grid, "2021-06-01", 12, 8, usable=usable)
    assert np.array_equal(m, train_mask(grid, "2021-06-01", 12, 8) & usable)


def test_test_mask_is_calendar_month(grid):
    for tau, days in [("2020-02-01", 29), ("2021-02-01", 28), ("2021-07-01", 31), ("2026-10-01", 31)]:
        m = window_test_mask(grid, tau)
        sel = grid[m]
        assert len(sel) == days * 24
        assert sel[0] == pd.Timestamp(tau, tz="UTC")
        assert sel[-1] == pd.Timestamp(tau, tz="UTC") + pd.DateOffset(months=1) - H
        # train and test rows never overlap and train labels are known by tau
        assert not (m & train_mask(grid, tau, 12, 8)).any()


@pytest.mark.parametrize("delta,expected", [
    (0, 1.0),
    (1, 335 / 366),   # [2020-02-01, 2021-01-01) / [2020-01-01, 2021-01-01), 2020 is a leap year
    (6, 184 / 366),   # [2020-07-01, 2021-01-01)
    (12, 0.0),
    (13, 0.0),
])
def test_overlap_fraction(delta, expected):
    a = pd.Timestamp("2021-01-01", tz="UTC")
    b = a + pd.DateOffset(months=delta)
    assert overlap_fraction(a, b, 12) == pytest.approx(expected, abs=1e-12)
    assert overlap_fraction(b, a, 12) == pytest.approx(expected, abs=1e-12)
    assert overlap_fraction(a, b, 12) == pytest.approx(max(0.0, 1 - delta / 12), abs=0.02)


# ---------------------------------------------------------------- bags


def roles(ts, h, vs):
    lt = label_time(ts, h)
    tail = np.asarray(ts >= vs)
    purged = ~tail & np.asarray(lt > vs)
    return tail, purged, ~tail & ~purged


def test_bags_tiny_hand_case():
    ts = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    b = time_ordered_bags(ts, horizon=2, n_bags=6, tail_frac=0.25, block_hours=4, bag_train_frac=0.5, seed=3)
    assert b.shape == (20, 6) and b.dtype == np.int8
    assert np.all(b[15:] == -1)            # val_start = ts[floor(0.75 * 20)] = ts[15]
    assert np.all(b[13:15] == 0)           # ts + 3h > ts[15] for rows 13, 14: purged
    # rows 0..12 form blocks [0-3], [4-7], [8-11], [12]; 2 of 4 blocks are +1 in every bag
    blocks = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 13)]
    for j in range(6):
        vals = [np.unique(b[s, j]) for s in blocks]
        assert all(len(v) == 1 for v in vals)
        assert sum(v[0] == 1 for v in vals) == 2


@pytest.mark.parametrize("which", ["window_ts", "gapped_ts"])
def test_bags_properties(which, request):
    ts = request.getfixturevalue(which)
    h, n_bags = 8, 14
    b = time_ordered_bags(ts, h, n_bags, seed=11, **VAL)
    n = len(ts)
    assert b.shape == (n, n_bags) and b.dtype == np.int8
    assert set(np.unique(b)) <= {-1, 0, 1}

    vs = tail_start(ts, 0.15)
    assert vs == ts[int(np.floor(0.85 * n))]
    tail, purged, elig = roles(ts, h, vs)
    # tail = the last 15% of the rows in time, -1 in every bag and nowhere else
    assert tail.sum() == n - int(np.floor(0.85 * n))
    assert np.all(tail[-tail.sum():])
    assert np.array_equal((b == -1).all(axis=1), tail) and np.array_equal((b == -1).any(axis=1), tail)
    # purge: no +1 row has its label after val_start; purged rows 0 in all bags
    assert np.all(label_time(ts[(b == 1).any(axis=1)], h) <= vs)
    assert np.all(b[purged] == 0)
    if which == "window_ts":
        assert purged.sum() == h and np.all(purged[-tail.sum() - h:-tail.sum()])

    # blocks are all-or-nothing and each bag uses round(0.85 * n_blocks) of them
    blk = np.asarray((ts[elig] - ts[0]) // H) // 168
    n_blocks = len(np.unique(blk))
    k = int(np.floor(0.85 * n_blocks + 0.5))
    for j in range(n_bags):
        col = b[elig, j]
        per_block = pd.Series(col).groupby(blk).agg(["min", "max"])
        assert (per_block["min"] == per_block["max"]).all()
        assert (per_block["max"] == 1).sum() == k
        assert abs(k / n_blocks - 0.85) <= 1 / n_blocks
    # bags differ across columns
    cols = {b[:, j].tobytes() for j in range(n_bags)}
    assert len(cols) == n_bags


def test_bags_deterministic(window_ts):
    a = time_ordered_bags(window_ts, 8, 14, seed=5, **VAL)
    assert np.array_equal(a, time_ordered_bags(window_ts, 8, 14, seed=5, **VAL))
    assert not np.array_equal(a, time_ordered_bags(window_ts, 8, 14, seed=6, **VAL))


def test_bags_explicit_val_start(window_ts):
    vs = pd.Timestamp("2020-11-01", tz="UTC")
    b = time_ordered_bags(window_ts, 8, 3, seed=1, val_start=vs, **VAL)
    assert np.array_equal((b == -1).all(axis=1), np.asarray(window_ts >= vs))
    assert np.all(label_time(window_ts[(b == 1).any(axis=1)], 8) <= vs)


def test_bags_require_tail():
    ts = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    with pytest.raises(AssertionError):
        time_ordered_bags(ts, 8, 2, seed=1, val_start=ts[-1] + H, **VAL)


# ---------------------------------------------------------------- half-samples


@pytest.mark.parametrize("which", ["window_ts", "gapped_ts"])
def test_half_sample(which, request):
    ts = request.getfixturevalue(which)
    h = 8
    mask, vs = half_sample(ts, h, 0.15, 168, seed=21)
    assert vs == tail_start(ts, 0.15)
    tail = np.asarray(ts >= vs)
    for part, origin in ((~tail, ts[0]), (tail, vs)):
        blk = np.asarray((ts[part] - origin) // H) // 168
        sel = pd.Series(mask[part]).groupby(blk).agg(["min", "max"])
        assert (sel["min"] == sel["max"]).all()          # whole blocks
        n_blocks = len(sel)
        assert sel["max"].sum() == max(1, int(np.floor(n_blocks / 2 + 0.5)))

    # bags on the selected rows: validation = selected tail rows, training only from pre-tail rows
    sub = ts[mask]
    b = time_ordered_bags(sub, h, 14, seed=3, val_start=vs, block_origin=ts[0], **VAL)
    assert np.array_equal((b == -1).all(axis=1), np.asarray(sub >= vs))
    train_rows = sub[(b == 1).any(axis=1)]
    assert np.all(train_rows < vs) and np.all(label_time(train_rows, h) <= vs)
    assert set(train_rows) <= set(ts[~tail])

    other, vs2 = half_sample(ts, h, 0.15, 168, seed=22)
    assert vs2 == vs and not np.array_equal(mask, other)
    assert np.array_equal(mask, half_sample(ts, h, 0.15, 168, seed=21)[0])
    assert 0.4 < mask.mean() < 0.6


# ---------------------------------------------------------------- xgb split, bootstrap


def test_xgb_split(gapped_ts):
    ts, h = gapped_ts, 8
    tr, va = xgb_split(ts, h, 0.15)
    vs = tail_start(ts, 0.15)
    tail, purged, elig = roles(ts, h, vs)
    assert np.array_equal(va, np.flatnonzero(tail))
    assert np.array_equal(tr, np.flatnonzero(elig))
    b = time_ordered_bags(ts, h, 2, seed=0, **VAL)
    assert np.array_equal(np.flatnonzero((b == -1).all(axis=1)), va)


def test_stationary_bootstrap_counts(window_ts):
    ts, h = window_ts, 8
    c = stationary_bootstrap_counts(ts, h, 0.15, 24, seed=4)
    tail, purged, elig = roles(ts, h, tail_start(ts, 0.15))
    assert c.shape == (len(ts),) and c.dtype == np.int8
    assert np.all(c[tail] == -1) and np.all(c[purged] == 0)
    assert c[elig].min() >= 0 and c[elig].sum() == elig.sum()
    assert c[elig].max() > 1                           # replication counts, not a 0/1 mask
    # share of rows drawn at least once: expectation 1 - (1 - 1/m)^m ≈ 1 - 1/e ≈ 0.632
    assert 0.5 < (c[elig] > 0).mean() < 0.75
    assert np.array_equal(c, stationary_bootstrap_counts(ts, h, 0.15, 24, seed=4))
    assert not np.array_equal(c, stationary_bootstrap_counts(ts, h, 0.15, 24, seed=5))


def test_stationary_bootstrap_mean_block_length():
    # m = 20000 eligible rows; runs of consecutive drawn rows have mean length ~ mean_block
    ts = pd.date_range("2020-01-01", periods=24000, freq="h", tz="UTC")
    c = stationary_bootstrap_counts(ts, 0, 1 / 6, 50, seed=9)
    elig = c[c >= 0]
    drawn = (elig > 0).astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, drawn]) == 1)
    assert len(starts) > 0
    # with overlap of blocks runs merge, so the mean drawn-run length is at least mean_block / 2
    assert drawn.sum() / len(starts) > 25


# ---------------------------------------------------------------- seeds


def test_stable_seed_basic():
    a = stable_seed(20261108, "BTCUSDT", "2020-01-01T00:00:00+00:00", "dir8", "seed", 0)
    assert a == stable_seed(20261108, "BTCUSDT", "2020-01-01T00:00:00+00:00", "dir8", "seed", 0)
    assert 0 <= a < 2**31
    assert stable_seed(1) != stable_seed("1")
    assert stable_seed("a", "b") != stable_seed("b", "a") != stable_seed("ab")
    seeds = {stable_seed(20261108, "BTCUSDT", r) for r in range(1000)}
    assert len(seeds) == 1000
    with pytest.raises(TypeError):
        stable_seed(1.5)
    # pinned value: changing the hashing scheme changes every fit in the study
    assert stable_seed(20261108, "BTCUSDT", "x", 0) == stable_seed(np.int64(20261108), "BTCUSDT", "x", 0)
    assert stable_seed(0) == 155685518
    assert stable_seed(20261108, "BTCUSDT", "2021-01-01T00:00:00+00:00", "dir8", "seed", 0) == 782372550


def test_stable_seed_independent_of_process():
    code = "from ebmregime.splits import stable_seed; print(stable_seed(20261108, 'ETHUSDT', 'x', 3))"
    outs = set()
    for hs in ("0", "123"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outs.add(int(out.stdout.strip()))
    assert outs == {stable_seed(20261108, "ETHUSDT", "x", 3)}


def test_replicate_plan(prereg):
    cfg = prereg["replicates"]
    tau = pd.Timestamp("2021-01-01", tz="UTC")
    plan = replicate_plan("BTCUSDT", tau, "dir8", cfg)
    assert [p["protocol"] for p in plan] == ["seed"] * 5 + ["half"] * 10
    assert [p["r"] for p in plan] == list(range(5)) + list(range(10))
    assert all(p["sample_seed"] is None for p in plan[:5])
    assert plan[0]["seed"] == stable_seed(cfg["base_seed"], "BTCUSDT", tau.isoformat(), "dir8", "seed", 0)
    assert plan[5]["seed"] == stable_seed(cfg["base_seed"], "BTCUSDT", tau.isoformat(), "dir8", "half", 0)
    all_seeds = [p["seed"] for p in plan] + [p["sample_seed"] for p in plan[5:]]
    assert len(set(all_seeds)) == len(all_seeds)
    assert plan == replicate_plan("BTCUSDT", "2021-01-01", "dir8", cfg)
    other = replicate_plan("ETHUSDT", tau, "dir8", cfg)
    assert not {p["seed"] for p in plan} & {p["seed"] for p in other}
