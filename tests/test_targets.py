"""Targets: hand-computed values, strict validity, no look-ahead, scale invariance, shuffled control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ebmregime.series import forward_mean_strict, hourly_inputs
from ebmregime.targets import build_targets, direction, forward_vol, shuffle_offset, shuffled
from ebmregime.targets.build import _forward_mean_exact

T0 = pd.Timestamp("2025-01-01", tz="UTC")
SHUF = {"min_days": 90, "max_days": 400, "forbid_multiples_of_hours": [24, 168], "seed": 20261108}
PREREG = {"targets": {"horizons": {"primary": 8, "secondary": [24]},
                      "direction": {"drop_zero_returns": True},
                      "vol": {"trailing_hours": 168},
                      "shuffled": SHUF}}


def frame(close, rv=None, valid=None) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(T0, periods=n, freq="1h", name="ts")
    return pd.DataFrame({"close": np.asarray(close, float),
                         "rv": np.ones(n) if rv is None else np.asarray(rv, float),
                         "valid": np.ones(n, bool) if valid is None else np.asarray(valid, bool)},
                        index=idx)


def random_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h = frame(c, rv=rng.lognormal(-9, 0.5, n))
    for col in ("open", "high", "low", "volume"):
        h[col] = rng.lognormal(0, 1, n)
    return h


def same(a: pd.Series, b: pd.Series) -> bool:
    return bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))


# --- hand-computed values -----------------------------------------------------------------------

def test_direction_hand_computed_zero_and_tail():
    #          0    1    2    3    4    5    6
    h = frame([100, 110, 99, 100, 120, 120, 90])
    y = direction(h, 2)
    # t=0: 99/100 <1 ->0; t=1: 100/110 ->0; t=2: 120/99 ->1; t=3: 120/100 ->1; t=4: 90/120 ->0
    assert y.iloc[:5].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert y.iloc[5:].isna().all() and y.name == "y_dir_h2" and y.dtype == "float64"
    h0 = frame([100, 105, 100, 101])
    y0 = direction(h0, 2)                        # t=0: exact zero return -> NaN
    assert np.isnan(y0.iloc[0]) and y0.iloc[1] == 0.0
    assert direction(h0, 2, drop_zero=False).iloc[0] == 0.0


def test_direction_strict_window_masks():
    c = [100, 101, 102, 103, 104, 105, 106, 107]
    v = [True] * 8
    v[3] = False                                 # interior hour of the window for t=1,2 (h=2)
    h = frame(np.where(v, c, np.nan), valid=v)   # prices are NaN on invalid hours
    y = direction(h, 2)
    # t=0 uses rows 1,2 -> ok. t=1 uses 2,3 (3 invalid). t=2 uses 3,4: endpoints 2,4 fine but 3 invalid.
    # t=3 is itself invalid. t=4 uses 5,6 -> ok. t=6,7 have no full window.
    assert y.iloc[0] == 1.0 and np.isnan(y.iloc[1]) and np.isnan(y.iloc[2]) and np.isnan(y.iloc[3])
    assert y.iloc[4] == 1.0 and y.iloc[5] == 1.0 and y.iloc[6:].isna().all()
    # Invalid flag alone (prices present) must also mask: interior hour of the window.
    h2 = frame(c, valid=v)
    y2 = direction(h2, 2)
    assert np.isnan(y2.iloc[2]) and np.isnan(y2.iloc[3]) and y2.iloc[4] == 1.0


def test_forward_vol_hand_computed():
    rv = [1.0, 3.0, 2.0, 4.0, 8.0, 1.0, 1.0]
    h = frame(np.full(7, 100.0), rv=rv)
    y = forward_vol(h, 2, trailing_hours=2)
    # t=1: fwd mean(rv2, rv3) = 3, trail mean(rv0, rv1) = 2 -> log(1.5)
    assert np.isclose(y.iloc[1], np.log(3 / 2))
    # t=2: fwd (4+8)/2 = 6, trail (3+2)/2 = 2.5
    assert np.isclose(y.iloc[2], np.log(6 / 2.5))
    # t=4: fwd (1+1)/2 = 1, trail (4+8)/2 = 6
    assert np.isclose(y.iloc[4], np.log(1 / 6))
    assert np.isnan(y.iloc[0])                   # trailing window incomplete (2 of 2 required)
    assert y.iloc[5:].isna().all() and y.name == "y_vol_h2"


def test_forward_vol_masks():
    rv = np.arange(1.0, 11.0)
    v = np.ones(10, bool)
    v[4] = False
    h = frame(np.full(10, 100.0), rv=rv, valid=v)   # rv itself still present at row 4
    y = forward_vol(h, 2, trailing_hours=2)
    assert np.isnan(y.iloc[2]) and np.isnan(y.iloc[3]) and np.isnan(y.iloc[4])
    assert np.isfinite(y.iloc[1]) and np.isfinite(y.iloc[5])
    rv2 = rv.copy()
    rv2[7] = np.nan                                  # missing rv inside the forward window
    y2 = forward_vol(frame(np.full(10, 100.0), rv=rv2), 2, trailing_hours=2)
    assert np.isnan(y2.iloc[5]) and np.isnan(y2.iloc[6]) and np.isnan(y2.iloc[7])
    rv3 = rv.copy()
    rv3[[2, 3]] = 0.0                                # zero forward mean -> NaN, not -inf
    assert np.isnan(forward_vol(frame(np.full(10, 100.0), rv=rv3), 2, trailing_hours=2).iloc[1])


def test_forward_mean_exact_matches_series_helper():
    x = pd.Series(np.random.default_rng(7).lognormal(-9, 1, 300))
    x[[10, 150]] = np.nan
    for hz in (1, 8, 24):
        a, b = _forward_mean_exact(x, hz), forward_mean_strict(x, hz)
        assert a.isna().equals(b.isna())
        assert np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True)


# --- no look-ahead ------------------------------------------------------------------------------

@pytest.mark.parametrize("hz", [8, 24])
def test_no_lookahead_and_horizon_reaches(hz):
    rng = np.random.default_rng(1)
    base = random_frame(600, seed=2)
    yd0, yv0 = direction(base, hz), forward_vol(base, hz)
    for t in rng.integers(200, 600 - hz - 2, size=15):
        p = base.copy()
        k = slice(t + hz + 1, None)
        n = len(p) - (t + hz + 1)
        for col in ("close", "open", "high", "low", "volume", "rv"):
            p.iloc[k, p.columns.get_loc(col)] = p[col].iloc[k].to_numpy() * rng.lognormal(0, 1, n)
        p.iloc[k, p.columns.get_loc("valid")] = rng.random(n) > 0.3
        assert same(direction(p, hz).iloc[: t + 1], yd0.iloc[: t + 1])
        assert same(forward_vol(p, hz).iloc[: t + 1], yv0.iloc[: t + 1])

        # The horizon really reaches t+h: flip the sign of the return via C_{t+h}.
        q = base.copy()
        ct = q["close"].iloc[t]
        r = np.log(q["close"].iloc[t + hz] / ct)
        q.iloc[t + hz, q.columns.get_loc("close")] = ct * np.exp(-r)
        assert direction(q, hz).iloc[t] == 1.0 - yd0.iloc[t]
        # ... and rv at t+h enters the forward mean.
        q.iloc[t + hz, q.columns.get_loc("rv")] *= 50
        assert forward_vol(q, hz).iloc[t] > yv0.iloc[t] + 0.1
        # ... and an invalid flag anywhere in t+1..t+h masks both.
        for j in (t + 1, t + hz):
            w = base.copy()
            w.iloc[j, w.columns.get_loc("valid")] = False
            assert np.isnan(direction(w, hz).iloc[t]) and np.isnan(forward_vol(w, hz).iloc[t])


# --- scale invariance (through the real hourly-input builder) -------------------------------------

def test_scale_invariance_through_hourly_inputs():
    rng = np.random.default_rng(3)
    n15 = 4 * 400
    c15 = 50 * np.exp(np.cumsum(rng.normal(0, 0.004, n15)))
    i15 = pd.date_range(T0, periods=n15, freq="15min", name="ts")
    v15 = rng.random(n15) > 0.002
    g = pd.DataFrame({"close": c15, "valid": v15}, index=i15)
    i1 = pd.date_range(T0, periods=n15 // 4, freq="1h", name="ts")
    hv = pd.Series(v15, index=i15).groupby(i15.floor("1h")).all().reindex(i1).to_numpy()
    c1 = np.where(hv, c15[3::4], np.nan)
    bars = pd.DataFrame({"close": c1, "valid": hv, "volume_suspect": False}, index=i1)
    a = hourly_inputs(g, bars)
    g2, bars2 = g.copy(), bars.copy()
    g2["close"] *= 37.3
    bars2["close"] *= 37.3
    b = hourly_inputs(g2, bars2)
    for hz in (8, 24):
        assert same(direction(a, hz), direction(b, hz))
        ya, yb = forward_vol(a, hz), forward_vol(b, hz)
        assert ya.notna().sum() > 50
        assert np.allclose(ya, yb, equal_nan=True, rtol=0, atol=1e-10)


# --- shuffled control ---------------------------------------------------------------------------

def test_shuffle_offset_deterministic_bounds_and_forbidden_multiples():
    syms = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"] + [f"SYM{i}" for i in range(40)]
    offs = {s: shuffle_offset(100_000, s, SHUF) for s in syms}
    assert offs == {s: shuffle_offset(100_000, s, SHUF) for s in syms}
    assert len(set(offs.values())) > 0.9 * len(syms)
    assert offs["BTCUSDT"] != offs["SOLUSDT"] and offs["BTCUSDT"] != offs["ETHUSDT"]
    for k in offs.values():
        assert 90 * 24 <= k <= 400 * 24 and k % 24 and k % 168
    assert shuffle_offset(100_000, "BTCUSDT", {**SHUF, "seed": 1}) != offs["BTCUSDT"]
    # n_rows constraint: the offset is always below n_rows.
    for s in syms[:10]:
        assert shuffle_offset(2200, s, SHUF) < 2200
    with pytest.raises(ValueError):
        shuffle_offset(2161, "BTCUSDT", SHUF)      # only 2160 = 90*24 fits, and it is forbidden


def test_shuffled_restriction_multiset_and_mapping():
    rng = np.random.default_rng(4)
    n = 500
    idx = pd.date_range(T0, periods=n, freq="1h", name="ts")
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx, name="y_dir_h8")
    y[rng.random(n) < 0.1] = np.nan
    start = str(idx[100])
    s = shuffled(y, start, 37)
    assert s.name == "y_shuf_h8"
    assert s.iloc[:100].isna().all()
    r = y.iloc[100:].to_numpy()
    for i in (0, 5, 362, 363, 399):
        assert same(pd.Series([s.iloc[100 + i]]), pd.Series([r[(i + 37) % 400]]))
    # Multiset of labels (and of NaNs) on the restricted range is identical.
    assert sorted(s.iloc[100:].dropna()) == sorted(y.iloc[100:].dropna())
    assert s.iloc[100:].isna().sum() == y.iloc[100:].isna().sum()
    with pytest.raises(ValueError):
        shuffled(y, start, 400)


def test_shuffled_lag1_autocorrelation_up_to_wrap_point():
    rng = np.random.default_rng(5)
    n, k = 3000, 977
    idx = pd.date_range(T0, periods=n, freq="1h", name="ts")
    # Persistent binary series (overlapping-label-like autocorrelation).
    r = (np.convolve(rng.normal(size=n + 7), np.ones(8), "valid") > 0).astype(float)
    y = pd.Series(r, index=idx)
    s = shuffled(y, str(idx[0]), k).to_numpy()
    rc = r - r.mean()
    sc = s - s.mean()                            # same mean: same multiset
    circ = np.sum(rc * np.roll(rc, -1))          # all N circular lag-1 pairs of the original
    lin = np.sum(sc[:-1] * sc[1:])               # N-1 linear pairs of the shuffled series
    assert np.isclose(lin, circ - rc[k - 1] * rc[k])   # the only missing pair is the wrap point
    ac_orig = circ / np.sum(rc**2)
    ac_shuf = lin / np.sum(sc**2)
    assert abs(ac_orig - ac_shuf) <= 5.0 / n
    assert ac_orig > 0.5                         # the test series really is autocorrelated


def test_build_targets_and_shuffled_uncorrelated_with_direction():
    n = 20_000
    h = random_frame(n, seed=6)
    start = str(h.index[1000].date())
    data_cfg = {"symbols": {"BTCUSDT": {"analysis_start": start}}}
    df = build_targets(h, "BTCUSDT", PREREG, data_cfg)
    assert list(df.columns) == ["y_dir_h8", "y_vol_h8", "y_shuf_h8", "y_dir_h24", "y_vol_h24", "y_shuf_h24"]
    offs = df.attrs["shuffle_offset_hours"]
    assert set(offs) == {8, 24} and offs[8] == offs[24]
    n_r = int((h.index >= pd.Timestamp(start, tz="UTC")).sum())
    assert offs[8] == shuffle_offset(n_r, "BTCUSDT", PREREG["targets"]["shuffled"])
    assert df["y_shuf_h8"].loc[: pd.Timestamp(start, tz="UTC") - pd.Timedelta("1h")].isna().all()
    assert same(df["y_dir_h8"], direction(h, 8)) and same(df["y_vol_h24"], forward_vol(h, 24, 168))
    for hz in (8, 24):
        d = df[[f"y_dir_h{hz}", f"y_shuf_h{hz}"]].dropna()
        assert len(d) > 15_000
        assert abs(d.corr().iloc[0, 1]) < 3 * np.sqrt(hz / len(d))   # ~3 SE with overlap
    assert df.attrs["shuffle_offset_hours"] == build_targets(h, "BTCUSDT", PREREG, data_cfg).attrs[
        "shuffle_offset_hours"]
