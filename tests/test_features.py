"""Tests for ebmregime.features (PLAN §5.1). Synthetic canonical hourly frames only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config
from ebmregime.features import FEATURE_NAMES, NOMINAL, build_features, rolling_pct_rank
from ebmregime.series import MIN_FRAC, MIN_FRAC_HISTORY, trailing_mean

T0 = pd.Timestamp("2025-01-01", tz="UTC")  # a Wednesday
CFG = load_config("features")
CONT = [f for f in FEATURE_NAMES if f not in NOMINAL]
PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]


# ----------------------------------------------------------------------------- frame builders
def canon(close, rv, *, high=None, low=None, qv=None, tbq=None, valid=None, suspect=None,
          start=T0) -> pd.DataFrame:
    """A canonical hourly input frame (as series.hourly_inputs returns) from explicit columns."""
    close = np.asarray(close, float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="1h", name="ts")
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    suspect = np.zeros(n, bool) if suspect is None else np.asarray(suspect, bool)
    high = close if high is None else np.asarray(high, float)
    low = close if low is None else np.asarray(low, float)
    qv = np.full(n, 1e6) if qv is None else np.asarray(qv, float)
    tbq = qv / 2 if tbq is None else np.asarray(tbq, float)
    rv = np.broadcast_to(np.asarray(rv, float), (n,))

    def px(a):
        return np.where(valid, a, np.nan)

    return pd.DataFrame({
        "open": px(close), "high": px(high), "low": px(low), "close": px(close),
        "volume": qv / 100, "quote_volume": qv, "taker_buy_base": tbq / 100, "taker_buy_quote": tbq,
        "n_trades": np.full(n, 10.0), "valid": valid, "volume_suspect": suspect,
        "volume_ok": valid & ~suspect, "rv": np.where(valid, rv, np.nan),
    }, index=idx)


def random_frame(n: int, seed: int, start=T0, p_invalid=0.004, p_suspect=0.004,
                 outages: tuple[tuple[int, int], ...] = ()) -> pd.DataFrame:
    """Random-walk canonical frame with sprinkled invalid and volume_suspect hours, plus optional
    outage blocks ``(first_row, length)`` of consecutive invalid hours."""
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))          # slowly varying vol level
    r = rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    rv = (rng.normal(0, 1, (n, 4)) ** 2).sum(axis=1) * vol**2 / 4
    qv = np.exp(rng.normal(13, 0.6, n))
    tbq = qv * rng.uniform(0.3, 0.7, n)
    valid = rng.random(n) > p_invalid
    for a, length in outages:
        valid[a: a + length] = False
    suspect = valid & (rng.random(n) < p_suspect)
    h = canon(close, rv, high=high, low=low, qv=qv, tbq=tbq, valid=valid, suspect=suspect, start=start)
    h["open"] = np.where(valid, open_, np.nan)
    return h


# ----------------------------------------------------------------------------- brute force
def brute_features(h: pd.DataFrame, partner: pd.DataFrame | None, cfg: dict = CFG) -> pd.DataFrame:
    """Row-by-row reference implementation written directly from the spec (slow)."""
    spec = {f["name"]: f for f in cfg["features"]}
    sh = cfg["sigma_hours"]
    need = lambda k: math.ceil(MIN_FRAC * k)             # window rule  # noqa: E731
    need_hist = lambda n: math.ceil(MIN_FRAC_HISTORY * n)  # history rule  # noqa: E731

    def tmean(x, t, k):
        w = x[max(0, t - k + 1): t + 1]
        w = w[~np.isnan(w)]
        return w.mean() if len(w) >= need(k) else np.nan

    def arrays(f):
        v = f["valid"].to_numpy()
        return (np.where(v, f["close"], np.nan), np.where(v, f["high"], np.nan),
                np.where(v, f["low"], np.nan), np.where(v, f["rv"], np.nan))

    def rets(c, rv, k):
        out = np.full(len(c), np.nan)
        for t in range(k, len(c)):
            s = math.sqrt(tmean(rv, t, sh)) if not np.isnan(tmean(rv, t, sh)) else np.nan
            out[t] = np.log(c[t] / c[t - k]) / (s * math.sqrt(k))
        return out

    n = len(h)
    C, H, L, RV = arrays(h)
    vok = (h["volume_ok"] & h["valid"]).to_numpy()
    QV = np.where(vok, h["quote_volume"], np.nan)
    TB = np.where(vok, h["taker_buy_quote"], np.nan)
    o = {name: np.full(n, np.nan) for name in FEATURE_NAMES}
    for name in ("ret_1h", "ret_4h", "ret_24h", "ret_168h"):
        o[name] = rets(C, RV, spec[name]["k"])
    rv168 = np.array([tmean(RV, t, 168) for t in range(n)])
    sig = np.sqrt(np.array([tmean(RV, t, sh) for t in range(n)]))
    for t in range(n):
        o["rv_ratio_4h"][t] = np.log(tmean(RV, t, 4) / rv168[t])
        o["rv_ratio_24h"][t] = np.log(tmean(RV, t, 24) / rv168[t])
        hh = spec["rv_level_pct"]["history_hours"]
        w = rv168[max(0, t - hh + 1): t + 1]
        w = w[~np.isnan(w)]
        if not np.isnan(rv168[t]) and len(w) >= need_hist(hh):
            o["rv_level_pct"][t] = (w <= rv168[t]).mean()
        # qv_z_1h
        x = np.log(QV)
        ref = x[max(0, t - 168): t]
        ref = ref[~np.isnan(ref)]
        if len(ref) >= need(168) and not np.isnan(x[t]) and ref.std(ddof=1) > 0:
            o["qv_z_1h"][t] = (x[t] - ref.mean()) / ref.std(ddof=1)
        o["range_1h"][t] = np.log(H[t] / L[t]) / sig[t]
        for name, k in (("pos_range_24h", 24), ("pos_range_168h", 168)):
            wh, wl = H[max(0, t - k + 1): t + 1], L[max(0, t - k + 1): t + 1]
            if (~np.isnan(wh)).sum() >= need(k) and not np.isnan(C[t]):
                hk, lk = np.nanmax(wh), np.nanmin(wl)
                o[name][t] = 0.5 if hk == lk else (C[t] - lk) / (hk - lk)
        tq, qq = TB[max(0, t - 3): t + 1], QV[max(0, t - 3): t + 1]
        if (~np.isnan(qq)).sum() == 4:
            o["taker_imb_4h"][t] = tq.sum() / qq.sum() - 0.5
    # qv_z_24h
    y = np.array([np.log(tmean(QV, t, 24) * 24) for t in range(n)])
    for t in range(n):
        ref = np.array([y[t - 24 * j] if t - 24 * j >= 0 else np.nan for j in range(1, 31)])
        ref = ref[~np.isnan(ref)]
        if len(ref) >= need_hist(30) and not np.isnan(y[t]) and ref.std(ddof=1) > 0:
            o["qv_z_24h"][t] = (y[t] - ref.mean()) / ref.std(ddof=1)
    o["hour"] = h.index.hour.to_numpy().astype(np.int64)
    o["dow"] = h.index.day_of_week.to_numpy().astype(np.int64)
    if partner is not None:
        pc, _, _, prv = arrays(partner)
        pr = pd.Series(rets(pc, prv, 4), index=partner.index)
        o["xret_4h"] = pr.reindex(h.index).to_numpy()
    f = pd.DataFrame(o, index=h.index)[FEATURE_NAMES]
    f["features_ok"] = h["valid"].to_numpy() & np.isfinite(f.to_numpy(float)).all(axis=1)
    return f


def assert_features_close(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), col
        np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=col)


# ----------------------------------------------------------------------------- API
def test_feature_names_match_config():
    assert FEATURE_NAMES == [f["name"] for f in CFG["features"]]
    assert NOMINAL == [f["name"] for f in CFG["features"] if f.get("nominal")]


def test_output_layout_and_dtypes():
    h = random_frame(400, 0)
    f = build_features(h, h, CFG)
    assert list(f.columns) == FEATURE_NAMES + ["features_ok"]
    assert f.index.equals(h.index)
    assert f["hour"].dtype == np.int64 and f["dow"].dtype == np.int64
    assert f["features_ok"].dtype == bool
    assert not np.isinf(f[CONT].to_numpy()).any()


def test_gap_constants():
    assert (MIN_FRAC, MIN_FRAC_HISTORY) == (0.90, 0.50)  # D-43; the hand tests below assume these


def test_matches_brute_force_on_random_frame():
    # outages of 10-30 h cross the window rule (17+ of 168 missing) and thin the histories
    h = random_frame(4700, 1, outages=((1000, 20), (2000, 12), (2100, 10), (3500, 30), (4300, 17)))
    p = random_frame(4800, 2, start=T0 + pd.Timedelta(hours=5))  # partner starts later
    f = build_features(h, p, CFG)
    b = brute_features(h, p)
    assert_features_close(f, b)
    # every feature is non-trivially populated at the end of the sample
    tail = f.iloc[-150:]
    assert tail[FEATURE_NAMES].notna().mean().min() > 0.8
    assert f["features_ok"].iloc[-150:].mean() > 0.8
    assert f["xret_4h"].iloc[:5].isna().all()  # no partner row yet


# ----------------------------------------------------------------------------- hand-computed values
def test_returns_and_range_by_hand():
    n = 200
    logc = 0.001 * np.arange(n) ** 1.3
    hi, lo = np.exp(logc + 0.01), np.exp(logc - 0.01)
    f = build_features(canon(np.exp(logc), 1e-4, high=hi, low=lo), None, CFG)  # sigma = 0.01
    t = n - 1
    for k in (1, 4, 24, 168):
        assert np.isclose(f[f"ret_{k}h"].iloc[t], (logc[t] - logc[t - k]) / (0.01 * math.sqrt(k)), rtol=1e-12)
    assert np.isclose(f["range_1h"].iloc[t], 0.02 / 0.01, rtol=1e-12)
    # sigma needs ceil(0.90*168) = 152 rv values: row 150 has 151 -> NaN, row 151 -> value
    assert np.isnan(f["ret_1h"].iloc[150]) and np.isfinite(f["ret_1h"].iloc[151])
    assert f["xret_4h"].isna().all() and not f["features_ok"].any()  # partner=None


def test_rv_ratio_by_hand():
    rv = np.full(200, 1e-4)
    rv[-4:] = 4e-4
    f = build_features(canon(np.full(200, 100.0), rv), None, CFG)
    assert np.isclose(f["rv_ratio_4h"].iloc[-1], np.log(4 * 168 / 180))
    assert np.isclose(f["rv_ratio_24h"].iloc[-1], np.log(1.5 * 168 / 180))


def test_rolling_pct_rank_by_hand():
    x = pd.Series([1.0, np.nan, 3.0, 2.0, 2.0, np.nan, 5.0, 1.0])
    r = rolling_pct_rank(x, 4, 3)
    exp = [np.nan, np.nan, np.nan, 2 / 3, 2 / 3, np.nan, 1.0, 1 / 3]
    np.testing.assert_allclose(r.to_numpy(), exp, equal_nan=True)


def test_rolling_pct_rank_equals_brute_force():
    rng = np.random.default_rng(7)
    x = pd.Series(np.round(rng.normal(size=3000), 1))  # rounding creates many ties
    x[rng.random(3000) < 0.04] = np.nan
    x[1000:1030] = np.nan                                # a long gap -> min_count binds
    w, m = 200, 190
    r = rolling_pct_rank(x, w, m).to_numpy()
    xv = x.to_numpy()
    exp = np.full(len(x), np.nan)
    for t in range(len(x)):
        win = xv[max(0, t - w + 1): t + 1]
        win = win[~np.isnan(win)]
        if not np.isnan(xv[t]) and len(win) >= m:
            exp[t] = (win <= xv[t]).sum() / len(win)
    assert np.array_equal(np.isnan(r), np.isnan(exp))
    np.testing.assert_allclose(r, exp, rtol=1e-12, equal_nan=True)
    assert np.isfinite(r).sum() > 2000


def test_qv_z_1h_by_hand():
    x = np.r_[np.tile([0.0, 1.0], 84), 2.0]  # 168 reference values then the current one
    f = build_features(canon(np.full(169, 100.0), 1e-4, qv=np.exp(x)), None, CFG)
    sd = math.sqrt(168 * 0.25 / 167)
    assert np.isclose(f["qv_z_1h"].iloc[168], 1.5 / sd)
    assert np.isnan(f["qv_z_1h"].iloc[151])     # 151 < ceil(0.90*168) = 152 reference values
    assert np.isfinite(f["qv_z_1h"].iloc[152])


def test_qv_z_24h_by_hand():
    days = np.arange(31)
    level = np.where(days == 30, 3.0, days % 2)            # log hourly QV per day
    qv = np.exp(np.repeat(level, 24))
    f = build_features(canon(np.full(len(qv), 100.0), 1e-4, qv=qv), None, CFG)
    sd = math.sqrt(30 * 0.25 / 29)
    assert np.isclose(f["qv_z_24h"].iloc[-1], (3.0 - 0.5) / sd)


def test_pos_range_by_hand():
    n = 200
    c = np.full(n, 100.0)
    hi, lo = c + 1, c - 1
    hi[-10], lo[-5], c[-1] = 110.0, 90.0, 105.0
    hi[-1], lo[-1] = 106.0, 104.0
    f = build_features(canon(c, 1e-4, high=hi, low=lo), None, CFG)
    assert np.isclose(f["pos_range_24h"].iloc[-1], 0.75)
    assert np.isclose(f["pos_range_168h"].iloc[-1], 0.75)
    flat = build_features(canon(np.full(n, 100.0), 1e-4), None, CFG)
    assert (flat["pos_range_24h"].iloc[23:] == 0.5).all()
    assert np.isnan(flat["pos_range_24h"].iloc[20])  # 21 < ceil(0.90*24) = 22 bars
    assert flat["pos_range_24h"].iloc[21] == 0.5


def test_taker_imb_by_hand():
    qv = np.r_[np.ones(10), 1.0, 2.0, 3.0, 4.0]
    tbq = np.r_[np.ones(10) / 2, 1.0, 1.0, 1.0, 1.0]
    f = build_features(canon(np.full(14, 100.0), 1e-4, qv=qv, tbq=tbq), None, CFG)
    assert np.isclose(f["taker_imb_4h"].iloc[-1], 4 / 10 - 0.5)
    assert np.isclose(f["taker_imb_4h"].iloc[5], 0.0)
    assert np.isnan(f["taker_imb_4h"].iloc[2])  # fewer than 4 hours


def test_time_features():
    f = build_features(canon(np.full(30, 100.0), 1e-4), None, CFG)
    assert f["hour"].iloc[0] == 0 and f["dow"].iloc[0] == 2  # 2025-01-01 is a Wednesday
    assert f["hour"].iloc[25] == 1 and f["dow"].iloc[25] == 3


def test_xret_is_partner_ret_4h_aligned():
    h = random_frame(300, 3)
    p = random_frame(300, 4, start=T0 + pd.Timedelta(hours=50))
    fh, fp = build_features(h, p, CFG), build_features(p, None, CFG)
    exp = fp["ret_4h"].reindex(h.index)
    pd.testing.assert_series_equal(fh["xret_4h"], exp, check_names=False)
    assert fh["xret_4h"].iloc[:50].isna().all()


# ----------------------------------------------------------------------------- gap tolerance
def _base(n=1000):
    rng = np.random.default_rng(11)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return c, c * 1.003, c / 1.003, rng.uniform(5e-5, 2e-4, n), np.exp(rng.normal(13, 0.5, n))


def test_one_invalid_hour_inside_window_still_gives_values():
    c, hi, lo, rv, qv = _base()
    valid = np.ones(1000, bool)
    valid[800] = False
    f = build_features(canon(c, rv, high=hi, low=lo, qv=qv, valid=valid), None, CFG)
    t = 899  # the 168h windows ending here (732..899) contain the invalid row 800
    # (rv_level_pct needs ~4104 h of history and is NaN in this short frame; see the brute-force test)
    for col in ("ret_1h", "ret_24h", "ret_168h", "rv_ratio_4h", "rv_ratio_24h", "qv_z_1h", "qv_z_24h",
                "range_1h", "pos_range_24h", "pos_range_168h", "taker_imb_4h"):
        assert np.isfinite(f[col].iloc[t]), col
    assert f["features_ok"].iloc[t] == False  # noqa: E712  (partner=None -> xret_4h NaN)
    assert f.loc[f.index[800], CONT].drop("xret_4h").isna().any()  # the invalid row itself
    assert np.isnan(f["ret_168h"].iloc[800 + 168])                 # C_{t-168} is the invalid hour
    assert np.isfinite(f["ret_168h"].iloc[800 + 169])
    assert np.isnan(f["ret_1h"].iloc[801]) and np.isfinite(f["ret_1h"].iloc[802])
    # qv_z_24h: the 24h sum tolerates up to 2 missing hours out of 24 (ceil(0.90*24) = 22)
    assert np.isfinite(f["qv_z_24h"].iloc[810])


def test_window_rule_boundary_16_ok_17_nan():
    c, hi, lo, rv, qv = _base()
    t = 880                                    # 168h windows ending at t (and t-1) contain 800..816
    cols = ["ret_1h", "ret_4h", "rv_ratio_4h", "range_1h", "pos_range_168h", "qv_z_1h"]
    valid = np.ones(1000, bool)
    valid[800:817] = False                     # 17 invalid -> 151 < 152 present
    f = build_features(canon(c, rv, high=hi, low=lo, qv=qv, valid=valid), None, CFG)
    for col in cols:
        assert np.isnan(f[col].iloc[t]), col
    assert np.isfinite(f["pos_range_24h"].iloc[t])          # the 24h window is clean
    valid[816] = True                          # 16 invalid -> 152 present: values again
    f = build_features(canon(c, rv, high=hi, low=lo, qv=qv, valid=valid), None, CFG)
    assert np.isfinite(f.loc[f.index[t], cols].to_numpy(float)).all()


@pytest.mark.parametrize("outages, expect_rv168_nan", [(((3000, 10),), 0), (((3000, 24), (4000, 24)), 318)])
def test_long_outage_does_not_blank_rv_level_pct(outages, expect_rv168_nan):
    """History rule: rv_level_pct is NaN only where RV_168 itself is NaN. Two 24h outages blank
    2 x 159 = 318 RV_168 values, more than the 216 the old 0.95 rule allowed in a 4320h history."""
    h = random_frame(6500, 12, p_invalid=0.0, p_suspect=0.0, outages=outages)
    f = build_features(h, None, CFG)
    rv168 = trailing_mean(h["rv"].where(h["valid"]), 168)
    warm = 2400                                # RV_168 from row 151, + 2159 more values needed
    assert np.isnan(f["rv_level_pct"].iloc[warm - 100])
    assert int(rv168.iloc[warm:].isna().sum()) == expect_rv168_nan
    np.testing.assert_array_equal(f["rv_level_pct"].iloc[warm:].isna(), rv168.iloc[warm:].isna())


def test_volume_suspect_excluded_from_volume_features_only():
    c, hi, lo, rv, qv = _base()
    p = canon(c * 1.1, rv)
    suspect = np.zeros(1000, bool)
    suspect[850] = True
    clean = build_features(canon(c, rv, high=hi, low=lo, qv=qv), p, CFG)
    qv_bad = qv.copy()
    qv_bad[850] = 1e30
    f = build_features(canon(c, rv, high=hi, low=lo, qv=qv_bad, tbq=qv / 2, suspect=suspect), p, CFG)
    # price features are untouched by the flag
    price = [x for x in FEATURE_NAMES if x not in ("qv_z_1h", "qv_z_24h", "taker_imb_4h")]
    pd.testing.assert_frame_equal(f[price], clean[price])
    # volume features ignore the suspect hour's value
    assert np.isnan(f["qv_z_1h"].iloc[850])
    # the 24h sum tolerates one missing hour, so qv_z_24h stays defined but ignores the value
    assert np.isfinite(f["qv_z_24h"].iloc[850])
    qv_other = qv.copy()
    qv_other[850] = 1e-30
    g = build_features(canon(c, rv, high=hi, low=lo, qv=qv_other, tbq=qv / 2, suspect=suspect), p, CFG)
    pd.testing.assert_frame_equal(f, g, check_exact=True)
    assert f["taker_imb_4h"].iloc[850:854].isna().all() and np.isfinite(f["taker_imb_4h"].iloc[854])
    assert np.isfinite(f["qv_z_1h"].iloc[851:].to_numpy()).all()
    assert abs(f["qv_z_1h"].iloc[851:]).max() < 10                # 1e30 did not leak into the stats
    assert np.isfinite(f["qv_z_24h"].iloc[851:].to_numpy()).all()
    assert abs(f["qv_z_24h"].iloc[851:]).max() < 10
    # rv_level_pct needs ~4104 h of history, so features_ok is False everywhere in this short frame
    others = [x for x in FEATURE_NAMES if x != "rv_level_pct"]
    assert np.isfinite(f[others].iloc[854].to_numpy(float)).all()
    assert np.isfinite(clean[others].iloc[850].to_numpy(float)).all()


# ----------------------------------------------------------------------------- property tests
def _perturb_after(df: pd.DataFrame, t: int, rng: np.random.Generator) -> pd.DataFrame:
    """Scramble every column at rows > t."""
    d = df.copy()
    m = len(d) - t - 1
    if m <= 0:
        return d
    sl = slice(t + 1, None)
    for col in PRICE_COLS:
        d.iloc[sl, d.columns.get_loc(col)] = d[col].to_numpy()[t + 1:] * rng.uniform(0.5, 2.0, m)
    for col in VOLUME_COLS + ["n_trades"]:
        d.iloc[sl, d.columns.get_loc(col)] = rng.uniform(0, 1e9, m)
    d.iloc[sl, d.columns.get_loc("rv")] = rng.uniform(0, 1e-2, m)
    for col in ("valid", "volume_suspect", "volume_ok"):
        d.iloc[sl, d.columns.get_loc(col)] = rng.random(m) < 0.5
    return d


def test_no_lookahead():
    h = random_frame(4700, 5)
    p = random_frame(4700, 6, start=T0 - pd.Timedelta(hours=3))
    base = build_features(h, p, CFG)
    rng = np.random.default_rng(99)
    ts = [0, 200, 1000, 4400] + sorted(rng.integers(4350, 4699, 3).tolist())
    for t in ts:
        hp = _perturb_after(h, t, rng)
        pt = p.index.get_loc(h.index[t])
        pp = _perturb_after(p, pt, rng)
        f = build_features(hp, pp, CFG)
        pd.testing.assert_frame_equal(f.iloc[: t + 1], base.iloc[: t + 1], check_exact=True)
        # and truncating the future entirely gives the same rows
        g = build_features(h.iloc[: t + 1], p.iloc[: pt + 1], CFG)
        pd.testing.assert_frame_equal(g, base.iloc[: t + 1], check_exact=True)
    assert base["features_ok"].iloc[4400:].mean() > 0.8  # the checked rows are populated


@pytest.mark.parametrize("c, exact", [(4.0, True), (3.7, False)])
def test_price_scale_invariance(c, exact):
    h = random_frame(4600, 8)
    p = random_frame(4600, 9)
    base = build_features(h, p, CFG)
    hs, ps = h.copy(), p.copy()
    hs[PRICE_COLS] *= c
    ps[PRICE_COLS] *= c   # rv is a variance of log returns: unchanged by price scaling
    f = build_features(hs, ps, CFG)
    if exact:
        pd.testing.assert_frame_equal(f, base, check_exact=True)
    else:
        assert_features_close(f, base)


@pytest.mark.parametrize("c", [8.0, 123.4])
def test_volume_scale_invariance(c):
    h = random_frame(4600, 10)
    base = build_features(h, None, CFG)
    hs = h.copy()
    hs[VOLUME_COLS] *= c
    f = build_features(hs, None, CFG)
    # log(c * QV) = log c + log QV is not bit-exact, so compare with a tolerance
    assert_features_close(f, base)
    assert f["qv_z_24h"].notna().sum() > 3000
