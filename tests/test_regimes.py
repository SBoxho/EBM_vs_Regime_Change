"""Causal regimes R1/R2/R3 (PLAN §8, D-20, D-43, D-44) on synthetic data with hand-computed values.

Gap tolerance (D-43): daily aggregates need ceil(0.90*24) = 22 hours; multi-day windows over daily
values need ceil(0.50*n) (s: 15 of 30, T: 45 of 90, ILLIQ: 15 of 30, dv_z: 183 of 365); the
percentile histories need 365 defined values.
"""

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config
from ebmregime.regimes import (daily_inputs, daily_regimes, primary_r1, regime_change_covariates,
                               to_hourly, window_shares)

CFG = {
    "primary": "r1_vol_2y",
    "r1_vol": {"rv_days": 30, "min_history_days": 365, "cuts": [0.3333333333, 0.6666666667]},
    "r1_vol_2y": {"rv_days": 30, "history_days": 730, "min_history_days": 365,
                  "cuts": [0.3333333333, 0.6666666667]},
    "r2_trend": {"days": 90, "t_threshold": 1.0},
    "r3_liq": {"days": 30, "min_history_days": 365, "cuts": [0.3333333333, 0.6666666667]},
    "r3_dv_z": {"history_days": 365},
}
D0 = pd.Timestamp("2021-01-01", tz="UTC")
STATES = ["r1_vol", "r1_vol_2y", "r2_trend", "r3_liq"]
# s (30 days, >= 15) is first defined at day 14; at day d it has d - 13 values -> 365 at d = 378.
FIRST_P = 14 + 364


def hourly_frame(n_hours, start=D0, rv=1.0, close=100.0, qv=10.0):
    idx = pd.date_range(start, periods=n_hours, freq="1h", name="ts")
    h = pd.DataFrame(index=idx)
    h["rv"] = np.broadcast_to(rv, n_hours).astype(float)
    h["close"] = np.broadcast_to(close, n_hours).astype(float)
    h["valid"] = True
    h["quote_volume"] = np.broadcast_to(qv, n_hours).astype(float)
    h["volume_ok"] = True
    return h


def daily_frame(n, RV=1.0, r=0.0, DV=1.0):
    idx = pd.date_range(D0, periods=n, freq="D", name="day")
    return pd.DataFrame({"RV": np.broadcast_to(RV, n).astype(float),
                         "r": np.broadcast_to(r, n).astype(float),
                         "DV": np.broadcast_to(DV, n).astype(float)}, index=idx)


def random_hourly(n_days, seed):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, n_days))).repeat(24) * 1e-3
    r = rng.normal(0, 1, n) * vol
    h = hourly_frame(n, rv=r ** 2 * rng.uniform(0.5, 1.5, n), close=100 * np.exp(np.cumsum(r)),
                     qv=np.exp(rng.normal(10, 0.5, n)))
    return h


# --- config -----------------------------------------------------------------------------------

def test_primary_r1_from_config_and_real_prereg_section_is_accepted():
    assert primary_r1(CFG) == "r1_vol_2y"
    with pytest.raises(ValueError):
        primary_r1({**CFG, "primary": "r2_trend"})
    cfg = load_config("prereg")["regimes"]                  # has the scalar key "primary"
    reg = daily_regimes(daily_frame(400, RV=np.arange(1.0, 401)), cfg)
    assert reg.attrs["primary_r1"] == primary_r1(cfg) == "r1_vol_2y"


# --- daily aggregation ------------------------------------------------------------------------

def test_daily_inputs_aggregation_and_22_of_24_tolerance():
    h = hourly_frame(72, rv=2.0)
    h.loc[h.index[0:24], "rv"] = 1.0
    h.loc[h.index[[30, 40]], "rv"] = np.nan                    # day 1: 22 of 24 -> mean*24
    h.loc[h.index[[50, 60, 70]], "rv"] = np.nan                # day 2: 21 of 24 -> NaN
    h.loc[h.index[23], "close"] = 110.0                        # day 0 close
    h.loc[h.index[47], "close"] = 121.0                        # day 1 close
    h.loc[h.index[71], "valid"] = False                        # day 2 close invalid -> NaN
    qv = np.arange(1.0, 73.0)
    h["quote_volume"] = qv
    h.loc[h.index[[5, 6]], "volume_ok"] = False                # day 0: 22 ok hours
    h.loc[h.index[[30, 31, 32]], "volume_ok"] = False          # day 1: 21 ok hours -> NaN
    d = daily_inputs(h)

    assert list(d.index) == list(pd.date_range(D0, periods=3, freq="D"))
    assert str(d.index.tz) == "UTC"
    assert d["RV"].iloc[0] == pytest.approx(24.0)
    assert d["RV"].iloc[1] == pytest.approx(48.0)
    assert np.isnan(d["RV"].iloc[2])
    assert d["n_rv"].tolist() == [24, 22, 21]
    assert d["close"].iloc[0] == 110.0 and d["close"].iloc[1] == 121.0
    assert np.isnan(d["close"].iloc[2])
    assert np.isnan(d["r"].iloc[0]) and np.isnan(d["r"].iloc[2])
    assert d["r"].iloc[1] == pytest.approx(np.log(1.1))
    assert d["DV"].iloc[0] == pytest.approx(np.delete(qv[:24], [5, 6]).mean() * 24)
    assert np.isnan(d["DV"].iloc[1])
    assert d["DV"].iloc[2] == pytest.approx(qv[48:72].sum())


def test_daily_inputs_partial_first_day_is_nan():
    h = hourly_frame(24 + 19, start=D0 + pd.Timedelta(hours=5))   # day 0 has 19 hours
    d = daily_inputs(h)
    assert d.index[0] == D0 and len(d) == 2
    assert np.isnan(d["RV"].iloc[0]) and np.isnan(d["DV"].iloc[0])
    assert d["RV"].iloc[1] == pytest.approx(24.0)


# --- R1 ---------------------------------------------------------------------------------------

def test_r1_increasing_vol_is_high_after_365_values():
    n = 500
    reg = daily_regimes(daily_frame(n, RV=np.arange(1.0, n + 1)), CFG)
    assert reg["s"].iloc[:14].isna().all() and reg["s"].iloc[14:].notna().all()
    assert reg["s"].iloc[14] == pytest.approx(np.sqrt(np.arange(1.0, 16).mean()))  # 15 of 30
    assert reg["s"].iloc[29] == pytest.approx(np.sqrt(np.arange(1.0, 31).mean()))
    assert reg["p"].iloc[:FIRST_P].isna().all()
    assert (reg["p"].iloc[FIRST_P:] == 1.0).all()
    assert reg["r1_vol"].iloc[:FIRST_P].isna().all()
    assert (reg["r1_vol"].iloc[FIRST_P:] == 2).all()
    assert reg["r1_vol_2y"].iloc[:FIRST_P].isna().all()
    assert (reg["r1_vol_2y"].iloc[FIRST_P:] == 2).all()
    assert str(reg["r1_vol"].dtype) == "Int8" and str(reg["r1_vol_2y"].dtype) == "Int8"


def test_r1_decreasing_vol_is_low_and_2y_window():
    n = 1000
    reg = daily_regimes(daily_frame(n, RV=np.arange(n, 0, -1.0)), CFG)
    assert reg["p"].iloc[:FIRST_P].isna().all()
    assert (reg["r1_vol"].iloc[FIRST_P:] == 0).all()
    assert reg["p"].iloc[FIRST_P] == pytest.approx(1 / 365)
    d = 900
    assert reg["p"].iloc[d] == pytest.approx(1 / (d - 13))   # expanding: all values since day 14
    assert reg["p_2y"].iloc[d] == pytest.approx(1 / 730)      # trailing 730 days
    assert reg["r1_vol_2y"].iloc[d] == 0


def test_r1_window_tolerance_is_half_the_window():
    # Alternating pattern -> all codes valid.
    rv = np.tile([1.0, 2.0, 3.0], 150)
    assert daily_regimes(daily_frame(450, RV=rv), CFG)["r1_vol"].dropna().isin([0, 1, 2]).all()
    # 16 NaN RV days in the 30-day window ending at day 55 -> 14 < 15 -> s NaN; 15 NaN -> defined.
    rv2 = np.ones(60)
    rv2[40:56] = np.nan
    reg2 = daily_regimes(daily_frame(60, RV=rv2), CFG)
    assert np.isnan(reg2["s"].iloc[55]) and reg2["s"].iloc[39] == 1.0
    rv2[55] = 1.0
    assert daily_regimes(daily_frame(60, RV=rv2), CFG)["s"].iloc[55] == 1.0


def test_one_outage_day_no_longer_blanks_r1_for_30_days():
    """D-43: outages that kill two days' RV (5 days apart) leave s, p and both R1 states defined.
    Under the old 29-of-30 rule, s and R1 would have been undefined on days 425..449."""
    h = random_hourly(460, seed=5)
    day = 420 * 24
    out = h.index[day + 3:day + 13]                                     # 10 h outage on day 420
    h.loc[out, ["rv", "close"]] = np.nan
    h.loc[out, ["valid", "volume_ok"]] = False
    h.loc[h.index[day + 5 * 24:day + 5 * 24 + 4], "rv"] = np.nan        # 4 h of rv on day 425
    d = daily_inputs(h)
    assert np.isnan(d["RV"].iloc[420]) and np.isnan(d["RV"].iloc[425])
    reg = daily_regimes(d, CFG)
    after = reg.iloc[FIRST_P + 1:]                 # day 0 is complete, but r[0] is NaN
    for c in ["s", "p", "p_2y", "T", "ILLIQ", "r1_vol", "r1_vol_2y", "r2_trend"]:
        assert after[c].notna().all(), c


# --- R2 ---------------------------------------------------------------------------------------

def test_r2_bull_bear_and_t_formula():
    rng = np.random.default_rng(0)
    n = 200
    r = 0.01 + rng.normal(0, 0.002, n)
    reg = daily_regimes(daily_frame(n, r=r), CFG)
    assert reg["T"].iloc[:44].isna().all()                    # need ceil(0.50*90) = 45 values
    assert reg["T"].iloc[44] == pytest.approx(r[:45].mean() / (r[:45].std(ddof=1) / np.sqrt(45)))
    assert (reg["r2_trend"].iloc[44:] == 1).all()
    w = r[-90:]
    assert reg["T"].iloc[-1] == pytest.approx(w.mean() / (w.std(ddof=1) / np.sqrt(90)))
    assert (daily_regimes(daily_frame(n, r=-r), CFG)["r2_trend"].iloc[44:] == -1).all()

    r2 = r.copy()
    r2[-90:-45] = np.nan                                       # 45 of 90 present
    T = daily_regimes(daily_frame(n, r=r2), CFG)["T"].iloc[-1]
    w = r2[-45:]
    assert T == pytest.approx(w.mean() / (w.std(ddof=1) / np.sqrt(45)))
    r2[-45] = np.nan                                           # 44 of 90 -> NaN
    assert np.isnan(daily_regimes(daily_frame(n, r=r2), CFG)["T"].iloc[-1])

    sideways = rng.normal(0, 0.01, n)
    reg_s = daily_regimes(daily_frame(n, r=sideways), CFG)
    T_s = reg_s["T"].iloc[44:]
    st = reg_s["r2_trend"].iloc[44:].astype(int)
    assert ((T_s > 1) == (st == 1)).all() and ((T_s < -1) == (st == -1)).all()
    assert (st == 0).any()


# --- R3 ---------------------------------------------------------------------------------------

def test_r3_illiq_and_dv_z_formulas():
    rng = np.random.default_rng(1)
    n = 500
    r = rng.normal(0, 0.02, n)
    dv = np.exp(rng.normal(10, 0.3, n))
    reg = daily_regimes(daily_frame(n, r=r, DV=dv), CFG)
    d = 450
    assert reg["ILLIQ"].iloc[d] == pytest.approx(np.mean(np.abs(r[d - 29:d + 1]) / dv[d - 29:d + 1]))
    assert reg["ILLIQ"].iloc[:14].isna().all()
    assert reg["ILLIQ"].iloc[14] == pytest.approx(np.mean(np.abs(r[:15]) / dv[:15]))  # 15 of 30
    ill = reg["ILLIQ"]
    hist = ill.iloc[:d + 1].dropna()
    assert reg["p_illiq"].iloc[d] == pytest.approx((hist <= ill.iloc[d]).mean())
    lg = np.log(dv)
    prev = lg[d - 365:d]
    assert reg["dv_z"].iloc[d] == pytest.approx((lg[d] - prev.mean()) / prev.std(ddof=1))
    assert reg["dv_z"].iloc[:183].isna().all() and reg["dv_z"].iloc[183:].notna().all()  # 183/365
    assert reg["dv_z"].iloc[183] == pytest.approx((lg[183] - lg[:183].mean()) / lg[:183].std(ddof=1))
    assert reg["r3_liq"].iloc[:FIRST_P].isna().all()
    assert reg["r3_liq"].iloc[FIRST_P:].notna().all()
    # Monotonically falling dollar volume with constant |r| -> rising ILLIQ -> state 2.
    reg2 = daily_regimes(daily_frame(n, r=np.where(np.arange(n) % 2, 0.01, -0.01),
                                     DV=np.linspace(1e6, 1e4, n)), CFG)
    assert (reg2["r3_liq"].dropna() == 2).all()



# --- hourly mapping ---------------------------------------------------------------------------

def test_to_hourly_maps_day_d_to_hours_of_day_d_plus_1():
    days = pd.date_range(D0, periods=3, freq="D", name="day")
    ds = pd.DataFrame({"r1_vol": pd.array([0, 1, 2], dtype="Int8"), "s": [0.1, 0.2, 0.3]},
                      index=days)
    hidx = pd.date_range(D0, periods=4 * 24, freq="1h", name="ts")
    hs = to_hourly(ds, hidx)
    assert hs.index.equals(hidx)
    assert str(hs["r1_vol"].dtype) == "Int8"
    assert hs["r1_vol"].iloc[:24].isna().all()                # day 0 hours: no previous day
    assert (hs["r1_vol"].iloc[24:48] == 0).all()             # day 1 hours carry day 0
    assert (hs["r1_vol"].iloc[48:72] == 1).all()
    assert (hs["r1_vol"].iloc[72:96] == 2).all()
    assert (hs["s"].iloc[72:96] == 0.3).all()
    assert not (hs["r1_vol"].iloc[48:72] == 2).any()         # day 2 hours never see day 2


# --- causality --------------------------------------------------------------------------------

def _pipeline(h):
    return to_hourly(daily_regimes(daily_inputs(h), CFG), h.index)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_causality_perturbing_future_hours_leaves_past_states_unchanged(seed):
    n_days = 470
    h = random_hourly(n_days, seed)
    base = _pipeline(h)
    rng = np.random.default_rng(seed + 100)
    t0 = h.index[rng.integers(400 * 24, (n_days - 2) * 24)]
    hp = h.copy()
    m = hp.index >= t0
    k = int(m.sum())
    hp.loc[m, "rv"] *= rng.uniform(0.01, 100, k)
    hp.loc[m, "close"] *= np.exp(rng.normal(0, 0.5, k))
    hp.loc[m, "quote_volume"] *= rng.uniform(0.01, 100, k)
    hp.loc[m, "valid"] = rng.random(k) > 0.05
    hp.loc[m, "volume_ok"] = rng.random(k) > 0.05
    pert = _pipeline(hp)

    cut = t0.floor("D") + pd.Timedelta(days=1)
    before = base.index < cut
    pd.testing.assert_frame_equal(base[before], pert[before], check_exact=True)
    # Non-vacuous: states are defined before the cut, and something changes after it.
    for c in STATES:
        assert base.loc[before, c].notna().any()
    assert not np.allclose(base.loc[~before, "s"].to_numpy(float),
                           pert.loc[~before, "s"].to_numpy(float), equal_nan=True)


# --- window shares and regime-change covariates -----------------------------------------------

def _hand_state_series():
    idx = pd.date_range("2021-01-01", "2021-08-01", freq="1h", inclusive="left", tz="UTC")
    st = pd.Series(pd.array([pd.NA] * len(idx), dtype="Int8"), index=idx)
    st[(idx >= "2021-01-01") & (idx < "2021-02-01")] = 2            # Jan: high
    st[(idx >= "2021-02-01") & (idx < "2021-02-15")] = 2            # Feb: half high,
    st[(idx >= "2021-02-15") & (idx < "2021-03-01")] = 0            #      half low
    st[(idx >= "2021-03-01") & (idx < "2021-04-01")] = 1            # Mar: mid
    st[(idx >= "2021-04-11") & (idx < "2021-05-01")] = 2            # Apr: 10 days NA, 20 high
    st[(idx >= "2021-05-01") & (idx < "2021-07-01")] = 0            # May, Jun: low
    st[(idx >= "2021-07-01") & (idx < "2021-08-01")] = 2            # Jul: high
    return st


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_window_shares_hand_built():
    st = _hand_state_series()
    w = window_shares(st, ts("2021-02-01"), ts("2021-03-01"), [0, 1, 2])
    assert w == {0: 0.5, 1: 0.0, 2: 0.5, "coverage": 1.0}
    w = window_shares(st, ts("2021-04-01"), ts("2021-05-01"), [0, 1, 2])
    assert w[2] == 1.0 and w[0] == 0.0 and w["coverage"] == pytest.approx(20 / 30)
    w = window_shares(st, ts("2021-07-01"), ts("2021-08-31"), [2])   # runs past the series end
    assert w[2] == 1.0 and w["coverage"] == pytest.approx(31 / 61)
    w = window_shares(st, ts("2021-04-01"), ts("2021-04-11"), [2])   # nothing defined
    assert np.isnan(w[2]) and w["coverage"] == 0.0


def test_regime_change_covariates_hand_built():
    st = _hand_state_series()
    taus = [ts(s) for s in ["2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01", "2021-07-01"]]
    cov = regime_change_covariates(st, taus, train_months=2, high_state=2)
    assert list(cov.index) == taus
    jan, feb, mar, apr, may, jun = 744, 672, 744, 720, 744, 720
    pi = cov["pi_high_window"]
    assert pi.iloc[0] == pytest.approx((jan + feb / 2) / (jan + feb))
    assert pi.iloc[1] == pytest.approx((feb / 2) / (feb + mar))
    assert pi.iloc[2] == pytest.approx(480 / (mar + 480))
    assert cov["coverage_window"].iloc[2] == pytest.approx((mar + 480) / (mar + apr))
    assert pi.iloc[3] == pytest.approx(480 / (480 + may))
    assert pi.iloc[4] == 0.0
    # C: |pi(added month) - pi(dropped month)|
    C = cov["C"]
    assert np.isnan(C.iloc[0])
    assert C.iloc[1] == pytest.approx(abs(0.0 - 1.0))   # added Mar (0), dropped Jan (1)
    assert C.iloc[2] == pytest.approx(abs(1.0 - 0.5))   # added Apr (1, defined part), dropped Feb
    assert C.iloc[3] == pytest.approx(abs(0.0 - 0.0))   # added May, dropped Mar
    assert C.iloc[4] == pytest.approx(abs(0.0 - 1.0))   # added Jun, dropped Apr
    # E: state at tau vs state at previous tau. States: Mar1=1, Apr1=NA, May1=0, Jun1=0, Jul1=2
    assert cov["state_tau"].iloc[0] == 1 and np.isnan(cov["state_tau"].iloc[1])
    E = cov["E"]
    assert np.isnan(E.iloc[0]) and np.isnan(E.iloc[1]) and np.isnan(E.iloc[2])
    assert E.iloc[3] == 0.0 and E.iloc[4] == 1.0
