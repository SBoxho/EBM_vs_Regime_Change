import numpy as np
import pandas as pd

from ebmregime.series import (forward_mean_strict, hourly_inputs, returns_15m, rv_hourly,
                              trailing_mean, trailing_sum)

T0 = pd.Timestamp("2025-01-01", tz="UTC")


def grid15(closes, valid=None):
    idx = pd.date_range(T0, periods=len(closes), freq="15min", name="ts")
    return pd.DataFrame({"close": closes, "valid": True if valid is None else valid}, index=idx)


def test_rv_hour_uses_returns_ending_in_hour_including_previous_close():
    c = np.exp(np.array([0.0, 0.01, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.0]))  # 9 bars = 2h + 1
    g = grid15(c)
    idx = pd.date_range(T0, periods=3, freq="1h", name="ts")
    rv = rv_hourly(g, idx)
    # Hour 0 has only 3 returns (no close before the first bar): NaN.
    assert np.isnan(rv.iloc[0])
    # Hour 1 returns: bar3->4 (-0.02), 4->5 (0), 5->6 (0.03), 6->7 (-0.03)
    assert np.isclose(rv.iloc[1], 0.02**2 + 0 + 0.03**2 + 0.03**2)
    # Hour 2 contains only one 15m bar: NaN.
    assert np.isnan(rv.iloc[2])


def test_invalid_subbar_kills_both_adjacent_returns():
    c = np.exp(np.linspace(0, 0.08, 9))
    v = np.ones(9, bool); v[5] = False
    r = returns_15m(grid15(c, v))
    assert r.isna().tolist() == [True, False, False, False, False, True, True, False, False]


def test_hourly_inputs_masks_rv_on_invalid_hour():
    c = np.exp(np.linspace(0, 0.08, 9))
    idx = pd.date_range(T0, periods=2, freq="1h", name="ts")
    bars = pd.DataFrame({"valid": [True, False], "volume_suspect": [True, False]}, index=idx)
    h = hourly_inputs(grid15(c), bars)
    assert h["volume_ok"].tolist() == [False, False]
    assert h["rv"].isna().all()  # hour 0: only 3 returns; hour 1: invalid hour


def test_trailing_tolerance_and_forward_strict():
    x = pd.Series(np.arange(1.0, 21.0))
    x[[3, 4]] = np.nan
    m = trailing_mean(x, 20)                     # 18/20 present >= 90%
    assert np.isclose(m.iloc[-1], x.drop([3, 4]).mean())
    assert np.isclose(trailing_sum(x, 20).iloc[-1], x.drop([3, 4]).mean() * 20)
    x[5] = np.nan                                # 17/20 < 90%
    assert np.isnan(trailing_mean(x, 20).iloc[-1])
    f = forward_mean_strict(pd.Series([1.0, 2.0, 3.0, np.nan, 5.0]), 2)
    assert f.iloc[0] == 2.5 and np.isnan(f.iloc[1]) and np.isnan(f.iloc[2]) and np.isnan(f.iloc[4])
