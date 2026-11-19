"""Unit tests of the D-62 pooled rank statistic and median-referenced ratio."""

import numpy as np
import pandas as pd
import pytest

from ebmregime import surfloor as S


def test_pooled_stats_hand_example():
    w = pd.DataFrame({"a": [1.0, 4.0], "b": [2.0, 2.0], "c": [3.0, 1.0]})
    st = S.pooled_stats(w)
    # pair 0: a lowest (u=0), b middle (0.5), c top (1); pair 1: a top, b middle, c lowest
    assert st.loc["a", "U"] == pytest.approx(0.5) and st.loc["b", "U"] == pytest.approx(0.5)
    # r_a = [1/2.5, 4/1.5]; median over the two pairs = mean
    assert st.loc["a", "M"] == pytest.approx((1 / 2.5 + 4 / 1.5) / 2)
    assert st.loc["b", "M"] == pytest.approx((2 / 2 + 2 / 2.5) / 2)


def test_pooled_stats_ties_and_exact_centre():
    w = pd.DataFrame(np.ones((5, 4)), columns=list("abcd"))
    st = S.pooled_stats(w)
    assert np.allclose(st.U, 0.5) and np.allclose(st.M, 1.0)
    rng = np.random.default_rng(1)
    w = pd.DataFrame(rng.lognormal(0, 1.5, (60, 21)))   # heavily skewed, exchangeable
    st = S.pooled_stats(w)
    assert st.U.mean() == pytest.approx(0.5)             # exact by construction
    assert 0.8 < st.M.median() < 1.25                    # median ratio stays centred despite skew


def test_pooled_stats_detects_shift_and_drops_incomplete():
    rng = np.random.default_rng(2)
    w = pd.DataFrame(rng.gamma(2, 1, (50, 10)))
    w[0] *= 3.0
    w.iloc[3, 5] = np.nan
    st = S.pooled_stats(w)
    assert st.loc[0, "U"] == st.U.max() and st.loc[0, "M"] > 2
    assert (st.n == 49).all()


def test_rank_checks_on_null_panel():
    from tests.test_surfloor import _pf

    tot = S.totals(_pf(seed=3))
    pr = {"hypotheses": {"g0": {"pass_ci_within": [0.8, 1.25]}}, "stability": {"headline_lag_months": 12}}
    cal = S.check_calibration_rank(tot, pr)
    assert cal["pass"] and set(cal["gate"]) == {"dir", "vol"}
    neg = S.check_negative_control_rank(tot, pr)
    assert neg["gate"]["U_within_null"] and 1 <= neg["gate"]["rank_from_top"] <= 21
    big = S.totals(_pf(seed=3, scale_real=4.0))
    neg = S.check_negative_control_rank(big, pr)
    assert neg["gate"]["rank_from_top"] == 1 and not neg["pass"]


def test_real_summary_rank_shapes_and_status():
    from tests.test_surfloor import _pf

    pf = _pf(seed=4, scale_real=3.0)
    real = pf[pf.data == "real"]
    pf = pd.concat([pf[pf.data != "real"]] + [real.assign(target=t) for t in ("dir", "vol")], ignore_index=True)
    pr = {"stability": {"headline_lag_months": 12}}
    prec = {"dir": {"precise": False}, "vol": {"precise": True}}
    summ, feat = S.real_summary_rank(S.totals(pf), pf, prec, pr)
    assert len(summ) == 4 and len(feat) == 6
    assert (summ.rank_from_top == 1).all() and np.allclose(summ.p_upper, 1 / 21)
    assert summ[summ.target == "dir"].status.str.startswith("exploratory").all()
    assert (summ[summ.target == "vol"].status == "precise").all()
