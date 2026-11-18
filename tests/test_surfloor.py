"""Unit tests of the surrogate-referenced floor helpers (D-57)."""

import numpy as np
import pandas as pd
import pytest

from ebmregime import surfloor as S


def _pf(n_sur=20, n_pairs=40, n_feat=3, scale_real=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    taus = pd.date_range("2020-01-01", periods=n_pairs + 12, freq="MS").strftime("%Y-%m-%d")
    for d in S.datasets(n_sur) + ["real"]:
        for i in range(n_pairs):
            for j in range(n_feat):
                g = rng.gamma(20, 1 / 20) * (j + 1) * (scale_real if d == "real" else 1.0)
                rows.append({"data": d, "target": "shuf" if d == "real" else "dir", "tau_a": taus[i],
                             "tau_b": taus[i + 12], "lag": 12, "feature": f"f{j}", "g": g,
                             "s2_seed_a": 0.01, "s2_seed_b": 0.01})
    pf = pd.DataFrame(rows)
    # surrogates also carry shuf and vol copies so every check has data
    extra = [pf[pf.data != "real"].assign(target=t) for t in ("vol", "shuf")]
    return pd.concat([pf] + extra, ignore_index=True)


def test_totals_and_dr_sur_exact():
    pf = _pf()
    tot = S.totals(pf)
    sub = pf[(pf.target == "dir") & (pf.lag == 12)]
    G = sub.groupby(["data", "tau_a"]).g.sum().unstack("data")
    refs = [f"surrogate:{i}" for i in range(1, 20)]
    expect = G["surrogate:0"] / G[refs].mean(axis=1)
    got = S.dr_sur(tot, "surrogate:0", "dir", refs, 12)
    np.testing.assert_allclose(got.values, expect.sort_index().values, rtol=1e-12)
    adj = S.dr_sur(tot, "surrogate:0", "dir", refs, 12, adj=True)
    np.testing.assert_allclose(adj.values, ((G["surrogate:0"] - 0.02 * 3) / (G[refs] - 0.06).mean(axis=1)).sort_index().values)


def test_feature_ratio_matches_by_hand():
    pf = _pf()
    refs = [f"surrogate:{i}" for i in range(1, 20)]
    r = S.dr_sur_feature(pf, "surrogate:0", "dir", refs, 12)
    sub = pf[(pf.target == "dir") & (pf.feature == "f1")].pivot(index="tau_a", columns="data", values="g")
    np.testing.assert_allclose(r["f1"].values, (sub["surrogate:0"] / sub[refs].mean(axis=1)).values)


def test_calibration_passes_on_exchangeable_data_and_negative_control_detects_scale():
    pf = _pf()
    tot = S.totals(pf)
    cal = S.check_calibration(tot, with_ci=False)
    assert cal["pass"], cal["gate"]
    neg = S.check_negative_control(tot, with_ci=False)
    assert neg["pass"], neg["gate"]
    tot2 = S.totals(_pf(scale_real=2.0))
    neg = S.check_negative_control(tot2, with_ci=False)
    assert not neg["pass"] and not neg["gate"]["within_null"]
    assert neg["gate"]["m_real"] == pytest.approx(2.0, rel=0.1)


def test_loo_medians_are_exchangeable_null():
    tot = S.totals(_pf())
    m = S.loo_medians(tot, "dir", 12)
    assert len(m) == 20 and 0.9 < m.median() < 1.1


def test_rank_p():
    tot = S.totals(_pf(scale_real=5.0))
    tot = pd.concat([tot, tot[tot.data == "real"].assign(target="dir")])
    assert S.rank_p(tot, "real", "dir", 12) == pytest.approx(1 / 21)


def test_tail_rows_bags_repeats_only_tail():
    from ebmregime.floorexp_tail import tail_rows_bags

    n = 500
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    bags = np.ones((n, 3), dtype=np.int8)
    bags[:100] = 2
    bags[400:] = -1
    bags[390:400] = 0
    rep, br = tail_rows_bags(ts, bags, 1)
    assert np.all(np.diff(rep) >= 0)                       # time order kept
    head = rep[rep < 400]
    np.testing.assert_array_equal(head, np.arange(400))    # non-tail rows exactly once
    assert len(rep[rep >= 400]) == 100                     # n tail draws
    assert (br[rep >= 400] == -1).all()
    np.testing.assert_array_equal(br[rep < 400], bags[:400])
