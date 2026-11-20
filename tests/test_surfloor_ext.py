"""D-65 extension: indices, targets and the 121-member summary on a toy panel."""

import numpy as np
import pandas as pd

from ebmregime import surfloor_ext as E


def test_ext_datasets_and_jobs():
    d = E.ext_datasets()
    assert d[0] == "surrogate:20" and d[-1] == "surrogate:119" and len(d) == 100
    jobs = E.ext_jobs(20, 21)
    assert jobs and {j.target for j in jobs} == {"vol"} and {j.data for j in jobs} == {"surrogate:20"}


def test_summarise_toy_panel(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    taus = pd.date_range("2020-01-01", periods=30, freq="MS").strftime("%Y-%m-%d")
    rows = []
    for d in ["real"] + [f"surrogate:{i}" for i in range(E.HI)]:
        for i in range(18):
            for f in ("f0", "f1"):
                g = rng.gamma(20, 1 / 20) * (3.0 if d == "real" else 1.0)
                rows.append({"data": d, "target": "vol", "tau_a": taus[i], "tau_b": taus[i + 12], "lag": 12,
                             "feature": f, "g": g, "s2_seed_a": 0.01, "s2_seed_b": 0.01})
    monkeypatch.setattr(E.S, "OUT", tmp_path)
    pr = {"hypotheses": {"g0": {"pass_ci_within": [0.8, 1.25]}}, "stability": {"headline_lag_months": 12}}
    out = E.summarise(pd.DataFrame(rows), pr)
    assert out["n_pool"] == 121 and out["a"]["pass"]
    assert out["lag12"]["rank_from_top"] == 1 and abs(out["lag12"]["p_upper"] - 1 / 121) < 1e-12
