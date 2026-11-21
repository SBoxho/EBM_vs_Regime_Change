"""D-67: SNR floors and decision on toy data; block-length keys and panels."""

import numpy as np
import pandas as pd

from ebmregime import surfloor as S
from ebmregime import surfloor_snr as N


def _pairs(n_sur=3, taus=26, scale=None):
    t = pd.date_range("2020-01-01", periods=taus, freq="MS").strftime("%Y-%m-%d")
    rows = []
    for d in S.datasets(n_sur):
        for tg in ("dir", "shuf"):
            for i in range(taus - 12):
                g = (scale or {}).get(d, 1.0) * (i + 1.0)
                rows.append({"data": d, "target": tg, "lag": 12, "tau_a": t[i], "tau_b": t[i + 12],
                             "feature": "f0", "g": g})
    return pd.DataFrame(rows), list(t)


def test_floors_average_pairs_containing_tau_and_leave_one_out():
    pf, t = _pairs(scale={"surrogate:2": 3.0})
    fl = N.floors(pf, "dir", S.datasets(3))
    # tau index 12 sits in pairs (0,12) and (12,24): g = 1 and 13 per member; mean over members (1+1+3)/3
    assert np.isclose(fl.loc[(t[12], "f0")], 0.5 * np.mean([1, 13]) * (5 / 3))
    lo = N.floors(pf, "dir", S.datasets(3), exclude="surrogate:2")
    assert np.isclose(lo.loc[(t[12], "f0")], 0.5 * np.mean([1, 13]))


def test_decide_flags_signal_only_when_dir_tops_the_pool():
    pf, t = _pairs(n_sur=20)
    rows = []
    for d in ["real"] + S.datasets(20):
        for tg in (("dir", "shuf") if d == "real" else ("shuf",)):
            for tau in t:
                rows.append({"data": d, "target": tg, "tau": tau, "feature": "f0",
                             "A2_mean": 10.0 if (d, tg) == ("real", "dir") else 1.0 + 0.01 * int(d[-1] if d[-1].isdigit() else 0)})
    tab = N.snr_table(pd.DataFrame(rows), pf, n_sur=20)
    out, feat = N.decide(tab)
    assert out["pool"] == 22 and out["signal"] and out["rank_from_top"] == 1
    tab.loc[tab.member == "real dir", "snr"] *= 0.0
    out, _ = N.decide(tab)
    assert not out["signal"] and out["rank_from_top"] == 22


def test_block_keys_differ_by_block_and_from_perm():
    from ebmregime import surfloor_block as B
    from ebmregime import surfloor_perm as Q

    j = B.jobs()[0]
    keys = {B.key_fn_for(720)(j), B.key_fn_for(2160)(j), Q.key_fn(j)}
    assert len(keys) == 3
