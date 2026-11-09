"""M1 acceptance: an independent re-derivation of features, targets and regimes for random real rows.

Written by the lead as the verifier. It does NOT use ``ebmregime.series`` or pandas rolling
windows. It recomputes every quantity from the raw 15m grid with explicit numpy indexing, following
the definitions in configs/features.yaml, configs/prereg.yaml and PLAN §5 and §8. It is a
deliberately different code path, so a shared bug cannot pass silently.

Needs the prepared data (scripts/download_data.py, scripts/build_bars.py).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ebmregime.config import load_config, repo_path

DATA_CFG = load_config("data")
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (repo_path(DATA_CFG["paths"]["interim"]) / "build_info.json").exists(),
                       reason="data not prepared"),
]
N_ROWS = 25


def need(k: int) -> int:
    """Window rule (D-43): aggregations over k raw hourly values need ceil(0.90 k)."""
    return math.ceil(0.90 * k)


def need_hist(n: int) -> int:
    """History rule (D-43): windows over derived values need ceil(0.50 n)."""
    return math.ceil(0.50 * n)


class Oracle:
    """Hourly arrays rebuilt directly from the 15m grid."""

    def __init__(self, sym: str):
        from ebmregime.data import store
        g, b = store.load("grid_15m", sym), store.load("bars_1h", sym)
        self.idx = b.index
        off = int((b.index[0] - g.index[0]) / pd.Timedelta("15min"))
        assert off >= 0
        c15 = np.where(g["valid"].to_numpy(), g["close"].to_numpy(dtype=float), np.nan)
        n = len(b)
        rv = np.full(n, np.nan)
        valid = b["valid"].to_numpy()
        for i in range(n):
            p = off + 4 * i
            if p < 1 or p + 3 >= len(c15) or not valid[i]:
                continue
            seg = c15[p - 1:p + 4]
            if np.isfinite(seg).all():
                rv[i] = float(np.sum(np.diff(np.log(seg)) ** 2))
        self.rv = rv
        self.valid = valid
        self.C = np.where(valid, b["close"].to_numpy(dtype=float), np.nan)
        self.H = np.where(valid, b["high"].to_numpy(dtype=float), np.nan)
        self.L = np.where(valid, b["low"].to_numpy(dtype=float), np.nan)
        vok = valid & ~b["volume_suspect"].to_numpy()
        self.vok = vok
        self.QV = np.where(vok, b["quote_volume"].to_numpy(dtype=float), np.nan)
        self.TBQ = np.where(vok, b["taker_buy_quote"].to_numpy(dtype=float), np.nan)

    @staticmethod
    def tmean(x, i, k):
        w = x[max(0, i - k + 1):i + 1]
        if i - k + 1 < 0:
            return np.nan
        ok = w[np.isfinite(w)]
        return ok.mean() if len(ok) >= need(k) else np.nan

    def RV(self, i, k):
        return self.tmean(self.rv, i, k)

    def sigma(self, i):
        return math.sqrt(self.RV(i, 168))

    def ret(self, i, k):
        return math.log(self.C[i] / self.C[i - k]) / (self.sigma(i) * math.sqrt(k))

    def features(self, i, partner: "Oracle | None") -> dict:
        f = {f"ret_{k}h": self.ret(i, k) for k in (1, 4, 24, 168)}
        f["rv_ratio_4h"] = math.log(self.RV(i, 4) / self.RV(i, 168))
        f["rv_ratio_24h"] = math.log(self.RV(i, 24) / self.RV(i, 168))
        hist = np.array([self.RV(j, 168) for j in range(i - 4319, i + 1)])
        ok = hist[np.isfinite(hist)]
        f["rv_level_pct"] = (ok <= self.RV(i, 168)).mean() if len(ok) >= need_hist(4320) else np.nan
        x = np.log(self.QV)
        ref = x[i - 168:i]
        ref = ref[np.isfinite(ref)]
        f["qv_z_1h"] = (x[i] - ref.mean()) / ref.std(ddof=1) if len(ref) >= need(168) else np.nan
        qv24 = lambda j: self.tmean(self.QV, j, 24) * 24
        y = np.array([math.log(qv24(i - 24 * j)) for j in range(1, 31)])
        y = y[np.isfinite(y)]
        f["qv_z_24h"] = (math.log(qv24(i)) - y.mean()) / y.std(ddof=1) if len(y) >= need_hist(30) else np.nan
        f["range_1h"] = math.log(self.H[i] / self.L[i]) / self.sigma(i)
        for k in (24, 168):
            hw, lw = self.H[i - k + 1:i + 1], self.L[i - k + 1:i + 1]
            hk, lk = np.nanmax(hw), np.nanmin(lw)
            f[f"pos_range_{k}h"] = 0.5 if hk == lk else (self.C[i] - lk) / (hk - lk)
        f["hour"] = self.idx[i].hour
        f["dow"] = self.idx[i].day_of_week
        f["taker_imb_4h"] = np.nansum(self.TBQ[i - 3:i + 1]) / np.nansum(self.QV[i - 3:i + 1]) - 0.5
        if partner is not None:
            j = partner.idx.get_loc(self.idx[i])
            f["xret_4h"] = partner.ret(j, 4)
        return f

    def targets(self, i, h) -> dict:
        if not self.valid[i:i + h + 1].all():
            return {"y_dir": np.nan, "y_vol": np.nan}
        r = math.log(self.C[i + h] / self.C[i])
        fwd = self.rv[i + 1:i + h + 1]
        y_vol = math.log(fwd.mean() / self.RV(i, 168)) if np.isfinite(fwd).all() else np.nan
        return {"y_dir": np.nan if r == 0 else float(r > 0), "y_vol": y_vol}


@pytest.fixture(scope="module")
def btc():
    return Oracle("BTCUSDT"), Oracle("ETHUSDT")


@pytest.fixture(scope="module")
def module_outputs():
    from ebmregime.data import store
    from ebmregime.features.build import build_features
    from ebmregime.series import hourly_inputs
    from ebmregime.targets.build import build_targets
    h = hourly_inputs(store.load("grid_15m", "BTCUSDT"), store.load("bars_1h", "BTCUSDT"))
    p = hourly_inputs(store.load("grid_15m", "ETHUSDT"), store.load("bars_1h", "ETHUSDT"))
    feats = build_features(h, p, load_config("features"))
    targs = build_targets(h, "BTCUSDT", load_config("prereg"), DATA_CFG, horizons=[8, 24])
    return h, feats, targs


def sample_rows(feats, n, seed=7):
    start = DATA_CFG["symbols"]["BTCUSDT"]["analysis_start"]
    pos = np.flatnonzero(feats["features_ok"].to_numpy() & (feats.index >= pd.Timestamp(start, tz="UTC")))
    return np.random.default_rng(seed).choice(pos, size=n, replace=False)


def test_features_match_independent_rederivation(btc, module_outputs):
    o, partner = btc
    _, feats, _ = module_outputs
    assert (feats.index == o.idx).all()
    for i in sample_rows(feats, N_ROWS):
        want = o.features(i, partner)
        for k, v in want.items():
            got = feats[k].iloc[i]
            assert np.isclose(got, v, rtol=1e-9, atol=1e-12), (o.idx[i], k, got, v)


def test_targets_match_independent_rederivation(btc, module_outputs):
    o, _ = btc
    _, feats, targs = module_outputs
    for h in (8, 24):
        for i in sample_rows(feats, N_ROWS, seed=h):
            want = o.targets(i, h)
            for k in ("y_dir", "y_vol"):
                got, v = targs[f"{k}_h{h}"].iloc[i], want[k]
                assert (np.isnan(got) and np.isnan(v)) or np.isclose(got, v, rtol=1e-9), (o.idx[i], k, h, got, v)


def test_feature_rows_ok_share(module_outputs):
    _, feats, _ = module_outputs
    rng = feats.loc[DATA_CFG["symbols"]["BTCUSDT"]["analysis_start"]:]
    assert rng["features_ok"].mean() > 0.98


def test_regime_r1_and_r2_match_independent_rederivation(btc):
    from ebmregime.data import store
    from ebmregime.regimes.causal import daily_inputs, daily_regimes, to_hourly
    from ebmregime.series import hourly_inputs
    o, _ = btc
    h = hourly_inputs(store.load("grid_15m", "BTCUSDT"), store.load("bars_1h", "BTCUSDT"))
    cfg = load_config("prereg")["regimes"]
    daily = daily_inputs(h)
    reg = daily_regimes(daily, cfg)
    hourly = to_hourly(reg, h.index)
    # Independent daily RV and close from the oracle's hourly arrays.
    days = pd.DatetimeIndex(sorted(set(o.idx.floor("1D"))))
    rvd, cd = [], []
    for d in days:
        sl = (o.idx >= d) & (o.idx < d + pd.Timedelta("1D"))
        x = o.rv[sl]
        ok = x[np.isfinite(x)]
        rvd.append(ok.mean() * 24 if (len(x) == 24 and len(ok) >= need(24)) else np.nan)
        last = np.flatnonzero(sl)[-1]
        cd.append(o.C[last] if o.idx[last].hour == 23 else np.nan)
    rvd, cd = np.array(rvd), np.array(cd)
    r = np.r_[np.nan, np.log(cd[1:] / cd[:-1])]
    s = np.array([np.sqrt(np.mean(rvd[k - 29:k + 1][np.isfinite(rvd[k - 29:k + 1])]))
                  if k >= 29 and np.isfinite(rvd[k - 29:k + 1]).sum() >= need_hist(30) else np.nan
                  for k in range(len(days))])
    rng = np.random.default_rng(3)
    for k in rng.choice(np.arange(400, len(days) - 2), size=15, replace=False):
        def state(hist):
            hist = hist[np.isfinite(hist)]
            p = (hist <= s[k]).mean() if len(hist) >= 365 and np.isfinite(s[k]) else np.nan
            return np.nan if np.isnan(p) else (0 if p < 1 / 3 else (2 if p > 2 / 3 else 1))
        st_exp = state(s[:k + 1])                       # r1_vol: expanding history (sensitivity)
        st_2y = state(s[max(0, k - 729):k + 1])         # r1_vol_2y: trailing 730 days (primary, D-44)
        w = r[k - 89:k + 1]
        w = w[np.isfinite(w)]
        T = w.mean() / (w.std(ddof=1) / math.sqrt(len(w))) if len(w) >= need_hist(90) else np.nan
        tr = np.nan if np.isnan(T) else (1 if T > 1 else (-1 if T < -1 else 0))
        # The state computed at the end of day k applies to every hour of day k+1, never to day k.
        nxt = days[k] + pd.Timedelta("1D")
        hrs = hourly.loc[nxt:nxt + pd.Timedelta("23h")]
        assert len(hrs) == 24
        for got, want in ((hrs["r1_vol"], st_exp), (hrs["r1_vol_2y"], st_2y), (hrs["r2_trend"], tr)):
            vals = pd.to_numeric(got, errors="coerce").to_numpy(dtype=float)
            assert np.all(np.isnan(vals)) if np.isnan(want) else np.all(vals == want), (days[k], want, vals[:3])


@pytest.mark.parametrize("cut", ["2019-05-15 03:00", "2021-04-23 01:00", "2023-03-24 12:00", "2025-01-01 00:00"])
def test_truncating_raw_data_changes_nothing_before_the_cut(cut):
    """End-to-end no-lookahead on real data, at outage dates and the µs-switch date.

    Drop every raw 15m and 1h bar from ``cut`` onward (for the symbol and its partner), rebuild
    everything, and compare: features at rows < cut, labels at rows whose label time <= cut, and
    regime states at hours of days that end before the cut. They must be bit-identical.
    """
    from ebmregime.data import store
    from ebmregime.features.build import build_features
    from ebmregime.regimes.causal import daily_inputs, daily_regimes, to_hourly
    from ebmregime.series import hourly_inputs
    from ebmregime.targets.build import direction, forward_vol

    cut = pd.Timestamp(cut, tz="UTC")
    g, b = store.load("grid_15m", "BTCUSDT"), store.load("bars_1h", "BTCUSDT")
    pg, pb = store.load("grid_15m", "ETHUSDT"), store.load("bars_1h", "ETHUSDT")
    fcfg, rcfg = load_config("features"), load_config("prereg")["regimes"]

    def run(g, b, pg, pb):
        h, p = hourly_inputs(g, b), hourly_inputs(pg, pb)
        reg = to_hourly(daily_regimes(daily_inputs(h), rcfg), h.index)
        t = pd.concat([direction(h, 8), forward_vol(h, 8)], axis=1)
        return build_features(h, p, fcfg), t, reg

    f_all, t_all, r_all = run(g, b, pg, pb)
    f_cut, t_cut, r_cut = run(g[g.index < cut], b[b.index < cut], pg[pg.index < cut], pb[pb.index < cut])
    rows = f_cut.index[f_cut.index < cut]
    pd.testing.assert_frame_equal(f_all.loc[rows], f_cut.loc[rows], check_exact=True)
    lab = t_cut.index[t_cut.index + pd.Timedelta(hours=9) <= cut]
    pd.testing.assert_frame_equal(t_all.loc[lab], t_cut.loc[lab], check_exact=True)
    hrs = r_cut.index[r_cut.index < cut.floor("1D")]
    pd.testing.assert_frame_equal(r_all.loc[hrs], r_cut.loc[hrs], check_exact=True)
