"""Stability metrics, noise floor and drift ratio (PLAN §6.2-§6.4; D-14, D-16, D-17, D-50).

Synthetic shapes only; known answers plus a Monte Carlo check of the drift ratio.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
from scipy import stats

from ebmregime.config import load_config
from ebmregime.stability import (
    amplitude,
    cell_noise_sd,
    cross_mean_d2,
    d2,
    drift_pair,
    importance,
    importance_shares,
    local_monotonicity_agreement,
    monotonicity_agreement,
    nd2,
    pair_mean_d2,
    shape_corr,
    shape_spearman,
    sigma2,
    sign_agreement,
    spearman_rank,
    topk_jaccard,
    tv_distance,
    weighted_tau,
)


def recentre(f, w):
    f = np.asarray(f, dtype=float)
    return f - np.sum(w * f, axis=-1, keepdims=True)


@pytest.fixture
def grid():
    rng = np.random.default_rng(1)
    K = 20
    w = rng.dirichlet(np.ones(K))
    f = recentre(rng.normal(size=K), w)
    return f, w


# --------------------------------------------------------------------------------------------
# Defaults mirror the frozen config
# --------------------------------------------------------------------------------------------


def test_defaults_match_prereg():
    st = load_config("prereg")["stability"]
    import inspect

    assert inspect.signature(monotonicity_agreement).parameters["min_abs"].default == st[
        "monotonicity_min_abs_spearman"
    ]
    assert inspect.signature(local_monotonicity_agreement).parameters["mult"].default == st[
        "local_monotonicity_noise_multiple"
    ]
    sig = inspect.signature(drift_pair).parameters
    assert sig["min_abs_spearman"].default == st["monotonicity_min_abs_spearman"]
    assert list(sig["top_k"].default) == st["top_k_jaccard"]
    assert st["rank_metric"] == "weightedtau_hyperbolic"


def test_package_does_not_import_surrogate():
    import subprocess
    import sys

    code = "import sys, ebmregime.stability; print('ebmregime.stability.surrogate' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------------------------
# Per-term metrics: known answers
# --------------------------------------------------------------------------------------------


def test_identical_shapes(grid):
    f, w = grid
    assert d2(f, f, w) == 0.0
    assert shape_corr(f, f, w) == pytest.approx(1.0)
    assert shape_spearman(f, f) == pytest.approx(1.0)
    sa, cov = sign_agreement(f, f, w, 0.0)
    assert sa == 1.0 and cov == pytest.approx(1.0)
    assert nd2(f, f, w) == 0.0


def test_sign_flip(grid):
    f, w = grid
    A = amplitude(f, w)
    assert d2(f, -f, w) == pytest.approx(4 * A**2)
    assert shape_corr(f, -f, w) == pytest.approx(-1.0)
    sa, _ = sign_agreement(f, -f, w, 0.0)
    assert sa == 0.0
    assert nd2(f, -f, w) == pytest.approx(4.0)


def test_orthogonal_equal_amplitude():
    w = np.full(4, 0.25)
    fa = np.array([1.0, -1.0, 1.0, -1.0])
    fb = np.array([1.0, 1.0, -1.0, -1.0])
    assert np.sum(w * fa) == 0 and np.sum(w * fb) == 0 and np.sum(w * fa * fb) == 0
    assert amplitude(fa, w) == amplitude(fb, w) == 1.0
    assert nd2(fa, fb, w) == pytest.approx(2.0)
    assert shape_corr(fa, fb, w) == pytest.approx(0.0)


def test_unrelated_shapes_nd2_about_two():
    rng = np.random.default_rng(2)
    K = 50
    w = np.full(K, 1 / K)
    fa = recentre(rng.normal(size=(4000, K)), w)
    fb = recentre(rng.normal(size=(4000, K)), w)
    assert np.mean(nd2(fa, fb, w)) == pytest.approx(2.0, abs=0.02)


def test_zero_weight_cells_do_not_count():
    w = np.array([0.5, 0.0, 0.5, 0.0])
    fa = np.array([1.0, 7.0, -1.0, 3.0])
    fb = np.array([1.0, -5.0, -1.0, 9.0])
    assert d2(fa, fb, w) == 0.0
    assert shape_corr(fa, fb, w) == pytest.approx(1.0)
    sa, cov = sign_agreement(fa, fb, w, 0.0)
    assert sa == 1.0 and cov == 1.0
    assert importance(fa, w) == importance(fb, w) == 1.0


def test_broadcasting_over_leading_dims(grid):
    f, w = grid
    F = np.stack([f, -f, 2 * f])
    np.testing.assert_allclose(d2(F, f, w), np.array([0, 4, 1]) * amplitude(f, w) ** 2)
    np.testing.assert_allclose(amplitude(F, w), np.array([1, 1, 2]) * amplitude(f, w))
    np.testing.assert_allclose(shape_corr(F, f, w), [1, -1, 1])
    np.testing.assert_allclose(nd2(F, f, w), [0, 4, 1 / 2.5])


def test_degenerate_cases_are_nan():
    w = np.full(5, 0.2)
    z = np.zeros(5)
    f = np.array([2.0, -1.0, 0.0, -1.0, 0.0])
    assert np.isnan(nd2(z, z, w))
    assert nd2(f, z, w) == pytest.approx(2.0)  # d2 = A², mean amplitude² = A²/2
    assert np.isnan(shape_corr(f, z, w))
    assert np.isnan(shape_corr(f, np.full(5, 3.0), w))  # constant, not exactly zero after centring
    assert np.isnan(shape_spearman(f, z))
    sa, cov = sign_agreement(f, -f, w, 10.0)
    assert np.isnan(sa) and cov == 0.0
    assert np.all(np.isnan(importance_shares(np.zeros(3))))


def test_weighted_pearson_matches_expanded_data():
    """Density-weighted Pearson = ordinary Pearson on data replicated by integer weights."""
    rng = np.random.default_rng(3)
    counts = rng.integers(1, 6, size=12)
    w = counts / counts.sum()
    fa, fb = rng.normal(size=12), rng.normal(size=12)
    expected = np.corrcoef(np.repeat(fa, counts), np.repeat(fb, counts))[0, 1]
    assert shape_corr(fa, fb, w) == pytest.approx(expected)


def test_shape_spearman_ties_average_ranks():
    fa = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0])  # piecewise-constant shape with ties
    fb = np.array([0.5, 0.1, 0.9, 1.2, 1.1, 3.0])
    assert shape_spearman(fa, fb) == pytest.approx(stats.spearmanr(fa, fb).statistic)


def test_sign_agreement_excludes_noise_cells_and_reports_coverage():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    fa = np.array([0.05, 1.0, -1.0, 0.05])
    fb = np.array([-0.05, 1.0, 1.0, 0.5])
    tau = np.array([0.1, 0.1, 0.1, 0.1])
    # cell 0: both below tau -> excluded; cell 3: fb above tau -> kept (agree); cell 2 disagrees.
    sa, cov = sign_agreement(fa, fb, w, tau)
    assert cov == pytest.approx(0.9)
    assert sa == pytest.approx((0.2 + 0.4) / 0.9)
    sa0, cov0 = sign_agreement(fa, fb, w, 0.0)  # no exclusion: cell 0 counts as a disagreement
    assert cov0 == pytest.approx(1.0) and sa0 == pytest.approx(0.6)
    # per-cell tau: a large tau on cell 2 only removes the disagreement
    sa2, cov2 = sign_agreement(fa, fb, w, np.array([0.1, 0.1, 5.0, 0.1]))
    assert cov2 == pytest.approx(0.6) and sa2 == 1.0


def test_monotonicity_agreement_qualifying_logic():
    x = np.arange(10.0)
    up = x**2
    down = -np.sqrt(x)
    wiggle = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1.0])  # |Spearman with x| ≈ 0.17
    assert abs(stats.spearmanr(x, wiggle).statistic) <= 0.3
    assert monotonicity_agreement(x, up, x + 1) == 1.0
    assert monotonicity_agreement(x, up, down) == 0.0
    assert np.isnan(monotonicity_agreement(x, up, wiggle))
    assert np.isnan(monotonicity_agreement(x, wiggle, up))
    assert np.isnan(monotonicity_agreement(x, up, np.zeros(10)))  # flat: Spearman undefined
    # threshold is strict: |rho| must exceed min_abs
    rho = stats.spearmanr(x, wiggle).statistic
    assert monotonicity_agreement(x, up, wiggle, min_abs=abs(rho) - 1e-9) in (0.0, 1.0)
    assert np.isnan(monotonicity_agreement(x, up, wiggle, min_abs=abs(rho)))


def test_local_monotonicity_agreement():
    fa = np.array([0.0, 1.0, 2.0, 2.1, 1.0])
    fb = np.array([0.0, 1.0, 0.0, 0.05, -1.0])
    # steps a: +1, +1, +0.1, -1.1 ; b: +1, -1, +0.05, -1.05 ; threshold 2*0.2 = 0.4
    # qualifying: 0 (agree), 1 (disagree), 3 (agree); step 2 is noise.
    assert local_monotonicity_agreement(fa, fb, 0.2) == pytest.approx(2 / 3)
    assert local_monotonicity_agreement(fa, fb, np.array([0.2, 10, 0.2, 0.2])) == 1.0
    assert np.isnan(local_monotonicity_agreement(fa, fb, 100.0))


# --------------------------------------------------------------------------------------------
# Importances and ranking
# --------------------------------------------------------------------------------------------


def test_importance_and_shares(grid):
    f, w = grid
    assert importance(f, w) == pytest.approx(np.sum(w * np.abs(f)))
    I = np.array([3.0, 1.0, 0.0, 4.0])
    np.testing.assert_allclose(importance_shares(I), I / 8)
    assert importance_shares(I).sum() == pytest.approx(1.0)


def test_rank_metrics():
    rng = np.random.default_rng(4)
    I = rng.gamma(2.0, size=16)
    assert weighted_tau(I, I) == pytest.approx(1.0)
    assert weighted_tau(I, 2 * I) == pytest.approx(1.0)
    assert weighted_tau(I, -I) == pytest.approx(-1.0)
    assert spearman_rank(I, I) == pytest.approx(1.0)
    assert spearman_rank(I, -I) == pytest.approx(-1.0)
    J = I.copy()
    J[np.argsort(I)[:2]] = J[np.argsort(I)[:2]][::-1]  # swap the two least important
    K = I.copy()
    K[np.argsort(I)[-2:]] = K[np.argsort(I)[-2:]][::-1]  # swap the two most important
    # top-weighting: churn at the bottom costs less than the same churn at the top (D-14)
    assert weighted_tau(I, J) > weighted_tau(I, K)
    assert spearman_rank(I, J) == pytest.approx(spearman_rank(I, K))
    assert weighted_tau(I, K) == pytest.approx(stats.weightedtau(I, K).statistic)


def test_topk_jaccard_and_tv():
    Ia = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    Ib = np.array([5.0, 0.0, 3.0, 4.0, 1.0, 2.0])
    assert topk_jaccard(Ia, Ia, 3) == 1.0
    assert topk_jaccard(Ia, Ib, 3) == pytest.approx(2 / 4)  # {0,1,2} vs {0,3,2}
    assert topk_jaccard(Ia, Ib, 5) == pytest.approx(4 / 6)  # {0,1,2,3,4} vs {0,3,2,5,4}
    sa, sb = importance_shares(Ia), importance_shares(Ib)
    assert tv_distance(sa, sa) == 0.0
    assert tv_distance(sa, sb) == pytest.approx(0.5 * np.abs(sa - sb).sum())
    assert tv_distance(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])) == 1.0


# --------------------------------------------------------------------------------------------
# Floor building blocks: closed-form identities against explicit pair loops
# --------------------------------------------------------------------------------------------


def test_pair_identities_match_loops():
    rng = np.random.default_rng(5)
    K = 13
    w = rng.dirichlet(np.ones(K))
    F = rng.normal(size=(7, K))
    G = rng.normal(size=(4, K)) + 0.3
    loop_pairs = np.mean([d2(F[r], F[s], w) for r, s in itertools.combinations(range(7), 2)])
    loop_cross = np.mean([d2(F[r], G[s], w) for r in range(7) for s in range(4)])
    assert pair_mean_d2(F, w) == pytest.approx(loop_pairs, rel=1e-12)
    assert cross_mean_d2(F, G, w) == pytest.approx(loop_cross, rel=1e-12)
    loop_sd = np.sqrt(
        0.5 * np.mean([(F[r] - F[s]) ** 2 for r, s in itertools.combinations(range(7), 2)], axis=0)
    )
    np.testing.assert_allclose(cell_noise_sd(F), loop_sd, rtol=1e-12)
    s = sigma2(G, F, w)  # seed = G (4 fits), half = F (7 fits)
    assert s["seed"] == pytest.approx(0.5 * pair_mean_d2(G, w))
    assert s["tot"] == pytest.approx(0.5 * loop_pairs)
    assert s["samp"] == max(0.0, s["tot"] - s["seed"])
    assert sigma2(F, G, w)["samp"] >= 0.0


def random_pair(rng, K, R_s=5, R_b=10):
    """Random windows a, b with per-feature shapes of K_j cells and weights."""
    w = [rng.dirichlet(np.ones(k)) for k in K]

    def win(shift):
        return {
            "seed": [recentre(rng.normal(shift, 1.0, size=(R_s, k)) * rng.gamma(2), wj) for k, wj in zip(K, w)],
            "half": [recentre(rng.normal(shift, 1.5, size=(R_b, k)), wj) for k, wj in zip(K, w)],
        }

    return win(0.0), win(0.2), w


def test_drift_pair_matches_metric_functions():
    rng = np.random.default_rng(6)
    K = [50, 37, 12, 7, 2, 50]
    a, b, w = random_pair(rng, K)
    x_rep = [np.sort(rng.normal(size=k)) for k in K[:-1]] + [None]
    # make feature 0 monotone in both windows so that MA qualifies there
    for side, sgn in ((a, 1.0), (b, -1.0)):
        side["seed"][0] = recentre(side["seed"][0] + sgn * 10 * np.arange(50.0), w[0])
    rho_ov = 0.4
    out = drift_pair(a, b, w, rho_ov, x_rep)

    for j, k in enumerate(K):
        Sa, Sb, Ha, Hb = a["seed"][j], b["seed"][j], a["half"][j], b["half"][j]
        ma, mb = Sa.mean(0), Sb.mean(0)
        ga, gb = sigma2(Sa, Ha, w[j]), sigma2(Sb, Hb, w[j])
        cross = cross_mean_d2(Sa, Sb, w[j])
        assert out["d2_cross"][j] == pytest.approx(cross, rel=1e-12)
        assert out["s2_seed_a"][j] == pytest.approx(ga["seed"])
        assert out["s2_tot_b"][j] == pytest.approx(gb["tot"])
        assert out["s2_samp_a"][j] == pytest.approx(ga["samp"])
        assert out["DR"][j] == pytest.approx(cross / (ga["tot"] + gb["tot"]))
        den_adj = 2 * ((1 - rho_ov) * (ga["samp"] + gb["samp"]) / 2 + (ga["seed"] + gb["seed"]) / 2)
        assert out["DR_adj"][j] == pytest.approx(cross / den_adj)
        assert out["A_a"][j] == pytest.approx(amplitude(ma, w[j]))
        assert out["snr_b"][j] == pytest.approx(np.mean(amplitude(Sb, w[j]) ** 2) / gb["tot"])
        assert out["nd2"][j] == pytest.approx(nd2(ma, mb, w[j]))
        if k > 2:
            assert out["rho"][j] == pytest.approx(shape_corr(ma, mb, w[j]))
            assert out["rho_s"][j] == pytest.approx(shape_spearman(ma, mb))
        tau = np.sqrt(0.5 * (cell_noise_sd(Ha) ** 2 + cell_noise_sd(Hb) ** 2))
        sa, cov = sign_agreement(ma, mb, w[j], tau)
        np.testing.assert_allclose([out["SA"][j], out["SA_cov"][j]], [sa, cov])
        if x_rep[j] is None:
            assert np.isnan(out["MA"][j])
        else:
            np.testing.assert_equal(out["MA"][j], monotonicity_agreement(x_rep[j], ma, mb))
        assert out["I_a"][j] == pytest.approx(importance(ma, w[j]))
    assert out["MA"][0] == 0.0  # increasing in a, decreasing in b

    tot = out["s2_tot_a"] + out["s2_tot_b"]
    assert out["DR_tot"] == pytest.approx(out["d2_cross"].sum() / tot.sum())
    assert out["DR_adj_tot"] == pytest.approx(np.sum(out["d2_cross"]) / np.sum(out["d2_cross"] / out["DR_adj"]))
    assert out["seed_share_a"] == pytest.approx(out["s2_seed_a"].sum() / out["s2_tot_a"].sum())
    Ia, Ib = out["I_a"], out["I_b"]
    assert out["wtau"] == pytest.approx(weighted_tau(Ia, Ib))
    assert out["spearman_imp"] == pytest.approx(spearman_rank(Ia, Ib))
    assert out["jacc3"] == topk_jaccard(Ia, Ib, 3) and out["jacc5"] == topk_jaccard(Ia, Ib, 5)
    assert out["tv"] == pytest.approx(tv_distance(importance_shares(Ia), importance_shares(Ib)))

    # per-feature input layout gives the same result; rho = 0 gives DR_adj = DR when samp is unclipped
    per_feat = [{"seed": a["seed"][j], "half": a["half"][j]} for j in range(len(K))]
    out2 = drift_pair(per_feat, b, w, rho_ov, x_rep)
    np.testing.assert_equal(out2["DR"], out["DR"])
    out0 = drift_pair(a, b, w, 0.0)
    unclipped = (out0["s2_tot_a"] >= out0["s2_seed_a"]) & (out0["s2_tot_b"] >= out0["s2_seed_b"])
    np.testing.assert_allclose(out0["DR_adj"][unclipped], out0["DR"][unclipped])
    assert np.all(np.isnan(out0["MA"]))


def test_drift_pair_identical_windows():
    rng = np.random.default_rng(7)
    K = [30, 30, 5]
    a, _, w = random_pair(rng, K)
    out = drift_pair(a, a, w, 1.0)
    np.testing.assert_allclose(out["nd2"], 0.0, atol=1e-15)
    np.testing.assert_allclose(out["rho"], 1.0)
    np.testing.assert_allclose(out["SA"], 1.0)
    assert out["wtau"] == pytest.approx(1.0) and out["tv"] == 0.0 and out["jacc3"] == 1.0
    # only within-window seed spread is left in the cross distance: d2_cross = 2·(R-1)/R·s2_seed
    np.testing.assert_allclose(out["d2_cross"], 2 * 4 / 5 * out["s2_seed_a"])


def test_drift_pair_input_validation():
    rng = np.random.default_rng(8)
    a, b, w = random_pair(rng, [10, 10])
    with pytest.raises(ValueError):
        drift_pair(a, b, w, 1.5)
    with pytest.raises(ValueError):
        drift_pair(a, b, [w[0] * 2, w[1]], 0.0)
    with pytest.raises(ValueError):
        drift_pair(a, b, w[:1], 0.0)
    with pytest.raises(ValueError):
        drift_pair(a, b, w, 0.0, [np.arange(9.0), None])


# --------------------------------------------------------------------------------------------
# Monte Carlo: E[DR] = 1 without drift, 1 + d2(δ)/(2σ²_tot) with drift, DR_adj under overlap
# --------------------------------------------------------------------------------------------
#
# Generative model for one feature on K cells (all noise iid over cells):
#   full-window estimate of window m:     g_m + e_m,                e_m ~ N(0, σ²_samp)
#   seed-only fit r (protocol (i)):       g_m + e_m + u_r,          u_r ~ N(0, σ²_seed)
#   half-sample fit r (protocol (iii)):   g_m + e_m + h_r + u'_r,   h_r ~ N(0, σ²_samp)
# The seed fits share the window's sampling error e_m (same data, different seeds). Each half-
# sample estimate deviates from the full-window estimate by h_r, whose variance equals the
# full-window sampling variance σ²_samp (PLAN §6.4: m = n/2 without replacement gives
# (σ²/m)(1 − m/n) = σ²/n), and the h_r are independent across draws (independent half-samples,
# conditionally uncorrelated given the data). So pairs of half fits differ by h_r − h_s + u'_r − u'_s
# and σ²_tot = σ²_samp + σ²_seed, while the seed spread gives σ²_seed.
# Overlap ρ: the two windows share a fraction ρ of their rows, so Corr(e_a, e_b) = ρ; simulated as
# e_m = σ_samp (√ρ c + √(1−ρ) z_m) with a common c.
# Cross distance: E[d̄²_cross] = d2(δ) + 2(1−ρ)σ²_samp + 2σ²_seed per unit cell variance, hence
#   E[DR] ≈ (1−ρ)σ²_samp/σ²_tot + σ²_seed/σ²_tot + d2(δ)/(2σ²_tot),   E[DR_adj] ≈ 1 + d2(δ)/(2·…).
# Every replicate is re-centred under w; that maps all noise through the same projection, which
# scales every expected variance by c_P = 1 − Σω² and leaves the ratios unchanged (the drift δ is
# itself re-centred, so d2(δ) is its distance on the re-centred scale).

SIG_SEED2, SIG_SAMP2 = 0.4, 0.6
R_S, R_B = 5, 10


def simulate_pair(rng, w_list, g_list, delta_list, rho):
    a = {"seed": [], "half": []}
    b = {"seed": [], "half": []}
    ss, sp = np.sqrt(SIG_SEED2), np.sqrt(SIG_SAMP2)
    for w, g, delta in zip(w_list, g_list, delta_list):
        K = w.size
        c = rng.normal(size=K)
        for side, shift in ((a, 0.0), (b, 1.0)):
            e = sp * (np.sqrt(rho) * c + np.sqrt(1 - rho) * rng.normal(size=K))
            base = g + shift * delta + e
            side["seed"].append(recentre(base + ss * rng.normal(size=(R_S, K)), w))
            half = base + sp * rng.normal(size=(R_B, K)) + ss * rng.normal(size=(R_B, K))
            side["half"].append(recentre(half, w))
    return a, b


@pytest.fixture(scope="module")
def mc_setup():
    rng = np.random.default_rng(20261117)
    K = [30, 30, 20, 30]
    w_list = [rng.dirichlet(np.full(k, 5.0)) for k in K]
    x = [np.linspace(-1, 1, k) for k in K]
    g_list = [recentre(amp * np.tanh(2 * xi), w) for amp, xi, w in zip([2.0, 0.5, 1.0, 0.0], x, w_list)]
    c_P = np.array([1 - np.sum(w**2) for w in w_list])
    return rng, w_list, g_list, x, c_P


def run_mc(mc_setup, delta_list, rho, n_sims=300):
    rng, w_list, g_list, _, _ = mc_setup
    res = [drift_pair(*simulate_pair(rng, w_list, g_list, delta_list, rho), w_list, rho) for _ in range(n_sims)]
    return {k: np.array([r[k] for r in res]) for k in ("DR_tot", "DR_adj_tot", "DR", "seed_share_a", "s2_tot_a")}


def test_mc_no_drift_disjoint_windows(mc_setup):
    _, w_list, _, _, c_P = mc_setup
    zero = [np.zeros(w.size) for w in w_list]
    r = run_mc(mc_setup, zero, rho=0.0)
    assert abs(np.median(r["DR_tot"]) - 1.0) < 0.1
    assert abs(np.median(r["DR_adj_tot"]) - 1.0) < 0.1
    assert np.all(np.abs(np.median(r["DR"], axis=0) - 1.0) < 0.15)
    assert abs(np.median(r["seed_share_a"]) - SIG_SEED2 / (SIG_SEED2 + SIG_SAMP2)) < 0.05
    # floor level: σ²_tot = (σ²_seed + σ²_samp)·(1 − Σω²) after re-centring
    np.testing.assert_allclose(np.mean(r["s2_tot_a"], axis=0), (SIG_SEED2 + SIG_SAMP2) * c_P, rtol=0.05)


def test_mc_drift(mc_setup):
    _, w_list, _, x, c_P = mc_setup
    delta = [recentre(s * xi, w) for s, xi, w in zip([1.6, 0.8, 2.0, 1.2], x, w_list)]
    d2_delta = np.array([d2(dl, 0.0, w) for dl, w in zip(delta, w_list)])
    s2_tot = (SIG_SEED2 + SIG_SAMP2) * c_P
    expected_j = 1 + d2_delta / (2 * s2_tot)
    expected_tot = 1 + d2_delta.sum() / (2 * s2_tot.sum())
    assert expected_tot > 1.3  # the test has power
    r = run_mc(mc_setup, delta, rho=0.0)
    assert np.median(r["DR_tot"]) == pytest.approx(expected_tot, rel=0.05)
    np.testing.assert_allclose(np.median(r["DR"], axis=0), expected_j, rtol=0.08)


def test_mc_overlap_adjustment(mc_setup):
    _, w_list, _, _, _ = mc_setup
    zero = [np.zeros(w.size) for w in w_list]
    rho = 0.6
    r = run_mc(mc_setup, zero, rho=rho)
    assert abs(np.median(r["DR_adj_tot"]) - 1.0) < 0.1
    # the unadjusted ratio is pulled below 1 by the shared sampling noise
    expected_dr = ((1 - rho) * SIG_SAMP2 + SIG_SEED2) / (SIG_SAMP2 + SIG_SEED2)
    assert np.median(r["DR_tot"]) == pytest.approx(expected_dr, abs=0.04)
    assert np.median(r["DR_tot"]) < 0.85


# --------------------------------------------------------------------------------------------
# Cost: drift_pair runs ~3,000 times per slice
# --------------------------------------------------------------------------------------------


def test_drift_pair_speed():
    rng = np.random.default_rng(9)
    K = [50] * 14 + [24, 7]
    a, b, w = random_pair(rng, K)
    x_rep = [np.arange(float(k)) for k in K[:14]] + [None, None]
    drift_pair(a, b, w, 0.5, x_rep)
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        drift_pair(a, b, w, 0.5, x_rep)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.05
