"""Noise floor, variance decomposition and drift ratio (PLAN §6.4; D-16, D-17, D-50).

Shapes of one feature arrive as ``(R, K)`` arrays: R replicate fits (rows) of re-centred cell
shapes on a common grid of K cells with weights ``w`` (``Σ w = 1``). Protocol (i) gives the
``seed`` replicates (R_s = 5 full-window fits), protocol (iii) the ``half`` replicates
(R_b = 10 fits on independent half-samples).

Pair means are computed in closed form instead of looping over pairs. For one cell, with
replicate values x_1..x_R, mean x̄ and ``s² = Σ (x_r − x̄)² / (R − 1)`` (ddof = 1):

* **(I1) within-set pairs:** ``Σ_{r<s} (x_r − x_s)² = R Σ_r (x_r − x̄)²``, so the mean over the
  R(R−1)/2 distinct pairs is ``2 s²``  (= ``2·R/(R−1)·var₀`` with the ddof-0 variance var₀).
  Hence ``pair_mean_d2(F, w) = 2 Σ_k ω_k s_k²`` and ``σ² = ½·pair_mean_d2 = Σ_k ω_k s_k²``.
* **(I2) cross pairs:** for sets x (R_a) and y (R_b),
  ``mean_{r,s} (x_r − y_s)² = (x̄ − ȳ)² + var₀(x) + var₀(y)`` (ddof 0), so
  ``cross_mean_d2(Fa, Fb, w) = Σ_k ω_k [(x̄_k − ȳ_k)² + var₀ x_k + var₀ y_k]``.
* **(I3) mean amplitude:** ``mean_r A_r² = Σ_k ω_k mean_r f_r(k)²`` (A² is linear in f²).

Both identities are exact (tests compare them with explicit loops over pairs).

``drift_pair`` concatenates the J features' cells into one axis and reduces per feature with
``np.add.reduceat``, so its cost hardly grows with J; only the rank-based metrics (MA, ρ_S) loop
over features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .metrics import (
    _FLAT_RTOL,
    _arr,
    _rank,
    _safe_div,
    _weighted_pearson,
    importance_shares,
    spearman_rank,
    topk_jaccard,
    tv_distance,
    weighted_tau,
)

PROTOCOLS = ("seed", "half")


# --------------------------------------------------------------------------------------------
# Single-feature building blocks
# --------------------------------------------------------------------------------------------


def _check_reps(F, name="F") -> np.ndarray:
    F = _arr(F)
    if F.ndim != 2 or F.shape[0] < 2:
        raise ValueError(f"{name} must have shape (R, K) with R >= 2, got {F.shape}")
    return F


def pair_mean_d2(F, w):
    """Mean of ``d2(F[r], F[s], w)`` over the R(R−1)/2 distinct pairs of rows (identity I1)."""
    F = _check_reps(F)
    return 2.0 * float(np.sum(_arr(w) * F.var(axis=0, ddof=1)))


def cross_mean_d2(Fa, Fb, w):
    """Mean of ``d2(Fa[r], Fb[s], w)`` over all Ra × Rb cross pairs (identity I2)."""
    Fa, Fb = _arr(Fa), _arr(Fb)
    if Fa.ndim != 2 or Fb.ndim != 2 or Fa.shape[1] != Fb.shape[1]:
        raise ValueError(f"need (Ra, K) and (Rb, K) arrays, got {Fa.shape} and {Fb.shape}")
    diff = Fa.mean(axis=0) - Fb.mean(axis=0)
    return float(np.sum(_arr(w) * (diff * diff + Fa.var(axis=0) + Fb.var(axis=0))))


def sigma2(F_seed, F_half, w) -> dict[str, float]:
    """Variance decomposition of one term in one window (PLAN §6.4).

    ``seed = ½·pair_mean_d2(F_seed)``, ``tot = ½·pair_mean_d2(F_half)``,
    ``samp = max(0, tot − seed)``.
    """
    seed = 0.5 * pair_mean_d2(F_seed, w)
    tot = 0.5 * pair_mean_d2(F_half, w)
    return {"seed": seed, "tot": tot, "samp": max(0.0, tot - seed)}


def cell_noise_sd(F_half):
    """Per-cell noise SD ``sqrt(½ · mean over half pairs of (f_r(k) − f_s(k))²)``, shape (K,).

    By identity I1 this is the ddof-1 SD of the replicates in each cell. It is the τ_jk of the
    sign-agreement exclusion (PLAN §6.2).
    """
    return np.sqrt(_check_reps(F_half, "F_half").var(axis=0, ddof=1))


# --------------------------------------------------------------------------------------------
# Full per-pair computation
# --------------------------------------------------------------------------------------------


def _by_protocol(shapes) -> dict[str, list[np.ndarray]]:
    """Accept ``{"seed": [F_j], "half": [F_j]}`` or ``[{"seed": F_j, "half": F_j}, ...]``."""
    if isinstance(shapes, Mapping):
        return {p: [_arr(F) for F in shapes[p]] for p in PROTOCOLS}
    return {p: [_arr(s[p]) for s in shapes] for p in PROTOCOLS}


def drift_pair(
    shapes_a,
    shapes_b,
    weights: Sequence[np.ndarray],
    overlap: float,
    x_rep: Sequence[np.ndarray | None] | None = None,
    *,
    min_abs_spearman: float = 0.3,
    top_k: Sequence[int] = (3, 5),
) -> dict[str, np.ndarray | float]:
    """All per-pair stability quantities for windows a and b (PLAN §6.2-§6.4).

    Parameters
    ----------
    shapes_a, shapes_b
        Re-centred cell shapes of each window on the pair's common cells, either as
        ``{"seed": [F_0, ..., F_{J-1}], "half": [...]}`` or as a per-feature list
        ``[{"seed": F_0, "half": H_0}, ...]``. ``seed`` arrays are ``(R_s, K_j)`` (protocol (i)),
        ``half`` arrays ``(R_b, K_j)`` (protocol (iii)); R ≥ 2, K_j ≥ 1.
    weights
        Per feature the cell weights ``(K_j,)``, each summing to 1.
    overlap
        ρ = ``splits.overlap_fraction(tau_a, tau_b, train_months)`` in [0, 1].
    x_rep
        Per feature the cell representative values ``(K_j,)`` for monotonicity agreement, or
        None (nominal features, or all features) → MA = NaN.
    min_abs_spearman, top_k
        ``prereg.stability.monotonicity_min_abs_spearman`` and ``top_k_jaccard``.

    Returns
    -------
    dict
        Per-feature float64 arrays of length J:
        ``d2_cross`` (mean over the R_s×R_s cross pairs of seed fits), ``s2_seed_a``,
        ``s2_tot_a``, ``s2_samp_a``, ``s2_seed_b``, ``s2_tot_b``, ``s2_samp_b``, ``DR``,
        ``DR_adj``, ``A_a``, ``A_b`` (amplitude of the seed-mean shape), ``snr_a``, ``snr_b``
        (mean over seed fits of A² / s2_tot), ``nd2``, ``rho``, ``rho_s`` (Spearman over cells,
        secondary), ``SA``, ``SA_cov``, ``MA``, ``I_a``, ``I_b`` (importance of the seed-mean
        shapes); scalars ``DR_tot``, ``DR_adj_tot``, ``seed_share_a``, ``seed_share_b``,
        ``wtau``, ``spearman_imp``, ``jacc{k}`` for k in ``top_k`` (``jacc3``, ``jacc5``),
        ``tv``. nd2, ρ, ρ_S, SA, MA and the importance metrics use the seed-mean shapes.

    Definitions: ``DR_j = d2_cross / (s2_tot_a + s2_tot_b)``;
    ``DR_adj_j = d2_cross / (2[(1 − ρ)·s̄2_samp + s̄2_seed])`` with s̄ the average of the two
    windows; ``*_tot`` = Σ numerators / Σ denominators over features; SA's per-cell τ =
    ``sqrt(½(cell_noise_sd_a² + cell_noise_sd_b²))`` from the half replicates. Any ratio with a
    zero denominator is NaN.
    """
    rho_ov = float(overlap)
    if not 0.0 <= rho_ov <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}")
    sa, sb = _by_protocol(shapes_a), _by_protocol(shapes_b)
    w_list = [_arr(w) for w in weights]
    J = len(w_list)
    K = np.array([w.shape[0] for w in w_list])
    if J == 0 or np.any(K < 1):
        raise ValueError("need at least one feature and one cell per feature")
    for side in (sa, sb):
        for p in PROTOCOLS:
            if len(side[p]) != J:
                raise ValueError(f"{p}: expected {J} features, got {len(side[p])}")
            for j, F in enumerate(side[p]):
                if F.ndim != 2 or F.shape[1] != K[j] or F.shape[0] < 2:
                    raise ValueError(f"{p} shapes of feature {j}: expected (R>=2, {K[j]}), got {F.shape}")
    if x_rep is not None and len(x_rep) != J:
        raise ValueError(f"x_rep: expected {J} entries, got {len(x_rep)}")

    starts = np.concatenate([[0], np.cumsum(K)[:-1]])
    seg = np.repeat(np.arange(J), K)
    w = np.concatenate(w_list)
    if not np.allclose(np.add.reduceat(w, starts), 1.0, rtol=0, atol=1e-9):
        raise ValueError("cell weights of every feature must sum to 1")

    def per_feature(x):
        return np.add.reduceat(x, starts, axis=-1)

    seed_a = np.concatenate(sa["seed"], axis=1)
    seed_b = np.concatenate(sb["seed"], axis=1)
    half_a = np.concatenate(sa["half"], axis=1)
    half_b = np.concatenate(sb["half"], axis=1)

    # Variance decomposition (identity I1): sigma2 = Σ ω s² with ddof-1 per-cell variances.
    v_half_a = half_a.var(axis=0, ddof=1)
    v_half_b = half_b.var(axis=0, ddof=1)
    s2_seed_a = per_feature(w * seed_a.var(axis=0, ddof=1))
    s2_seed_b = per_feature(w * seed_b.var(axis=0, ddof=1))
    s2_tot_a = per_feature(w * v_half_a)
    s2_tot_b = per_feature(w * v_half_b)
    s2_samp_a = np.maximum(0.0, s2_tot_a - s2_seed_a)
    s2_samp_b = np.maximum(0.0, s2_tot_b - s2_seed_b)

    # Cross distance over the R_s x R_s seed pairs (identity I2).
    ma = seed_a.mean(axis=0)
    mb = seed_b.mean(axis=0)
    diff = ma - mb
    dd = per_feature(w * diff * diff)  # d2 of the seed-mean shapes
    d2_cross = dd + per_feature(w * (seed_a.var(axis=0) + seed_b.var(axis=0)))

    den_dr = s2_tot_a + s2_tot_b
    den_adj = 2.0 * ((1.0 - rho_ov) * 0.5 * (s2_samp_a + s2_samp_b) + 0.5 * (s2_seed_a + s2_seed_b))

    # Amplitudes of the seed-mean shapes and SNR (identity I3).
    A2_a = per_feature(w * ma * ma)
    A2_b = per_feature(w * mb * mb)
    snr_a = _safe_div(per_feature(w * np.mean(seed_a * seed_a, axis=0)), s2_tot_a)
    snr_b = _safe_div(per_feature(w * np.mean(seed_b * seed_b, axis=0)), s2_tot_b)

    # Weighted Pearson per feature (means subtracted explicitly, as metrics.shape_corr).
    ca = ma - per_feature(w * ma)[seg]
    cb = mb - per_feature(w * mb)[seg]
    va = per_feature(w * ca * ca)
    vb = per_feature(w * cb * cb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.clip(per_feature(w * ca * cb) / np.sqrt(va * vb), -1.0, 1.0)
    flat_a = va <= (_FLAT_RTOL * np.maximum.reduceat(np.abs(ma), starts)) ** 2
    flat_b = vb <= (_FLAT_RTOL * np.maximum.reduceat(np.abs(mb), starts)) ** 2
    rho = np.where(flat_a | flat_b, np.nan, rho)

    # Sign agreement with the per-cell noise SD of the half replicates, averaged over windows.
    tau = np.sqrt(0.5 * (v_half_a + v_half_b))
    keep = ~((np.abs(ma) < tau) & (np.abs(mb) < tau))
    agree = np.sign(ma) == np.sign(mb)
    sa_cov = per_feature(w * keep)
    sa_val = _safe_div(per_feature(w * (keep & agree)), sa_cov)

    # Rank-based per-feature metrics (loop over features; ranks are cheap at K ~ 50).
    rho_s = np.full(J, np.nan)
    ma_agree = np.full(J, np.nan)
    for j in range(J):
        sl = slice(starts[j], starts[j] + K[j])
        u = np.full(K[j], 1.0 / K[j])
        ra, rb = _rank(ma[sl]), _rank(mb[sl])
        rho_s[j] = _weighted_pearson(ra, rb, u)
        xr = None if x_rep is None else x_rep[j]
        if xr is not None:
            rx = _rank(xr)
            if rx.shape[0] != K[j]:
                raise ValueError(f"x_rep of feature {j}: expected {K[j]} cells, got {rx.shape[0]}")
            qa = float(_weighted_pearson(rx, ra, u))
            qb = float(_weighted_pearson(rx, rb, u))
            if abs(qa) > min_abs_spearman and abs(qb) > min_abs_spearman:
                ma_agree[j] = 1.0 if np.sign(qa) == np.sign(qb) else 0.0

    # Importances of the seed-mean shapes and ranking metrics.
    I_a = per_feature(w * np.abs(ma))
    I_b = per_feature(w * np.abs(mb))

    out: dict[str, np.ndarray | float] = {
        "d2_cross": d2_cross,
        "s2_seed_a": s2_seed_a,
        "s2_tot_a": s2_tot_a,
        "s2_samp_a": s2_samp_a,
        "s2_seed_b": s2_seed_b,
        "s2_tot_b": s2_tot_b,
        "s2_samp_b": s2_samp_b,
        "DR": _safe_div(d2_cross, den_dr),
        "DR_adj": _safe_div(d2_cross, den_adj),
        "A_a": np.sqrt(A2_a),
        "A_b": np.sqrt(A2_b),
        "snr_a": snr_a,
        "snr_b": snr_b,
        "nd2": _safe_div(dd, 0.5 * (A2_a + A2_b)),
        "rho": rho,
        "rho_s": rho_s,
        "SA": sa_val,
        "SA_cov": sa_cov,
        "MA": ma_agree,
        "I_a": I_a,
        "I_b": I_b,
        "DR_tot": float(_safe_div(d2_cross.sum(), den_dr.sum())),
        "DR_adj_tot": float(_safe_div(d2_cross.sum(), den_adj.sum())),
        "seed_share_a": float(_safe_div(s2_seed_a.sum(), s2_tot_a.sum())),
        "seed_share_b": float(_safe_div(s2_seed_b.sum(), s2_tot_b.sum())),
        "wtau": weighted_tau(I_a, I_b) if J > 1 else np.nan,
        "spearman_imp": spearman_rank(I_a, I_b) if J > 1 else np.nan,
    }
    for k in top_k:
        out[f"jacc{k}"] = topk_jaccard(I_a, I_b, k)
    out["tv"] = float(tv_distance(importance_shares(I_a), importance_shares(I_b)))
    return out
