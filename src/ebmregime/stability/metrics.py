"""Per-term shape metrics and importance-ranking metrics (PLAN §6.2, §6.3; D-14, D-15).

Every shape function takes **re-centred cell shapes** of one feature on a common cell grid
(``f`` of shape ``(K,)`` or ``(..., K)``; cells on the last axis) and the cell weights ``w`` of
shape ``(K,)`` with ``Σ w = 1`` (``explain.cells.CellGrid.weights[j]``). Leading dimensions
broadcast. Nothing here re-centres: the caller passes f̃ (PLAN §6.1 step 4).

Defaults that are pre-registered numbers (``min_abs=0.3``, ``mult=2.0``) mirror
``prereg.stability``; ``tests/test_stability.py`` checks them against the config.

Degenerate cases return NaN rather than raising:
* ``nd2`` when both amplitudes are 0;
* ``shape_corr`` / ``shape_spearman`` when either shape is flat (constant over the cells);
* ``sign_agreement`` when no cell qualifies (coverage 0);
* ``monotonicity_agreement`` when either |Spearman(x_rep, f)| ≤ ``min_abs`` (or a shape is flat);
* ``local_monotonicity_agreement`` when no step qualifies;
* ``importance_shares`` when Σ I = 0.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# A shape counts as flat when its density-weighted SD is at most this fraction of its largest
# absolute value: a constant shape re-centred in floating point leaves residuals of ~1e-16·|c|.
_FLAT_RTOL = 1e-12


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _safe_div(num, den):
    """num / den with NaN where den == 0 (no warnings)."""
    num = _arr(num)
    den = _arr(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(den == 0, np.nan, out)


# --------------------------------------------------------------------------------------------
# Per-term shape metrics (PLAN §6.2)
# --------------------------------------------------------------------------------------------


def d2(fa, fb, w):
    """Density-weighted squared L2 distance ``Σ_k ω_k (fa(k) − fb(k))²`` over the last axis."""
    diff = _arr(fa) - _arr(fb)
    return np.sum(_arr(w) * diff * diff, axis=-1)


def amplitude(f, w):
    """Amplitude ``A = (Σ_k ω_k f(k)²)^{1/2}``: the density-weighted SD of a re-centred shape."""
    f = _arr(f)
    return np.sqrt(np.sum(_arr(w) * f * f, axis=-1))


def nd2(fa, fb, w):
    """Normalised distance ``d² / ((A_a² + A_b²) / 2)`` in [0, 4]; NaN if both amplitudes are 0."""
    fa, fb, w = _arr(fa), _arr(fb), _arr(w)
    den = 0.5 * (np.sum(w * fa * fa, axis=-1) + np.sum(w * fb * fb, axis=-1))
    return _safe_div(d2(fa, fb, w), den)


def _flat(var, f):
    """True where a shape with weighted variance ``var`` is constant up to rounding."""
    scale = np.max(np.abs(f), axis=-1)
    return var <= (_FLAT_RTOL * scale) ** 2


def _weighted_pearson(a, b, w):
    a, b, w = np.broadcast_arrays(_arr(a), _arr(b), _arr(w))
    ca = a - np.sum(w * a, axis=-1, keepdims=True)
    cb = b - np.sum(w * b, axis=-1, keepdims=True)
    va = np.sum(w * ca * ca, axis=-1)
    vb = np.sum(w * cb * cb, axis=-1)
    cov = np.sum(w * ca * cb, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(va * vb)
    r = np.clip(r, -1.0, 1.0)
    return np.where(_flat(va, a) | _flat(vb, b), np.nan, r)


def shape_corr(fa, fb, w):
    """Density-weighted Pearson correlation over cells (ρ_j); NaN if either shape is flat.

    The weighted means are subtracted explicitly, so it is also correct for shapes that are not
    exactly re-centred.
    """
    return _weighted_pearson(fa, fb, w)


def _rank(x):
    """Average ranks along the last axis (ties share the mean rank)."""
    return stats.rankdata(_arr(x), method="average", axis=-1)


def shape_spearman(fa, fb):
    """Spearman correlation over cells (unweighted, average ranks for ties; secondary metric).

    NaN if either shape is constant over the cells.
    """
    ra, rb = _rank(fa), _rank(fb)
    k = ra.shape[-1]
    return _weighted_pearson(ra, rb, np.full(k, 1.0 / k))


def sign_agreement(fa, fb, w, tau):
    """Sign agreement and its coverage: ``(SA, Σ_{k∈K} ω_k)``.

    ``K`` keeps cell k unless **both** ``|fa(k)| < tau_k`` and ``|fb(k)| < tau_k`` (PLAN §6.2);
    ``tau`` is a scalar or a ``(K,)`` array of per-cell noise SDs (``floor.cell_noise_sd``).
    ``SA = Σ_{k∈K} ω_k 1[sign fa(k) = sign fb(k)] / Σ_{k∈K} ω_k``; NaN if the coverage is 0.
    ``np.sign`` is used, so an exact 0 has its own sign (it agrees only with another 0).
    """
    fa, fb, w = _arr(fa), _arr(fb), _arr(w)
    tau = _arr(tau)
    keep = ~((np.abs(fa) < tau) & (np.abs(fb) < tau))
    agree = np.sign(fa) == np.sign(fb)
    cov = np.sum(w * keep, axis=-1)
    sa = _safe_div(np.sum(w * (keep & agree), axis=-1), cov)
    return sa, cov


def monotonicity_agreement(x_rep, fa, fb, min_abs: float = 0.3):
    """Monotonicity agreement MA_j for one feature: 1.0 or 0.0, NaN if not qualifying.

    ``ρ_m = Spearman(x_rep, f_m)`` over the cells (unweighted, average ranks). The term qualifies
    only if ``|ρ_a| > min_abs`` and ``|ρ_b| > min_abs``; then MA = 1[sign ρ_a = sign ρ_b].
    A flat shape (Spearman undefined) does not qualify. 1-D inputs.
    """
    rx = _rank(x_rep)
    k = rx.shape[-1]
    u = np.full(k, 1.0 / k)
    ra = float(_weighted_pearson(rx, _rank(fa), u))
    rb = float(_weighted_pearson(rx, _rank(fb), u))
    if not (abs(ra) > min_abs and abs(rb) > min_abs):  # also False for NaN
        return np.nan
    return 1.0 if np.sign(ra) == np.sign(rb) else 0.0


def local_monotonicity_agreement(fa, fb, step_sd, mult: float = 2.0):
    """Local monotonicity agreement (secondary): share of qualifying steps with equal sign.

    Steps are ``Δ_k = f(k+1) − f(k)`` between consecutive entries of the given shapes (the caller
    passes decile-aggregated shapes for the PLAN's decile-to-decile version). A step qualifies if
    ``max(|Δ_a|, |Δ_b|) > mult · step_sd_k``; ``step_sd`` is a scalar or ``(K−1,)`` array, e.g.
    ``floor.cell_noise_sd(np.diff(F_half, axis=1))``. Unweighted share; NaN if no step qualifies.
    """
    da = np.diff(_arr(fa), axis=-1)
    db = np.diff(_arr(fb), axis=-1)
    q = np.maximum(np.abs(da), np.abs(db)) > mult * _arr(step_sd)
    n = np.sum(q, axis=-1)
    return _safe_div(np.sum(q & (np.sign(da) == np.sign(db)), axis=-1), n)


# --------------------------------------------------------------------------------------------
# Importances and ranking (PLAN §6.3)
# --------------------------------------------------------------------------------------------


def importance(f, w):
    """Unified importance ``I = Σ_k ω_k |f(k)|`` on the common reference density."""
    return np.sum(_arr(w) * np.abs(_arr(f)), axis=-1)


def importance_shares(I):
    """Shares ``s_j = I_j / Σ_j I_j`` over the last axis; NaN if Σ I = 0."""
    I = _arr(I)
    return _safe_div(I, np.sum(I, axis=-1, keepdims=True))


def weighted_tau(Ia, Ib) -> float:
    """Weighted Kendall τ of two importance vectors (D-14).

    ``scipy.stats.weightedtau`` with its defaults: hyperbolic weigher ``1/(r+1)``, additive
    weights, and ``rank=True`` (the weighted τ is averaged over the rankings by decreasing ``Ia``
    and by decreasing ``Ib``). Returns the statistic.
    """
    return float(stats.weightedtau(_arr(Ia), _arr(Ib)).statistic)


def spearman_rank(Ia, Ib) -> float:
    """Spearman correlation of two importance vectors (average ranks for ties)."""
    return float(shape_spearman(Ia, Ib))


def _topk(I, k):
    # Descending order; ties broken by feature index (stable sort), so the result is deterministic.
    return set(np.argsort(-_arr(I), kind="stable")[:k].tolist())


def topk_jaccard(Ia, Ib, k: int) -> float:
    """Jaccard index ``|T_a ∩ T_b| / |T_a ∪ T_b|`` of the top-k features by importance.

    Ties at the boundary are broken by feature index (lower index first).
    """
    a, b = _topk(Ia, k), _topk(Ib, k)
    return len(a & b) / len(a | b)


def tv_distance(sa, sb):
    """Total-variation distance ``½ Σ_j |s_a,j − s_b,j|`` between importance shares."""
    return 0.5 * np.sum(np.abs(_arr(sa) - _arr(sb)), axis=-1)
