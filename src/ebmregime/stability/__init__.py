"""Stability metrics, noise floor and drift ratio (PLAN §6.2-§6.4).

``stability.surrogate`` (PLAN §9.3) is a separate module and is deliberately not imported here;
import it as ``from ebmregime.stability import surrogate``.
"""

from .floor import cell_noise_sd, cross_mean_d2, drift_pair, pair_mean_d2, sigma2
from .metrics import (
    amplitude,
    d2,
    importance,
    importance_shares,
    local_monotonicity_agreement,
    monotonicity_agreement,
    nd2,
    shape_corr,
    shape_spearman,
    sign_agreement,
    spearman_rank,
    topk_jaccard,
    tv_distance,
    weighted_tau,
)

__all__ = [
    "amplitude",
    "cell_noise_sd",
    "cross_mean_d2",
    "d2",
    "drift_pair",
    "importance",
    "importance_shares",
    "local_monotonicity_agreement",
    "monotonicity_agreement",
    "nd2",
    "pair_mean_d2",
    "shape_corr",
    "shape_spearman",
    "sigma2",
    "sign_agreement",
    "spearman_rank",
    "topk_jaccard",
    "tv_distance",
    "weighted_tau",
]
