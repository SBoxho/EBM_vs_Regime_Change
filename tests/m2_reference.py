"""Lead's independent reference implementation of the PLAN §6 metrics (naive loops, no vectorisation).

Written from PLAN §6.2-§6.4 and D-50 only, without reading the implementers' code, so that
``tests/test_m2_verify.py`` can compare the two. Slow by design; for small synthetic inputs only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats


def d2(fa, fb, w):
    return float(sum(w[k] * (fa[k] - fb[k]) ** 2 for k in range(len(w))))


def amp(f, w):
    return math.sqrt(sum(w[k] * f[k] ** 2 for k in range(len(w))))


def nd2(fa, fb, w):
    den = (amp(fa, w) ** 2 + amp(fb, w) ** 2) / 2
    return d2(fa, fb, w) / den if den > 0 else float("nan")


def wcorr(fa, fb, w):
    ma = sum(w[k] * fa[k] for k in range(len(w)))
    mb = sum(w[k] * fb[k] for k in range(len(w)))
    cov = sum(w[k] * (fa[k] - ma) * (fb[k] - mb) for k in range(len(w)))
    va = sum(w[k] * (fa[k] - ma) ** 2 for k in range(len(w)))
    vb = sum(w[k] * (fb[k] - mb) ** 2 for k in range(len(w)))
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def sign_agreement(fa, fb, w, tau):
    num = den = 0.0
    for k in range(len(w)):
        if abs(fa[k]) < tau[k] and abs(fb[k]) < tau[k]:
            continue
        den += w[k]
        num += w[k] * (np.sign(fa[k]) == np.sign(fb[k]))
    return (num / den if den > 0 else float("nan")), den


def mono_agreement(x, fa, fb, thr=0.3):
    ra = stats.spearmanr(x, fa).statistic
    rb = stats.spearmanr(x, fb).statistic
    if not (abs(ra) > thr and abs(rb) > thr):
        return float("nan")
    return float(np.sign(ra) == np.sign(rb))


def importance(f, w):
    return sum(w[k] * abs(f[k]) for k in range(len(w)))


def jaccard_topk(Ia, Ib, k):
    A = set(np.argsort(-np.asarray(Ia), kind="stable")[:k])
    B = set(np.argsort(-np.asarray(Ib), kind="stable")[:k])
    return len(A & B) / len(A | B)


def tv(Ia, Ib):
    sa = np.asarray(Ia) / sum(Ia)
    sb = np.asarray(Ib) / sum(Ib)
    return 0.5 * float(sum(abs(sa - sb)))


def half_mean_pair_d2(F, w):
    """½ · mean over distinct pairs (r < s) of d2(F[r], F[s])."""
    pairs = list(itertools.combinations(range(len(F)), 2))
    return 0.5 * sum(d2(F[r], F[s], w) for r, s in pairs) / len(pairs)


def cross_d2(Fa, Fb, w):
    return sum(d2(a, b, w) for a in Fa for b in Fb) / (len(Fa) * len(Fb))


def cell_tau(F_half_a, F_half_b):
    """Per-cell noise SD: sqrt of the two-window average of ½·mean over half pairs of the cell diff²."""
    K = len(F_half_a[0])
    out = []
    for k in range(K):
        v = []
        for F in (F_half_a, F_half_b):
            pairs = list(itertools.combinations(range(len(F)), 2))
            v.append(0.5 * sum((F[r][k] - F[s][k]) ** 2 for r, s in pairs) / len(pairs))
        out.append(math.sqrt(sum(v) / 2))
    return out


def drift(seed_a, half_a, seed_b, half_b, weights, overlap):
    """Per-feature lists in, dict of per-feature arrays and totals out (PLAN §6.4, D-50)."""
    J = len(weights)
    r = {k: [] for k in ("d2_cross", "s2_seed_a", "s2_tot_a", "s2_samp_a", "s2_seed_b", "s2_tot_b",
                         "s2_samp_b", "DR", "DR_adj", "snr_a", "snr_b", "A_a", "A_b", "rho", "nd2")}
    num_adj = den_adj = num = den = 0.0
    for j in range(J):
        w = weights[j]
        sa, ta = half_mean_pair_d2(seed_a[j], w), half_mean_pair_d2(half_a[j], w)
        sb, tb = half_mean_pair_d2(seed_b[j], w), half_mean_pair_d2(half_b[j], w)
        pa, pb = max(0.0, ta - sa), max(0.0, tb - sb)
        dc = cross_d2(seed_a[j], seed_b[j], w)
        dadj_den = 2 * ((1 - overlap) * (pa + pb) / 2 + (sa + sb) / 2)
        ma = np.mean(seed_a[j], axis=0)
        mb = np.mean(seed_b[j], axis=0)
        r["d2_cross"].append(dc)
        r["s2_seed_a"].append(sa); r["s2_tot_a"].append(ta); r["s2_samp_a"].append(pa)
        r["s2_seed_b"].append(sb); r["s2_tot_b"].append(tb); r["s2_samp_b"].append(pb)
        r["DR"].append(dc / (ta + tb))
        r["DR_adj"].append(dc / dadj_den)
        r["snr_a"].append(np.mean([amp(f, w) ** 2 for f in seed_a[j]]) / ta)
        r["snr_b"].append(np.mean([amp(f, w) ** 2 for f in seed_b[j]]) / tb)
        r["A_a"].append(amp(ma, w)); r["A_b"].append(amp(mb, w))
        r["rho"].append(wcorr(ma, mb, w)); r["nd2"].append(nd2(ma, mb, w))
        num += dc; den += ta + tb; num_adj += dc; den_adj += dadj_den
    out = {k: np.asarray(v) for k, v in r.items()}
    out["DR_tot"] = num / den
    out["DR_adj_tot"] = num_adj / den_adj
    out["seed_share_a"] = sum(r["s2_seed_a"]) / sum(r["s2_tot_a"])
    out["seed_share_b"] = sum(r["s2_seed_b"]) / sum(r["s2_tot_b"])
    Ia = [importance(np.mean(seed_a[j], axis=0), weights[j]) for j in range(J)]
    Ib = [importance(np.mean(seed_b[j], axis=0), weights[j]) for j in range(J)]
    out["wtau"] = stats.weightedtau(Ia, Ib).statistic
    out["jacc3"] = jaccard_topk(Ia, Ib, 3)
    out["tv"] = tv(Ia, Ib)
    return out
