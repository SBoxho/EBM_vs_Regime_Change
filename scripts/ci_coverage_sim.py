"""Coverage of the prereg stationary-bootstrap median CI on lag-12 pair series built from
overlapping 12-month windows (D-58). Usage: python scripts/ci_coverage_sim.py"""
import numpy as np
from ebmregime.stats import median_ci
rng = np.random.default_rng(3)
K, M, L = 30, 95, 12          # shape dims, months, window length; 83 windows, 71 lag-12 pairs
def series(n_sets):
    e = rng.standard_normal((n_sets, M, K)) * rng.uniform(0.5, 1.5, K)   # monthly shocks
    c = np.cumsum(e, axis=1); c = np.concatenate([np.zeros((n_sets, 1, K)), c], 1)
    w = (c[:, L:] - c[:, :-L]) / L                                         # 84 window means
    w = w + 0.3 * rng.standard_normal(w.shape) / np.sqrt(L)                # fit noise not tied to months
    return ((w[:, :-12] - w[:, 12:]) ** 2).sum(-1)[:, :71]                  # G of lag-12 pairs
big = series(21 * 50)
true = np.median((big[0::21] / np.stack([big[i::21] for i in range(1, 21)]).mean(0)).ravel())
print("true median", round(true, 4))
for mult in (1, 2):
    c1 = []; ct = []
    for k in range(200):
        G = series(21)
        x = G[0] / G[1:].mean(0)
        ci = median_ci(x, delta=12, reps=1999, seed=k, multiplier=mult)
        c1.append(ci["lo"] <= 1 <= ci["hi"]); ct.append(ci["lo"] <= true <= ci["hi"])
    print("block multiplier", mult, "cover 1", np.mean(c1), "cover true median", np.mean(ct))
