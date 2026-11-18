"""Diagnosis for D-59: train/validation-tail near-duplicates in surrogate windows vs the lag-12
distance (control arms only). Usage: python scripts/surrogate_leakage_diag.py"""
import numpy as np, pandas as pd
from ebmregime import panel as P
from ebmregime.config import load_config
from ebmregime.runner.compare import _panel
from ebmregime.stability.surrogate import surrogate_rows, surrogate_bounds, _range_rows
from ebmregime.splits import stable_seed
from ebmregime.splits.bags import tail_start
pr = load_config("prereg")
real = P.load_panel("BTCUSDT")
start, end = surrogate_bounds("BTCUSDT", pr)
rows = _range_rows(real, start, end); idx = real.index[rows]
pf = pd.read_parquet("results/surfloor/control_pairs.parquet")
pf = pf[(pf.lag == 12)]
G = pf.groupby(["data", "target", "tau_a", "tau_b"]).g.sum().reset_index()
taus = sorted(pf.tau_a.unique().tolist() + pf.tau_b.unique().tolist())
taus = sorted(set(taus))
dup = {}
for s in range(20):
    d = f"surrogate:{s}"
    seed = stable_seed(int(pr["surrogates"]["seed"]), "BTCUSDT", "surrogate", s)
    src = surrogate_rows(len(rows), float(pr["surrogates"]["mean_block_hours"]), seed)
    sp = _panel("BTCUSDT", d)
    for tau in taus:
        tr = P.train_rows(sp, tau, "shuf", 8, 12)
        pos = np.flatnonzero(tr); ts = sp.index[pos]
        tail = np.asarray(ts >= tail_start(ts, float(pr["validation"]["tail_frac"])))
        s_src = src[idx.get_indexer(ts)]
        trs = np.sort(s_src[~tail])
        k = np.searchsorted(trs, s_src[tail])
        lo = trs[np.clip(k - 1, 0, len(trs) - 1)]; hi = trs[np.clip(k, 0, len(trs) - 1)]
        dist = np.minimum(np.abs(s_src[tail] - lo), np.abs(s_src[tail] - hi))
        dup[(d, tau)] = (np.mean(dist == 0), np.mean(dist <= 8))
G = G[G.data != "real"].copy()
G["dup"] = [max(dup[(d, a)][0], dup[(d, b)][0]) for d, a, b in zip(G.data, G.tau_a, G.tau_b)]
G["near"] = [max(dup[(d, a)][1], dup[(d, b)][1]) for d, a, b in zip(G.data, G.tau_a, G.tau_b)]
realG = pf[pf.data == "real"].groupby(["tau_a"]).g.sum().median()
print("real shuf median G12", round(realG, 5))
for t in ("shuf", "dir", "vol"):
    x = G[G.target == t]
    clean = x[x.near == 0]
    print(t, "pairs", len(x), "| both windows with no tail row within 8h of a training source row:", len(clean),
          "median G", round(clean.g.median(), 5), "| all:", round(x.g.median(), 5),
          "| near>0.3:", round(x[x.near > 0.3].g.median(), 5), f"(n={int((x.near > 0.3).sum())})")
print("share of windows with near-dup tail rows:", round(np.mean([v[1] > 0 for v in dup.values()]), 2),
      " median near-dup share:", round(np.median([v[1] for v in dup.values()]), 3))
G["bin"] = pd.cut(G.near, [0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 1.0])
print(G.pivot_table(index="bin", columns="target", values="g", aggfunc="median", observed=True).round(5))
print(G[G.target == "shuf"].groupby("bin", observed=True).size())
print("spearman near vs G:", {t: round(G[G.target == t].near.corr(G[G.target == t].g, method="spearman"), 2) for t in ("shuf", "dir", "vol")})
