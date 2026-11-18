"""Candidate noise-floor resampling schemes for the floor fix (D-55; lead-owned, experimental).

Kept outside ``splits/`` (and so outside the runner's fit-code hash) while the candidates are being
evaluated, so that the cached slice fits stay valid. The adopted scheme moves into ``splits/``.

* :func:`bootstrap_bags` (candidate B): full-n stationary-bootstrap replication counts combined with
  the time-ordered outer bags.
* :func:`disjoint_parts` (candidate R): a random partition of the window's 168-h blocks into ``k``
  disjoint parts, pre-tail and validation-tail blocks partitioned separately.
"""

from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import panel as P
from .config import REPO_ROOT as ROOT
from .config import load_config
from .splits import retrain_schedule, stable_seed
from .splits.bags import _block_ids, _sorted_ts, stationary_bootstrap_counts, tail_start, time_ordered_bags


def bootstrap_bags(ts, horizon: int, n_bags: int, tail_frac: float, block_hours: int, bag_train_frac: float,
                   mean_block: float, seed: int) -> np.ndarray:
    """int8 ``(n_rows, n_bags)``: time-ordered bags whose +1 entries carry bootstrap replication counts.

    Counts come from :func:`splits.bags.stationary_bootstrap_counts` over the pre-tail unpurged rows
    (mean block ``mean_block`` rows). Validation tail stays -1, purged rows 0. Rows drawn 0 times are
    0 in every bag, so ``fit_ebm`` drops them (D-45) and the fit sees only the resampled rows.
    """
    counts = stationary_bootstrap_counts(ts, horizon, tail_frac, mean_block, seed)
    base = time_ordered_bags(ts, horizon, n_bags, tail_frac, block_hours, bag_train_frac,
                             seed=int(seed) ^ 0x5EED)
    bags = np.where(base == 1, counts[:, None].astype(np.int16), base.astype(np.int16))
    if bags.max() > 127:
        raise AssertionError("replication count exceeds int8")
    bags = bags.astype(np.int8)
    if not (bags > 0).any(axis=0).all():
        raise AssertionError("a bag has no in-bag rows")
    return bags


def disjoint_parts(ts, tail_frac: float, block_hours: int, k: int, seed: int) -> tuple[list[np.ndarray], pd.Timestamp]:
    """``(masks, val_start)``: ``k`` disjoint boolean masks over ``ts`` that together cover every row.

    Pre-tail blocks (anchored at ``ts[0]``) and validation-tail blocks (anchored at ``val_start``) are
    each shuffled and dealt round-robin to the k parts, so part sizes differ by at most one block and
    every part has validation rows. Build each part's bags with
    ``time_ordered_bags(ts[mask], ..., val_start=val_start, block_origin=ts[0])``.
    """
    idx = _sorted_ts(ts)
    vs = tail_start(idx, tail_frac)
    tail = np.asarray(idx >= vs)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xD15C, int(k)]))
    part_of_row = np.full(len(idx), -1, dtype=np.int64)
    for region, origin in ((~tail, idx[0]), (tail, vs)):
        rows = np.flatnonzero(region)
        blocks = _block_ids(idx[rows], origin, block_hours)
        uniq, inverse = np.unique(blocks, return_inverse=True)
        if len(uniq) < k:
            raise AssertionError(f"only {len(uniq)} blocks for {k} parts")
        perm = rng.permutation(len(uniq))
        part_of_block = np.empty(len(uniq), dtype=np.int64)
        part_of_block[perm] = np.arange(len(uniq)) % k
        part_of_row[rows] = part_of_block[inverse]
    assert (part_of_row >= 0).all()
    return [part_of_row == i for i in range(k)], vs


# --------------------------------------------------------------------- experiment fits (D-55)

SYMBOL, H = "BTCUSDT", 8
ARMS = [("real", "shuf"), ("surrogate:0", "dir"), ("surrogate:0", "vol")]
N_BOOT, N_HALF_PART, N_QUART_PART = 10, 4, 2
MEAN_BLOCK = 168


def code_tag() -> str:
    h = hashlib.sha256()
    for rel in ("src/ebmregime/floorexp.py", "src/ebmregime/models/ebm.py", "src/ebmregime/splits/bags.py",
                "src/ebmregime/panel.py", "src/ebmregime/stability/surrogate.py"):
        h.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def fit_dir() -> Path:
    return ROOT / "data" / "cache" / "floor_exp" / code_tag()


def fit_path(data, target, tau, proto, p, part) -> Path:
    return fit_dir() / f"{data.replace(':', '-')}_{target}" / tau / f"{proto}_{p}_{part}.npz"


def tasks_for(prereg):
    taus = [str(t.date()) for t in retrain_schedule(prereg["windows"]["first_tau"][SYMBOL], prereg["windows"]["last_tau"])]
    out = []
    for data, target in sorted(ARMS, key=lambda a: a[1] != "vol"):          # vol (slowest) first
        for tau in taus:
            out += [(data, target, tau, "boot", r, 0) for r in range(N_BOOT)]
            out += [(data, target, tau, "half", p, k) for p in range(N_HALF_PART) for k in range(2)]
            out += [(data, target, tau, "quart", p, k) for p in range(N_QUART_PART) for k in range(4)]
    return taus, out


@lru_cache(maxsize=4)
def _panel(data):
    if data == "real":
        return P.load_panel(SYMBOL)
    from ebmregime.stability.surrogate import surrogate_panel
    return surrogate_panel(SYMBOL, int(data.split(":")[1]))


def fit_task(data, target, tau, proto, p, part, prereg):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ebmregime.models.ebm import export_fit, fit_ebm, kind_for_target
    from ebmregime.runner.cache import write_atomic

    path = fit_path(data, target, tau, proto, p, part)
    if path.exists():
        return None
    v = prereg["validation"]
    base = int(prereg["replicates"]["base_seed"])
    pnl = _panel(data)
    tr = P.train_rows(pnl, tau, target, H, int(prereg["windows"]["train_months"]))
    pos = np.flatnonzero(tr)
    ts = pnl.index[pos]
    seed = stable_seed(base, "floorexp", SYMBOL, tau, target, data, proto, p, part)
    common = dict(tail_frac=float(v["tail_frac"]), block_hours=int(v["block_hours"]),
                  bag_train_frac=float(v["bag_train_frac"]))
    if proto == "boot":
        rows = pos
        bags = bootstrap_bags(ts, H, int(v["outer_bags"]), mean_block=MEAN_BLOCK, seed=seed, **common)
    else:
        k = 2 if proto == "half" else 4
        psd = stable_seed(base, "floorexp", SYMBOL, tau, target, data, proto, p, "partition")
        masks, vs = disjoint_parts(ts, float(v["tail_frac"]), int(v["block_hours"]), k, psd)
        m = masks[part]
        rows = pos[m]
        bags = time_ordered_bags(ts[m], H, int(v["outer_bags"]), seed=seed, val_start=vs, block_origin=ts[0], **common)
    X = pnl[P.feature_names()].iloc[rows]
    y = pnl[P.target_column(target, H)].to_numpy()[rows]
    kind = kind_for_target(target)
    if kind == "classifier":
        y = y.astype(np.int64)
    t0 = time.perf_counter()
    model, info = fit_ebm(X, y, bags, kind, seed)
    arrays = dict(export_fit(model))
    arrays["best_iteration"] = info["best_iteration"]
    meta = {"data": data, "target": target, "tau": tau, "proto": proto, "p": p, "part": part, "seed": seed,
            "n_rows_fit": info["n_rows_fit"], "fit_seconds": info["fit_seconds"], "code_tag": code_tag()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, arrays, meta)
    return time.perf_counter() - t0


def cmd_fit(workers):
    from joblib import Parallel, delayed

    prereg = load_config("prereg")
    _, tasks = tasks_for(prereg)
    todo = [t for t in tasks if not fit_path(*t).exists()]
    print(f"{len(tasks)} fits, {len(todo)} to run, dir {fit_dir()}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky", verbose=5, batch_size=1)(
        delayed(fit_task)(*t, prereg) for t in todo)
    secs = [r for r in res if r is not None]
    print(f"done: {len(secs)} fits in {(time.time() - t0) / 60:.1f} min wall; mean {np.mean(secs) if secs else 0:.2f} s", flush=True)
