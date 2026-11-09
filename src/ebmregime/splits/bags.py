"""Time-ordered validation bags, half-samples and seeds for one training window (D-13, D-17).

All functions take ``ts``: the sorted timestamps of the training rows of ONE window (the rows that
will be passed to the model, in that order). ``ts`` may have gaps where unusable rows were dropped.

Layout of a window (D-13):

* **Validation tail.** ``val_start = ts[floor((1 - tail_frac) * n)]`` unless given. Rows with
  ``ts >= val_start`` are validation (-1 in every bag).
* **Purge.** Rows with ``ts < val_start`` but ``label_time(ts) > val_start`` are 0 in every bag
  (their labels overlap the validation period). On a gap-free grid these are the last h rows.
* **Blocks.** The remaining rows are grouped into calendar blocks of ``block_hours`` hours,
  ``block = floor((ts - origin) / block_hours)`` with ``origin = ts[0]`` by default. Each bag
  independently sets ``round(bag_train_frac * n_blocks)`` blocks (drawn without replacement) to +1.

"round" means round-half-up (``floor(x + 0.5)``), not Python's banker's rounding.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .windows import HOUR, _utc, _utc_index, label_time

_EPS = 1e-9  # guards floor((1 - tail_frac) * n) against float error, e.g. 0.85 * 100


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _sorted_ts(ts) -> pd.DatetimeIndex:
    idx = _utc_index(ts)
    if len(idx) == 0:
        raise ValueError("ts is empty")
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("ts must be strictly increasing")
    return idx


def tail_start(ts, tail_frac: float) -> pd.Timestamp:
    """``ts[floor((1 - tail_frac) * n)]``: start of the last ``tail_frac`` of the rows in time."""
    idx = _sorted_ts(ts)
    if not 0.0 < tail_frac < 1.0:
        raise ValueError("tail_frac must be in (0, 1)")
    k = int(math.floor((1.0 - tail_frac) * len(idx) + _EPS))
    return idx[min(k, len(idx) - 1)]


def _roles(idx: pd.DatetimeIndex, horizon: int, val_start: pd.Timestamp):
    """Boolean masks (tail, purged, eligible) for rows of ``idx`` relative to ``val_start``."""
    tail = np.asarray(idx >= val_start)
    purged = ~tail & np.asarray(label_time(idx, horizon) > val_start)
    eligible = ~tail & ~purged
    return tail, purged, eligible


def _block_ids(idx: pd.DatetimeIndex, origin: pd.Timestamp, block_hours: int) -> np.ndarray:
    if block_hours <= 0:
        raise ValueError("block_hours must be positive")
    hours = np.asarray((idx - origin) // HOUR, dtype=np.int64)
    return np.floor_divide(hours, int(block_hours))


def time_ordered_bags(ts, horizon: int, n_bags: int, tail_frac: float, block_hours: int,
                      bag_train_frac: float, seed: int, val_start=None,
                      block_origin=None) -> np.ndarray:
    """EBM ``bags`` matrix, int8 of shape ``(n_rows, n_bags)`` (the shape interpret 0.7.8 requires).

    Values: -1 validation tail, 0 purged or not drawn, +1 training. ``block_origin`` (default
    ``ts[0]``) anchors the block grid; pass the full window's first timestamp when ``ts`` is a
    half-sample so that blocks line up with the half-sample's blocks.
    """
    idx = _sorted_ts(ts)
    n = len(idx)
    if n_bags < 1:
        raise ValueError("n_bags must be >= 1")
    if not 0.0 < bag_train_frac <= 1.0:
        raise ValueError("bag_train_frac must be in (0, 1]")
    vs = tail_start(idx, tail_frac) if val_start is None else _utc(val_start)
    origin = idx[0] if block_origin is None else _utc(block_origin)

    tail, _, eligible = _roles(idx, horizon, vs)
    bags = np.zeros((n, n_bags), dtype=np.int8)
    bags[tail, :] = -1

    elig_rows = np.flatnonzero(eligible)
    blocks = _block_ids(idx[elig_rows], origin, block_hours)
    uniq, inverse = np.unique(blocks, return_inverse=True)
    n_blocks = len(uniq)
    k = min(n_blocks, max(1, _round_half_up(bag_train_frac * n_blocks))) if n_blocks else 0
    for b in range(n_bags):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), b]))
        chosen = np.zeros(n_blocks, dtype=bool)
        chosen[rng.choice(n_blocks, size=k, replace=False)] = True
        bags[elig_rows[chosen[inverse]], b] = 1

    if not (bags == 1).any(axis=0).all():
        raise AssertionError("a bag has no training (+1) rows")
    if not (bags == -1).any(axis=0).all():
        raise AssertionError("a bag has no validation (-1) rows")
    return bags


def half_sample(ts, horizon: int, tail_frac: float, block_hours: int,
                seed: int) -> tuple[np.ndarray, pd.Timestamp]:
    """Half-sample of contiguous calendar blocks for noise-floor protocol (iii) (D-17).

    The window is split at the full-window ``val_start`` into a pre-tail part (``ts < val_start``,
    including the rows the purge will later zero) and a tail part. Blocks are assigned separately:
    pre-tail blocks are anchored at ``ts[0]``, tail blocks at ``val_start``. In each part
    ``max(1, round(n_blocks / 2))`` blocks are drawn without replacement, independently.

    Returns ``(mask, val_start)``. The caller then builds the bags on the selected rows with
    ``time_ordered_bags(ts[mask], ..., val_start=val_start, block_origin=ts[0])``. ``horizon`` is
    accepted for interface symmetry; the purge itself is applied by ``time_ordered_bags``.
    """
    idx = _sorted_ts(ts)
    vs = tail_start(idx, tail_frac)
    tail = np.asarray(idx >= vs)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB10C]))
    mask = np.zeros(len(idx), dtype=bool)
    for part, origin in ((~tail, idx[0]), (tail, vs)):
        rows = np.flatnonzero(part)
        if len(rows) == 0:
            raise AssertionError("half_sample: a part of the window is empty")
        blocks = _block_ids(idx[rows], origin, block_hours)
        uniq, inverse = np.unique(blocks, return_inverse=True)
        k = max(1, _round_half_up(len(uniq) / 2))
        chosen = np.zeros(len(uniq), dtype=bool)
        chosen[rng.choice(len(uniq), size=k, replace=False)] = True
        mask[rows[chosen[inverse]]] = True
    return mask, vs


def xgb_split(ts, horizon: int, tail_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Row indices ``(train, val)``: pre-tail unpurged rows, and the validation-tail rows."""
    idx = _sorted_ts(ts)
    tail, _, eligible = _roles(idx, horizon, tail_start(idx, tail_frac))
    return np.flatnonzero(eligible), np.flatnonzero(tail)


def stationary_bootstrap_counts(ts, horizon: int, tail_frac: float, mean_block: float,
                                seed: int) -> np.ndarray:
    """Replication counts for protocol (ii): one ``bags`` column, int8 of shape ``(n_rows,)``.

    Politis-Romano stationary bootstrap over the m pre-tail unpurged rows taken in row order
    (gaps ignored): geometric block lengths with mean ``mean_block`` rows, uniform block starts,
    circular wrap; m draws in total. Counts per row; tail rows are -1, purged rows 0.
    """
    idx = _sorted_ts(ts)
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    tail, _, eligible = _roles(idx, horizon, tail_start(idx, tail_frac))
    rows = np.flatnonzero(eligible)
    m = len(rows)
    if m == 0:
        raise AssertionError("no eligible training rows")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB007]))
    new_block = rng.random(m) < 1.0 / mean_block
    new_block[0] = True
    starts = rng.integers(0, m, size=m)
    block_no = np.cumsum(new_block) - 1
    first_pos = np.flatnonzero(new_block)            # draw position where each block starts
    offset = np.arange(m) - first_pos[block_no]
    draws = (starts[first_pos][block_no] + offset) % m
    counts = np.bincount(draws, minlength=m)
    if counts.max() > 127:
        raise AssertionError("a replication count exceeds the int8 range of interpret bags")
    out = np.zeros(len(idx), dtype=np.int8)
    out[tail] = -1
    out[rows] = counts.astype(np.int8)
    return out


def stable_seed(*parts) -> int:
    """Deterministic seed in ``[0, 2**31 - 1]`` (a valid signed 32-bit int) from str/int parts.

    sha256 over the type-tagged parts, so it is independent of the process, PYTHONHASHSEED and the
    order of execution, and ``1`` and ``"1"`` give different seeds. Order of ``parts`` matters.
    """
    tokens = []
    for p in parts:
        if isinstance(p, bool) or not isinstance(p, (int, np.integer, str)):
            raise TypeError(f"stable_seed parts must be str or int, got {type(p).__name__}")
        tokens.append(f"s:{p}" if isinstance(p, str) else f"i:{int(p)}")
    digest = hashlib.sha256("\x1f".join(tokens).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def replicate_plan(symbol: str, tau, target: str, cfg_replicates: dict) -> list[dict]:
    """Fits per (symbol, window, target) for the noise floor (D-17).

    ``cfg_replicates`` is the ``replicates`` block of ``configs/prereg.yaml``. Returns
    ``seed_only`` dicts with protocol ``"seed"`` (full data) followed by ``half_samples`` dicts with
    protocol ``"half"``. Each dict has ``protocol``, ``r``, ``seed`` (model random_state) and
    ``sample_seed`` (half-sample draw; None for ``"seed"``).
    """
    base = int(cfg_replicates["base_seed"])
    t = _utc(tau).isoformat()
    plan = []
    for protocol, n in (("seed", int(cfg_replicates["seed_only"])),
                        ("half", int(cfg_replicates["half_samples"]))):
        for r in range(n):
            plan.append({
                "protocol": protocol,
                "r": r,
                "seed": stable_seed(base, symbol, t, target, protocol, r),
                "sample_seed": (stable_seed(base, symbol, t, target, protocol, r, "sample")
                                if protocol == "half" else None),
            })
    return plan
