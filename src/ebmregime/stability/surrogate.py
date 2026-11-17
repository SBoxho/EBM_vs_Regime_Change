"""Stationary surrogates of a symbol's panel for the G0 floor calibration (PLAN §9.3, D-18, D-50).

A surrogate is a Politis-Romano stationary block bootstrap of the panel **rows** with
``start <= ts < end``, taken in time order: geometric block lengths with mean ``mean_block_hours``
rows, uniform block starts, circular wrap, n draws for n rows. Every column of a source row
(features, ``features_ok``, all targets, regime states, ``valid``) moves jointly, so masked rows
stay masked. The resampled rows are placed on the **original timestamps** of ``[start, end)``, so
windows, purge, bags and test months are built exactly as for real data.

For a symbol, ``[start, end) = [first_tau - train_months, last_tau + 1 month)`` from
``prereg.windows`` (BTCUSDT: ``[2018-12-01, 2026-11-01)``). Note that ``hour`` and ``dow`` are
columns like any other: they travel with their source row and need not match the new timestamp.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import panel as _panel
from ..config import load_config
from ..splits.bags import stable_seed
from ..splits.windows import _utc

_SEED_TAG = 0x5022  # separates this stream from other uses of the same integer seed


def surrogate_rows(n: int, mean_block: float, seed: int) -> np.ndarray:
    """Source positions (int64, length ``n``, values in ``[0, n)``) of a stationary bootstrap.

    Position 0 starts a block; every later position starts a new block with probability
    ``1 / mean_block`` (geometric block lengths, mean ``mean_block``), otherwise it continues the
    previous block at the next source position modulo ``n`` (circular wrap). Block starts are
    uniform on ``[0, n)``. ``mean_block = inf`` gives a single block, i.e. a rotation.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if not mean_block >= 1:  # also rejects NaN
        raise ValueError("mean_block must be >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SEED_TAG]))
    p_new = 0.0 if math.isinf(mean_block) else 1.0 / float(mean_block)
    new_block = rng.random(n) < p_new
    new_block[0] = True
    starts = rng.integers(0, n, size=n)
    block_no = np.cumsum(new_block) - 1
    first_pos = np.flatnonzero(new_block)              # draw position where each block starts
    offset = np.arange(n, dtype=np.int64) - first_pos[block_no]
    return ((starts[first_pos][block_no] + offset) % n).astype(np.int64)


def _range_rows(panel: pd.DataFrame, start, end) -> np.ndarray:
    idx = panel.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("panel index must be a tz-aware DatetimeIndex")
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("panel index must be strictly increasing")
    a, b = _utc(start), _utc(end)
    if b <= a:
        raise ValueError(f"end {b} must be after start {a}")
    rows = np.flatnonzero(np.asarray(idx >= a) & np.asarray(idx < b))
    if len(rows) == 0:
        raise ValueError(f"no panel rows in [{a}, {b})")
    return rows


def make_surrogate(panel: pd.DataFrame, start, end, mean_block_hours: float,
                   seed: int) -> pd.DataFrame:
    """Stationary-bootstrap surrogate of the rows of ``panel`` with ``start <= ts < end``.

    Returns a new frame indexed by the original timestamps of those rows; row i holds all columns
    of source row ``surrogate_rows(n, mean_block_hours, seed)[i]`` (positions within the range).
    Rows outside the range are not included. Column order and dtypes are preserved.
    """
    rows = _range_rows(panel, start, end)
    sub = panel.iloc[rows]
    pos = surrogate_rows(len(sub), mean_block_hours, seed)
    out = sub.iloc[pos].copy()
    out.index = sub.index.copy()
    return out


def surrogate_bounds(symbol: str, prereg: dict | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[first_tau - train_months, last_tau + 1 month)`` of ``symbol`` from ``prereg.windows``."""
    cfg = load_config("prereg") if prereg is None else prereg
    w = cfg["windows"]
    start = _utc(w["first_tau"][symbol]) - pd.DateOffset(months=int(w["train_months"]))
    end = _utc(w["last_tau"]) + pd.DateOffset(months=1)
    return start, end


@lru_cache(maxsize=8)
def _surrogate_cached(symbol: str, index: int, start: pd.Timestamp, end: pd.Timestamp,
                      mean_block: float, seed: int) -> pd.DataFrame:
    return make_surrogate(_panel.load_panel(symbol), start, end, mean_block, seed)


def surrogate_panel(symbol: str, index: int, prereg: dict | None = None) -> pd.DataFrame:
    """Surrogate number ``index`` of ``symbol``'s panel (cached per process; do not mutate it).

    Range from :func:`surrogate_bounds`, mean block ``prereg.surrogates.mean_block_hours``, seed
    ``stable_seed(prereg.surrogates.seed, symbol, "surrogate", index)``.
    """
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError(f"index must be a non-negative int, got {index!r}")
    cfg = load_config("prereg") if prereg is None else prereg
    s = cfg["surrogates"]
    start, end = surrogate_bounds(symbol, cfg)
    seed = stable_seed(int(s["seed"]), symbol, "surrogate", int(index))
    return _surrogate_cached(symbol, int(index), start, end, float(s["mean_block_hours"]), seed)
