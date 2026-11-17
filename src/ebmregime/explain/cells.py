"""Reference sample, pooled equal-probability cells and cell-averaged shapes (PLAN §6.1; D-15, D-46, D-50).

* ``make_xref``: X_ref of a pair = union of the rows each window passes to ``fit``, capped by a sorted
  uniform subsample without replacement (D-50).
* ``make_cells``: per continuous feature, G pooled equal-probability cells from the quantiles of X_ref,
  ties merged, cells with fewer than ``min_rows`` rows merged into the smaller adjacent neighbour, and
  the PLAN §9.7 fallback 3 (G = 20) when fewer than ``min_cells`` cells remain. Nominal features get
  one cell per level present, weighted by pooled frequency.
* ``cell_shapes`` / ``cell_shapes_many``: cell means of a contribution column, re-centred under the
  cell weights ω (Σ_k ω_k f̃(k) = 0). One sparse product per stack, because this runs ~10⁵ times.

Min-rows merge rule (the PLAN says only "merge cells with fewer than 20 rows"): repeatedly take the
smallest cell below ``min_rows`` (leftmost on ties) and merge it into its adjacent neighbour with fewer
rows (the left one on ties; an end cell has one neighbour), until no cell is below ``min_rows`` or one
cell is left. Merging removes the interior edge between the two cells, so the final cells are still
``np.searchsorted(edges, x, side="right")`` of the surviving edges. Empty cells (possible after tie
merging, e.g. a mass point at the minimum) are always merged, even with ``min_rows <= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import load_config

__all__ = ["CellGrid", "cell_shapes", "cell_shapes_many", "explain_settings", "make_cells", "make_xref"]


@lru_cache(maxsize=1)
def _prereg_explain() -> dict:
    return load_config("prereg")["explain"]


def explain_settings() -> dict:
    """The pre-registered settings used here (``prereg.explain``; PLAN §9.7 fallback 3)."""
    e = _prereg_explain()
    return {
        "cap_rows": int(e["x_ref"]["cap_rows"]),
        "seed": int(e["x_ref"]["seed"]),
        "n_cells": int(e["cells"]["n"]),
        "min_rows": int(e["cells"]["min_rows"]),
        "fallback_cells": 20,  # prereg.fallbacks: "cell merging leaves < 10 cells -> G = 20"
        "min_cells": 10,
    }


# ----------------------------------------------------------------------------------------------- X_ref


def make_xref(fit_masks: list[np.ndarray], cap: int, seed: int) -> np.ndarray:
    """Sorted int64 row positions of the union of ``fit_masks``, capped at ``cap`` rows (D-50).

    Above the cap: a uniform subsample without replacement drawn with ``np.random.default_rng(seed)``,
    returned sorted.
    """
    if len(fit_masks) == 0:
        raise ValueError("make_xref needs at least one mask")
    masks = [np.asarray(m) for m in fit_masks]
    n = masks[0].shape[0]
    union = np.zeros(n, dtype=bool)
    for m in masks:
        if m.ndim != 1 or m.shape[0] != n:
            raise ValueError("all masks must be 1-D and of equal length")
        if m.dtype != bool:
            raise TypeError("masks must be boolean")
        union |= m
    pos = np.flatnonzero(union).astype(np.int64)
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if pos.size > cap:
        rng = np.random.default_rng(seed)
        pick = rng.choice(pos.size, size=cap, replace=False)
        pos = np.sort(pos[pick])
    return pos


# ----------------------------------------------------------------------------------------------- cells


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class CellGrid:
    """Cells of one reference sample. Per-feature entries are tuples aligned with ``features``.

    ``cell_of_row[j]``: int64 ``(n_ref,)`` in 0..K_j−1; ``weights[j]``: float64 ``(K_j,)``, ω_k = n_k/n_ref;
    ``x_rep[j]``: float64 ``(K_j,)``, cell median (nominal: the level); ``edges[j]``: continuous: the
    interior cut values (``K_j − 1``), nominal: the levels (``K_j``); ``is_nominal[j]``;
    ``fallback_used[j]``: the G = ``fallback_cells`` rebuild was applied (PLAN §9.7 fallback 3).
    ``offsets``: int64 ``(J+1,)``, start of feature j's cells in the concatenated cell axis.
    """

    features: tuple[str, ...]
    n_ref: int
    cell_of_row: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    x_rep: tuple[np.ndarray, ...]
    edges: tuple[np.ndarray, ...]
    is_nominal: tuple[bool, ...]
    fallback_used: tuple[bool, ...]
    offsets: np.ndarray
    _mean_op: sp.csr_matrix = field(repr=False)   # (n_ref*J, ΣK): row i*J + j → cell, value 1/n_k
    _w_all: np.ndarray = field(repr=False)        # (ΣK,) concatenated weights

    @property
    def n_cells(self) -> tuple[int, ...]:
        return tuple(int(w.size) for w in self.weights)

    @property
    def n_features(self) -> int:
        return len(self.features)


def _quantile_edges(x: np.ndarray, g: int) -> np.ndarray:
    """Interior edges at quantiles k/g, k = 1..g−1 (method "linear"), duplicates merged."""
    q = np.arange(1, g) / g
    return np.unique(np.quantile(x, q, method="linear"))


def _merge_small(counts: np.ndarray, min_rows: int) -> np.ndarray:
    """Indices of the edges to keep after the min-rows merge. Edge e sits between cells e and e+1."""
    thr = max(int(min_rows), 1)
    c = [int(v) for v in counts]
    keep = list(range(len(c) - 1))       # keep[i] = original edge index between current cells i and i+1
    while len(c) > 1:
        small = [i for i, v in enumerate(c) if v < thr]
        if not small:
            break
        i = min(small, key=lambda t: (c[t], t))
        if i == 0:
            nb = 1
        elif i == len(c) - 1:
            nb = i - 1
        else:
            nb = i - 1 if c[i - 1] <= c[i + 1] else i + 1
        lo, hi = min(i, nb), max(i, nb)
        c[lo] += c[hi]
        del c[hi]
        del keep[lo]                       # the edge between lo and hi disappears
    return np.asarray(keep, dtype=np.int64)


def _continuous_cells(x: np.ndarray, g: int, min_rows: int) -> tuple[np.ndarray, np.ndarray]:
    edges = _quantile_edges(x, g)
    cell = np.searchsorted(edges, x, side="right")
    counts = np.bincount(cell, minlength=edges.size + 1)
    edges = edges[_merge_small(counts, min_rows)]
    cell = np.searchsorted(edges, x, side="right").astype(np.int64)
    return edges, cell


def _cell_medians(x: np.ndarray, cell: np.ndarray, k: int) -> np.ndarray:
    # cells are contiguous in sorted x (cell is monotone in x), so medians come from one sort
    xs = np.sort(x)
    counts = np.bincount(cell, minlength=k)
    start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    lo = start + (counts - 1) // 2
    hi = start + counts // 2
    return 0.5 * (xs[lo] + xs[hi])


def make_cells(X_ref: pd.DataFrame, n_cells: int = 50, min_rows: int = 20, nominal: list[str] | None = None,
               fallback_cells: int = 20, min_cells: int = 10) -> CellGrid:
    """Pooled cells of ``X_ref`` (PLAN §6.1 step 2). ``nominal=None`` → ``panel.nominal_features()``."""
    if nominal is None:
        from ..panel import nominal_features
        nominal = nominal_features()
    nominal_set = set(nominal)
    features = tuple(str(c) for c in X_ref.columns)
    n = len(X_ref)
    if n == 0:
        raise ValueError("X_ref is empty")

    cell_of_row, weights, x_rep, edges, is_nom, fb, counts = [], [], [], [], [], [], []
    for name in features:
        col = X_ref[name]
        if name in nominal_set:
            vals = col.to_numpy()
            if pd.isna(vals).any():
                raise ValueError(f"nominal feature {name!r} has missing values in X_ref")
            levels, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
            cell = inv.astype(np.int64).ravel()
            e = levels
            rep = levels.astype(np.float64)
            used = False
        else:
            x = col.to_numpy(dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"continuous feature {name!r} has non-finite values in X_ref")
            e, cell = _continuous_cells(x, n_cells, min_rows)
            used = False
            if e.size + 1 < min_cells:
                e, cell = _continuous_cells(x, fallback_cells, min_rows)
                used = True
            cnt = np.bincount(cell, minlength=e.size + 1)
            rep = _cell_medians(x, cell, e.size + 1)
        cell_of_row.append(_ro(cell))
        counts.append(np.asarray(cnt, dtype=np.float64))
        weights.append(_ro(cnt.astype(np.float64) / n))
        x_rep.append(_ro(rep))
        edges.append(_ro(np.asarray(e)))
        is_nom.append(name in nominal_set)
        fb.append(used)

    k = np.array([w.size for w in weights], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(k))).astype(np.int64)
    J = len(features)
    # mean operator: contrib.reshape(R, n*J) @ M gives all cell means at once
    gcell = np.stack([cell_of_row[j] + offsets[j] for j in range(J)], axis=1).ravel()   # (n*J,) in (i, j) order
    counts_all = np.concatenate(counts)
    vals = 1.0 / counts_all[gcell]
    mean_op = sp.csr_matrix((vals, (np.arange(n * J), gcell)), shape=(n * J, int(offsets[-1])))
    return CellGrid(
        features=features, n_ref=n, cell_of_row=tuple(cell_of_row), weights=tuple(weights),
        x_rep=tuple(x_rep), edges=tuple(edges), is_nominal=tuple(is_nom), fallback_used=tuple(fb),
        offsets=_ro(offsets), _mean_op=mean_op, _w_all=_ro(np.concatenate(weights)),
    )


# ---------------------------------------------------------------------------------------------- shapes


def _recentred_flat(contribs: np.ndarray, grid: CellGrid) -> np.ndarray:
    """(R, n, J) → (R, ΣK) re-centred cell means."""
    c = np.asarray(contribs, dtype=np.float64)
    R, n, J = c.shape
    if n != grid.n_ref or J != grid.n_features:
        raise ValueError(f"contribs has shape {c.shape}; grid expects (R, {grid.n_ref}, {grid.n_features})")
    f = np.asarray(grid._mean_op.T @ c.reshape(R, n * J).T).T          # (R, ΣK) cell means
    off = grid.offsets
    centre = np.add.reduceat(f * grid._w_all, off[:-1], axis=1)       # (R, J): Σ_k ω_k f(k)
    f -= np.repeat(centre, np.diff(off), axis=1)
    return f


def cell_shapes_many(contribs: np.ndarray, grid: CellGrid) -> list[np.ndarray]:
    """Re-centred cell shapes of a stack ``(R, n_ref, J)``: per feature j an ``(R, K_j)`` array."""
    c = np.asarray(contribs)
    if c.ndim != 3:
        raise ValueError("contribs must be (R, n_ref, J)")
    f = _recentred_flat(c, grid)
    off = grid.offsets
    return [f[:, off[j]:off[j + 1]] for j in range(grid.n_features)]


def cell_shapes(contrib: np.ndarray, grid: CellGrid) -> list[np.ndarray]:
    """Re-centred cell shapes f̃_j of one ``(n_ref, J)`` contribution matrix (PLAN §6.1 steps 3–4)."""
    c = np.asarray(contrib)
    if c.ndim != 2:
        raise ValueError("contrib must be (n_ref, J)")
    f = _recentred_flat(c[None], grid)[0]
    off = grid.offsets
    return [f[off[j]:off[j + 1]] for j in range(grid.n_features)]
