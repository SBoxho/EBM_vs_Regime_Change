"""Per-row term contributions of a stored EBM fit (PLAN §6.1 step 3; D-46)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ebm_contrib", "eta"]


def ebm_contrib(export: dict, X_ref: pd.DataFrame) -> np.ndarray:
    """``(n, 16)`` float64 contributions c_{m,j}(x_i) of an exported EBM on ``X_ref``.

    Thin wrapper of ``models.ebm.eval_terms_from_export`` (D-46 lookup: ``digitize(x, cuts) + 1``,
    ``right=False``; NaN → 0; unseen nominal level → last).
    """
    from ..models.ebm import eval_terms_from_export  # lazy: models is a separate module

    return np.asarray(eval_terms_from_export(export, X_ref), dtype=np.float64)


def eta(export: dict, contrib: np.ndarray) -> np.ndarray:
    """Raw margin η = intercept + sum of the term contributions over the last axis."""
    c = np.asarray(contrib, dtype=np.float64)
    return float(np.asarray(export["intercept"]).ravel()[0]) + c.sum(axis=-1)
