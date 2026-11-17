"""Stationary-bootstrap confidence intervals over the window index t (PLAN §9.1, D-22).

* Expected block length: ``arch.bootstrap.optimal_block_length`` (stationary, ``b_sb``; Politis-White
  2004 with the Patton-Politis-White 2009 correction), floored at ``max(3, delta)``. A degenerate
  estimate (NaN, inf, < 1) falls back to the floor (PLAN §9.7 fallback 2).
* ``multiplier`` scales the block length for the pre-registered 2x sensitivity.
* Intervals are percentile intervals of the statistic over bootstrap resamples.
"""

from __future__ import annotations

import warnings

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length


def block_length(x, delta: int = 1, multiplier: float = 1.0) -> tuple[float, bool]:
    """``(block, degenerate)``: max(b_sb, 3, delta) * multiplier; ``degenerate`` if b_sb was unusable."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    floor = float(max(3, int(delta)))
    degenerate = False
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)   # constant series: 0/0 inside arch
            b = float(optimal_block_length(x)["stationary"].iloc[0]) if len(x) >= 10 else np.nan
    except Exception:  # noqa: BLE001 - arch raises on constant or tiny series
        b = np.nan
    if not np.isfinite(b) or b < 1:
        degenerate, b = True, floor
    return max(b, floor) * float(multiplier), degenerate


def median_ci(x, delta: int = 1, level: float = 0.95, reps: int = 9999, seed: int = 20261108,
              multiplier: float = 1.0, stat=np.median) -> dict:
    """Point estimate and stationary-bootstrap percentile CI of ``stat(x)`` (NaNs dropped first)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"n": len(x), "est": float(stat(x)) if len(x) else np.nan, "lo": np.nan, "hi": np.nan,
                "block": np.nan, "degenerate": True}
    b, degenerate = block_length(x, delta, multiplier)
    bs = StationaryBootstrap(b, x, seed=np.random.default_rng(seed))
    draws = np.array([stat(data[0][0]) for data in bs.bootstrap(reps)])
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [a, 1.0 - a])
    return {"n": len(x), "est": float(stat(x)), "lo": float(lo), "hi": float(hi), "block": float(b),
            "degenerate": degenerate}
