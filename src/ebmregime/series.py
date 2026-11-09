"""Shared causal primitives on the hourly grid. Every feature, target and regime module uses these.

Time convention (applies to the whole package)
----------------------------------------------
* The hourly frame is indexed by bar **open** time ``ts``. Row ``t`` holds bar t, which covers
  ``[ts, ts + 1h)``. Its close ``C_t`` is known at ``ts + 1h``.
* A **feature** at row t may use bars ``<= t`` only.
* A **label** at row t with horizon h uses bars ``t+1 .. t+h`` and becomes known at
  ``ts_t + (h + 1) hours``, the close of bar t+h. The purge in ``ebmregime.splits`` uses this.

Canonical hourly input frame (built by :func:`hourly_inputs`)
--------------------------------------------------------------
The frame has a regular, complete 1h UTC index and these columns:
``open high low close volume quote_volume taker_buy_base taker_buy_quote n_trades`` (from the 1h
bars; prices are NaN on invalid hours), ``valid`` (bool), ``volume_suspect`` (bool), and:

* ``volume_ok``: ``valid & ~volume_suspect``. Volume-based quantities use only these hours.
* ``rv``: realised variance of hour t, the sum of the four squared 15m log returns that **end**
  inside the hour. The first of them runs from the previous hour's last 15m close. ``rv`` is NaN
  unless all four returns exist, all their 15m bars are valid, and hour t itself is valid.

Gap tolerance (D-06, revised by D-43)
-------------------------------------
* **Window rule** (``MIN_FRAC = 0.90``): an aggregation over k raw hourly values (sums, means,
  max/min, daily aggregates of hours) uses the present values when at least ``ceil(0.90 k)`` exist,
  otherwise NaN. A 168-hour window tolerates up to 16 missing hours. The longest outage inside the
  analysis ranges is 10 h.
* **History rule** (``MIN_FRAC_HISTORY = 0.50``): a reference or history window over values that are
  themselves window aggregates (percentile histories, lagged reference sets, multi-day windows of
  daily values) needs at least half of its values. A single outage blanks a derived series for a
  whole window length, so applying the 0.90 rule to derived values would turn one outage into
  months of NaN.
* Forward (label) windows are strict: any invalid hour gives NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
MIN_FRAC = 0.90          # window rule (D-43)
MIN_FRAC_HISTORY = 0.50  # history rule (D-43)


def returns_15m(grid15: pd.DataFrame) -> pd.Series:
    """15m log close-to-close returns on the complete 15m grid. NaN if either bar is invalid."""
    c = grid15["close"].where(grid15["valid"])
    return np.log(c).diff().rename("r15")


def rv_hourly(grid15: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """Per-hour realised variance from the four 15m returns ending in each hour (NaN unless all 4 exist)."""
    r2 = returns_15m(grid15) ** 2
    rv = r2.groupby(r2.index.floor(HOUR)).sum(min_count=4)
    return rv.reindex(hourly_index).rename("rv")


def hourly_inputs(grid15: pd.DataFrame, bars1h: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical hourly input frame (see the module docstring)."""
    h = bars1h.copy()
    h["volume_ok"] = h["valid"] & ~h["volume_suspect"]
    h["rv"] = rv_hourly(grid15, h.index).where(h["valid"])
    return h


def trailing_mean(x: pd.Series, k: int, min_frac: float = MIN_FRAC) -> pd.Series:
    """Mean of ``x`` over the trailing window ``t-k+1 .. t``, using NaN-free values when at least
    ``ceil(min_frac * k)`` of the k values are present; otherwise NaN."""
    return x.rolling(k, min_periods=math.ceil(min_frac * k)).mean()


def trailing_sum(x: pd.Series, k: int, min_frac: float = MIN_FRAC) -> pd.Series:
    """Sum over ``t-k+1 .. t``, rescaled to k hours when a few values are missing (``mean * k``)."""
    return trailing_mean(x, k, min_frac) * k


def forward_mean_strict(x: pd.Series, h: int) -> pd.Series:
    """Mean of ``x`` over ``t+1 .. t+h``. NaN if any of those values is NaN, or if the window runs
    past the end of the series.

    Each window is summed on its own (``sliding_window_view``). A running rolling sum over the
    reversed series would carry about 1e-16 of rounding from rows after t+h into row t. That is not
    information, but it breaks bit-exact no-lookahead tests (found during M1 by the targets
    implementer).
    """
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > h:
        win = np.lib.stride_tricks.sliding_window_view(v[1:], h)  # row t -> v[t+1 .. t+h]
        out[: len(win)] = win.sum(axis=1) / h
    return pd.Series(out, index=x.index, name=x.name)


def require_regular_hourly(df: pd.DataFrame) -> None:
    """Raise unless ``df`` has a complete, strictly increasing 1h UTC index."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex")
    if len(idx) > 1 and not (np.diff(idx.asi8) == np.diff(idx.asi8)[0]).all():
        raise ValueError("index is not a regular grid")
    if len(idx) > 1 and pd.Timedelta(idx[1] - idx[0]) != HOUR:
        raise ValueError("index step is not 1 hour")
