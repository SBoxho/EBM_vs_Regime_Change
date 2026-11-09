"""Rolling retraining windows (PLAN §7, D-19, D-34).

Time convention: see :mod:`ebmregime.series`. Row t is the bar opening at ``ts_t``; a label with
horizon h is known at ``ts_t + (h + 1)`` hours (the close of bar t+h). All timestamps are UTC.

* Retrain cut-offs ``tau`` are UTC calendar-month starts.
* Training rows of the window at ``tau``: ``tau - L <= ts`` (L calendar months) and
  ``label_time(ts) <= tau`` (the purge: every training label is observed by ``tau``).
* Test rows: ``tau <= ts < tau + 1 calendar month``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def _utc(x) -> pd.Timestamp:
    """Timestamp in UTC (naive input is taken to be UTC)."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _utc_index(ts) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(ts)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _apply_usable(mask: np.ndarray, usable) -> np.ndarray:
    if usable is None:
        return mask
    u = np.asarray(usable, dtype=bool)
    if u.shape != mask.shape:
        raise ValueError(f"usable has shape {u.shape}, expected {mask.shape}")
    return mask & u


def retrain_schedule(first_tau: str, last_tau: str, step_months: int = 1) -> list[pd.Timestamp]:
    """UTC month-start cut-offs from ``first_tau`` to ``last_tau`` inclusive, every ``step_months``."""
    a, b = _utc(first_tau), _utc(last_tau)
    for t in (a, b):
        if t != t.normalize() or t.day != 1:
            raise ValueError(f"tau must be a UTC month start, got {t}")
    if step_months < 1:
        raise ValueError("step_months must be >= 1")
    if b < a:
        raise ValueError(f"last_tau {b} is before first_tau {a}")
    out = []
    t = a
    while t <= b:
        out.append(t)
        t = t + pd.DateOffset(months=step_months)
    return out


def label_time(ts, horizon: int) -> pd.DatetimeIndex:
    """Time at which the horizon-``horizon`` label of each row is known: ``ts + (horizon + 1) h``."""
    return _utc_index(ts) + (horizon + 1) * HOUR


def train_mask(ts, tau, train_months: int, horizon: int, usable=None) -> np.ndarray:
    """Rows with ``ts >= tau - train_months`` (calendar months) and ``label_time <= tau`` (and ``usable``)."""
    idx = _utc_index(ts)
    tau = _utc(tau)
    start = tau - pd.DateOffset(months=train_months)
    m = np.asarray(idx >= start) & np.asarray(label_time(idx, horizon) <= tau)
    return _apply_usable(m, usable)


def test_mask(ts, tau, usable=None) -> np.ndarray:
    """Rows with ``tau <= ts < tau + 1 calendar month`` (and ``usable``)."""
    idx = _utc_index(ts)
    tau = _utc(tau)
    end = tau + pd.DateOffset(months=1)
    m = np.asarray(idx >= tau) & np.asarray(idx < end)
    return _apply_usable(m, usable)


test_mask.__test__ = False  # not a pytest test despite the name


def overlap_fraction(tau_a, tau_b, train_months: int) -> float:
    """Share of the earlier window's training interval that the later window also covers.

    Definition: with ``I(tau) = [tau - L, tau)`` (L = ``train_months`` calendar months) and
    ``e = min(tau_a, tau_b)``, the result is ``|I(tau_a) ∩ I(tau_b)| / |I(e)|``, both lengths in
    hours. It is 1 for equal taus, 0 when the taus are L or more months apart, and
    ``≈ 1 - Δ/L`` in between (exact up to unequal calendar-month lengths). It ignores the
    label purge and missing rows; it is a calendar overlap.
    """
    a, b = sorted((_utc(tau_a), _utc(tau_b)))
    L = pd.DateOffset(months=train_months)
    early_len = (a - (a - L)) / HOUR
    inter = (a - (b - L)) / HOUR  # [b-L, a) when b-L < a
    return float(max(0.0, inter) / early_len)
