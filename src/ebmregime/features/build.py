"""The fixed 16-feature set of PLAN §5.1 (D-07, D-08, D-35), computed causally on the hourly grid.

Conventions (see :mod:`ebmregime.series` for the time convention and gap tolerance)
-----------------------------------------------------------------------------------
* Row t is bar t. A feature at row t uses bars ``<= t`` only.
* ``C, H, L`` are close/high/low masked to NaN on invalid hours.
* ``RV_k(t) = series.trailing_mean(rv, k)``: per-hour realised variance over bars ``t-k+1 .. t``.
* ``sigma_t = sqrt(RV_168(t))`` (``sigma_hours`` in the config).
* Gap tolerance (D-06 as revised by D-43; constants imported from :mod:`ebmregime.series`):

  - **window rule**: an aggregation over k raw hourly values (RV_k and sigma, the qv_z_1h reference
    window, the pos_range max/min, QV24, the taker_imb sums) uses the present values when at least
    ``ceil(MIN_FRAC * k)`` exist, else NaN;
  - **history rule**: a reference window over n derived values (the RV_168 history of rv_level_pct,
    the lagged log-QV24 set of qv_z_24h) needs at least ``ceil(MIN_FRAC_HISTORY * n)`` of them.
* Volume quantities use only ``volume_ok`` hours (``valid & ~volume_suspect``); other hours count as
  missing.
* Any non-finite result (e.g. ``log(0)``, division by a zero sigma) is returned as NaN, so every
  continuous feature column is either finite or NaN.
* ``hour`` and ``dow`` are returned as ``int64`` (never missing).

``features_ok = h.valid & (all 16 features finite)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ebmregime.series import (MIN_FRAC, MIN_FRAC_HISTORY, require_regular_hourly, trailing_mean,
                              trailing_sum)

FEATURE_NAMES: list[str] = [
    "ret_1h", "ret_4h", "ret_24h", "ret_168h",
    "rv_ratio_4h", "rv_ratio_24h", "rv_level_pct",
    "qv_z_1h", "qv_z_24h",
    "range_1h", "pos_range_24h", "pos_range_168h",
    "hour", "dow",
    "taker_imb_4h", "xret_4h",
]
NOMINAL: list[str] = ["hour", "dow"]


def _min_count(k: int) -> int:
    """Window rule: minimum present values in a window of k raw hourly values, ``ceil(MIN_FRAC*k)``."""
    return math.ceil(MIN_FRAC * k)


def _min_count_history(n: int) -> int:
    """History rule: minimum present values in a reference window of n derived values,
    ``ceil(MIN_FRAC_HISTORY*n)``."""
    return math.ceil(MIN_FRAC_HISTORY * n)


def _finite_or_nan(x: pd.Series) -> pd.Series:
    """Replace +-inf by NaN."""
    return x.where(np.isfinite(x))


def _prices(h: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """``C, H, L``: close, high, low, NaN on invalid hours."""
    v = h["valid"].astype(bool)
    return h["close"].where(v), h["high"].where(v), h["low"].where(v)


def _rv(h: pd.DataFrame) -> pd.Series:
    """Hourly realised variance, NaN on invalid hours (already so in the canonical frame)."""
    return h["rv"].where(h["valid"].astype(bool))


def _volume_ok(h: pd.DataFrame) -> pd.Series:
    """``volume_ok & valid`` (the second term is defensive; the canonical frame already implies it)."""
    return h["volume_ok"].astype(bool) & h["valid"].astype(bool)


def sigma(h: pd.DataFrame, sigma_hours: int = 168) -> pd.Series:
    """``sigma_t = sqrt(RV_{sigma_hours}(t))``, the trailing per-hour volatility."""
    return np.sqrt(trailing_mean(_rv(h), sigma_hours)).rename("sigma")


def scaled_return(close: pd.Series, sig: pd.Series, k: int) -> pd.Series:
    """``log(C_t / C_{t-k}) / (sigma_t * sqrt(k))``. NaN if either close or sigma is NaN."""
    return _finite_or_nan(np.log(close / close.shift(k)) / (sig * math.sqrt(k)))


def rolling_pct_rank(x: pd.Series, window: int, min_count: int) -> pd.Series:
    """Trailing percentile of ``x_t`` within ``x_{t-window+1} .. x_t``.

    ``pct_t = #{s in window : x_s present and x_s <= x_t} / #{s in window : x_s present}``
    (inclusive of t, so ties count as "below" and the maximum gets 1.0). NaN if ``x_t`` is NaN or if
    fewer than ``min_count`` values in the window are present. Implemented with pandas' rolling
    rank (``method="max"``, ``pct=True``); a test checks it against a brute-force loop.
    """
    r = x.rolling(window, min_periods=min_count).rank(method="max", pct=True)
    return r.where(x.notna())


def _rv_level_pct(rv168: pd.Series, history_hours: int) -> pd.Series:
    """Percentile of ``RV_168(t)`` among the ``RV_168`` values of bars ``t-history_hours+1 .. t``,
    requiring ``ceil(MIN_FRAC_HISTORY * history_hours)`` present values (history rule)."""
    return rolling_pct_rank(rv168, history_hours, _min_count_history(history_hours))


def _qv_z_1h(qv_ok: pd.Series, history_hours: int) -> pd.Series:
    """``x_t = log(QV_t)`` (NaN if not volume_ok or ``QV_t <= 0``);
    ``z_t = (x_t - mean(x_{t-n..t-1})) / sd(x_{t-n..t-1})`` with ``n = history_hours``, ddof=1,
    the reference window excluding t and requiring ``ceil(MIN_FRAC * n)`` present values (window rule over raw hours). NaN if sd == 0."""
    x = np.log(qv_ok.where(qv_ok > 0))
    ref = x.shift(1).rolling(history_hours, min_periods=_min_count(history_hours))
    mu, sd = ref.mean(), ref.std(ddof=1)
    return _finite_or_nan((x - mu) / sd.where(sd > 0))


def _qv_z_24h(qv_ok: pd.Series, window_hours: int, history_days: int) -> pd.Series:
    """``QV24(t) = trailing_sum(QV on volume_ok hours, 24)``, ``y_t = log QV24(t)``.
    Reference values ``y_{t-24j}``, j = 1..history_days; mean and sd (ddof=1) over the present ones,
    requiring ``ceil(MIN_FRAC_HISTORY * history_days)`` of them (history rule). ``z_t = (y_t - mean) / sd``; NaN if sd == 0."""
    qv24 = trailing_sum(qv_ok, window_hours)
    y = np.log(qv24.where(qv24 > 0))
    ref = np.column_stack([y.shift(window_hours * j).to_numpy() for j in range(1, history_days + 1)])
    present = np.isfinite(ref)
    cnt = present.sum(axis=1)
    ok = cnt >= _min_count_history(history_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(present, ref, 0.0)
        mu = filled.sum(axis=1) / cnt
        dev = np.where(present, ref - mu[:, None], 0.0)
        sd = np.sqrt((dev**2).sum(axis=1) / (cnt - 1))
        z = (y.to_numpy() - mu) / np.where(sd > 0, sd, np.nan)
    z = np.where(ok, z, np.nan)
    return _finite_or_nan(pd.Series(z, index=y.index))


def _pos_range(c: pd.Series, hi: pd.Series, lo: pd.Series, k: int) -> pd.Series:
    """``Hk = max(H_{t-k+1..t})``, ``Lk = min(L_{t-k+1..t})`` (each needing ``ceil(MIN_FRAC*k)`` present);
    ``pos_t = (C_t - Lk) / (Hk - Lk)``, and 0.5 when ``Hk == Lk``. NaN if C_t, Hk or Lk is NaN."""
    m = _min_count(k)
    hk = hi.rolling(k, min_periods=m).max()
    lk = lo.rolling(k, min_periods=m).min()
    width = hk - lk
    pos = (c - lk) / width.where(width != 0)
    pos = pos.mask((width == 0) & c.notna(), 0.5)
    return _finite_or_nan(pos)


def _taker_imb(tbq_ok: pd.Series, qv_ok: pd.Series, k: int) -> pd.Series:
    """``trailing_sum(taker_buy_quote, k) / trailing_sum(quote_volume, k) - 0.5`` on volume_ok hours
    (window rule; with k = 4 all four hours are required)."""
    return _finite_or_nan(trailing_sum(tbq_ok, k) / trailing_sum(qv_ok, k) - 0.5)


def _check_cfg(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    names = [f["name"] for f in cfg["features"]]
    if names != FEATURE_NAMES:
        raise ValueError(f"config feature list {names} does not match FEATURE_NAMES {FEATURE_NAMES}")
    return {f["name"]: f for f in cfg["features"]}


def build_features(h: pd.DataFrame, partner: pd.DataFrame | None, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute the 16 features of PLAN §5.1 for every row of ``h`` (row t uses bars <= t only).

    Parameters
    ----------
    h, partner
        Canonical hourly input frames (:func:`ebmregime.series.hourly_inputs`). ``partner`` supplies
        ``xret_4h``; ``None`` gives an all-NaN ``xret_4h`` (and hence ``features_ok`` all False).
    cfg
        ``load_config("features")``. Window lengths are read from it.

    Definitions (sigma_t = sqrt(RV_168(t)), RV_k = trailing_mean(rv, k))
    ---------------------------------------------------------------------
    * ``ret_k = log(C_t / C_{t-k}) / (sigma_t sqrt(k))``, k = 1, 4, 24, 168.
    * ``rv_ratio_4h = log(RV_4 / RV_168)``, ``rv_ratio_24h = log(RV_24 / RV_168)``.
    * ``rv_level_pct``: share of present RV_168 values in bars t-4319..t that are ``<= RV_168(t)``.
    * ``qv_z_1h``: z-score of ``log QV_t`` vs mean/sd (ddof=1) of ``log QV`` over t-168..t-1.
    * ``qv_z_24h``: z-score of ``log QV24(t)`` vs mean/sd (ddof=1) of ``log QV24(t-24j)``, j=1..30.
    * ``range_1h = log(H_t / L_t) / sigma_t``.
    * ``pos_range_k = (C_t - L_k) / (H_k - L_k)`` (0.5 if H_k == L_k), k = 24, 168.
    * ``hour``, ``dow``: UTC hour (0..23) and weekday (0 = Monday) of the bar's open time, int64.
    * ``taker_imb_4h = sum_4 taker_buy_quote / sum_4 quote_volume - 0.5``.
    * ``xret_4h``: the partner's ``ret_4h`` (with the partner's own sigma), aligned on timestamp.

    Returns a frame on ``h.index`` with columns ``FEATURE_NAMES + ["features_ok"]``.
    """
    require_regular_hourly(h)
    spec = _check_cfg(cfg)
    sig_hours = int(cfg["sigma_hours"])

    c, hi, lo = _prices(h)
    rv = _rv(h)
    sig = sigma(h, sig_hours)
    out: dict[str, pd.Series] = {}

    for name in ("ret_1h", "ret_4h", "ret_24h", "ret_168h"):
        out[name] = scaled_return(c, sig, int(spec[name]["k"]))

    for name in ("rv_ratio_4h", "rv_ratio_24h"):
        s = spec[name]
        out[name] = _finite_or_nan(np.log(trailing_mean(rv, int(s["short"])) / trailing_mean(rv, int(s["long"]))))

    s = spec["rv_level_pct"]
    out["rv_level_pct"] = _rv_level_pct(trailing_mean(rv, int(s["rv_hours"])), int(s["history_hours"]))

    vok = _volume_ok(h)
    qv_ok = h["quote_volume"].where(vok)
    tbq_ok = h["taker_buy_quote"].where(vok)
    out["qv_z_1h"] = _qv_z_1h(qv_ok, int(spec["qv_z_1h"]["history_hours"]))
    s = spec["qv_z_24h"]
    out["qv_z_24h"] = _qv_z_24h(qv_ok, int(s["window_hours"]), int(s["history_days"]))

    out["range_1h"] = _finite_or_nan(np.log(hi / lo) / sig)
    for name in ("pos_range_24h", "pos_range_168h"):
        out[name] = _pos_range(c, hi, lo, int(spec[name]["k"]))

    idx = h.index
    out["hour"] = pd.Series(np.asarray(idx.hour, dtype=np.int64), index=idx)
    out["dow"] = pd.Series(np.asarray(idx.day_of_week, dtype=np.int64), index=idx)

    out["taker_imb_4h"] = _taker_imb(tbq_ok, qv_ok, int(spec["taker_imb_4h"]["k"]))

    k = int(spec["xret_4h"]["k"])
    if partner is None:
        out["xret_4h"] = pd.Series(np.nan, index=idx)
    else:
        require_regular_hourly(partner)
        pc, _, _ = _prices(partner)
        pret = scaled_return(pc, sigma(partner, sig_hours), k)
        out["xret_4h"] = pret.reindex(idx)

    f = pd.DataFrame({n: out[n] for n in FEATURE_NAMES}, index=idx)
    finite = np.isfinite(f.to_numpy(dtype=float)).all(axis=1)
    f["features_ok"] = h["valid"].astype(bool).to_numpy() & finite
    return f
