"""Causal market regimes R1 (volatility), R2 (trend) and R3 (liquidity) (PLAN §8, D-20).

Timing
------
Regimes are computed once per UTC calendar day. Day ``d`` covers the hourly bars with open times
``[d 00:00, d+1 00:00)``; its last bar (23:00) closes at ``d+1 00:00``. The regime computed from
days ``<= d`` is therefore known at ``d+1 00:00`` and is applied to all 24 hours of day ``d+1``
(:func:`to_hourly`). The hourly state at hour ``t`` uses only days ``< floor_day(t)``, i.e. only
bars that closed at or before ``floor_day(t) <= t``. Strictly causal.

Daily inputs (:func:`daily_inputs`), from the canonical hourly frame of :mod:`ebmregime.series`
--------------------------------------------------------------------------------------------
* ``RV_d  = 24 * mean(rv_t : t in day d, rv_t not NaN)`` if at least ceil(0.90 * 24) = 22 of the
  24 hours have a non-NaN ``rv``; else NaN. (With all 24 present this is the plain sum.)
* ``C_d   = close of the 23:00 bar of day d`` (NaN if that bar is invalid or absent).
* ``r_d   = log(C_d / C_{d-1})`` (calendar-consecutive days; NaN if either close is NaN).
* ``DV_d  = 24 * mean(quote_volume_t : t in day d, volume_ok_t)`` if at least 22 such hours have a
  non-NaN quote volume; else NaN.

Gap tolerance (D-43; constants imported from :mod:`ebmregime.series`)
--------------------------------------------------------------------
* Daily aggregates of raw hours use the **window rule** ``MIN_FRAC = 0.90``: >= ceil(0.90 * 24)
  = 22 present hours.
* Multi-day windows over daily values ("n-day window ending at d" = days ``d-n+1 .. d``
  inclusive) use the **history rule** ``MIN_FRAC_HISTORY = 0.50``: the statistic is computed from
  the non-NaN values when at least ``ceil(0.50 n)`` are present, else NaN. Days before the start
  of the data count as missing. So: s 30 -> 15, T 90 -> 45, ILLIQ 30 -> 15, dv_z 365 -> 183.
* The percentile histories keep their explicit count ``min_history_days`` (365 defined values),
  which is a count, not a fraction.

Regimes (:func:`daily_regimes`)
------------------------------
* R1 ``r1_vol`` (expanding; sensitivity under D-44): ``s_d = sqrt(mean(RV over d-29..d))``
  (>= 15 days) ``p_d = #{j in H_d : s_j <= s_d} / |H_d|``
  where ``H_d`` = days ``<= d`` with non-NaN ``s`` (expanding history, includes d). ``p_d`` is NaN
  if ``s_d`` is NaN or ``|H_d| < 365``. State 0 (low) if ``p < cut1``, 2 (high) if ``p > cut2``,
  else 1 (mid).
* R1 ``r1_vol_2y`` (**primary under D-44**; ``regimes.primary`` in the config, see
  :func:`primary_r1`): same, with ``H_d`` restricted to days ``d-729 .. d`` (still >= 365 values).
* R2 ``r2_trend``: ``T_d = mean(r) / (sd(r, ddof=1) / sqrt(m))`` over the m non-NaN returns in
  ``d-89 .. d`` (m >= 45). State +1 (bull) if ``T > 1``, -1 (bear) if ``T < -1``, else 0.
* R3 ``r3_liq``: ``ILLIQ_d = mean(|r_j| / DV_j over d-29..d)`` (>= 15 non-NaN terms; ``|r|/DV``
  is NaN when ``DV <= 0``). ``p_illiq`` = expanding percentile as in R1 (min 365 values).
  States 0/1/2 as R1, 2 = most illiquid.
* ``r3_dv_z`` (alternative, continuous only): ``(log DV_d - mean) / sd`` with mean and sd
  (ddof=1) of ``log DV`` over ``d-365 .. d-1`` (>= ceil(0.50 * 365) = 183 values).

States are pandas nullable ``Int8`` columns; ``<NA>`` means undefined.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ebmregime.series import MIN_FRAC, MIN_FRAC_HISTORY

DAY = pd.Timedelta(days=1)
HOURS_PER_DAY = 24
MIN_HOURS_PER_DAY = math.ceil(MIN_FRAC * HOURS_PER_DAY)   # window rule (D-43): 22 of 24

STATE_COLUMNS = ("r1_vol", "r1_vol_2y", "r2_trend", "r3_liq")
R1_COLUMNS = ("r1_vol", "r1_vol_2y")
R1_LOW, R1_MID, R1_HIGH = 0, 1, 2
R2_BEAR, R2_SIDEWAYS, R2_BULL = -1, 0, 1


# ---------------------------------------------------------------------------------------------
# Daily inputs
# ---------------------------------------------------------------------------------------------

def _day_matrix(x: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    """Hourly values as an ``(n_days, 24)`` array; hours absent from ``x`` are NaN."""
    grid = pd.date_range(days[0], periods=len(days) * HOURS_PER_DAY, freq="1h")
    return x.reindex(grid).to_numpy(dtype=float).reshape(len(days), HOURS_PER_DAY)


def _daily_rescaled_sum(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``24 * mean`` of the non-NaN values of each row if >= 22 are present, else NaN; and counts.

    Computed row by row in numpy (not pandas groupby), so each day's value depends bitwise only
    on that day's hours. (pandas groupby-mean picks a different summation path depending on NaNs
    anywhere in the column, which changes early values in the last bit when later data change.)
    """
    cnt = np.count_nonzero(~np.isnan(m), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(m, axis=1) / cnt
    return np.where(cnt >= MIN_HOURS_PER_DAY, mean * HOURS_PER_DAY, np.nan), cnt


def daily_inputs(h: pd.DataFrame) -> pd.DataFrame:
    """Daily RV, close, log return and dollar volume from the canonical hourly frame.

    Parameters
    ----------
    h : hourly frame from :func:`ebmregime.series.hourly_inputs` (needs ``rv``, ``close``,
        ``valid``, ``quote_volume``, ``volume_ok``), indexed by tz-aware UTC bar open time.

    Returns
    -------
    DataFrame indexed by a complete daily range of UTC midnights (``day``), from the first to the
    last calendar day touched by ``h``, with columns:

    * ``RV``    = 24 * mean of non-NaN hourly ``rv`` in the day, if >= 22 present, else NaN;
    * ``close`` = close of the day's 23:00 bar (NaN if invalid or absent);
    * ``r``     = log(close_d / close_{d-1});
    * ``DV``    = 24 * mean of ``quote_volume`` over ``volume_ok`` hours, if >= 22, else NaN;
    * ``n_rv``, ``n_dv``: number of hours used for RV and DV.
    """
    idx = h.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise ValueError("h must have a tz-aware DatetimeIndex")
    days = pd.date_range(idx[0].floor("D"), idx[-1].floor("D"), freq="D", name="day")

    if not idx.is_unique:
        raise ValueError("h has duplicate timestamps")

    rv_m = _day_matrix(h["rv"].astype(float), days)
    dv_m = _day_matrix(h["quote_volume"].astype(float).where(h["volume_ok"].astype(bool)), days)
    close_m = _day_matrix(h["close"].astype(float).where(h["valid"].astype(bool)), days)

    close = close_m[:, HOURS_PER_DAY - 1]
    r = np.full(len(days), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = np.log(close[1:] / close[:-1])

    RV, n_rv = _daily_rescaled_sum(rv_m)
    DV, n_dv = _daily_rescaled_sum(dv_m)
    return pd.DataFrame({"RV": RV, "close": close, "r": r, "DV": DV,
                         "n_rv": n_rv.astype(int), "n_dv": n_dv.astype(int)}, index=days)


# ---------------------------------------------------------------------------------------------
# Windowed statistics and percentiles
# ---------------------------------------------------------------------------------------------

def _min_count(n: int, min_frac: float = MIN_FRAC_HISTORY) -> int:
    return math.ceil(min_frac * n)


def _trailing_stats(x: pd.Series, n: int, with_sd: bool = False,
                    min_frac: float = MIN_FRAC_HISTORY
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Count, mean and sd (ddof=1) of the non-NaN values in each window ``d-n+1 .. d``.

    Days before the start of ``x`` count as missing. Mean and sd are NaN unless the count is at
    least ``ceil(min_frac n)`` (default: history rule, 0.50, for windows over daily values). Each window is reduced independently (numpy, two-pass sd), so a value
    depends bitwise only on its own window: strictly causal, even at the last-bit level.
    """
    v = x.to_numpy(dtype=float)
    w = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(n - 1, np.nan), v]), n)
    cnt = np.count_nonzero(~np.isnan(w), axis=1)
    ok = cnt >= _min_count(n, min_frac)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, np.nansum(w, axis=1) / cnt, np.nan)
        sd = None
        if with_sd:
            ss = np.nansum((w - mean[:, None]) ** 2, axis=1)
            sd = np.where(ok & (cnt >= 2), np.sqrt(ss / (cnt - 1)), np.nan)
    return cnt, mean, sd


def _trailing_mean(x: pd.Series, n: int) -> pd.Series:
    """Mean over ``d-n+1 .. d`` of the non-NaN values, if at least ceil(0.50 n) present."""
    return pd.Series(_trailing_stats(x, n)[1], index=x.index)


def percentile_rank(s: pd.Series, min_history: int, history_days: int | None = None) -> pd.Series:
    """Causal percentile of ``s_d`` within its own history.

    ``p_d = #{j in H_d : s_j <= s_d} / |H_d|`` where ``H_d`` holds the non-NaN values of ``s`` on
    days ``<= d`` (expanding, ``history_days=None``) or on days ``d-history_days+1 .. d``. NaN if
    ``s_d`` is NaN or ``|H_d| < min_history``. Exact, O(n^2) worst case (n ~ 3,400 days: ~10 ms).
    """
    v = s.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    ok = ~np.isnan(v)
    for d in range(n):
        if not ok[d]:
            continue
        lo = 0 if history_days is None else max(0, d - history_days + 1)
        w = v[lo:d + 1]
        w = w[ok[lo:d + 1]]
        if len(w) >= min_history:
            out[d] = np.count_nonzero(w <= v[d]) / len(w)
    return pd.Series(out, index=s.index)


def _tercile_state(p: pd.Series, cuts: Sequence[float]) -> pd.Series:
    """0 if p < cuts[0], 2 if p > cuts[1], else 1; <NA> where p is NaN."""
    c1, c2 = float(cuts[0]), float(cuts[1])
    st = pd.Series(pd.array([pd.NA] * len(p), dtype="Int8"), index=p.index)
    st[p < c1] = R1_LOW
    st[(p >= c1) & (p <= c2)] = R1_MID
    st[p > c2] = R1_HIGH
    return st


def trend_tstat(r: pd.Series, n: int) -> pd.Series:
    """``T_d = mean / (sd(ddof=1) / sqrt(m))`` of the m non-NaN values of ``r`` in ``d-n+1..d``.

    NaN unless m >= ceil(0.50 n) (history rule). If sd == 0, T = +/-inf (sign of the mean) or NaN (mean 0).
    """
    m, mean, sd = _trailing_stats(r, n, with_sd=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(m))
    return pd.Series(t, index=r.index)  # NaN wherever mean/sd are NaN (m < ceil(0.50 n))


# ---------------------------------------------------------------------------------------------
# Daily regimes
# ---------------------------------------------------------------------------------------------

def primary_r1(cfg_regimes: dict[str, Any]) -> str:
    """Name of the primary R1 state column, read from ``cfg_regimes["primary"]`` (D-44:
    ``"r1_vol_2y"``). Raises if the key is missing or not an R1 column. Callers should use this
    rather than hard-coding a column, so the pre-registered choice lives in the config only."""
    name = cfg_regimes.get("primary")
    if name not in R1_COLUMNS:
        raise ValueError(f"regimes.primary must be one of {R1_COLUMNS}, got {name!r}")
    return str(name)


def daily_regimes(daily: pd.DataFrame, cfg_regimes: dict[str, Any]) -> pd.DataFrame:
    """Continuous regime scores and integer states per day, from :func:`daily_inputs`.

    ``cfg_regimes`` is the ``regimes`` section of ``configs/prereg.yaml``. The row for day ``d``
    uses days ``<= d`` only; it applies to the hours of day ``d+1`` (see :func:`to_hourly`).

    Returns a frame on ``daily.index`` with continuous columns

    * ``s``        = sqrt(mean RV over d-29..d) (daily vol);
    * ``p``        = expanding percentile of ``s`` (min 365 values);
    * ``p_2y``     = percentile of ``s`` within the trailing 730 days (min 365 values);
    * ``T``        = 90-day trend t-statistic of daily log returns;
    * ``ILLIQ``    = mean of |r|/DV over d-29..d;
    * ``p_illiq``  = expanding percentile of ``ILLIQ`` (min 365 values);
    * ``dv_z``     = z-score of log DV_d against log DV over d-365..d-1;

    and nullable ``Int8`` state columns ``r1_vol``, ``r1_vol_2y`` (0 low / 1 mid / 2 high),
    ``r2_trend`` (-1 bear / 0 sideways / +1 bull), ``r3_liq`` (0 / 1 / 2 = most illiquid).
    ``<NA>`` = undefined. Formulas are in the module docstring. Only the named sub-sections of
    ``cfg_regimes`` are read (scalar keys such as ``primary`` are ignored here); the primary R1
    column name is stored in ``out.attrs["primary_r1"]`` when ``primary`` is present.
    """
    c1 = cfg_regimes["r1_vol"]
    c1b = cfg_regimes["r1_vol_2y"]
    c2 = cfg_regimes["r2_trend"]
    c3 = cfg_regimes["r3_liq"]
    c3z = cfg_regimes["r3_dv_z"]

    out = pd.DataFrame(index=daily.index)

    # R1 ----------------------------------------------------------------------------------
    if int(c1b["rv_days"]) != int(c1["rv_days"]):
        raise ValueError("r1_vol and r1_vol_2y must use the same rv_days")
    s = np.sqrt(_trailing_mean(daily["RV"].astype(float), int(c1["rv_days"])))
    out["s"] = s
    out["p"] = percentile_rank(s, int(c1["min_history_days"]))
    out["p_2y"] = percentile_rank(s, int(c1b["min_history_days"]), int(c1b["history_days"]))

    # R2 ----------------------------------------------------------------------------------
    T = trend_tstat(daily["r"].astype(float), int(c2["days"]))
    out["T"] = T

    # R3 ----------------------------------------------------------------------------------
    dv = daily["DV"].astype(float)
    dv_pos = dv.where(dv > 0)
    illiq_d = daily["r"].astype(float).abs() / dv_pos
    illiq = _trailing_mean(illiq_d, int(c3["days"]))
    out["ILLIQ"] = illiq
    out["p_illiq"] = percentile_rank(illiq, int(c3["min_history_days"]))

    k = int(c3z["history_days"])
    log_dv = np.log(dv_pos)
    _, mu_prev, sd_prev = _trailing_stats(log_dv.shift(1), k, with_sd=True)  # days d-k .. d-1
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dv_z"] = (log_dv.to_numpy() - mu_prev) / sd_prev

    # States ------------------------------------------------------------------------------
    out["r1_vol"] = _tercile_state(out["p"], c1["cuts"])
    out["r1_vol_2y"] = _tercile_state(out["p_2y"], c1b["cuts"])
    thr = float(c2["t_threshold"])
    st2 = pd.Series(pd.array([pd.NA] * len(T), dtype="Int8"), index=T.index)
    st2[T.notna()] = R2_SIDEWAYS
    st2[T > thr] = R2_BULL
    st2[T < -thr] = R2_BEAR
    out["r2_trend"] = st2
    out["r3_liq"] = _tercile_state(out["p_illiq"], c3["cuts"])
    if "primary" in cfg_regimes:
        out.attrs["primary_r1"] = primary_r1(cfg_regimes)
    return out


# ---------------------------------------------------------------------------------------------
# Hourly mapping and window summaries
# ---------------------------------------------------------------------------------------------

def to_hourly(daily_states: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map daily rows to hours: hour ``t`` gets the row of day ``floor_day(t) - 1 day``.

    Hours whose previous day is absent from ``daily_states`` get NaN / <NA>. Dtypes are kept.
    """
    if not isinstance(hourly_index, pd.DatetimeIndex) or hourly_index.tz is None:
        raise ValueError("hourly_index must be a tz-aware DatetimeIndex")
    src_day = hourly_index.floor("D") - DAY
    out = daily_states.reindex(src_day)
    out.index = hourly_index
    return out


def _hours_in(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(hours=1))


def window_shares(hourly_state: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                  states: Iterable[int]) -> dict[Any, float]:
    """Share of each state over the hours ``[start, end)``.

    ``share[s] = #{t in [start, end) : state_t == s} / #{t in [start, end) : state_t defined}``
    (NaN if no hour is defined), and ``coverage = #defined / N`` where N is the number of hours in
    ``[start, end)`` on the calendar (hours missing from the series count as undefined).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    x = hourly_state[(hourly_state.index >= start) & (hourly_state.index < end)]
    defined = x.notna()
    n_def = int(defined.sum())
    n_all = _hours_in(start, end)
    res: dict[Any, float] = {}
    xd = x[defined]
    for s in states:
        res[s] = float((xd == s).sum()) / n_def if n_def else float("nan")
    res["coverage"] = n_def / n_all if n_all > 0 else float("nan")
    return res


def _state_at(hourly_state: pd.Series, t: pd.Timestamp) -> float:
    if t not in hourly_state.index:
        return float("nan")
    v = hourly_state.loc[t]
    return float("nan") if pd.isna(v) else float(v)


def regime_change_covariates(hourly_state: pd.Series, taus: Sequence[pd.Timestamp],
                             train_months: int, high_state: int = R1_HIGH) -> pd.DataFrame:
    """Window labels and regime-change covariates per retrain time tau (PLAN §8).

    For each ``tau_i`` (training window ``W_i = [tau_i - train_months, tau_i)``):

    * ``pi_high_window`` = share of ``high_state`` among defined hours of ``W_i``;
    * ``coverage_window`` = defined hours / calendar hours of ``W_i``;
    * ``pi_high_added``   = share in the added month ``[tau_{i-1}, tau_i)``;
    * ``pi_high_dropped`` = share in the dropped month ``[tau_{i-1} - m, tau_i - m)``;
    * ``C = |pi_high_added - pi_high_dropped|`` (NaN for i = 0 or if either share is undefined);
    * ``state_tau`` = hourly state at hour ``tau_i`` (the ex-ante test-month label);
    * ``E = 1[state_tau_i != state_tau_{i-1}]`` (NaN for i = 0 or if either state is undefined).

    ``m = pd.DateOffset(months=train_months)``. Shares use :func:`window_shares`.
    """
    off = pd.DateOffset(months=int(train_months))
    taus = [pd.Timestamp(t) for t in taus]
    rows = []
    for i, tau in enumerate(taus):
        w = window_shares(hourly_state, tau - off, tau, [high_state])
        row = {"tau": tau, "pi_high_window": w[high_state], "coverage_window": w["coverage"],
               "pi_high_added": np.nan, "pi_high_dropped": np.nan, "C": np.nan,
               "state_tau": _state_at(hourly_state, tau), "E": np.nan}
        if i > 0:
            prev = taus[i - 1]
            a = window_shares(hourly_state, prev, tau, [high_state])[high_state]
            d = window_shares(hourly_state, prev - off, tau - off, [high_state])[high_state]
            row["pi_high_added"], row["pi_high_dropped"] = a, d
            row["C"] = abs(a - d)  # NaN propagates
            s0, s1 = rows[-1]["state_tau"], row["state_tau"]
            row["E"] = np.nan if (np.isnan(s0) or np.isnan(s1)) else float(s1 != s0)
        rows.append(row)
    cols = ["pi_high_window", "coverage_window", "pi_high_added", "pi_high_dropped", "C",
            "state_tau", "E"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="tau"))
    return pd.DataFrame(rows).set_index("tau")[cols]
