"""Target construction on the canonical hourly frame (PLAN §5.2; D-09, D-33, D-36).

Time convention (see :mod:`ebmregime.series`): row t is bar t, indexed by its open time ``ts_t``.
A label at row t with horizon h uses bars ``t+1 .. t+h`` (plus the close ``C_t`` of bar t, known
at ``ts_t + 1h``) and becomes known at ``ts_t + (h + 1)`` hours, the close of bar t+h.

Validity (strict forward window, D-06)
--------------------------------------
Every label at row t is NaN unless bar t **and** every bar ``t+1 .. t+h`` is ``valid``. This holds
even for the direction label, whose formula only uses the endpoints ``C_t`` and ``C_{t+h}``: a gap
inside the horizon masks the label. The last h rows are therefore always NaN.

Targets
-------
* Direction:  ``y_dir_t = 1[ log(C_{t+h} / C_t) > 0 ]``; exact zero returns give NaN
  (``drop_zero_returns``).
* Normalised forward vol:
  ``y_vol_t = log( mean(rv_{t+1..t+h}) / RV168_t )`` with ``mean(rv_{t+1..t+h})`` strict (same
  definition as :func:`ebmregime.series.forward_mean_strict`, computed by :func:`_forward_mean_exact`
  so no float rounding from rows after t+h leaks in) and ``RV168_t`` from
  :func:`ebmregime.series.trailing_mean` over ``t-167 .. t`` (95 % gap tolerance). Both are
  per-hour realised variances, so the ratio is scale-free.
* Shuffled control: the direction label circularly shifted by a fixed per-symbol offset (in hours)
  over the rows with ``ts >= analysis_start``; see :func:`shuffled` and :func:`shuffle_offset`.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pandas as pd

from ..series import require_regular_hourly, trailing_mean

MAX_DRAWS = 10_000


def _strict_window_ok(valid: pd.Series, horizon: int) -> pd.Series:
    """True at row t iff bar t and every bar ``t+1 .. t+horizon`` exist and are valid.

    Uses integer cumulative counts of invalid hours, so it is exact:
    ``bad(t..t+h) = cs[t+h+1] - cs[t]`` with ``cs[k] = #invalid in rows 0..k-1``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    v = valid.fillna(False).to_numpy(dtype=bool)
    n = len(v)
    cs = np.concatenate([[0], np.cumsum(~v)])
    ok = np.zeros(n, dtype=bool)
    m = n - horizon                      # rows 0 .. m-1 have a complete forward window
    if m > 0:
        ok[:m] = (cs[horizon + 1: horizon + 1 + m] - cs[:m]) == 0
    return pd.Series(ok, index=valid.index, name="window_ok")


def _forward_mean_exact(x: pd.Series, horizon: int) -> pd.Series:
    """Strict forward mean ``mean(x_{t+1} .. x_{t+h})``; NaN if any value is NaN or the window runs
    past the end.

    Same values as :func:`ebmregime.series.forward_mean_strict` up to float rounding, but each
    window is summed on its own. ``forward_mean_strict`` runs pandas' online rolling mean over the
    reversed series, so its value at t carries ~1e-16 rounding from rows after t+h; here the value
    at t depends bit-for-bit only on rows t+1 .. t+h.
    """
    a = x.to_numpy(dtype="float64")
    out = np.full(len(a), np.nan)
    if len(a) > horizon:
        w = np.lib.stride_tricks.sliding_window_view(a[1:], horizon)   # w[t] = a[t+1 : t+1+h]
        out[: len(w)] = w.sum(axis=1) / horizon
    return pd.Series(out, index=x.index)


def direction(h: pd.DataFrame, horizon: int, drop_zero: bool = True) -> pd.Series:
    """Direction label ``y_t = 1[log(C_{t+h}/C_t) > 0]`` (float64: 1.0, 0.0 or NaN).

    ``h`` is the canonical hourly frame (needs ``close`` and ``valid``). The label is NaN unless
    bar t and every bar ``t+1 .. t+h`` is valid. With ``drop_zero`` (the pre-registered
    ``drop_zero_returns: true``) an exact zero log return gives NaN; otherwise it gives 0.0.
    """
    c = h["close"].astype("float64")
    r = np.log(c.shift(-horizon)) - np.log(c)
    y = pd.Series(np.nan, index=h.index, dtype="float64")
    y[r > 0] = 1.0
    y[r < 0] = 0.0
    if not drop_zero:
        y[r == 0] = 0.0
    y = y.where(_strict_window_ok(h["valid"], horizon))
    return y.rename(f"y_dir_h{horizon}")


def forward_vol(h: pd.DataFrame, horizon: int, trailing_hours: int = 168) -> pd.Series:
    """Normalised forward volatility ``y_t = log( mean(rv_{t+1..t+h}) / trailing_mean(rv, K)_t )``.

    ``K = trailing_hours`` (168). The forward mean (:func:`_forward_mean_exact`) is strict (any NaN
    rv in ``t+1..t+h`` gives NaN); the trailing mean over ``t-K+1 .. t`` uses the 95 % gap
    tolerance of :func:`ebmregime.series.trailing_mean`. The label is NaN if either mean is NaN, if bar t or any
    bar in ``t+1 .. t+h`` is invalid, or if the log is not finite (a zero forward or trailing mean,
    i.e. a perfectly flat price path, would give +-inf).
    """
    rv = h["rv"].astype("float64")
    fwd = _forward_mean_exact(rv, horizon)
    trail = trailing_mean(rv, trailing_hours)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(fwd / trail)
    y = y.where(np.isfinite(y)).where(_strict_window_ok(h["valid"], horizon))
    return y.astype("float64").rename(f"y_vol_h{horizon}")


def shuffle_offset(n_rows: int, symbol: str, cfg_shuffled: dict[str, Any]) -> int:
    """Circular-shift offset in hours for one symbol (D-09).

    Draws ``k`` uniformly from the integers ``[min_days*24, max_days*24]`` (both inclusive) with
    ``np.random.default_rng([seed, zlib.crc32(symbol)])`` and redraws until ``k`` is not a multiple
    of any value in ``forbid_multiples_of_hours`` and ``k < n_rows``. Deterministic in
    ``(seed, symbol)``; ``n_rows`` only matters if it is below ``max_days*24`` (redraws then depend
    on it). Raises ``ValueError`` if no admissible offset exists.
    """
    lo = int(cfg_shuffled["min_days"]) * 24
    hi = int(cfg_shuffled["max_days"]) * 24
    forbid = [int(m) for m in cfg_shuffled.get("forbid_multiples_of_hours", [])]
    top = min(hi, n_rows - 1)
    cand = np.arange(lo, top + 1) if top >= lo else np.arange(0)
    if not any(all(k % m for m in forbid) for k in cand):
        raise ValueError(f"no admissible offset in [{lo}, {top}] hours for n_rows={n_rows}")
    rng = np.random.default_rng([int(cfg_shuffled["seed"]), zlib.crc32(symbol.encode())])
    for _ in range(MAX_DRAWS):
        k = int(rng.integers(lo, hi, endpoint=True))
        if k < n_rows and all(k % m for m in forbid):
            return k
    raise RuntimeError("shuffle_offset: too many redraws")  # practically unreachable


def _utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def shuffled(y_dir: pd.Series, analysis_start: str, offset: int) -> pd.Series:
    """Shuffled control: circular shift of ``y_dir`` over the rows with ``ts >= analysis_start``.

    Let ``r`` be ``y_dir`` restricted to those rows (length N, NaNs kept). Then
    ``y_shuf[i] = r[(i + offset) mod N]`` on the restricted rows and NaN before ``analysis_start``.
    NaNs travel with their labels, so the label's marginal distribution, autocorrelation and
    overlap structure are kept (the only broken adjacency is the single wrap point).
    """
    mask = y_dir.index >= _utc(analysis_start)
    r = y_dir.to_numpy(dtype="float64")[mask]
    n = len(r)
    if not 0 < offset < n:
        raise ValueError(f"offset must be in (0, {n}); got {offset}")
    out = np.full(len(y_dir), np.nan)
    out[mask] = np.roll(r, -offset)      # np.roll(r, -k)[i] == r[(i + k) % n]
    name = str(y_dir.name).replace("y_dir", "y_shuf") if y_dir.name else "y_shuf"
    return pd.Series(out, index=y_dir.index, name=name)


def build_targets(h: pd.DataFrame, symbol: str, prereg_cfg: dict[str, Any], data_cfg: dict[str, Any],
                  horizons: list[int] | None = None) -> pd.DataFrame:
    """All targets for one symbol on the full hourly index of ``h``.

    Columns ``y_dir_h{h}``, ``y_vol_h{h}``, ``y_shuf_h{h}`` per horizon (default: the primary then
    the secondary horizons of ``prereg_cfg["targets"]["horizons"]``). ``y_dir`` and ``y_vol`` are
    defined on every row (also before ``analysis_start``); ``y_shuf`` is NaN before it.

    The shuffle offset is drawn **once per symbol** (with N = number of rows from
    ``analysis_start``) and reused for every horizon. ``df.attrs["shuffle_offset_hours"]`` maps
    horizon -> offset (hours).
    """
    require_regular_hourly(h)
    tcfg = prereg_cfg["targets"]
    if horizons is None:
        hz = tcfg["horizons"]
        horizons = [int(hz["primary"])] + [int(x) for x in hz.get("secondary", [])]
    drop_zero = bool(tcfg["direction"].get("drop_zero_returns", True))
    trailing = int(tcfg["vol"]["trailing_hours"])
    start = data_cfg["symbols"][symbol]["analysis_start"]
    n_rows = int((h.index >= _utc(start)).sum())
    offset = shuffle_offset(n_rows, symbol, tcfg["shuffled"])

    cols: dict[str, pd.Series] = {}
    for hz_ in horizons:
        yd = direction(h, hz_, drop_zero=drop_zero)
        cols[f"y_dir_h{hz_}"] = yd
        cols[f"y_vol_h{hz_}"] = forward_vol(h, hz_, trailing)
        cols[f"y_shuf_h{hz_}"] = shuffled(yd, start, offset)
    out = pd.DataFrame(cols, index=h.index)
    out.attrs["shuffle_offset_hours"] = {int(hz_): offset for hz_ in horizons}
    out.attrs["symbol"] = symbol
    out.attrs["analysis_start"] = str(start)
    return out
