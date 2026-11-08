"""Regular bar grids, validity masks and 15m → 1h aggregation (D-04, D-06).

``valid`` means the bar exists on the grid and is not off-grid, truncated or an outage
placeholder. A 1h bar is valid only if all four of its 15m sub-bars are valid. Prices are never
forward-filled across gaps; invalid bars keep NaN prices, and later stages mask any feature or label
that touches them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parse import PRICE_VOL, STEP_MS

FLAGS = ["off_grid", "truncated", "placeholder"]
SUM_COLS = ["volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "n_trades"]


def to_grid(df: pd.DataFrame, interval: str, start: pd.Timestamp | None = None,
            end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Reindex cleaned klines onto the complete UTC grid ``[start, end]`` (defaults: first/last bar).

    Adds ``present`` (a row existed) and ``valid`` (present and not flagged).
    """
    freq = pd.Timedelta(milliseconds=STEP_MS[interval])
    start = df["ts"].iloc[0] if start is None else pd.Timestamp(start, tz="UTC")
    end = df["ts"].iloc[-1] if end is None else pd.Timestamp(end, tz="UTC")
    grid = pd.date_range(start, end, freq=freq, name="ts")
    out = df.set_index("ts").reindex(grid)
    out["present"] = out["open"].notna()
    for f in FLAGS:
        out[f] = out[f].astype("boolean").fillna(False).astype(bool)
    out["valid"] = out["present"] & ~out[FLAGS].any(axis=1)
    return out


def aggregate(grid: pd.DataFrame, base: str = "15m", target: str = "1h") -> pd.DataFrame:
    """Aggregate a base-interval grid (from :func:`to_grid`) into target bars.

    OHLC are first/max/min/last and volumes are sums. ``n_sub_valid`` counts valid sub-bars, and the
    target bar is ``valid`` only when every sub-bar is valid. The target grid is aligned to whole
    target periods: partial periods at the edges are dropped.
    """
    k = STEP_MS[target] // STEP_MS[base]
    if k * STEP_MS[base] != STEP_MS[target]:
        raise ValueError(f"{target} is not a multiple of {base}")
    g = grid.copy()
    key = g.index.floor(pd.Timedelta(milliseconds=STEP_MS[target]))
    grp = g.groupby(key)
    out = pd.DataFrame({
        "open": grp["open"].first(),
        "high": grp["high"].max(),
        "low": grp["low"].min(),
        "close": grp["close"].last(),
        **{c: grp[c].sum(min_count=1) for c in SUM_COLS},
        "n_sub": grp["open"].size(),
        "n_sub_valid": grp["valid"].sum(),
    })
    out.index.name = "ts"
    out = out[out["n_sub"] == k]  # drop partial periods at the grid edges
    # first()/last() skip NaN, so a missing first or last sub-bar would silently give a wrong
    # open or close. Such bars are invalid anyway; blank them so nobody uses them.
    out["valid"] = out["n_sub_valid"] == k
    out.loc[~out["valid"], ["open", "high", "low", "close"]] = np.nan
    return out.drop(columns="n_sub")


def crosscheck(agg: pd.DataFrame, archive: pd.DataFrame, rtol: float = 1e-9) -> pd.DataFrame:
    """Compare aggregated bars with the archive's own bars of the same interval.

    Only bars valid in both are compared. Returns, per column, the number of compared bars and of
    mismatches. Prices must match exactly; volumes may differ by float summation error (``rtol``).
    """
    both = agg.index.intersection(archive.index[archive["valid"]])
    both = both[agg.loc[both, "valid"].to_numpy()]
    rows = []
    for c in ["open", "high", "low", "close", *SUM_COLS]:
        a = agg.loc[both, c].to_numpy(dtype=float)
        b = archive.loc[both, c].to_numpy(dtype=float)
        tol = 0.0 if c in ("open", "high", "low", "close") else rtol
        bad = ~np.isclose(a, b, rtol=tol, atol=0.0)
        rows.append({"column": c, "compared": len(both), "mismatches": int(bad.sum())})
    return pd.DataFrame(rows)


def _disagree(agg: pd.DataFrame, archive: pd.DataFrame, cols: list[str], rtol: float) -> pd.Series:
    both = agg.index.intersection(archive.index[archive["valid"]])
    flag = pd.Series(False, index=agg.index)
    for c in cols:
        a = agg.loc[both, c].to_numpy(dtype=float)
        b = archive.loc[both, c].to_numpy(dtype=float)
        flag.loc[both[~np.isclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)]] = True
    return flag & agg["valid"]


def volume_suspect(agg: pd.DataFrame, archive: pd.DataFrame, rtol: float = 1e-9) -> pd.Series:
    """Hours whose aggregated volume or trade count disagrees with the archive's own 1h file.

    In a handful of hours per symbol the archive's 1h bar reports more volume and trades than the
    sum of its own 15m bars (D-40). We cannot tell which file is right, so these hours are flagged and
    volume-based features treat them as missing. Hours with no valid archive bar are not flagged.
    """
    return _disagree(agg, archive, SUM_COLS, rtol).rename("volume_suspect")


def price_conflict(agg: pd.DataFrame, archive: pd.DataFrame) -> pd.Series:
    """Hours whose aggregated OHLC differ from the archive's 1h bar. These are marked invalid (D-40)."""
    return _disagree(agg, archive, ["open", "high", "low", "close"], 0.0).rename("price_conflict")


def apply_archive_flags(agg: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Add ``volume_suspect`` and ``price_conflict``. Price-conflict hours become invalid, with blank prices."""
    out = agg.copy()
    out["volume_suspect"] = volume_suspect(agg, archive)
    out["price_conflict"] = price_conflict(agg, archive)
    out.loc[out["price_conflict"], "valid"] = False
    out.loc[out["price_conflict"], ["open", "high", "low", "close"]] = np.nan
    return out


__all__ = ["to_grid", "aggregate", "crosscheck", "volume_suspect", "price_conflict",
           "apply_archive_flags", "FLAGS", "PRICE_VOL"]
