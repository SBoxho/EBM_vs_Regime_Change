"""Parse and clean Binance spot kline files (D-06).

The schema has 12 columns: open_time, open, high, low, close, volume, close_time, quote_volume,
n_trades, taker_buy_base, taker_buy_quote, ignore. Spot files have no header row, but the parser
skips a non-numeric first line anyway (futures files have one).

**Epoch units.** Spot timestamps are milliseconds up to the 2024-12 file and **microseconds from
the 2025-01 file onward**. Units are detected from the magnitude of the values, not from the file
date. A file that mixes units is rejected. Parsing a µs value with ``unit="ms"`` would give a
year-58,000 timestamp without any error, which is why the detection is explicit.

**Cleaning flags.**
* ``off_grid``: open_time is not on the bar grid (e.g. bars shifted +28m14s after the 2018-02
  outage). The bar is snapped down to the grid and flagged.
* ``truncated``: close_time − open_time + 1 ≠ step (short bars at outages).
* ``placeholder``: zero volume, zero trades and O=H=L=C (outage placeholders).
A snapped grid collision raises; it must never pass silently.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
PRICE_VOL = ["open", "high", "low", "close", "volume", "quote_volume",
             "taker_buy_base", "taker_buy_quote"]
STEP_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}

# Plausible open_time range in ms: 2017-01-01 .. 2100-01-01.
_MIN_MS, _MAX_MS = 1_483_228_800_000, 4_102_444_800_000


class ParseError(ValueError):
    pass


def detect_unit(values: np.ndarray) -> str:
    """Return ``"ms"`` or ``"us"`` for an array of epoch integers. Raises on mixed or implausible values."""
    v = np.asarray(values, dtype=np.int64)
    is_us = v > 10**14
    if is_us.all():
        unit, ms = "us", v // 1000
    elif not is_us.any():
        unit, ms = "ms", v
    else:
        raise ParseError("mixed ms/µs timestamps within one file")
    if ms.min() < _MIN_MS or ms.max() > _MAX_MS:
        raise ParseError(f"timestamps outside plausible range: {ms.min()}..{ms.max()} ms")
    return unit


def to_ms(values: np.ndarray, unit: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    return v // 1000 if unit == "us" else v


def read_kline_csv(fh: io.BufferedIOBase) -> tuple[pd.DataFrame, str]:
    """Read one kline CSV stream. Returns ``(frame with open_time/close_time in ms, detected unit)``."""
    raw = fh.read()
    first = raw[: raw.find(b"\n") if b"\n" in raw else len(raw)]
    skip = 0 if first[:1].isdigit() else 1
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS, skiprows=skip,
                     dtype={"open_time": "int64", "close_time": "int64", "n_trades": "int64",
                            **{c: "float64" for c in PRICE_VOL}, "ignore": "float64"})
    if df.empty:
        return df.drop(columns="ignore"), "ms"
    unit = detect_unit(df["open_time"].to_numpy())
    if detect_unit(df["close_time"].to_numpy()) != unit:
        raise ParseError("open_time and close_time use different units")
    df["open_time"] = to_ms(df["open_time"].to_numpy(), unit)
    df["close_time"] = to_ms(df["close_time"].to_numpy(), unit)
    return df.drop(columns="ignore"), unit


def read_kline_zip(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a monthly kline zip. The archive puts exactly one CSV in each zip; anything else raises."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ParseError(f"{path.name}: expected one member, found {names}")
        with z.open(names[0]) as fh:
            return read_kline_csv(fh)


def clean_klines(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Snap to the bar grid, add quality flags, sort, and drop exact duplicates.

    Output columns: ``ts`` (bar open, tz-aware UTC), the OHLCV columns, ``n_trades``, and the flags
    ``off_grid``, ``truncated`` and ``placeholder``.
    """
    step = STEP_MS[interval]
    out = df.copy()
    snapped = out["open_time"] - out["open_time"] % step
    out["off_grid"] = snapped != out["open_time"]
    out["truncated"] = (out["close_time"] - out["open_time"] + 1) != step
    out["placeholder"] = ((out["volume"] == 0) & (out["n_trades"] == 0)
                          & (out["open"] == out["high"]) & (out["high"] == out["low"])
                          & (out["low"] == out["close"]))
    out["open_ms"] = snapped
    out = out.drop_duplicates()  # identical rows (e.g. a bar repeated in adjacent files)
    if out["open_ms"].duplicated().any():
        dup = out.loc[out["open_ms"].duplicated(keep=False), "open_ms"].unique()[:5]
        raise ParseError(f"{len(dup)}+ grid collisions after snapping, e.g. open_ms={list(dup)}")
    out = out.sort_values("open_ms").reset_index(drop=True)
    out.insert(0, "ts", pd.to_datetime(out["open_ms"], unit="ms", utc=True))
    return out.drop(columns=["open_time", "close_time", "open_ms"])


def load_symbol(paths: list[Path], interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse and clean every monthly zip of one symbol and interval.

    Returns ``(cleaned klines, per-file table of file name, detected unit and row count)``.
    """
    frames, info = [], []
    for p in sorted(paths):
        df, unit = read_kline_zip(p)
        frames.append(df)
        info.append({"file": p.name, "unit": unit, "rows": len(df)})
    return clean_klines(pd.concat(frames, ignore_index=True), interval), pd.DataFrame(info)
