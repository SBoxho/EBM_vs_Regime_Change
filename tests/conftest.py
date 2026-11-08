"""Shared fixtures: synthetic kline files in the archive's exact format (no real data in tests)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

STEP = {"15m": 900_000, "1h": 3_600_000}
T0 = 1_735_689_600_000  # 2025-01-01 00:00 UTC in ms


def kline_rows(start_ms: int, n: int, interval: str = "15m", unit: str = "ms", price: float = 100.0,
               skip: set[int] = frozenset()) -> list[list]:
    """Rows in Binance spot kline format. ``skip`` drops bar indices to create gaps."""
    step = STEP[interval]
    mult = 1000 if unit == "us" else 1
    rows = []
    for i in range(n):
        if i in skip:
            continue
        o = start_ms + i * step
        p = price + i
        rows.append([o * mult, p, p + 2, p - 1, p + 1, 10.0 + i, (o + step) * mult - 1,
                     (10.0 + i) * p, 5 + i, 4.0, 4.0 * p, 0])
    return rows


def write_zip(path: Path, rows: list[list], header: bool = False) -> Path:
    buf = io.StringIO()
    if header:
        buf.write("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                  "taker_buy_volume,taker_buy_quote_volume,ignore\n")
    for r in rows:
        buf.write(",".join(str(x) for x in r) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.with_suffix(".csv").name, buf.getvalue())
    return path


@pytest.fixture
def make_zip(tmp_path):
    def _make(name: str, rows: list[list], header: bool = False) -> Path:
        return write_zip(tmp_path / name, rows, header)
    return _make
