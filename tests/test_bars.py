import numpy as np
import pandas as pd

from conftest import T0, kline_rows
from ebmregime.data.audit import runs, summarize, unit_switch
from ebmregime.data.bars import aggregate, apply_archive_flags, crosscheck, to_grid, volume_suspect
from ebmregime.data.parse import clean_klines

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]


def frame(rows, interval="15m"):
    return clean_klines(pd.DataFrame([r[:-1] for r in rows], columns=COLS), interval)


def test_grid_marks_missing_bars_and_runs():
    g = to_grid(frame(kline_rows(T0, 12, skip={4, 5, 9})), "15m")
    assert len(g) == 12 and int((~g["present"]).sum()) == 3
    r = runs(~g["present"])
    assert r["n_bars"].tolist() == [2, 1] and r["start"].iloc[0] == g.index[4]
    s = summarize(g)
    assert (s["missing"], s["missing_runs"], s["invalid"]) == (3, 2, 3)


def test_aggregate_matches_hand_computed_ohlcv():
    rows = kline_rows(T0, 8)  # two full hours
    bars = aggregate(to_grid(frame(rows), "15m"))
    h0 = rows[:4]
    assert len(bars) == 2 and bars["valid"].all()
    assert bars["open"].iloc[0] == h0[0][1] and bars["close"].iloc[0] == h0[-1][4]
    assert bars["high"].iloc[0] == max(r[2] for r in h0) and bars["low"].iloc[0] == min(r[3] for r in h0)
    assert np.isclose(bars["volume"].iloc[0], sum(r[5] for r in h0))
    assert bars["n_trades"].iloc[0] == sum(r[8] for r in h0)


def test_hour_with_a_missing_or_flagged_subbar_is_invalid_and_blanked():
    rows = kline_rows(T0, 12, skip={3})          # hour 0 loses its last 15m bar
    p = rows[6][1]                               # hour 1: one placeholder sub-bar (index 7 → row 6)
    rows[6][2:5] = [p, p, p]; rows[6][5] = 0.0; rows[6][8] = 0
    bars = aggregate(to_grid(frame(rows), "15m"))
    assert bars["valid"].tolist() == [False, False, True]
    assert bars["n_sub_valid"].tolist() == [3, 3, 4]
    assert bars[["open", "close"]].iloc[0].isna().all()   # never a wrong close from a partial hour


def test_crosscheck_against_archive_1h():
    rows15 = kline_rows(T0, 8)
    bars = aggregate(to_grid(frame(rows15), "15m"))
    h = [[T0 + i * 3_600_000, rows15[4 * i][1], max(r[2] for r in rows15[4 * i:4 * i + 4]),
          min(r[3] for r in rows15[4 * i:4 * i + 4]), rows15[4 * i + 3][4],
          sum(r[5] for r in rows15[4 * i:4 * i + 4]), T0 + (i + 1) * 3_600_000 - 1,
          sum(r[7] for r in rows15[4 * i:4 * i + 4]), sum(r[8] for r in rows15[4 * i:4 * i + 4]),
          16.0, sum(r[10] for r in rows15[4 * i:4 * i + 4]), 0] for i in range(2)]
    arch = to_grid(frame(h, "1h"), "1h")
    assert crosscheck(bars, arch)["mismatches"].sum() == 0
    assert not volume_suspect(bars, arch).any()
    arch.loc[arch.index[1], "close"] += 0.5
    arch.loc[arch.index[0], "volume"] += 7.0
    xc = crosscheck(bars, arch).set_index("column")
    assert xc.loc["close", "mismatches"] == 1
    assert volume_suspect(bars, arch).tolist() == [True, False]
    flagged = apply_archive_flags(bars, arch)
    assert flagged["price_conflict"].tolist() == [False, True]
    assert flagged["valid"].tolist() == [True, False] and np.isnan(flagged["close"].iloc[1])


def test_unit_switch_detects_first_microsecond_month():
    fu = pd.DataFrame({"file": ["A-15m-2024-11.zip", "A-15m-2025-01.zip", "A-15m-2024-12.zip"],
                       "unit": ["ms", "us", "ms"]})
    assert unit_switch(fu) == "2025-01"
