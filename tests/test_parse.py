import pandas as pd
import pytest

from conftest import T0, kline_rows
from ebmregime.data.parse import ParseError, clean_klines, detect_unit, load_symbol, read_kline_zip


def test_ms_and_us_files_parse_to_identical_frames(make_zip):
    a, ua = read_kline_zip(make_zip("A-15m-2024-12.zip", kline_rows(T0, 8, unit="ms")))
    b, ub = read_kline_zip(make_zip("A-15m-2025-01.zip", kline_rows(T0, 8, unit="us")))
    assert (ua, ub) == ("ms", "us")
    pd.testing.assert_frame_equal(a, b)
    assert pd.to_datetime(a["open_time"].iloc[0], unit="ms", utc=True) == pd.Timestamp("2025-01-01", tz="UTC")


def test_header_line_is_skipped(make_zip):
    df, _ = read_kline_zip(make_zip("H.zip", kline_rows(T0, 3), header=True))
    assert len(df) == 3 and df["open_time"].iloc[0] == T0


def test_mixed_units_and_implausible_epochs_raise():
    with pytest.raises(ParseError):
        detect_unit([T0, T0 * 1000])
    with pytest.raises(ParseError):
        detect_unit([T0 * 1_000_000])  # nanoseconds are not a valid unit here


def test_flags_offgrid_truncated_placeholder():
    rows = kline_rows(T0, 4)
    rows[1][0] += 314_000              # +5m14s off-grid (1h bars were +28m14s after the 2018-02 outage)
    rows[1][6] += 314_000
    rows[2][6] = rows[2][0] + 60_000 - 1  # one-minute (truncated) bar
    p = rows[3][1]
    rows[3][2:5] = [p, p, p]; rows[3][5] = 0.0; rows[3][8] = 0  # O=H=L=C, no volume, no trades
    raw = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                      "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    out = clean_klines(raw.drop(columns="ignore"), "15m")
    assert out["off_grid"].tolist() == [False, True, False, False]
    assert out["truncated"].tolist() == [False, False, True, False]
    assert out["placeholder"].tolist() == [False, False, False, True]
    # The off-grid bar is snapped down to the 15m grid.
    assert (out["ts"].diff().dropna() == pd.Timedelta("15min")).all()


def test_exact_duplicates_dropped_but_collisions_raise():
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote"]
    rows = [r[:-1] for r in kline_rows(T0, 3)]
    dup = pd.DataFrame(rows + [rows[1]], columns=cols)
    assert len(clean_klines(dup, "15m")) == 3
    clash = [r.copy() for r in rows]
    clash[2][0] = clash[1][0] + 60_000  # snaps onto bar 1 with different content
    with pytest.raises(ParseError):
        clean_klines(pd.DataFrame(clash, columns=cols), "15m")


def test_load_symbol_reports_units_per_file(make_zip):
    p1 = make_zip("A-15m-2024-12.zip", kline_rows(T0 - 4 * 900_000, 4, unit="ms"))
    p2 = make_zip("A-15m-2025-01.zip", kline_rows(T0, 4, unit="us"))
    kl, units = load_symbol([p2, p1], "15m")
    assert units["unit"].tolist() == ["ms", "us"] and len(kl) == 8
    assert kl["ts"].is_monotonic_increasing
