"""M0 acceptance checks against the real downloaded archive (PLAN §13, M0 "done when").

These tests are skipped unless the data has been prepared with:
    python scripts/download_data.py && python scripts/build_bars.py
The expected counts were measured independently during planning (2026-11-08) with a separate
parser, so they also check this package's parser.
"""

from __future__ import annotations

import pytest

from ebmregime.config import load_config, repo_path
from ebmregime.data import store
from ebmregime.data.audit import summarize, unit_switch
from ebmregime.data.download import verify_against_manifest

CFG = load_config("data")
INTERIM = repo_path(CFG["paths"]["interim"])
SYMBOLS = list(CFG["symbols"])

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (INTERIM / "build_info.json").exists(), reason="data not prepared"),
]

# Missing bars against a perfect grid between each symbol's first and last bar (full history).
EXPECTED_MISSING = {
    ("BTCUSDT", "archive_1h"): 127, ("ETHUSDT", "archive_1h"): 127,
    ("XRPUSDT", "archive_1h"): 87, ("SOLUSDT", "archive_1h"): 19,
    ("BTCUSDT", "grid_15m"): 562, ("SOLUSDT", "grid_15m"): 93,
}
FIRST_BAR = {"BTCUSDT": "2017-08-17 04:00", "XRPUSDT": "2018-05-04 08:00", "SOLUSDT": "2020-08-11 06:00"}


def test_raw_files_match_manifest():
    assert verify_against_manifest(repo_path(CFG["paths"]["manifest"]), repo_path(CFG["paths"]["raw"])) == []


@pytest.mark.parametrize("sym,kind", sorted(EXPECTED_MISSING))
def test_missing_bar_counts_reproduce_planning_audit(sym, kind):
    assert summarize(store.load(kind, sym))["missing"] == EXPECTED_MISSING[(sym, kind)]


@pytest.mark.parametrize("sym", sorted(FIRST_BAR))
def test_first_bar(sym):
    assert str(summarize(store.load("archive_1h", sym))["first_bar"])[:16] == FIRST_BAR[sym]


@pytest.mark.parametrize("sym", SYMBOLS)
@pytest.mark.parametrize("interval", ["15m", "1h"])
def test_microsecond_switch_is_2025_01(sym, interval):
    assert unit_switch(store.file_units(sym, interval)) == "2025-01"


@pytest.mark.parametrize("sym", SYMBOLS)
def test_archive_inconsistencies_are_rare_flagged_and_masked(sym):
    # D-40: in a few hours the archive's own 1h bar disagrees with the sum of its 15m bars.
    bars = store.load("bars_1h", sym)
    xc = store.crosscheck_table(sym).set_index("column")
    assert xc["compared"].min() > 40_000
    n_vol, n_px = int(bars["volume_suspect"].sum()), int(bars["price_conflict"].sum())
    assert n_vol <= 10 and n_px <= 2
    assert xc.loc[["open", "high", "low", "close"], "mismatches"].max() == n_px
    # Price conflicts only occur inside hours that also disagree on volume, and they are masked.
    assert not (bars["price_conflict"] & ~bars["volume_suspect"]).any()
    assert not (bars["price_conflict"] & bars["valid"]).any()


@pytest.mark.parametrize("sym", SYMBOLS)
def test_analysis_range_is_almost_complete(sym):
    bars = store.load("bars_1h", sym)
    rng = bars.loc[CFG["symbols"][sym]["analysis_start"]:]
    assert rng.index[-1].strftime("%Y-%m") == CFG["last_month"]
    assert (~rng["valid"]).mean() < 0.005
