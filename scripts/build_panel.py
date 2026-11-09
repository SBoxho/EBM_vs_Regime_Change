"""Assemble the per-symbol modelling panel: features, targets and causal regimes on the 1h grid.

Writes, per symbol (under data/interim, never committed):
    panel/<SYM>.parquet      16 features + features_ok, y_{dir,vol,shuf}_h{8,24}, regime states
                             (r1_vol_2y [primary], r1_vol, r2_trend, r3_liq), valid
    panel/<SYM>.json         shuffle offset, primary R1, manifest hash, row counts

Usage:
    python scripts/build_panel.py [--symbols BTCUSDT ...]
Requires scripts/build_bars.py to have run.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ebmregime.config import load_config, repo_path
from ebmregime.data import store
from ebmregime.data.download import manifest_hash
from ebmregime.features.build import FEATURE_NAMES, build_features
from ebmregime.regimes import daily_inputs, daily_regimes, primary_r1, to_hourly
from ebmregime.series import hourly_inputs
from ebmregime.targets.build import build_targets

STATE_COLS = ["r1_vol_2y", "r1_vol", "r2_trend", "r3_liq"]


def build_symbol(sym: str, dcfg: dict, fcfg: dict, pcfg: dict) -> tuple[pd.DataFrame, dict]:
    h = hourly_inputs(store.load("grid_15m", sym), store.load("bars_1h", sym))
    partner_sym = fcfg["cross_partner"][sym]
    partner = hourly_inputs(store.load("grid_15m", partner_sym), store.load("bars_1h", partner_sym))
    feats = build_features(h, partner, fcfg)
    targs = build_targets(h, sym, pcfg, dcfg)
    states = to_hourly(daily_regimes(daily_inputs(h), pcfg["regimes"]), h.index)[STATE_COLS]
    panel = pd.concat([feats, targs, states, h[["valid"]]], axis=1)
    start = pd.Timestamp(dcfg["symbols"][sym]["analysis_start"], tz="UTC")
    rng = panel.loc[start:]
    info = {
        "symbol": sym,
        "partner": partner_sym,
        "primary_r1": primary_r1(pcfg["regimes"]),
        "shuffle_offset_hours": targs.attrs["shuffle_offset_hours"],
        "analysis_start": str(start),
        "rows_from_analysis_start": len(rng),
        "features_ok_share": round(float(rng["features_ok"].mean()), 4),
        "usable_share_h8": round(float((rng["features_ok"] & rng["y_dir_h8"].notna()
                                        & rng["y_vol_h8"].notna()).mean()), 4),
        "feature_names": FEATURE_NAMES,
    }
    return panel, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="+")
    args = ap.parse_args(argv)
    dcfg, fcfg, pcfg = load_config("data"), load_config("features"), load_config("prereg")
    out = repo_path(dcfg["paths"]["interim"]) / "panel"
    out.mkdir(parents=True, exist_ok=True)
    mhash = manifest_hash(repo_path(dcfg["paths"]["manifest"]))
    for sym in args.symbols or list(dcfg["symbols"]):
        panel, info = build_symbol(sym, dcfg, fcfg, pcfg)
        info["manifest_hash"] = mhash
        panel.to_parquet(out / f"{sym}.parquet")
        (out / f"{sym}.json").write_text(json.dumps(info, indent=1))
        print(f"{sym}: {len(panel)} rows; from {info['analysis_start'][:10]}: features_ok "
              f"{info['features_ok_share']:.4f}, usable (features + h8 labels) {info['usable_share_h8']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
