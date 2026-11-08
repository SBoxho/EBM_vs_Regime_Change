"""Read the processed files that ``scripts/build_bars.py`` writes under data/interim."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import load_config, repo_path

KINDS = {"grid_15m", "bars_1h", "archive_1h"}


def interim_dir(config: str = "data") -> Path:
    return repo_path(load_config(config)["paths"]["interim"])


def load(kind: str, symbol: str, config: str = "data") -> pd.DataFrame:
    """Load ``grid_15m``, ``bars_1h`` or ``archive_1h`` for one symbol."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    return pd.read_parquet(interim_dir(config) / kind / f"{symbol}.parquet")


def file_units(symbol: str, interval: str, config: str = "data") -> pd.DataFrame:
    return pd.read_csv(interim_dir(config) / "file_units" / f"{symbol}_{interval}.csv")


def crosscheck_table(symbol: str, config: str = "data") -> pd.DataFrame:
    return pd.read_csv(interim_dir(config) / "crosscheck" / f"{symbol}.csv")
