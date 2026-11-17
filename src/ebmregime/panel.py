"""Shared access to the M1 panel for M2 (lead-owned primitives, like ``series.py`` at M1).

The panel ``data/interim/panel/<SYM>.parquet`` (built by ``scripts/build_panel.py``) has a UTC
hourly index ``ts`` (bar open time), the 16 features, ``features_ok``, the targets
``y_{dir,vol,shuf}_h{8,24}``, the regime states and ``valid``.

Row roles for one window (PLAN §7, D-13, D-45, D-46):

* **usable**: ``features_ok`` and the target is not NaN;
* **train rows** of the window at ``tau``: usable and ``splits.train_mask`` (calendar window plus
  the label purge ``ts + (h+1) h <= tau``);
* **fit rows**: train rows minus those that would be 0 in every bag, i.e. the rows before the
  validation tail whose labels overlap it (D-45). These are the rows passed to ``fit`` for a
  full-window fit, and the rows whose union forms a pair's X_ref (D-50);
* **test rows**: usable rows in ``[tau, tau + 1 month)``.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, repo_path
from .splits.bags import tail_start
from .splits.windows import label_time, test_mask, train_mask

TARGETS = ("dir", "vol", "shuf")


@lru_cache(maxsize=1)
def _features_cfg() -> dict:
    return load_config("features")


def feature_names() -> list[str]:
    """The 16 features in the frozen order of ``configs/features.yaml``."""
    return [f["name"] for f in _features_cfg()["features"]]


def nominal_features() -> list[str]:
    return [f["name"] for f in _features_cfg()["features"] if f.get("nominal", False)]


def feature_types() -> list[str]:
    """interpret ``feature_types``: ``"nominal"`` for hour and dow, else ``"continuous"``."""
    nom = set(nominal_features())
    return ["nominal" if n in nom else "continuous" for n in feature_names()]


def feature_families() -> dict[str, str]:
    return {f["name"]: f["family"] for f in _features_cfg()["features"]}


def panel_path(symbol: str) -> Path:
    return repo_path(f"data/interim/panel/{symbol}.parquet")


@lru_cache(maxsize=8)
def _load(symbol: str) -> pd.DataFrame:
    p = panel_path(symbol)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/download_data.py, build_bars.py, build_panel.py")
    return pd.read_parquet(p)


def load_panel(symbol: str) -> pd.DataFrame:
    """The panel of ``symbol`` (cached per process; callers must not mutate it)."""
    return _load(symbol)


def panel_info(symbol: str) -> dict:
    return json.loads(repo_path(f"data/interim/panel/{symbol}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def data_hash(symbol: str) -> str:
    """sha256 over the data-manifest hash and the bytes of the symbol's panel file (D-26)."""
    h = hashlib.sha256()
    h.update(panel_info(symbol)["manifest_hash"].encode())
    h.update(panel_path(symbol).read_bytes())
    return h.hexdigest()


def target_column(target: str, horizon: int) -> str:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return f"y_{target}_h{int(horizon)}"


def usable_mask(panel: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    return panel["features_ok"].to_numpy(dtype=bool) & panel[target_column(target, horizon)].notna().to_numpy()


def train_rows(panel: pd.DataFrame, tau, target: str, horizon: int, train_months: int) -> np.ndarray:
    """Boolean mask over ``panel`` rows: usable training rows of the window at ``tau``."""
    return train_mask(panel.index, tau, train_months, horizon, usable_mask(panel, target, horizon))


def purged_before_tail(ts, horizon: int, tail_frac: float) -> np.ndarray:
    """Boolean mask over the train rows ``ts``: rows that are 0 in every bag (D-45).

    Same rule as ``splits.bags``: rows before the validation-tail start whose label time is after it.
    """
    idx = pd.DatetimeIndex(ts)
    vs = tail_start(idx, tail_frac)
    return np.asarray(idx < vs) & np.asarray(label_time(idx, horizon) > vs)


def fit_rows(panel: pd.DataFrame, tau, target: str, horizon: int, train_months: int,
             tail_frac: float) -> np.ndarray:
    """Boolean mask over ``panel`` rows: the rows a full-window fit at ``tau`` passes to ``fit``."""
    tr = train_rows(panel, tau, target, horizon, train_months)
    pos = np.flatnonzero(tr)
    keep = ~purged_before_tail(panel.index[pos], horizon, tail_frac)
    out = np.zeros(len(panel), dtype=bool)
    out[pos[keep]] = True
    return out


def test_rows(panel: pd.DataFrame, tau, target: str, horizon: int) -> np.ndarray:
    return test_mask(panel.index, tau, usable_mask(panel, target, horizon))


test_rows.__test__ = False  # not a pytest test despite the name
