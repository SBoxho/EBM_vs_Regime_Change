"""Data-quality summaries for the audit notebook and the M0 acceptance checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .listing import month_of


def runs(mask: pd.Series) -> pd.DataFrame:
    """Runs of consecutive ``True`` values in a boolean series indexed by time.

    Returns ``start`` (first bar of the run) and ``n_bars``, longest first.
    """
    m = mask.to_numpy(dtype=bool)
    if not m.any():
        return pd.DataFrame({"start": pd.Series(dtype="datetime64[ns, UTC]"), "n_bars": pd.Series(dtype=int)})
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = pd.DataFrame({"start": mask.index[starts], "n_bars": ends - starts})
    return out.sort_values(["n_bars", "start"], ascending=[False, True]).reset_index(drop=True)


def summarize(grid: pd.DataFrame) -> dict:
    """Headline counts for a grid produced by :func:`ebmregime.data.bars.to_grid`."""
    missing = ~grid["present"]
    return {
        "first_bar": grid.index[grid["present"]][0],
        "last_bar": grid.index[grid["present"]][-1],
        "expected": len(grid),
        "present": int(grid["present"].sum()),
        "missing": int(missing.sum()),
        "missing_runs": len(runs(missing)),
        "placeholder": int(grid["placeholder"].sum()),
        "truncated": int(grid["truncated"].sum()),
        "off_grid": int(grid["off_grid"].sum()),
        "invalid": int((~grid["valid"]).sum()),
    }


def by_year(grid: pd.DataFrame) -> pd.DataFrame:
    """Missing and invalid bars per calendar year."""
    y = grid.index.year
    return pd.DataFrame({
        "missing": (~grid["present"]).groupby(y).sum(),
        "invalid": (~grid["valid"]).groupby(y).sum(),
        "expected": grid["present"].groupby(y).size(),
    }).rename_axis("year")


def unit_switch(file_units: pd.DataFrame) -> str | None:
    """First month whose file uses µs timestamps, from the per-file table of ``load_symbol``.

    Raises if any ms file follows a µs file: the switch must happen exactly once.
    """
    fu = file_units.sort_values("file").reset_index(drop=True)
    is_us = (fu["unit"] == "us").to_numpy()
    if not is_us.any():
        return None
    first = int(np.argmax(is_us))
    if not is_us[first:].all():
        raise AssertionError("an ms file appears after the first µs file")
    return month_of(fu.loc[first, "file"])
