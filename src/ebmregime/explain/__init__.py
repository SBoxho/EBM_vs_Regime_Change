"""Comparable shapes: reference sample, pooled cells, cell-averaged contributions (PLAN §6.1)."""

from .cells import CellGrid, cell_shapes, cell_shapes_many, explain_settings, make_cells, make_xref
from .contributions import ebm_contrib, eta

__all__ = [
    "CellGrid",
    "cell_shapes",
    "cell_shapes_many",
    "ebm_contrib",
    "eta",
    "explain_settings",
    "make_cells",
    "make_xref",
]
