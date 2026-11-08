"""Configuration loading. Paths in configs are relative to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by name (``"data"`` -> ``configs/data.yaml``) or by path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = REPO_ROOT / "configs" / f"{p}.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(rel: str | Path) -> Path:
    """Resolve a repository-relative path."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
