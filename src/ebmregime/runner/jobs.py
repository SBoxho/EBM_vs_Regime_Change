"""Job specs, grid expansion and the content-addressed cache key (PLAN §12, D-26, M2_SPEC §5).

One job is one fit: model x arm x symbol x window (``tau``) x target x horizon x replicate x data.

Cache key (D-26 as specified in M2_SPEC §5)::

    key = sha256(canonical_json({
        "prereg": prereg_hash(),             # sha256(LF(configs/prereg.yaml) + LF(configs/features.yaml))
        "code":   fit_code_hash(),           # the files that determine a fit, plus interpret.__version__
        "data":   panel.data_hash(symbol),   # data-manifest hash + panel parquet bytes
        "job":    dataclasses.asdict(job),
    })).hexdigest()

``canonical_json`` is ``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)``. LF-normalisation replaces CRLF by LF, so a Windows checkout with
``core.autocrlf`` gives the same keys as a Unix one.

``fit_code_hash`` covers only the code a fit depends on (``FIT_CODE_FILES``), not all of
``src/``: changing metric or report code must not invalidate thousands of fits; the metric stage
has its own key. Files are hashed in sorted order of their path relative to ``src/ebmregime``
(POSIX separators); each contributes ``"<relpath>\\0<n_bytes>\\0"`` followed by its LF-normalised
bytes; the string ``"interpret\\0<version>"`` is appended last. A listed file that does not exist
is skipped (so the hash changes when it is created).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import load_config, repo_path
from ..splits import replicate_plan, retrain_schedule

TARGETS = ("dir", "vol", "shuf")
PROTOCOLS = ("seed", "half")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # src/ebmregime

# Paths relative to src/ebmregime (glob patterns allowed). See module docstring.
FIT_CODE_FILES = (
    "config.py",
    "series.py",
    "panel.py",
    "splits/*.py",
    "models/*.py",
    "stability/surrogate.py",
    "runner/jobs.py",
    "runner/execute.py",
)
PREREG_FILES = ("configs/prereg.yaml", "configs/features.yaml")

_DATA_RE = re.compile(r"^(real|surrogate:(0|[1-9][0-9]*))$")
_TAU_RE = re.compile(r"^\d{4}-\d{2}-01$")


@dataclass(frozen=True, kw_only=True)
class Job:
    """One fit. Field order is the spec's; all fields enter the cache key."""

    model: str = "ebm"
    arm: str = "primary"
    symbol: str
    tau: str                      # ISO date of the retrain cut-off, e.g. "2019-12-01"
    target: str                   # dir | vol | shuf
    horizon: int = 8
    protocol: str                 # seed | half
    r: int
    seed: int                     # model random_state, from splits.replicate_plan
    sample_seed: int | None       # half-sample draw; None for protocol "seed"
    data: str = "real"            # "real" or "surrogate:<i>"

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if (self.sample_seed is None) != (self.protocol == "seed"):
            raise ValueError("sample_seed must be None for protocol 'seed' and an int for 'half'")
        if not _DATA_RE.match(self.data):
            raise ValueError(f"data must be 'real' or 'surrogate:<i>', got {self.data!r}")
        if not _TAU_RE.match(self.tau):
            raise ValueError(f"tau must be an ISO month-start date 'YYYY-MM-01', got {self.tau!r}")
        for name in ("horizon", "r", "seed") + (("sample_seed",) if self.sample_seed is not None else ()):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"{name} must be a plain int, got {type(v).__name__}")

    @property
    def surrogate_index(self) -> int | None:
        return None if self.data == "real" else int(self.data.split(":", 1)[1])

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


# ---------------------------------------------------------------- hashes and key

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def prereg_hash(paths: Iterable[str | Path] | None = None) -> str:
    """sha256 of the LF-normalised bytes of ``configs/prereg.yaml`` followed by ``configs/features.yaml``."""
    if paths is None:
        return _prereg_hash_default()
    h = hashlib.sha256()
    for p in paths:
        h.update(_lf_bytes(repo_path(p)))
    return h.hexdigest()


@lru_cache(maxsize=1)
def _prereg_hash_default() -> str:
    return prereg_hash(PREREG_FILES)


def fit_code_files(root: str | Path | None = None) -> list[str]:
    """Existing files matched by ``FIT_CODE_FILES`` under ``root`` (default src/ebmregime), sorted."""
    base = Path(root) if root is not None else PACKAGE_ROOT
    found = set()
    for pat in FIT_CODE_FILES:
        for p in base.glob(pat):
            if p.is_file():
                found.add(p.relative_to(base).as_posix())
    return sorted(found)


def _interpret_version() -> str:
    import interpret

    return str(interpret.__version__)


def fit_code_hash(root: str | Path | None = None, interpret_version: str | None = None) -> str:
    """Hash of the code that determines a fit (see module docstring). Cached for the defaults."""
    if root is None and interpret_version is None:
        return _fit_code_hash_default()
    base = Path(root) if root is not None else PACKAGE_ROOT
    version = interpret_version if interpret_version is not None else _interpret_version()
    h = hashlib.sha256()
    for rel in fit_code_files(base):
        data = _lf_bytes(base / rel)
        h.update(f"{rel}\0{len(data)}\0".encode("utf-8"))
        h.update(data)
    h.update(f"interpret\0{version}".encode("utf-8"))
    return h.hexdigest()


@lru_cache(maxsize=1)
def _fit_code_hash_default() -> str:
    return fit_code_hash(PACKAGE_ROOT, _interpret_version())


def _data_hash(symbol: str) -> str:
    from .. import panel

    return panel.data_hash(symbol)


def job_key(job: Job, *, prereg: str | None = None, code: str | None = None,
            data: str | Callable[[str], str] | None = None) -> str:
    """Content-addressed key of ``job`` (64 hex chars). The keyword overrides exist for tests and
    for callers that have already computed the hashes; ``data`` may be a str or ``symbol -> str``."""
    if data is None:
        d = _data_hash(job.symbol)
    elif callable(data):
        d = data(job.symbol)
    else:
        d = data
    doc = {
        "prereg": prereg if prereg is not None else prereg_hash(),
        "code": code if code is not None else fit_code_hash(),
        "data": d,
        "job": job.as_dict(),
    }
    return hashlib.sha256(canonical_json(doc).encode("ascii")).hexdigest()


# ---------------------------------------------------------------- grid

def load_grid(path: str | Path) -> dict:
    return load_config(repo_path(path))


def grid_taus(symbol: str, prereg: dict, taus="all") -> list[str]:
    """ISO dates of the retrain cut-offs: every tau of ``prereg.windows`` for ``symbol``, or a list."""
    if taus == "all":
        w = prereg["windows"]
        sched = retrain_schedule(w["first_tau"][symbol], w["last_tau"], int(w["step_months"]))
        return [t.strftime("%Y-%m-%d") for t in sched]
    out = []
    for t in taus:
        ts = pd.Timestamp(t)
        out.append(ts.strftime("%Y-%m-%d"))
    return out


def expand_grid(grid_cfg: dict, prereg: dict | None = None) -> list[Job]:
    """All jobs of a grid config (``configs/grid_*.yaml``). Seeds come only from
    ``splits.replicate_plan`` with ``prereg.replicates``. Order: symbol, data entry, target, tau,
    replicate (seed fits then half-samples). If the config has ``expected_jobs``, the count is checked."""
    prereg = prereg if prereg is not None else load_config("prereg")
    model = grid_cfg.get("model", "ebm")
    arm = grid_cfg.get("arm", "primary")
    horizon = int(grid_cfg.get("horizon", prereg["targets"]["horizons"]["primary"]))
    jobs: list[Job] = []
    for symbol in grid_cfg["symbols"]:
        taus = grid_taus(symbol, prereg, grid_cfg.get("taus", "all"))
        for entry in grid_cfg["data"]:
            for target in entry["targets"]:
                for tau in taus:
                    for rep in replicate_plan(symbol, tau, target, prereg["replicates"]):
                        jobs.append(Job(model=model, arm=arm, symbol=symbol, tau=tau, target=target,
                                        horizon=horizon, protocol=rep["protocol"], r=int(rep["r"]),
                                        seed=int(rep["seed"]),
                                        sample_seed=(None if rep["sample_seed"] is None
                                                     else int(rep["sample_seed"])),
                                        data=str(entry["source"])))
    if len(set(jobs)) != len(jobs):
        raise AssertionError("expand_grid produced duplicate jobs")
    exp = grid_cfg.get("expected_jobs")
    if exp is not None and int(exp) != len(jobs):
        raise AssertionError(f"grid expands to {len(jobs)} jobs, config expects {exp}")
    return jobs


def dispatch_order(jobs: Iterable[Job]) -> list[Job]:
    """Slowest first (PLAN §11.1): vol (regressor) before dir/shuf, full-data seed fits before
    half-samples; otherwise the input order is kept (stable sort)."""
    return sorted(jobs, key=lambda j: (j.target != "vol", j.protocol != "seed"))


__all__ = [
    "FIT_CODE_FILES",
    "Job",
    "PREREG_FILES",
    "canonical_json",
    "dispatch_order",
    "expand_grid",
    "fit_code_files",
    "fit_code_hash",
    "grid_taus",
    "job_key",
    "load_grid",
    "prereg_hash",
]
