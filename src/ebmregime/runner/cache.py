"""Fit cache: one ``.npz`` per job key, written atomically (PLAN §12, D-26, M2_SPEC §5).

Layout under the cache root (default ``data/cache/fits``; overridden by the ``root`` argument or the
environment variable ``EBMREGIME_FIT_CACHE``)::

    <root>/<key[:2]>/<key>.npz                  finished fit: arrays + ``meta`` (1-element str array, JSON)
    <root>/<key[:2]>/<key>.npz.tmp-<pid>-<hex>  in-progress write (never read; removed on startup if stale)
    <root>/progress.jsonl                       one JSON line per job finished by ``run.run_jobs``

Atomicity. ``write_atomic`` writes the whole archive to a temp file in the *same directory*, flushes
and fsyncs it, then ``os.replace``-s it onto the final name. ``os.replace`` is an atomic rename on
NTFS and POSIX, so a reader (or the runner's skip check) sees either no file or a complete one,
whatever happens to the writer: a killed process leaves at most a ``.tmp-`` file, never a partial
``<key>.npz``. Nothing is pickled: arrays must not be of object dtype and ``read`` loads with
``allow_pickle=False``.

Duplicate writes are harmless. Two writers of the same key (for example a worker of a killed run
that is still finishing while a restarted run recomputes the same job) each write their own temp
file; the last ``os.replace`` wins and both payloads are complete fits of the same job (identical
arrays; only provenance fields such as time, pid and host in ``meta`` differ). On Windows,
``os.replace`` fails with ``PermissionError`` while the destination is open by another process; the
writer retries for a few seconds and, if the destination then exists, discards its own temp file.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np

from ..config import repo_path

DEFAULT_ROOT = "data/cache/fits"
ENV_VAR = "EBMREGIME_FIT_CACHE"
PROGRESS_NAME = "progress.jsonl"

_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_TMP_RE = re.compile(r"^(?P<final>[0-9a-f]{64}\.npz)\.tmp-(?P<pid>\d+)-[0-9a-f]{32}$")


def cache_root(root: str | os.PathLike | None = None) -> Path:
    """``root`` if given, else ``$EBMREGIME_FIT_CACHE``, else ``data/cache/fits`` (repo-relative)."""
    if root is None:
        root = os.environ.get(ENV_VAR) or DEFAULT_ROOT
    return repo_path(root)


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise ValueError(f"not a cache key (64 lowercase hex chars): {key!r}")
    return key


def path_for(key: str, root=None) -> Path:
    _check_key(key)
    return cache_root(root) / key[:2] / f"{key}.npz"


def progress_path(root=None) -> Path:
    return cache_root(root) / PROGRESS_NAME


def exists(key: str, root=None) -> bool:
    return path_for(key, root).is_file()


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"meta value of type {type(o).__name__} is not JSON serialisable")


def _replace(tmp: Path, path: Path, tries: int = 50, delay: float = 0.1) -> bool:
    """``os.replace`` with retries for Windows sharing violations. Returns False if the destination
    stayed locked but exists (another writer finished the same key); the caller drops ``tmp``."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            if i == tries - 1:
                if path.is_file():
                    return False
                raise
            time.sleep(delay)
    return False  # pragma: no cover


def _fsync_dir(d: Path) -> None:
    if os.name == "nt":  # directories cannot be opened for fsync on Windows; NTFS journals the rename
        return
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: str | os.PathLike, arrays: dict, meta: dict) -> bool:
    """Write ``arrays`` plus ``meta`` (as JSON in a 1-element str array ``meta``) to ``path``.

    Temp file ``<path>.tmp-<pid>-<uuid>`` in the same directory, ``np.savez`` (uncompressed),
    flush + fsync, ``os.replace``. On any error the temp file is removed and the error re-raised.
    Returns True if this call placed the file, False if an existing copy was kept (see module doc).
    """
    path = Path(path)
    if "meta" in arrays:
        raise ValueError("'meta' is reserved for the metadata entry")
    payload = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"array {k!r} has object dtype (would need pickle)")
        payload[k] = a
    payload["meta"] = np.array([json.dumps(meta, sort_keys=True, default=_json_default)])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        # an open file object, so np.savez does not append ".npz" to the temp name
        with open(tmp, "xb") as fh:
            np.savez(fh, **payload)
            fh.flush()
            os.fsync(fh.fileno())
        placed = _replace(tmp, path)
        if not placed:
            tmp.unlink(missing_ok=True)
        else:
            _fsync_dir(path.parent)
        return placed
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write(key: str, arrays: dict, meta: dict, root=None) -> Path:
    p = path_for(key, root)
    write_atomic(p, arrays, meta)
    return p


def read_path(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"][0]))
    return arrays, meta


def read(key: str, root=None) -> tuple[dict[str, np.ndarray], dict]:
    """``(arrays, meta)`` of a finished fit. The file is closed on return."""
    return read_path(path_for(key, root))


def finished_keys(root=None) -> set[str]:
    """Keys with a final ``<key>.npz`` under the root (one directory scan, no file opens)."""
    base = cache_root(root)
    out: set[str] = set()
    if not base.is_dir():
        return out
    with os.scandir(base) as shards:
        for shard in shards:
            if not (shard.is_dir() and len(shard.name) == 2):
                continue
            with os.scandir(shard.path) as files:
                for f in files:
                    name = f.name
                    if name.endswith(".npz") and _KEY_RE.match(name[:-4]) and name.startswith(shard.name):
                        out.add(name[:-4])
    return out


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return False  # our own temp files cannot be in flight while we clean up
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover - psutil is a declared dependency
        return True  # unknown: be conservative and keep the file


def tmp_files(root=None) -> list[Path]:
    base = cache_root(root)
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*/*.npz.tmp-*") if _TMP_RE.match(p.name))


def clean_stale_tmp(root=None) -> dict[str, int]:
    """Remove temp files left by dead writers. Temp files whose writer pid is still alive (e.g. an
    orphaned worker of a killed run that is finishing its fit) are kept: that writer will rename or
    delete its own file. Files that cannot be deleted (open elsewhere on Windows) are kept too.
    Returns counts ``removed`` and ``kept``."""
    removed = kept = 0
    for p in tmp_files(root):
        m = _TMP_RE.match(p.name)
        if _pid_alive(int(m["pid"])):
            kept += 1
            continue
        try:
            p.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            kept += 1
    return {"removed": removed, "kept": kept}


__all__ = [
    "DEFAULT_ROOT",
    "ENV_VAR",
    "cache_root",
    "clean_stale_tmp",
    "exists",
    "finished_keys",
    "path_for",
    "progress_path",
    "read",
    "read_path",
    "tmp_files",
    "write",
    "write_atomic",
]
