"""Run jobs on joblib loky workers, skipping finished keys (PLAN §12, D-25, D-26, M2_SPEC §5).

Resume and kill-safety:

* The **worker** writes its fit itself with :func:`cache.write_atomic` (temp file in the same
  directory, fsync, ``os.replace``) and returns only a small summary. Killing the parent (or the
  whole process tree, ``taskkill /T /F`` on Windows) therefore loses only the fits in flight, at most
  ``n_workers``; it can leave ``*.npz.tmp-<pid>-<hex>`` files but never a partial ``<key>.npz``.
* On start, ``run_jobs`` removes temp files whose writer pid is dead, then lists the finished keys
  with one directory scan and dispatches only the missing ones.
* A worker that outlives a killed parent (not possible with ``/T``, which kills the tree; loky
  workers also exit once their parent's pipe breaks) can at worst finish its current fit and
  rename it into place. That write is complete and equivalent to the restarted run's own write of
  the same key, so the late duplicate is harmless (last ``os.replace`` wins); its temp file is not
  deleted by the restart because its pid is alive.
* ``progress.jsonl`` is a log, not the source of truth: the npz files are. A kill can truncate its
  last line; readers skip unparsable lines and the next run starts on a fresh line.

Workers are single-threaded: ``OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`` in
the parent's environment while the pool is created, and joblib's ``inner_max_num_threads=1``
(which passes the same limits explicitly to the loky workers).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from . import cache
from .jobs import Job, dispatch_order, job_key

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextlib.contextmanager
def single_thread_env():
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def execute_one(job_fn: Callable[[Job], tuple[dict, dict]], job: Job, key: str, root: str) -> dict:
    """Worker side: run ``job_fn(job)``, write ``<root>/<key[:2]>/<key>.npz`` atomically, return a
    small summary. Exceptions are caught and returned as ``status="error"`` so that one bad job
    does not stop the pool (the parent stops after ``max_errors``)."""
    t0 = time.perf_counter()
    pid = os.getpid()
    try:
        arrays, meta = job_fn(job)
        job_seconds = time.perf_counter() - t0
        meta = dict(meta)
        meta.setdefault("job", job.as_dict())
        meta["key"] = key
        meta["worker_pid"] = pid
        meta["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
        meta["job_seconds"] = round(job_seconds, 4)
        placed = cache.write_atomic(cache.path_for(key, root), arrays, meta)
        fit_seconds = (meta.get("fit_info") or {}).get("fit_seconds", job_seconds)
        return {"key": key, "status": "ok", "fit_seconds": float(fit_seconds),
                "job_seconds": round(job_seconds, 4), "worker_pid": pid, "placed": bool(placed)}
    except Exception as e:  # noqa: BLE001 - reported to the parent and logged
        return {"key": key, "status": "error", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=20), "worker_pid": pid,
                "job_seconds": round(time.perf_counter() - t0, 4)}


def _fmt_s(s: float) -> str:
    if s != s or s == float("inf"):
        return "?"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "ab")
    if fh.tell() > 0:  # a killed run may have left a partial last line
        with open(path, "rb") as r:
            r.seek(-1, os.SEEK_END)
            if r.read(1) != b"\n":
                fh.write(b"\n")
    return fh


def read_progress(path) -> list[dict]:
    """Parsed lines of a progress log; partial or corrupt lines are skipped."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def plan_jobs(jobs: Iterable[Job], root=None, key_fn: Callable[[Job], str] | None = None
              ) -> tuple[list[tuple[str, Job]], set[str]]:
    """``(keyed, finished)``: ``(key, job)`` for every job in dispatch order (slowest first), and the
    keys already in the cache. Raises if two jobs share a key."""
    key_fn = key_fn or job_key
    keyed = [(key_fn(j), j) for j in dispatch_order(jobs)]
    if len({k for k, _ in keyed}) != len(keyed):
        raise AssertionError("duplicate cache keys among jobs")
    return keyed, cache.finished_keys(root)


def run_jobs(jobs: Iterable[Job], n_workers: int = 16, log_path=None, *,
             job_fn: Callable[[Job], tuple[dict, dict]] | None = None, root=None,
             key_fn: Callable[[Job], str] | None = None, limit: int | None = None,
             progress_every: int = 50, max_errors: int = 20, verbose: bool = True,
             stream=None) -> dict:
    """Run every job whose key is not cached yet; return counts.

    ``job_fn(job) -> (arrays, meta)`` defaults to :func:`execute.run_fit_job`; it runs in the
    workers and must be picklable (module-level function). ``root`` is the cache root
    (:func:`cache.cache_root`); ``log_path`` defaults to ``<root>/progress.jsonl``. ``limit`` caps
    the number of jobs run (after skipping cached ones, in dispatch order). ``key_fn`` defaults to
    :func:`jobs.job_key` (it runs in the parent only). The run stops with ``RuntimeError`` once more
    than ``max_errors`` jobs failed; failed jobs are logged with ``status="error"`` and not cached.
    """
    if job_fn is None:
        from .execute import run_fit_job as job_fn
    stream = stream or sys.stdout
    root_p = cache.cache_root(root)
    root_p.mkdir(parents=True, exist_ok=True)
    log_p = Path(log_path) if log_path is not None else cache.progress_path(root_p)

    def say(msg: str):
        if verbose:
            print(f"[runner] {msg}", file=stream, flush=True)

    cleaned = cache.clean_stale_tmp(root_p)
    keyed, finished = plan_jobs(jobs, root_p, key_fn)
    todo = [(k, j) for k, j in keyed if k not in finished]
    n_cached = len(keyed) - len(todo)
    if limit is not None:
        todo = todo[: max(0, int(limit))]
    run_id = uuid.uuid4().hex[:12]
    say(f"{len(keyed)} jobs, {n_cached} cached, {len(todo)} to run on {n_workers} workers "
        f"(root {root_p}; stale tmp removed {cleaned['removed']}, kept {cleaned['kept']}; run {run_id})")
    summary = {"run_id": run_id, "n_jobs": len(keyed), "n_cached": n_cached, "n_todo": len(todo),
               "n_ok": 0, "n_errors": 0, "wall_seconds": 0.0, "stale_tmp": cleaned}
    if not todo:
        return summary

    import joblib

    jobs_by_key = dict(todo)
    t0 = time.perf_counter()
    n_done = 0
    fit_total = 0.0
    errors: list[dict] = []
    with single_thread_env(), joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        par = joblib.Parallel(n_jobs=n_workers, return_as="generator_unordered")
        gen = par(joblib.delayed(execute_one)(job_fn, j, k, str(root_p)) for k, j in todo)
        fh = _open_log(log_p)
        try:
            for res in gen:
                n_done += 1
                elapsed = time.perf_counter() - t0
                job = jobs_by_key[res["key"]]
                line = {"key": res["key"], "job": job.as_dict(), "status": res["status"],
                        "fit_seconds": res.get("fit_seconds"), "job_seconds": res.get("job_seconds"),
                        "worker_pid": res.get("worker_pid"),
                        "wall_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                        "elapsed_s": round(elapsed, 3), "run_id": run_id}
                if res["status"] == "ok":
                    summary["n_ok"] += 1
                    fit_total += float(res.get("fit_seconds") or 0.0)
                else:
                    line["error"] = res.get("error")
                    errors.append(res)
                    summary["n_errors"] += 1
                    say(f"ERROR {res['key'][:12]} {job}: {res.get('error')}")
                fh.write((json.dumps(line, sort_keys=True) + "\n").encode("utf-8"))
                fh.flush()
                if n_done % max(1, progress_every) == 0 or n_done == len(todo):
                    rate = n_done / elapsed if elapsed > 0 else float("nan")
                    eta = (len(todo) - n_done) / rate if rate and rate == rate else float("nan")
                    say(f"{n_done}/{len(todo)} done ({summary['n_errors']} errors), "
                        f"{rate * 60:.1f} fits/min, elapsed {_fmt_s(elapsed)}, ETA {_fmt_s(eta)}")
                if summary["n_errors"] > max_errors:
                    raise RuntimeError(f"more than {max_errors} jobs failed; first error:\n"
                                       f"{errors[0].get('traceback')}")
        finally:
            gen.close()  # aborts outstanding tasks if we leave early
            fh.close()
    summary["wall_seconds"] = round(time.perf_counter() - t0, 3)
    summary["fit_seconds_total"] = round(fit_total, 3)
    summary["errors"] = [{"key": e["key"], "error": e.get("error")} for e in errors]
    say(f"finished: {summary['n_ok']} ok, {summary['n_errors']} errors in {_fmt_s(summary['wall_seconds'])}")
    return summary


__all__ = ["THREAD_VARS", "execute_one", "plan_jobs", "read_progress", "run_jobs", "single_thread_env"]
