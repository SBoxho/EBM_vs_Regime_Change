"""Kill-and-resume trial of the runner on a real grid (M2 go/no-go criterion 1; lead's verification).

Starts ``scripts/run_grid.py <grid>`` as a subprocess, hard-kills its whole process tree after
``--kill-after`` seconds (``taskkill /T /F`` on Windows, SIGKILL to the process group elsewhere), then
reports what the kill left behind:

* finished fits on disk before and after the kill (no fit may appear or vanish after the kill);
* workers that were mid-fit at the kill = fits whose work was lost (each worker runs one fit at a
  time, and a fit only reaches the cache when complete, via an atomic rename);
* stale temporary files and surviving worker processes;
* fits on disk that the parent had not logged (written by a worker just before the kill; kept).

The resume itself is a plain second ``run_grid.py`` call, run afterwards by the caller.
Output: a JSON summary on stdout and in ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebmregime.runner import cache  # noqa: E402
from ebmregime.runner.run import read_progress  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--kill-after", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/slice/kill_resume_trial.json")
    args = ap.parse_args()

    before = cache.finished_keys()
    t0 = time.time()
    log = open(ROOT / "results" / "slice" / "run_killed.log", "w", encoding="utf-8")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_grid.py"), args.grid,
                             "--workers", str(args.workers)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **kw)
    time.sleep(args.kill_after)

    parent = psutil.Process(proc.pid)
    tree = [parent] + parent.children(recursive=True)
    busy = []
    for p in tree[1:]:
        try:
            busy.append((p.pid, p.cpu_percent(interval=None)))
        except psutil.NoSuchProcess:
            pass
    time.sleep(1.0)
    busy = []
    for p in tree[1:]:
        try:
            if "python" in p.name().lower():
                busy.append({"pid": p.pid, "cpu_percent": p.cpu_percent(interval=None)})
        except psutil.NoSuchProcess:
            pass
    at_kill = cache.finished_keys()
    t_kill = time.time()
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=60)
    time.sleep(3.0)
    survivors = [p.pid for p in tree if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    after = cache.finished_keys()
    logged = {r["key"] for r in read_progress(cache.progress_path()) if r.get("status", "ok") == "ok"}
    new = after - before
    worker_cpu_busy = [b for b in busy if b["cpu_percent"] > 50]
    summary = {
        "kill_after_s": round(t_kill - t0, 1),
        "finished_before_run": len(before),
        "finished_at_kill": len(at_kill),
        "finished_after_kill": len(after),
        "new_fits_this_run": len(new),
        "appeared_after_kill_snapshot": len(after - at_kill),
        "vanished_after_kill": len(at_kill - after),
        "on_disk_not_logged_by_parent": len(after - logged),
        "worker_processes_at_kill": len(busy),
        "workers_busy_at_kill": len(worker_cpu_busy),
        "lost_fits_upper_bound": len(worker_cpu_busy),
        "stale_tmp_files": len(cache.tmp_files()),
        "surviving_processes": survivors,
        "fit_rate_per_min": round(len(new) / ((t_kill - t0) / 60), 1),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
