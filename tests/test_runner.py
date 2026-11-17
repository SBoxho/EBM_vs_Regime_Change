"""Grid runner (M2_SPEC §5; PLAN §12; D-25, D-26). Synthetic data and dummy jobs only.

The kill-and-resume test runs a dummy job function (sleep 0.2 s, return a small array) on 4 loky
workers in a subprocess, kills the whole process tree mid-run (``taskkill /T /F`` on Windows,
``killpg(SIGKILL)`` elsewhere), restarts it, and checks: every job done exactly once in the cache,
no partial or temp files, no worker outliving the kill, at most ``n_workers`` fits lost.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ebmregime import panel as P
from ebmregime.config import load_config, repo_path
from ebmregime.runner import cache, execute
from ebmregime.runner.jobs import (
    FIT_CODE_FILES,
    PACKAGE_ROOT,
    Job,
    canonical_json,
    dispatch_order,
    expand_grid,
    fit_code_files,
    fit_code_hash,
    job_key,
    load_grid,
    prereg_hash,
)
from ebmregime.runner.run import execute_one, read_progress, run_jobs
from ebmregime.splits import half_sample, replicate_plan, time_ordered_bags

PREREG = load_config("prereg")
FAKE = {"prereg": "p" * 64, "code": "c" * 64, "data": "d" * 64}
PY = sys.executable


def fake_key(job: Job) -> str:
    return job_key(job, **FAKE)


def mk_job(**kw) -> Job:
    base = dict(symbol="BTCUSDT", tau="2020-01-01", target="dir", protocol="seed", r=0, seed=123,
                sample_seed=None)
    base.update(kw)
    return Job(**base)


def dummy_jobs(n: int) -> list[Job]:
    return [mk_job(r=i, seed=1000 + i) for i in range(n)]


def dummy_fn(job: Job):
    """Cheap job function (module level so loky workers can import it)."""
    return ({"x": np.arange(5, dtype=np.float64) + job.r, "i": np.array([job.r], dtype=np.int32)},
            {"fit_info": {"fit_seconds": 0.0}, "omp_seen": os.environ.get("OMP_NUM_THREADS")})


# ---------------------------------------------------------------- jobs and keys

def test_job_defaults_and_validation():
    j = mk_job()
    assert (j.model, j.arm, j.horizon, j.data) == ("ebm", "primary", 8, "real")
    assert [f.name for f in dataclasses.fields(Job)] == [
        "model", "arm", "symbol", "tau", "target", "horizon", "protocol", "r", "seed", "sample_seed", "data"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        j.r = 1
    assert Job.from_dict(j.as_dict()) == j
    assert mk_job(data="surrogate:3").surrogate_index == 3 and j.surrogate_index is None
    for bad in (dict(target="up"), dict(protocol="boot"), dict(protocol="half"),       # half needs sample_seed
                dict(sample_seed=5), dict(data="surrogate"), dict(data="surrogate:01"),
                dict(tau="2020-01-15"), dict(tau="2020-01-01T00:00:00+00:00")):
        with pytest.raises(ValueError):
            mk_job(**bad)
    with pytest.raises(TypeError):
        mk_job(seed=np.int64(3))


def test_key_recipe_and_stability():
    j = mk_job()
    k = job_key(j, **FAKE)
    doc = {**FAKE, "job": dataclasses.asdict(j)}
    assert k == hashlib.sha256(canonical_json(doc).encode()).hexdigest()
    assert len(k) == 64 and k == job_key(mk_job(), **FAKE)
    # canonical JSON: key order of the document does not matter
    assert canonical_json({"b": 1, "a": [1, None]}) == '{"a":[1,null],"b":1}'
    # default prereg/code hashes are used when not overridden
    assert job_key(j, data=FAKE["data"]) == job_key(j, prereg=prereg_hash(), code=fit_code_hash(),
                                                    data=FAKE["data"])
    assert job_key(j, prereg="x", code="y", data=lambda s: f"h-{s}") == job_key(j, prereg="x", code="y",
                                                                               data="h-BTCUSDT")


def test_key_sensitive_to_every_job_field():
    base = mk_job(protocol="half", sample_seed=7)
    changes = dict(model="xgb", arm="interactions", symbol="ETHUSDT", tau="2020-02-01", target="vol",
                   horizon=24, protocol="seed", r=1, seed=124, sample_seed=8, data="surrogate:0")
    assert set(changes) == {f.name for f in dataclasses.fields(Job)}
    keys = {fake_key(base)}
    for name, v in changes.items():
        kw = {name: v}
        if name == "protocol":
            kw["sample_seed"] = None
        keys.add(fake_key(dataclasses.replace(base, **kw)))
    assert len(keys) == 1 + len(changes)


@pytest.mark.parametrize("part", ["prereg", "code", "data"])
def test_key_sensitive_to_hashes(part):
    j = mk_job()
    assert job_key(j, **FAKE) != job_key(j, **{**FAKE, part: FAKE[part][:-1] + "0"})


def test_prereg_hash_lf_normalised(tmp_path):
    a, b = tmp_path / "prereg.yaml", tmp_path / "features.yaml"
    a.write_bytes(b"x: 1\ny: 2\n")
    b.write_bytes(b"f: [a]\n")
    h = prereg_hash([a, b])
    assert h == hashlib.sha256(b"x: 1\ny: 2\nf: [a]\n").hexdigest()
    a.write_bytes(b"x: 1\r\ny: 2\r\n")
    assert prereg_hash([a, b]) == h
    a.write_bytes(b"x: 1\ny: 3\n")
    assert prereg_hash([a, b]) != h
    # the default covers configs/prereg.yaml then configs/features.yaml
    assert prereg_hash() == prereg_hash([repo_path("configs/prereg.yaml"), repo_path("configs/features.yaml")])


def test_fit_code_files_listed():
    files = fit_code_files()
    for f in ("config.py", "series.py", "panel.py", "splits/bags.py", "splits/windows.py",
              "splits/__init__.py", "runner/jobs.py", "runner/execute.py"):
        assert f in files
    for f in ("runner/run.py", "runner/cache.py", "runner/__init__.py", "stability/metrics.py",
              "features/build.py"):
        assert f not in files
    assert FIT_CODE_FILES  # documented list


def test_fit_code_hash_sensitivity(tmp_path):
    root = tmp_path / "ebmregime"
    for rel in fit_code_files() + ["runner/run.py"]:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE_ROOT / rel, dst)
    v = "0.7.8"
    h = fit_code_hash(root, v)
    assert h == fit_code_hash(PACKAGE_ROOT, v)
    assert fit_code_hash(root, "0.7.9") != h                                  # interpret version
    (root / "runner/run.py").write_text("# not fit code\n")                  # unlisted file
    assert fit_code_hash(root, v) == h
    bags = root / "splits/bags.py"
    raw = bags.read_bytes().replace(b"\r\n", b"\n")
    bags.write_bytes(raw.replace(b"\n", b"\r\n"))                            # CRLF checkout
    assert fit_code_hash(root, v) == h
    bags.write_bytes(raw + b"\n# changed\n")                                 # listed file
    assert fit_code_hash(root, v) != h
    bags.write_bytes(raw)
    (root / "models").mkdir(exist_ok=True)
    (root / "models/xgb.py").write_text("# new model file\n")                # new file under models/
    assert fit_code_hash(root, v) != h


# ---------------------------------------------------------------- grid

def test_expand_grid_slice_counts():
    grid = load_grid("configs/grid_slice.yaml")
    jobs = expand_grid(grid, PREREG)
    assert len(jobs) == 6225 == 83 * 15 * 3 + 83 * 15 * 2 == grid["expected_jobs"]
    c = Counter((j.data, j.target, j.protocol) for j in jobs)
    for data, targets in (("real", ("dir", "vol", "shuf")), ("surrogate:0", ("dir", "vol"))):
        for t in targets:
            assert c[(data, t, "seed")] == 83 * 5 and c[(data, t, "half")] == 83 * 10
    assert len(c) == 10
    taus = sorted({j.tau for j in jobs})
    assert len(taus) == 83 and taus[0] == "2019-12-01" and taus[-1] == "2026-10-01"
    assert {(j.symbol, j.horizon, j.model, j.arm) for j in jobs} == {("BTCUSDT", 8, "ebm", "primary")}
    # seeds only from replicate_plan
    j0 = next(j for j in jobs if j.data == "surrogate:0" and j.target == "vol" and j.tau == "2021-03-01"
              and j.protocol == "half" and j.r == 4)
    plan = replicate_plan("BTCUSDT", "2021-03-01", "vol", PREREG["replicates"])
    rep = next(p for p in plan if p["protocol"] == "half" and p["r"] == 4)
    assert (j0.seed, j0.sample_seed) == (rep["seed"], rep["sample_seed"])
    assert len({fake_key(j) for j in jobs}) == len(jobs)


def test_expand_grid_expected_count_checked():
    grid = {"symbols": ["BTCUSDT"], "horizon": 8, "taus": ["2020-01-01"],
            "data": [{"source": "real", "targets": ["dir"]}], "expected_jobs": 16}
    with pytest.raises(AssertionError):
        expand_grid(grid, PREREG)
    grid["expected_jobs"] = 15
    assert len(expand_grid(grid, PREREG)) == 15


def test_dispatch_order_vol_first():
    jobs = [mk_job(target=t, protocol=p, r=i, sample_seed=(None if p == "seed" else i))
            for i, (t, p) in enumerate([("dir", "half"), ("vol", "half"), ("shuf", "seed"), ("vol", "seed")])]
    order = [(j.target, j.protocol) for j in dispatch_order(jobs)]
    assert order == [("vol", "seed"), ("vol", "half"), ("shuf", "seed"), ("dir", "half")]


# ---------------------------------------------------------------- cache

def test_cache_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(cache.ENV_VAR, raising=False)
    assert cache.cache_root() == repo_path("data/cache/fits")
    monkeypatch.setenv(cache.ENV_VAR, str(tmp_path / "env"))
    assert cache.cache_root() == tmp_path / "env"
    assert cache.cache_root(tmp_path / "arg") == tmp_path / "arg"
    k = "ab" + "0" * 62
    assert cache.path_for(k, tmp_path) == tmp_path / "ab" / f"{k}.npz"
    with pytest.raises(ValueError):
        cache.path_for("AB" + "0" * 62, tmp_path)


def test_write_read_roundtrip(tmp_path):
    k = fake_key(mk_job())
    arrays = {"f32": np.arange(4, dtype=np.float32), "i64": np.array([1, 2], dtype=np.int64),
              "f64": np.array([0.1]), "m": np.zeros((2, 3), dtype=np.int32)}
    meta = {"a": 1, "np": np.int64(3), "x": np.float32(0.5), "none": None, "s": "é"}
    p = cache.write(k, arrays, meta, root=tmp_path)
    assert p.is_file() and cache.exists(k, tmp_path) and cache.finished_keys(tmp_path) == {k}
    got, m = cache.read(k, tmp_path)
    assert set(got) == set(arrays)
    for name, a in arrays.items():
        assert got[name].dtype == a.dtype and np.array_equal(got[name], a)
    assert m == {"a": 1, "np": 3, "x": 0.5, "none": None, "s": "é"}
    assert cache.tmp_files(tmp_path) == []
    os.remove(p)  # read() closed the file
    with pytest.raises(TypeError):
        cache.write(k, {"o": np.array([object()])}, {}, root=tmp_path)
    with pytest.raises(ValueError):
        cache.write(k, {"meta": np.zeros(1)}, {}, root=tmp_path)
    assert not cache.exists(k, tmp_path) and cache.tmp_files(tmp_path) == []


def test_write_error_leaves_nothing(tmp_path, monkeypatch):
    def broken_savez(fh, **kw):
        fh.write(b"PK\x03\x04partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken_savez)
    k = fake_key(mk_job())
    with pytest.raises(OSError):
        cache.write(k, {"a": np.zeros(3)}, {}, root=tmp_path)
    assert not cache.exists(k, tmp_path)
    assert list(tmp_path.rglob("*")) == [tmp_path / k[:2]]


def test_writer_killed_midwrite_leaves_no_partial_file(tmp_path):
    """Simulated kill: the writer process dies (os._exit) after writing part of the archive."""
    k = fake_key(mk_job())
    final = cache.path_for(k, tmp_path)
    script = textwrap.dedent("""
        import os, sys
        import numpy as np
        from ebmregime.runner import cache
        def savez_then_die(fh, **kw):
            fh.write(b"PK\\x03\\x04" + b"x" * 100000)
            fh.flush()
            os._exit(3)
        np.savez = savez_then_die
        cache.write_atomic(sys.argv[1], {"a": np.arange(10)}, {"m": 1})
    """)
    r = subprocess.run([PY, "-c", script, str(final)], capture_output=True, timeout=60)
    assert r.returncode == 3, r.stderr.decode(errors="replace")
    assert not final.exists() and not cache.exists(k, tmp_path)
    tmps = cache.tmp_files(tmp_path)
    assert len(tmps) == 1 and tmps[0].stat().st_size > 0 and tmps[0].name.startswith(final.name + ".tmp-")
    assert cache.finished_keys(tmp_path) == set()
    assert cache.clean_stale_tmp(tmp_path) == {"removed": 1, "kept": 0}
    assert cache.tmp_files(tmp_path) == []


def test_clean_stale_tmp_keeps_live_writers(tmp_path):
    k = fake_key(mk_job())
    d = tmp_path / k[:2]
    d.mkdir()
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    live = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    try:
        (d / f"{k}.npz.tmp-{dead.pid}-{'a' * 32}").write_bytes(b"partial")
        (d / f"{k}.npz.tmp-{live.pid}-{'b' * 32}").write_bytes(b"in flight")
        (d / "unrelated.txt").write_bytes(b"keep")
        assert cache.clean_stale_tmp(tmp_path) == {"removed": 1, "kept": 1}
        assert sorted(p.name for p in d.iterdir()) == sorted([f"{k}.npz.tmp-{live.pid}-{'b' * 32}",
                                                              "unrelated.txt"])
    finally:
        live.kill()
        live.wait()


@pytest.mark.skipif(os.name != "nt", reason="sharing violations on os.replace are Windows behaviour")
def test_late_duplicate_write_onto_locked_file_is_harmless(tmp_path, monkeypatch):
    k = fake_key(mk_job())
    cache.write(k, {"a": np.arange(3)}, {"writer": 1}, root=tmp_path)
    monkeypatch.setattr(cache.time, "sleep", lambda s: None)
    with open(cache.path_for(k, tmp_path), "rb"):                  # a reader holds the file open
        placed = cache.write_atomic(cache.path_for(k, tmp_path), {"a": np.arange(3)}, {"writer": 2})
    assert placed is False
    arrays, meta = cache.read(k, tmp_path)
    assert meta == {"writer": 1} and np.array_equal(arrays["a"], np.arange(3))
    assert cache.tmp_files(tmp_path) == []


# ---------------------------------------------------------------- run

def test_run_jobs_skips_finished_keys(tmp_path):
    calls = []

    def fn(job):
        calls.append(job.r)
        return dummy_fn(job)

    jobs = dummy_jobs(6)
    s1 = run_jobs(jobs, n_workers=1, root=tmp_path, job_fn=fn, key_fn=fake_key, verbose=False)
    assert (s1["n_cached"], s1["n_todo"], s1["n_ok"]) == (0, 6, 6) and sorted(calls) == list(range(6))
    assert cache.finished_keys(tmp_path) == {fake_key(j) for j in jobs}
    arrays, meta = cache.read(fake_key(jobs[2]), tmp_path)
    assert np.array_equal(arrays["x"], np.arange(5) + 2.0)
    assert meta["key"] == fake_key(jobs[2]) and meta["job"] == jobs[2].as_dict()

    calls.clear()
    s2 = run_jobs(jobs, n_workers=1, root=tmp_path, job_fn=fn, key_fn=fake_key, verbose=False)
    assert (s2["n_cached"], s2["n_todo"]) == (6, 0) and calls == []

    os.remove(cache.path_for(fake_key(jobs[4]), tmp_path))
    s3 = run_jobs(jobs, n_workers=1, root=tmp_path, job_fn=fn, key_fn=fake_key, verbose=False)
    assert (s3["n_cached"], s3["n_ok"]) == (5, 1) and calls == [4]

    log = read_progress(cache.progress_path(tmp_path))
    assert len(log) == 7 and all(r["status"] == "ok" for r in log)
    assert set(log[0]) >= {"key", "job", "fit_seconds", "worker_pid", "wall_utc", "run_id"}


def test_run_jobs_limit_and_dispatch_order(tmp_path):
    calls = []

    def fn(job):
        calls.append(job.target)
        return dummy_fn(job)

    jobs = [mk_job(target=t, r=i, seed=i) for i, t in enumerate(["dir", "shuf", "vol", "dir", "vol"])]
    s = run_jobs(jobs, n_workers=1, root=tmp_path, job_fn=fn, key_fn=fake_key, limit=2, verbose=False)
    assert s["n_ok"] == 2 and calls == ["vol", "vol"]


def test_run_jobs_errors_logged_not_cached(tmp_path):
    def fn(job):
        if job.r % 2:
            raise ValueError(f"bad {job.r}")
        return dummy_fn(job)

    jobs = dummy_jobs(4)
    s = run_jobs(jobs, n_workers=1, root=tmp_path, job_fn=fn, key_fn=fake_key, verbose=False)
    assert (s["n_ok"], s["n_errors"]) == (2, 2)
    assert cache.finished_keys(tmp_path) == {fake_key(j) for j in jobs if j.r % 2 == 0}
    errs = [r for r in read_progress(cache.progress_path(tmp_path)) if r["status"] == "error"]
    assert len(errs) == 2 and "ValueError: bad" in errs[0]["error"]
    with pytest.raises(RuntimeError):
        run_jobs(dummy_jobs(8), n_workers=1, root=tmp_path / "b", job_fn=fn, key_fn=fake_key,
                 max_errors=1, verbose=False)


def test_progress_log_survives_partial_line(tmp_path):
    log = cache.progress_path(tmp_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b'{"key": "x", "status": "ok"}\n{"key": "trunc')       # killed mid-line
    run_jobs(dummy_jobs(2), n_workers=1, root=tmp_path, job_fn=dummy_fn, key_fn=fake_key, verbose=False)
    rows = read_progress(log)
    assert [r["key"] for r in rows][0] == "x" and len(rows) == 3


def test_run_jobs_loky_workers_single_threaded(tmp_path):
    env_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    before = {v: os.environ.get(v) for v in env_vars}
    jobs = dummy_jobs(6)
    s = run_jobs(jobs, n_workers=2, root=tmp_path, job_fn=dummy_fn, key_fn=fake_key, verbose=False)
    assert s["n_ok"] == 6
    pids = set()
    for j in jobs:
        _, meta = cache.read(fake_key(j), tmp_path)
        assert meta["omp_seen"] == "1" and meta["omp_num_threads"] == "1"
        pids.add(meta["worker_pid"])
    assert os.getpid() not in pids                                  # ran in loky workers
    assert {v: os.environ.get(v) for v in env_vars} == before       # parent environment restored


RESUME_SCRIPT = textwrap.dedent("""
    import os, sys, time, uuid
    import numpy as np
    from ebmregime.runner.jobs import Job, job_key
    from ebmregime.runner.run import run_jobs

    def sleepy_job(job):
        marks = os.environ["RUNNER_TEST_MARKS"]
        open(os.path.join(marks, f"{job.r}-{os.getpid()}-{uuid.uuid4().hex}"), "w").close()
        time.sleep(0.2)
        return {"x": np.full(100, job.r, dtype=np.float64)}, {"fit_info": {"fit_seconds": 0.2}}

    def key(job):
        return job_key(job, prereg="p" * 64, code="c" * 64, data="d" * 64)

    if __name__ == "__main__":
        root, n_workers, n_jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        jobs = [Job(symbol="BTCUSDT", tau="2020-01-01", target="dir", protocol="seed", r=i, seed=i,
                    sample_seed=None) for i in range(n_jobs)]
        s = run_jobs(jobs, n_workers=n_workers, root=root, job_fn=sleepy_job, key_fn=key, progress_every=10)
        print("SUMMARY", s["n_cached"], s["n_ok"], s["n_errors"], flush=True)
""")


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=60)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=60)


def test_kill_and_resume(tmp_path):
    import psutil

    n_workers, n_jobs = 4, 60
    root, marks = tmp_path / "fits", tmp_path / "marks"
    marks.mkdir()
    script = tmp_path / "resume_job.py"
    script.write_text(RESUME_SCRIPT)
    env = {**os.environ, "RUNNER_TEST_MARKS": str(marks)}
    keys = {job_key(Job(symbol="BTCUSDT", tau="2020-01-01", target="dir", protocol="seed", r=i, seed=i,
                        sample_seed=None), **FAKE) for i in range(n_jobs)}

    out1 = open(tmp_path / "run1.log", "wb")
    kw = {"start_new_session": True} if os.name != "nt" else {}
    p = subprocess.Popen([PY, str(script), str(root), str(n_workers), str(n_jobs)], stdout=out1,
                         stderr=subprocess.STDOUT, env=env, **kw)
    deadline = time.time() + 90
    while len(cache.finished_keys(root)) < 12:
        assert p.poll() is None, (tmp_path / "run1.log").read_text(errors="replace")
        assert time.time() < deadline, "run 1 made no progress"
        time.sleep(0.05)
    tree = psutil.Process(p.pid).children(recursive=True)
    _kill_tree(p)
    out1.close()
    gone, alive = psutil.wait_procs(tree, timeout=10)
    assert not alive, f"processes outlived the kill: {alive}"

    done1 = cache.finished_keys(root)
    time.sleep(0.5)
    assert cache.finished_keys(root) == done1                     # nothing writes after the kill
    assert 12 <= len(done1) < n_jobs, "the kill must land mid-run"
    assert done1 <= keys
    for k in done1:                                               # finals are complete archives
        arrays, meta = cache.read(k, root)
        assert meta["key"] == k and arrays["x"].shape == (100,)
    starts1 = len(list(marks.iterdir()))

    r2 = subprocess.run([PY, str(script), str(root), str(n_workers), str(n_jobs)], capture_output=True,
                        env=env, timeout=180)
    log2 = r2.stdout.decode(errors="replace") + r2.stderr.decode(errors="replace")
    assert r2.returncode == 0, log2
    assert f"SUMMARY {len(done1)} {n_jobs - len(done1)} 0" in log2

    finals = sorted(root.glob("*/*.npz"))
    assert cache.finished_keys(root) == keys and len(finals) == n_jobs   # all done, one file per key
    assert cache.tmp_files(root) == []                                    # stale tmp files removed
    for k in keys:
        arrays, meta = cache.read(k, root)
        assert np.all(arrays["x"] == meta["job"]["r"])
    starts = len(list(marks.iterdir()))
    lost = starts - n_jobs                    # fits started but never finished (re-run after restart)
    assert 0 <= lost <= n_workers, (starts1, len(done1), starts)
    log = read_progress(cache.progress_path(root))
    ok = [r for r in log if r["status"] == "ok"]
    run_ids = {r["run_id"] for r in ok}
    assert len(run_ids) == 2
    second = [r["key"] for r in ok if r["run_id"] != ok[0]["run_id"]]
    assert len(second) == len(set(second)) == n_jobs - len(done1) and not set(second) & done1


# ---------------------------------------------------------------- fit job

def synth_panel(start="2019-01-01", end="2020-02-01", seed=0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left", name="ts").as_unit("us")
    n = len(idx)
    rng = np.random.default_rng(seed)
    names = P.feature_names()
    df = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names, index=idx)
    df["hour"] = np.asarray(idx.hour, dtype=np.int64)
    df["dow"] = np.asarray(idx.day_of_week, dtype=np.int64)
    df["features_ok"] = True
    df.iloc[:50, df.columns.get_loc("features_ok")] = False
    logit = 0.8 * df["ret_4h"].to_numpy() + 0.4 * np.sin(2 * np.pi * df["hour"].to_numpy() / 24)
    y_dir = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    df["y_dir_h8"] = y_dir
    df["y_vol_h8"] = 0.5 * df["rv_ratio_24h"].to_numpy() + 0.5 * rng.normal(size=n)
    df["y_shuf_h8"] = np.roll(y_dir, 2000)
    for c in ("y_dir_h8", "y_vol_h8", "y_shuf_h8"):
        df.iloc[-8:, df.columns.get_loc(c)] = np.nan
    return df


class _FakeModel:
    def __init__(self, kind):
        self.kind = kind

    def predict_proba(self, X):
        p = np.full(len(X), 0.25)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return np.full(len(X), -0.5)


@pytest.mark.parametrize("protocol,target,data", [("seed", "dir", "real"), ("half", "vol", "surrogate:1"),
                                                  ("half", "shuf", "real")])
def test_run_fit_job_plumbing(monkeypatch, protocol, target, data):
    """Rows, bags, kind, test arrays and meta, with a fake fit (no model needed)."""
    panel = synth_panel()
    seen = {}

    def fake_fit(X, y, bags, kind, random_state):
        seen.update(X=X, y=y, bags=bags, kind=kind, rs=random_state)
        keep = ~(bags == 0).all(axis=1)
        return _FakeModel(kind), {"n_rows_in": len(X), "n_rows_dropped": int((~keep).sum()),
                                  "n_rows_fit": int(keep.sum()), "fit_mask": keep, "fit_seconds": 0.01,
                                  "bag_counts": np.zeros((14, 3), np.int32),
                                  "best_iteration": np.ones((1, 14), np.int32)}

    monkeypatch.setattr(execute, "_model_api", lambda: (fake_fit, lambda m: {"intercept": np.zeros(1)}))
    surro_calls = []
    monkeypatch.setattr(execute, "_surrogate_panel", lambda s, i: surro_calls.append((s, i)) or panel)
    monkeypatch.setattr(P, "load_panel", lambda s: panel)

    job = mk_job(tau="2020-01-01", target=target, protocol=protocol, r=3, seed=77,
                 sample_seed=(None if protocol == "seed" else 99), data=data)
    arrays, meta = execute.run_fit_job(job)
    assert surro_calls == ([("BTCUSDT", 1)] if data != "real" else [])

    v = PREREG["validation"]
    tr = P.train_rows(panel, job.tau, target, 8, PREREG["windows"]["train_months"])
    pos = np.flatnonzero(tr)
    ts_tr = panel.index[pos]
    common = dict(n_bags=v["outer_bags"], tail_frac=v["tail_frac"], block_hours=v["block_hours"],
                  bag_train_frac=v["bag_train_frac"])
    if protocol == "seed":
        rows = pos
        bags = time_ordered_bags(ts_tr, 8, seed=77, **common)
    else:
        mask, vs = half_sample(ts_tr, 8, v["tail_frac"], v["block_hours"], 99)
        rows = pos[mask]
        bags = time_ordered_bags(ts_tr[mask], 8, seed=77, val_start=vs, block_origin=ts_tr[0], **common)
        assert meta["val_start"] == vs.isoformat()
    assert seen["X"].index.equals(panel.index[rows]) and list(seen["X"].columns) == P.feature_names()
    assert np.array_equal(seen["bags"], bags) and seen["bags"].dtype == np.int8
    ycol = f"y_{target}_h8"
    assert np.array_equal(seen["y"], panel[ycol].to_numpy()[rows])
    assert seen["kind"] == ("regressor" if target == "vol" else "classifier") == meta["kind"]
    assert seen["rs"] == 77

    te = P.test_rows(panel, job.tau, target, 8)
    assert arrays["test_ts"].dtype == np.int64 and arrays["test_pred"].dtype == np.float32
    assert np.array_equal(arrays["test_ts"], panel.index[te].as_unit("ns").asi8)
    assert len(arrays["test_ts"]) == meta["n_test"] == te.sum() == 31 * 24 - 8
    assert np.allclose(arrays["test_pred"], -0.5 if target == "vol" else 0.25)
    assert np.array_equal(arrays["test_y"], panel[ycol].to_numpy()[te].astype(np.float32))
    assert arrays["best_iteration"].dtype == np.int32 and arrays["bag_counts"].dtype == np.int32
    keep = ~(bags == 0).all(axis=1)
    assert meta["y_train_mean"] == pytest.approx(panel[ycol].to_numpy()[rows][keep].mean())
    assert meta["job"] == job.as_dict()
    assert meta["fit_info"] == {"n_rows_in": len(rows), "n_rows_dropped": int((~keep).sum()),
                                "n_rows_fit": int(keep.sum()), "fit_seconds": 0.01}
    for f in ("git_sha", "git_dirty", "interpret_version", "host", "utc"):
        assert f in meta
    json.dumps(meta)


def test_model_kind_from_prereg():
    assert execute.model_kind("dir") == execute.model_kind("shuf") == "classifier"
    assert execute.model_kind("vol") == "regressor"


def _models_ebm_available() -> bool:
    try:
        from ebmregime.models.ebm import export_fit, fit_ebm, predict_from_export  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _models_ebm_available(), reason="ebmregime.models.ebm not available yet")
def test_real_ebm_fit_job_synthetic(tmp_path, monkeypatch):
    """One real EBM fit (classifier, seed protocol, 14 bags) on a synthetic 13-month panel, through
    the worker wrapper and the cache."""
    from ebmregime.models.ebm import predict_from_export

    panel = synth_panel()
    monkeypatch.setattr(execute, "job_panel", lambda job: panel)
    job = mk_job(tau="2020-01-01", target="dir", protocol="seed", r=0, seed=5)
    key = fake_key(job)
    res = execute_one(execute.run_fit_job, job, key, str(tmp_path))
    assert res["status"] == "ok", res.get("traceback")
    arrays, meta = cache.read(key, tmp_path)
    n_feat = len(P.feature_names())
    assert arrays["intercept"].shape == (1,) and arrays["term_features"].shape == (n_feat,)
    for j, t in enumerate(P.feature_types()):
        assert arrays[f"scores_{j}"].dtype == np.float32
        assert (f"cats_{j}" if t == "nominal" else f"cuts_{j}") in arrays
    te = P.test_rows(panel, job.tau, "dir", 8)
    X_te = panel[P.feature_names()].iloc[np.flatnonzero(te)]
    pred = predict_from_export(arrays, X_te, "classifier")
    assert np.allclose(arrays["test_pred"], pred, atol=1e-6)
    assert arrays["bag_counts"].shape == (14, 3)
    assert meta["kind"] == "classifier" and meta["key"] == key
    assert meta["fit_info"]["n_rows_fit"] == meta["fit_info"]["n_rows_in"] - meta["fit_info"]["n_rows_dropped"]
    assert meta["fit_info"]["n_rows_dropped"] == 8          # the h-row purge before the validation tail
