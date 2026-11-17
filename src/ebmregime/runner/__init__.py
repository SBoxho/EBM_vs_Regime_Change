"""Grid runner: one job = one fit, content-addressed and resumable (PLAN §12, D-25, D-26).

* :mod:`.jobs`    -- the ``Job`` spec, grid expansion and the cache key;
* :mod:`.cache`   -- atomic npz writes and reads under the fit-cache root;
* :mod:`.execute` -- ``run_fit_job``: one EBM fit and what is stored from it;
* :mod:`.run`     -- ``run_jobs``: skip finished keys, run the rest on joblib loky workers.

``compare.py`` (the pair-metric stage) is the lead's. Submodules are imported on use, so importing
this package does not import the model or surrogate code.
"""

from .jobs import Job, expand_grid, fit_code_hash, job_key, prereg_hash

__all__ = ["Job", "expand_grid", "fit_code_hash", "job_key", "prereg_hash"]
