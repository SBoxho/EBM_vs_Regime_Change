# M2 implementation spec (interfaces between modules)

Written by the lead before the module sub-agents start. Each module is owned by one implementer and lives in its
own files. Normative sources: `docs/PLAN.md` §6, §7, §9.3, §12; `docs/DECISIONS.md` D-11, D-13, D-15, D-17,
D-18, D-26, D-45, D-46, D-50; `configs/prereg.yaml` (frozen, never edited). Where this spec and the PLAN disagree,
the PLAN wins: stop and report instead of guessing.

General rules:
- Python 3.13, numpy 2.5, pandas 3.1, scipy 1.18, interpret 0.7.8, joblib 1.6, arch 8.0. No new dependencies.
- Use the venv: `.venv/Scripts/python.exe -m pytest`. The full suite must stay green.
- Every module reads its settings from `configs/prereg.yaml` via `ebmregime.config.load_config("prereg")`;
  no hard-coded copies of pre-registered numbers except as defaults that tests check against the config.
- Do not edit files you do not own. Shared primitives (`ebmregime/series.py`, `ebmregime/panel.py`,
  `ebmregime/splits/*`, `ebmregime/config.py`) are the lead's; ask if you need a change.
- No model may be fitted on real data in unit tests except where this spec says so; use synthetic data.
- Never write into `data/` from tests except under pytest's `tmp_path`.

## Shared primitives (exist; lead-owned)

`ebmregime.panel`: `feature_names()` (16, frozen order), `feature_types()` (`"nominal"` for `hour`, `dow`),
`feature_families()`, `load_panel(symbol)`, `data_hash(symbol)`, `target_column(target, h)` with target in
`("dir", "vol", "shuf")` giving `y_dir_h8` etc., `usable_mask`, `train_rows`, `fit_rows`, `test_rows`,
`purged_before_tail`. Panel index `ts` is a UTC `DatetimeIndex` (bar open time, hourly). `hour` and `dow` are
int64. Usable rows have no NaN features.

`ebmregime.splits`: `retrain_schedule`, `train_mask`, `test_mask`, `label_time`, `overlap_fraction`,
`time_ordered_bags`, `half_sample`, `replicate_plan`, `stable_seed`.

## 1. `models/ebm.py` (models implementer)

```python
make_ebm(kind: str, random_state: int, cfg_models: dict | None = None)  # kind in {"classifier", "regressor"}
fit_ebm(X: pd.DataFrame, y: np.ndarray, bags: np.ndarray, kind: str, random_state: int,
        cfg_models: dict | None = None) -> tuple[estimator, dict]          # (model, fit_info)
export_fit(model) -> dict[str, np.ndarray]
eval_terms_from_export(export: dict, X: pd.DataFrame) -> np.ndarray     # (n, 16) float64
predict_from_export(export: dict, X: pd.DataFrame, kind: str) -> np.ndarray
```

- `make_ebm`: interpret 0.7.8 defaults with the overrides of `prereg.models.ebm.overrides`: `interactions=0`,
  `n_jobs=1`, explicit `random_state`, `feature_names=panel.feature_names()`,
  `feature_types=panel.feature_types()`. It asserts that the estimator's params equal
  `prereg.models.ebm.defaults_relied_on` (classifier/regressor values where they differ) and that
  `interpret.__version__ == prereg.models.ebm.interpret_version`.
- `fit_ebm`: `X` has exactly the 16 feature columns in frozen order; `bags` int8 of shape
  `(n_rows, outer_bags)`. **Drops rows that are 0 in every bag before calling `fit` (D-45)**, then passes the
  remaining rows and the matching bag rows. `fit_info` = `n_rows_in`, `n_rows_dropped`, `n_rows_fit`,
  `bag_counts` (int32 `(outer_bags, 3)`: counts of −1, 0, +1 per bag among fitted rows), `best_iteration`
  (int32 array as in `best_iteration_`), `fit_seconds`.
- `export_fit` (the stored fit; nothing is pickled):
  - `intercept`: float64 `(1,)`;
  - for feature index `j` = 0..15: `scores_{j}` = `term_scores_[j]` as **float32**; continuous: `cuts_{j}` =
    `bins_[j][0]` as **float64** (exact cuts, so bin membership never changes); nominal: `cats_{j}` = int64
    array of the category values in score-index order (score index 1..n_cats; 0 = missing, last = unseen);
  - `term_features`: int32 `(16,)` sanity copy (must be 0..15, main effects only).
- `eval_terms_from_export`: lookup per D-46: continuous score index `np.digitize(x, cuts) + 1` (`right=False`),
  NaN → 0; nominal: category → its index, unknown → last (unseen). Must equal `model.eval_terms(X)` computed
  with the model's scores cast to float32, exactly, and `model.eval_terms(X)` itself to ≤ 1e-6 absolute.
- `predict_from_export`: η = intercept + row sum of terms; classifier → `1/(1+exp(−η))`, regressor → η.
  Must match `predict_proba[:, 1]` / `predict` to ≤ 1e-6.
- Tests `tests/test_models_ebm.py` (synthetic data, small): D-45 drop is applied (intercept equals a fit on
  the pre-dropped rows; differs from passing them); export round-trip for continuous, tied and nominal
  features including values on cuts, beyond range, NaN, unseen nominal level; params match prereg; both kinds.

## 2. `stability/surrogate.py` (surrogate implementer)

```python
surrogate_rows(n: int, mean_block: float, seed: int) -> np.ndarray      # int64 source positions, len n
make_surrogate(panel: pd.DataFrame, start, end, mean_block_hours: float, seed: int) -> pd.DataFrame
surrogate_panel(symbol: str, index: int, prereg: dict | None = None) -> pd.DataFrame
```

- PLAN §9.3 and D-50: stationary (Politis–Romano) block bootstrap of the **rows** of the panel with
  `start <= ts < end`, taken in time order: geometric block lengths with mean `mean_block_hours` rows, uniform
  block starts, circular wrap, n draws for n rows.
- `make_surrogate` returns a frame with the **original timestamps** of the rows in `[start, end)` as index and,
  at position i, **all columns** of source row `surrogate_rows(...)[i]` (features, `features_ok`, every target,
  regime states, `valid`) moved jointly. Rows outside `[start, end)` are not included. dtypes preserved.
- `surrogate_panel(symbol, index)`: `start = first_tau(symbol) − train_months`, `end = last_tau + 1 month`
  (from `prereg.windows`); `mean_block_hours = prereg.surrogates.mean_block_hours` (252);
  `seed = splits.stable_seed(prereg.surrogates.seed, symbol, "surrogate", index)`. Cached per process.
- Tests `tests/test_surrogate.py`: determinism; permutation-with-repetition property (every output row equals
  some source row, all columns jointly); mean run length of consecutive source positions ≈ `mean_block`;
  marginal distribution preserved in expectation; index equals the original grid; circular wrap; bounds.

## 3. `explain/` (explain implementer): `cells.py`, `contributions.py`

```python
make_xref(fit_masks: list[np.ndarray], cap: int, seed: int) -> np.ndarray   # sorted row positions
make_cells(X_ref: pd.DataFrame, n_cells: int = 50, min_rows: int = 20,
           nominal: list[str] | None = None, fallback_cells: int = 20, min_cells: int = 10) -> CellGrid
cell_shapes(contrib: np.ndarray, grid: CellGrid) -> list[np.ndarray]     # per feature j: (K_j,) re-centred
cell_shapes_many(contribs: np.ndarray, grid: CellGrid) -> list[np.ndarray]  # (R, n, J) -> per j: (R, K_j)
```

- `make_xref` (D-50): union of the given boolean row masks (one per window; `panel.fit_rows`), then if larger
  than `cap`, a sorted uniform subsample without replacement with `np.random.default_rng(seed)`.
- `CellGrid` (frozen dataclass): `features` (names, frozen order); per feature j: `cell_of_row[j]` (int array,
  len n_ref, values 0..K_j−1), `weights[j]` (float64 `(K_j,)`, ω_k = n_k / n_ref, sums to 1), `x_rep[j]`
  (median of X_ref values in the cell; for nominal the level value), `edges[j]` (continuous: interior cut
  values; nominal: levels), `is_nominal[j]`, `fallback_used[j]` (bool).
- Continuous features, PLAN §6.1 step 2: interior edges at the pooled quantiles k/G, k = 1..G−1 of X_ref
  (`np.quantile`, method "linear"); duplicate edges merged (ties); cell = `np.searchsorted(edges, x, side="right")`;
  then cells with fewer than `min_rows` rows merged into an adjacent cell (the smaller neighbour; repeat until
  none remain); if fewer than `min_cells` cells remain, rebuild with `G = fallback_cells` and set
  `fallback_used` (PLAN §9.7 fallback 3; if still < `min_cells`, keep the result and leave the flag set).
  Nominal features: one cell per level present in X_ref, weight = pooled frequency.
- `cell_shapes`: f_j(k) = mean of `contrib[:, j]` over rows in cell k; then f̃ = f − Σ_k ω_k f(k) (PLAN §6.1
  step 4). Vectorised (`np.bincount`), because it runs ~10⁵ times.
- `contributions.py` also provides `ebm_contrib(export, X_ref)` = `models.ebm.eval_terms_from_export` (thin
  wrapper; may import lazily) and `eta(export, contrib)` = intercept + row sum.
- Tests `tests/test_explain.py`: equal-probability cells on continuous data (weights ≈ 1/G); ties merge;
  min-rows merge; fallback; nominal levels; re-centring gives Σ ω f̃ = 0; a contribution that depends only on
  x gives the analytic cell means; invariance: two step functions with different bin edges that agree as
  functions give identical shapes.

## 4. `stability/metrics.py` and `stability/floor.py` (stability implementer)

All functions take re-centred cell shapes and the cell weights ω of one feature (`f: (K,)` or `(R, K)`,
`w: (K,)`, Σw = 1). Pure numpy/scipy, vectorised where cheap.

`metrics.py` (PLAN §6.2, §6.3):
```python
d2(fa, fb, w)                      # Σ ω (fa − fb)²; broadcasting over leading dims
amplitude(f, w)                    # sqrt(Σ ω f²)
nd2(fa, fb, w)                     # d2 / ((A_a² + A_b²)/2); NaN if both amplitudes are 0
shape_corr(fa, fb, w)              # density-weighted Pearson; NaN if either is flat
shape_spearman(fa, fb)             # Spearman over cells, average ranks for ties (secondary)
sign_agreement(fa, fb, w, tau)     # -> (SA, coverage); cells where both |f| < tau excluded; NaN if none
monotonicity_agreement(x_rep, fa, fb, min_abs=0.3)   # 1.0 / 0.0, NaN if not qualifying (either |ρ_S| ≤ 0.3)
local_monotonicity_agreement(fa, fb, step_sd, mult=2.0)  # share of qualifying steps with equal sign; NaN if none
importance(f, w)                   # Σ ω |f|
importance_shares(I)               # I / ΣI
weighted_tau(Ia, Ib)               # scipy.stats.weightedtau (hyperbolic), returns the statistic
spearman_rank(Ia, Ib)
topk_jaccard(Ia, Ib, k)
tv_distance(sa, sb)                # ½ Σ |sa − sb|
```

`floor.py` (PLAN §6.4). Shapes arrive per feature as `(R, K_j)` arrays for each protocol and window:
```python
pair_mean_d2(F, w)                 # mean over the R(R−1)/2 distinct pairs of rows of F of d2
cross_mean_d2(Fa, Fb, w)           # mean over all Ra×Rb cross pairs
sigma2(F_seed, F_half, w)          # -> dict seed=½·pair_mean_d2(F_seed), tot=½·pair_mean_d2(F_half),
                                   #    samp=max(0, tot − seed)
cell_noise_sd(F_half)              # (K,) per-cell SD: sqrt(½ mean over half pairs of (f_r(k) − f_r'(k))²)
drift_pair(shapes_a, shapes_b, weights, overlap)   # full per-pair computation, see below
```
- `drift_pair` input: for windows a and b, per feature j: `seed` (5, K_j) and `half` (10, K_j) shapes on the
  pair's common cells, and `weights[j]`; `overlap` = ρ = `splits.overlap_fraction`. Output: a dict with per
  feature arrays (length J) `d2_cross`, `s2_seed_a`, `s2_tot_a`, `s2_samp_a`, same for b, `DR`, `DR_adj`,
  `A_a`, `A_b` (amplitude of the seed-mean shape), `snr_a`, `snr_b` (mean over seed fits of A² / s2_tot),
  `nd2` (seed-mean shapes), `rho`, `SA`, `SA_cov`, `MA`; and scalars `DR_tot`, `DR_adj_tot`, `seed_share_a`
  (Σ s2_seed / Σ s2_tot), `seed_share_b`, plus importance metrics on seed-mean shapes: `wtau`, `spearman_imp`,
  `jacc3`, `jacc5`, `tv`.
  - DR_j = d2_cross / (s2_tot_a + s2_tot_b); DR_tot = Σ d2_cross / Σ (s2_tot_a + s2_tot_b).
  - DR_adj_j = d2_cross / (2[(1 − ρ)·s̄2_samp + s̄2_seed]) with s̄ the average of the two windows;
    DR_adj_tot = Σ numerators / Σ denominators.
  - τ_jk for SA = sqrt of the average over the two windows of `cell_noise_sd²`.
  - MA needs `x_rep[j]`; pass `x_rep` (list, None for nominal → MA NaN).
- Tests `tests/test_stability.py` with known answers: identical shapes → d2 = 0, ρ = 1, SA = 1; sign flip
  f → −f → d2 = 4A², ρ = −1, SA = 0, nd2 = 4; unrelated equal-amplitude orthogonal shapes → nd2 = 2; density
  weighting: a difference confined to cells with weight 0 gives d2 = 0; analytic DR for a Gaussian synthetic
  (shape + iid noise with known variance: E[DR] = 1 with no drift, = 1 + δ²/(2σ²)·… with drift).

## 5. `runner/` (runner implementer): `jobs.py`, `cache.py`, `execute.py`, `run.py`; `scripts/run_grid.py`; `configs/grid_slice.yaml`

`runner/compare.py` is reserved for the lead.

- **Job** (frozen dataclass, `jobs.py`): `model="ebm"`, `arm="primary"`, `symbol`, `tau` (ISO date
  `"2019-12-01"`), `target` (`dir|vol|shuf`), `horizon` (8), `protocol` (`seed|half`), `r`, `seed`,
  `sample_seed` (None for seed), `data` (`"real"` or `"surrogate:<i>"`). `expand_grid(grid_cfg, prereg)` builds
  all jobs from `configs/grid_slice.yaml` with `splits.replicate_plan` (seeds come only from there).
- **Cache key** (D-26): sha256 of canonical JSON of `{prereg: sha256 of LF-normalised configs/prereg.yaml +
  configs/features.yaml, code: fit_code_hash(), data: panel.data_hash(symbol), job: asdict(job)}`.
  `fit_code_hash()` = sha256 over sorted (relative path, LF-normalised bytes) of the files that determine a fit:
  `config.py, series.py, panel.py, splits/*.py, models/*.py, stability/surrogate.py, runner/jobs.py,
  runner/execute.py`, plus `interpret.__version__`. (Hashing all of `src/` would invalidate every fit whenever
  metric code changes; the metric stage has its own key. Logged by the lead as a decision.)
- **Cache** (`cache.py`): `data/cache/fits/<key[:2]>/<key>.npz`. `write_atomic(path, arrays, meta)`: write to
  `<path>.tmp-<pid>-<uuid>` in the same directory with `np.savez` (uncompressed is fine), fsync, `os.replace`.
  `meta` is stored as a JSON string array `meta`. `exists(key)`, `read(key) -> (arrays, meta)`. Stale tmp files
  are ignored and removed on startup.
- **Fit job** (`execute.py`, `run_fit_job(job) -> (arrays, meta)`):
  1. panel = `panel.load_panel(symbol)` or `stability.surrogate.surrogate_panel(symbol, i)`;
  2. `tr = panel.train_rows(...)`; for `half`: `mask, vs = half_sample(ts_tr, h, tail_frac, block_hours,
     sample_seed)` then `bags = time_ordered_bags(ts_tr[mask], h, outer_bags, tail_frac, block_hours,
     bag_train_frac, seed=seed, val_start=vs, block_origin=ts_tr[0])`; for `seed`:
     `bags = time_ordered_bags(ts_tr, h, ..., seed=seed)`;
  3. kind = classifier for dir/shuf, regressor for vol; `fit_ebm` (drops all-zero rows);
  4. arrays = `export_fit(model)` + `test_ts` (int64 ns), `test_pred` (float32), `test_y` (float32) on
     `panel.test_rows`, `best_iteration`, `bag_counts`;
  5. meta = job fields, key, kind, `fit_info` scalars, `y_train_mean` (mean of y over the fitted rows),
     `n_test`, interpret version, git HEAD SHA and dirty flag (informational, not in the key), host, UTC time.
- **Run** (`run.py`): `run_jobs(jobs, n_workers=16, log_path=...)`: skip keys already in the cache; dispatch the
  rest with `joblib.Parallel(n_jobs=n_workers, backend="loky", return_as="generator_unordered")`; set
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` for workers; the **worker** writes its own npz
  atomically (so a parent kill loses only in-flight fits) and returns a small summary; the parent appends one
  JSON line per finished job to `data/cache/fits/progress.jsonl` (key, job, fit_seconds, worker pid, wall time)
  and prints progress every ~50 fits with ETA. Order jobs so that the slowest (vol) start first.
- **`scripts/run_grid.py configs/grid_slice.yaml [--workers 16] [--limit N] [--dry-run]`**; `--dry-run` prints
  job counts per (data, target, protocol) and how many are cached.
- **`configs/grid_slice.yaml`**: symbols `[BTCUSDT]`, horizon 8, arm primary; data `real` with targets
  `[dir, vol, shuf]`; data `surrogate:0` with targets `[dir, vol]`; all taus of `prereg.windows` for the symbol;
  replicates from `prereg.replicates`. Expected: 83 × 15 × 3 + 83 × 15 × 2 = 6,225 jobs.
- Tests `tests/test_runner.py`: key stability and sensitivity (changes with any job field, prereg, data hash,
  code hash); atomic write leaves no partial file when the writer is killed mid-write (simulate); skip of
  finished keys; `expand_grid` counts; a resume test with a cheap dummy job function (the run function must
  accept a job function argument for this) killed in a subprocess and restarted: no duplicates, all done,
  lost ≤ n_workers. Real EBM fits only in one small marked test on synthetic data.
