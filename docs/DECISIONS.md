# Design decisions

Numbered log. Each entry gives the decision, the reason, and the evidence where it was checked. Later changes are added as new entries that reference the old number. Entries are never rewritten silently. After the `prereg-v1` tag, changes to frozen items are marked **DEVIATION**.

Sources of evidence:
- **[data]** data-feasibility check of the live archive, 2026-11-08;
- **[compute]** timing runs on this laptop;
- **[verify]** API checks the lead ran on interpret 0.7.8;
- **[method]** the method-design draft;
- **[lit]** `LITERATURE.md`.

---

## Data

**D-01. Source: Binance public archive, spot monthly klines. No raw data in the repository.**
- *Reason:* it is public, has long history, and has checksummed monthly files. Spot data avoids funding and perp-basis confounds.
- *Evidence [data]:* S3 listing via `s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=…`; `.CHECKSUM` files are sha256sum-format and all of about 435 downloaded files verified.
- *Parser note:* objects uploaded from 2025-02 onward carry `ChecksumAlgorithm`/`ChecksumType` XML elements that silently broke a regex parser. Use an XML parser.

*(Superseded by D-34.)* **D-02. Symbols: BTCUSDT, ETHUSDT, BNBUSDT, LTCUSDT, XRPUSDT. Reserves: ADAUSDT, TRXUSDT.**
- *Reason:* the five longest continuous USDT spot histories among liquid pairs.
- *Evidence [data]:* first months are BTC and ETH 2017-08, BNB 2017-11, LTC 2017-12, XRP 2018-05; all have files through 2026-10 with no missing months. The lead independently re-listed the XRP and LTC first/last files.
- SOL (2020-08) is too short for 83 windows.
- Open question O-3 (BNB).

*(Amended by D-34 for SOL.)* **D-03. Common range 2018-06-01 to 2026-10-31.**
- *Reason:* it starts after XRP's partial first month and after the 2018-02-08/09 outage (32 h).
- *Evidence [data]:* gaps are exchange-wide (BTC and ETH identical). None occur in 2022 or from 2024 onward. Inside the range the largest gaps are 10 h.

**D-04. Download 15m klines, model on 1h bars aggregated from them.**
- *Reason:* 1h modelling bars carry less microstructure noise and fewer gap bars, while 15m sub-bars give better realised-variance estimates for features and targets.
- *Evidence:*
  - [compute] row count barely drives EBM cost (4× rows ≈ 2.7× time), so this is a statistical choice, not a computational one;
  - the data sub-agent and the method draft both recommended 1h.
- 15m modelling (h = 16) is a Tier 2 sensitivity arm.
- Open question O-1.

*(Superseded by D-33.)* **D-05. Primary horizon h = 4 hours. Secondary h = 24.**
- *Reason:* with overlapping labels, n_eff per 12-month window is about n/h. That gives about 2,190 at h = 4 against about 365 at h = 24.
- *Conflict resolved:* the method draft suggested h = 24 as primary. Rejected, because n_eff ≈ 365 per window would leave direction-target shapes almost entirely noise.
- One shared h across targets keeps the purge and overlap structure identical.

*(Gap tolerance revised by D-43.)* **D-06. Cleaning rules.**
- *Rules:* units normalised by magnitude (values above 1e14 are µs); off-grid bars snapped to the grid; O=H=L=C zero-volume bars treated as missing; truncated bars flagged; no forward-fill across gaps; 1h bars valid only if all four 15m sub-bars are valid; feature and label validity masks.
- *Evidence [data]:*
  - spot timestamps are µs from the 2025-01 file onward, documented in the README and confirmed in the files (`1735689600000000`); futures are unaffected;
  - 2018-02-09/11 bars are offset by +28m14s;
  - BTC 1h has 4 zero-volume placeholders and 15 truncated bars.
- Each file's sha256 and Last-Modified go into a manifest, because Binance replaces files occasionally (README `updates/` note).

## Features and targets

*(Amended by D-35: 16 features.)* **D-07. A fixed set of 14 scale-free features** (PLAN §5.1).
- *Reason:* it matches your generic list (returns, RV, volume z-scores, range and position in range, time). It is large enough for interesting shapes and small enough to compare.
- Every continuous feature is normalised by its own trailing history, so a vol regime shift does not slide feature distributions along the x-axis and look like shape drift [method, pitfall 1].
- Feature selection is out of scope.

*(Day of week confirmed under D-35.)* **D-08. Hour of day and day of week are nominal features.**
- *Reason:* a sin/cos pair produces two shape functions that cannot be read on their own.
- Day of week is added beyond your list because crypto has a strong weekend vol pattern. Confirm under O-4.

**D-09. Three targets.**
- Direction (near-noise).
- Normalised forward log RV, log(RV_fwd / RV_168h) (positive control).
- Direction circularly shifted by a fixed random offset of 90–400 days that is not a multiple of 24 h or 168 h (no-signal control).
- *Reason:*
  - the vol target is known to be learnable [lit: Corsi 2009; Dudek et al. 2025], so stable explanations there are not "stable because nothing was learned";
  - the circular shift keeps the label's autocorrelation and overlap, so the control has the same n_eff as direction; an iid permutation would not.
- Normalised rather than raw vol: open question O-2.

## Models

**D-10. Hyperparameters fixed across all windows. No per-window tuning.**
- *Reason:* per-window tuning creates explanation drift by itself [lit: Müller et al. 2023].

**D-11. Primary EBM = interpret 0.7.8 defaults (outer_bags = 14, max_bins = 1024, …) with `interactions=0`, explicit seeds and `n_jobs=1`.**
- *Reason:*
  - Q2 is about the variance a practitioner sees from a single default fit;
  - interactions=0 gives a clean GAM with identifiable main effects and the same model class as logistic regression.
- *Conflict resolved:* the compute sub-agent recommended a reduced config (outer_bags = 4, interactions = 10, max_bins = 256) to save time. Rejected for the primary arm. Measured cost of 14 bags without pairs is 3.0 s per fit at 8,760 rows [verify], so defaults are affordable, and reducing bags would change the very variance we measure.
- *Evidence [compute]:* same-seed fits are bit-identical at n_jobs = 1 and −1, so parallelising across fits is safe.

*(Pair count for 16 features in D-35.)* **D-12. Interactions are studied in a separate arm with `interactions=5`. Pairs are purified before pair-shape comparison.**
- *Reason:* the default "3x" means 42 pairs for 14 features, which dilutes everything and costs about 85 s per fit.
- *Evidence [verify]:* on 0.7.8, main-effect `term_scores_` are bit-identical (max abs diff 0.0) between interactions = 0 and 5 with the same seed and bags. Pairs are fitted after the mains, with the mains frozen. Only the intercept moves, by about 3e-4. So the pair arm cannot contaminate the primary analysis.
- Pairs are only mean-centred by default (`purify_result=False` [method, source-read]), hence explicit purification [lit: Lengerich et al. 2020].

**D-13. Internal validation uses time-ordered `bags`. Layout: last 15% of the window is validation, h-row purge before it, per-bag 85% of contiguous 168-row training blocks. Array shape (n_samples, outer_bags).**
- *Reason:* EBM's default random 15% validation split leaks with overlapping labels, makes early stopping overrun, and inflates instability.
- *Correction [verify]:* the method draft read the GitHub develop branch and claimed the shape must be (outer_bags, n_samples). On the pinned 0.7.8 that shape raises `ValueError: bags has 8760 bags and self.outer_bags is 14 bags`, while (n_samples, outer_bags) works, matching the documentation.
- This behaviour is pinned in `tests/test_interpret_api.py`.
- At least one −1 row is required, otherwise early stopping is disabled [method, source-read].

## Metrics

*(16 features since D-35.)* **D-14. Primary rank metric: weighted Kendall τ, with hyperbolic top-weighting.**
- *Reason:* the bottom of a 14-feature ranking is near-zero noise, and plain Spearman or Kendall penalise churn there as much as at the top. Weighted τ handles ties and is one library call.
- RBO is a sensitivity check [lit: Webber et al. 2010].

*(Centring wording amended by D-46.)* **D-15. Shapes are compared as cell-averaged contributions on 50 pooled equal-probability cells of a pair-specific reference sample, re-centred under that common density.**
- *Reason:* fits have different bin edges, and each fit is centred under its own window density.
- *Evidence [verify]:* the bin-weighted mean of a term is about 1e-17, i.e. each fit is centred under its own density. len(term_scores_) = cuts + 3, with missing and unseen bins at the ends.
- Equal-probability cells make every summary density-weighted automatically, so sparse tails cannot dominate.
- The same estimator is used for EBM (`eval_terms`), XGB (SHAP) and LR (β·x).

**D-16. Primary drift statistic: DR_tot = between-window shape distance / noise-floor distance. Headline at Δ = 12 months (zero overlap). Equivalence margin [0.8, 1.25]; "drift" requires the CI lower bound above 1.25.**
- *Reason:*
  - normalising by an empirical floor answers the question as posed and is comparable across features, targets and models;
  - Δ = 12 removes the artificial similarity of overlapping windows.
- *Change from the method draft:* the draft used "lower bound > 1" for drift. Changed to 1.25 for symmetry with the equivalence margin, so that "drift" means material drift. A CI inside [0.8, 1.25] that excludes 1 is reported as "detectable but negligible".

**D-17. Noise floor: R_s = 5 seed-only full fits, plus R_b = 10 fits on independent half-samples of contiguous 168-h blocks.**
- *Reason:*
  - half-samples drawn without replacement reproduce the full-n sampling variance (σ²/n);
  - they avoid duplicate rows that interact with EBM binning and min_samples_leaf;
  - 10 replicates give about 18 df for a pooled two-window floor.
- The internal outer-bag SD (`standard_deviations_`) is only a diagnostic. It understates sampling noise.

**D-18. The floor is calibrated empirically: G0 gate on the shuffled control and on 3 stationary surrogates per symbol (BTC, ETH). Pre-specified fallback DR* = DR / median surrogate DR.**
- *Reason:* the σ²/n argument holds only for smooth estimators, and early-stopped boosting may not obey it.

## Design

*(Amended by D-34: SOL has 56 retrains.)* **D-19. Rolling 12-month windows, 1-month step: 83 retrains per symbol (τ = 2019-12 to 2026-10). Expanding windows only as illustration.**
- *Reason:* expanding windows stabilise by construction. A 12-month window gives about 8,700 rows. With a 1-month step, consecutive retrains differ by one month added and one dropped, which makes regime-change covariates precise.
- 6- and 24-month windows are exploratory.

*(Amended by D-44: the trailing-2-year R1 is primary.)* **D-20. Regimes: R1 = trailing 30-day RV percentile within the symbol's expanding history (primary); R2 = 90-day trend t-statistic; R3 = Amihud liquidity percentile. The full-sample HMM is descriptive only.**
- *Reason:* the three are simple, causal and distinct. Volatility is the most persistent and best-measured state.
- A full-sample HMM uses the future, so it lives in a separate module and every figure that uses it is labelled.

*(Direction-skill SE updated by D-33: about 0.06 AUC at h = 8.)* **D-21. Early-warning test.**
- Only leads k ≥ 0 count, because k = −1 is mechanical.
- Primary test: an incremental panel regression controlling for regime change, vol level, covariate shift (PSI) and prediction drift.
- Placebos: circular shifts of at least 6 months (999 draws), time-shuffled windows, and drift computed from the shuffled-target models.
- Vol target as primary. An equivalence bound (|partial r| < 0.2) defines "no useful early warning".
- *Reason:* drift and decay are both driven by the regime shift itself, so a raw correlation would fool us. n_eff ≈ 25–35 limits power, so "inconclusive" must be a possible outcome. Monthly direction-target skill is too noisy (SE ≈ 0.04 AUC).

**D-22. Statistics.**
- Stationary bootstrap over windows, with `arch` optimal block length floored at max(3, Δ); identical time blocks across symbols.
- n_eff reported for every estimate.
- Holm on the six primary tests; BH q = 0.10 on everything else.
- *Reason:* overlapping targets and autocorrelated windows invalidate naive SEs [lit: Politis & Romano 1994; Politis & White 2004; Patton et al. 2009].
- Crypto symbols are not independent replications.

**D-23. The XGBoost arm uses interventional TreeSHAP with a common 100-row background drawn from each pair's X_ref, on about 2,000-row evaluation sets, and only lags Δ ∈ {1, 12}.**
- *Reason:* path-dependent SHAP uses each model's own training cover as the background, which confounds drift.
- *Evidence [verify]:* interventional SHAP costs 1.3 s per 2,000 rows (3–9 s per 5,000) against about 0.5 s for the fit, so restricting lags keeps the arm at about 1.5 wall-h.

**D-24. In XGBoost, hour and day of week are integer codes. In the linear models they are one-hot, and coefficients are mapped back to raw units.**
- *Reason:* shap's interventional explainer support for native XGBoost categorical splits is not established. Integer codes still let trees isolate hours.
- Converting LR coefficients to raw units stops each window's scaler from redefining the axis.

## Engineering

**D-25. Parallelism across fits: 16 joblib loky workers, each fit single-threaded.**
- *Evidence [compute]:* about 4,300 reduced-config fits per hour, against 680–990 for sequential multi-threaded fits. About 10.5 effective cores. Results are deterministic across n_jobs.

**D-26. Content-addressed cache with one job per fit.**
- The key is a hash of the prereg config, the git SHA of `src/`, the data-manifest hash and the job fields.
- Writes are atomic, and the runner skips finished keys.
- A second cached stage computes pair metrics.
- *Reason:* the grid is resumable after a kill, and results are traceable to code and data versions.

*(Amended by D-37: plus a Zenodo record.)* **D-27. Pre-registration is frozen as git tag `prereg-v1` before the first slice touches real data. Only the fallbacks in PLAN §9.7 may change afterwards. Anything else is a logged DEVIATION, and the affected hypothesis is demoted to exploratory.**
- *Reason:* the slice uses BTC data that is also in the confirmatory grid. Freezing first keeps the slice from steering the hypotheses.

*(Amended by D-33: h = 8.)* **D-28. First slice: BTCUSDT, h = 4, EBM interactions = 0, all three targets, all 83 windows, R = 5 + 10, plus 1 surrogate. Under 1 wall-hour.**
- *Reason:* it is the minimum that can show whether the idea holds. A noise floor plus drift on the near-noise target alone could not tell "the market changed" from "there was never a signal". The vol and shuffled targets add only about 0.2 h, so they are included.
- The go/no-go criteria are in PLAN M2.

*(Confirmed and extended by D-38.)* **D-29. Licensing: MIT for code (proposed); data not hosted; derived artefacts under CC BY-NC-SA 4.0 with attribution to Binance Vision.**
- *Evidence [data, re-checked by lead]:* the Binance Vision Dataset Terms v1.0 (last updated 2026-08-26) license the datasets under CC BY-NC-SA 4.0, prohibit hosting, and require derivative works to keep the licence and credit. https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md
- Open question O-6.

**D-30. Pinned environment: interpret-core 0.7.8, xgboost 3.4.1, shap 0.54.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0, Python 3.13.2; `arch` to be added at M0.**
- *Reason:* D-12, D-13 and D-15 depend on version-specific `interpret` behaviour that differs between the docs, the develop branch and the release. Any version bump must pass `test_interpret_api.py`.

**D-31. Literature: arXiv 2512.15780 was rejected.**
- The sub-agent described it as showing explanation drift ahead of AUROC drops in financial ML. Its landing page is an adversarial-robustness paper.
- The lead also re-checked the five most load-bearing preprints, and added three adjacent works it verified itself [lit].

**D-32. Only the timing script is committed at the planning stage (`scripts/time_fits.py`, synthetic data only).**
- The lead's API verification checks (D-12, D-13, D-15) become pytest tests in M1 rather than a loose script.

---

## Revision 1: your answers of 2026-11-08

**D-33. Primary horizon h = 8 hours on 1h bars; secondary h = 24. Supersedes D-05.**
- *Reason:* your choice (O-1).
- *Consequences:*
  - n_eff per 12-month window ≈ 8,760 / 8 ≈ 1,095;
  - the purge and the validation-tail gap are 8 rows;
  - the forward-vol target uses 32 fifteen-minute sub-bars;
  - one test month has n_eff ≈ 90, so monthly direction-skill SE ≈ 0.06 AUC. That strengthens the case for running H4 on the vol target;
  - the 15m sensitivity arm uses h = 32 bars.
- Compute is unchanged.

**D-34. Symbols are BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, each on its own range, analysed as an unbalanced panel with a balanced-panel sensitivity check. Supersedes D-02, amends D-03 and D-19.**
- *Reason:* your choice (O-3).
- SOL is used from 2020-09-01, the first full month, giving τ_1 = 2022-03-01 and 56 retrains (44 Δ = 12 pairs). BTC, ETH and XRP keep 83. Total: 305 retrains.
- Cutting every symbol to SOL's range would have discarded 27 retrains each from BTC, ETH and XRP, including the 2020–21 cycle.
- *Evidence [lead, 2026-11-08]:*
  - S3 listing: SOLUSDT 1h and 15m monthly files cover 2020-08 to 2026-10 (75 each), and all 150 sha256 checks pass;
  - first bar 2020-08-11 06:00 UTC;
  - 1h: 19 missing bars out of 54,546, in 10 runs; 15m: 93 out of 218,184;
  - 3 zero-volume placeholders and 6 truncated bars;
  - every gap falls on the exchange-wide outage dates already seen in BTC (largest: 2021-08-13, 4 h).
- BNB and LTC are dropped.
- The regime label R1 for SOL exists from 2021-09.

**D-35. Two features added, for 16 in total. Day of week kept. Amends D-07 and D-08.**
- `taker_imb_4h`: share of taker-buy quote volume over the last 4h, minus 0.5.
- `xret_4h`: the partner asset's vol-normalised 4h return. The partner is BTC for ETH, XRP and SOL, and ETH for BTC.
- *Reason:* your choice (O-4).
- The partner for BTC is ETH because a BTC-on-BTC copy would duplicate `ret_4h`, and EBM splits shape mass between duplicates at random, which would create artificial instability.
- Drift is measured within symbol, where the feature is fixed, so the asymmetry does not bias the main metrics.
- With 16 features there are P = 120 possible pairs, and the default "3x" setting would select 48.
- The timing runs already used 16 features, so the compute estimates stand.

**D-36. Positive-control target confirmed: log(RV_fwd,h / RV_168h). Raw log RV is exploratory. Confirms D-09.**
- *Reason:* your choice (O-2).

**D-37. Pre-registration: annotated git tag `prereg-v1` + SHA-256 hash list + an embargoed Zenodo record that you deposit. Amends D-27.**
- *Reason:* you asked for my recommendation (O-5).
  - A git tag alone can be rewritten.
  - An embargoed Zenodo record has public metadata and a DOI immediately, keeps the files private until a chosen date, and the hashes in its description prove the content.
  - Researchers already use Zenodo embargoes for exactly this, e.g. https://zenodo.org/records/23839234 [lead, web check].
- *Why not OSF:* OSF is running a "Project Transition". A library guide (https://lib.guides.umd.edu/c.php?g=1136217&p=8292722) reports no new OSF projects from 2026-11-16 and read-only projects from 2027-02-19, and OSF's help page says a registration creates a connected project (https://help.osf.io/article/330-welcome-to-registrations). Too uncertain to depend on now.
- Creating the Zenodo record needs your account. I prepare the files.

**D-38. Licensing confirmed: MIT for code; derived data, figures and model artefacts under CC BY-NC-SA 4.0 with attribution to Binance Vision; no raw or processed data in the repository. Extends D-29.**
- *Reason:* your answer (O-6).

**D-39. Compute budget accepted: Tier 1 ≈ 3.6 h (≈ 5 h with margin), Tier 2 ≈ 4.7 h (≈ 6.5 h). Tier 2 is kept in full.**
- *Reason:* your answer (O-7). Recomputed for 305 retrains.

---

## M0: data pipeline (2026-11-08)

**D-40. Archive self-inconsistency rule: hours whose prices disagree between the archive's 1h and 15m files are invalid; hours where only volume disagrees are flagged `volume_suspect`. Extends D-06.**
- *Evidence [M0 build]:* each 1h bar aggregated from the 15m files was compared with the archive's own 1h file.
  - Hours compared, valid in both: BTC 80,466; ETH 80,406; XRP 74,352; SOL 54,514.
  - Prices match exactly everywhere except one XRP hour (2022-04-29 09:00: close 0.6315 from the 15m files, 0.6313 in the 1h file).
  - Volume and trade counts disagree in 8 (BTC), 7 (ETH), 4 (XRP) and 4 (SOL) hours. In every case the 1h file reports *more*, by a factor of 1.000–1.52 (e.g. BTC 2017-12-04 09:00).
  - Several of these hours are shared across symbols (2021-04-22 02:00, 2021-04-23 01:00 and 02:00).
  - The one XRP price conflict falls in an hour that also disagrees on volume.
- *Reason:*
  - Our aggregation is verified against hand-computed bars in `tests/test_bars.py`, so the disagreement lies between the archive's own files.
  - We cannot tell which file is right. Prices drive returns, RV and targets, so any price doubt invalidates the hour. Volume doubt only affects volume features, which treat flagged hours as missing.
- This affects 1 invalid hour and ≤ 8 flagged hours per symbol. It is pinned in `tests/test_data_archive.py`.

**D-41. The full available history is downloaded, not only the analysis range. The archive's 1h files are downloaded only as a cross-check.**
- *Reason:*
  - The regime label R1 uses each symbol's expanding history from its first bar.
  - The cross-check is what found D-40.
  - Extra cost: 798 files, 74 MB.
- Download policy:
  - The terms (§7.1) publish no numeric rate limit. We use 4 threads, retries with exponential backoff on 429/5xx, and a generic User-Agent.
  - Every file is verified against its `.CHECKSUM` before it is kept.
  - Re-runs re-verify and skip existing files.

**D-42. M0 acceptance results.**
- All 798 files verified (manifest hash `bf30026133152753…`).
- The µs switch is at the 2025-01 file for all 4 symbols and both intervals.
- Missing-bar counts reproduce the independent planning audit exactly (1h archive: BTC 127, ETH 127, XRP 87, SOL 19; 15m: BTC 562, SOL 93).
- First bars: BTC 2017-08-17 04:00, XRP 2018-05-04 08:00, SOL 2020-08-11 06:00.
- Invalid hours inside the analysis ranges: BTC, ETH and XRP 113 of 73,800 each (0.153%, the same exchange-wide outages); SOL 32 of 54,048 (0.059%). The longest invalid run is 10 h (2018-06-26).
- Candidate volume-regime breaks for R3 are listed in `notebooks/00_data_audit.ipynb` §6, for example BTC 2023-04 (log quote volume −1.59). Their causes are not yet established; they must be dated before H3 is interpreted.

---

## M1: features, targets, regimes, splits (2026-11-09)

Implemented by four sub-agents (one per module, each with its own files). The lead wrote the shared primitives (`series.py`) and the configs, reviewed the code, and verified it independently (D-47).

**D-43. Gap tolerance has two classes. Revises D-06.**
- **Window rule:** an aggregation over k raw hourly values needs ceil(0.90·k) of them (`series.MIN_FRAC`).
- **History rule:** a reference or history window over derived values needs ceil(0.50·n) (`series.MIN_FRAC_HISTORY`). This covers the 180-day `rv_level_pct` history, the 30 lagged `qv_z_24h` values, and the 30/90/365-day windows of daily regime inputs.
- Labels stay strict.
- *Reason [features and regimes implementers, confirmed by lead]:* the single 0.95 rule cascaded. One outage of 9 h or more blanked σ for a week, `rv_level_pct` for up to 180 days, and R1 for 30 days. BTC lost 13% of its feature rows (37% in 2019; 124 undefined R1 days after first definition).
- *After the change:*
  - `features_ok` from analysis_start: BTC 0.990, ETH 0.990, XRP 0.969, SOL 0.962.
  - No undefined R1 days after first definition.
  - The longest in-range outage (10 h) is below the 16 h a 168 h window tolerates.
- Decided before any model was fitted.

**D-44. Primary R1 is the trailing-2-year percentile (`r1_vol_2y`). The expanding-history version (`r1_vol`) becomes a sensitivity check. Amends D-20 and PLAN §8.**
- *Reason [regimes implementer's smoke run]:* crypto volatility trends down, so against an expanding history the "high" state almost disappears after 2022. BTC has 0% high in 2023 and 2025, the H3 covariate C has median 0, and only 13 retrains have C ≥ 0.5. That label could not test H3.
- *Under the 2-year version (BTC):* high share 23% overall; C mean 0.33, median 0.10; 29 retrains with C ≥ 0.5.
- The two versions are identical until 730 days of history exist.
- The decision uses only the distribution of the regime labels, never any model output or target relationship.
- **Confirmed by the user on 2026-11-09.**

**D-45. Rows that are 0 in every bag must not be passed to `fit`. The M2 model wrapper drops them.**
- *Evidence [lead, pinned in `tests/test_interpret_api.py::test_labels_of_rows_excluded_from_every_bag`]:* in interpret 0.7.8 the labels of such rows (the purge before the validation tail) leave every shape function bit-identical, but they move the **intercept**, for the classifier and the regressor alike.
- Dropping the rows removes the effect exactly.
- Validation (−1) rows also enter the intercept. That is acceptable: their labels are observed by τ.

**D-46. Further interpret 0.7.8 facts pinned by the splits implementer.**
- **Centring:** `bin_weights_` counts every row passed to `fit` (+1, 0 and −1), so each fit is centred under the density of all rows passed, not only its +1 rows. This amends the wording of D-15. Cross-window comparisons re-centre under the pooled density anyway.
- **Bin lookup:** score index = `np.digitize(x, cuts) + 1` with `right=False` (a value exactly on a cut goes to the upper bin); NaN goes to index 0.
- **Replication counts:** counts > 1 in `bags` are accepted and act approximately like integer sample weights, so protocol (ii) is usable.
- **Determinism:** `n_jobs` does not change results.
- **Early stopping:** `best_iteration_` ≈ 1,200 rounds against `max_rounds` 50,000 with time-ordered bags.
- **Purge size:** the h-row purge from `label_time(ts) = ts + (h+1) h ≤ τ` drops exactly h rows on a full grid. PLAN §7's "i + h ≤ τ" is shorthand for this.

**D-47. M1 verification by the lead.**
- `tests/test_m1_handcheck.py` re-derives every feature, both targets at h = 8 and h = 24, and the R1 (both versions) and R2 states for 25 random real BTC rows or 15 random days. It uses a separate numpy code path from the raw 15m grid and none of the package's helpers.
  - Agreement is within 1e-9 relative.
  - A deliberate off-by-one (hour t inside the `qv_z_1h` reference window) would change the value by 6%, so the check can fail.
- End-to-end truncation test on real data at 2019-05-15 03:00, 2021-04-23 01:00, 2023-03-24 12:00 and 2025-01-01:
  - features before the cut, labels whose label time is at or before the cut, and regime states of days ending before the cut are bit-identical with and without all later raw data.
- `series.forward_mean_strict` was rewritten after the targets implementer found that a running rolling sum carried about 1e-16 from beyond t+h. That is not information leakage, but it broke bit-exact no-lookahead tests.
- Full suite: all tests pass (see M1 summary).

**D-48. M1 descriptive facts, for the record (no models fitted).**
- **Shuffle offsets:** BTC 239.6 days, SOL 231.5 days.
- **Direction base rate:** BTC h=8 is 0.515 from 2018-06.
- **Exact-zero 8h returns** are dropped: SOL 0.34% (tick-size discreteness at low prices), BTC 5 rows.
- **Mean of `y_vol_h8`:** BTC −0.40 (a log-mean below a log-mean of a right-skewed variance; expected).
- **Usable training rows per window (features plus h=8 labels):** 8,415–8,775 across all 305 windows.
- **Primary R1 coverage of the first training window:** BTC and ETH 100%, XRP 54%, SOL 51%, because those symbols need 365 days of history first (PLAN §8).
- **R1 flicker:** R1 changes state about 10–17 times a year. Short runs near the tercile cuts inflate the event indicator E. C, the primary H3 covariate, is a share-based measure and less affected. No hysteresis is added (not pre-registered).

---

## Pre-registration freeze (2026-11-17)

**D-49. Consistency pass before the freeze. No model had been fitted on real data.**
- PLAN §2, §3, §9 and `configs/prereg.yaml` were checked against D-01 to D-48. Fixed:
  - PLAN §8 still called the expanding-history R1 primary and the trailing-2-year version the sensitivity check, the reverse of D-44. Corrected; PLAN §2 now names `r1_vol_2y` as the primary R1, and H3's covariate C uses it.
  - PLAN §8 said only SOL's first windows are partly labelled. XRP's are too (D-48). The first defined R1 days, read from the panel (regime labels only), are 2018-09-05 (BTC), 2018-09-08 (ETH), 2019-05-19 (XRP) and 2021-08-26 (SOL).
  - PLAN §12 says `prereg.yaml` holds "metrics, thresholds", but it held only targets, windows, validation, replicates and regimes. Added, transcribed from PLAN §2, §6, §7, §9, §13 and D-11, D-23, D-24, D-43, D-45, D-46: symbols, gap tolerance, the purge rule, the EBM, XGBoost and linear configurations, the explain and stability settings, the surrogate design, every hypothesis threshold, the statistics settings, the §9.7 fallbacks and the M2 go/no-go criteria. Existing keys are unchanged.
  - The EBM defaults recorded in `prereg.yaml` were read from the installed interpret 0.7.8 signatures (`inspect.signature`, no fitting) and match PLAN §7.
  - PLAN §6.1 said each term is centred under its own *training* weights; D-46 found the weights count every row passed to `fit`. Wording corrected, and the D-46 lookup convention (`right=False`, NaN to index 0) added.
  - PLAN §7 now states the exact purge (D-46) and the D-45 rule that rows which are 0 in every bag are dropped before `fit`.
  - `configs/features.yaml` described the old 0.95 gap tolerance. Corrected to D-43 (0.90 window rule, 0.50 history rule).
  - Amendment markers added in front of D-06, D-12, D-14, D-15, D-20 and D-21, pointing to the later entries that changed them (D-21's SE of 0.04 AUC was for h = 4).
  - "DRAFT" note removed from `prereg.yaml`; PLAN status line updated.
- The test suite passes unchanged (145 tests).

**D-50. Clarifications written into the pre-registration before the freeze.** Each fills a gap the plan left open. None uses any model output.
- **Surrogate mean block length: 252 hours**, the midpoint of the pre-stated "1–2 weeks". Rows from [τ_1 − 12 months, τ_last + 1 month) are resampled circularly in time order, features and targets jointly, and placed on the original timestamps so that windows, purge, bags and test months are built as for real data.
- **G0 fallback check.** The shuffled control has no surrogate counterpart, so its DR* uses the direction-target surrogate. DR* passes if the 95% CI of its median at Δ = 12 lies inside [0.8, 1.25] for the shuffled control and for the surrogates themselves. The surrogate check tests only dispersion, because its median is 1 by construction.
- **X_ref for a pair** is the union of the rows the two windows pass to `fit` (overlapping rows count once), capped at 20,000.
- **Per-window SNR** (used by H1-dir and M2 criterion 3): A²_j is the mean of A²_{m,j} over the 5 seed-only fits; A²_j and σ²_tot,j are both computed on X_ref = the window's own rows. Summaries over windows are medians.
- **Primary skill** uses the seed-ensemble mean prediction (mean of the 5 seed-only full-window fits) on the test month; per-fit skill is also reported. Vol climatology is the training-window mean of `y_vol`; the direction base rate is the training-window mean of `y_dir`.
- **M2 wall-time estimate** for criterion 4, from the PLAN §11.2 unit costs: 6,225 fits, about 20,200 CPU-seconds, **0.53 wall-hours** at 10.5 effective cores (PLAN §13).

**D-51. Freeze.** `configs/prereg.yaml`, `configs/features.yaml`, `docs/PLAN.md`, `docs/DECISIONS.md` and `src/` are frozen by the annotated tag `prereg-v1`. Their SHA-256 hashes, and how to recompute them, are in `docs/PREREG_HASHES.txt`. The Zenodo deposit is prepared in `docs/ZENODO_DEPOSIT.md`, added after the tag because it quotes the tagged commit; the DOI is recorded in a later entry. From here on, any change to a frozen item other than a PLAN §9.7 fallback is a **DEVIATION** (D-27).

---

## After the freeze

**D-52. DEVIATION from D-37 and PLAN §2 (pre-registration rule) and §13: no Zenodo record. The pre-registration rests on the annotated git tag `prereg-v1` and the SHA-256 list in `docs/PREREG_HASHES.txt` only.** (2026-11-17)
- *Reason:* your decision, before any model was fitted on real data.
- *Consequence:* there is no third-party timestamp. A reader has to trust the git history (the tag, its commit date, and the hashes) that the plan was fixed before the slice. The paper must say so plainly and must not call the study "registered" in the sense of a public registry.
- *Scope:* this changes how the freeze is evidenced, not what is frozen. No hypothesis, metric, target, horizon, feature or test changes, so no hypothesis is demoted to exploratory.
- M2 starts without the DOI condition of PLAN §13; open question O-9 is closed.

---

## M2: first slice (2026-11-17)

Implemented by five sub-agents (models, runner, explain, stability metrics, surrogate), one per module, each against the interface spec `docs/M2_SPEC.md` written by the lead. The lead wrote the shared primitives (`panel.py`, `stats/bootstrap.py`), the comparison stage (`runner/compare.py`), the go/no-go evaluation (`report/slice.py`), reviewed the code, and verified it independently (D-54).

**D-53. Engineering choices made during M2 (none touches a frozen item).**
- **Fit cache key (amends D-26).** The key hashes the files that determine a fit (`config.py`, `series.py`, `panel.py`, `splits/*`, `models/*`, `stability/surrogate.py`, `runner/jobs.py`, `runner/execute.py`, plus the interpret version), not the git SHA of all of `src/`. *Reason:* a key over all of `src/` would invalidate every fit whenever metric or report code changed, and the metric stage reads fits without changing them. The git HEAD SHA and a dirty flag are still stored in every fit's metadata. The prereg part of the key hashes `configs/prereg.yaml` and `configs/features.yaml` with LF line endings; the data part hashes the manifest hash and the panel file.
- **Storage.** Term scores are float32 (PLAN §11.2); continuous bin cuts are stored as float64, so that bin membership of any value is exactly what the fitted model used; nominal categories as int64. Nothing is pickled. About 190 KB per fit.
- **Classifier labels.** The panel stores `y_dir` and `y_shuf` as float 0.0/1.0. interpret 0.7.8 turns float labels into the string classes `'0.0'`/`'1.0'`, so the fit job casts them to int64 (after checking they are 0/1).
- **Surrogates.** `hour` and `dow` travel with their source row like every other column (PLAN §9.3: "(feature, target) rows" resampled jointly), so in a surrogate they need not match the new timestamp.
- **Not implemented at M2 (secondary metrics):** the noise-floor ratio for the importance TV distance (PLAN §6.3 does not define its floor), the feature-family distance, local monotonicity on decile steps. They are added before M3.

**D-54. M2 slice results and verification (BTCUSDT, h = 8; 6,225 fits; `notebooks/01_slice_noise_floor_and_drift.ipynb`).**
- *Lead's verification (independent of the implementers):*
  - `tests/test_m2_verify.py` with the naive reference `tests/m2_reference.py` (written from PLAN §6 without reading the implementers' code). Known answers: identical shapes give 0; a sign flip gives d² = 4A², ρ = −1, SA = 0, nd² = 4; a difference beyond the data is invisible, and one confined to the top cell gives exactly δ²ω(1 − ω); step functions with different bin edges, or with a centring offset, give identical shapes; `drift_pair` matches the reference to 1e-9.
  - `scripts/verify_floor_synthetic.py`: real EBM fits (prereg settings) on a stationary synthetic process, 48 windows, 1,440 fits. Implementation agrees with the reference to 1e-15. At Δ = 12: DR = 1.11 [0.93, 1.24] (regression target), 0.83 [0.72, 1.08] (weak-signal classifier). At Δ < 12, DR_adj is well above 1 (1.4–2.3): shifting a window by a month moves an early-stopped EBM more than its 1/12 data turnover predicts. Short-lag γ therefore has to be read against the surrogate's γ at the same Δ (PLAN §9.3).
  - Kill-and-resume on the real grid (`scripts/kill_resume_trial.py`): hard kill of the process tree after 151 s. No partial or temporary files, no surviving processes, no fit appeared or vanished after the kill; the restart skipped all 467 finished fits. **16 fits were lost** (one per busy worker), which is the design bound but one more than the pre-registered "fewer than 16": criterion 1 fails by one.
- *Go / no-go (PLAN §13 M2):*
  1. End-to-end run with resume: passes except the strict count (16 lost, criterion says < 16).
  2. **G0 fails, and so does the DR* fallback.** Median DR_tot at Δ = 12 (n = 71, 95% CI): shuffled 0.44 [0.28, 0.60]; surrogate dir 0.64 [0.49, 1.03]; surrogate vol 1.25 [1.15, 1.34]. DR*: shuffled 0.69 [0.44, 0.94]; surrogate dir CI [0.77, 1.61]; surrogate vol [0.92, 1.07].
  3. Positive control passes: 8 features with median vol SNR > 4 (hour 17.1, range_1h 16.5, qv_z_1h 12.6, dow 11.0, rv_ratio_4h 9.4, rv_ratio_24h 8.4, rv_level_pct 6.2, pos_range_168h 4.6); vol OOS R² > 0 in 83 of 83 months, median 0.35. Direction skill is about 0 (median log-loss skill 0.002), the shuffled control 0.000.
  4. Wall time passes: 0.64 h against 0.53 h (ratio 1.21).
- *Diagnosis of the G0 failure (control arms only).* The full-window fits behave as σ²/n estimators: on the shuffled target the cross-window distance at Δ = 12 is about twice the fitted-noise amplitude A² of a full fit, as theory says. The half-sample floor does not: half-sample fits carry 4.6× (not 2×) the full fit's noise amplitude, and σ²_tot is 2.8× A²_full (theory: 1). Boosting rounds do not shrink with n (≈ 2,800 for classifiers, ≈ 9,700 for regressors, half and full alike), so with half the rows an EBM fits disproportionately more noise per bin. The PLAN §6.4 argument that half-samples reproduce σ²/n holds only for smooth estimators, as §9.3 warned. Where signal dominates (surrogate vol) the bias reverses, which is why a single DR* rescaling cannot fix both.
- *Consequence (PLAN §2):* the floor must be fixed before any real-target drift result is looked at. Real-target γ(Δ), DR over time, per-feature DR and the shape gallery are withheld from the notebook. **Disclosure:** a summary printout during the run also showed the real dir and vol γ medians to the lead; they have not been interpreted or used.
- The seed share is about 2% for every target, but it is σ²_seed / σ²_tot with the inflated σ²_tot, so it is not interpretable either until the floor is fixed.
- *Re-timed compute (PLAN §11.4):* mean single-thread fit 4.3 s (classifier, full window), 3.3 s (classifier, half-sample), 8.2 s (regressor, full), 7.5 s (regressor, half): 1.4–1.9× the synthetic timings of §11.1, because real data runs more boosting rounds. The 16 loky workers deliver 14.1 effective cores, not 10.5, so wall time grows only 1.2×. Tier 1 EBM arms: about 1.2× the §11.2 figures.

---

## Fixing the noise floor (2026-11-17)

**D-55. DEVIATION (pre-specified before any result): candidate replacements for the half-sample floor, tested on control arms only, with a fixed adoption rule.** Written before any candidate was fitted.
- *Why:* G0 and its DR* fallback failed (D-54). Half-sample fits carry 4.6× (not 2×) the full fit's fitted noise, so the floor is inflated on noise-dominated shapes. PLAN §2 requires the floor to be fixed before any real-target drift result is looked at.
- *What stays unchanged:* the full-window seed-only fits (protocol i) and the cross-window distance d̄²_cross, the cells, the metrics, the targets, G0's statistic and band. Only σ²_tot,j (the denominator of DR) is replaced.
- *Candidate B, full-n stationary bootstrap (the pre-registered protocol ii, used as the floor):* 10 fits per window. Each draws stationary-bootstrap replication counts over the pre-tail training rows (mean block 168 h, the prereg block length) with a fresh model seed; each of the 14 outer bags uses the counts on its own time-ordered 85% of blocks; validation tail −1. Rows drawn 0 times are 0 in every bag and are dropped before `fit` (D-45). σ²_B,j = ½ · mean over replicate pairs of d²_j.
- *Candidate R, rate-corrected disjoint subsampling:* per window, 4 random partitions of the 168-h blocks into 2 disjoint halves (8 fits) and 2 random partitions into 4 disjoint quarters (8 fits); pre-tail and validation-tail blocks are partitioned separately, so every part has its own tail; fresh model seed per fit. Only fits from the same partition are paired, so paired fits share no rows. V2,j = ½ · mean over disjoint half pairs of d²_j and V4,j likewise for quarters. The convergence rate per window, pooled over features: β = log2(Σ_j V4,j / Σ_j V2,j). Floor at full n: σ²_R,j = V2,j · 2^(−β).
- *Evaluation:* on the three control arms only (real `shuf`, `surrogate:0` dir, `surrogate:0` vol), for BTCUSDT: median DR_tot at Δ = 12 over the 71 zero-overlap pairs, with the pre-registered 95% stationary-bootstrap CI. DR_tot = Σ_j d̄²_cross,j / Σ_j (σ²_a,j + σ²_b,j), every quantity computed on the pair's X_ref as before.
- *Adoption rule:* a candidate passes if its CI lies inside [0.8, 1.25] on all three arms. If exactly one passes, adopt it. If both pass, adopt the one with the smaller max over arms of |log median DR|. If neither passes, stop and report; no further candidate is tried without the user's decision.
- *Scope of the deviation:* the adopted floor replaces protocol (iii) in DR, DR_adj (σ²_samp = max(0, σ² − σ²_seed)), SNR, the seed share and the sign-agreement noise SD, for all primary analyses. Whether H1–H5 keep confirmatory status is the user's decision, recorded in a later entry.
- *Implementation:* `src/ebmregime/floorexp.py` (resampling) and `scripts/floor_experiment.py` (fits and evaluation), kept outside the fit-code hash of the runner so that the 6,225 cached slice fits stay valid during the experiment. If a candidate is adopted, it moves into `splits/` and the runner before M3.

**D-56. Floor-fix result (D-55): neither candidate passes; stopped per the adoption rule.** 6,474 candidate fits on the control arms (89 min wall, 0 errors; `data/cache/floor_exp/31eb6e7c1d9e`), evaluated by `scripts/floor_experiment.py eval` (`results/floor_exp/`). Real dir and vol targets were not touched.
- *G0 at Δ = 12 (median DR_tot over 71 zero-overlap pairs, 95% CI; band [0.8, 1.25]):*

  | arm | (iii) half-sample | B bootstrap | R rate-corrected |
  |---|---|---|---|
  | real shuf | 0.44 [0.28, 0.60] | 1.11 [0.78, 1.44] | 0.44 [0.24, 0.53] |
  | surrogate dir | 0.64 [0.49, 1.03] | 2.14 [1.25, 2.27] | 0.49 [0.39, 0.86] |
  | surrogate vol | 1.25 [1.15, 1.34] | 1.62 [1.58, 1.80] | 0.73 [0.67, 0.89] |

- *B* is about right where there is no signal (shuf: median 1.11, though the CI's upper end is 1.44), but too small where there is signal (surrogate dir 2.1, vol 1.6). The bootstrap replicates vary less than independent windows do. One possible cause, not tested: every replicate keeps the same validation tail, so early stopping is correlated across replicates.
- *R* overstates the floor on every arm. The per-window rate β = log2(ΣV4/ΣV2) is very noisy: 10th–90th percentile −0.54 to 3.17 (shuf), −1.50 to 2.94 (dir), −0.27 to 0.74 (vol). For the vol regressor it is about 0 (median 0.09): quarter-sample fits are barely noisier than half-sample fits, so the rate cannot be extrapolated to full n.
- *Status:* G0 is still unmet. The real-target drift results stay withheld. No further candidate is tried without the user's decision (D-55).

**D-57. DEVIATION (pre-specified before any new surrogate is fitted and before any real dir or vol drift result is examined): a surrogate-referenced floor replaces the within-window floor.** The user chose this route after D-56.
- *Why:* three within-window floors (half-sample, bootstrap, rate-corrected subsamples) failed in different directions (D-54, D-56). An early-stopped boosted EBM is not a smooth σ²/n estimator, so a resampling scheme inside one window does not predict how far two independent windows' shapes lie apart. A stationary surrogate runs the exact pipeline on data without regime change, so its cross-window distance contains every source of fit-to-fit variation (seeds, early stopping on the tail, binning, sample composition). This is the standard surrogate-data test.
- *Surrogates:* BTCUSDT, h = 8, indices 0–19 (20 in total; prereg `surrogates.per_symbol` was 3), generated by the frozen generator (`stability/surrogate.py`, mean block 252 h, prereg seed stream). Per surrogate and target (dir, vol, shuf): the 5 seed-only fits per window (protocol i, prereg replicate seeds), 83 windows. Half-sample fits are not needed. Surrogate 0's existing dir and vol fits are reused.
- *Statistic, per data set D (real or surrogate s), target and pair (τa, τb) at lag Δ:* g_D,j = d̄²_cross,j and G_D = Σ_j g_D,j, computed exactly as in the slice (pair X_ref, 50 cells, re-centred shapes, the seed-only fits of both windows). The seed term is S_D = Σ_j (σ²_seed,a,j + σ²_seed,b,j).
  - DR_sur(pair) = G_real / mean_s G_s; per feature, DR_sur,j = g_real,j / mean_s g_s,j. The adjusted version is DR_sur,adj = (G_real − S_real) / mean_s (G_s − S_s).
  - Summary at each Δ: the median over pairs with the prereg 95% stationary-bootstrap CI (as G0).
  - Rank test: T_D(Δ) = median over pairs of G_D; one-sided p = (1 + #{s : T_s ≥ T_real}) / 21.
  - At Δ < 12 the surrogate has the same window overlap as the real data, so short-lag γ needs no separate correction.
  - The per-window SNR becomes A²_j(τ) / σ²_sur,j(τ), with σ²_sur,j(τ) = ½ · the mean over surrogates and over the Δ = 12 pairs containing τ of g_s,j. A² comes from the window's own cells, the floor from the pair cells; this mismatch is accepted.
- *Checks; real dir and vol drift is computed only if (a), (b) and (c) all pass:*
  - (a) *Calibration, leave one surrogate out.* For each s in 0–19 and each target in {dir, vol}, compute DR_sur of surrogate s against the other 19. Passes if, for each target, (i) the median over s of the held-out median DR_sur at Δ = 12 lies in [0.9, 1.11], and (ii) at least 18 of the 20 held-out 95% CIs contain 1. Reported at every lag.
  - (b) *Negative control on real data (replaces G0).* Real `shuf`, referenced to `shuf` on all 20 surrogates. Passes if the 95% CI of the median DR_sur at Δ = 12 lies inside [0.8, 1.25] (the frozen G0 band).
  - (c) *Power, synthetic planted break.* The stationary synthetic process of `scripts/verify_floor_synthetic.py`, lengthened to 2019-01 to 2026-01 (vol target, 72 windows, τ from 2020-01 to 2025-12), with a break at 2022-07-01: the `rv_level_pct` effect flips sign (+0.3x to −0.3x, the strong change) and the `rv_ratio_4h` effect halves (0.4 tanh to 0.2 tanh, the moderate change). Every other feature is unchanged. 9 surrogates of the synthetic panel come from the same generator (252 h). Fits are the 5 seed-only fits per window. The "opposite-side" pairs are those at Δ ∈ {12, 24} whose two windows lie on opposite sides of the break, neither containing it. Passes if, on the opposite-side pairs, the 95% CI lower bound of the median DR_sur,j for `rv_level_pct` is above 1.25 (the H1 threshold) and that of DR_sur,tot is above 1.
  - Reported, not gating: the moderate feature on the opposite-side pairs; the unchanged features; same-side pairs at Δ = 12. The surrogate of a series with a break mixes both regimes, so its windows vary more than either regime alone. The reference is therefore conservative under real non-stationarity, and the same-side pairs measure how conservative.
- *Exploratory, not eligible for adoption:* bootstrap floor B with a resampled validation tail ("B-tail"). Tail rows are repeated according to stationary-bootstrap counts (mean block 168 h) drawn over the tail, every copy is −1 in all bags, and training rows are as in B. 10 fits per window on the three control arms; median DR at Δ = 12 is reported against the D-56 table. This tests whether B's under-dispersion comes from the shared tail (D-56). It cannot replace D-57.
- *Later sensitivity check (not run now):* the surrogate mean block at 1 and 3 months (720 h, 2,160 h), 9 surrogates each, dir and vol.
- *Scope:* if the checks pass, DR_sur replaces DR wherever PLAN uses DR (γ(Δ), DR over time, H1 thresholds, SNR, the seed share). The 252-h surrogate defines "no regime change": the claim is drift beyond a stationary process with dependence up to about 10 days. Whether H1–H5 stay confirmatory (under an amended pre-registration with disclosed deviations) is the user's decision.
- *Code:* lead-owned `src/ebmregime/surfloor.py` and `scripts/surrogate_floor.py`, outside the fit-code hash; fits run through the runner cache.

**D-58. DEVIATION (written before any surrogate result was computed): the D-57 gates use the surrogates' own null distribution, not the bootstrap CI.**
- *Finding (simulation only, no project data):* the prereg CI for a median (stationary bootstrap, block floor max(3, Δ), percentile) under-covers badly on the lag-12 pair series:
  - Two pairs share a window up to 2Δ = 24 months apart, so 71 pairs from 95 months carry about 4 independent pieces of information.
  - `scripts/ci_coverage_sim.py` builds the same overlap structure: 30-dimensional monthly shocks, 12-month window means, pair G = squared distance, ratio to the mean of 20 independent references, 200 replications. 95% intervals covered the true median 76% of the time with the prereg block and 70% with the pre-registered sensitivity block (×2).
  - On i.i.d. series the coverage was 90% (n = 71); on AR(0.9) series it was 69%.
  - Under the null, the median of a ratio of skewed distances also sits slightly below 1 (0.96–0.985 in these simulations).
  - So D-57 gate (a)(ii) ("at least 18 of 20 CIs contain 1") would test the bootstrap, not the floor. The prereg CIs of G0 and H1 are too narrow in general (anti-conservative). Fixing that for H1–H5 is the user's decision; this entry changes only the D-57 gates.
- *Null distribution:* m = median over pairs of DR_sur. For each surrogate s, m_s is the median of surrogate s's DR_sur against the other 19 (leave one out). The 20 values m_s form the null distribution of m, already including its small downward bias. Inference compares an observed m with them, and does not need a bootstrap.
- *Revised gates (replace D-57 (a)–(c); the bootstrap CIs are still reported alongside):*
  - (a) For dir and vol at Δ = 12, the median of the 20 m_s lies in [0.9, 1.11] (D-57 (a)(i), kept). (a)(ii) is dropped: coverage by exchangeable held-out surrogates is guaranteed by construction, so it tests nothing.
  - (b) Real `shuf` at Δ = 12, against `shuf` on all 20 surrogates. Three conditions must all hold:
    1. m_real lies within [min, max] of the 20 shuf m_s (two-sided rank test, size about 2/21);
    2. m_real lies in [0.8, 1.25];
    3. precision: the 2.5% and 97.5% empirical quantiles of the 20 shuf m_s lie within [0.8, 1.25], which is the analogue of "CI inside the band".
  - (c) Synthetic break, opposite-side pairs. The null values are the leave-one-out m of the 9 synthetic surrogates (each against the other 8) on the same pairs. The strong feature's m must exceed both 1.25 and the largest null value, and the total m must exceed both 1 and the largest null value.
- *For later analyses (proposal, the user's decision):* H1-type inference by the same rank comparison against held-out surrogates, which needs more surrogates per symbol (39 for a one-sided 2.5% test).

**D-59. Surrogate-floor result (D-57/D-58): checks (a) and (b) fail; the surrogate generator leaks training rows into the validation tail.** 24,070 seed-only fits on surrogates 0–19 (0 errors), 34,282 control pairs (`results/surfloor/`). Real dir and vol drift not computed.
- *Gate (a), calibration (m = median over the 71 lag-12 pairs of DR_sur, leave one surrogate out):*
  - vol: median of the 20 m_s = 0.96 (2.5–97.5%: 0.81–1.14). Passes.
  - dir: median of the 20 m_s = 0.48 (0.31–1.21). Fails.
  - Across lags, vol m_s rises from 0.64 (Δ = 1) to about 0.97 (Δ ≥ 12); dir stays at 0.34–0.53.
- *Gate (b), real shuf against 20 surrogates:* m_real = 0.17, below every null value (null 0.21–1.63, median 0.36), outside the band, and the null is far wider than [0.8, 1.25]. Fails. The pattern holds at every lag.
- *Diagnosis (control arms only, `scripts/surrogate_leakage_diag.py`):*
  - Across surrogates, classifier distances are extremely heavy-tailed. At a given pair, the largest surrogate distance is typically 10× the median (90th percentile 26× for dir, 51× for shuf); for vol it is 1.9×.
  - Real shuf's median distance (0.0054) lies below every surrogate's (0.0060–0.031). Every feature is affected.
  - *Cause:* the surrogate draws 252-h blocks with replacement from the whole 8-year range. A 12-month window's validation tail can therefore contain source rows that also sit in its training part, or within 8 h of them (overlapping 8-h labels). Every surrogate window has some such tail rows (median 7% of the tail). Real windows have none, because of the purge.
  - Early stopping on a tail that repeats training rows rewards memorising: when more than 30% of the tail is near-duplicate, boosting rounds rise (3,270 against 2,820) and the lag-12 distance is 3–9× larger. With little leakage (under 10%), surrogate shuf distances are 1.2–2× real shuf.
  - The noise-dominated classifiers are hit hardest; the signal-dominated vol regressor much less (rank correlation of leakage with distance: 0.36 shuf, 0.43 dir, 0.20 vol).
- *Consequences:*
  - (i) The frozen surrogate is not a valid null for early-stopped fits. Its windows are leakier than real windows by construction.
  - (ii) This also contaminates every earlier surrogate-arm result: G0's surrogate arms (D-54) and candidates B and R on the surrogate arms (D-56). On the one uncontaminated control, real shuf, candidate B gave 1.11 [0.78, 1.44].
  - (iii) Gate (a) for vol passing shows the leave-one-out machinery works where leakage matters little.
- *Status:* stopped. No new surrogate generator or floor is tried without the user's decision. Checks (c) and the exploratory B-tail run as pre-specified and are reported when done; (c) uses the same with-replacement generator on synthetic data.
- *Check (c), power on the synthetic break (3,600 fits, `results/surfloor/check_c_power.csv`): passes.*
  - On the 14 opposite-side pairs, the strong change gives m = 30.9 for `rv_level_pct` (null max 3.2) and 6.3 in total (null max 1.28).
  - The moderate change (`rv_ratio_4h` effect halved) is also detected: 1.84 against a null max of 1.47.
  - The unchanged features give 0.23–1.24, every one below its null max.
  - On the 37 same-side pairs, total m = 0.72 (null 0.80–1.17): the reference is conservative, as expected for surrogates that mix both regimes and, here, also leak. One of the 16 features (`taker_imb_4h`, 1.44) exceeds its null max of 1.25, consistent with a 9-surrogate null.
  - The method has power, but this check does not rescue (a) and (b).
- *Exploratory B-tail (2,490 fits, `results/floor_exp/g0_Btail.csv`; not eligible for adoption):* median DR at Δ = 12 is 0.44 [0.35, 0.69] for real shuf, 1.06 [0.70, 1.24] for surrogate dir and 1.39 [1.30, 1.45] for surrogate vol.
  - Resampling the validation tail makes the replicates much noisier: on real shuf the floor grows about 2.5× over B, which gave 1.11.
  - The tail realisation that early stopping sees is a large source of fit-to-fit variation. A fixed tail understates it; a bootstrapped tail with repeated rows overstates it.
  - The surrogate-arm numbers carry the D-59 leakage and are not interpreted.

**D-60. DEVIATION (pre-specified before any fit with it): a leakage-free block-permutation surrogate replaces the with-replacement surrogate for the D-57/D-58 checks.** The user approved the rebuild after D-59.
- *Generator ("perm"):* the rows with `start <= ts < end` (as in D-50; BTCUSDT `[2018-12-01, 2026-11-01)`) are cut into consecutive source blocks. Block lengths are geometric with mean 252 rows (the prereg mean block), drawn until the range is covered, with the last block truncated. The blocks are put in a uniformly random order without replacement and laid end to end on the original timestamps, so every source row appears exactly once.
  - *Purge:* for every block except the one that ends at the end of the range, all `y_*_h{H}` targets of its last H source rows are set to NaN (H = 8 and 24), so no label window reaches into another block. In the synthetic check (c) the same rule applies to `y_vol` with H = 8.
  - Seed: `stable_seed(prereg.surrogates.seed, symbol, "perm_surrogate", i)`; synthetic `"SYNTH_BREAK"`, `"perm_surrogate"`, i.
  - *Property:* no validation-tail row of any window is within H source rows of a training row from another block. A unit test asserts this.
- *Fits:* surrogates perm:0–19, targets dir, vol and shuf, 5 seed-only fits per window (prereg replicate seeds), all 83 windows. They run through the unchanged runner fit function (`execute.run_fit_job`) with the surrogate panel swapped for the perm panel, in a separate cache (`data/cache/fits_perm`). The key is the normal key with the code hash extended by the perm module's hash. The frozen runner files, and so the cached real and slice fits, are unchanged.
- *Statistic, checks and gates:* identical to D-57 as amended by D-58: (a) calibration for dir and vol, (b) real shuf against the 20 shuf surrogates, (c) power on the synthetic break with 9 perm surrogates of the synthetic panel. Real dir and vol drift is computed only if all three pass. The leakage diagnostic of D-59 is rerun on the perm surrogates.
- *Not run now:* re-testing candidate B against the perm surrogates (exploratory only, D-59 (ii)) and the block-length sensitivity check (D-57).

**D-61. Perm-surrogate result (D-60): leakage removed, real shuf now exchangeable with the null, but gates (a) and (b) still fail as written; (c) passes.** 24,898 seed-only fits (0 errors), 34,282 pairs, 3,590 synthetic fits (`results/surfloor/perm_*`). Real dir and vol drift not computed.
- *Leakage, corrected measure* (`surfloor_perm.leakage`; all 20 surrogates × 83 windows × 3 targets):
  - The measure counts tail rows that copy a fitted training row, or share label hours with one (fewer than H source rows apart). Fitted rows exclude those purged before the tail, and the tail start is computed as in the fit.
  - perm generator: no leakage in any window.
  - Frozen generator: duplicates in 71% of windows; label overlap in 72%, median 6.6% of the tail, up to 70%.
  - *Correction to D-59:* its diagnostic also counted the window's own purged boundary rows, so "every surrogate window" and "median 7%" are replaced by these figures. The conclusions of D-59 stand.
- *Fit cost:* the perm fits used 133k CPU-seconds against 472k for the frozen generator's: leaky tails made early stopping run longer.
- *Gate (a), lag 12, median of the 20 leave-one-out m_s:*
  - vol: 0.96 (2.5–97.5%: 0.84–1.15). Passes.
  - dir: 0.73 (0.41–1.03). Fails; it was 0.48 with the frozen generator.
- *Gate (b), real shuf:*
  - m_real = 0.44, inside the null range 0.29–0.93 (rank 13 of 21; ranks 12–18 at every lag; with the frozen generator it was below every null value). So condition 1 passes.
  - Conditions 2 (in [0.8, 1.25]) and 3 (null 2.5–97.5% inside the band) fail.
- *Why the gates still fail:* across surrogates, classifier distances remain right-skewed. At a given pair the largest surrogate distance is 5–7× the median (frozen generator: 10–12×), and the mean is 1.4–1.7× the median (vol: 1.02).
  - DR_sur divides by the surrogates' mean, so every classifier ratio, the null's included, centres well below 1 (shuf null median 0.53). The fixed bands of D-58 assumed a centre near 1.
  - Real shuf / median surrogate = 0.83. This is a property of the statistic, not of the null.
  - The classifier null is also wide (factor about 3 between its 2.5% and 97.5% quantiles): with 5 seed fits and 12-month windows, drift in noise-dominated shapes is imprecise.
- *Gate (c), power on the synthetic break:* passes.
  - Strong change: m = 26.2 (null max 3.4). Total: 7.3 (null max 1.32). Moderate change: 2.31 (null max 1.76).
  - No unchanged feature on the opposite-side pairs exceeds its null max; 1 of 17 on the same-side pairs does.
- *Status:* stopped; the gates are not changed without the user's decision. Any change now would be made after seeing control-arm results only (real dir and vol never computed).

**D-62. DEVIATION (pre-specified after the control-arm results of D-61 and before any real dir or vol drift result): the D-58 gates use a pooled rank statistic and a median-referenced ratio.** The user approved amending the gates after D-61.
- *Why:* D-61 showed that the gates failed because of the statistic, not the null. Classifier distances are right-skewed across surrogates, so a ratio to the surrogates' *mean* centres below 1 even on exchangeable data. A rank is free of this bias, and a ratio to the *median* is robust to it. The frozen G0 fallback already referenced the median surrogate (`hypotheses.g0.fallback`).
- *Disclosure:* this is chosen after seeing control-arm results only. Known before writing: real shuf / median surrogate ≈ 0.83 at Δ = 12 (D-61), and real shuf ranks 12–18 of 21 at every lag. Not known: any of the statistics below for dir, vol or the synthetic break, and every real dir or vol result.
- *Pool:* the data sets compared together. Calibration (a) and precision (d): the 20 perm surrogates. Real targets (b) and later: real plus the 20 perm surrogates (21). Power (c): the synthetic panel plus its 9 perm surrogates (10).
- *Statistic, for each data set D in a pool, target and lag Δ* (G = Σ_j g_j as in D-57; per feature, g_j in place of G):
  - Per pair p: u_D(p) = (#{E ≠ D : G_E(p) < G_D(p)} + ½ #{ties}) / (k − 1), with k the pool size, the mid-rank of D among the other members. r_D(p) = G_D(p) / median_{E ≠ D} G_E(p).
  - Ū_D = mean over pairs of u_D(p) (rank statistic; 0.5 in expectation under exchangeability). M_D = median over pairs of r_D(p) (effect size, "DR_med").
  - Every member is referenced to the other k − 1, so under the null (no drift beyond the surrogate) all k values are exchangeable. The test of D is exact: p_upper = #{E : Ū_E ≥ Ū_D} / k, rank from the top among the k.
  - Seed-adjusted versions (G − S) are reported, not gating. The prereg bootstrap CI is no longer reported (D-58: it under-covers).
- *Gates (replace D-58 (a)–(c); real dir and vol drift is computed only if (a), (b) and (c) all pass):*
  - (a) *Calibration.* For dir and vol at Δ = 12, the median of the 20 surrogate M_s lies in [0.9, 1.11] (band unchanged). The rank statistic needs no calibration check: its centre is exact by construction.
  - (b) *Negative control.* Real shuf in the 21-member shuf pool at Δ = 12. Passes if (1) Ū_real lies within [min, max] of the 20 surrogates' Ū (two-sided rank test, size about 2/21) and (2) M_real lies in [0.8, 1.25] (band unchanged).
  - (c) *Power.* Synthetic break, opposite-side pairs, 10-member pool. Passes if, for `rv_level_pct` and for the total, the synthetic panel's Ū is the largest of the 10, and M_synth exceeds both the largest surrogate M and the threshold (1.25 for `rv_level_pct`, 1 for the total).
- *Precision (d), per target, not a stop rule:* a target is precise if the 2.5% and 97.5% quantiles of its 20 surrogate M_s at Δ = 12 lie within [0.8, 1.25] (D-58 (b)(3), moved from shuf to each target). A target that is not precise cannot separate drift of 1.25 from the null. Its later drift results are exploratory, reported with its null range. Real shuf's precision is reported.
- *Reported, not gating:* every lag, the seed-adjusted versions, and per-feature Ū and M; for (c), the moderate feature, the unchanged features and the same-side pairs, as in D-57.
- *Scope:* for later analyses, M replaces DR_sur as the effect size and the pooled rank replaces the bootstrap CI for inference. With 21 members, the smallest attainable one-sided p is 1/21 ≈ 0.048. Whether H1–H5 stay confirmatory, and whether more surrogates are fitted for a finer p (D-58), is the user's decision.
- *Code:* `surfloor.pooled_stats`, `check_calibration_rank`, `check_negative_control_rank`, `surfloor_power.summarise_rank`; outputs `results/surfloor/perm_rank_*`. The D-58 functions and outputs stay as they are.

**D-63. Amended-gate result (D-62): gates (a), (b) and (c) pass; vol is precise, dir is not.** Computed on the stored perm pairs of D-61 (no new fits; `scripts/surrogate_floor.py check-rank`, `results/surfloor/perm_rank_*`). The D-62 text was hashed before this run (sha256 `715587149a37…`, 2026-11-19T06:08Z). Real dir and vol drift not computed.
- *Gate (a), Δ = 12, median of the 20 surrogate M_s:* dir 1.05, vol 0.98. Both pass. Every lag lies in 0.94–1.06 for every target.
- *Precision (d), 2.5–97.5% of the 20 surrogate M_s at Δ = 12:*
  - vol: 0.87–1.17. Precise.
  - dir: 0.62–1.53. Not precise, so dir drift results are exploratory.
  - shuf: 0.53–1.78. Not precise.
- *Gate (b), real shuf in the 21-member pool, Δ = 12:* passes.
  - Ū_real = 0.49, within the surrogates' 0.40–0.61. Rank 12 of 21 (p_upper = 0.57).
  - M_real = 0.83, in [0.8, 1.25] (known approximately before D-62, as disclosed there).
  - At every lag, real shuf ranks 11–19 of 21. M_real is 0.72–0.87, below the surrogate median at every lag, but always inside the surrogate range.
- *Gate (c), synthetic break, opposite-side pairs, 10-member pool:* passes.
  - Strong change: Ū = 1 (rank 1); M = 52.5 against a surrogate max of 4.0.
  - Total: Ū = 1 (rank 1); M = 8.3 against 1.43.
  - Moderate change: rank 1; M = 2.57 against 2.00.
  - The 14 unchanged features: none ranks first by Ū; one (`pos_range_168h`, M = 2.42 against 2.28) exceeds its surrogate max by M.
- *Caveat on per-feature results, same-side pairs (Δ = 12, 37 pairs, no change inside any pair):*
  - Total: rank 10 of 10, M = 0.73 below the surrogate min of 0.80. The reference is conservative, as D-57 expected.
  - But 5 of the 16 features rank first by Ū (chance: about 1.6), each only just above the surrogate max (Ū 0.61–0.66 against 0.60–0.65). By M, 1 of 16 does (`taker_imb_4h`).
  - So per-feature rank results can flag unchanged features when the real series has a regime change elsewhere. A plausible cause: surrogates that mix regimes fit a weaker main effect, which changes early stopping and so the noise in every other shape. This is one synthetic series and is not tested further.
  - Per-feature Ū results on real data are therefore exploratory, reported with M.
- *Status:* by D-62, real dir and vol drift may now be computed (`scripts/surrogate_floor.py real-rank`; every real seed fit is already cached). It has not been run. vol is eligible for confirmatory status; dir is exploratory by (d). Whether H1–H5 stay confirmatory is the user's decision.

**D-64. Real dir and vol drift under the D-62 gates (first look at any real dir or vol drift result beyond the D-54 printout).** Run after D-62/D-63 were committed (`d4230a0`). 1,124 real pairs from the cached seed fits (`scripts/surrogate_floor.py real-rank`, `results/surfloor/perm_rank_real*.csv`). Pool: real plus the 20 perm surrogates. The smallest attainable p is 1/21 = 0.048.
- *vol (precise, D-63):*
  - Δ = 12: M = 1.31, rank 1 of 21 (p = 0.048), above every surrogate (max 1.17). Ū = 0.67 against a surrogate max of 0.63. The seed-adjusted version is the same (1.31).
  - Drift grows with lag: M = 0.98 at Δ = 6 (rank 11), 1.47 at Δ = 18, 1.48 at Δ = 24, 2.14 at Δ = 36. Every lag from 12 up ranks first.
  - At Δ = 1–3, real vol ranks last or near last (M = 0.84–0.88, at or below the surrogate minimum). Nearby windows are more alike than in any surrogate. On the synthetic panel the same-side pairs showed the same pattern (D-63). Surrogates that mix regimes are noisier than either regime, so the reference is conservative and M understates drift.
  - *Against H1 as frozen:* the point estimate 1.31 is just above the 1.25 threshold, but D-62 gives no interval. So "CI lower bound above 1.25" cannot be assessed. With the Holm family of six, p = 0.048 cannot reach 0.05/6 ≈ 0.0083, which would need at least 120 pool members.
- *dir (exploratory, not precise):* no drift beyond the surrogates at any lag. At Δ = 12, M = 0.87 (rank 17 of 21); across lags, M = 0.80–1.00 (ranks 8–19). A dir drift of 1.25 cannot be separated from the null (surrogate range 0.53–1.51). The prereg SNR condition for dir (`h1.dir_requires_snr_above_shuffled`) was not evaluated.
- *Per feature, Δ = 12 (exploratory, D-63 caveat):*
  - vol, rank 1 of 21: `hour` (M = 2.27 against a surrogate max of 1.34), `xret_4h` (1.75 against 1.19), `pos_range_168h` (1.77 against 1.64), `rv_level_pct` (1.34, below its max of 1.71). That is 4 of 16, against about 0.8 by chance; the synthetic panel showed 5 of 16 on unchanged features.
  - The three that exceed their surrogate max by M as well are the most credible. The largest is the intraday (`hour`) profile of volatility.
  - dir, rank 1 of 21: only `ret_4h` (M = 2.32 against 1.50), about what chance gives.
- *Status:* vol drifts beyond a stationary 252-h block surrogate at lags of 12 months and more, and dir does not. Confirmatory status of H1–H5, the H1 interval rule under D-62, and more surrogates for a Holm-level p are the user's decisions.

**D-65. DEVIATION (pre-specified before any new fit): 100 more perm surrogates for vol, so the pooled rank test can reach the Holm level.** The user approved this after D-64.
- *Disclosure:* this is decided after seeing D-64, where real vol ranked first of 21 at Δ ≥ 12. Extending the null after a minimum-p result is a sequential choice. So the 21-member result of D-64 stays on record, and the extended result is reported whatever it shows. No other extension follows from it.
- *Fits:* perm surrogates 20–119 (indices as in D-60, same generator, seed stream and cache), vol only, 5 seed-only fits per window, all 83 windows (41,500 fits). Pairs at every prereg lag, as in D-61. Code: `src/ebmregime/surfloor_ext.py`; `surfloor_perm.py` is not changed, so its cache keys stay valid.
- *Pool and statistic:* D-62 unchanged, with real plus surrogates 0–119 (121 members). The smallest attainable p is 1/121 ≈ 0.0083, at or below the Holm bound for the smallest p in a family of six (0.05/6 ≈ 0.0083).
- *Conditions, Δ = 12, using the 120 surrogates:*
  - D-62 (a): the median of the 120 vol M_s lies in [0.9, 1.11]. If this fails, the 121-member result is reported but not used.
  - D-62 (d): precision is recomputed and reported.
  - Gates (b) and (c) are unaffected (shuf and synthetic).
- *Reported:* U, M, rank and p_upper for real vol at every lag, plain and seed-adjusted, and per feature at Δ = 12 (exploratory, D-63). Whether p at Δ = 12 meets Holm within the H1–H5 family, and the H1 interval rule, remain the user's decisions.

**D-66. Extended-null result (D-65): real vol drift is significant at the Holm level at Δ ≥ 18, but not at the pre-registered Δ = 12 (p = 0.033).** 41,500 vol fits on perm surrogates 20–119 (0 errors; 11.6 h wall, about 5 h of it with the machine asleep) and 56,200 pairs (`results/surfloor/perm_ext_*`). Pool: real plus surrogates 0–119 (121 members); the smallest attainable p is 1/121 = 0.0083.
- *Condition (a), 120 surrogates, Δ = 12:* median M_s = 0.99 (2.5–97.5%: 0.86–1.17). Passes.
- *Precision (d):* 0.86–1.17, within [0.8, 1.25]. Precise.
- *Real vol, Δ = 12 (H1 lag):*
  - M = 1.30, Ū = 0.65; rank 4 of 121, so p_upper = 0.033. Three surrogates have a larger Ū, and the largest surrogate M is 1.29.
  - With 21 members (D-64) it ranked first. The extended null resolves it to 0.033, which does not reach 0.05/6 ≈ 0.0083.
  - The seed-adjusted version is the same (M = 1.30, rank 4).
- *Other lags:*
  - Δ = 18, 24, 36: M = 1.39, 1.46, 2.03; every one ranks 1 of 121 (p = 0.0083).
  - Δ = 6: M = 0.95 (rank 57).
  - Δ = 1–3: M = 0.83–0.88, ranks 111–121. Real windows close in time are more alike than surrogate windows: the conservative reference of D-64.
- *Per feature, Δ = 12 (exploratory, D-63 caveat):*
  - `hour`: M = 2.31, rank 1 (p = 0.0083), above every surrogate (max 1.95).
  - `pos_range_168h`: 1.70, rank 3 (p = 0.025).
  - `xret_4h`: 1.59, rank 4 (p = 0.033).
  - `rv_level_pct`: 1.34, rank 7 (p = 0.058).
  - Every other feature ranks 23 or below.
- *Reading:* vol drift beyond the stationary 252-h block surrogate grows steadily with lag. It is clear from 18 months on. At 12 months it is borderline: nominally significant (p = 0.033), with point estimate M = 1.30 just above the H1 threshold of 1.25.
- *Status:* as frozen, H1-vol is tested at Δ = 12. With Holm over six hypotheses it is not significant, unless the family or the lag is changed, which would be a post-hoc deviation. The confirmatory reading of H1–H5 is the user's decision. dir stays exploratory (D-64).
