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

**D-06. Cleaning rules.**
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

**D-12. Interactions are studied in a separate arm with `interactions=5`. Pairs are purified before pair-shape comparison.**
- *Reason:* the default "3x" means 42 pairs for 14 features, which dilutes everything and costs about 85 s per fit.
- *Evidence [verify]:* on 0.7.8, main-effect `term_scores_` are bit-identical (max abs diff 0.0) between interactions = 0 and 5 with the same seed and bags. Pairs are fitted after the mains, with the mains frozen. Only the intercept moves, by about 3e-4. So the pair arm cannot contaminate the primary analysis.
- Pairs are only mean-centred by default (`purify_result=False` [method, source-read]), hence explicit purification [lit: Lengerich et al. 2020].

**D-13. Internal validation uses time-ordered `bags`. Layout: last 15% of the window is validation, h-row purge before it, per-bag 85% of contiguous 168-row training blocks. Array shape (n_samples, outer_bags).**
- *Reason:* EBM's default random 15% validation split leaks with overlapping labels, makes early stopping overrun, and inflates instability.
- *Correction [verify]:* the method draft read the GitHub develop branch and claimed the shape must be (outer_bags, n_samples). On the pinned 0.7.8 that shape raises `ValueError: bags has 8760 bags and self.outer_bags is 14 bags`, while (n_samples, outer_bags) works, matching the documentation.
- This behaviour is pinned in `tests/test_interpret_api.py`.
- At least one −1 row is required, otherwise early stopping is disabled [method, source-read].

## Metrics

**D-14. Primary rank metric: weighted Kendall τ, with hyperbolic top-weighting.**
- *Reason:* the bottom of a 14-feature ranking is near-zero noise, and plain Spearman or Kendall penalise churn there as much as at the top. Weighted τ handles ties and is one library call.
- RBO is a sensitivity check [lit: Webber et al. 2010].

**D-15. Shapes are compared as cell-averaged contributions on 50 pooled equal-probability cells of a pair-specific reference sample, re-centred under that common density.**
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

**D-20. Regimes: R1 = trailing 30-day RV percentile within the symbol's expanding history (primary); R2 = 90-day trend t-statistic; R3 = Amihud liquidity percentile. The full-sample HMM is descriptive only.**
- *Reason:* the three are simple, causal and distinct. Volatility is the most persistent and best-measured state.
- A full-sample HMM uses the future, so it lives in a separate module and every figure that uses it is labelled.

**D-21. Early-warning test.**
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
