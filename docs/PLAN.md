# Research and implementation plan

**Project:** Do glass-box explanations survive regime change?
**Status:** draft for approval (2026-11-08). No implementation has started.
**Revision 1 (2026-11-08)** incorporates your answers to O-1 to O-7:
- 1h bars with an **8-hour** horizon;
- normalised vol target;
- symbols **BTC, ETH, XRP, SOL**;
- taker-buy imbalance and a cross-asset return feature added;
- pre-registration on git plus Zenodo;
- licensing and compute as proposed.

The changes are logged as D-33 to D-38.
**Companion files:**
- [`DECISIONS.md`](DECISIONS.md): numbered design decisions, cited below as D-xx.
- [`LITERATURE.md`](LITERATURE.md): verified references.

**Provenance of the evidence.**
- Facts about the data were measured against the live archive on 2026-11-08 (§4).
- Compute numbers were measured on this laptop (§11).
- API behaviour was checked against `interpret` 0.7.8 by running code (D-13, D-12).

Where a sub-agent's draft was wrong, the correction is logged in DECISIONS.

---

## 1. Summary

An Explainable Boosting Machine (EBM) is a generalised additive model with optional pairwise terms. It is often presented as showing *what the model learned about the market*. We test whether that claim holds when the model is retrained over time.

**Design.**
- Fit EBMs on rolling 12-month windows of hourly Binance spot data for four liquid pairs: BTC, ETH and XRP from 2018-06, and SOL from 2020-09, all through 2026-10. That gives 83 retrains per symbol, and 56 for SOL.
- Hold the features and hyperparameters fixed.
- Measure how much the explanations move between windows. The explanations are the shape functions, term importances and selected pairs.

**Key methodological choice.** Every comparison between windows is expressed relative to a measured noise floor. The floor is the spread between retrains on the *same* window under different seeds and resamples. The headline statistic is a drift ratio:
- about 1 means stable up to retraining noise;
- above 1 means real temporal change.

The drift ratio is calibrated on stationary surrogate data and on a shuffled-target control.

**Three targets** separate "the market changed" from "nothing was learned":
- forward return direction (near-noise);
- forward realised volatility, normalised (learnable positive control);
- a time-shifted direction target (no signal by construction).

**Further questions.**
- Is drift concentrated where the volatility, trend or liquidity regime changes?
- Does drift *lead* out-of-sample decay?
- Is the effect specific to EBM? Comparators are logistic regression and XGBoost with interventional TreeSHAP.

**Cost and staging.** The confirmatory grid costs about 3.6 wall-clock hours on this laptop (about 5 h with margin). The secondary arms add about 4.7 h. A first slice runs end to end in under an hour before the rest is built: BTCUSDT, one horizon, EBM only, noise floor plus drift.

---

## 2. Research questions and pre-registered hypotheses

### Questions

- **Q1 (main).** Are EBM explanations temporally stable across rolling retrains, beyond retraining noise?
- **Q2 (noise floor).** How large is within-window retraining variability? How much of it is the algorithm (seed) and how much is the data (sampling)?
- **Q3 (regimes).** Is explanation drift concentrated where the volatility, trend or liquidity regime changes?
- **Q4 (early warning).** Does explanation drift at a retrain predict out-of-sample decay of that model? Does it do so beyond regime, vol-level and covariate-shift controls?
- **Q5 (EBM-specific?).** Do logistic regression and XGBoost+SHAP, trained on the same windows, drift more, less or the same?
- **Q6 (positive control).** Does a learnable target give high explanation signal-to-noise and interpretable drift, while direction and shuffled targets do not?

### Pre-registration rule

This section, §3 and §9 (with `configs/prereg.yaml`) are frozen before the first slice touches real data (D-37):
- an annotated git tag `prereg-v1`;
- an **embargoed Zenodo record**. You deposit the frozen files; the record shows their SHA-256 hashes publicly; the DOI and timestamp are public immediately; the files open on a date you choose.

A git history alone can be rewritten, so the Zenodo record is the timestamp that an outside reviewer can trust.

After the freeze:
- The slice (§13, M2) may only trigger the pre-specified fallbacks listed in §9.7.
- Any other change to hypotheses, primary metrics, the target, horizon, features or tests is logged in DECISIONS as a **deviation**. The affected hypothesis becomes exploratory.
- Everything not listed below is exploratory.

### Scope of the primary analysis

Unless stated otherwise, the primary analysis uses:
- EBM with `interactions=0`;
- 1h bars, primary horizon h = 8 bars (8 hours);
- primary regime R1;
- all 4 symbols pooled as an unbalanced panel (SOL enters from 2022-03), with symbol treated as a blocking factor.

### Hypotheses

**G0: validity gate (not counted as a hypothesis).**
- Statistic: median drift ratio at Δ = 12 months for the shuffled-target control and for the stationary surrogates (§9.3).
- Pass: 95% CI inside [0.8, 1.25].
- If it fails, the pre-specified fallback applies: use the surrogate-calibrated ratio (§9.3).
- If even the calibrated ratio fails the gate, stop and fix the floor before looking at any real-target result.

**H1: temporal drift beyond noise.** Tested separately for the vol target (H1-vol) and the direction target (H1-dir).
- Statistic: median over t of DR_tot(M_t, M_{t+12}) at zero data overlap.
- Decision:
  - CI lower bound above 1.25: **drift**;
  - CI inside [0.8, 1.25]: **stable up to retraining noise** (an equivalence result). If that CI also excludes 1, the result is reported as "detectable but practically negligible drift";
  - otherwise: **inconclusive**.
- The margin of 1.25 on the drift side is set for symmetry with the equivalence margin (D-16).
- H1-dir is only interpreted if direction-target explanations carry signal, i.e. the median shape SNR for direction exceeds the shuffled control (CI excludes 0). If they do not, H1-dir is reported as **vacuous**: there was never a signal to be stable.

**H2: algorithmic share of the noise floor.**
- Statistic: median seed share σ²_seed / σ²_tot for EBM main-effect shapes (§6), on the vol target.
- Decision:
  - CI upper bound below 0.25: a single default fit is algorithmically reliable;
  - CI lower bound above 0.5: retraining noise alone matters to practitioners;
  - otherwise: report the estimate.

**H3: drift concentrated at regime change.**
- Model, on consecutive retrains (Δ = 1): log DR_t = a + b·C_t + c·log σ_τ + symbol FE + ε.
- C_t is the change in high-vol fraction between the month added and the month dropped (§8).
- Decision:
  - b > 0 with Holm-adjusted p < 0.05: **concentrated**;
  - CI on b containing 0 with upper bound below 0.25 SD(log DR) per unit C: **not concentrated**;
  - otherwise: inconclusive.

**H4: early warning.**
- Model: ΔSkill_t = a + b·D_t + controls (§9.5). Vol target.
- Decision:
  - b < 0, with block-bootstrap CI excluding 0 **and** placebo-shift p < 0.05: **early warning**;
  - 90% CI upper bound on |partial r| below 0.2: **no useful early warning**;
  - otherwise: **inconclusive**.
- This is an estimation hypothesis, honest about power: n_eff is about 25–35 per symbol.

**H5: EBM-specificity.**
- Statistic: paired-by-t difference in log DR_tot(Δ = 12) between EBM and XGB-SHAP, and between EBM and LR. Vol target.
- Decision:
  - CI excluding 0: model-specific;
  - CI inside ±log 1.25: drift is a property of the data, not of EBM.
- Must agree with the SNR comparison (pitfall: a model with a larger floor mechanically gets a smaller DR).

**Multiplicity.** H1-vol, H1-dir, H2, H3, H4 and H5 form one family under Holm, FWER 0.05. Everything else uses BH-FDR q = 0.10 and is labelled exploratory.

**Expectations, written down so we can be surprised.**
- Vol-target shapes have high SNR; hour-of-day and vol-ratio shapes are stable (DR near 1).
- Direction-target shapes have SNR close to the shuffled control.
- If H4 holds, it holds more for gradual than for abrupt regime changes, following Szandała & Raczkowski 2026.

---

## 3. Primary and secondary metrics

All metrics are defined precisely in §6.

| Role | Metric | Used in |
|---|---|---|
| **Primary** | DR_tot: noise-floor-normalised, density-weighted main-effect shape distance, summed over terms | H1, H3, H5, G0 |
| **Primary** | Seed share σ²_seed / σ²_tot | H2 |
| **Primary** | Out-of-sample skill: R²_OOS of log normalised RV against training climatology (vol); log-loss skill against base rate (direction) | H4 |
| Secondary | Per-feature DR_j; drift variogram γ(Δ) for Δ ∈ {1,2,3,6,12,18,24,36} months | Q1 |
| Secondary | Shape amplitude A_j and SNR_j | always reported next to stability metrics |
| Secondary | Shape correlation ρ_j; sign agreement SA_j; monotonicity agreement MA_j | Q1 |
| Secondary | Importance stability: weighted Kendall τ (primary rank metric); Spearman; top-3 Jaccard; TV distance of importance shares | Q1, Q5 |
| Secondary | Interaction stability: Jaccard of selected pair sets; per-pair selection frequency π; Pearson correlation of π vectors | interaction arm |
| Secondary | Prediction distance d²_pred; feature-family distance | separating explanation drift from prediction drift and Rashomon shuffling |
| Secondary | Brier score, AUC, QLIKE against an HAR-style benchmark | Q4 robustness |

---

## 4. Data specification

All facts below were measured on 2026-11-08. Sources, commands and full gap tables are in D-01 to D-06.

- **Source.** Binance public archive, spot monthly klines:
  `https://data.binance.vision/data/spot/monthly/klines/<SYM>/<IV>/<SYM>-<IV>-YYYY-MM.zip`
  - Files are listed with an S3 `ListObjects` query against `s3-ap-northeast-1.amazonaws.com/data.binance.vision`. Use a real XML parser: objects uploaded from 2025-02 onward carry extra checksum tags that break regex parsers.
  - Each zip has a `.CHECKSUM` file in `sha256sum` format.
- **Symbols (4).** BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT (D-34).
  - Spot monthly files start 2017-08 (BTC, ETH), 2018-05 (XRP; first bar 2018-05-04) and 2020-08 (SOL; first bar 2020-08-11 06:00). All run through 2026-10 with no missing months.
  - Reserves: ADAUSDT (from 2018-04), TRXUSDT (from 2018-06).
- **Range, per symbol (D-34).**
  - BTC, ETH, XRP: 2018-06-01 00:00 to 2026-10-31 23:45 UTC.
  - SOL: 2020-09-01 00:00 to 2026-10-31 23:45 UTC.
  - The panel is unbalanced. A balanced-panel sensitivity analysis uses only the window cut-offs SOL also has (τ from 2022-03).
- **Downloaded interval.** 15m. Modelling bars are 1h, aggregated from 15m. This lets realised variance use 15-minute returns (D-04).
  - 15m history for the 4 symbols is about 58 MB of zips.
- **SOL quality (checked by the lead; all 75 monthly 1h files and 75 15m files pass their checksums).**
  - 1h: 19 missing bars out of 54,546, in 10 runs. 15m: 93 missing out of 218,184.
  - 3 zero-volume placeholder bars at 1h and 6 truncated bars.
  - Every gap coincides with the exchange-wide outages seen in BTC. The largest are 4 h on 2021-08-13 and 3 h on 2020-12-21 and 2021-04-25.
- **Measured quality, BTCUSDT.**
  - 15m: 562 missing bars out of 322,832 (0.17%).
  - 1h: 127 missing out of 80,708, in 28 runs. Gaps are exchange-wide (identical for BTC and ETH) and none occur in 2022 or from 2024 onward.
  - The largest gap is 2018-02-08/09 (32 h), which falls before our range.
  - Inside our range the worst gaps are 10 h (2018-06-26 and 2019-05-15).
  - There are 15 truncated bars, 4 zero-volume O=H=L=C placeholder bars at 1h, and off-grid bars (+28m14s) on 2018-02-09/11.
- **Timestamp trap.** Spot `open_time`/`close_time` switch from **milliseconds to microseconds from the 2025-01 file onward**. This was confirmed in the files and in the README. Futures files are unaffected.
- **Cleaning rules (D-06).**
  1. Normalise units by magnitude: values above 1e14 are µs.
  2. Skip a non-numeric first line, defensively.
  3. Snap open_time down to the bar grid and assert there are no collisions.
  4. Dedupe and sort.
  5. Reindex to the full UTC grid.
  6. Treat zero-volume O=H=L=C bars as missing.
  7. Flag truncated bars.
  8. Never forward-fill prices across gaps.
  9. A 1h bar is valid only if all four 15m sub-bars are valid.
  10. Gap tolerance (revised by D-43): aggregations over raw hours need 90% of their hours, and history windows over derived values need 50%. Rows whose forward horizon contains any invalid hour are masked. They are excluded from training and testing and counted in the data audit.
- **Archive self-inconsistency (D-40).** In a few hours (≤ 8 per symbol) the archive's 1h file reports more volume and trades than its own 15m files. In one XRP hour it also reports a different close.
  - Hours with a price conflict are invalid.
  - Hours with a volume-only conflict are flagged `volume_suspect`, and volume features treat them as missing.
- **Integrity.**
  - Verify sha256 against `.CHECKSUM` on every download.
  - Keep `data/manifest.csv` with file, sha256, Last-Modified and download date. Binance does occasionally replace archive files.
  - The data-manifest hash is part of every cache key.
- **Licence (D-29, D-38).** The archive is CC BY-NC-SA 4.0 under the Binance Vision Dataset Terms v1.0 (2026-08-26), which also forbid hosting the data. The repository therefore ships **download and verify scripts only**, never raw or processed data.
  - Published derived data, figures and model artefacts carry attribution to Binance Vision and CC BY-NC-SA 4.0.
  - Code is MIT.

---

## 5. Feature set and targets

### 5.1 Features (fixed, 16, all causal; D-07, D-08, D-35)

Computed at the close of 1h bar t from data up to and including bar t:
- σ_t is the trailing 7-day hourly vol, from 15m returns.
- RV_k is the realised variance over the last k hours, from 15m squared log returns, expressed per hour.
- Volume is quote-asset volume.

| # | Name | Definition | Family |
|---|---|---|---|
| 1 | `ret_1h` | log(C_t / C_{t-1}) / σ_t | returns |
| 2 | `ret_4h` | log(C_t / C_{t-4}) / (σ_t·√4) | returns |
| 3 | `ret_24h` | log(C_t / C_{t-24}) / (σ_t·√24) | returns |
| 4 | `ret_168h` | log(C_t / C_{t-168}) / (σ_t·√168) | returns |
| 5 | `rv_ratio_4h` | log(RV_4h / RV_168h) | vol |
| 6 | `rv_ratio_24h` | log(RV_24h / RV_168h) | vol |
| 7 | `rv_level_pct` | percentile of RV_168h within the trailing 180 days | vol |
| 8 | `qv_z_1h` | z-score of log QV_1h against the trailing 168h mean/SD | volume |
| 9 | `qv_z_24h` | z-score of log QV_24h against the trailing 30-day mean/SD of daily log QV | volume |
| 10 | `range_1h` | log(H_t / L_t) / σ_t | range |
| 11 | `pos_range_24h` | (C_t − L_24h) / (H_24h − L_24h) | range |
| 12 | `pos_range_168h` | same, over 168h | range |
| 13 | `hour` | UTC hour of day, nominal (24 levels) | time |
| 14 | `dow` | day of week, nominal (7 levels) | time |
| 15 | `taker_imb_4h` | Σ taker-buy quote volume / Σ quote volume over the last 4h, minus 0.5 | flow |
| 16 | `xret_4h` | partner asset's log(C_t / C_{t-4}) / (σ^partner_t · √4). Partner = BTC for ETH, XRP and SOL; ETH for BTC | cross |

**Notes on the two added features.**
- `taker_imb_4h` is bounded in [−0.5, 0.5]. Its level can shift with venue programmes (market-maker incentives, fee changes), which the data audit flags.
- `xret_4h` uses a different partner for BTC, because a BTC-on-BTC feature would duplicate `ret_4h`. Exact duplicates split shape mass at random between the two copies, which shows up as artificial instability.
  - The feature is fixed **within** each symbol across windows, which is where drift is measured.
  - Pooled cross-symbol summaries of this one feature are reported with BTC separately.
  - The partner's bar must be valid at the same timestamp; outages are exchange-wide, so this costs almost nothing.

**Why these features are scale-free.** Every continuous feature is a ratio, a z-score or a percentile against its own trailing history. Without this, a volatility regime shift would slide each feature's distribution along the x-axis, and the shapes would appear to drift for that reason alone (method pitfall 1).

**Leakage test.** A pytest property test perturbs all data after t and asserts that the features at t are unchanged.

Feature-selection stability is out of scope (§15).

### 5.2 Targets and horizons (D-09, D-33, D-36)

h = forward horizon in hours. **Primary: h = 8. Secondary: h = 24.** For h = 8 there are 32 sub-bars ahead.

- **Direction (near-noise):** y = 1[log(C_{t+h}/C_t) > 0]. Exact zeros are dropped.
- **Normalised forward volatility (positive control):** y = log(RV_fwd,h / RV_168h,t), with both terms per hour.
  - This is the volatility *surprise* relative to the trailing level.
  - It is learnable through intraday and weekly seasonality, mean reversion of the short/long vol ratio, and vol-of-vol. Corsi's HAR and Dudek et al. (2025) give the prior.
  - It shares the scale-free property of the features.
  - Fitted as an EBM regressor.
  - Sensitivity check: raw log RV_fwd (exploratory).
- **Shuffled control (no signal):** the direction target circularly shifted against the features. For each symbol the offset is drawn uniformly from [90, 400] days, rounded so that it is **not** a multiple of 24 h or 168 h, and fixed by seed.
  - This keeps the label's marginal distribution, autocorrelation and overlap structure, so the effective n is the same as for direction.
  - It shows what pure-noise drift looks like.
  - An iid within-window permutation is a secondary variant.

**Effective sample size per 12-month window.** About 8,700 rows, i.e. n/h ≈ 1,095 independent labels at h = 8 and ≈ 365 at h = 24 (D-33).

---

## 6. Stability metrics (definitions and rationale)

Notation:
- Models a and b are trained on windows w_a and w_b.
- j indexes a main-effect term.
- c_{m,j}(x_i) is model m's contribution of term j at row i.

### 6.1 How shapes become comparable

The fitted shapes are not directly comparable, for two reasons:
- Each EBM fit has its own bin edges (`bins_`).
- `interpret` centres each term to mean zero under *its own* training bin weights. We verified this in 0.7.8: the bin-weighted mean of a term is about 1e-17.

So we never compare `term_scores_` index by index. Instead:

1. **Reference sample.** X_ref = the pooled training rows of w_a and w_b, capped at 20,000 with a fixed seed. Both windows exist by the later window's training cut-off, so the comparison is causal.
2. **Cells.** For each continuous feature, define G = 50 equal-probability cells C_{jk} from pooled quantiles of X_ref. Merge ties, and merge cells with fewer than 20 rows. Nominal features use their levels, weighted by pooled frequency.
3. **Cell-averaged contribution.** f_{m,j}(k) = mean over rows i in C_{jk} of c_{m,j}(x_i), where:
   - **EBM:** c = `eval_terms(X_ref)[:, j]`, cross-checked by a manual lookup `term_scores_[j][np.digitize(x, bins_[j][0]) + 1]` in tests. Index 0 is the missing bin and the last index is the unseen bin (verified: len = cuts + 3).
   - **XGBoost:** c = interventional TreeSHAP value on the raw margin, with one common background of 100 rows drawn from X_ref (D-23). Path-dependent SHAP would use each model's own training cover as the background, which confounds explanation drift with background drift.
   - **Logistic or ridge regression:** c = β_j^raw · x_ij, with coefficients converted back to raw feature units so that each window's scaler does not redefine the axis. Nominal features use one coefficient per level.
4. **Re-centre under the common density.** f̃_{m,j}(k) = f_{m,j}(k) − Σ_k ω_k f_{m,j}(k). For equal-probability cells ω_k = 1/G, so plain averages over cells are already density-weighted. Sparse tails get little weight by construction, which is what we want. The intercept is reported separately and never counts as drift.

### 6.2 Per-term metrics

**Distance** (building block):

  d²_j(a,b) = Σ_k ω_k (f̃_{a,j}(k) − f̃_{b,j}(k))²

Units are logit² for direction and log-vol² for vol.

**Amplitude:**

  A_{m,j} = (Σ_k ω_k f̃²_{m,j}(k))^{1/2}

This is the density-weighted SD of the shape. It is always reported next to any stability number, so that a shape which is "stable" only because it is flat is visible (pitfall 2).

**Normalised distance** (secondary):

  nd²_j = d²_j / ((A²_a + A²_b) / 2)

It lies in [0, 4]. It is about 2 for two unrelated shapes of equal amplitude.

**Shape correlation:** ρ_j is the density-weighted Pearson correlation of f̃_a and f̃_b over the cells. Spearman over cells is secondary; EBM shapes are piecewise constant, so ties take average ranks. It is only interpreted where SNR_j > 4.

**Sign agreement:**

  SA_j = Σ_{k∈K} ω_k 1[sign f̃_a(k) = sign f̃_b(k)] / Σ_{k∈K} ω_k

K excludes cells where both |f̃| are below the cell's noise SD τ_jk (from §6.4). The coverage Σ_{k∈K} ω_k is reported alongside. Without the exclusion, sign agreement on near-zero cells would be a coin flip.

**Monotonicity agreement:** MA_j = 1[sign of Spearman(x_k, f̃_a) equals sign of Spearman(x_k, f̃_b)]. It is evaluated only where |Spearman| > 0.3 in both shapes, and the number of qualifying terms is reported.
- A local version (secondary) compares the signs of the decile-to-decile steps Δ_k. It counts only steps where max(|Δ_a|, |Δ_b|) exceeds twice the step's noise SD.

**Linear models:** ρ and MA are degenerate (always ±1), so only d², DR and the slope sign are used.

### 6.3 Importances, interactions and companion distances

**Unified importance** (the same for all model types):

  I_{m,j} = Σ_k ω_k |f̃_{m,j}(k)|

This is computed on the common X_ref. It is the EBM analogue of mean |SHAP|.

`interpret`'s own `term_importances()` (avg_weight) uses each window's own density and centring, so it is reported only, never compared across windows.

**Rank comparison** is on main effects only:
- Primary: weighted Kendall τ (hyperbolic top-weighting, `scipy.stats.weightedtau`). The bottom of a 16-feature ranking is near-zero noise, and plain Spearman penalises that churn as much as churn at the top (D-14).
- Also reported: Spearman; top-3 Jaccard (top-5 secondary); TV = ½ Σ_j |s_{a,j} − s_{b,j}| on importance shares s_j = I_j / Σ I. TV also gets a noise-floor ratio.
- RBO (Webber et al. 2010) is a sensitivity check.

**Interactions** (interaction arm only, D-12):
- With K = 5 pairs fixed:
  - Jaccard J of the selected sets, reported against its null expectation under random selection (expected overlap K²/P ≈ 0.21 pairs with P = 120 possible pairs, so J ≈ 0.02);
  - per-pair selection frequency π over the R replicates of a window, with "stable pairs" defined as π > 0.6;
  - Jaccard of stable sets and Pearson correlation of π vectors between windows.
- For a pair selected in both windows: purify both tensors under the pooled 16×16 cell density, then compute a density-weighted distance. This is exploratory.
- We verified that main-effect scores are bit-identical with and without 5 pairs (same seed and bags). So the pair arm cannot contaminate the primary main-effect analysis.

**Companion distances:**
- **Prediction distance:** d²_pred = mean over X_ref of (η_a − η_b)², where η is the full additive predictor.
- **Feature-family distance:** d² of the summed contributions within each family (returns, vol, volume, range, time).

A stable family with unstable members is Rashomon redistribution among correlated lookbacks (Caraker et al. 2026; Dong & Rudin 2020), not regime drift. These are reported separately.

### 6.4 Noise floor and drift ratio

**Protocols per window.** Fits are cached and reused across every comparison; only the X_ref evaluation is redone per pair.
- **(i) Seed-only:** R_s = 5 full-window fits with explicit seeds.
- **(iii) Data + seed, the primary floor:** R_b = 10 fits, each on an independent half-sample of contiguous 168-hour blocks, drawn without replacement and with a fresh seed.
  - Validation-tail blocks are sampled separately.
  - The purge is applied.
  - With m = n/2 drawn without replacement, the spread around the full-window estimate has variance (σ²/m)(1 − m/n) = σ²/n, matching the sampling variance of a full fit.
  - Draws are independent; complementary halves would double the variance.
  - Half-sampling avoids the duplicate rows a bootstrap creates, which would interact with EBM's binning and min_samples_leaf.
- **(ii) Data-only stationary bootstrap:** passed as replication counts through `bags`. Secondary, used to check that the components are additive.

**Variance decomposition for term j** (computed on the pair-specific X_ref):
- σ²_seed,j = ½ · mean over protocol-(i) pairs of d²_j
- σ²_tot,j = ½ · mean over protocol-(iii) pairs of d²_j
- σ²_samp,j = max(0, σ²_tot,j − σ²_seed,j)
- **Seed share** = σ²_seed / σ²_tot (H2).

**Cross distance:** d̄²_cross,j(a,b) is the mean over the 5 × 5 cross pairs of protocol-(i) full-window fits.

**Drift ratio:**

  DR_j = d̄²_cross,j / (σ²_tot,j(w_a) + σ²_tot,j(w_b))

  DR_tot = Σ_j d̄²_cross,j / Σ_j (σ²_tot,j(w_a) + σ²_tot,j(w_b))

E[DR] ≈ 1 under no drift with disjoint windows. DR_tot is the main-effect aggregate; it weights high-amplitude terms more, which is what we want.

**Overlap-adjusted ratio** for Δ < 12, where the two windows share a fraction ρ = 1 − Δ/12 of their data:

  DR_adj = d̄²_cross / (2[(1 − ρ)·σ̂²_samp + σ̂²_seed])

Each σ̂² is averaged over the two windows. At Δ = 12, ρ = 0 and DR_adj = DR.

**Signal-to-noise:** SNR_j = A²_j / σ²_tot,j. Shapes are "interpretable" where SNR > 4, i.e. the amplitude exceeds two noise SDs.

### 6.5 Why these metrics

- A density-weighted L2 distance on a shared, pooled-quantile grid is the simplest distance that ignores empty regions and is invariant to each fit's binning.
- Normalising by an empirically measured floor rather than by amplitude makes the ratio comparable across features, targets and model classes. It also answers the question as posed: is this change larger than what retraining on the same data would produce?
- Correlation, sign and monotonicity answer the practitioner's question ("would I tell the same story?"). They are gated by SNR because they are meaningless on flat shapes.

---

## 7. Retraining design

- **Windows (D-19).** Rolling, length L = 12 months, step S = 1 month.
  - Model M_t is trained at cut-off τ_t on rows i with τ_t − L ≤ i and i + h ≤ τ_t. Requiring the label to be observed by τ_t is the purge.
  - M_t is tested on [τ_t, τ_{t+1}), one calendar month.
  - BTC, ETH, XRP: features are valid from 2018-12-01 (180-day burn-in after 2018-06-01). So τ_1 = 2019-12-01 and τ_last = 2026-10-01, giving **83 retrains** and 71 zero-overlap Δ = 12 pairs.
  - SOL: features are valid from 2021-03-01. So τ_1 = 2022-03-01, giving **56 retrains** and 44 Δ = 12 pairs.
  - In total that is 305 retrains and 257 zero-overlap pairs.
  - The training purge is h = 8 rows (D-33).
  - Expanding windows are run once for BTC as an illustration of their built-in stabilisation. They are not used for inference.
  - Window lengths of 6 and 24 months are exploratory sensitivity checks (BTC).
- **Internal validation (D-13).** EBM's default `validation_size = 0.15` draws a random row split. With overlapping h-bar labels this leaks: neighbouring rows share h − 1 of their forward bars. Leakage makes early stopping overrun and inflates instability. Instead, every fit passes a time-ordered `bags` matrix:
  - the last 15% of the window (in time) is validation (−1);
  - the h rows before it are excluded (0, the purge);
  - training rows are +1 on a per-bag random 85% of contiguous 168-row blocks, and 0 elsewhere. This keeps the outer-bag diversity that random splits would have provided.
  - **Verified on interpret 0.7.8: shape (n_samples, outer_bags).** The method draft's (outer_bags, n_samples) is rejected with a `ValueError`. A pytest test pins this behaviour.
  - XGBoost uses the same validation tail for early stopping.
  - The random-split default is kept as the **naive-practitioner arm**, to quantify leakage.
- **Replicates per window and target:** R_s = 5 seeds + R_b = 10 half-samples, i.e. 15 fits. LR is deterministic, so it gets the 10 half-samples plus 1 full fit; its seed floor is exactly zero, and this is stated.
- **Fixed hyperparameters (D-10, D-11).** No per-window tuning, since tuning creates drift by itself.
  - **EBM (primary): `interpret` 0.7.8 defaults**, except:
    - `interactions=0`;
    - explicit `random_state`;
    - time-ordered `bags`;
    - `n_jobs=1`, because we parallelise across fits; same-seed fits were bit-identical at n_jobs = 1 and −1.
  - Defaults kept, for the classifier and regressor respectively:
    - outer_bags = 14, inner_bags = 0, max_bins = 1024;
    - learning_rate = 0.015 / 0.04;
    - smoothing_rounds = 75 / 500;
    - early_stopping_rounds = 100, tolerance 1e-5;
    - min_samples_leaf = 4, max_leaves = 2.
  - Reason: Q2 is about the variance a practitioner gets from a single default fit, and 14 bags without pairs is cheap (3.0 s per fit).
  - **EBM interaction arm:** same settings with `interactions=5`.
  - **EBM naive arm:** pure defaults, i.e. random validation split and "3x"/"5x" pairs.
  - **XGBoost:** `hist`, max_depth = 3, eta = 0.05, up to 1,000 trees with early stopping after 50 rounds on the time-ordered validation tail, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5, n_jobs = 1. Hour and day of week enter as integer codes (D-24).
  - **Logistic regression / ridge:** L2 penalty on standardised continuous features (C = 1 or α = 1), one-hot hour and day of week. Coefficients are mapped back to raw units for shapes.
- **Performance per test month.** Predictions are stored on the test block for each full-window fit. Skill is reported per fit and for the seed ensemble mean.

---

## 8. Regime definitions (all causal)

Each state at τ uses only data up to τ (D-20).

- **R1, volatility (primary).** *Revised by D-44:* the primary percentile history is the **trailing 2 years**, and the expanding history is the sensitivity check.
  - σ^30d_τ = trailing 30-day realised vol from 15m returns.
  - p_τ = its percentile within the symbol's expanding history, with at least 365 days of history (earlier bars are "undefined" and excluded from regime analyses). For SOL, labels exist from 2021-09, so the training windows of its first 6 retrains are only partly labelled.
  - States: low (p < 1/3), mid, high (p > 2/3).
  - Sensitivity: a trailing 2-year history instead of expanding.
- **R2, trend.**
  - T_τ = t-statistic of the trailing 90-day mean daily log return.
  - States: bull (T > 1), bear (T < −1), sideways otherwise.
  - Sensitivity: the sign of the 60-day return.
- **R3, liquidity.**
  - Trailing 30-day Amihud illiquidity, mean_d |r_d| / DV_d, ranked by percentile within expanding history.
  - Alternative: log dollar volume z-scored against the trailing year.
  - Caveat: Binance fee promotions (e.g. zero-fee BTC pairs) distort volume. These episodes are to be dated and flagged in the data audit.
- **Window labels.**
  - π^s_w = the fraction of window w's training bars in state s.
  - The test month's ex-ante label is the state at τ_t. The realised share in the test month is descriptive only.
- **Regime-change covariates.**
  - Consecutive retrains differ only by the month added and the month dropped. So the primary H3 covariate is C_t = |π^high(added month) − π^high(dropped month)|.
  - Also: the event indicator E_t = 1[state at τ_t ≠ state at τ_{t−1}].
  - For disjoint pairs: Δπ = |π^high_a − π^high_b|.
- **Descriptive only (labelled as such everywhere).**
  - A 2–3 state Gaussian HMM on daily log RV, fitted on the full sample, used as a plot overlay.
  - A causal variant (exploratory) uses filtered, not smoothed, probabilities from an HMM re-estimated on expanding data at each τ.

---

## 9. Statistical methods

### 9.1 Dependence

- **Over windows.** Summaries over t are medians of DR and regression coefficients, and the series are strongly autocorrelated: consecutive windows share 11/12 of their data, and lag-Δ pairs share models.
- **CIs.** Stationary bootstrap over the t index (Politis & Romano 1994; `arch.bootstrap.StationaryBootstrap`).
  - Expected block length from `arch.bootstrap.optimal_block_length` (`b_sb`; Politis–White 2004 with the Patton–Politis–White 2009 correction).
  - Floored at max(3, Δ). With about 80 points the estimate is unstable, so a 2× block-length sensitivity is reported.
- **Pooling across symbols.** The same time blocks are resampled for all symbols, because crypto symbols are highly cross-correlated. Pooling adds far less information than ×4.
- **Unbalanced panel.** Before 2022-03 only BTC, ETH and XRP contribute, and the bootstrap resamples calendar blocks, so this is handled naturally. A balanced-panel sensitivity analysis restricts all symbols to τ ≥ 2022-03.
- **Within a test month.** Stationary bootstrap of per-row losses with block length max(b_sb, h). AUC is recomputed on each resample.

### 9.2 Effective sample size (reported for every estimate)

- **Overlapping h-bar labels:** n_eff ≈ n/h. For sums of iid returns the autocorrelation is ρ_k = 1 − k/h, so 1 + 2Σρ ≈ h.
- **General series:**

  n_eff = n / (1 + 2 Σ_{k=1}^{K} (1 − k/n) ρ̂_k)

  K follows Geyer's initial-positive-sequence rule: stop at the first k with ρ̂_k + ρ̂_{k+1} < 0. K is capped at n/10.
- Bayley & Hammersley (1946) is cited for n_eff.

### 9.3 Calibration of the floor: G0

The half-sample argument is exact only for smooth estimators. EBM with early stopping may deviate. Calibration therefore has two arms:
- **Stationary surrogates.** Stationary-block-bootstrap the full sequence of precomputed (feature, target) rows for BTC and ETH:
  - mean block length 1–2 weeks (≫ h);
  - target resampled jointly with its features;
  - 3 surrogates per symbol.

  Mixing the regimes uniformly makes the surrogate stationary by construction. The full pipeline is then run on it, and its DR at each Δ is the empirical null.
- **Shuffled-target arm** (on real data, all symbols).

**Pre-specified fallback.** If G0 fails, report the surrogate-calibrated ratio:

  DR* = DR / median surrogate DR (at the same Δ and target)

### 9.4 Drift structure (H1, Q1)

- γ(Δ) = median over t of DR_adj(M_t, M_{t+Δ}). The headline is Δ = 12.
- Under stationarity γ ≈ 1 for Δ ≥ 12. A γ that keeps rising beyond 12 indicates slow structural drift.
- Comparing each window against a fixed reference (the first post-burn-in window) is secondary.

### 9.5 Early-warning test (H4)

**Timing.**
- D_t = log DR_adj,tot(M_{t−1}, M_t) is computable at τ_t: both models, their floors and X_ref all use data up to τ_t.
- Skill_t is measured on [τ_t, τ_{t+1}) and is known at τ_{t+1} + h.

**Mechanical confound.** The month [τ_{t−1}, τ_t) is both M_{t−1}'s test block and new training data for M_t. So corr(D_t, Skill_{t−1}) is partly mechanical. **Only leads k ≥ 0 count as early warning.**

**Decay.** ΔSkill_t = Skill_t − Skill^avail_{t−1}, where Skill^avail uses only rows of block t−1 whose labels are observed by τ_t.

**Tests.**
1. **Descriptive cross-correlation.** CCF r(k) = corr(D_t, ΔSkill_{t+k}) for k = −3..3, with stationary-bootstrap CIs.
2. **Placebo.** Circularly shift D against Skill by random offsets of at least 6 months (999 draws) and recompute max over k ∈ {0, 1, 2} of r(k). The p-value is the share of draws at or above the observed value. Circular shifts preserve both series' autocorrelation. Time-shuffled windows (a block permutation of t) serve as a second placebo.
3. **Primary incremental regression** (panel with symbol fixed effects):

   ΔSkill_t = a + b·D_t + c·Skill^avail_{t−1} + d·C_t + e·log σ_τt + f·PSI_t + g·D^pred_t + ε

   - PSI_t is the population stability index of the added month against the window, averaged over features: covariate shift.
   - D^pred_t is the prediction drift.
   - b is identified net of regime change, vol level, covariate shift and prediction drift. This is what separates "explanations tell you something" from "the regime moved and everything moved".
   - CIs from the time-block bootstrap; Newey–West with lag = block length as a sensitivity check.
4. **Placebo explanations.** D_t computed from the shuffled-target models must not predict the real model's ΔSkill.

**Target.** H4 runs on the vol target. Monthly direction skill has an SE of roughly 0.58/√90 ≈ 0.06 AUC at h = 8 (720 rows / 8), which swamps any plausible decay signal. The direction version is exploratory.

**Power.** With about 80 retrains of persistent series per symbol, n_eff is about 25–35, so the detectable |partial r| at α = 0.05 is about 0.35–0.4. That is why H4 includes an equivalence bound and an "inconclusive" outcome.

### 9.6 Multiple testing

- **Primary family:** H1-vol, H1-dir, H2, H3, H4, H5, under Holm at FWER 0.05.
- **Exploratory, under BH-FDR q = 0.10:** per-feature, per-symbol, per-regime-definition, per-horizon, per-model results, the interaction arm, 15m and window-length sensitivities.

### 9.7 Pre-specified fallbacks: the only changes allowed after `prereg-v1`

1. G0 fails → surrogate-calibrated DR*.
2. The block-length estimate is degenerate → use the floor values in §9.1.
3. A cell-merging rule leaves fewer than 10 cells for a feature → use G = 20 for that feature.
4. A symbol loses more than 10% of its windows to data masking → drop that symbol from pooling and report it separately.
5. Wall-time overrun beyond 2× the estimate → apply the reductions in §11.3 in the listed order. These never touch the primary arm.

---

## 10. Baselines and controls

| Arm | Role | Scope |
|---|---|---|
| EBM, interactions=0 | primary object | all symbols, h = 8 and 24, 3 targets |
| Logistic / ridge regression | linear, deterministic baseline; same features with one-hot time | all |
| XGBoost + interventional TreeSHAP | black box with post-hoc explanation | all symbols, h = 8, 3 targets, Δ ∈ {1, 12} (cost, D-23) |
| Vol target | positive control: real, learnable signal | everywhere |
| Shuffled target | negative control: what pure-noise drift looks like; G0 | everywhere |
| Stationary surrogates | null calibration of DR | BTC, ETH; h = 8; direction and vol |
| EBM interactions=5 | pair-selection stability | BTC, ETH; direction and vol; R = 5 + 5 |
| EBM naive defaults | size of the random-validation leakage and of default pairs | BTC; 3-month step; R_s = 5 |
| EBM outer_bags = 1 | what bagging buys in stability | BTC (exploratory) |
| Expanding windows | shows built-in stabilisation | BTC (illustration) |
| HAR-style OLS on vol features | skill benchmark for the vol target (QLIKE secondary) | all |

---

## 11. Compute estimate

### 11.1 Measured on this laptop (2026-11-08)

**Setup.**
- Hardware: Intel Core Ultra 9 185H (6P + 8E + 2LP cores, 22 logical), 31.4 GB RAM, on AC power.
- Software: Python 3.13.2, interpret-core 0.7.8, xgboost 3.4.1, shap 0.54.1, scikit-learn 1.9.1.
- Data: synthetic, with 16 AR(1) features.
- Script: [`scripts/time_fits.py`](../scripts/time_fits.py), plus the lead's verification run (D-12, D-13).

| Fit, single-threaded | n = 8,760 (1y of 1h) | n = 35,040 (1y of 15m) |
|---|---|---|
| EBM classifier, interactions=0, 14 bags, time bags | **3.0 s** (half-sample: 2.2 s) | 8.1 s |
| EBM regressor, interactions=0, 14 bags | **5.3 s** | ~14 s (est.) |
| EBM classifier, interactions=5, 14 bags | 12.5 s | — |
| EBM classifier, interpret defaults ("3x" pairs) | 85.6 s | 104 s |
| XGBoost (early stopping, about 90 trees) | — | 0.47 s |
| Interventional TreeSHAP, 2,000 rows, background 100 | 1.3 s | — |
| Logistic regression | — | 0.02 s |
| EBM `eval_terms` / explain / predict | milliseconds | milliseconds |

**Parallelism.**
- The best strategy is **16 concurrent single-threaded fits** (joblib loky). That is about 10.5 effective cores and 4–6× the throughput of sequential multi-threaded fits.
- Same-seed EBM fits were bit-identical at n_jobs = 1 and −1, so parallelising across fits does not change results.
- Memory is not a constraint (about 5–60 MB per single-threaded fit).

**Row count barely matters.** EBM cost is driven by boosting rounds and pairs, not rows: 4× the rows costs only about 2.7×. The reasons for 1h bars are therefore statistical, not computational (D-04).

### 11.2 Grid cost

Wall-hours = single-thread CPU-hours / 10.5.

Per window and target, the 5 + 10 replicates cost 5 × 3.0 + 10 × 2.2 ≈ 37 s for direction or shuffled, and about 66 s for vol.

**Tier 1, confirmatory**

| Arm | Fits | Wall-h |
|---|---|---|
| A. EBM primary: 305 retrains (3 × 83 + 56) × 3 targets × 15 replicates, h = 8 | 13,725 | **1.1** |
| B. Stationary surrogates: 3 × (BTC, ETH) × 83 × (direction, vol) × 15 | 14,940 | 1.35 |
| C. XGB + SHAP: 305 × 3 × 15, SHAP on 4 pair X_refs per fit | 13,725 | 1.1 |
| D. LR / ridge / HAR | ~17k | <0.05 |
| **Tier 1 total** | | **≈ 3.6 h, about 5 h with a 1.4× margin for thermals and real-data rounds** |

**Tier 2, secondary and exploratory**

| Arm | Wall-h |
|---|---|
| E. EBM primary at h = 24 (all symbols) | 1.1 |
| F. Interaction arm (K = 5; BTC, ETH; direction and vol; R = 5 + 5) | 1.2 |
| G. 15m-bar sensitivity (BTC, h = 32 bars = 8 h) | 0.75 |
| H. Naive-defaults arm (BTC, 3-month step, R_s = 5, direction and vol) | 0.8 |
| I. Window length 6 and 24 months (BTC) | 0.6 |
| J. outer_bags = 1, expanding windows (BTC) | 0.2 |
| **Tier 2 total** | **≈ 4.7 h (about 6.5 h with margin)** |

The whole programme is about 8.2 wall-hours on paper, or about 11.5 h with margin. It runs as two overnight batches, with Tier 1 first. You accepted this budget (D-39).

The timing runs used 16 synthetic features, the same count as the revised feature set, so the per-fit figures in §11.1 carry over unchanged.

**Storage.** Each EBM fit is stored as float32 `term_scores_`, `bins_` and the intercept, plus test-month predictions, at roughly 60 KB. That is about 0.8 GB for arm A and about 2.5 GB in total, kept in gitignored `data/cache/`.

### 11.3 Reductions if something overruns (in this order; primary arm A untouched)

1. Shuffled target in Tier 2 arms only on every 3rd window (−22% of those arms).
2. XGB R_b 10 → 5.
3. Surrogates 3 → 2 per symbol.
4. Tier 2 step 1 → 2 months.
5. Drop arm I.

### 11.4 Caveats

- The timings are on synthetic data and short bursts. Real features (ties, heavy tails) change the number of boosting rounds, and multi-hour runs throttle.
- M2 re-times one real window and updates this table before the full grid is launched.

---

## 12. Repository layout

```
README.md               project description, status, data notice
LICENSE                 MIT (code)
DATA_NOTICE.md          Binance Vision attribution, CC BY-NC-SA 4.0, no data hosted
pyproject.toml          package metadata (src layout), pytest/ruff config
requirements.lock       fully pinned environment (interpret-core==0.7.8, xgboost==3.4.1,
                        shap==0.54.1, scikit-learn==1.9.1, numpy, pandas, scipy, arch,
                        pyarrow, joblib, pytest, …), generated with pip-tools/uv
configs/
  data.yaml             symbols, range, interval, cleaning thresholds
  features.yaml         the 16 features, lookbacks and cross-asset partners
  prereg.yaml           frozen primary settings (h, L, S, R, metrics, thresholds)
  grid_slice.yaml       M2 slice
  grid_tier1.yaml       confirmatory grid
  grid_tier2.yaml       secondary arms
src/ebmregime/
  data/        listing.py (S3 XML), download.py (checksum, manifest), parse.py
               (µs/ms, snapping, placeholders), bars.py (15m→1h, validity masks)
  features/    build.py      (causal features; no pandas rolling that peeks)
  targets/     build.py      (direction, normalised fwd vol, shuffled; horizon masks)
  regimes/     causal.py     (R1–R3), descriptive.py (full-sample HMM, labelled)
  splits/      windows.py    (rolling/expanding, purge), bags.py (time-ordered bags,
               half-samples, stationary-bootstrap replication counts)
  models/      ebm.py, xgb.py, linear.py   (fixed configs from prereg.yaml)
  explain/     contributions.py (EBM eval_terms, interventional SHAP, linear),
               cells.py (pooled-quantile cells, re-centring)
  stability/   metrics.py (d², A, nd², ρ, SA, MA, importances, TV, Jaccard, π),
               floor.py (σ² decomposition, DR, DR_adj, SNR), surrogate.py
  stats/       bootstrap.py, neff.py, leadlag.py (CCF, placebo, incremental regression),
               multitest.py (Holm, BH)
  runner/      jobs.py (job specs → content hash), cache.py (atomic npz/parquet writes,
               manifest), run.py (joblib loky, 16 workers, resumable, progress log)
  report/      figures.py, tables.py
scripts/
  time_fits.py          compute-estimate timing script (exists)
  download_data.py      fetch + verify + manifest
  run_grid.py           `python scripts/run_grid.py configs/grid_slice.yaml`
notebooks/              narrative only; import from src/
  00_data_audit.ipynb
  01_slice_noise_floor_and_drift.ipynb
  02_regimes.ipynb
  03_early_warning.ipynb
  04_baselines_and_controls.ipynb
  05_figures.ipynb
tests/                  pytest
  test_parse.py         µs/ms flip, header sniffing, snapping, placeholder bars
  test_no_lookahead.py  perturb-the-future property tests for features, targets, regimes
  test_splits.py        purge, embargo, time-ordered bags, half-sample independence
  test_interpret_api.py pins: bags shape (n, outer_bags); mains unchanged by pairs;
                        own-density centring; len(term_scores_) = cuts + 3; n_jobs determinism
  test_metrics.py       identical shapes → 0; sign flip; density weighting ignores empty
                        tails; invariance to bin edges; known DR on synthetic
  test_floor.py         stationary synthetic process → DR ≈ 1 (small, fast)
  test_cache.py         resume after kill; cache key changes with config/data hash
data/                   gitignored: raw/, interim/, cache/
results/                gitignored except results/summary/*.csv used by notebooks
docs/                   PLAN.md, DECISIONS.md, LITERATURE.md, (later) PREREG deviations
```

**Caching and resumability (D-26).**
- One job equals one fit: model × config × symbol × window × target × horizon × replicate.
- The job's key is a hash of the prereg config, code version (git SHA of `src/`), data-manifest hash and job fields.
- A finished job writes `cache/<key>.npz` atomically (temp file plus rename). The runner skips existing keys, so killing and restarting loses at most the in-flight fits.
- Comparison metrics are a second, cheap cached stage that reads fits and writes per-pair rows to parquet.

---

## 13. Milestones (first slice first)

**M0. Environment and data (≈ 0.5 day).** ✅ Done 2026-11-08: all checks pass, see D-40 to D-42 and `notebooks/00_data_audit.ipynb`.
- Pinned lock file, package skeleton, `download_data.py` with checksums and manifest, `00_data_audit`.
- **Done when:**
  - gap counts reproduce §4: BTC has 127 missing 1h bars over its full history; SOL has 19;
  - the µs flip appears at 2025-01;
  - all checksums pass;
  - tests run in CI locally.

**M1. Features, targets, regimes, splits (≈ 1 day).** ✅ Done 2026-11-09: see D-43 to D-48, `scripts/build_panel.py`.
- Leakage property tests, the `interpret` API pin tests, time-ordered `bags`.
- **Done when:** all tests pass, and a hand-checked row matches the feature definitions.

**Pre-registration freeze.**
- I commit `configs/prereg.yaml` and §2/§3/§9 of this plan, tag `prereg-v1`, and write the SHA-256 hashes into `docs/PREREG_HASHES.txt`.
- You deposit those files on Zenodo as an embargoed record, with the hashes in the public description and an embargo date of your choice (for example the planned submission date).
- M2 starts only after the Zenodo DOI exists. The DOI goes into README and DECISIONS.

**M2. First slice (≈ 1 day; under 1 wall-hour of compute).**
- Scope: BTCUSDT, h = 8, EBM `interactions=0`, direction + vol + shuffled targets, all 83 windows, R = 5 + 10, plus 1 stationary surrogate.
- BTC's cross feature needs ETH, so M0 downloads ETH too.
- Produces: noise floor (σ² decomposition, seed share), γ(Δ) for all Δ, the DR_tot distribution, per-feature SNR, shape gallery, and the re-timed compute table.
- **Go / no-go for the full grid. All must hold:**
  1. The pipeline runs end to end, and a kill/resume test loses fewer than 16 fits.
  2. G0 passes on the shuffled control, or the pre-specified DR* fallback passes on the surrogate.
  3. The positive control learns: vol-target median SNR > 4 for at least 3 features, and OOS R² > 0 in at least 80% of months.
  4. Measured wall time is within 2× of the estimate.
- If (3) fails, the feature set or target definition is wrong. We stop and come back to you; we do not tune silently.

**M3. Tier 1, EBM (arms A, B) (≈ 1 day plus one overnight run).**
- H1, H2, H3, and `02_regimes`.

**M4. Baselines (arms C, D) (≈ 1 day).**
- H5, `04_baselines_and_controls`.

**M5. Early warning (≈ 1 day).**
- H4 with placebos, `03_early_warning`.

**M6. Tier 2 arms (≈ 1–2 days plus one overnight run).**
- Interactions, h = 24, 15m, naive defaults, window lengths, outer_bags = 1.

**M7. Write-up (≈ 2–3 days).**
- `05_figures`, results tables with CIs and n_eff, a deviations log, and a public-release check: no data in the repo, licences in place.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **"Stable because nothing was learned."** Flat shapes have small distances. | Amplitude and SNR are reported with every metric. Correlation, sign and monotonicity only where SNR > 4. The H1-dir "vacuous" clause. Vol positive control. Shuffled control shows the noise-only pattern. |
| **The noise floor is mis-calibrated**, so the DR null is not 1. | G0 gate on shuffled data and stationary surrogates. Pre-specified DR* fallback. Synthetic test in `test_floor.py`. |
| **Rashomon redistribution among correlated lookbacks** looks like drift. | Feature-family distance and prediction distance are reported separately. Seed floor. XGB first-mover bias noted (Caraker et al. 2026). |
| **EBM non-determinism.** | Explicit seeds everywhere. Verified bit-identical across n_jobs on 0.7.8. A pin test re-checks after any version bump. |
| **`interpret` API drift** (bags shape, centring, mains being refit after pairs). | Pinned version. `test_interpret_api.py` encodes each verified behaviour, so a version bump fails loudly. |
| **Leakage through random early-stopping validation.** | Time-ordered `bags` with purge. The naive arm quantifies the effect. |
| **Look-ahead in features or regimes.** | Perturb-the-future property tests. Full-sample HMM confined to `regimes/descriptive.py` and labelled in every figure. |
| **Feature distribution shift moves the grid.** | Scale-free features. Pair-specific pooled X_ref. |
| **Overlap between rolling windows makes adjacent models look alike.** | Headline at Δ = 12 (zero overlap). DR_adj for Δ < 12. |
| **Archive gaps, the µs flip, replaced files.** | Cleaning rules §4. Validity masks on features and labels. Checksums and a manifest hash in every cache key. |
| **Compute blow-up.** | Measured costs. Parallelism across fits. Tiered grid. Ordered reduction list §11.3. Resumable cache. |
| **Few effective windows** (n_eff ≈ 25–35) for H3 and H4. | Equivalence bounds and "inconclusive" outcomes pre-registered. Effect sizes reported, not just p-values. Symbols pooled with shared time blocks (no false ×5). |
| **Shared crypto regimes:** the symbols are not independent replications. | Stated. Results reported per symbol and pooled. Bootstrap uses the same time blocks for all symbols. |
| **Daylight-saving shifts in UTC hour-of-day shapes.** | Exploratory check: compare hour shapes in EU/US summer and winter months. Interpret any hour drift with this in mind. |
| **SHAP background dependence; SHAP absorbs half of interactions.** | Common interventional background per pair. Within-cell SHAP dispersion reported as an interaction indicator. |
| **Venue artefacts:** fee promotions, stablecoin or quote changes distort volume features. | Dated and flagged in the data audit. Volume-feature drift interpreted against them. |
| **Licence:** the data cannot be hosted. | Scripts only; DATA_NOTICE; derived artefacts under CC BY-NC-SA. |
| **SOL has a shorter, more idiosyncratic history:** 56 retrains, and the FTX collapse (Nov 2022) hit SOL especially hard. | Results reported per symbol. Balanced-panel sensitivity. SOL's FTX-period windows flagged in figures. |
| **Cross feature and taker imbalance:** the partner differs for BTC, and taker-share levels can shift with venue programmes. | Within-symbol comparisons only for `xret_4h`. Abrupt level shifts in `taker_imb_4h` dated in the data audit. Drift in these two features interpreted against those dates. |
| **Results are null.** | Pre-registered so that "stable", "drifts" and "vacuous" are all reportable conclusions. |

---

## 15. Out of scope (for later)

- Feature-selection stability (e.g. mRMR re-run per window).
- Perpetual futures, funding and open interest.
- Trading performance and costs. This study is about explanations and forecast skill only, with no strategy P&L.
- Deep models.
- Cross-sectional or multi-asset pooled models.

---

## 16. Open questions

### Answered on 2026-11-08

| # | Question | Answer | Logged as |
|---|---|---|---|
| O-1 | Bar size and horizon | 1h bars, **8-hour** primary horizon (24 h stays secondary) | D-33 |
| O-2 | Vol target | Forward vol normalised by trailing vol (raw log RV stays exploratory) | D-36 |
| O-3 | Symbols | **BTC, ETH, XRP, SOL**. SOL checked against the archive: clean, starts 2020-08-11 | D-34 |
| O-4 | Extra features | Add taker-buy imbalance and a cross-asset return. Day of week kept | D-35 |
| O-5 | Pre-registration | My recommendation: git tag plus an embargoed Zenodo record. OSF is not used because it is mid-way through a project transition | D-37 |
| O-6 | Licensing | MIT for code. CC BY-NC-SA 4.0 for derived data, figures and model artefacts. No raw data hosted | D-38 |
| O-7 | Compute | About 11.5 h with margin accepted. Tier 2 kept in full | D-39 |

### Still open

- **O-8. A last check before we say "this is new" (does not block M0–M2).**
  - Before the paper claims that nobody has studied this, someone should check that nobody has.
  - My searches covered free sources: arXiv, publisher landing pages and the open web. They cannot see inside paywalled finance journals, and I cannot search Google Scholar properly.
  - Example of what I might have missed: a 2024 article in a journal such as *Quantitative Finance* titled something like "Do GAM shape functions persist across market regimes?". If it is only visible behind the paywall, I would not have found it.
  - **What I'm asking:** do you have Google Scholar, or university or company library access? If yes, I'll give you 4–5 ready-made searches (10 minutes of your time) before the write-up (M7).
  - If not, we word the claim carefully ("to our knowledge") and repeat my free search just before submission.
- **O-9. Zenodo deposit (your action, at the freeze before M2).**
  - Creating the record needs your Zenodo account. I'll prepare the files and the hash list.
  - Pick an embargo end date (for example your planned submission date).
