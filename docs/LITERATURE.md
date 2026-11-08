# Literature: verified annotated references

**Compiled:** 2026-11-08.

**How the references were checked.** A literature sub-agent drafted this list. For every entry it opened a landing page during the search: an arXiv abstract, a DOI resolver, a publisher page, ACM, PMLR, JMLR, Project Euclid or RePEc. On that page it confirmed the title, authors, year and venue. Nothing here is cited from memory.

**Re-checked by the lead.** I fetched the arXiv pages for the most load-bearing or most surprising entries myself (marked ✔): 2610.16534, 2508.01851, 2011.09903, 2511.20105 and 2603.22346. I also added three adjacent preprints (marked ➕) after verifying them.

**Removed.** One candidate the sub-agent suggested is not on this list. arXiv 2512.15780 had been described as "explanation drift visible before AUROC drops in financial ML". Its landing page shows a paper on adversarial robustness in financial ML, unrelated to explanation drift.

**Before submission:** run a final Google Scholar / SSRN pass to confirm novelty. The search did not cover paywalled Elsevier and Springer journals exhaustively.

**Legend:**
- ★ essential
- *(preprint)* not peer-reviewed
- ✔ landing page re-checked by the lead
- ➕ found and verified by the lead

---

## 1. Stability and robustness of post-hoc explanations

- ★ **Alvarez-Melis, D. & Jaakkola, T. S. (2018).** *On the Robustness of Interpretability Methods.* ICML Workshop on Human Interpretability. https://arxiv.org/abs/1806.08049
  Defines explanation robustness as local Lipschitz continuity and shows that common explainers violate it. **Here:** we keep the idea but change what is perturbed: the training window, seed and resample instead of the input.

- **Slack, D., Hilgard, S., Jia, E., Singh, S. & Lakkaraju, H. (2020).** *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AIES '20. https://doi.org/10.1145/3375627.3375830
  Shows that perturbation-based explainers can be manipulated. **Here:** a reason to prefer intrinsic EBM terms, and a warning that drift in the SHAP comparison arm may partly come from the explainer itself.

- ★ **Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S. & Lakkaraju, H. (2024).** *The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.* TMLR. https://arxiv.org/abs/2202.01602
  Defines top-k feature agreement, rank agreement and sign agreement metrics, and finds that disagreement between explainers is common. **Here:** the source of our top-k and sign-agreement metrics (PLAN §5).

- **Agarwal, C., Johnson, N., Pawelczyk, M., Krishna, S., Saxena, E., Zitnik, M. & Lakkaraju, H. (2022).** *Rethinking Stability for Attribution-based Explanations.* ICLR 2022 PAIR^2Struct Workshop. https://arxiv.org/abs/2203.06877
  Relative stability: change in the explanation divided by change in the input, representation or output. **Here:** the reason we report prediction distance next to explanation distance.

- **Visani, G., Bagli, E., Chesani, F., Poluzzi, A. & Capuzzo, D. (2022).** *Statistical stability indices for LIME.* Journal of the Operational Research Society 73(1). https://doi.org/10.1080/01605682.2020.1865846
  Stability indices for repeated explainer runs on credit-risk data. **Here:** a finance precedent for measuring a repeated-run noise floor.

- **Lundberg, S. M. et al. (2020).** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence 2(1). https://doi.org/10.1038/s42256-019-0138-9
  TreeSHAP and SHAP interaction values. **Here:** the reference method for the XGBoost comparison arm.

- **Yasodhara, A., Asgarian, A., Huang, D. & Sobhani, P. (2021).** *On the Trustworthiness of Tree Ensemble Explainability Methods.* CD-MAKE 2021. https://arxiv.org/abs/2110.00086
  Gain importance and SHAP for tree ensembles are both unstable under data, seed and hyperparameter perturbations; SHAP less so. **Here:** a precedent for the seed noise floor in the XGBoost arm.

- ✔ **Lin, L. & Wang, Y. (2025).** *SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model.* *(preprint)* https://arxiv.org/abs/2508.01851
  SHAP rank consistency across many seeds depends on how important a feature is. **Here:** we stratify stability metrics by amplitude/SNR. Low-importance features are expected to be noisy.

- ✔ **Liu, B. & Udell, M. (2020).** *Impact of Accuracy on Model Interpretations.* *(preprint)* https://arxiv.org/abs/2011.09903
  Interpretation quality depends on predictive accuracy. **Here:** the theoretical reason for the positive control. On a near-noise target, unstable explanations are expected whatever the regime.

- ✔ **Caraker, D., Arnold, B. & Rhoads, D. (2026).** *First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution.* *(preprint)* https://arxiv.org/abs/2603.22346
  Sequential residual fitting concentrates SHAP importance on whichever correlated feature is fitted early, so explanations become unstable. **Here:** a direct threat for our correlated lookback features in the XGBoost arm. It motivates the feature-family distance (PLAN §5.6).

- **Webber, W., Moffat, A. & Zobel, J. (2010).** *A similarity measure for indefinite rankings.* ACM TOIS 28(4). https://doi.org/10.1145/1852102.1852106
  Rank-biased overlap (RBO). **Here:** a top-weighted rank similarity we considered. We use weighted Kendall τ as primary (DECISIONS D-14) and RBO as a sensitivity check.

## 2. Rashomon effect, model multiplicity, underspecification

- **Breiman, L. (2001).** *Statistical Modeling: The Two Cultures.* Statistical Science 16(3). https://doi.org/10.1214/ss/1009213726
  Origin of the "Rashomon" framing: many roughly equally good models tell different stories.

- **Fisher, A., Rudin, C. & Dominici, F. (2019).** *All Models are Wrong, but Many are Useful.* JMLR 20(177). https://jmlr.org/papers/v20/18-760.html
  Model class reliance: variable importance as a range across good models. **Here:** report importance as a distribution over replicates.

- ★ **Dong, J. & Rudin, C. (2020).** *Exploring the cloud of variable importance for the set of all good models.* Nature Machine Intelligence 2(12). https://doi.org/10.1038/s42256-020-00264-0
  Importance can vary widely across equally accurate models. **Here:** the main threat to validity. Apparent temporal drift may be nothing more than a different Rashomon draw, which is why the noise floor comes first.

- **D'Amour, A. et al. (2022).** *Underspecification Presents Challenges for Credibility in Modern Machine Learning.* JMLR 23(226). https://jmlr.org/papers/v23/20-1335.html
  Seed-only differences give equal held-out accuracy but different behaviour under shift. **Here:** justifies the seed protocol, and predicts a larger seed spread in stress regimes.

- **Marx, C., Calmon, F. & Ustun, B. (2020).** *Predictive Multiplicity in Classification.* ICML 2020, PMLR 119. https://proceedings.mlr.press/v119/marx20a.html
  Ambiguity and discrepancy measures. **Here:** report prediction multiplicity alongside explanation multiplicity.

- **Donnelly, J., Katta, S., Rudin, C. & Browne, E. P. (2023).** *The Rashomon Importance Distribution.* NeurIPS 2023. https://arxiv.org/abs/2309.13775
  Combines bootstrap resampling with Rashomon sets to get a distribution of importance. **Here:** a template for the combined seed × resample noise floor.

- **Müller, S., Toborek, V., Beckh, K., Jakobs, M., Bauckhage, C. & Welke, P. (2023).** *An Empirical Evaluation of the Rashomon Effect in Explainable Machine Learning.* ECML PKDD 2023. https://doi.org/10.1007/978-3-031-43418-1_28
  Explanation diversity depends on hyperparameter tuning and on the metric chosen. **Here:** hyperparameters are fixed across windows (D-10), and several dissimilarity metrics are reported.

- **Zhong, C., Chen, Z., Liu, J., Seltzer, M. & Rudin, C. (2023).** *Exploring and Interacting with the Set of Good Sparse Generalized Additive Models.* NeurIPS 2023. https://arxiv.org/abs/2303.16047
  The Rashomon set of sparse GAMs, and how shape functions vary within it. **Here:** the closest work on within-dataset GAM shape multiplicity. We add the time and regime axis.

## 3. Explanation drift under concept drift and data shift

- **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014).** *A survey on concept drift adaptation.* ACM Computing Surveys 46(4). https://doi.org/10.1145/2523813
  Vocabulary of drift types (sudden, gradual, recurring) and prequential evaluation.

- **Lu, J., Liu, A., Dong, F., Gu, F., Gama, J. & Zhang, G. (2019).** *Learning under Concept Drift: A Review.* IEEE TKDE 31(12). https://doi.org/10.1109/TKDE.2018.2876857
  Divides the field into detection, understanding and adaptation. This project is about drift *understanding*.

- ★ **Mougan, C., Broelemann, K., Masip, D., Kasneci, G., Thiropanis, T. & Staab, S. (2023).** *Explanation Shift: How Did the Distribution Shift Impact the Model?* *(preprint; arXiv lists no journal ref)* https://arxiv.org/abs/2303.08081
  Shifts in the distribution of SHAP explanations detect harmful shift better than input or prediction shift. **Difference:** that work uses a fixed model on shifted data. We compare retrained models.

- ★ **Nigenda, D. et al. (2021).** *Amazon SageMaker Model Monitor.* *(preprint)* https://arxiv.org/abs/2111.13657
  A production system that monitors feature-attribution drift (NDCG of rankings) as an early warning. **Here:** an industry precedent that asserts the "drift leads decay" claim but does not test it. This project tests it (H4).

- **Haug, J., Braun, A., Zürn, S. & Kasneci, G. (2022).** *Change Detection for Local Explainability in Evolving Data Streams.* CIKM 2022. https://arxiv.org/abs/2209.02764
  Local attributions go stale under drift. **Difference:** local attributions in streams, not global glass-box shapes.

- **Meyer, A. P., Ley, D., Srinivas, S. & Lakkaraju, H. (2023).** *On Minimizing the Impact of Dataset Shifts on Actionable Explanations.* UAI 2023. https://arxiv.org/abs/2306.06716
  Explanation instability under retraining scales with the size of the shift. **Here:** suggests drift should grow with regime distance, which H3 tests.

- ★ ✔ **Szandała, T. & Raczkowski, D. (2026).** *Does the Explanation Drift Before the Guarantee? Label-Free Early Warning on Evolving Graphs.* *(preprint, Oct 2026)* https://arxiv.org/abs/2610.16534
  The closest prior work on explanation drift as an early warning. It measures lead times and finds leads under gradual drift but no lead under sudden shifts. **Difference:** graph models and conformal coverage. **Here:** carry over their "no lead for abrupt shifts" result as an expectation for crash regimes, and borrow their approach of measuring lead time.

## 4. EBM / GAM interpretability

- **Lou, Y., Caruana, R. & Gehrke, J. (2012).** *Intelligible models for classification and regression.* KDD '12. https://doi.org/10.1145/2339530.2339556
  Bagged, boosted shallow-tree GAMs, the basis of EBM.

- ★ **Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2013).** *Accurate intelligible models with pairwise interactions.* KDD '13. https://doi.org/10.1145/2487575.2487579
  GA2M and FAST interaction detection. **Here:** FAST is the mechanism whose selection stability we measure.

- **Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M. & Elhadad, N. (2015).** *Intelligible Models for HealthCare.* KDD '15. https://doi.org/10.1145/2783258.2788613
  GA2M shapes exposing a learned artefact (the asthma case). **Here:** shapes are read as "knowledge", so it matters whether they are reliable.

- ★ **Nori, H., Jenkins, S., Koch, P. & Caruana, R. (2019).** *InterpretML: A Unified Framework for Machine Learning Interpretability.* *(preprint)* https://arxiv.org/abs/1909.09223
  The software citation for `interpret` and EBM.

- ★ **Chang, C.-H., Tan, S., Lengerich, B., Goldenberg, A. & Caruana, R. (2021).** *How Interpretable and Trustworthy are GAMs?* KDD '21. https://doi.org/10.1145/3447548.3467453
  Equally accurate GAM algorithms learn different shapes; inductive bias decides what they learn. **Difference:** variation across algorithms on static data. We vary time, seed and resample while keeping EBM fixed.

- **Lengerich, B., Tan, S., Chang, C.-H., Hooker, G. & Caruana, R. (2020).** *Purifying Interaction Effects with the Functional ANOVA.* AISTATS 2020, PMLR 108. https://proceedings.mlr.press/v108/lengerich20a.html
  Effects can move between main and pair terms; purification restores identifiability. **Here:** the reason the primary arm uses interactions=0 and pairs are purified before comparison (D-12).

- **Rudin, C. (2019).** *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence 1(5). https://doi.org/10.1038/s42256-019-0048-x
  The argument for glass-box models. **Here:** H5 tests one of its implications, that glass-box explanations are more trustworthy.

## 5. Finance, crypto, non-stationarity, volatility

- **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
  Purged and embargoed cross-validation for overlapping labels. Cite the book; chapter numbers have not been checked against the book itself.

- ★ **Timmermann, A. (2008).** *Elusive return predictability.* International Journal of Forecasting 24(1). https://ideas.repec.org/a/eee/intfor/v24y2008i1p1-18.html
  Return predictability exists only in short local pockets. **Here:** the prior that direction-target explanations are mostly noise, which the positive control needs to separate out.

- **Lo, A. W. (2004).** *The Adaptive Markets Hypothesis.* Journal of Portfolio Management 30(5). Author PDF: http://stat.wharton.upenn.edu/~steele/Courses/434/434Context/EfficientMarket/AndyLoJPM2004.pdf
  Theory of time-varying market efficiency.

- **Corsi, F. (2009).** *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics 7(2). https://doi.org/10.1093/jjfinec/nbp001
  The HAR-RV model. **Here:** the positive-control target is learnable, and we know in advance roughly which shapes to expect. HAR is the benchmark for volatility skill.

- **Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. (2003).** *Modeling and Forecasting Realized Volatility.* Econometrica 71(2). https://doi.org/10.1111/1468-0262.00418
  Definition of the realised-volatility target.

- ✔ **Dudek, G., Kasprzyk, M. & Pełka, P. (2025).** *Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting.* Expert Systems with Applications (DOI 10.1016/j.eswa.2025.130404). https://arxiv.org/abs/2511.20105
  Bitcoin realised volatility is learnable with boosting; volume and lagged volatility dominate importance, which is computed only statically. **Here:** a prior for the positive-control shapes.

- **Liu, Y., Tsyvinski, A. & Wu, X. (2022).** *Common Risk Factors in Cryptocurrency.* Journal of Finance 77(2). https://doi.org/10.1111/jofi.13119
  Market, size and momentum factors in crypto.

- **Wen, Z., Bouri, E., Xu, Y. & Zhao, Y. (2022).** *Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both.* North American Journal of Economics and Finance 62. https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html
  Intraday crypto momentum and reversal depend on the state (jumps, FOMC, liquidity). **Here:** an a priori reason to expect shapes to depend on regime.

- **Akyildirim, E., Goncu, A. & Sensoy, A. (2021).** *Prediction of cryptocurrency returns using machine learning.* Annals of Operations Research 297. https://doi.org/10.1007/s10479-020-03575-y
  Direction-prediction baselines, including logistic regression.

- **Dessain, J., Bentaleb, N. & Vinas, F. (2023).** *Cost of Explainability in AI: An Example with Credit Scoring Models.* xAI 2023, CCIS. https://doi.org/10.1007/978-3-031-44064-9_26
  EBM comes close to XGBoost in credit scoring. Static setting; stability is not studied. (The DOI resolves to Springer behind an auth wall; metadata confirmed through the OUCI record.)

- ➕ **Attolico, L. (2025).** *Explainable Machine Learning for Macroeconomic and Financial Nowcasting: A Decision-Grade Framework for Business and Policy.* *(preprint)* https://arxiv.org/abs/2512.00399
  A review and framework that lists the temporal stability of explanations as an open research agenda item. **Here:** independent support for the gap.

- ➕ **Kim, S., Cho, S. H., Park, S. & Kim, Y.-M. (2026).** *Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions.* *(preprint)* https://arxiv.org/abs/2609.38869
  Tracks changes in normalised global SHAP importance of an XGBoost stock model over six months and uses them as narrative evidence. It applies no noise floor and runs no stability test. **Here:** an adjacent use of temporal SHAP change.

## 6. Statistics for dependent data, multiple testing, pre-registration

- **Künsch, H. R. (1989).** *The Jackknife and the Bootstrap for General Stationary Observations.* Annals of Statistics 17(3). https://doi.org/10.1214/aos/1176347265
  The moving block bootstrap.

- ★ **Politis, D. N. & Romano, J. P. (1994).** *The Stationary Bootstrap.* JASA 89(428). https://doi.org/10.1080/01621459.1994.10476870
  The resampler used for all CIs over windows and for within-block performance (PLAN §9).

- **Politis, D. N. & White, H. (2004).** *Automatic Block-Length Selection for the Dependent Bootstrap.* Econometric Reviews 23(1). https://doi.org/10.1081/ETC-120028836
- **Patton, A., Politis, D. N. & White, H. (2009).** *Correction to "Automatic Block-Length Selection for the Dependent Bootstrap".* Econometric Reviews 28(4). https://doi.org/10.1080/07474930802459016
  Data-driven block length, as implemented in `arch.bootstrap.optimal_block_length`. Cite both.

- **Bayley, G. V. & Hammersley, J. M. (1946).** *The "Effective" Number of Independent Observations in an Autocorrelated Time Series.* Supplement to JRSS 8(2). https://doi.org/10.2307/2983560
  The origin of effective sample size.

- **Diebold, F. X. & Mariano, R. S. (1995).** *Comparing Predictive Accuracy.* JBES 13(3). https://doi.org/10.1080/07350015.1995.10524599
  Test for out-of-sample comparisons between models (secondary, per window).

- **Newey, W. K. & West, K. D. (1987).** *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica 55(3). https://doi.org/10.2307/1913610
  HAC standard errors, used as a sensitivity check for the H3 and H4 regressions.

- **Benjamini, Y. & Hochberg, Y. (1995).** *Controlling the False Discovery Rate.* JRSS B 57(1). https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
  BH-FDR correction for the exploratory family.

- **Harvey, C. R., Liu, Y. & Zhu, H. (2016).** *… and the Cross-Section of Expected Returns.* Review of Financial Studies 29(1). https://doi.org/10.1093/rfs/hhv059
  Multiple testing in finance. Justifies pre-registration and strict thresholds.

- **Hofman, J. M., Chatzimparmpas, A., Sharma, A., Watts, D. J. & Hullman, J. (2023).** *Pre-registration for Predictive Modeling.* *(preprint)* https://arxiv.org/abs/2311.18807
  A template for the pre-registration in PLAN §2.

---

## Gap this project fills

No work found retrains an EBM/GA2M over rolling windows and measures how stable its shape functions, term importances and selected interactions are over time. This holds in finance and elsewhere, and no EBM application to crypto was found. Work on GAM variability looks either across algorithms (Chang et al. 2021) or within one dataset's Rashomon set (Zhong et al. 2023), never along time. Explanation-drift monitoring tracks post-hoc SHAP of a fixed or incrementally updated model (Mougan et al. 2023; Nigenda et al. 2021; Haug et al. 2022). None of it subtracts a seed/resample noise floor, so Rashomon variation (Dong & Rudin 2020) cannot be separated from real change. Industry asserts that "explanation drift precedes decay" (SageMaker). The one direct test found is an unreviewed preprint on graph streams (Szandała & Raczkowski 2026). Using a learnable target as a positive control next to a near-noise target, to validate explanation-stability metrics, appears new. Attolico (2025) explicitly lists temporal stability of explanations as an open problem.

## Closest prior work and how this differs

| Work | What it does | Difference |
|---|---|---|
| Chang et al. 2021 (KDD) | Equally accurate GAMs disagree in their shapes | Varies the algorithm on static data. We vary time, seed and resample with EBM fixed. |
| Mougan et al. 2023; Nigenda et al. 2021 | SHAP-distribution shift as a monitor | Fixed model, no noise floor, lead over realised decay not tested. |
| Szandała & Raczkowski 2026 *(preprint)* | Explanation drift as early warning, with measured lead times | Graph models with conformal coverage and injected drift. We use tabular glass-box models on natural regimes. |
| Meyer et al. 2023 (UAI) | Explanation stability under retraining after a shift | Counterfactual explanations and one-off synthetic shifts. We use GA2M terms over many rolling windows. |
| Zhong et al. 2023 (NeurIPS) | Rashomon set of sparse GAMs | One dataset, optimisation view. We use an empirical ensemble as the Rashomon proxy and add the time axis. |

## Could not verify (not cited)

- Bergsli, Lind, Molnár & Polasik (2022), *Forecasting volatility of Bitcoin*, RIBAF 59. The landing page returned 403.
- A SUERF policy brief on LSTM+SHAP for the Bitcoin price. The PDF could not be read.
- Hinder et al., *Model Based Explanations of Concept Drift* (arXiv 2303.09331). Seen only in search listings.
- arXiv 2603.06671 (ERP-RiskBench). Seen in search listings only.
- Rejected: arXiv 2512.15780. Its content does not match the claim made about it (see the top of this file).
