# Grand Banquet: Full-Mode Search and Advanced Optimisation Guide

This guide accompanies the source-only `Grand Banquet` v0.0 release. It explains the current Full mode, how input `Group` and `Split` columns are interpreted, why pipeline ranking uses cross-validation rather than independent validation, and a recommended roadmap for a future dynamic-search implementation.

## 1. What the current Full mode does

Full mode evaluates a fixed, transparent candidate library. For each analytical task it enumerates:

```
15 preprocessing methods × 6 variable-selection methods × 11 task-specific models = 990 pipeline structures
```

`990` refers to pipeline structures, not to the total number of fitted estimators. Each structure can require multiple fits because Savitzky–Golay (SG) settings, variable selection, model hyperparameters, and two-level cross-validation are all evaluated within the calibration data.

The processing order is sequential:

1. Fit imputation and preprocessing using the current training subset only.
2. For SG-based preprocessing, select SG parameters using inner cross-validation.
3. Fit variable selection using the same training subset only.
4. Search the model hyperparameters using inner cross-validation.
5. Obtain cross-validation predictions from the outer level and use them for pipeline ranking.

## 2. Current Full-mode preprocessing settings

The 15 preprocessing choices are:

1. Raw
2. Baseline correction
3. Max normalization
4. Area normalization
5. Range normalization
6. Center and scale
7. Log transform
8. Centering
9. Pareto scaling
10. SNV
11. MSC
12. SG derivative
13. Simple 1st derivative
14. Simple 2nd derivative
15. SG smoothing

For `Baseline correction`, `SG derivative`, and `SG smoothing`, Full mode selects the SG setting from these candidates within the training data:

| Parameter | Candidate values |
|---|---|
| `sg_window` | 5, 7, 9, 11, 13, 15 |
| `sg_polyorder` | 2, 3 |

The code automatically makes an SG window valid for the available number of predictors (it must be odd and cannot exceed the number of features). Missing predictor values are imputed with the training-subset median for every preprocessing method.

## 3. Current Full-mode variable-selection settings

| Method | Full-mode setting |
|---|---|
| `None` | Retain all variables. |
| `VIP` | VIP threshold = 1.0; retain at least 5 variables. |
| `iPLS` | Compare up to 10 contiguous intervals and retain the best interval. |
| `GA` | 14 individuals, 8 generations, 4% bit-mutation probability, 5–60 variables (also bounded by one third of the available variables). |
| `CARS` | 24 sampling runs with exponentially decreasing retained-variable proportion. |
| `SFLA` | 14 frogs and 8 generations; 5–60 variables (with the same feature-count bounds as GA). |

Selection fitness is calculated by cross-validation inside the current training subset. Classification uses MCC; regression uses negative RMSE. Both include the feature-count penalty `feature_penalty = 0.02 × selected_fraction`. A selector failure is recorded as a failure; it is not silently replaced by another selector.

## 4. Current Full-mode model grids

All grid searches occur inside the calibration data. `LDA`, `Naive Bayes`, `Linear Regression`, and `Bayesian Ridge` have no exposed grid in the current implementation and therefore use their fixed library defaults.

### Classification

| Model | Full-mode candidates |
|---|---|
| PLS-DA | `n_components`: 2, 5, 8, 10, 12 |
| LDA | Fixed parameters |
| Linear SVM-DA | `C`: 0.1, 1, 10 |
| RBF SVM-DA | `C`: 1, 10; `gamma`: `scale`, 0.01, 0.1 |
| Random Forest | `max_depth`: None, 10, 20; `max_features`: `sqrt`, 0.5; `n_estimators`: 120, 200 |
| XGBoost | `max_depth`: 3, 6, 8; `learning_rate`: 0.05, 0.1; `n_estimators`: 100, 200 |
| Gradient Boosting | `learning_rate`: 0.05, 0.1; `max_depth`: 2, 3 |
| KNN | `n_neighbors`: 3, 5, 7 |
| Logistic Regression | `C`: 0.1, 1, 10 |
| Naive Bayes | Fixed parameters |
| Neural Network | `alpha`: 0.0001, 0.001, 0.01; `hidden_layer_sizes`: (50,), (80,), (120,); `max_iter`: 2000, 5000 |

### Regression

| Model | Full-mode candidates |
|---|---|
| PLS Regression | `n_components`: 2, 5, 8, 10, 12 |
| Linear Regression | Fixed parameters |
| Linear SVR | `C`: 0.1, 1, 10 |
| RBF SVR | `C`: 1, 10; `gamma`: `scale`, 0.01, 0.1 |
| Random Forest Regression | `max_depth`: None, 10, 20; `max_features`: `sqrt`, 0.7; `n_estimators`: 120, 200 |
| XGBoost Regression | `max_depth`: 3, 6, 8; `learning_rate`: 0.05, 0.1; `n_estimators`: 100, 200 |
| Gradient Boosting Regression | `learning_rate`: 0.05, 0.1; `max_depth`: 2, 3 |
| KNN Regression | `n_neighbors`: 3, 5, 7 |
| Ridge Regression | `alpha`: 0.1, 1, 10 |
| Bayesian Ridge | Fixed parameters |
| Neural Network Regression | `alpha`: 0.0001, 0.001, 0.01; `hidden_layer_sizes`: (50,), (80,), (120,); `max_iter`: 2000, 5000 |

XGBoost is an explicit dependency. If it is unavailable, its pipelines are reported as failed; the software does not substitute another model.

## 5. Input data: `Group` and `Split`

### Required columns

- Classification: `Sample name`, `Class`, and numeric predictor columns.
- Regression: `Sample name`, numeric `Target`, and numeric predictor columns.

### Optional `Group`

`Group` identifies repeated measurements, matched samples, or observations from the same independent experimental unit. Examples include repeated spectra from one loaf, repeated scans from one fish, technical replicates from one biological sample, or a matched case-control pair.

If `Group` is present, GB keeps every group intact when it creates the calibration/validation partition and when it performs cross-validation. This prevents the same experimental unit from appearing in both model fitting and evaluation subsets.

If `Group` is absent, GB **does not invent or infer groups**. It treats each row as an independent observation. This is appropriate only when rows are genuinely independent. Users must create the `Group` column themselves when repeated or matched observations exist.

### Optional `Split`

`Split` may predefine the calibration and independent-validation sets. Accepted calibration labels are `calibration`, `cal`, `train`, and `training`. Accepted independent-validation labels are `validation`, `val`, and `test`.

- If `Split` is present, GB uses these labels and verifies that both subsets are non-empty and disjoint. If `Group` is also present, it raises an error when a group crosses from calibration into validation.
- If `Split` is absent, GB creates one reproducible calibration/validation partition from `validation_ratio` and `random_state` (default ratio: 0.20; permitted range: greater than 0 and less than 0.50). With `Group`, it uses a group-aware split; without `Group`, it randomly splits rows. The automatic split is then recorded in the manifest.

Therefore, GB can automatically create a **split**, but it cannot automatically create scientifically meaningful **groups**.

## 6. Cross-validation and independent validation have different jobs

GB uses two-level cross-validation entirely within the calibration data:

- The inner level chooses SG settings and model hyperparameters.
- The outer level produces performance estimates for observations not used to fit that outer model.
- All data-dependent operations—imputation, preprocessing, variable selection, and tuning—are fitted using each training subset only.

The ranking is then frozen from `CV_*` metrics and cross-validation stability:

| Task | `Selection_Score` components |
|---|---|
| Classification | Kappa 30%, MCC 25%, macro-F1 20%, balanced accuracy 15%, stability 10% |
| Regression | inverse RMSE 30%, R2 25%, RPD 20%, RPIQ 15%, stability 10% |

Each component is converted to a within-task percentile rank in the direction where higher is better. For classification, stability is based on outer-fold MCC variation; for regression, it is based on outer-fold RMSE variation.

After `Selection_Rank` is frozen, every successfully ranked pipeline is refitted on all calibration observations. GB then reports apparent `Cal_*` metrics and applies that refitted pipeline once to the independent validation observations to obtain `Val_*` metrics.

`Val_*` values must not be used to recompute `Selection_Score` or `Selection_Rank`. Although validation data are independent of model fitting, selecting a different pipeline after inspecting many validation results would make validation data a second-stage model-selection set. 

## 7. Recommended next step: dynamic parameter search

The current static Full grid is deliberately simple, reviewable, and reproducible. It is not a dynamic optimiser. The following is a recommended extension for advanced users and future development.

### Phase A: define legal, data-dependent ranges before evaluation

Build parameter bounds from the calibration training subset only. Examples:

| Component | Suggested dynamic range |
|---|---|
| PLS components | Integers from 1 to `min(20, n_training_samples - 1, n_selected_features)`. |
| SG window | Odd integers from 5 to `min(51, n_features_or_spectral_channels)`; restrict further using instrument resolution and peak width. |
| SG polynomial order | Integers 2–4, always less than the SG window. |
| SVM/SVR `C` | Log-uniform range 1e-3 to 1e3. |
| RBF `gamma` | `scale` plus a log-uniform numeric range based on training-feature variance, for example 1e-5 to 1e1. |
| Ridge / NN `alpha` | Log-uniform 1e-6 to 1e2. |
| KNN neighbors | Integers from 3 to `min(31, floor(sqrt(n_training_samples)))`, using odd values where appropriate. |
| Tree depth | `None` plus integers 2–20. |
| Tree count | 100–1000, with an early stopping or convergence rule for added trees. |
| Boosting learning rate | Log-uniform 0.005–0.3, paired with 100–1000 estimators. |
| Variable count | A range such as 5 to `min(200, n_features, floor(n_training_samples / 2))`; use a penalty or multi-objective criterion to avoid unnecessarily large subsets. |

Do not set these ranges using independent-validation performance, and do not use all rows before cross-validation to determine scaling-sensitive bounds.

### Phase B: use a budgeted search rather than a larger fixed grid

Recommended choices are randomized search, successive halving, or Bayesian optimisation. For each model family, define a fixed computational budget before the run, for example 30–100 candidates per outer-training subset, then report the budget and search seed. This is generally more efficient than an extremely large grid for continuous parameters such as `C`, `gamma`, learning rate, and regularisation.

### Phase C: extend selectors carefully

For GA, CARS, and SFLA, a future dynamic mode could adapt population size, generations, sampling runs, or the maximum feature count to the number of predictors and available calibration groups. Suggested starting budgets are 20–50 individuals/frogs, 15–30 generations, and 30–80 CARS runs for sufficiently large datasets. These changes should be evaluated by calibration cross-validation only and reported explicitly, because they substantially increase computation and can magnify variance in small datasets.

### Phase D: preserve the evaluation contract

Dynamic search does not justify using validation values to choose hyperparameters or pipeline families. The correct sequence remains:

1. Fix the independent validation set once.
2. Use calibration-only two-level cross-validation for all search, selection, and ranking decisions.
3. Freeze the selected pipeline or a predeclared tier of pipelines.
4. Fit using all calibration data.
5. Evaluate independent validation once for the frozen decision.

## 8. Practical checklist before publication

1. Confirm that rows are genuinely independent; otherwise provide `Group`.
2. Decide whether `Split` is pre-specified or should be created automatically, and record the seed and ratio.
3. Use Full mode as a transparent baseline before adding dynamic search.
4. Keep imputation, preprocessing, selection, and tuning inside calibration cross-validation.
5. Rank pipelines only from `CV_*` evidence and stability.
6. Treat `Cal_*` as apparent calibration fit, not generalisation evidence.
7. Report `Val_*` as independent-validation performance without using it to re-rank candidates.
8. Archive `pipeline_results.csv`, `leaderboard.csv`, `project_state.json`, the manifest, and the exact source version.
