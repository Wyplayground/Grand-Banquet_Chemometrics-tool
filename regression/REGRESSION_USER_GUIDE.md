# Grand Banquet Regression User Guide

## Workflow

1. In **Data Loading**, select a CSV/XLSX/XLS file and project directory. Choose Exploration (reduced settings), Standard, or Full mode.
2. Load and split the data. Confirm the displayed calibration and independent-validation sample counts.
3. Select preprocessing methods. Imputation and learned preprocessing parameters are fitted using each cross-validation training subset only.
4. Select variable-selection methods. VIP, GA, iPLS, CARS, and SFLA use only the target values available in the applicable training subset.
5. Select regression models and the outer/inner cross-validation counts, then start training.
6. Review `Selection_Rank`, `Selection_Score`, Cal/CV/Val metrics, stability, selected variables, backend, and failures in **Results Analysis**.
7. Generate plots, export results, load the saved Rank 1 pipeline, predict new samples, or optionally create a Qwen scientific summary.

## Input

Required columns:

- `Sample name`: row identifier.
- `Target`: numeric response.
- Numeric predictor columns.

Optional columns:

- `Group`: assigns repeated measurements from the same independent object to one group. A group is never divided across calibration, validation, or cross-validation held-out subsets.
- `Split`: accepts `calibration`, `cal`, `train`, or `training` for calibration and `validation`, `val`, or `test` for independent validation. Without this column, GB creates the split from the selected ratio and seed.

## Evaluation modes

- **Exploration** uses reduced variable-selection budgets, fixed SG settings, and fixed model settings.
- **Standard** uses full variable-selection budgets, searches SG settings, and applies medium model grids.
- **Full** uses full variable-selection budgets, searches SG settings, and applies the expanded model grids documented in the root `to_the_user.md`.

## Result interpretation

- `Cal_*`: apparent performance after fitting all calibration observations; not used for ranking.
- `CV_*`: calibration-set two-level cross-validation performance; the ranking basis.
- `Selection_Score`: weighted percentile score based on inverse RMSE, R2, RPD, RPIQ, and stability.
- `Selection_Rank`: candidate order fixed before validation is evaluated.
- `Val_*`: performance on the predefined independent-validation observations; descriptive only and unable to change the rank.
- `Overfit_*`: Cal/CV/Val diagnostic differences; not used for ranking.

Regression metrics are R2, RMSE, MAE, RPD, RPIQ, RER, Bias, Slope, and Intercept.

Rows that fail before a valid `CV_*` result have `Eligible_For_Ranking=False`. A later validation problem is recorded in `Val_Status` and `Val_Error` while the previously assigned rank remains unchanged.

## Saved pipeline and prediction

`best_regression_pipeline.joblib` contains the fitted preprocessor, selector, regressor, feature schema, parameters, software/schema versions, and dataset fingerprint.

New data must contain the same predictor names as the saved bundle. Use **Load Pipeline / Predict New Data** in Results Analysis or:

```powershell
python regression\grand_banquet_regression.py --predict-model regression_output\best_regression_pipeline.joblib --predict-data new_samples.csv
```

The saved preprocessing and selector are applied directly; variable selection is not rerun.

## Outputs and scientific summary

The project directory contains `pipeline_results.csv`, `leaderboard.csv`, `leaderboard.html`, prediction files, fold metadata, manifests, plots, and the Rank 1 bundle. Together, these files form the Reproducible Analysis Record.

The optional Qwen function receives a structured summary of the results. It is instructed to distinguish CV-based ranking from later validation and not to re-rank candidates using `Val_*`.
