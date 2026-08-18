# Grand Banquet v0.0 evaluation methods

This document defines the evaluation contract for software version **0.0.0** and result schema **0.0**. Result tables use `Cal_*`, `CV_*`, and `Val_*` prefixes for apparent calibration, calibration-set cross-validation, and independent-validation metrics.

## Candidate Pipeline Evaluation

GB evaluates each requested preprocessing–variable-selection–model structure by two-level cross-validation within the calibration data. The outer level supplies predictions for samples not used to fit that outer model. The inner level selects SG settings and model hyperparameters.

Imputation, preprocessing, and supervised variable selection are fitted using each training subset only and then applied to its corresponding held-out subset. During inner tuning, these steps are refitted from the original predictors for every inner training subset. Related observations remain together when a `Group` column is provided.

Baseline correction, SG derivative, and SG smoothing search Savitzky–Golay window and polynomial order in Standard and Full modes. Exploration uses fixed defaults. Variable-selection algorithms use only the response values available in their training subset.

Classification candidates are summarized by Accuracy, Balanced Accuracy, Kappa, MCC, macro-F1, Sensitivity, and Specificity. Regression candidates are summarized by R2, RMSE, MAE, RPD, RPIQ, RER, Bias, Slope, and Intercept.

## Selection Score from Calibration-Set Cross-Validation

Cross-validation metrics are converted to within-run percentile ranks with their direction aligned so that larger percentiles are better. Equal numerical values receive the average percentile.

Classification:

`0.30 × Kappa + 0.25 × MCC + 0.20 × macro-F1 + 0.15 × balanced accuracy + 0.10 × stability`

Regression:

`0.30 × inverse RMSE + 0.25 × R2 + 0.20 × RPD + 0.15 × RPIQ + 0.10 × stability`

Classification stability is based on the inverse percentile of the outer-subset MCC standard deviation. Regression stability is based on the inverse percentile of the outer-subset RMSE standard deviation. `Cal_*`, `Val_*`, and overfit diagnostic columns are not read by the ranking function.

`Selection_Rank` is frozen immediately after all eligible `CV_*` rows have been ranked. A validation failure cannot remove a rank or change its value.

## Independent Validation after Pipeline Ranking

After ranks are frozen, every successfully ranked pipeline is refitted using all calibration observations. The refitted pipeline provides apparent `Cal_*` metrics and is then applied once to the predefined independent validation observations to obtain `Val_*` metrics.

Validation values are reported for descriptive comparison and diagnosis only. They do not change `Selection_Score` or `Selection_Rank`. If users inspect validation values and choose a different pipeline, those data have become a second-stage selection set and should not be described as an untouched final assessment.

`Val_Evaluated`, `Val_Status`, and `Val_Error` record validation separately from ranking eligibility. Rank 1 is additionally refitted and saved as the task-specific `.joblib` bundle together with `rank1_validation_predictions.csv`.

## Failure Reporting and Reproducible Execution

Every requested pipeline structure produces a result row. Dependency, preprocessing, variable-selection, inner-tuning, model-fit, outer-cross-validation, and validation problems are recorded explicitly; GB does not silently substitute another model or use training predictions as held-out predictions.

Failures before a valid calibration-set cross-validation result set `Eligible_For_Ranking=False`. A failure during the later independent-validation phase is stored in `Val_Status` and `Val_Error` without changing the frozen rank.

The Reproducible Analysis Record contains the software and schema versions, dataset fingerprint, input feature schema, calibration and validation membership, group assignments, random seed, evaluation mode, cross-validation settings, preprocessing and variable-selection choices, model backend, selected hyperparameters, performance metrics, timings, and failure status. SQLite checkpoints store completed stages atomically so an interrupted run can resume with the same scientific configuration.
