# Grand Banquet evaluation methods

This note defines the evaluation workflow and output fields in Grand Banquet. The software version is v0.0. Metrics use the `Cal_*`, `CV_*`, and `Val_*` prefixes.

## Configured candidate library

The complete configured library for each task contains **15 preprocessing methods × 6 variable-selection routes × 11 task-specific models = 990 structural candidates**. The number evaluated in a run depends on the selected methods. Full/Audit mode expands finite tuning grids; it does not automatically select every method. Use `--full-grid` on the command line or **Select All Options** in the GUI to include all 990 structures. Neither the library nor the grids exhaust all chemometric choices. Explicit method definitions, failures, metrics and reusable artifacts reduce reliance on manual pipeline selection; analysts still determine available methods, groups, grids, score weights and decisions.

## Candidate-level calibration CV and its boundaries

Every candidate is evaluated on the calibration partition with outer CV. For each outer training subset, fitted preprocessing, variable selection, and model tuning use that subset; its held-out observations are predicted without fitting those components on the outer held-out labels. Supplied Group labels are used by group-aware splitters. Model-grid scoring refits preprocessing and selection on each inner-training subset before predicting the corresponding inner-held-out subset.

SG window/order selection is sequential. In Standard and Audit modes it is chosen using internal CV within the current outer-training boundary, then held fixed during model-grid scoring within that boundary. Thus the SG choice has already used the available inner-CV outcomes; model-grid inner scores are not an untouched evaluation of SG selection. Exploration uses fixed SG defaults.

Selector fitness (GA, iPLS, CARS and SFLA where applicable) evaluates models on already preprocessed inputs supplied to the selector. It does not refit the preprocessor within each selector-fitness split. Supervised subset generation may use the full current selector-training subset before fitness comparison. These internal scores guide search within the current training boundary; they are not independent evaluation scores.

Outer calibration-CV predictions are concatenated to produce each candidate's `CV_*` metrics. They are candidate-level estimates. There is no extra resampling loop that selects among the whole library within each outer split and evaluates that library-selection procedure on a separate outer test partition. Reusing candidate CV metrics to select a displayed/exported candidate may make the selected estimate optimistic.

### CARS feature importance

CARS derives predictor importance from PLS regression coefficients. It accepts both feature-by-target and target-by-feature `coef_` arrays, identifies the feature axis, and calculates one importance value per predictor before variable-subset generation and fitness comparison.

## Full-calibration fit and internal holdout

After a candidate completes its calibration CV, it is fitted on all calibration observations. Apparent `Cal_*` and within-source internal-holdout `Val_*` metrics are calculated during candidate evaluation, **before aggregate Selection_Score calculation and sorting**. Automatic export may refit the first eligible candidate in the default CV order afterward. That export does not turn the previous candidate evaluation into an evaluation of the whole selection procedure.

Classification reports Kappa, MCC, macro-F1, balanced accuracy, sensitivity and specificity (and accuracy); regression reports R2, RMSE, MAE, RPD, RPIQ, RER, bias, slope and intercept. `Cal_*` is descriptive apparent performance. `Val_*` is a one-partition descriptive check and never enters the default score. Separation of rows or supplied groups within one source is internal holdout, not evidence of a separately sourced external dataset.

`Val_Status` and `Val_Error` isolate holdout failures from calibration-CV success: eligible CV evidence can remain available when holdout evaluation fails. A candidate whose calibration computation or required full-calibration fit fails remains an explicit failed row. Do not infer that every ranked candidate has a successful holdout prediction.

Classification labels present only in the holdout are recorded as a holdout failure rather than aborting otherwise valid calibration CV. This does not create an estimate for the unseen class.

Repeated spectra are observations, not independent specimens. Group counts and source context describe supplied metadata; their physical interpretation requires source documentation. A count alone does not establish specimen-level independence or confidence intervals. The software does not automatically calculate bootstrap confidence intervals or run a separate benchmark study.

## Default CV score and display order

Within the current task, metric values are converted to percentile ranks with direction aligned so larger values are better. Ties receive the average percentile. Default weights are:

- Classification: `0.30*Kappa + 0.25*MCC + 0.20*macro-F1 + 0.15*balanced accuracy + 0.10*stability`.
- Regression: `0.30*inverse RMSE + 0.25*R2 + 0.20*RPD + 0.15*RPIQ + 0.10*stability`.

Stability is the inverse percentile of outer-fold MCC standard deviation for classification, or outer-fold RMSE standard deviation for regression. Holdout metrics are excluded from this function. `Overfit_*` fields are descriptive comparisons and do not affect ordering.

`Selection_Rank` records the default CV-score browsing order. Being first does not imply a universally best model or an unbiased performance estimate after library-wide selection. Inspect the full result table using metrics appropriate to the scientific question. Examining all `Val_*` and then choosing a candidate consumes that holdout for selection; a further untouched evaluation is needed for that decision.

## Explicit candidate export

`EvaluationEngine.export_candidate` accepts a Pipeline ID or selection by `Selection_Score` / a supported numeric `CV_*` metric. Direction is inferred for recognized metrics, such as lower for `CV_RMSE` and higher for `CV_R2` or `CV_MCC`; other numeric CV metrics require an explicit direction. The CLI loads an existing run with `--export-from` and `--data`, and selects a candidate with `--export-pipeline-id` or `--export-metric`. Use `--export-direction higher` or `--export-direction lower` to specify or override the direction; `--export-output` and `--selection-reason` set the export destination and record the rationale. The GUI provides **Export Candidate** with `auto`, `higher` and `lower` direction options. Export refits a selected eligible candidate and saves its fitted bundle, predictions, decision/refit context and manifest without rerunning the entire library or changing the supplied score columns. Record a prespecified reason where possible.

Export verifies the result table against the current dataset fingerprint, scientific configuration, candidate specifications/library and run identifier. Importing an existing result table requires its `source_manifest`, the recorded run identifier, and a matching dataset fingerprint and candidate library. The CLI loads this evidence from the source run directory. Evaluation and subsequent refitting retain separate provenance records.

Automatic default export writes `best_*_pipeline.joblib` and `rank1`-named artifacts for the first eligible candidate in the default CV order. These names identify that candidate and confer no independent superiority claim.

## Failure records and reproducibility

Every attempted specification retains status/error information. Missing dependencies and failures of preprocessing, selection, tuning, calibration fit, outer-CV, holdout or export are not replaced by a different backend or by training predictions disguised as holdout predictions. Checkpoints are atomic and resumable. Run identifiers use dataset/scientific configuration and candidate definitions; retain source provenance as well and verify compatibility before resuming after scientific source changes.

Checkpoint run identity includes an evaluation-compatibility identifier. A matching compatibility identifier is required when resuming; exact source-file hashes are retained separately. Before a candidate is reattempted, its current per-ID CV/holdout prediction and fold-metadata files are cleared. Before default export, existing `best_*` / `rank1` files are cleared, so a failed attempt cannot leave stale files presented as its output.

Successful cached rows are reused only when a per-candidate provenance marker matches the current run ID, dataset fingerprint, evaluation revision, validation status, and prediction/fold-file hashes. If another run sharing that output directory has overwritten those artifacts, or a marker is missing/corrupt, the affected candidate is recomputed. Candidate IDs must be unique within a run. Explicit cached failures retain their previous state; use `--no-resume` to retry them after fixing their cause.

Run manifests record dependency versions plus hashes of the executing copy's Python sources in `gb_core`, `classification` and `regression`. The combined SHA-256 covers sorted POSIX relative paths, a NUL separator, each raw-byte file SHA-256 and a newline. Absolute installation paths do not enter that digest. Tests, upload folders and caches are excluded. Git HEAD is recorded only when the exact source root is itself a valid checkout; hashes remain the description of actual working files.

Optional DeepSeek interpretation uses `https://api.deepseek.com`, with credentials from the masked GUI field or `DEEPSEEK_API_KEY`. The GUI defaults to `deepseek-flash`; `deepseek-v4-pro` is also available. The reporting APIs are `build_deepseek_prompt` and `call_deepseek`. Provider details are documented in the [official API guide](https://api-docs.deepseek.com/) and [chat-completion reference](https://api-docs.deepseek.com/api/create-chat-completion/).

Calls record the actual structured input and system/user prompts, requested model and request parameters, request/response dates, returned model identity and response in `ai_reports/call_records`, without credentials. Model aliases are not immutable-version guarantees: retain the returned identity for each call. Failure records describe actual attempted calls. These records support auditing the actual call. Remote-service output is not guaranteed to repeat exactly.
