# Grand Banquet Chemometrics

**Software version: v0.0.** Grand Banquet supports graphical and command-line workflows for classification and regression.

Grand Banquet (GB) provides a glass-box workflow for comparing classification and regression pipelines, inspecting failures and retaining reproducible artifacts. The full configured library has **15 preprocessing methods × 6 variable-selection routes × 11 task-specific models = 990 structural candidates**. This finite library is not exhaustive over preprocessing, variable subsets or continuous hyperparameters. Its practical contribution is explicit, reusable comparison across this configured library; it reduces reliance on manual pipeline selection while analyst choices remain in the library, grids, groups, metrics and interpretation.

Two applications share `gb_core/`:

- `classification/grand_banquet_classification.py`
- `regression/grand_banquet_regression.py`

The accompanying manuscript, *Grand Banquet: A Glass-Box Chemometric Framework for Analytical Data Matrices*, describes five tasks reconstructed from three public datasets. Its reported runs used all 990 structures per task, three outer and three inner calibration-CV folds, and fixed group-aware calibration/internal-holdout partitions. The five manuscript task results have not been recomputed for this source update. Synthetic examples are software checks and should not be interpreted as the manuscript measurements. See [`METHODS.md`](METHODS.md) for evaluation boundaries and the task-specific user guides for operation.

No Windows executable is included in this source copy.

## Installation and running

Use Python 3.10–3.13. From this source root in your chosen environment:

```powershell
python -m pip install -r requirements.txt
python classification\grand_banquet_classification.py
python regression\grand_banquet_regression.py
```

Full mode is called `audit` on the command line. The mode controls tuning budgets; the number of structural candidates depends on the selected methods. Use `--full-grid` to evaluate the complete 990-candidate library, or **Select All Options** in the GUI. Choosing Full mode alone does not select every method.

```powershell
python classification\grand_banquet_classification.py --headless-demo classification_output --data your_data.csv --mode audit --full-grid --outer-folds 3 --inner-folds 3
python regression\grand_banquet_regression.py --headless-demo regression_output --data your_data.csv --mode audit --full-grid --outer-folds 3 --inner-folds 3
```

The GUI and Standard/Audit command-line runs default to five outer calibration-CV folds and three inner tuning folds. The examples explicitly use three outer and three inner folds, matching the fold counts in the manuscript. Reproducing a reported analysis also requires its reconstructed input table with the recorded Group/Split assignments, run mode and seed. Exploration command-line runs default to three outer folds and use reduced internal settings.

`--jobs 0` uses most CPUs. Interrupted runs can resume from checkpoints; `--no-resume` starts over. Resume identity does not replace source provenance: retain manifests and do not reuse checkpoints across scientific source changes without verifying compatibility.

| Mode | Intended use | Model grid | Selector budget |
|---|---|---|---|
| Exploration | Quick software check | No model-grid search | Reduced |
| Standard (default) | Routine candidate comparison | Medium | Full |
| Audit (GUI: Full) | Expanded candidate comparison | Expanded | Full |

Mode labels do not establish statistical validity for a particular dataset.

## Evaluation and interpretation

Each candidate receives outer calibration-CV predictions (`CV_*`). Model-parameter scoring refits preprocessing and variable selection on each inner-training subset. SG settings and selector fitness have additional sequential/internal boundaries explained in `METHODS.md`; the implementation does not claim every internal score is leakage-proof.

For each candidate completing calibration CV, a full-calibration fit produces apparent `Cal_*` metrics and computes internal-holdout `Val_*` metrics **before aggregate score calculation and sorting**. `Val_*` never enters `Selection_Score`. A holdout failure is recorded separately in `Val_Status` / `Val_Error` and does not erase successful calibration-CV evidence. Full-calibration fit failures remain explicit.

An unseen classification label in the holdout produces a holdout failure while allowing otherwise valid calibration CV to complete; the software does not invent an estimate for that class.

`Selection_Score` and `Selection_Rank` provide the default CV-based display order. They do not prove a universally best pipeline. Selecting among 990 candidates using the same CV results may make the selected candidate's CV estimate optimistic: this is not an additional outer evaluation of the whole library-selection procedure. All metrics remain available for inspection. Choosing candidates after examining `Val_*` changes the holdout into a further selection set; it then cannot provide an untouched evaluation of that decision.

Default classification weights are Kappa 30%, MCC 25%, macro-F1 20%, balanced accuracy 15%, and fold stability 10%. Default regression weights are inverse RMSE 30%, R2 25%, RPD 20%, RPIQ 15%, and fold stability 10%, applied to within-task metric percentiles. `Overfit_*` comparisons are descriptive and do not affect the score.

## Input and groups

Classification requires `Sample name`, `Class`, and numeric feature columns; regression uses `Target` instead of `Class`. Optional `Group` and `Split` columns describe dependence and the calibration/holdout partition.

```csv
Sample name,Target,Group,Split,Feature1,Feature2
Sample_001,12.3,Fish_01,validation,0.123,0.456
```

`Group` is recommended for repeated or matched observations. Supplied groups cannot cross the calibration/holdout partition or the group-aware CV boundaries. Without Group, splitting is by row and cannot establish independence among replicates. Group counts describe supplied labels, not verified physical specimens. The run records group/source context when available; a CSV identifier alone does not establish specimen provenance.

Accepted calibration labels include `calibration` and `train`; holdout labels include `validation`, `val`, `test`, and `holdout`. These names do not establish an external dataset. A within-source partition is termed **internal holdout**. External validation requires separately documented source independence.

The CLI can generate synthetic example CSV files with `--make-demo-data`; these are software examples, not laboratory measurements. The repository does not redistribute manuscript datasets.

## Candidate export and prediction

The GUI **Export Candidate** action allows selection by Pipeline ID or a permitted calibration-CV metric. The shared API is `EvaluationEngine.export_candidate`. CLI export uses existing result evidence and the original data for refitting; it does not rerun the complete candidate library:

```powershell
python regression\grand_banquet_regression.py --export-from regression_output --data your_data.csv --export-pipeline-id PIPELINE_ID --export-output chosen_candidate --selection-reason "Prespecified candidate for deployment"
python regression\grand_banquet_regression.py --export-from regression_output --data your_data.csv --export-metric CV_RMSE --export-direction lower --export-output chosen_candidate --selection-reason "Lowest calibration CV RMSE"
```

`Selection_Score` or a supported numeric `CV_*` field may be used as the export metric. The direction is inferred for recognized metrics, for example lower for `CV_RMSE` and higher for `CV_R2` or `CV_MCC`. Use `--export-direction lower` or `--export-direction higher` to override it; other numeric CV metrics require an explicit direction. The GUI offers `auto`, `higher` and `lower`. `Cal_*` and `Val_*` are not export-ranking metrics. Record the reason and retain the resulting selection/refit evidence. Metric-based selection still uses the available CV estimates and does not remove selection optimism.

Every export verifies the dataset, scientific configuration, candidate library/specifications and run identifier. Importing an existing result table requires its source manifest, a matching dataset fingerprint and candidate library, and the recorded run identifier. The CLI reads this evidence from the source directory. Evaluation and subsequent refitting retain separate provenance records.

Default automatic export writes artifacts for the first eligible candidate in the default CV order. Filenames containing `rank1` or `best` and `RunOutputs.best_pipeline` identify that candidate and do not establish statistical superiority.

```powershell
python regression\grand_banquet_regression.py --predict-model regression_output\best_regression_pipeline.joblib --predict-data new_samples.csv
```

Prediction reuses the stored feature schema, fitted preprocessing, selector and model. New data must contain every original input predictor column recorded in the bundle, including predictors subsequently excluded by variable selection. The GUI and CLI select and reorder those columns by name; additional columns such as sample identifiers or metadata may remain in the input file. Direct calls to `FinalPipelineBundle.predict_frame` require a predictor-only DataFrame with the stored feature names and no extra columns.

## Outputs and provenance

Runs retain `pipeline_results_v3.csv`, `leaderboard_v3.csv`, an HTML view, per-candidate CV predictions/fold metadata, explicit failed rows, checkpoints, and manifests. Default-export artifacts use the names `rank1_validation_predictions.csv`, `best_*_pipeline.joblib`, and `rank1_final_refit.json`.

Run manifests include dependency versions and `source_provenance`: the active source root, per-file raw-byte SHA-256 hashes for Python files in `gb_core`, `classification`, and `regression`, and a deterministic combined digest. Tests, upload folders and caches are excluded. A Git commit is added only if this exact source root is itself a valid Git checkout; an unrelated ancestor repository is never used. Hashes identify the executing source files.

Checkpoint run IDs include an evaluation-compatibility identifier, together with the dataset fingerprint, scientific configuration and candidate definitions. Exact source hashes remain the provenance evidence. Current per-candidate prediction/metadata files are cleared before reattempt, and existing default-export files are cleared before export, preventing stale files from being shown as the new attempt's outputs.

CARS handles both target-by-feature and feature-by-target PLS coefficient arrays and calculates one importance value per predictor. See `METHODS.md` for the evaluation and selection procedures.

Optional DeepSeek interpretation sends structured result evidence by default to `https://api.deepseek.com`. The GUI defaults to `deepseek-flash` and also offers `deepseek-v4-pro`. `gb_core.reporting.build_deepseek_prompt` builds the prompt and `call_deepseek` performs the optional call. API credentials are read from the masked GUI field or `DEEPSEEK_API_KEY` and are not saved in run state or call records. See the [official DeepSeek API documentation](https://api-docs.deepseek.com/) and [chat-completion API reference](https://api-docs.deepseek.com/api/create-chat-completion/) for provider details.

Calls save the actual structured input, system/user prompts, requested model and request parameters, request/response dates, returned model identifier, and response in `ai_reports/call_records`, including failure records where applicable. The requested model name is a service alias; record the identity returned for each call rather than treating the alias as an immutable model version. Inspect the supplied evidence and generated text before scientific use; recorded settings do not guarantee identical future remote output.

Code is licensed under Apache License 2.0. Retain source hashes, dependency versions, seed, Group/Split provenance, CV settings, run mode, selection rationale, and failure records when reporting results.

## Citation and contact

For the current manuscript and data context, cite: Jia, W.; Hong, Y.; Yao, Z.; Koidis, A.; Wu, D.; Huang, H. *Grand Banquet: A Glass-Box Chemometric Framework for Analytical Data Matrices*. ChemRxiv 2026. https://doi.org/10.26434/chemrxiv.15000013/v4. Add the journal citation when available.

Contact: wjia02@qub.ac.uk.
