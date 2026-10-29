# Grand Banquet Regression User Guide

## Workflow

1. In **Data Loading**, choose a CSV/XLSX/XLS file, output directory and Exploration, Standard or Full mode. Input requires `Sample name`, `Target` and numeric predictors.
2. Supply `Group` for repeated or matched observations and document `Split` / source context. Supplied groups are kept apart at the calibration/internal-holdout and group-aware CV boundaries; labels alone do not verify physical specimens.
3. Select preprocessing, variable-selection and model methods, CV settings and seed, then run. The full library is 15 × 6 × 11 = 990 structures. Actual candidate counts depend on the selected methods; **Select All Options** selects the entire structural library. Full mode controls tuning budgets and does not automatically select every method. The library is finite.
4. Review all result metrics in **Results Analysis**. The default score/order is one browsing rule; use criteria appropriate to the scientific question.
5. Use **Export Candidate** to choose a Pipeline ID or permitted CV metric and record the selection reason. Direction can be `auto` for recognized metrics or explicitly set to `higher` or `lower`; other numeric CV metrics require an explicit direction. Export refits that candidate and retains decision/refit evidence; it does not rerun the whole library.
6. Use **Load Pipeline / Predict New Data** to reuse a fitted bundle with all original input predictor columns.
7. Optionally review the structured evidence in **AI Report Generation** before generating a DeepSeek interpretation. The GUI defaults to `deepseek-flash` and offers `deepseek-v4-pro`. Use the masked credential field or `DEEPSEEK_API_KEY`. Calls retain actual request/response evidence in `ai_reports/call_records` without credentials.

## Result interpretation

- `Cal_*`: apparent full-calibration performance; not used for ordering.
- `CV_*`: candidate-level calibration CV; selection among candidates using these values may be optimistic.
- `Selection_Score`: default weighted calibration-CV percentile score; uses the weights defined in `METHODS.md`.
- `Selection_Rank`: position in the default score display order, calculated after candidate Cal/Val evaluation.
- `Val_*`: descriptive internal-holdout metrics; excluded from the default score.
- `Val_Status` / `Val_Error`: holdout success/failure independent of available CV evidence.
- `Overfit_*`: descriptive Cal/CV/Val comparisons; not part of the score.

A holdout failure may coexist with successful CV evidence and eligibility. Calibration computation or required full-calibration fit failures remain explicit failed rows. No training prediction substitutes for a holdout prediction.

The candidate outer test is not used to fit preprocessing, selection or tuning. Model-grid inner scoring refits preprocessing/selection on inner-training data. SG choice precedes model-grid scoring, and selector fitness operates on already preprocessed inputs; their internal scores have the limits described in root `METHODS.md`. No extra outer evaluation of whole-library selection is performed. CARS calculates one importance value per predictor and supports both orientations of the PLS coefficient array.

Default automatic `best_regression_pipeline.joblib` / `rank1` exports describe the first eligible candidate in the default CV order and do not prove optimality. Explicit CV-metric export still selects on the available CV estimates. Choosing after inspecting all holdout results consumes that holdout for further selection; report that decision and obtain an untouched evaluation if needed.

## Prediction and records

The fitted preprocessing, selected variables and model are reused during prediction. New data must contain every original input predictor column recorded in the bundle, including predictors subsequently excluded by variable selection. The GUI and CLI select and reorder these columns by name; additional sample-identifier or metadata columns may remain in the input file. Retain source/dependency provenance, the fitted bundle, dataset fingerprint, Group/Split context, selected candidate/reason and refit evidence. Source hashes identify the executing Python files.

DeepSeek reporting uses `https://api.deepseek.com` and the `build_deepseek_prompt` / `call_deepseek` APIs. Call records retain actual structured input, system/user prompts, request parameters, requested/returned model identities, dates and response. Model aliases are not immutable-version guarantees; retain each returned identity. Retained call records support auditing; remote output is not guaranteed to repeat exactly. See root `README.md` for official provider references and interpretation limits.
