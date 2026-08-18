# Grand Banquet Chemometrics

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue.svg)](README.md)
[![Version](https://img.shields.io/badge/Version-v0.0-orange.svg)](README.md)

Grand Banquet (GB) is a glass-box chemometrics framework for classification and regression. It combines preprocessing, variable selection, modelling, and optional LLM summaries in one reproducible workflow.

The source package contains two applications backed by a shared engine:

- `classification/grand_banquet_classification.py`
- `regression/grand_banquet_regression.py`
- `gb_core/` for shared data handling, transforms, selectors, models, evaluation, and reporting

The image `123.png` is used by the graphical interface.
   > **Tip**: Place `123.png` in the **same folder**. The image `123.png` is inspired by David Hockney's masterpiece *Portrait of an Artist (Pool with Two Figures)*, paying homage to this iconic work. Much like one standing by a pool, gazing at one's own reflection in the water, we too observe data. However sophisticated our methods may be, it remains akin to viewing the data through rippling waters: a pursuit of truth through the wave of uncertainty.

## Scientific workflow

Each task can enumerate 15 preprocessing methods, 6 variable-selection routes, and 11 task-specific models, giving 990 candidate pipeline structures.

| Mode | Intended use | Model hyperparameters | Variable-selection budget |
|---|---|---|---|
| Exploration | Reduced settings | Fixed model settings | Reduced |
| Standard | Routine analysis | Medium grid | Full |
| Full | Publication-oriented analysis | Expanded grid | Full |

The calculation separates pipeline ranking from independent validation:

1. Imputation, preprocessing, variable selection, and hyperparameter tuning use only the relevant cross-validation training subset.
2. Each candidate receives calibration-set cross-validation predictions and `CV_*` metrics.
3. `Selection_Score` and `Selection_Rank` are calculated from `CV_*` metrics and cross-validation stability only.
4. Ranks are frozen before the independent validation data are evaluated.
5. Every successfully ranked pipeline is refitted using all calibration data and receives `Cal_*` and `Val_*` metrics.
6. Rank 1 is additionally exported as a fitted `.joblib` bundle with `rank1_validation_predictions.csv`.

Classification ranking weights are Kappa 30%, MCC 25%, macro-F1 20%, balanced accuracy 15%, and stability 10%. Regression weights are inverse RMSE 30%, R2 25%, RPD 20%, RPIQ 15%, and stability 10%.

See `Evaluation methods.md` for the evaluation contract and [`to_the_user.md`](to_the_user.md) for the Full-mode design, advanced dynamic-search guidance, Group/Split behaviour, and ranking options.

## Installation

```powershell
git clone https://github.com/Wyplayground/Grand-Banquet_Chemometrics-tool.git
cd Grand-Banquet_Chemometrics-tool
conda create -n gb python=3.11 -y
conda activate gb
python -m pip install -r requirements.txt
```

Python 3.10–3.13 are supported. After `conda activate gb`, run the commands below from the repository root.

## Running

GUI:

```powershell
python classification\grand_banquet_classification.py
python regression\grand_banquet_regression.py
```

Training uses most available CPUs automatically. Interrupted GUI runs resume from the project directory unless you reset the project or choose a new output folder.

## Graphical interface

Classification and regression share a six-step layout. Screenshots are in `gui_screenshots/`.

| Step | Classification | Regression |
|---|---|---|
| 1 Data loading | [png](gui_screenshots/gui_classification_step1_data_loading.png) | [png](gui_screenshots/gui_regression_step1_data_loading.png) |
| 2 Preprocessing | [png](gui_screenshots/gui_classification_step2_preprocessing.png) | [png](gui_screenshots/gui_regression_step2_preprocessing.png) |
| 3 Variable selection | [png](gui_screenshots/gui_classification_step3_variable_selection.png) | [png](gui_screenshots/gui_regression_step3_variable_selection.png) |
| 4 Model training | [png](gui_screenshots/gui_classification_step4_model_training.png) | [png](gui_screenshots/gui_regression_step4_model_training.png) |
| 5 Results analysis | [png](gui_screenshots/gui_classification_step5_results_analysis.png) | [png](gui_screenshots/gui_regression_step5_results_analysis.png) |
| 6 AI report | [png](gui_screenshots/gui_classification_step6_ai_report_generation.png) | [png](gui_screenshots/gui_regression_step6_ai_report_generation.png) |

Task-specific notes: [`classification/CLASSIFICATION_USER_GUIDE.md`](classification/CLASSIFICATION_USER_GUIDE.md), [`regression/REGRESSION_USER_GUIDE.md`](regression/REGRESSION_USER_GUIDE.md).

## Input format

Classification:

```csv
Sample name,Class,Group,Split,Feature1,Feature2
Sample_001,Fresh,Batch_01,calibration,0.123,0.456
```

Regression:

```csv
Sample name,Target,Group,Split,Feature1,Feature2
Sample_001,12.3,Fish_01,validation,0.123,0.456
```

- `Sample name` identifies each row. Classification requires `Class`; regression requires numeric `Target`.
- `Group` is optional. Repeated measurements, paired observations, or spectra from the same independent object should share one group identifier. A group is kept intact when calibration, validation, and cross-validation subsets are formed.
- `Split` is optional. Calibration labels are `calibration`, `cal`, `train`, or `training`; independent-validation labels are `validation`, `val`, or `test`. If `Split` is absent, GB creates the split from the configured ratio and random seed, using groups when supplied.
- Every remaining column is treated as a numeric predictor. A feature column with no numeric values is rejected; missing numeric values are imputed from the applicable training subset.

## Outputs

Each completed run writes:

- `pipeline_results.csv` and `leaderboard.csv`
- `leaderboard.html`
- per-pipeline cross-validation predictions and fold metadata
- per-pipeline validation predictions for successfully evaluated pipelines
- `rank1_validation_predictions.csv`
- the fitted Rank 1 `.joblib` bundle
- `run_manifest.json` and `rank1_final_refit.json`
- diagnostic plots and explicit failed-stage records

`Schema_Version` is `0.0`. The manifest also records software version `0.0.0`, dataset fingerprint, seed, mode, split membership, model backend, pipeline definitions, and validation success/failure counts, forming the Reproducible Analysis Record.

## Data availability

This source repository does not include example datasets or the datasets used in the corresponding article.

## Qwen scientific summaries

The optional reporting function sends structured result summaries rather than raw feature matrices by default. API keys are read from the masked GUI field or `DASHSCOPE_API_KEY` and are not written to project state.

## Citation

If you use Grand Banquet in your research, please cite the ChemRxiv preprint below. Update or add the peer-reviewed journal citation when it is available.

Jia, W.; Hong, Y.; Yao, Z.; Koidis, T.; Wu, D.; Huang, H. Grand Banquet: A Glass-Box Chemometric Framework for Analytical Data Matrices. *ChemRxiv* **2026**. https://doi.org/10.26434/chemrxiv.15000013/v4

```bibtex
@article{jia2026grandbanquet,
  title   = {Grand Banquet: A Glass-Box Chemometric Framework for Analytical Data Matrices},
  author  = {Jia, Wenyang and Hong, Yunhe and Yao, Ziwei and Koidis, Tassos and Wu, Di and Huang, Honggang},
  year    = {2026},
  journal = {ChemRxiv},
  doi     = {10.26434/chemrxiv.15000013/v4},
  url     = {https://doi.org/10.26434/chemrxiv.15000013/v4},
  note    = {Preprint}
}
```

## Licence

Code is licensed under the Apache License 2.0. See `LICENSE`.

## Contact

- Email: [wjia02@qub.ac.uk](mailto:wjia02@qub.ac.uk)

## Acknowledgements

Grand Banquet integrates established chemometric methods and machine learning algorithms from the scientific community. Special thanks to the open-source community for the libraries that make this project possible.
