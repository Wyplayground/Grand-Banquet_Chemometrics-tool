from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .artifacts import load_bundle
from .contracts import (
    CLASSIFICATION_MODELS,
    PREPROCESSING_METHODS,
    REGRESSION_MODELS,
    SELECTOR_METHODS,
    EvaluationMode,
    PipelineSpec,
    RunConfig,
    TaskType,
    enumerate_specs,
)
from .data import create_demo_data, load_dataset, split_dataset
from .engine import EvaluationEngine
from .plotting import create_visualization_suite


QUICK_PREPROCESSING = ["Raw", "SNV", "MSC", "SG smoothing"]
QUICK_SELECTORS = ["None", "VIP"]
QUICK_MODELS = {
    TaskType.CLASSIFICATION: ["PLS-DA", "SVM-DA (rbf)", "Random Forest", "Gradient Boosting"],
    TaskType.REGRESSION: ["PLS Regression", "SVR (rbf)", "Random Forest Regression", "Gradient Boosting Regression"],
}


def _csv_list(value: Optional[str], defaults):
    return [item.strip() for item in value.split(",") if item.strip()] if value else list(defaults)


def load_run_manifest(output_dir: str | Path, task: Optional[TaskType] = None):
    """Rebuild scientific settings from the saved run, never current UI controls."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config_values = {key: value for key, value in manifest["config"].items()
                     if key in {field.name for field in fields(RunConfig)}}
    config_values["output_dir"] = output_dir
    config = RunConfig(**config_values)
    if task is not None and config.task is not TaskType(task):
        raise ValueError("Stored run task does not match this application.")
    spec_fields = {field.name for field in fields(PipelineSpec)}
    specs = []
    for saved in manifest["pipeline_specs"]:
        values = {key: value for key, value in saved.items() if key in spec_fields}
        values["task"] = TaskType(values["task"])
        spec = PipelineSpec(**values)
        if spec.task is not config.task or saved.get("pipeline_id") != spec.pipeline_id:
            raise ValueError("Stored pipeline specification or identifier is inconsistent.")
        specs.append(spec)
    if not specs:
        raise ValueError("Stored run contains no pipeline specifications.")
    return config, specs, manifest


def load_stored_run(output_dir: str | Path, data_path: str | Path, task: TaskType, logger=print):
    """Load existing CV rows and verify their data binding without rerunning CV."""
    config, specs, manifest = load_run_manifest(output_dir, task)
    dataset = load_dataset(data_path, config.task)
    split = split_dataset(dataset, config)
    if split.fingerprint != manifest.get("dataset_fingerprint"):
        raise ValueError("Data fingerprint differs from the stored run; export was cancelled.")
    for key, values in [("calibration_samples", split.sample_cal),
                        ("validation_samples", split.sample_validation)]:
        if key in manifest and list(map(str, manifest[key])) != values.astype(str).tolist():
            raise ValueError("Recreated calibration/internal holdout partition differs from the stored run.")
    results = pd.read_csv(Path(output_dir) / "pipeline_results_v3.csv",
                          dtype={"Run_ID": str, "Pipeline_ID": str})
    return EvaluationEngine(split, config, logger=logger), specs, results, manifest


def _export_options(args):
    return dict(pipeline_id=args.export_pipeline_id, metric=args.export_metric,
                higher_is_better=None if args.export_direction is None else args.export_direction == "higher",
                output_dir=args.export_output, selection_reason=args.selection_reason)


def _print_export(exported) -> None:
    print(f"Exported candidate: {exported.pipeline_id}")
    print(f"Pipeline bundle: {exported.bundle_path}")
    print(f"Export record: {exported.manifest_path}")
    print(f"Internal holdout status: {exported.validation_status}")
    if exported.validation_error:
        print(f"Internal holdout evaluation: {exported.validation_error}", file=sys.stderr)


def _predict(bundle_path: str, data_path: str, output_path: Optional[str]) -> Path:
    bundle = load_bundle(bundle_path)
    path = Path(data_path)
    data = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    sample = data["Sample name"].astype(str) if "Sample name" in data else pd.Series([f"Sample_{i + 1}" for i in range(len(data))])
    features = data.loc[:, bundle.feature_names]
    prediction = bundle.predict_frame(features)
    name = "Predicted_Class" if bundle.task is TaskType.CLASSIFICATION else "Predicted_Target"
    result = pd.DataFrame({"Sample name": sample, name: prediction})
    destination = Path(output_path) if output_path else path.with_name(path.stem + "_gb_predictions.csv")
    result.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def build_parser(task: TaskType) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Grand Banquet vNext {task.value} GUI and headless workflow")
    parser.add_argument("--make-demo-data", metavar="FILE")
    parser.add_argument("--headless-demo", metavar="DIR")
    parser.add_argument("--data", metavar="CSV_OR_EXCEL")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--mode", choices=[mode.value for mode in EvaluationMode], default=EvaluationMode.STANDARD.value)
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--validation-ratio",
        "--holdout-ratio",
        dest="validation_ratio",
        type=float,
        default=0.2,
        help="same-source internal holdout fraction; the legacy --holdout-ratio spelling is accepted",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=0, help="pipeline workers; 0 uses up to 80%% of logical CPUs")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--legacy-export", action="store_true", help="also write a v2 compatibility CSV")
    parser.add_argument("--preprocessing", help="comma-separated public preprocessing names")
    parser.add_argument("--selectors", help="comma-separated selector names")
    parser.add_argument("--models", help="comma-separated model names")
    parser.add_argument("--predict-model", metavar="JOBLIB")
    parser.add_argument("--predict-data", metavar="CSV_OR_EXCEL")
    parser.add_argument("--prediction-output", metavar="CSV")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--export-pipeline-id", metavar="ID", help="export an eligible candidate using its existing CV results")
    selection.add_argument("--export-metric", metavar="CV_METRIC", help="choose by Selection_Score or a numeric CV_* metric")
    parser.add_argument("--export-direction", choices=["higher", "lower"], help="metric preference; omit to infer direction")
    parser.add_argument("--export-from", metavar="DIR", help="export from a stored run without repeating library CV; requires --data")
    parser.add_argument("--export-output", metavar="DIR", help="parent folder for candidate exports")
    parser.add_argument("--selection-reason", help="optional explanation recorded with the exported candidate")
    return parser


def main_for_task(task: TaskType, argv: Optional[Sequence[str]] = None) -> int:
    task = TaskType(task)
    parser = build_parser(task)
    args = parser.parse_args(argv)
    requested_export = any([args.export_pipeline_id, args.export_metric, args.export_direction,
                            args.export_output, args.selection_reason])
    if args.export_direction and not args.export_metric:
        parser.error("--export-direction requires --export-metric")
    if args.export_from and (args.headless_demo or args.make_demo_data or args.predict_model or args.predict_data):
        parser.error("--export-from cannot be combined with a new run, demo generation, or prediction")
    if requested_export and not (args.export_from or args.headless_demo):
        parser.error("candidate export options require --export-from or --headless-demo")
    if requested_export and (args.make_demo_data or args.predict_model or args.predict_data):
        parser.error("candidate export options cannot be combined with demo generation or prediction")
    if args.export_from:
        if not args.data:
            parser.error("--export-from requires --data containing the original run data")
        try:
            engine, specs, results, manifest = load_stored_run(args.export_from, args.data, task)
            exported = engine.export_candidate(results, specs, source_manifest=manifest, **_export_options(args))
            _print_export(exported)
            return 0
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(f"Candidate export failed: {exc}", file=sys.stderr)
            return 2
    if args.predict_model or args.predict_data:
        if not args.predict_model or not args.predict_data:
            parser.error("--predict-model and --predict-data must be provided together")
        destination = _predict(args.predict_model, args.predict_data, args.prediction_output)
        print(f"Predictions written: {destination}")
        return 0
    if args.make_demo_data:
        destination = create_demo_data(args.make_demo_data, task)
        print(f"Demo data written: {destination}")
        return 0
    if args.headless_demo:
        output_dir = Path(args.headless_demo)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = Path(args.data) if args.data else output_dir / f"gb_{task.value}_demo.csv"
        if not data_path.exists():
            if args.data:
                parser.error("--data file does not exist")
            create_demo_data(data_path, task)
        mode = EvaluationMode(args.mode)
        outer_folds = args.outer_folds or (3 if mode is EvaluationMode.EXPLORATION else 5)
        config = RunConfig(
            task=task,
            mode=mode,
            validation_ratio=args.validation_ratio,
            outer_folds=outer_folds,
            inner_folds=args.inner_folds,
            random_state=args.seed,
            n_jobs=args.jobs,
            output_dir=output_dir,
            strict=args.strict,
            resume=not args.no_resume,
            legacy_export=args.legacy_export,
        )
        dataset = load_dataset(data_path, task)
        split = split_dataset(dataset, config)
        if args.full_grid:
            specs = enumerate_specs(task, seed=args.seed)
        else:
            models = CLASSIFICATION_MODELS if task is TaskType.CLASSIFICATION else REGRESSION_MODELS
            specs = enumerate_specs(
                task,
                _csv_list(args.preprocessing, QUICK_PREPROCESSING),
                _csv_list(args.selectors, QUICK_SELECTORS),
                _csv_list(args.models, QUICK_MODELS[task] if not args.models else models),
                seed=args.seed,
            )
        engine = EvaluationEngine(split, config, logger=print)
        try:
            outputs = engine.run(specs)
            if requested_export:
                _print_export(engine.export_candidate(outputs.results, specs, **_export_options(args)))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        create_visualization_suite(outputs.results, output_dir / "plots", task)
        print(f"Results: {outputs.results_csv}")
        print(f"Leaderboard: {outputs.leaderboard_csv}")
        print(f"Default example pipeline: {outputs.default_pipeline}")
        return 0

    from .gui import launch_gui

    launch_gui(task)
    return 0
