from __future__ import annotations

import argparse
import sys
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
    RunConfig,
    TaskType,
    enumerate_specs,
)
from .data import load_dataset, split_dataset
from .engine import EvaluationEngine
from .plotting import create_visualization_suite


def _csv_list(value: Optional[str], defaults):
    return [item.strip() for item in value.split(",") if item.strip()] if value else list(defaults)


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
    parser = argparse.ArgumentParser(description=f"Grand Banquet v0.0 {task.value} GUI and command-line workflow")
    parser.add_argument("--data", metavar="CSV_OR_EXCEL")
    parser.add_argument("--output-dir", metavar="DIR")
    parser.add_argument("--mode", choices=[mode.value for mode in EvaluationMode], default=EvaluationMode.STANDARD.value)
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="independent validation fraction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=0, help="pipeline workers; 0 uses up to 80%% of logical CPUs")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preprocessing", help="comma-separated public preprocessing names")
    parser.add_argument("--selectors", help="comma-separated selector names")
    parser.add_argument("--models", help="comma-separated model names")
    parser.add_argument("--predict-model", metavar="JOBLIB")
    parser.add_argument("--predict-data", metavar="CSV_OR_EXCEL")
    parser.add_argument("--prediction-output", metavar="CSV")
    return parser


def main_for_task(task: TaskType, argv: Optional[Sequence[str]] = None) -> int:
    task = TaskType(task)
    parser = build_parser(task)
    args = parser.parse_args(argv)
    if args.predict_model or args.predict_data:
        if not args.predict_model or not args.predict_data:
            parser.error("--predict-model and --predict-data must be provided together")
        destination = _predict(args.predict_model, args.predict_data, args.prediction_output)
        print(f"Predictions written: {destination}")
        return 0
    if args.data:
        data_path = Path(args.data)
        output_dir = Path(args.output_dir) if args.output_dir else data_path.resolve().parent / f"{data_path.stem}_gb_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = EvaluationMode(args.mode)
        outer_folds = args.outer_folds or 5
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
        )
        dataset = load_dataset(data_path, task)
        split = split_dataset(dataset, config)
        models = CLASSIFICATION_MODELS if task is TaskType.CLASSIFICATION else REGRESSION_MODELS
        specs = enumerate_specs(
            task,
            _csv_list(args.preprocessing, PREPROCESSING_METHODS),
            _csv_list(args.selectors, SELECTOR_METHODS),
            _csv_list(args.models, models),
            seed=args.seed,
        )
        engine = EvaluationEngine(split, config, logger=print)
        try:
            outputs = engine.run(specs)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        create_visualization_suite(outputs.results, output_dir / "plots", task)
        print(f"Results: {outputs.results_csv}")
        print(f"Leaderboard: {outputs.leaderboard_csv}")
        print(f"Best pipeline: {outputs.best_pipeline}")
        return 0

    from .gui import launch_gui

    launch_gui(task)
    return 0
