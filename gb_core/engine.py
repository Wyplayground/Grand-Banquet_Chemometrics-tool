from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

from .artifacts import CheckpointStore, FinalPipelineBundle, save_bundle, write_manifest
from .contracts import (
    EvaluationMode,
    PipelineSpec,
    RunConfig,
    RunStatus,
    SCHEMA_VERSION,
    SG_PREPROCESSING_METHODS,
    TaskType,
    empty_result_row,
    mode_uses_hyperparameter_grid,
    result_columns,
)
from .data import DatasetSplit
from .metrics import (
    add_legacy_validation_aliases,
    add_overfit_diagnostics,
    add_stage_metrics,
    compute_metrics,
    primary_fold_value,
    rank_results,
    reorder_leaderboard_columns,
)
from .models import backend_name, cv_strategy, make_model, model_grid
from .reporting import write_leaderboard_html
from .selectors import GBSelector
from .transformers import (
    DEFAULT_SG_POLYORDER,
    DEFAULT_SG_WINDOW,
    GBPreprocessor,
    sg_param_candidates,
)


EVALUATION_REVISION = "aca-v2.5-review-2026-10-08-cars-layout-fix"


@dataclass
class RunOutputs:
    results: pd.DataFrame
    results_csv: Path
    leaderboard_csv: Path
    leaderboard_html: Path
    manifest: Path
    best_pipeline: Optional[Path]
    validation_predictions: Optional[Path]

    @property
    def default_pipeline(self) -> Optional[Path]:
        """Score-ordered default example; best_pipeline remains a compatibility alias."""
        return self.best_pipeline


@dataclass
class CandidateExport:
    pipeline_id: str
    bundle_path: Path
    manifest_path: Path
    validation_predictions: Optional[Path]
    validation_status: str
    validation_error: str
    calibration_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    selected_feature_names: List[str]
    hyperparameters: Dict[str, object]
    actual_backend: str


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluate_pipeline_worker(split, config, run_id, spec):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return EvaluationEngine(split, config).evaluate_one(run_id, spec)


class EvaluationEngine:
    def __init__(
        self,
        split: DatasetSplit,
        config: RunConfig,
        logger: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.split = split
        self.config = config
        self.logger = logger or (lambda message: None)
        self.should_stop = should_stop or (lambda: False)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Lazy: ProcessPool workers only call evaluate_one and must not open checkpoint.sqlite.
        self._checkpoint: Optional[CheckpointStore] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._validation_label_error: Optional[str] = None
        if config.task is TaskType.CLASSIFICATION:
            self.label_encoder = LabelEncoder().fit(np.asarray(split.y_cal))
            unseen = set(np.asarray(split.y_validation)).difference(self.label_encoder.classes_)
            if unseen:
                self._validation_label_error = f"Internal holdout contains unseen classes: {sorted(map(str, unseen))}"
            self.y_cal = self.label_encoder.transform(np.asarray(split.y_cal))
            self.y_validation = (
                np.asarray(split.y_validation) if unseen
                else self.label_encoder.transform(np.asarray(split.y_validation))
            )
        else:
            self.y_cal = np.asarray(split.y_cal, dtype=float)
            self.y_validation = np.asarray(split.y_validation, dtype=float)
        self.X_cal = split.X_cal.to_numpy(dtype=float)
        self.X_validation = split.X_validation.to_numpy(dtype=float)
        self.feature_names = [str(column) for column in split.X_cal.columns]

    @property
    def checkpoint(self) -> CheckpointStore:
        if self._checkpoint is None:
            self._checkpoint = CheckpointStore(self.output_dir / "checkpoint.sqlite")
        return self._checkpoint

    def _run_id(self, specs: Sequence[PipelineSpec], *, include_revision: bool = True) -> str:
        scientific_config = {
            "task": self.config.task.value,
            "schema_version": SCHEMA_VERSION,
            "mode": self.config.mode.value,
            "validation_ratio": self.config.validation_ratio,
            "outer_folds": self.config.outer_folds,
            "inner_folds": self.config.inner_folds,
            "random_state": self.config.random_state,
            "selector_penalty": self.config.selector_penalty,
        }
        if include_revision:
            scientific_config["evaluation_revision"] = EVALUATION_REVISION
        payload = json.dumps(
            {
                "fingerprint": self.split.fingerprint,
                "config": scientific_config,
                "pipelines": [spec.pipeline_id for spec in specs],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _outer_splitter(self):
        groups = self.split.groups_cal
        return cv_strategy(
            self.config.task,
            self.config.outer_folds,
            self.config.random_state,
            self.y_cal,
            None if groups is None else np.asarray(groups),
        )

    def _tuning_folds(self) -> int:
        return self.config.inner_folds if mode_uses_hyperparameter_grid(self.config.mode) else 2

    def _selector(self, spec: PipelineSpec, seed_offset: int = 0) -> GBSelector:
        return GBSelector(
            method=spec.selector,
            task=spec.task,
            model_name=spec.model,
            inner_folds=self._tuning_folds(),
            mode=self.config.mode,
            random_state=spec.seed + seed_offset,
            feature_penalty=self.config.selector_penalty,
        )

    def _inner_splits(self, spec: PipelineSpec, X, y, groups=None):
        splitter = cv_strategy(spec.task, self._tuning_folds(), spec.seed, y, groups)
        if isinstance(splitter, list):
            return splitter
        split_args = (X, y, groups) if groups is not None else (X, y)
        return list(splitter.split(*split_args))

    def _make_preprocessor(self, method: str, sg_params: Optional[Dict[str, int]] = None) -> GBPreprocessor:
        params = sg_params or {}
        return GBPreprocessor(
            method=method,
            sg_window=int(params.get("sg_window", DEFAULT_SG_WINDOW)),
            sg_polyorder=int(params.get("sg_polyorder", DEFAULT_SG_POLYORDER)),
        )

    def _search_sg_params(self, spec: PipelineSpec, X, y, groups=None) -> Dict[str, int]:
        """Fold-local supervised SG window/order search; Exploration keeps defaults.

        Uses the pipeline model on all features (no selector) so SG search stays
        cheap relative to the outer nested-CV + selector loop.
        """
        if spec.preprocessing not in SG_PREPROCESSING_METHODS:
            return {}
        if self.config.mode is EvaluationMode.EXPLORATION:
            return {"sg_window": DEFAULT_SG_WINDOW, "sg_polyorder": DEFAULT_SG_POLYORDER}

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        group_array = None if groups is None else np.asarray(groups)
        candidates = sg_param_candidates(X.shape[1])
        scored = []
        for window, order in candidates:
            fold_scores = []
            for fold, (train_idx, validation_idx) in enumerate(self._inner_splits(spec, X, y, group_array), 1):
                preprocessor = self._make_preprocessor(
                    spec.preprocessing, {"sg_window": window, "sg_polyorder": order}
                ).fit(X[train_idx], y[train_idx])
                X_train = preprocessor.transform(X[train_idx])
                X_validation = preprocessor.transform(X[validation_idx])
                model = make_model(spec.task, spec.model, random_state=spec.seed + fold)
                model.fit(X_train, y[train_idx])
                prediction = np.asarray(model.predict(X_validation)).ravel()
                metrics = compute_metrics(spec.task, y[validation_idx], prediction)
                score = metrics["MCC"] if spec.task is TaskType.CLASSIFICATION else -metrics["RMSE"]
                if np.isfinite(score):
                    fold_scores.append(float(score))
            if fold_scores:
                scored.append((float(np.mean(fold_scores)), -window, -order, window, order))
        if not scored:
            return {"sg_window": DEFAULT_SG_WINDOW, "sg_polyorder": DEFAULT_SG_POLYORDER}
        scored.sort(reverse=True)
        _, _, _, window, order = scored[0]
        return {"sg_window": int(window), "sg_polyorder": int(order)}

    def _tune_full_pipeline(
        self,
        spec: PipelineSpec,
        X,
        y,
        groups=None,
        sg_params: Optional[Dict[str, int]] = None,
    ) -> Dict[str, object]:
        """Tune model parameters with preprocessing and selection refitted per inner fold."""
        grid = model_grid(spec.task, spec.model, self.config.mode)
        if not grid:
            return {}

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        group_array = None if groups is None else np.asarray(groups)
        prepared_folds = []
        for fold, (train_idx, validation_idx) in enumerate(
            self._inner_splits(spec, X, y, group_array), 1
        ):
            preprocessor = self._make_preprocessor(spec.preprocessing, sg_params).fit(X[train_idx], y[train_idx])
            X_train = preprocessor.transform(X[train_idx])
            X_validation = preprocessor.transform(X[validation_idx])
            train_groups = None if group_array is None else group_array[train_idx]
            selector = self._selector(spec, seed_offset=fold).fit(
                X_train, y[train_idx], groups=train_groups
            )
            prepared_folds.append(
                (
                    selector.transform(X_train),
                    y[train_idx],
                    selector.transform(X_validation),
                    y[validation_idx],
                    fold,
                )
            )

        candidates = []
        for params in ParameterGrid(grid):
            fold_scores = []
            for X_train, y_train, X_validation, y_validation, fold in prepared_folds:
                model = make_model(spec.task, spec.model, random_state=spec.seed + fold)
                model.set_params(**params)
                model.fit(X_train, y_train)
                prediction = np.asarray(model.predict(X_validation)).ravel()
                metrics = compute_metrics(spec.task, y_validation, prediction)
                score = metrics["MCC"] if spec.task is TaskType.CLASSIFICATION else -metrics["RMSE"]
                if not np.isfinite(score):
                    raise ValueError(f"Non-finite inner-CV score for parameters {params}")
                fold_scores.append(float(score))
            candidates.append((float(np.mean(fold_scores)), json.dumps(params, sort_keys=True), dict(params)))
        if not candidates:
            raise ValueError(f"No hyperparameter candidates were generated for {spec.model}")
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _fit_components(self, spec: PipelineSpec, X, y, groups=None):
        sg_params = self._search_sg_params(spec, X, y, groups=groups)
        preprocessor = self._make_preprocessor(spec.preprocessing, sg_params).fit(X, y)
        X_pre = preprocessor.transform(X)
        selector = self._selector(spec).fit(X_pre, y, groups=groups)
        X_selected = selector.transform(X_pre)
        best_params = self._tune_full_pipeline(spec, X, y, groups=groups, sg_params=sg_params)
        merged = dict(best_params)
        merged.update(sg_params)
        model = make_model(spec.task, spec.model, random_state=spec.seed)
        model.set_params(**{key: value for key, value in best_params.items()})
        model.fit(X_selected, y)
        return preprocessor, selector, model, merged

    def _base_row(self, run_id: str, spec: PipelineSpec) -> Dict[str, object]:
        row = empty_result_row(spec.task)
        row.update(
            {
                "Schema_Version": SCHEMA_VERSION,
                "Run_ID": run_id,
                "Pipeline_ID": spec.pipeline_id,
                "Task": spec.task.value,
                "Mode": self.config.mode.value,
                "Preprocessing": spec.preprocessing,
                "Feature_Selection": "No selection" if spec.selector == "None" else spec.selector,
                "Model": spec.model,
                "Status": RunStatus.SUCCESS.value,
                "Eligible_For_Ranking": True,
                "Error_Message": "",
                "Traceback_Summary": "",
                "Actual_Backend": "",
                "Hyperparameters": "{}",
                "Seed": spec.seed,
                "N_Calibration": len(self.y_cal),
                "N_Validation": len(self.y_validation),
                "Input_Features": self.X_cal.shape[1],
                "Selected_Features": None,
                "Fit_Seconds": None,
                "Val_Evaluated": False,
                "Val_Status": "NOT_EVALUATED",
                "Val_Error": "",
            }
        )
        return row

    def _failure_row(self, row: Dict[str, object], status: RunStatus, exc: Exception, started: float) -> Dict[str, object]:
        row["Status"] = status.value
        row["Eligible_For_Ranking"] = False
        row["Error_Message"] = f"{type(exc).__name__}: {exc}"
        row["Traceback_Summary"] = "\n".join(traceback.format_exc().splitlines()[-12:])
        row["Fit_Seconds"] = time.time() - started
        return row

    def _evidence_files(self, spec: PipelineSpec) -> Dict[str, Path]:
        return {
            "cv": self.output_dir / "predictions" / f"{spec.pipeline_id}_cv.csv",
            "fold_metadata": self.output_dir / "fold_metadata" / f"{spec.pipeline_id}.json",
            "val": self.output_dir / "predictions" / f"{spec.pipeline_id}_val.csv",
        }

    def _evidence_marker(self, spec: PipelineSpec) -> Path:
        return self.output_dir / "predictions" / f"{spec.pipeline_id}_provenance.json"

    def _write_evidence_marker(self, run_id: str, spec: PipelineSpec, row: Dict[str, object]) -> None:
        hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            for name, path in self._evidence_files(spec).items()
        }
        self._evidence_marker(spec).write_text(json.dumps({
            "run_id": run_id, "pipeline_id": spec.pipeline_id,
            "dataset_fingerprint": self.split.fingerprint,
            "evaluation_revision": EVALUATION_REVISION,
            "validation_status": row["Val_Status"], "file_sha256": hashes,
        }, indent=2), encoding="utf-8")

    def _evidence_matches(self, run_id: str, spec: PipelineSpec, row: Dict[str, object]) -> bool:
        # Failures have no successful evidence to reuse; keep their explicit cached state.
        if row.get("Status") != RunStatus.SUCCESS.value:
            return True
        try:
            marker = json.loads(self._evidence_marker(spec).read_text(encoding="utf-8"))
            if not isinstance(marker, dict) or not isinstance(marker.get("file_sha256"), dict):
                return False
            if (marker.get("run_id") != run_id
                or marker.get("pipeline_id") != spec.pipeline_id
                or marker.get("dataset_fingerprint") != self.split.fingerprint
                or marker.get("evaluation_revision") != EVALUATION_REVISION
                or marker.get("validation_status") != row.get("Val_Status")):
                return False
            files = self._evidence_files(spec)
            for name, path in files.items():
                expected = marker["file_sha256"].get(name)
                if name in {"cv", "fold_metadata"} and expected is None:
                    return False
                if name == "val" and row.get("Val_Status") == "SUCCESS" and expected is None:
                    return False
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                if actual != expected:
                    return False
            return True
        except (OSError, ValueError, TypeError):
            return False

    def evaluate_one(self, run_id: str, spec: PipelineSpec) -> Dict[str, object]:
        row = self._base_row(run_id, spec)
        started = time.time()
        phase = RunStatus.FAILED_OUTER_CV
        try:
            # Never leave an earlier run's per-candidate evidence under this attempt's paths.
            for relative in (
                f"predictions/{spec.pipeline_id}_cv.csv",
                f"predictions/{spec.pipeline_id}_val.csv",
                f"fold_metadata/{spec.pipeline_id}.json",
                f"predictions/{spec.pipeline_id}_provenance.json",
            ):
                (self.output_dir / relative).unlink(missing_ok=True)
            oof = np.empty(len(self.y_cal), dtype=float if spec.task is TaskType.REGRESSION else int)
            oof_fold = np.zeros(len(self.y_cal), dtype=int)
            fold_values: List[float] = []
            selected_counts: List[int] = []
            fold_best_params: List[Dict[str, object]] = []
            fold_metadata: List[Dict[str, object]] = []
            last_model = None
            groups = np.asarray(self.split.groups_cal) if self.split.groups_cal is not None else None
            splitter = self._outer_splitter()
            split_args = (self.X_cal, self.y_cal, groups) if groups is not None else (self.X_cal, self.y_cal)
            split_iterator = splitter if isinstance(splitter, list) else splitter.split(*split_args)
            for fold, (train_idx, test_idx) in enumerate(split_iterator, 1):
                if self.should_stop():
                    raise InterruptedError("Run stopped by user.")
                phase = RunStatus.FAILED_PREPROCESSING
                train_groups = groups[train_idx] if groups is not None else None
                sg_params = self._search_sg_params(
                    spec, self.X_cal[train_idx], self.y_cal[train_idx], groups=train_groups
                )
                preprocessor = self._make_preprocessor(spec.preprocessing, sg_params).fit(
                    self.X_cal[train_idx], self.y_cal[train_idx]
                )
                X_train_pre = preprocessor.transform(self.X_cal[train_idx])
                X_test_pre = preprocessor.transform(self.X_cal[test_idx])
                phase = RunStatus.FAILED_SELECTOR
                selector = self._selector(spec).fit(X_train_pre, self.y_cal[train_idx], groups=train_groups)
                X_train_selected = selector.transform(X_train_pre)
                X_test_selected = selector.transform(X_test_pre)
                phase = RunStatus.FAILED_INNER_CV
                best_params = self._tune_full_pipeline(
                    spec,
                    self.X_cal[train_idx],
                    self.y_cal[train_idx],
                    groups=train_groups,
                    sg_params=sg_params,
                )
                merged_params = dict(best_params)
                merged_params.update(sg_params)
                phase = RunStatus.FAILED_MODEL_FIT
                model = make_model(spec.task, spec.model, random_state=spec.seed + fold)
                model.set_params(**best_params)
                model.fit(X_train_selected, self.y_cal[train_idx])
                prediction = np.asarray(model.predict(X_test_selected)).ravel()
                oof[test_idx] = prediction
                oof_fold[test_idx] = fold
                metrics = compute_metrics(spec.task, self.y_cal[test_idx], prediction)
                fold_values.append(primary_fold_value(spec.task, metrics))
                selected_counts.append(int(len(selector.get_support(indices=True))))
                fold_best_params.append(dict(merged_params))
                last_model = model
                fold_metadata.append(
                    {
                        "fold": fold,
                        "train_indices": train_idx.tolist(),
                        "test_indices": test_idx.tolist(),
                        "train_samples": self.split.sample_cal.iloc[train_idx].astype(str).tolist(),
                        "test_samples": self.split.sample_cal.iloc[test_idx].astype(str).tolist(),
                        "train_groups": None if groups is None else sorted(set(map(str, groups[train_idx]))),
                        "test_groups": None if groups is None else sorted(set(map(str, groups[test_idx]))),
                        "selected_indices": selector.get_support(indices=True).tolist(),
                        "selected_features": selector.get_feature_names_out(self.feature_names).tolist(),
                        "best_params": merged_params,
                        "metrics": metrics,
                    }
                )

            add_stage_metrics(row, "CV", compute_metrics(spec.task, self.y_cal, oof))
            row["Fold_Primary_SD"] = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
            row["Selected_Features"] = int(round(float(np.median(selected_counts))))
            row["Hyperparameters"] = json.dumps(_jsonable(fold_best_params), sort_keys=True)
            row["Actual_Backend"] = backend_name(last_model)

            # Apparent Cal metrics: full-calibration refit scored on calibration (not used for ranking).
            phase = RunStatus.FAILED_MODEL_FIT
            cal_preprocessor, cal_selector, cal_model, cal_params = self._fit_components(
                spec, self.X_cal, self.y_cal, groups=groups
            )
            cal_prediction = np.asarray(
                cal_model.predict(cal_selector.transform(cal_preprocessor.transform(self.X_cal)))
            ).ravel()
            add_stage_metrics(row, "Cal", compute_metrics(spec.task, self.y_cal, cal_prediction))
            row["Hyperparameters"] = json.dumps(_jsonable({"folds": fold_best_params, "full_cal": cal_params}), sort_keys=True)
            row["Selected_Features"] = int(len(cal_selector.get_support(indices=True)))
            row["Actual_Backend"] = backend_name(cal_model)

            phase = RunStatus.FAILED_EXPORT
            prediction_dir = self.output_dir / "predictions"
            metadata_dir = self.output_dir / "fold_metadata"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame = pd.DataFrame(
                {
                    "Sample name": self.split.sample_cal,
                    "Observed": self.y_cal,
                    "CV_Predicted": oof,
                    "Outer_Fold": oof_fold,
                }
            )
            if groups is not None:
                prediction_frame["Group"] = groups
            if self.label_encoder is not None:
                prediction_frame["Observed_Label"] = self.label_encoder.inverse_transform(self.y_cal.astype(int))
                prediction_frame["CV_Predicted_Label"] = self.label_encoder.inverse_transform(oof.astype(int))
            prediction_frame.to_csv(prediction_dir / f"{spec.pipeline_id}_cv.csv", index=False, encoding="utf-8-sig")
            (metadata_dir / f"{spec.pipeline_id}.json").write_text(
                json.dumps(_jsonable(fold_metadata), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # Persist calibration CV evidence before report-only internal holdout evaluation.
            # A holdout failure must not determine eligibility for CV-based ranking.
            row["Val_Evaluated"] = True
            try:
                self._check_validation_labels()
                val_prediction = np.asarray(
                    cal_model.predict(cal_selector.transform(cal_preprocessor.transform(self.X_validation)))
                ).ravel()
                val_metrics = compute_metrics(spec.task, self.y_validation, val_prediction)
                self._validation_frame(val_prediction, "Val_Predicted").to_csv(
                    prediction_dir / f"{spec.pipeline_id}_val.csv", index=False, encoding="utf-8-sig"
                )
                add_stage_metrics(row, "Val", val_metrics)
                row["Val_Status"] = "SUCCESS"
            except InterruptedError:
                raise
            except Exception as exc:
                row["Val_Status"] = RunStatus.FAILED_VALIDATION.value
                row["Val_Error"] = f"{type(exc).__name__}: {exc}"
                self.logger(f"HOLDOUT FAILED {spec.pipeline_id}: {row['Val_Error']}")
            self._write_evidence_marker(run_id, spec, row)
            row["Fit_Seconds"] = time.time() - started
            return row
        except InterruptedError as exc:
            return self._failure_row(row, RunStatus.STOPPED, exc, started)
        except ImportError as exc:
            return self._failure_row(row, RunStatus.FAILED_DEPENDENCY, exc, started)
        except Exception as exc:
            return self._failure_row(row, phase, exc, started)

    def _validation_frame(self, prediction, prediction_column="Predicted") -> pd.DataFrame:
        frame = pd.DataFrame({
            "Sample name": self.split.sample_validation,
            "Observed": self.y_validation,
            prediction_column: prediction,
        })
        if self.split.groups_validation is not None:
            frame["Group"] = np.asarray(self.split.groups_validation).astype(str)
        if self.label_encoder is not None:
            frame["Observed_Label"] = self.label_encoder.inverse_transform(self.y_validation.astype(int))
            frame[prediction_column + "_Label"] = self.label_encoder.inverse_transform(
                np.asarray(prediction).astype(int)
            )
        return frame

    def _check_validation_labels(self) -> None:
        if self._validation_label_error is not None:
            raise ValueError(self._validation_label_error)

    @staticmethod
    def _eligible_candidates(ranked: pd.DataFrame) -> pd.DataFrame:
        required = {"Pipeline_ID", "Status", "Eligible_For_Ranking", "Selection_Rank"}
        if not required.issubset(ranked.columns):
            raise ValueError("Candidate table is missing required result columns.")
        eligibility = ranked["Eligible_For_Ranking"].map(
            lambda value: str(value).lower() in {"true", "1"}
        )
        return ranked.loc[
            ranked["Status"].eq(RunStatus.SUCCESS.value)
            & eligibility & ranked["Selection_Rank"].notna()
        ].copy()

    def export_candidate(
        self,
        ranked: pd.DataFrame,
        specs: Sequence[PipelineSpec],
        pipeline_id: Optional[str] = None,
        metric: Optional[str] = None,
        higher_is_better: Optional[bool] = None,
        output_dir: Optional[str | Path] = None,
        selection_reason: Optional[str] = None,
        source_manifest: Optional[Dict[str, object]] = None,
    ) -> CandidateExport:
        """Refit a stored eligible candidate; do not rerun CV or alter its stored ranking.

        Selection may use an exact pipeline ID or a calibration CV metric. Viewing
        holdout results before choosing a candidate compromises confirmatory use.
        The metric table and specs must belong to the same verified dataset/run.
        """
        if pipeline_id is not None and metric is not None:
            raise ValueError("Choose a pipeline ID or a CV metric, not both.")
        if higher_is_better is not None and metric is None:
            raise ValueError("A metric is required when specifying its direction.")
        specs = list(specs)
        expected_run = self._run_id(specs)
        if source_manifest is not None:
            if source_manifest.get("dataset_fingerprint") != self.split.fingerprint:
                raise ValueError("Source manifest dataset fingerprint does not match current data.")
            manifest_ids = [item.get("pipeline_id") for item in source_manifest.get("pipeline_specs", [])]
            if manifest_ids != [spec.pipeline_id for spec in specs]:
                raise ValueError("Source manifest pipeline specifications do not match the supplied library.")
            expected_run = str(source_manifest.get("run_id", ""))
            accepted_runs = {self._run_id(specs), self._run_id(specs, include_revision=False)}
            if expected_run not in accepted_runs:
                raise ValueError("Source manifest does not match the current scientific configuration.")
        if "Run_ID" not in ranked or not ranked["Run_ID"].eq(expected_run).all():
            raise ValueError("Candidate results do not match the verified dataset/configuration/run.")
        candidates = self._eligible_candidates(ranked)
        if candidates.empty:
            raise ValueError("No eligible calibration-CV candidates are available.")
        if candidates["Pipeline_ID"].duplicated().any():
            raise ValueError("Candidate table contains duplicate pipeline IDs.")
        if pipeline_id is not None:
            candidates = candidates[candidates["Pipeline_ID"].eq(pipeline_id)]
            if candidates.empty:
                raise ValueError(f"Unknown or ineligible pipeline ID: {pipeline_id}")
            selection_basis = "explicit pipeline ID"
        elif metric is not None:
            if metric != "Selection_Score" and not metric.startswith("CV_"):
                raise ValueError("Selection metric must be Selection_Score or a CV_* metric.")
            if metric not in candidates.columns:
                raise ValueError(f"Unknown selection metric: {metric}")
            metric_values = pd.to_numeric(candidates[metric], errors="coerce")
            candidates = candidates.loc[np.isfinite(metric_values)].copy()
            if candidates.empty:
                raise ValueError(f"No finite values are available for {metric}.")
            candidates[metric] = metric_values.loc[candidates.index]
            if higher_is_better is None:
                lower_metrics = {"CV_RMSE", "CV_MAE", "CV_LogLoss"}
                higher_metrics = {
                    "Selection_Score", "CV_Accuracy", "CV_Balanced_Accuracy",
                    "CV_MCC", "CV_Kappa", "CV_Macro_F1", "CV_Sensitivity",
                    "CV_Specificity", "CV_R2", "CV_RPD", "CV_RPIQ", "CV_RER",
                }
                if metric in lower_metrics:
                    higher_is_better = False
                elif metric in higher_metrics:
                    higher_is_better = True
                else:
                    raise ValueError(f"Specify higher/lower direction explicitly for {metric}.")
            candidates = candidates.sort_values(
                [metric, "Selection_Rank", "Pipeline_ID"],
                ascending=[not higher_is_better, True, True], kind="mergesort",
            )
            selection_basis = "calibration CV metric" if metric.startswith("CV_") else "configured composite score"
        else:
            candidates = candidates.sort_values(["Selection_Rank", "Pipeline_ID"], kind="mergesort")
            selection_basis = "configured composite-score ordering (default example)"
        selected = candidates.iloc[0]
        selected_id = str(selected["Pipeline_ID"])
        matches = [spec for spec in specs if spec.pipeline_id == selected_id]
        if len(matches) != 1 or matches[0].task is not self.config.task:
            raise ValueError("Selected pipeline does not match exactly one specification for this task.")
        spec = matches[0]
        groups = None if self.split.groups_cal is None else np.asarray(self.split.groups_cal)
        preprocessor, selector, model, params = self._fit_components(
            spec, self.X_cal, self.y_cal, groups=groups
        )
        calibration_prediction = np.asarray(
            model.predict(selector.transform(preprocessor.transform(self.X_cal)))
        ).ravel()
        calibration_metrics = compute_metrics(spec.task, self.y_cal, calibration_prediction)
        names = selector.get_feature_names_out(self.feature_names).tolist()
        bundle = FinalPipelineBundle(
            task=spec.task, spec=spec.to_dict(), preprocessor=preprocessor,
            selector=selector, model=model, feature_names=self.feature_names,
            selected_feature_names=names, dataset_fingerprint=self.split.fingerprint,
            label_encoder=self.label_encoder,
        )
        base = Path(output_dir) if output_dir is not None else self.output_dir / "exports"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        export_dir = base / f"{selected_id}_{stamp}"
        export_dir.mkdir(parents=True, exist_ok=False)
        # Preserve a usable calibration-fitted model even if holdout prediction fails.
        bundle_path = save_bundle(bundle, export_dir / "pipeline.joblib")
        validation_path = None
        validation_metrics = {}
        validation_status = "SUCCESS"
        validation_error = ""
        try:
            self._check_validation_labels()
            prediction = np.asarray(
                model.predict(selector.transform(preprocessor.transform(self.X_validation)))
            ).ravel()
            metrics = compute_metrics(spec.task, self.y_validation, prediction)
            path = export_dir / "internal_holdout_predictions.csv"
            self._validation_frame(prediction).to_csv(path, index=False, encoding="utf-8-sig")
            validation_path = path
            validation_metrics = metrics
        except Exception as exc:
            validation_status = RunStatus.FAILED_VALIDATION.value
            validation_error = f"{type(exc).__name__}: {exc}"
            self.logger(f"HOLDOUT FAILED {selected_id}: {validation_error}")
        manifest_path = write_manifest(export_dir / "export_manifest.json", _jsonable({
            "pipeline_id": selected_id,
            "pipeline_spec": spec.to_dict(),
            "dataset_fingerprint": self.split.fingerprint,
            "source_run_id": selected.get("Run_ID"),
            "source_evaluation_revision": (
                source_manifest.get("evaluation_revision", "legacy working-tree implementation")
                if source_manifest is not None else EVALUATION_REVISION
            ),
            "evaluation_revision": EVALUATION_REVISION,
            "selection_basis": selection_basis,
            "selection_metric": metric,
            "higher_is_better": higher_is_better,
            "selection_reason": selection_reason or selection_basis,
            "stored_selection_rank": selected.get("Selection_Rank"),
            "stored_selection_score": selected.get("Selection_Score"),
            "stored_cv_metrics": {key: selected[key] for key in selected.index if key.startswith("CV_")},
            "cv_recomputed": False,
            "refit_role": "full-calibration fit of the selected candidate",
            "config": self.config.to_dict(),
            "calibration_observations": len(self.y_cal),
            "internal_holdout_observations": len(self.y_validation),
            "calibration_groups": None if groups is None else len(set(map(str, groups))),
            "internal_holdout_groups": None if self.split.groups_validation is None else int(pd.Series(self.split.groups_validation).nunique()),
            "calibration_metrics": calibration_metrics,
            "internal_holdout_metrics": validation_metrics,
            "validation_status": validation_status,
            "validation_error": validation_error,
            "validation_role": "same-source internal holdout; report-only, not used by automatic CV ranking",
            "selection_limit": "candidate CV is not an unbiased estimate of selecting from the entire library; repeated use of holdout results for choice is exploratory",
            "selected_features": names,
            "hyperparameters": params,
            "actual_backend": backend_name(model),
            "bundle_path": str(bundle_path),
            "validation_predictions": validation_path,
        }))
        return CandidateExport(
            selected_id, bundle_path, manifest_path, validation_path, validation_status,
            validation_error, calibration_metrics, validation_metrics, names, params,
            backend_name(model),
        )

    def _finalize_rank_one(
        self, ranked: pd.DataFrame, specs: Sequence[PipelineSpec],
    ) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
        """Export a score-ordered default example with legacy filename aliases."""
        result = ranked.copy()
        try:
            for name in (
                f"best_{self.config.task.value}_pipeline.joblib",
                "rank1_validation_predictions.csv", "rank1_final_refit.json",
            ):
                (self.output_dir / name).unlink(missing_ok=True)
            if self._eligible_candidates(result).empty:
                return result, None, None
            exported = self.export_candidate(result, specs)
            row_index = result.index[result["Pipeline_ID"].eq(exported.pipeline_id)][0]
            for prefix, metrics in (("Cal", exported.calibration_metrics), ("Val", exported.validation_metrics)):
                for name, value in metrics.items():
                    result.loc[row_index, f"{prefix}_{name}"] = value
            if exported.validation_status != "SUCCESS":
                for column in result.columns:
                    if column.startswith("Val_") and column not in {"Val_Status", "Val_Error", "Val_Evaluated"}:
                        result.loc[row_index, column] = np.nan
            result.loc[row_index, "Val_Evaluated"] = True
            result.loc[row_index, "Val_Status"] = exported.validation_status
            result.loc[row_index, "Val_Error"] = exported.validation_error
            result.loc[row_index, "Selected_Features"] = len(exported.selected_feature_names)
            result.loc[row_index, "Hyperparameters"] = json.dumps(_jsonable(exported.hyperparameters), sort_keys=True)
            result.loc[row_index, "Actual_Backend"] = exported.actual_backend
            # Older clients use these paths. "best" and "rank1" denote score order only.
            bundle_path = self.output_dir / f"best_{self.config.task.value}_pipeline.joblib"
            shutil.copy2(exported.bundle_path, bundle_path)
            validation_path = self.output_dir / "rank1_validation_predictions.csv"
            if exported.validation_predictions is not None:
                shutil.copy2(exported.validation_predictions, validation_path)
            else:
                validation_path.unlink(missing_ok=True)
                validation_path = None
            write_manifest(self.output_dir / "rank1_final_refit.json", {
                "pipeline_id": exported.pipeline_id,
                "selection_rank": 1,
                "selection_basis": "configured CV-based composite score; default illustrative candidate",
                "validation_role": "same-source internal holdout; report-only",
                "validation_status": exported.validation_status,
                "validation_error": exported.validation_error,
                "selected_features": exported.selected_feature_names,
                "hyperparameters": exported.hyperparameters,
                "export_manifest": str(exported.manifest_path),
                "legacy_filename_note": "rank1 and best are retained for compatibility, not a claim of unbiased superiority",
            })
            return result, bundle_path, validation_path
        except Exception as exc:
            self.logger(f"DEFAULT EXPORT FAILED: {type(exc).__name__}: {exc}")
            # Export or calibration-refit failure is not a holdout-prediction failure.
            return result, None, None

    def run(
        self,
        specs: Sequence[PipelineSpec],
        progress: Optional[Callable[[int, int, PipelineSpec], None]] = None,
    ) -> RunOutputs:
        specs = list(specs)
        if not specs:
            raise ValueError("At least one pipeline specification is required.")
        if any(spec.task is not self.config.task for spec in specs):
            raise ValueError("All pipeline specifications must match RunConfig.task.")
        if len({spec.pipeline_id for spec in specs}) != len(specs):
            raise ValueError("Pipeline specifications must have unique IDs.")
        run_id = self._run_id(specs)
        rows_by_index: Dict[int, Dict[str, object]] = {}
        total = len(specs)
        pending: List[Tuple[int, PipelineSpec]] = []
        completed = 0
        for index, spec in enumerate(specs):
            cached = self.checkpoint.get(run_id, spec.pipeline_id) if self.config.resume else None
            if cached is not None and not self._evidence_matches(run_id, spec, cached):
                self.logger(f"RECOMPUTE {spec.pipeline_id}: stored prediction/fold evidence is missing or belongs to another run")
                cached = None
            if cached is not None:
                rows_by_index[index] = cached
                completed += 1
                self.logger(f"RESUME {completed}/{total}: {spec.preprocessing} / {spec.selector} / {spec.model}")
                if progress:
                    progress(completed, total, spec)
            else:
                pending.append((index, spec))

        workers = min(self.config.resolved_n_jobs, max(1, len(pending)))
        self.logger(f"Pipeline workers: {workers} (logical CPUs: {os.cpu_count() or 1}, configured n_jobs: {self.config.n_jobs})")
        if workers == 1:
            for index, spec in pending:
                if self.should_stop():
                    break
                row = self.evaluate_one(run_id, spec)
                self.checkpoint.put(run_id, spec, _jsonable(row))
                rows_by_index[index] = row
                completed += 1
                self.logger(f"DONE {completed}/{total}: {spec.preprocessing} / {spec.selector} / {spec.model}")
                if progress:
                    progress(completed, total, spec)
        elif pending:
            iterator = iter(pending)
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for _ in range(workers):
                    try:
                        index, spec = next(iterator)
                    except StopIteration:
                        break
                    futures[executor.submit(_evaluate_pipeline_worker, self.split, self.config, run_id, spec)] = (index, spec)
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, spec = futures.pop(future)
                        row = future.result()
                        self.checkpoint.put(run_id, spec, _jsonable(row))
                        rows_by_index[index] = row
                        completed += 1
                        self.logger(f"DONE {completed}/{total}: {spec.preprocessing} / {spec.selector} / {spec.model}")
                        if progress:
                            progress(completed, total, spec)
                    if self.should_stop():
                        for future in futures:
                            future.cancel()
                        break
                    for _ in range(len(done)):
                        try:
                            index, spec = next(iterator)
                        except StopIteration:
                            break
                        futures[executor.submit(_evaluate_pipeline_worker, self.split, self.config, run_id, spec)] = (index, spec)

        rows = [rows_by_index[index] for index in sorted(rows_by_index)]

        frame = pd.DataFrame(rows, columns=result_columns(self.config.task))
        ranked = rank_results(frame, self.config.task)
        ranked, best_pipeline, validation_predictions = self._finalize_rank_one(ranked, specs)
        ranked = add_overfit_diagnostics(ranked, self.config.task)
        ranked = reorder_leaderboard_columns(ranked, self.config.task)
        results_csv = self.output_dir / "pipeline_results_v3.csv"
        leaderboard_csv = self.output_dir / "leaderboard_v3.csv"
        html_path = self.output_dir / "leaderboard_v3.html"
        ranked.to_csv(results_csv, index=False, encoding="utf-8-sig")
        ranked.to_csv(leaderboard_csv, index=False, encoding="utf-8-sig")
        if self.config.legacy_export:
            add_legacy_validation_aliases(ranked).to_csv(
                self.output_dir / "pipeline_results_v2_compatibility.csv",
                index=False,
                encoding="utf-8-sig",
            )
        write_leaderboard_html(ranked, html_path, self.config.task)
        manifest_path = write_manifest(
            self.output_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "evaluation_revision": EVALUATION_REVISION,
                "config": self.config.to_dict(),
                "dataset_fingerprint": self.split.fingerprint,
                "calibration_samples": self.split.sample_cal.tolist(),
                "validation_samples": self.split.sample_validation.tolist(),
                "calibration_observations": len(self.y_cal),
                "internal_holdout_observations": len(self.y_validation),
                "calibration_groups": None if self.split.groups_cal is None else int(pd.Series(self.split.groups_cal).nunique()),
                "internal_holdout_groups": None if self.split.groups_validation is None else int(pd.Series(self.split.groups_validation).nunique()),
                "pipeline_specs": [spec.to_dict() for spec in specs],
                "successful": int(ranked["Status"].eq(RunStatus.SUCCESS.value).sum()),
                "failed": int((~ranked["Status"].eq(RunStatus.SUCCESS.value)).sum()),
                "best_pipeline": str(best_pipeline) if best_pipeline else None,
                "default_pipeline": str(best_pipeline) if best_pipeline else None,
                "internal_holdout_failures": int(ranked["Val_Status"].eq(RunStatus.FAILED_VALIDATION.value).sum()),
                "default_export_failed": bool(self._eligible_candidates(ranked).shape[0] and best_pipeline is None),
                "validation_predictions": str(validation_predictions) if validation_predictions else None,
                "selection_role": "candidate-level calibration CV orders the configured library; the first row is a default example, not an unbiased estimate of library selection",
                "validation_role": "same-source internal holdout Val_* attempted per candidate before aggregate ranking; report-only; failure preserves CV eligibility; default example exported with legacy filenames",
                "calibration_role": "full-calibration apparent metrics shown as Cal_* and not used for ranking",
                "inner_cv_contract": "model tuning refits preprocessing and selection on inner-training folds; SG parameters are searched on the outer-training subset and held fixed during subsequent tuning; selector fitness sees outer-training preprocessing; no whole-library external resampling loop",
            },
        )
        failures = ranked[~ranked["Status"].eq(RunStatus.SUCCESS.value)]
        eligible_missing = ranked[
            ranked["Eligible_For_Ranking"].astype(bool)
            & ranked["Selection_Score"].isna()
        ]
        validation_failed = ranked["Val_Status"].eq(RunStatus.FAILED_VALIDATION.value).any()
        export_failed = not self._eligible_candidates(ranked).empty and best_pipeline is None
        if self.config.strict and (
            not failures.empty or not eligible_missing.empty or len(rows) != total or validation_failed or export_failed
        ):
            raise RuntimeError(
                f"Strict run failed: {len(failures)} failed pipelines, "
                f"{len(eligible_missing)} eligible rows with missing scores, "
                f"validation_failed={validation_failed}, export_failed={export_failed}, {len(rows)}/{total} completed."
            )
        return RunOutputs(
            ranked,
            results_csv,
            leaderboard_csv,
            html_path,
            manifest_path,
            best_pipeline,
            validation_predictions,
        )
