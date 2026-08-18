from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
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
    SOFTWARE_VERSION,
    TaskType,
    empty_result_row,
    mode_uses_hyperparameter_grid,
    result_columns,
)
from .data import DatasetSplit
from .metrics import (
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


@dataclass
class RunOutputs:
    results: pd.DataFrame
    results_csv: Path
    leaderboard_csv: Path
    leaderboard_html: Path
    manifest: Path
    best_pipeline: Optional[Path]
    validation_predictions: Optional[Path]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluate_pipeline_worker(split, config, run_id, spec):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return EvaluationEngine(split, config).evaluate_one(run_id, spec)


def _evaluate_final_metrics_worker(split, config, spec):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return EvaluationEngine(split, config).evaluate_final_metrics(spec)


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
        if config.task is TaskType.CLASSIFICATION:
            self.label_encoder = LabelEncoder().fit(np.asarray(split.y_cal))
            unseen = set(np.asarray(split.y_validation)).difference(self.label_encoder.classes_)
            if unseen:
                raise ValueError(f"Independent validation contains unseen classes: {sorted(map(str, unseen))}")
            self.y_cal = self.label_encoder.transform(np.asarray(split.y_cal))
            self.y_validation = self.label_encoder.transform(np.asarray(split.y_validation))
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

    def _run_id(self, specs: Sequence[PipelineSpec]) -> str:
        scientific_config = {
            "task": self.config.task.value,
            "software_version": SOFTWARE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "mode": self.config.mode.value,
            "validation_ratio": self.config.validation_ratio,
            "outer_folds": self.config.outer_folds,
            "inner_folds": self.config.inner_folds,
            "random_state": self.config.random_state,
            "selector_penalty": self.config.selector_penalty,
        }
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
        """Search SG window/order using only the current training data.

        Uses the pipeline model on all features (no selector) so SG search stays
        inexpensive relative to the outer two-level cross-validation and selector loop.
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

    def evaluate_one(self, run_id: str, spec: PipelineSpec) -> Dict[str, object]:
        row = self._base_row(run_id, spec)
        started = time.time()
        phase = RunStatus.FAILED_OUTER_CV
        try:
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

            row["Fit_Seconds"] = time.time() - started

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
            return row
        except InterruptedError as exc:
            return self._failure_row(row, RunStatus.STOPPED, exc, started)
        except ImportError as exc:
            return self._failure_row(row, RunStatus.FAILED_DEPENDENCY, exc, started)
        except Exception as exc:
            return self._failure_row(row, phase, exc, started)

    def evaluate_final_metrics(self, spec: PipelineSpec) -> Dict[str, object]:
        """Refit one ranked pipeline on all calibration data and report Cal/Val metrics.

        This method runs only after Selection_Rank has been fixed. Its return
        values never change ranking eligibility, Selection_Score, or Selection_Rank.
        """
        started = time.time()
        updates: Dict[str, object] = {
            "Val_Evaluated": True,
            "Val_Status": "NOT_EVALUATED",
            "Val_Error": "",
        }
        groups = np.asarray(self.split.groups_cal) if self.split.groups_cal is not None else None
        try:
            preprocessor, selector, model, best_params = self._fit_components(
                spec, self.X_cal, self.y_cal, groups=groups
            )
            cal_prediction = np.asarray(
                model.predict(selector.transform(preprocessor.transform(self.X_cal)))
            ).ravel()
            val_prediction = np.asarray(
                model.predict(selector.transform(preprocessor.transform(self.X_validation)))
            ).ravel()
            add_stage_metrics(updates, "Cal", compute_metrics(spec.task, self.y_cal, cal_prediction))
            add_stage_metrics(updates, "Val", compute_metrics(spec.task, self.y_validation, val_prediction))
            updates.update(
                {
                    "Val_Status": "SUCCESS",
                    "Selected_Features": int(len(selector.get_support(indices=True))),
                    "Actual_Backend": backend_name(model),
                    "_Full_Cal_Hyperparameters": best_params,
                    "_Final_Fit_Seconds": time.time() - started,
                }
            )

            prediction_dir = self.output_dir / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            val_frame = pd.DataFrame(
                {
                    "Sample name": self.split.sample_validation,
                    "Observed": self.y_validation,
                    "Val_Predicted": val_prediction,
                }
            )
            if self.split.groups_validation is not None:
                val_frame["Group"] = self.split.groups_validation.astype(str)
            if self.label_encoder is not None:
                val_frame["Observed_Label"] = self.label_encoder.inverse_transform(self.y_validation.astype(int))
                val_frame["Val_Predicted_Label"] = self.label_encoder.inverse_transform(val_prediction.astype(int))
            val_frame.to_csv(
                prediction_dir / f"{spec.pipeline_id}_val.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as exc:
            updates["Val_Status"] = RunStatus.FAILED_VALIDATION.value
            updates["Val_Error"] = f"{type(exc).__name__}: {exc}"
            updates["_Final_Fit_Seconds"] = time.time() - started
        return updates

    def _run_final_metrics(
        self,
        ranked: pd.DataFrame,
        specs: Sequence[PipelineSpec],
        run_id: str,
    ) -> pd.DataFrame:
        """Add Cal/Val results after ranking without changing the frozen rank."""
        result = ranked.copy()
        spec_by_id = {spec.pipeline_id: spec for spec in specs}
        pending: List[Tuple[int, PipelineSpec]] = []
        for row_index, row in result.iterrows():
            if not bool(row.get("Eligible_For_Ranking")) or pd.isna(row.get("Selection_Rank")):
                continue
            if bool(row.get("Val_Evaluated")):
                continue
            pipeline_id = str(row["Pipeline_ID"])
            pending.append((row_index, spec_by_id[pipeline_id]))

        def apply_updates(row_index: int, spec: PipelineSpec, updates: Dict[str, object]) -> None:
            full_cal_params = updates.pop("_Full_Cal_Hyperparameters", None)
            final_seconds = float(updates.pop("_Final_Fit_Seconds", 0.0) or 0.0)
            for key, value in updates.items():
                if key in result.columns:
                    result.at[row_index, key] = value
            if full_cal_params is not None:
                try:
                    fold_params = json.loads(str(result.at[row_index, "Hyperparameters"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    fold_params = []
                result.at[row_index, "Hyperparameters"] = json.dumps(
                    _jsonable({"folds": fold_params, "full_cal": full_cal_params}),
                    sort_keys=True,
                )
            prior_seconds = pd.to_numeric(pd.Series([result.at[row_index, "Fit_Seconds"]]), errors="coerce").iloc[0]
            result.at[row_index, "Fit_Seconds"] = (float(prior_seconds) if np.isfinite(prior_seconds) else 0.0) + final_seconds
            self.checkpoint.put(run_id, spec, _jsonable(result.loc[row_index].to_dict()))

        workers = min(self.config.resolved_n_jobs, max(1, len(pending)))
        if workers == 1:
            for completed, (row_index, spec) in enumerate(pending, 1):
                updates = self.evaluate_final_metrics(spec)
                apply_updates(row_index, spec, updates)
                self.logger(f"VALIDATION {completed}/{len(pending)}: {spec.preprocessing} / {spec.selector} / {spec.model}")
        elif pending:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for row_index, spec in pending:
                    future = executor.submit(_evaluate_final_metrics_worker, self.split, self.config, spec)
                    futures[future] = (row_index, spec)
                completed = 0
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        row_index, spec = futures.pop(future)
                        apply_updates(row_index, spec, future.result())
                        completed += 1
                        self.logger(
                            f"VALIDATION {completed}/{len(pending)}: {spec.preprocessing} / {spec.selector} / {spec.model}"
                        )
        return result

    def _finalize_rank_one(
        self,
        ranked: pd.DataFrame,
        specs: Sequence[PipelineSpec],
    ) -> Tuple[pd.DataFrame, Optional[Path], Optional[Path]]:
        """Export Rank 1 and refresh its Cal/Val metrics from the saved final fit.

        Every successful pipeline already carries report-only Val_* metrics;
        ranking was fixed previously using CV_* only.
        """
        result = ranked.copy()
        successful = result[
            result["Eligible_For_Ranking"].astype(bool) & result["Selection_Rank"].notna()
        ]
        if successful.empty:
            return result, None, None
        row_index = successful["Selection_Rank"].astype(float).idxmin()
        best_id = str(result.loc[row_index, "Pipeline_ID"])
        spec = next(item for item in specs if item.pipeline_id == best_id)
        groups = np.asarray(self.split.groups_cal) if self.split.groups_cal is not None else None
        try:
            preprocessor, selector, model, best_params = self._fit_components(
                spec, self.X_cal, self.y_cal, groups=groups
            )
            X_validation_selected = selector.transform(preprocessor.transform(self.X_validation))
            prediction = np.asarray(model.predict(X_validation_selected)).ravel()
            for name, value in compute_metrics(spec.task, self.y_validation, prediction).items():
                result.loc[row_index, f"Val_{name}"] = value
            # Refresh Cal metrics from the same final refit used for validation.
            cal_prediction = np.asarray(
                model.predict(selector.transform(preprocessor.transform(self.X_cal)))
            ).ravel()
            for name, value in compute_metrics(spec.task, self.y_cal, cal_prediction).items():
                result.loc[row_index, f"Cal_{name}"] = value
            result.loc[row_index, "Val_Evaluated"] = True
            result.loc[row_index, "Val_Status"] = "SUCCESS"
            result.loc[row_index, "Val_Error"] = ""
            result.loc[row_index, "Selected_Features"] = int(len(selector.get_support(indices=True)))
            try:
                recorded = json.loads(str(result.loc[row_index, "Hyperparameters"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                recorded = []
            fold_params = recorded.get("folds", []) if isinstance(recorded, dict) else recorded
            result.loc[row_index, "Hyperparameters"] = json.dumps(
                _jsonable({"folds": fold_params, "full_cal": best_params}),
                sort_keys=True,
            )
            result.loc[row_index, "Actual_Backend"] = backend_name(model)

            validation_frame = pd.DataFrame(
                {
                    "Sample name": self.split.sample_validation,
                    "Observed": self.y_validation,
                    "Predicted": prediction,
                }
            )
            if self.split.groups_validation is not None:
                validation_frame["Group"] = self.split.groups_validation.astype(str)
            if self.label_encoder is not None:
                validation_frame["Observed_Label"] = self.label_encoder.inverse_transform(
                    self.y_validation.astype(int)
                )
                validation_frame["Predicted_Label"] = self.label_encoder.inverse_transform(
                    prediction.astype(int)
                )
            validation_path = self.output_dir / "rank1_validation_predictions.csv"
            validation_frame.to_csv(validation_path, index=False, encoding="utf-8-sig")

            selected_names = selector.get_feature_names_out(self.feature_names).tolist()
            bundle = FinalPipelineBundle(
                task=spec.task,
                spec=spec.to_dict(),
                preprocessor=preprocessor,
                selector=selector,
                model=model,
                feature_names=self.feature_names,
                selected_feature_names=selected_names,
                dataset_fingerprint=self.split.fingerprint,
                label_encoder=self.label_encoder,
            )
            bundle_path = save_bundle(
                bundle, self.output_dir / f"best_{spec.task.value}_pipeline.joblib"
            )
            write_manifest(
                self.output_dir / "rank1_final_refit.json",
                {
                    "pipeline_id": best_id,
                    "selection_rank": 1,
                    "selection_basis": "calibration-set two-level cross-validation only (shown as CV metrics)",
                    "validation_role": "independent Val_* reported for all successfully ranked pipelines; Rank 1 bundle exported here",
                    "selected_features": selected_names,
                    "hyperparameters": best_params,
                },
            )
            return result, bundle_path, validation_path
        except Exception as exc:
            result.loc[row_index, "Val_Evaluated"] = True
            result.loc[row_index, "Val_Status"] = RunStatus.FAILED_VALIDATION.value
            result.loc[row_index, "Val_Error"] = f"{type(exc).__name__}: {exc}"
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
        run_id = self._run_id(specs)
        rows_by_index: Dict[int, Dict[str, object]] = {}
        total = len(specs)
        pending: List[Tuple[int, PipelineSpec]] = []
        completed = 0
        for index, spec in enumerate(specs):
            cached = self.checkpoint.get(run_id, spec.pipeline_id) if self.config.resume else None
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
        ranked = self._run_final_metrics(ranked, specs, run_id)
        ranked, best_pipeline, validation_predictions = self._finalize_rank_one(ranked, specs)
        ranked = add_overfit_diagnostics(ranked, self.config.task)
        ranked = reorder_leaderboard_columns(ranked, self.config.task)
        results_csv = self.output_dir / "pipeline_results.csv"
        leaderboard_csv = self.output_dir / "leaderboard.csv"
        html_path = self.output_dir / "leaderboard.html"
        ranked.to_csv(results_csv, index=False, encoding="utf-8-sig")
        ranked.to_csv(leaderboard_csv, index=False, encoding="utf-8-sig")
        write_leaderboard_html(ranked, html_path, self.config.task)
        validation_attempted = ranked["Val_Evaluated"].astype(bool)
        validation_successful = validation_attempted & ranked["Val_Status"].eq("SUCCESS")
        validation_failed_rows = validation_attempted & ~ranked["Val_Status"].eq("SUCCESS")
        manifest_path = write_manifest(
            self.output_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "config": self.config.to_dict(),
                "dataset_fingerprint": self.split.fingerprint,
                "calibration_samples": self.split.sample_cal.tolist(),
                "validation_samples": self.split.sample_validation.tolist(),
                "pipeline_specs": [spec.to_dict() for spec in specs],
                "successful": int(ranked["Status"].eq(RunStatus.SUCCESS.value).sum()),
                "failed": int((~ranked["Status"].eq(RunStatus.SUCCESS.value)).sum()),
                "validation_successful": int(validation_successful.sum()),
                "validation_failed": int(validation_failed_rows.sum()),
                "best_pipeline": str(best_pipeline) if best_pipeline else None,
                "validation_predictions": str(validation_predictions) if validation_predictions else None,
                "selection_role": "ranking uses calibration-set two-level cross-validation only (metrics shown as CV_*)",
                "validation_role": "independent Val_* is added after ranks are frozen and cannot change Selection_Rank; Rank 1 also exports a bundle and rank1_validation_predictions.csv",
                "calibration_role": "full-calibration apparent metrics shown as Cal_* and not used for ranking",
                "training_subset_contract": "imputation, preprocessing and variable selection are refitted using every inner training subset before hyperparameter scoring",
            },
        )
        failures = ranked[~ranked["Status"].eq(RunStatus.SUCCESS.value)]
        eligible_missing = ranked[
            ranked["Eligible_For_Ranking"].astype(bool)
            & ranked["Selection_Score"].isna()
        ]
        validation_failed = ranked.loc[ranked["Selection_Rank"].eq(1), "Val_Status"].ne("SUCCESS").any()
        if self.config.strict and (
            not failures.empty or not eligible_missing.empty or len(rows) != total or validation_failed
        ):
            raise RuntimeError(
                f"Strict run failed: {len(failures)} failed pipelines, "
                f"{len(eligible_missing)} eligible rows with missing scores, "
                f"validation_failed={validation_failed}, {len(rows)}/{total} completed."
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
