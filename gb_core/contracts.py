from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


SOFTWARE_VERSION = "0.0.0"
SCHEMA_VERSION = "0.0"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class EvaluationMode(str, Enum):
    FULL = "full"
    STANDARD = "standard"
    EXPLORATION = "exploration"


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED_DEPENDENCY = "FAILED_DEPENDENCY"
    FAILED_PREPROCESSING = "FAILED_PREPROCESSING"
    FAILED_SELECTOR = "FAILED_SELECTOR"
    FAILED_INNER_CV = "FAILED_INNER_CV"
    FAILED_MODEL_FIT = "FAILED_MODEL_FIT"
    FAILED_OUTER_CV = "FAILED_OUTER_CV"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_EXPORT = "FAILED_EXPORT"
    STOPPED = "STOPPED"


PREPROCESSING_METHODS = [
    "Raw",
    "Baseline correction",
    "Max normalization",
    "Area normalization",
    "Range normalization",
    "Center and scale",
    "Log transform",
    "Centering",
    "Pareto scaling",
    "SNV",
    "MSC",
    "SG derivative",
    "Simple 1st derivative",
    "Simple 2nd derivative",
    "SG smoothing",
]

SG_PREPROCESSING_METHODS = frozenset({"Baseline correction", "SG derivative", "SG smoothing"})

SELECTOR_METHODS = ["None", "VIP", "GA", "iPLS", "CARS", "SFLA"]

CLASSIFICATION_MODELS = [
    "PLS-DA",
    "LDA",
    "SVM-DA (linear)",
    "SVM-DA (rbf)",
    "Random Forest",
    "XGBoost",
    "Gradient Boosting",
    "KNN",
    "Logistic Regression",
    "Naive Bayes",
    "Neural Network",
]

REGRESSION_MODELS = [
    "PLS Regression",
    "Linear Regression",
    "SVR (linear)",
    "SVR (rbf)",
    "Random Forest Regression",
    "XGBoost Regression",
    "Gradient Boosting Regression",
    "KNN Regression",
    "Ridge Regression",
    "Bayesian Ridge",
    "Neural Network Regression",
]


@dataclass(frozen=True)
class PipelineSpec:
    task: TaskType
    preprocessing: str
    selector: str
    model: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    @property
    def pipeline_id(self) -> str:
        payload = json.dumps(
            {
                "task": self.task.value,
                "preprocessing": self.preprocessing,
                "selector": self.selector,
                "model": self.model,
                "hyperparameters": self.hyperparameters,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["task"] = self.task.value
        result["pipeline_id"] = self.pipeline_id
        return result


@dataclass
class RunConfig:
    task: TaskType
    mode: EvaluationMode = EvaluationMode.STANDARD
    validation_ratio: float = 0.2
    outer_folds: int = 5
    inner_folds: int = 3
    random_state: int = 42
    n_jobs: int = 1
    output_dir: Path = Path("gb_output")
    strict: bool = False
    resume: bool = True
    selector_penalty: float = 0.02
    def __post_init__(self) -> None:
        self.task = TaskType(self.task)
        self.mode = EvaluationMode(self.mode)
        self.output_dir = Path(self.output_dir)
        if not 0 < float(self.validation_ratio) < 0.5:
            raise ValueError("validation_ratio must be between 0 and 0.5.")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must be at least 2.")
        if self.n_jobs < 0:
            raise ValueError("n_jobs must be 0 (automatic) or a positive integer.")

    @property
    def resolved_n_jobs(self) -> int:
        # Auto: 80% of logical CPUs; hard cap 48 (Windows ProcessPool limit is 61).
        if self.n_jobs > 0:
            workers = self.n_jobs
        else:
            workers = max(1, math.floor((os.cpu_count() or 1) * 0.8))
        return max(1, min(workers, 48))

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["task"] = self.task.value
        result["mode"] = self.mode.value
        result["output_dir"] = str(self.output_dir)
        return result


COMMON_RESULT_COLUMNS = [
    "Schema_Version",
    "Run_ID",
    "Pipeline_ID",
    "Task",
    "Mode",
    "Preprocessing",
    "Feature_Selection",
    "Model",
    "Status",
    "Eligible_For_Ranking",
    "Error_Message",
    "Traceback_Summary",
    "Actual_Backend",
    "Hyperparameters",
    "Seed",
    "N_Calibration",
    "N_Validation",
    "Input_Features",
    "Selected_Features",
    "Fit_Seconds",
]

CLASSIFICATION_METRICS = [
    "Accuracy",
    "Balanced_Accuracy",
    "Kappa",
    "MCC",
    "Macro_F1",
    "Sensitivity",
    "Specificity",
]

REGRESSION_METRICS = [
    "R2",
    "RMSE",
    "MAE",
    "RPD",
    "RPIQ",
    "RER",
    "Bias",
    "Slope",
    "Intercept",
]

OVERFIT_DIAGNOSTIC_COLUMNS = [
    "Overfit_Cal_vs_CV",
    "Overfit_Cal_vs_Val",
    "Overfit_CV_vs_Val",
]


def result_columns(task: TaskType) -> List[str]:
    metrics = CLASSIFICATION_METRICS if TaskType(task) is TaskType.CLASSIFICATION else REGRESSION_METRICS
    columns = list(COMMON_RESULT_COLUMNS)
    for stage in ("Cal", "CV", "Val"):
        columns.extend(f"{stage}_{metric}" for metric in metrics)
    columns.extend(
        [
            "Fold_Primary_SD",
            "Stability_Percentile",
            "Selection_Score",
            "Selection_Rank",
            "Val_Evaluated",
            "Val_Status",
            "Val_Error",
        ]
    )
    columns.extend(OVERFIT_DIAGNOSTIC_COLUMNS)
    return columns


def empty_result_row(task: TaskType) -> Dict[str, Any]:
    return {column: None for column in result_columns(task)}


def leaderboard_front_columns(task: TaskType) -> List[str]:
    """User-facing column order: rank, pipeline, score, then Cal/CV/Val cores."""
    task = TaskType(task)
    if task is TaskType.CLASSIFICATION:
        cores = [
            "Cal_Kappa",
            "Cal_MCC",
            "CV_Kappa",
            "CV_MCC",
            "CV_Macro_F1",
            "CV_Balanced_Accuracy",
            "Val_Kappa",
            "Val_MCC",
            "Val_Macro_F1",
            "Val_Balanced_Accuracy",
        ]
    else:
        cores = [
            "Cal_R2",
            "Cal_RMSE",
            "CV_R2",
            "CV_RMSE",
            "CV_RPD",
            "Val_R2",
            "Val_RMSE",
            "Val_RPD",
        ]
    return [
        "Selection_Rank",
        "Preprocessing",
        "Feature_Selection",
        "Model",
        "Selection_Score",
        "Stability_Percentile",
        *cores,
        *OVERFIT_DIAGNOSTIC_COLUMNS,
        "Selected_Features",
        "Hyperparameters",
        "Status",
        "Val_Evaluated",
        "Pipeline_ID",
        "Mode",
        "Actual_Backend",
        "Fit_Seconds",
        "Error_Message",
    ]


def enumerate_specs(
    task: TaskType,
    preprocessing: Optional[List[str]] = None,
    selectors: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    seed: int = 42,
) -> List[PipelineSpec]:
    task = TaskType(task)
    preprocessing = preprocessing or list(PREPROCESSING_METHODS)
    selectors = selectors or list(SELECTOR_METHODS)
    if models is None:
        models = list(CLASSIFICATION_MODELS if task is TaskType.CLASSIFICATION else REGRESSION_MODELS)
    return [
        PipelineSpec(task=task, preprocessing=pre, selector=selector, model=model, seed=seed)
        for pre in preprocessing
        for selector in selectors
        for model in models
    ]


def mode_uses_full_selector_budget(mode: EvaluationMode) -> bool:
    return EvaluationMode(mode) in {EvaluationMode.FULL, EvaluationMode.STANDARD}


def mode_uses_hyperparameter_grid(mode: EvaluationMode) -> bool:
    return EvaluationMode(mode) is not EvaluationMode.EXPLORATION
