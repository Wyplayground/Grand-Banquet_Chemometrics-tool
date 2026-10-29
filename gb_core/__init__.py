"""Shared Grand Banquet engine for classification and regression."""

from .contracts import (
    CLASSIFICATION_MODELS,
    PREPROCESSING_METHODS,
    REGRESSION_MODELS,
    SELECTOR_METHODS,
    EvaluationMode,
    PipelineSpec,
    RunConfig,
    RunStatus,
    TaskType,
)

__version__ = "0.0.0"

__all__ = [
    "TaskType",
    "EvaluationMode",
    "RunStatus",
    "PipelineSpec",
    "RunConfig",
    "PREPROCESSING_METHODS",
    "SELECTOR_METHODS",
    "CLASSIFICATION_MODELS",
    "REGRESSION_MODELS",
]
