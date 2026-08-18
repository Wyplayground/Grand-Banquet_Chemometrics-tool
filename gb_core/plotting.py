from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .contracts import TaskType


COLORS = {
    "cyan": "#4FC3F7",
    "teal": "#00796B",
    "orange": "#FF7043",
    "coral": "#EF5350",
    "yellow": "#FFF176",
    "blue_grey": "#607D8B",
}


def _eligible(results: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    frame = results[results["Eligible_For_Ranking"].astype(bool)].copy()
    return frame.sort_values("Selection_Rank").head(top_n)


def compare_models_plot(results: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> Path:
    output_path = Path(output_path)
    frame = _eligible(results, top_n)
    labels = frame["Model"].astype(str) + "\n" + frame["Preprocessing"].astype(str) + " / " + frame["Feature_Selection"].astype(str)
    x = np.arange(len(frame))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    axes[0].bar(x, frame["Selection_Score"], color=COLORS["orange"], alpha=0.88)
    axes[0].set_ylabel("Selection Score")
    axes[0].set_xticks(x, labels, rotation=90, fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, frame["Stability_Percentile"], color=COLORS["cyan"], alpha=0.88)
    axes[1].set_ylabel("Fold-stability percentile")
    axes[1].set_xticks(x, labels, rotation=90, fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_visualization_suite(results: pd.DataFrame, output_dir: str | Path, task: TaskType) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task = TaskType(task)
    frame = _eligible(results, 20)
    paths = [compare_models_plot(results, output_dir / "model_score_comparison.png")]
    if frame.empty:
        return paths

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(frame))
    primary = "CV_MCC" if task is TaskType.CLASSIFICATION else "CV_RMSE"
    axes[0, 0].plot(x, frame[primary], marker="o", color=COLORS["teal"])
    axes[0, 0].set_title(f"{primary} across top pipelines")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].scatter(frame["Stability_Percentile"], frame["Selection_Score"], color=COLORS["teal"])
    axes[0, 1].set_xlabel("Fold-stability percentile")
    axes[0, 1].set_ylabel("Selection Score")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].scatter(frame["Selected_Features"], frame["Selection_Score"], color=COLORS["coral"])
    axes[1, 0].set_xlabel("Selected features")
    axes[1, 0].set_ylabel("Selection Score")
    axes[1, 0].grid(alpha=0.25)
    model_counts = frame["Model"].value_counts()
    axes[1, 1].barh(model_counts.index, model_counts.values, color=COLORS["yellow"], edgecolor=COLORS["teal"])
    axes[1, 1].set_title("Models represented in top pipelines")
    stage_path = output_dir / "cv_selection_diagnostics.png"
    fig.savefig(stage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(stage_path)

    best = frame.sort_values("Selection_Rank").iloc[0]
    prediction_path = output_dir.parent / "predictions" / f"{best['Pipeline_ID']}_cv.csv"
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        if task is TaskType.REGRESSION:
            observed = pd.to_numeric(predictions["Observed"], errors="coerce")
            predicted = pd.to_numeric(predictions["CV_Predicted"], errors="coerce")
            low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
            axes[0].scatter(observed, predicted, color=COLORS["teal"], alpha=0.75)
            axes[0].plot([low, high], [low, high], color=COLORS["coral"], linestyle="--")
            axes[0].set_xlabel("Observed")
            axes[0].set_ylabel("CV predicted")
            residual = predicted - observed
            axes[1].scatter(predicted, residual, color=COLORS["orange"], alpha=0.75)
            axes[1].axhline(0, color=COLORS["blue_grey"], linestyle="--")
            axes[1].set_xlabel("CV predicted")
            axes[1].set_ylabel("Residual")
        else:
            observed = predictions.get("Observed_Label", predictions["Observed"]).astype(str)
            predicted = predictions.get("CV_Predicted_Label", predictions["CV_Predicted"]).astype(str)
            labels = sorted(set(observed).union(predicted))
            matrix = confusion_matrix(observed, predicted, labels=labels)
            ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axes[0], colorbar=False, cmap="Blues")
            axes[0].set_title("Best pipeline CV confusion matrix")
            accuracy_by_class = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
            axes[1].bar(labels, accuracy_by_class, color=COLORS["orange"])
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("Class sensitivity")
            axes[1].tick_params(axis="x", rotation=45)
        diagnostic_path = output_dir / ("regression_cv_diagnostics.png" if task is TaskType.REGRESSION else "classification_cv_diagnostics.png")
        fig.savefig(diagnostic_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(diagnostic_path)
    return paths
