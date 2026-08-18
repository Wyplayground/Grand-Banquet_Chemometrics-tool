from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .contracts import (
    RunStatus,
    TaskType,
    leaderboard_front_columns,
    result_columns,
)


EPS = 1e-12


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    sensitivities: List[float] = []
    specificities: List[float] = []
    total = matrix.sum()
    for index in range(len(labels)):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        sensitivities.append(float(tp / (tp + fn)) if tp + fn else np.nan)
        specificities.append(float(tn / (tn + fp)) if tn + fp else np.nan)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Sensitivity": float(np.nanmean(sensitivities)),
        "Specificity": float(np.nanmean(specificities)),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    q75, q25 = np.percentile(y_true, [75, 25]) if len(y_true) else (0.0, 0.0)
    value_range = float(np.ptp(y_true)) if len(y_true) else 0.0
    bias = float(np.mean(y_pred - y_true)) if len(y_true) else np.nan
    if len(y_true) > 1 and np.std(y_true) > EPS:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
    else:
        slope, intercept = np.nan, np.nan
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "RMSE": rmse,
        "MAE": mae,
        "RPD": sd / rmse if rmse > EPS else np.inf,
        "RPIQ": float(q75 - q25) / rmse if rmse > EPS else np.inf,
        "RER": value_range / rmse if rmse > EPS else np.inf,
        "Bias": bias,
        "Slope": float(slope),
        "Intercept": float(intercept),
    }


def compute_metrics(task: TaskType, y_true, y_pred) -> Dict[str, float]:
    return classification_metrics(y_true, y_pred) if TaskType(task) is TaskType.CLASSIFICATION else regression_metrics(y_true, y_pred)


def add_stage_metrics(row: Dict[str, object], prefix: str, metrics: Dict[str, float]) -> None:
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def primary_fold_value(task: TaskType, metrics: Dict[str, float]) -> float:
    return float(metrics["MCC"] if TaskType(task) is TaskType.CLASSIFICATION else metrics["RMSE"])


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    # Quantize before ranking so mathematically tied metrics remain tied after CSV round-trips.
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).round(12)
    if values.notna().sum() <= 1:
        return pd.Series(np.where(values.notna(), 1.0, np.nan), index=values.index, dtype=float)
    return values.rank(method="average", pct=True, ascending=higher_is_better)


def rank_results(frame: pd.DataFrame, task: TaskType) -> pd.DataFrame:
    """Rank candidates using calibration-set cross-validation evidence only."""
    task = TaskType(task)
    result = frame.copy()
    eligible = (
        result["Status"].astype(str).eq(RunStatus.SUCCESS.value)
        & result["Eligible_For_Ranking"].astype(bool)
    )
    result["Stability_Percentile"] = np.nan
    result["Selection_Score"] = np.nan
    result["Selection_Rank"] = np.nan
    working = result.loc[eligible].copy()
    if working.empty:
        return result

    stability = _percentile(working["Fold_Primary_SD"], higher_is_better=False)
    if task is TaskType.CLASSIFICATION:
        components = {
            "kappa": _percentile(working["CV_Kappa"]),
            "mcc": _percentile(working["CV_MCC"]),
            "macro_f1": _percentile(working["CV_Macro_F1"]),
            "balanced_accuracy": _percentile(working["CV_Balanced_Accuracy"]),
            "stability": stability,
        }
        score = (
            0.30 * components["kappa"]
            + 0.25 * components["mcc"]
            + 0.20 * components["macro_f1"]
            + 0.15 * components["balanced_accuracy"]
            + 0.10 * components["stability"]
        )
    else:
        components = {
            "inverse_rmse": _percentile(working["CV_RMSE"], higher_is_better=False),
            "r2": _percentile(working["CV_R2"]),
            "rpd": _percentile(working["CV_RPD"]),
            "rpiq": _percentile(working["CV_RPIQ"]),
            "stability": stability,
        }
        score = (
            0.30 * components["inverse_rmse"]
            + 0.25 * components["r2"]
            + 0.20 * components["rpd"]
            + 0.15 * components["rpiq"]
            + 0.10 * components["stability"]
        )

    result.loc[working.index, "Stability_Percentile"] = stability.round(12)
    result.loc[working.index, "Selection_Score"] = score.round(12)
    rankable = result.loc[working.index].loc[result.loc[working.index, "Selection_Score"].notna()]
    task_tie = "CV_MCC" if task is TaskType.CLASSIFICATION else "CV_RMSE"
    ordered = rankable.sort_values(
        ["Selection_Score", task_tie, "Selected_Features", "Pipeline_ID"],
        ascending=[False, task is TaskType.REGRESSION, True, True],
        kind="mergesort",
    )
    result.loc[ordered.index, "Selection_Rank"] = np.arange(1, len(ordered) + 1)
    return result.sort_values(["Selection_Rank", "Pipeline_ID"], na_position="last", kind="mergesort").reset_index(drop=True)


def add_overfit_diagnostics(frame: pd.DataFrame, task: TaskType) -> pd.DataFrame:
    """Report-only Cal/CV/Val optimism diagnostics; never used for ranking."""
    task = TaskType(task)
    result = frame.copy()
    if task is TaskType.CLASSIFICATION:
        cal = pd.to_numeric(result.get("Cal_MCC"), errors="coerce")
        cv = pd.to_numeric(result.get("CV_MCC"), errors="coerce")
        val = pd.to_numeric(result.get("Val_MCC"), errors="coerce")
        result["Overfit_Cal_vs_CV"] = cal - cv
        result["Overfit_Cal_vs_Val"] = cal - val
        result["Overfit_CV_vs_Val"] = cv - val
    else:
        cal = pd.to_numeric(result.get("Cal_R2"), errors="coerce")
        cv = pd.to_numeric(result.get("CV_R2"), errors="coerce")
        val = pd.to_numeric(result.get("Val_R2"), errors="coerce")
        result["Overfit_Cal_vs_CV"] = cal - cv
        result["Overfit_Cal_vs_Val"] = cal - val
        result["Overfit_CV_vs_Val"] = cv - val
    return result


def reorder_leaderboard_columns(frame: pd.DataFrame, task: TaskType) -> pd.DataFrame:
    front = [column for column in leaderboard_front_columns(task) if column in frame.columns]
    remainder = [column for column in frame.columns if column not in front]
    # Prefer schema order for remainder when available.
    schema = [column for column in result_columns(task) if column in remainder]
    extras = [column for column in remainder if column not in schema]
    return frame.loc[:, front + schema + extras]
