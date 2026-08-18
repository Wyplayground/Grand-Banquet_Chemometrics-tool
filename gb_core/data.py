from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .contracts import RunConfig, TaskType


@dataclass
class LoadedDataset:
    sample_names: pd.Series
    target: pd.Series
    features: pd.DataFrame
    groups: Optional[pd.Series] = None
    split_labels: Optional[pd.Series] = None
    fingerprint: str = ""


@dataclass
class DatasetSplit:
    sample_cal: pd.Series
    sample_validation: pd.Series
    y_cal: pd.Series
    y_validation: pd.Series
    X_cal: pd.DataFrame
    X_validation: pd.DataFrame
    groups_cal: Optional[pd.Series]
    groups_validation: Optional[pd.Series]
    fingerprint: str


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported data format. Use CSV, XLSX, or XLS.")


def _fingerprint(data: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(data, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(map(str, data.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def load_dataset(path: str | Path, task: TaskType) -> LoadedDataset:
    path = Path(path)
    data = _read_table(path)
    task = TaskType(task)
    target_name = "Class" if task is TaskType.CLASSIFICATION else "Target"
    required = {"Sample name", target_name}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    reserved = {"Sample name", target_name, "Group", "Split"}
    feature_columns = [column for column in data.columns if column not in reserved]
    if not feature_columns:
        raise ValueError("At least one numeric feature column is required.")

    features = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    all_missing = [column for column in features if features[column].isna().all()]
    if all_missing:
        raise ValueError(f"Feature columns contain no numeric values: {', '.join(map(str, all_missing))}")

    target = data[target_name]
    if task is TaskType.REGRESSION:
        target = pd.to_numeric(target, errors="coerce")
    valid = target.notna()
    if not valid.all():
        data = data.loc[valid].reset_index(drop=True)
        features = features.loc[valid].reset_index(drop=True)
        target = target.loc[valid].reset_index(drop=True)
    if len(data) < 6:
        raise ValueError("At least six valid samples are required.")
    if task is TaskType.CLASSIFICATION and pd.Series(target).nunique() < 2:
        raise ValueError("Classification requires at least two classes.")
    if task is TaskType.REGRESSION and pd.Series(target).nunique() < 3:
        raise ValueError("Regression requires at least three distinct target values.")

    groups = data["Group"].reset_index(drop=True) if "Group" in data else None
    split_labels = data["Split"].astype(str).str.lower().reset_index(drop=True) if "Split" in data else None
    return LoadedDataset(
        sample_names=data["Sample name"].astype(str).reset_index(drop=True),
        target=pd.Series(target).reset_index(drop=True),
        features=features.reset_index(drop=True),
        groups=groups,
        split_labels=split_labels,
        fingerprint=_fingerprint(data.reset_index(drop=True)),
    )


def split_dataset(dataset: LoadedDataset, config: RunConfig) -> DatasetSplit:
    indices = np.arange(len(dataset.target))
    if dataset.split_labels is not None:
        validation_mask = dataset.split_labels.isin({"validation", "val", "test"}).to_numpy()
        cal_mask = dataset.split_labels.isin({"calibration", "cal", "train", "training"}).to_numpy()
        if not validation_mask.any() or not cal_mask.any() or np.any(validation_mask & cal_mask):
            raise ValueError("Split must define non-empty, disjoint calibration/train and independent validation rows.")
        cal_idx, validation_idx = indices[cal_mask], indices[validation_mask]
    elif dataset.groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.validation_ratio, random_state=config.random_state)
        cal_idx, validation_idx = next(splitter.split(indices, dataset.target, groups=dataset.groups))
    else:
        stratify = dataset.target if config.task is TaskType.CLASSIFICATION else None
        cal_idx, validation_idx = train_test_split(
            indices,
            test_size=config.validation_ratio,
            random_state=config.random_state,
            shuffle=True,
            stratify=stratify,
        )

    def take(series, idx):
        return series.iloc[idx].reset_index(drop=True)

    if dataset.groups is not None:
        cal_groups = set(dataset.groups.iloc[cal_idx].astype(str))
        validation_groups = set(dataset.groups.iloc[validation_idx].astype(str))
        overlap = cal_groups.intersection(validation_groups)
        if overlap:
            raise ValueError(f"Group leakage between calibration and validation: {sorted(overlap)[:5]}")

    return DatasetSplit(
        sample_cal=take(dataset.sample_names, cal_idx),
        sample_validation=take(dataset.sample_names, validation_idx),
        y_cal=take(dataset.target, cal_idx),
        y_validation=take(dataset.target, validation_idx),
        X_cal=take(dataset.features, cal_idx),
        X_validation=take(dataset.features, validation_idx),
        groups_cal=take(dataset.groups, cal_idx) if dataset.groups is not None else None,
        groups_validation=take(dataset.groups, validation_idx) if dataset.groups is not None else None,
        fingerprint=dataset.fingerprint,
    )
