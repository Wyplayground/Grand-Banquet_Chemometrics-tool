from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .contracts import EvaluationMode, TaskType


class AdaptivePLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_components = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))
        self.model_ = PLSRegression(n_components=n_components, max_iter=1000, tol=1e-6)
        self.model_.fit(X, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float)).ravel()


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        encoded = np.column_stack([(y == value).astype(float) for value in self.classes_])
        n_components = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))
        self.model_ = PLSRegression(n_components=n_components, max_iter=1000, tol=1e-6)
        self.model_.fit(X, encoded)
        return self

    def _scores(self, X):
        scores = np.asarray(self.model_.predict(np.asarray(X, dtype=float)))
        return scores if scores.ndim == 2 else scores.reshape(-1, 1)

    def predict(self, X):
        scores = self._scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._scores(X)
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return scores / np.sum(scores, axis=1, keepdims=True)


def _xgboost(task: TaskType, random_state: int):
    try:
        if task is TaskType.CLASSIFICATION:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=1,
                eval_metric="mlogloss",
            )
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=1,
            objective="reg:squarederror",
        )
    except Exception as exc:
        raise ImportError("XGBoost is required for the selected XGBoost pipeline; no fallback is permitted.") from exc


def make_model(task: TaskType, name: str, random_state: int = 42):
    task = TaskType(task)
    if task is TaskType.CLASSIFICATION:
        models = {
            "PLS-DA": PLSDAClassifier(n_components=5),
            "LDA": LinearDiscriminantAnalysis(),
            "SVM-DA (linear)": Pipeline(
                [("scaler", StandardScaler()), ("model", SVC(kernel="linear", C=1.0, random_state=random_state))]
            ),
            "SVM-DA (rbf)": Pipeline(
                [("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_state))]
            ),
            "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=random_state, n_jobs=1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state),
            "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
            "Logistic Regression": Pipeline(
                [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=random_state))]
            ),
            "Naive Bayes": GaussianNB(),
            "Neural Network": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(80,),
                            max_iter=2000,
                            tol=1e-4,
                            early_stopping=True,
                            n_iter_no_change=20,
                            validation_fraction=0.15,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        }
        if name == "XGBoost":
            return _xgboost(task, random_state)
    else:
        models = {
            "PLS Regression": AdaptivePLSRegressor(n_components=8),
            "Linear Regression": LinearRegression(),
            "SVR (linear)": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", C=1.0))]),
            "SVR (rbf)": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, gamma="scale"))]),
            "Random Forest Regression": RandomForestRegressor(n_estimators=120, max_depth=10, random_state=random_state, n_jobs=1),
            "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=random_state),
            "KNN Regression": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
            "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "Bayesian Ridge": Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())]),
            "Neural Network Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(80,),
                            max_iter=2000,
                            tol=1e-4,
                            early_stopping=True,
                            n_iter_no_change=20,
                            validation_fraction=0.15,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        }
        if name == "XGBoost Regression":
            return _xgboost(task, random_state)
    if name not in models:
        raise ValueError(f"Unknown {task.value} model: {name}")
    return models[name]


def model_grid(task: TaskType, name: str, mode: EvaluationMode) -> Dict[str, list]:
    mode = EvaluationMode(mode)
    if mode is EvaluationMode.EXPLORATION:
        return {}
    task = TaskType(task)
    audit = mode is EvaluationMode.AUDIT
    if task is TaskType.CLASSIFICATION:
        grids = {
            "PLS-DA": {"n_components": [2, 5, 8, 10, 12] if audit else [2, 5, 8]},
            "SVM-DA (linear)": {"model__C": [0.1, 1.0, 10.0]},
            "SVM-DA (rbf)": {"model__C": [1.0, 10.0], "model__gamma": ["scale", 0.01, 0.1]},
            "Random Forest": {
                "max_depth": [None, 10, 20] if audit else [None, 10],
                "max_features": ["sqrt", 0.5],
                **({"n_estimators": [120, 200]} if audit else {}),
            },
            "XGBoost": {
                "max_depth": [3, 6, 8] if audit else [3, 6],
                "learning_rate": [0.05, 0.1],
                **({"n_estimators": [100, 200]} if audit else {}),
            },
            "Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            "KNN": {"model__n_neighbors": [3, 5, 7]},
            "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
            "Neural Network": {
                "model__alpha": [0.0001, 0.001, 0.01] if audit else [0.0001, 0.01],
                **(
                    {
                        "model__hidden_layer_sizes": [(50,), (80,), (120,)],
                        "model__max_iter": [2000, 5000],
                    }
                    if audit
                    else {"model__max_iter": [2000]}
                ),
            },
        }
    else:
        grids = {
            "PLS Regression": {"n_components": [2, 5, 8, 10, 12] if audit else [2, 5, 8]},
            "SVR (linear)": {"model__C": [0.1, 1.0, 10.0]},
            "SVR (rbf)": {"model__C": [1.0, 10.0], "model__gamma": ["scale", 0.01, 0.1]},
            "Random Forest Regression": {
                "max_depth": [None, 10, 20] if audit else [None, 10],
                "max_features": ["sqrt", 0.7],
                **({"n_estimators": [120, 200]} if audit else {}),
            },
            "XGBoost Regression": {
                "max_depth": [3, 6, 8] if audit else [3, 6],
                "learning_rate": [0.05, 0.1],
                **({"n_estimators": [100, 200]} if audit else {}),
            },
            "Gradient Boosting Regression": {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            "KNN Regression": {"model__n_neighbors": [3, 5, 7]},
            "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
            "Neural Network Regression": {
                "model__alpha": [0.0001, 0.001, 0.01] if audit else [0.0001, 0.01],
                **(
                    {
                        "model__hidden_layer_sizes": [(50,), (80,), (120,)],
                        "model__max_iter": [2000, 5000],
                    }
                    if audit
                    else {"model__max_iter": [2000]}
                ),
            },
        }
    return grids.get(name, {})


def cv_strategy(task: TaskType, folds: int, random_state: int, y=None, groups=None):
    folds = max(2, int(folds))
    if groups is not None:
        unique_groups = np.unique(np.asarray(groups))
        folds = min(folds, len(unique_groups))
        if folds < 2:
            raise ValueError("Group-aware CV requires at least two distinct groups.")
        if TaskType(task) is TaskType.CLASSIFICATION:
            return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        y_array = np.asarray(y, dtype=float).ravel()
        group_array = np.asarray(groups)
        unique, inverse = np.unique(group_array, return_inverse=True)
        group_targets = np.asarray([np.median(y_array[inverse == index]) for index in range(len(unique))])
        values, counts = np.unique(group_targets, return_counts=True)
        if len(values) > 1 and int(np.min(counts)) >= folds:
            group_strata = np.searchsorted(values, group_targets)
        else:
            max_bins = min(10, max(2, len(unique) // folds))
            group_strata = None
            for bins in range(max_bins, 1, -1):
                order = np.argsort(np.argsort(group_targets, kind="mergesort"), kind="mergesort")
                candidate = np.minimum((order * bins) // max(len(unique), 1), bins - 1)
                if np.min(np.bincount(candidate)) >= folds:
                    group_strata = candidate
                    break
            if group_strata is None:
                return GroupKFold(n_splits=folds)
        row_strata = group_strata[inverse]
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(y_array)), row_strata, group_array))
    if TaskType(task) is TaskType.CLASSIFICATION:
        if y is not None:
            _, counts = np.unique(np.asarray(y), return_counts=True)
            folds = min(folds, int(np.min(counts)))
        if folds < 2:
            raise ValueError("Each class must contain at least two samples for cross-validation.")
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    if y is not None:
        folds = min(folds, len(y))
    if folds < 2:
        raise ValueError("At least two samples are required for cross-validation.")
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def backend_name(model) -> str:
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        return f"{final.__class__.__module__}.{final.__class__.__name__}"
    return f"{model.__class__.__module__}.{model.__class__.__name__}"
