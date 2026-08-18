from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score

from .contracts import EvaluationMode, SELECTOR_METHODS, TaskType, mode_uses_full_selector_budget
from .models import cv_strategy, make_model


EPS = 1e-12


def _vip_scores(X: np.ndarray, y: np.ndarray, n_components: int = 5) -> np.ndarray:
    components = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pls = PLSRegression(n_components=components, max_iter=1000, tol=1e-6)
    pls.fit(X, y)
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    explained = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total = float(np.sum(explained))
    if total <= EPS:
        return np.zeros(p)
    weights = np.array([(w[:, i] / max(np.linalg.norm(w[:, i]), EPS)) ** 2 for i in range(h)])
    return np.sqrt(p * (explained.T @ weights).ravel() / total)


class GBSelector(BaseEstimator, TransformerMixin):
    """Selector fitted from each cross-validation training subset."""

    def __init__(
        self,
        method: str = "None",
        task: TaskType = TaskType.REGRESSION,
        model_name: str = "PLS Regression",
        inner_folds: int = 3,
        mode: EvaluationMode = EvaluationMode.EXPLORATION,
        random_state: int = 42,
        feature_penalty: float = 0.02,
    ):
        self.method = method
        self.task = task
        self.model_name = model_name
        self.inner_folds = inner_folds
        self.mode = mode
        self.random_state = random_state
        self.feature_penalty = feature_penalty

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("Selector input must be two-dimensional.")
        if self.method not in SELECTOR_METHODS:
            raise ValueError(f"Unknown variable-selection method: {self.method}")
        self.n_features_in_ = X.shape[1]
        self.task_ = TaskType(self.task)
        self.mode_ = EvaluationMode(self.mode)
        self.rng_ = np.random.default_rng(self.random_state)
        self.groups_ = None if groups is None else np.asarray(groups)
        minimum = min(5, X.shape[1])

        if self.method == "None" or X.shape[1] <= minimum:
            selected = np.arange(X.shape[1])
            self.metadata_ = {"method": self.method, "reason": "all features retained"}
        elif self.method == "VIP":
            target = self._pls_target(y)
            scores = _vip_scores(X, target)
            selected = np.where(scores >= 1.0)[0]
            if len(selected) < minimum:
                selected = np.argsort(scores)[::-1][:minimum]
            self.metadata_ = {"vip_scores": scores.tolist(), "threshold": 1.0}
        elif self.method == "iPLS":
            selected = self._fit_ipls(X, y)
        elif self.method == "GA":
            selected = self._fit_ga(X, y)
        elif self.method == "CARS":
            selected = self._fit_cars(X, y)
        elif self.method == "SFLA":
            selected = self._fit_sfla(X, y)
        else:
            raise ValueError(f"Unknown selector: {self.method}")

        selected = np.unique(np.asarray(selected, dtype=int))
        if len(selected) < minimum:
            fallback = self._univariate_order(X, y)[:minimum]
            selected = np.unique(np.concatenate([selected, fallback]))
        self.selected_indices_ = np.sort(selected)
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[self.selected_indices_] = True
        self.metadata_["selected_count"] = int(len(self.selected_indices_))
        return self

    def _pls_target(self, y: np.ndarray) -> np.ndarray:
        if self.task_ is TaskType.REGRESSION:
            return np.asarray(y, dtype=float).ravel()
        classes, encoded = np.unique(y, return_inverse=True)
        return np.column_stack([(encoded == i).astype(float) for i in range(len(classes))])

    def _fitness(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> float:
        indices = np.unique(np.asarray(indices, dtype=int))
        if len(indices) < 1:
            return -np.inf
        try:
            model = make_model(self.task_, self.model_name, self.random_state)
            scoring = make_scorer(matthews_corrcoef) if self.task_ is TaskType.CLASSIFICATION else "neg_root_mean_squared_error"
            scores = cross_val_score(
                clone(model),
                X[:, indices],
                y,
                cv=cv_strategy(self.task_, self.inner_folds, self.random_state, y, self.groups_),
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                groups=self.groups_,
            )
            score = float(np.mean(scores))
            return score - self.feature_penalty * (len(indices) / X.shape[1])
        except Exception:
            return -np.inf

    def _univariate_order(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.task_ is TaskType.REGRESSION:
            target = np.asarray(y, dtype=float)
        else:
            _, target = np.unique(y, return_inverse=True)
        scores = []
        for column in range(X.shape[1]):
            corr = np.corrcoef(X[:, column], target)[0, 1]
            scores.append(abs(corr) if np.isfinite(corr) else 0.0)
        return np.argsort(scores)[::-1]

    def _fit_ipls(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        interval_count = min(10, X.shape[1])
        intervals = [part for part in np.array_split(np.arange(X.shape[1]), interval_count) if len(part)]
        scores = [self._fitness(X, y, interval) for interval in intervals]
        best = int(np.argmax(scores))
        self.metadata_ = {"interval_scores": scores, "best_interval": best}
        return intervals[best]

    def _fit_ga(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        population_size, generations = ((14, 8) if mode_uses_full_selector_budget(self.mode_) else (8, 4))
        min_features = min(5, X.shape[1])
        max_features = min(max(min_features, X.shape[1] // 3), 60, X.shape[1])

        def random_mask():
            count = int(self.rng_.integers(min_features, max_features + 1))
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[self.rng_.choice(X.shape[1], count, replace=False)] = True
            return mask

        population = [random_mask() for _ in range(population_size)]
        best_mask = population[0].copy()
        best_score = -np.inf
        history = []
        for _ in range(generations):
            scored = sorted(((self._fitness(X, y, np.where(mask)[0]), mask) for mask in population), key=lambda item: item[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best_mask = scored[0][0], scored[0][1].copy()
            history.append(float(best_score))
            survivors = [mask.copy() for _, mask in scored[: max(2, population_size // 3)]]
            population = survivors.copy()
            while len(population) < population_size:
                first, second = self.rng_.choice(len(survivors), 2, replace=True)
                child = np.where(self.rng_.random(X.shape[1]) < 0.5, survivors[first], survivors[second])
                child = np.logical_xor(child, self.rng_.random(X.shape[1]) < 0.04)
                selected = np.where(child)[0]
                if len(selected) < min_features:
                    child = random_mask()
                elif len(selected) > max_features:
                    keep = self.rng_.choice(selected, max_features, replace=False)
                    child[:] = False
                    child[keep] = True
                population.append(child)
        self.metadata_ = {"fitness": float(best_score), "history": history}
        return np.where(best_mask)[0]

    def _fit_cars(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        runs = 24 if mode_uses_full_selector_budget(self.mode_) else 10
        target = self._pls_target(y)
        candidates = []
        scores = []
        for run in range(runs):
            row_count = max(4, int(0.8 * len(y)))
            rows = self.rng_.choice(len(y), row_count, replace=True)
            components = max(1, min(5, row_count - 1, X.shape[1]))
            pls = PLSRegression(n_components=components, max_iter=1000, tol=1e-6)
            pls.fit(X[rows], target[rows])
            coefficients = np.abs(np.asarray(pls.coef_))
            importance = coefficients.mean(axis=1) if coefficients.ndim == 2 else coefficients.ravel()
            retain_ratio = math.exp(-2.5 * run / max(runs - 1, 1))
            keep_count = max(min(5, X.shape[1]), int(round(X.shape[1] * retain_ratio)))
            candidate = np.argsort(importance)[::-1][:keep_count]
            candidates.append(candidate)
            if self.task_ is TaskType.REGRESSION:
                score = self._pls_rmsecv(X[:, candidate], np.asarray(y, dtype=float))
            else:
                score = self._fitness(X, y, candidate)
            scores.append(score)
        best = int(np.argmax(scores))
        self.metadata_ = {"candidate_scores": scores, "best_run": best}
        return candidates[best]

    def _pls_rmsecv(self, X: np.ndarray, y: np.ndarray) -> float:
        components = max(1, min(5, X.shape[1], X.shape[0] - 1))
        model = PLSRegression(n_components=components, max_iter=1000, tol=1e-6)
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv_strategy(TaskType.REGRESSION, self.inner_folds, self.random_state, y, self.groups_),
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
                error_score="raise",
                groups=self.groups_,
            )
            return float(np.mean(scores)) - self.feature_penalty * (X.shape[1] / self.n_features_in_)
        except Exception:
            return -np.inf

    def _fit_sfla(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        frog_count, generations = ((14, 8) if mode_uses_full_selector_budget(self.mode_) else (8, 4))
        min_features = min(5, X.shape[1])
        max_features = min(max(min_features, X.shape[1] // 3), 60, X.shape[1])

        def decode(frog):
            count = int(np.clip(round(np.mean(frog) * max_features), min_features, max_features))
            return np.argsort(frog)[::-1][:count]

        frogs = self.rng_.random((frog_count, X.shape[1]))
        best_frog = frogs[0].copy()
        best_score = -np.inf
        history = []
        for _ in range(generations):
            scores = np.asarray([self._fitness(X, y, decode(frog)) for frog in frogs])
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > best_score:
                best_score = float(scores[order[0]])
                best_frog = frogs[order[0]].copy()
            history.append(best_score)
            for idx in order[len(order) // 2 :]:
                frogs[idx] = np.clip(
                    frogs[idx] + self.rng_.random(X.shape[1]) * (best_frog - frogs[idx]) + self.rng_.normal(0, 0.03, X.shape[1]),
                    0,
                    1,
                )
        self.metadata_ = {"fitness": best_score, "history": history}
        return decode(best_frog)

    def transform(self, X):
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("GBSelector must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {X.shape[1]}.")
        return X[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        return self.selected_indices_.copy() if indices else self.support_mask_.copy()

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        names = np.asarray(list(input_features) if input_features is not None else [f"F{i + 1}" for i in range(self.n_features_in_)])
        return names[self.selected_indices_]
