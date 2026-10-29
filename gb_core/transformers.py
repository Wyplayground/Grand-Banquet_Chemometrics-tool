from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

from .contracts import PREPROCESSING_METHODS, SG_PREPROCESSING_METHODS


EPS = 1e-12
DEFAULT_SG_WINDOW = 11
DEFAULT_SG_POLYORDER = 3
SG_WINDOW_CANDIDATES = (5, 7, 9, 11, 13, 15)
SG_POLYORDER_CANDIDATES = (2, 3)


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")
    return arr


def _savgol_params(n_features: int, requested: int = DEFAULT_SG_WINDOW, polyorder: int = DEFAULT_SG_POLYORDER) -> tuple[int, int]:
    if n_features < 3:
        return 1, 0
    window = min(int(requested), n_features)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)
    return window, min(int(polyorder), window - 1)


def sg_param_candidates(n_features: int) -> list[tuple[int, int]]:
    pairs = []
    seen = set()
    for requested in SG_WINDOW_CANDIDATES:
        for polyorder in SG_POLYORDER_CANDIDATES:
            window, order = _savgol_params(n_features, requested, polyorder)
            if window <= 1 or order < 1:
                continue
            key = (window, order)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    if not pairs:
        pairs.append(_savgol_params(n_features))
    return pairs


class GBPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe implementation of the 15 public Grand Banquet preprocessors."""

    def __init__(
        self,
        method: str = "Raw",
        sg_window: int = DEFAULT_SG_WINDOW,
        sg_polyorder: int = DEFAULT_SG_POLYORDER,
    ):
        self.method = method
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder

    def fit(self, X, y=None):
        X = _as_2d(X)
        if self.method not in PREPROCESSING_METHODS:
            raise ValueError(f"Unknown preprocessing method: {self.method}")
        self.n_features_in_ = X.shape[1]
        self.impute_values_ = np.nanmedian(X, axis=0)
        self.impute_values_ = np.where(np.isfinite(self.impute_values_), self.impute_values_, 0.0)
        clean = self._impute(X)
        if self.method in SG_PREPROCESSING_METHODS:
            self.resolved_sg_window_, self.resolved_sg_polyorder_ = _savgol_params(
                X.shape[1], self.sg_window, self.sg_polyorder
            )
        if self.method in {"Centering", "Center and scale", "Pareto scaling"}:
            self.center_ = np.mean(clean, axis=0)
        if self.method == "Center and scale":
            self.scale_ = np.std(clean, axis=0, ddof=1)
            self.scale_ = np.where(np.abs(self.scale_) < EPS, 1.0, self.scale_)
        if self.method == "Pareto scaling":
            std = np.std(clean, axis=0, ddof=1)
            self.scale_ = np.sqrt(np.where(np.abs(std) < EPS, 1.0, std))
        if self.method == "MSC":
            self.reference_ = np.mean(clean, axis=0)
        if self.method == "Log transform":
            minimum = float(np.min(clean))
            self.offset_ = 1.0 if minimum >= 0 else abs(minimum) + 1.0
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        rows, cols = np.where(~np.isfinite(X))
        if len(rows):
            X[rows, cols] = self.impute_values_[cols]
        return X

    def transform(self, X):
        if not hasattr(self, "impute_values_"):
            raise RuntimeError("GBPreprocessor must be fitted before transform.")
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {X.shape[1]}.")
        X = self._impute(X)
        method = self.method
        if method == "Raw":
            return X
        if method == "Baseline correction":
            window, order = self.resolved_sg_window_, self.resolved_sg_polyorder_
            if window <= 1:
                return X
            result = X - savgol_filter(X, window, order, axis=1)
            return np.maximum(result, 0.0)
        if method == "Max normalization":
            denom = np.max(np.abs(X), axis=1, keepdims=True)
            return X / np.where(denom < EPS, 1.0, denom)
        if method == "Area normalization":
            trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            area = trapezoid(np.abs(X), axis=1).reshape(-1, 1)
            return X / np.where(area < EPS, 1.0, area)
        if method == "Range normalization":
            minimum = np.min(X, axis=1, keepdims=True)
            span = np.max(X, axis=1, keepdims=True) - minimum
            return (X - minimum) / np.where(span < EPS, 1.0, span)
        if method == "Center and scale":
            return (X - self.center_) / self.scale_
        if method == "Log transform":
            shifted = X + self.offset_
            if np.any(shifted <= 0):
                raise ValueError("Transform values are outside the fitted log-transform domain.")
            return np.log10(shifted)
        if method == "Centering":
            return X - self.center_
        if method == "Pareto scaling":
            return (X - self.center_) / self.scale_
        if method == "SNV":
            center = np.mean(X, axis=1, keepdims=True)
            scale = np.std(X, axis=1, ddof=1, keepdims=True)
            return (X - center) / np.where(scale < EPS, 1.0, scale)
        if method == "MSC":
            design = np.column_stack([np.ones(self.reference_.shape[0]), self.reference_])
            corrected = np.empty_like(X)
            for row, spectrum in enumerate(X):
                intercept, slope = np.linalg.lstsq(design, spectrum, rcond=None)[0]
                corrected[row] = (spectrum - intercept) / slope if abs(slope) >= EPS else spectrum
            return corrected
        if method == "SG derivative":
            window, order = self.resolved_sg_window_, self.resolved_sg_polyorder_
            return X if window <= 1 else savgol_filter(X, window, order, deriv=1, axis=1)
        if method == "Simple 1st derivative":
            return np.pad(np.diff(X, axis=1), ((0, 0), (0, 1)))
        if method == "Simple 2nd derivative":
            if X.shape[1] < 3:
                return np.zeros_like(X)
            return np.pad(np.diff(X, n=2, axis=1), ((0, 0), (0, 2)))
        if method == "SG smoothing":
            window, order = self.resolved_sg_window_, self.resolved_sg_polyorder_
            return X if window <= 1 else savgol_filter(X, window, order, axis=1)
        raise ValueError(f"Unknown preprocessing method: {method}")

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"F{i + 1}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(list(input_features), dtype=object)
