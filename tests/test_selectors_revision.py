"""CARS feature-axis regression checks using controlled coefficients, offline."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gb_core.contracts import EvaluationMode, TaskType
from gb_core.selectors import GBSelector, _pls_coefficient_importance


class CoefficientLayoutTests(unittest.TestCase):
    def test_both_multitarget_layouts_average_targets_and_preserve_feature_indices(self):
        coefficients = np.array([[1., -2., 3., -18., 5., 6.],
                                 [-3., 4., -5., 22., 7., -8.]])
        expected = np.array([2., 3., 4., 20., 6., 7.])
        for layout in [coefficients, coefficients.T]:
            np.testing.assert_array_equal(_pls_coefficient_importance(layout, 6, 2), expected)
            self.assertEqual(np.argmax(_pls_coefficient_importance(layout, 6, 2)), 3)

    def test_single_target_matrix_and_vector_layouts(self):
        values = np.array([1., -2., 9., -3., 5., 4.])
        for layout in [values, values.reshape(1, -1), values.reshape(-1, 1)]:
            np.testing.assert_array_equal(_pls_coefficient_importance(layout, 6, 1), np.abs(values))
        np.testing.assert_array_equal(_pls_coefficient_importance([[-4.]], 1, 1), [4.])

    def test_ambiguous_invalid_and_nonfinite_layouts_fail_explicitly(self):
        for values, features, targets in [(np.ones((6, 6)), 6, 6),
                                          (np.ones((3, 4)), 6, 2),
                                          (np.ones(6), 6, 2),
                                          (np.ones((1, 2, 3)), 6, 1),
                                          (np.array([[np.nan]*6]), 6, 1),
                                          (np.array([[np.inf]*6]), 6, 1)]:
            with self.subTest(shape=np.shape(values), features=features, targets=targets):
                with self.assertRaises(ValueError):
                    _pls_coefficient_importance(values, features, targets)


class CARSExecutionTests(unittest.TestCase):
    def test_cars_selects_high_coefficient_features_in_both_tasks_and_layouts(self):
        # Important variables deliberately extend beyond the first one/two columns.
        feature_coefficients = np.array([.1, .2, .3, .4, .5, .6, .7, 7., 8., .8, 10., .9, 12., 1., 1.1, 15.])
        expected = [7, 8, 10, 12, 15]
        X = np.random.default_rng(20261008).normal(size=(24, 16))
        for task in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            y = np.arange(24) % 2 if task is TaskType.CLASSIFICATION else np.arange(24, dtype=float)
            target_first = np.vstack([feature_coefficients, -2*feature_coefficients]) if task is TaskType.CLASSIFICATION else feature_coefficients.reshape(1, -1)
            for layout in [target_first, target_first.T]:
                with self.subTest(task=task, shape=layout.shape):
                    pls = SimpleNamespace(coef_=layout, fit=Mock())
                    selector = GBSelector(method="CARS", task=task, mode=EvaluationMode.EXPLORATION)
                    # Prefer the smallest valid candidate; test actual CARS ranking,
                    # not scientific performance or the library's scoring policy.
                    with patch("gb_core.selectors.PLSRegression", return_value=pls), \
                         patch.object(selector, "_fitness", side_effect=lambda X, y, indices: -len(indices)), \
                         patch.object(selector, "_pls_rmsecv", side_effect=lambda X, y: -X.shape[1]):
                        selector.fit(X, y)
                    np.testing.assert_array_equal(selector.get_support(indices=True), expected)
                    self.assertEqual(pls.fit.call_count, 10)
                    self.assertEqual(selector.metadata_["selected_count"], 5)


if __name__ == "__main__":
    unittest.main()
