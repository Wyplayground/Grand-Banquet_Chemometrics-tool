"""Regression checks for export choice and calibration/holdout failure isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gb_core.artifacts import load_bundle
from gb_core.contracts import EvaluationMode, PipelineSpec, RunConfig, RunStatus, TaskType
from gb_core.data import DatasetSplit
from gb_core.engine import EVALUATION_REVISION, EvaluationEngine
from gb_core.metrics import rank_results


class HoldoutFailPreprocessor:
    def __init__(self, base, holdout_size):
        self.base = base
        self.holdout_size = holdout_size

    def transform(self, values):
        if len(values) == self.holdout_size:
            raise ValueError("injected holdout prediction failure")
        return self.base.transform(values)


class EngineRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        rng = np.random.default_rng(43)
        X = pd.DataFrame(rng.normal(size=(32, 6)), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(2 * X.f0 - X.f1 + rng.normal(scale=0.2, size=32))
        names = pd.Series([f"sample-{i}" for i in range(32)])
        groups = pd.Series(np.repeat(np.arange(16), 2))
        self.split = DatasetSplit(
            names[:24].reset_index(drop=True), names[24:].reset_index(drop=True),
            y[:24].reset_index(drop=True), y[24:].reset_index(drop=True),
            X[:24].reset_index(drop=True), X[24:].reset_index(drop=True),
            groups[:24].reset_index(drop=True), groups[24:].reset_index(drop=True),
            "synthetic-engine-test",
        )
        self.config = RunConfig(
            TaskType.REGRESSION, mode=EvaluationMode.EXPLORATION, outer_folds=3,
            inner_folds=2, output_dir=Path(self.temp.name), resume=False,
        )
        self.engine = EvaluationEngine(self.split, self.config)
        self.specs = [
            PipelineSpec(TaskType.REGRESSION, "Raw", "None", "Linear Regression"),
            PipelineSpec(TaskType.REGRESSION, "Center and scale", "None", "Ridge Regression"),
        ]

    def ranked(self):
        return rank_results(pd.DataFrame([
            self.engine.evaluate_one(self.engine._run_id(self.specs), spec) for spec in self.specs
        ]), TaskType.REGRESSION)

    def failing_fit(self):
        original = self.engine._fit_components

        def fit(*args, **kwargs):
            pre, selector, model, params = original(*args, **kwargs)
            return HoldoutFailPreprocessor(pre, len(self.split.y_validation)), selector, model, params
        return fit

    def test_holdout_failure_preserves_cv_and_group_evidence(self):
        reference = self.engine.evaluate_one("test-run", self.specs[0])
        with patch.object(self.engine, "_fit_components", side_effect=self.failing_fit()):
            failed = self.engine.evaluate_one("test-run", self.specs[0])
        self.assertEqual(failed["Status"], "SUCCESS")
        self.assertTrue(failed["Eligible_For_Ranking"])
        self.assertTrue(failed["Val_Evaluated"])
        self.assertEqual(failed["Val_Status"], "FAILED_VALIDATION")
        self.assertIn("injected", failed["Val_Error"])
        for column in ("CV_RMSE", "CV_R2", "Fold_Primary_SD"):
            self.assertEqual(reference[column], failed[column])
        cv_file = Path(self.temp.name) / "predictions" / f"{self.specs[0].pipeline_id}_cv.csv"
        self.assertEqual(len(pd.read_csv(cv_file)), 24)
        folds = json.loads((Path(self.temp.name) / "fold_metadata" / f"{self.specs[0].pipeline_id}.json").read_text())
        self.assertTrue(all(set(f["train_groups"]).isdisjoint(f["test_groups"]) for f in folds))

    def test_holdout_values_and_failure_do_not_change_cv_order(self):
        ranked = self.ranked()
        changed = ranked.copy()
        for column in changed:
            if column.startswith("Val_") and column not in {"Val_Evaluated", "Val_Status", "Val_Error"}:
                changed[column] = np.nan
        changed["Val_Status"] = "FAILED_VALIDATION"
        again = rank_results(changed, TaskType.REGRESSION)
        pd.testing.assert_frame_equal(ranked[["Pipeline_ID", "Selection_Score", "Selection_Rank"]],
                                      again[["Pipeline_ID", "Selection_Score", "Selection_Rank"]])

    def test_export_nondefault_id_does_not_recompute_cv_or_mutate_table(self):
        ranked = self.ranked()
        before = ranked.copy(deep=True)
        chosen = str(ranked.iloc[-1].Pipeline_ID)
        with patch.object(self.engine, "evaluate_one", side_effect=AssertionError("CV must not rerun")):
            exported = self.engine.export_candidate(ranked, self.specs, pipeline_id=chosen)
        self.assertEqual(exported.pipeline_id, chosen)
        bundle = load_bundle(exported.bundle_path)
        self.assertEqual(len(bundle.predict_frame(self.split.X_validation)), 8)
        manifest = json.loads(exported.manifest_path.read_text())
        self.assertFalse(manifest["cv_recomputed"])
        self.assertEqual(manifest["stored_selection_rank"], 2)
        self.assertEqual(manifest["source_run_id"], self.engine._run_id(self.specs))
        self.assertEqual(manifest["calibration_groups"], 12)
        pd.testing.assert_frame_equal(ranked, before)

    def test_metric_choice_infers_rmse_direction(self):
        ranked = self.ranked()
        expected = ranked.sort_values(["CV_RMSE", "Selection_Rank", "Pipeline_ID"]).iloc[0].Pipeline_ID
        exported = self.engine.export_candidate(ranked, self.specs, metric="CV_RMSE")
        self.assertEqual(exported.pipeline_id, expected)
        self.assertFalse(json.loads(exported.manifest_path.read_text())["higher_is_better"])

    def test_foreign_run_or_manifest_rejected_before_refit(self):
        ranked = self.ranked()
        foreign = ranked.copy()
        foreign["Run_ID"] = "foreign-data-run"
        with patch.object(self.engine, "_fit_components", side_effect=AssertionError("must not refit")):
            with self.assertRaisesRegex(ValueError, "verified"):
                self.engine.export_candidate(foreign, self.specs)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.engine.export_candidate(ranked, self.specs, source_manifest={"dataset_fingerprint": "foreign"})

    def test_legacy_run_requires_verified_manifest(self):
        ranked = self.ranked()
        legacy_run = self.engine._run_id(self.specs, include_revision=False)
        ranked["Run_ID"] = legacy_run
        with self.assertRaises(ValueError):
            self.engine.export_candidate(ranked, self.specs)
        manifest = {"dataset_fingerprint": self.split.fingerprint, "run_id": legacy_run,
                    "pipeline_specs": [spec.to_dict() for spec in self.specs]}
        exported = self.engine.export_candidate(ranked, self.specs, source_manifest=manifest)
        self.assertEqual(json.loads(exported.manifest_path.read_text())["source_run_id"], legacy_run)

    def test_failed_attempt_removes_stale_holdout_predictions(self):
        self.engine.evaluate_one("test-run", self.specs[0])
        path = Path(self.temp.name) / "predictions" / f"{self.specs[0].pipeline_id}_val.csv"
        self.assertTrue(path.exists())
        with patch.object(self.engine, "_fit_components", side_effect=self.failing_fit()):
            self.engine.evaluate_one("test-run", self.specs[0])
        self.assertFalse(path.exists())

    def test_default_export_failure_clears_stale_aliases(self):
        ranked = self.ranked()
        paths = [Path(self.temp.name) / name for name in
                 ("best_regression_pipeline.joblib", "rank1_validation_predictions.csv", "rank1_final_refit.json")]
        for path in paths:
            path.write_text("stale")
        with patch.object(self.engine, "_fit_components", side_effect=ValueError("refit failed")):
            self.engine._finalize_rank_one(ranked, self.specs)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_unseen_holdout_classes_preserve_classification_cv(self):
        from dataclasses import replace
        split = replace(self.split,
                        y_cal=pd.Series(np.repeat(["A", "B"], 12)),
                        y_validation=pd.Series(["C"] * 8))
        config = RunConfig(TaskType.CLASSIFICATION, mode=EvaluationMode.EXPLORATION,
                           outer_folds=3, inner_folds=2, output_dir=Path(self.temp.name))
        engine = EvaluationEngine(split, config)
        spec = PipelineSpec(TaskType.CLASSIFICATION, "Raw", "None", "Logistic Regression")
        row = engine.evaluate_one(engine._run_id([spec]), spec)
        self.assertEqual(row["Status"], "SUCCESS")
        self.assertTrue(row["Eligible_For_Ranking"])
        self.assertEqual(row["Val_Status"], "FAILED_VALIDATION")
        self.assertIn("unseen classes", row["Val_Error"])
        self.assertTrue(np.isfinite(row["CV_MCC"]))
        exported = engine.export_candidate(rank_results(pd.DataFrame([row]), TaskType.CLASSIFICATION), [spec])
        self.assertTrue(exported.bundle_path.exists())
        self.assertEqual(exported.validation_status, "FAILED_VALIDATION")

    def test_resume_reuses_only_matching_file_evidence(self):
        self.config.resume = True
        self.engine.run(self.specs)
        with patch.object(self.engine, "evaluate_one", side_effect=AssertionError("matching cache must be reused")):
            self.engine.run(self.specs)
        path = self.engine._evidence_files(self.specs[0])["cv"]
        path.write_text("foreign overwritten evidence")
        original = self.engine.evaluate_one
        with patch.object(self.engine, "evaluate_one", wraps=original) as evaluate:
            self.engine.run(self.specs)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[1].pipeline_id, self.specs[0].pipeline_id)
        self.assertTrue(self.engine._evidence_matches(self.engine._run_id(self.specs), self.specs[0],
            self.engine.checkpoint.get(self.engine._run_id(self.specs), self.specs[0].pipeline_id)))

    def test_shared_output_directory_other_run_evidence_forces_recompute(self):
        self.config.resume = True
        self.engine.run(self.specs)
        marker_path = self.engine._evidence_marker(self.specs[0])
        marker = json.loads(marker_path.read_text())
        marker["run_id"] = "other-run"
        marker_path.write_text(json.dumps(marker))
        with patch.object(self.engine, "evaluate_one", wraps=self.engine.evaluate_one) as evaluate:
            self.engine.run(self.specs)
            self.assertEqual(evaluate.call_count, 1)

    def test_malformed_evidence_marker_is_a_cache_miss(self):
        self.config.resume = True
        self.engine.run(self.specs)
        self.engine._evidence_marker(self.specs[0]).write_text("[]")
        with patch.object(self.engine, "evaluate_one", wraps=self.engine.evaluate_one) as evaluate:
            self.engine.run(self.specs)
            self.assertEqual(evaluate.call_count, 1)

    def test_duplicate_specs_rejected_before_running(self):
        with patch.object(self.engine, "evaluate_one", side_effect=AssertionError("must not evaluate")):
            with self.assertRaisesRegex(ValueError, "unique IDs"):
                self.engine.run([self.specs[0], self.specs[0]])

    def test_invalid_choices_rejected_before_refitting(self):
        ranked = self.ranked()
        with patch.object(self.engine, "_fit_components", side_effect=AssertionError("must not refit")):
            for kwargs in ({"pipeline_id": "missing"}, {"metric": "Val_R2"}, {"metric": "Cal_R2"},
                           {"metric": "CV_Bias"}, {"pipeline_id": self.specs[0].pipeline_id, "metric": "CV_R2"}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.engine.export_candidate(ranked, self.specs, **kwargs)

    def test_bundle_survives_holdout_export_failure(self):
        ranked = self.ranked()
        with patch.object(self.engine, "_fit_components", side_effect=self.failing_fit()):
            exported = self.engine.export_candidate(ranked, self.specs)
        self.assertTrue(exported.bundle_path.exists())
        self.assertEqual(exported.validation_status, "FAILED_VALIDATION")
        self.assertIsNone(exported.validation_predictions)
        self.assertEqual(exported.validation_metrics, {})
        self.assertEqual(json.loads(exported.manifest_path.read_text())["validation_status"], "FAILED_VALIDATION")

    def test_export_fit_failure_is_not_mislabeled_as_holdout_failure(self):
        ranked = self.ranked()
        with patch.object(self.engine, "_fit_components", side_effect=ValueError("refit failed")):
            refreshed, bundle, validation = self.engine._finalize_rank_one(ranked, self.specs)
        self.assertIsNone(bundle)
        self.assertIsNone(validation)
        pd.testing.assert_frame_equal(ranked, refreshed)

    def test_strict_reports_holdout_failures_while_preserving_ranks(self):
        self.config.strict = True
        with patch.object(self.engine, "_fit_components", side_effect=self.failing_fit()):
            with self.assertRaisesRegex(RuntimeError, "validation_failed=True"):
                self.engine.run(self.specs)
        results = pd.read_csv(Path(self.temp.name) / "pipeline_results_v3.csv")
        self.assertEqual(results.Status.tolist(), ["SUCCESS", "SUCCESS"])
        self.assertTrue(results.Selection_Rank.notna().all())
        manifest = json.loads((Path(self.temp.name) / "run_manifest.json").read_text())
        self.assertEqual(manifest["internal_holdout_failures"], 2)
        self.assertEqual(manifest["evaluation_revision"], EVALUATION_REVISION)

    def test_successful_run_keeps_legacy_alias_and_marks_default(self):
        outputs = self.engine.run(self.specs)
        self.assertEqual(outputs.default_pipeline, outputs.best_pipeline)
        self.assertTrue(outputs.default_pipeline.exists())
        self.assertTrue(outputs.validation_predictions.exists())
        manifest = json.loads(outputs.manifest.read_text())
        self.assertEqual(manifest["default_pipeline"], manifest["best_pipeline"])
        self.assertFalse(manifest["default_export_failed"])
        self.assertEqual(manifest["internal_holdout_failures"], 0)


if __name__ == "__main__":
    unittest.main()
