"""Headless CLI and saved-run binding checks; no GUI windows or library CV."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from gb_core import cli
from gb_core.contracts import EvaluationMode, PipelineSpec, RunConfig, TaskType
from gb_core.data import load_dataset, split_dataset
from gb_core.plotting import _eligible


class CandidateCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data_path = self.folder / "data.csv"
        pd.DataFrame({"Sample name": [f"sample_{i}" for i in range(12)],
                      "Target": list(range(12)), "feature": [i * 0.5 for i in range(12)],
                      "Group": [i // 2 for i in range(12)],
                      "Split": ["cal"] * 8 + ["holdout"] * 4}).to_csv(self.data_path, index=False)
        self.config = RunConfig(task=TaskType.REGRESSION, mode=EvaluationMode.EXPLORATION,
                                outer_folds=3, inner_folds=2, output_dir=self.folder)
        self.spec = PipelineSpec(TaskType.REGRESSION, "Raw", "None", "PLS Regression")
        split = split_dataset(load_dataset(self.data_path, TaskType.REGRESSION), self.config)
        self.manifest = {"run_id": "000012", "config": self.config.to_dict(),
                         "dataset_fingerprint": split.fingerprint,
                         "calibration_samples": split.sample_cal.tolist(),
                         "validation_samples": split.sample_validation.tolist(),
                         "pipeline_specs": [self.spec.to_dict()]}
        self.write_manifest()
        # The reader must preserve leading zeroes even for an all-numeric ID.
        pd.DataFrame({"Run_ID": ["000012"], "Pipeline_ID": ["0000001234567890"],
                      "Eligible_For_Ranking": [True], "CV_RMSE": [0.5]}).to_csv(
                          self.folder / "pipeline_results_v3.csv", index=False)
        self.export = SimpleNamespace(pipeline_id=self.spec.pipeline_id,
                                      bundle_path=self.folder / "candidate.joblib",
                                      manifest_path=self.folder / "export_manifest.json",
                                      validation_status="SUCCESS", validation_error=None)

    def write_manifest(self):
        (self.folder / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def call_main(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main_for_task(TaskType.REGRESSION, arguments)

    def test_id_and_metric_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser(TaskType.REGRESSION).parse_args(
                ["--export-pipeline-id", "id", "--export-metric", "CV_R2"])

    def test_export_requires_source_data_and_direction_requires_metric(self):
        for arguments in [["--export-from", str(self.folder)],
                          ["--export-from", str(self.folder), "--data", str(self.data_path),
                           "--export-direction", "lower"],
                          ["--export-pipeline-id", "id"]]:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.call_main(arguments)

    def test_manifest_restores_settings_and_filters_unknown_fields(self):
        self.manifest["config"]["future_option"] = "ignored"
        self.manifest["config"]["output_dir"] = "old_location"
        self.manifest["pipeline_specs"][0]["future_option"] = "ignored"
        self.write_manifest()
        config, specs, manifest = cli.load_run_manifest(self.folder, TaskType.REGRESSION)
        self.assertEqual(config.output_dir, self.folder)
        self.assertEqual(config.outer_folds, 3)
        self.assertEqual(config.inner_folds, 2)
        self.assertEqual(specs, [self.spec])
        self.assertEqual(manifest["run_id"], "000012")

    def test_manifest_rejects_wrong_task_and_modified_spec(self):
        with self.assertRaisesRegex(ValueError, "task"):
            cli.load_run_manifest(self.folder, TaskType.CLASSIFICATION)
        self.manifest["pipeline_specs"][0]["model"] = "Linear Regression"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identifier"):
            cli.load_run_manifest(self.folder)

    def test_stored_run_preserves_string_ids_without_running_cv(self):
        with patch.object(cli, "EvaluationEngine") as engine_class:
            engine, specs, results, manifest = cli.load_stored_run(
                self.folder, self.data_path, TaskType.REGRESSION)
        self.assertIs(engine, engine_class.return_value)
        self.assertEqual(results.loc[0, "Run_ID"], "000012")
        self.assertEqual(results.loc[0, "Pipeline_ID"], "0000001234567890")
        self.assertEqual(specs, [self.spec])
        engine.run.assert_not_called()

    def test_wrong_data_and_partition_are_rejected_before_engine_creation(self):
        for key, value in [("dataset_fingerprint", "wrong"),
                           ("calibration_samples", ["different"] * 8)]:
            original = self.manifest[key]
            self.manifest[key] = value
            self.write_manifest()
            with patch.object(cli, "EvaluationEngine") as engine_class, self.assertRaises(ValueError):
                cli.load_stored_run(self.folder, self.data_path, TaskType.REGRESSION)
            engine_class.assert_not_called()
            self.manifest[key] = original

    def test_stored_export_passes_provenance_and_explicit_metric(self):
        with patch.object(cli, "EvaluationEngine") as engine_class:
            engine = engine_class.return_value
            engine.export_candidate.return_value = self.export
            result = self.call_main(["--export-from", str(self.folder), "--data", str(self.data_path),
                                     "--export-metric", "CV_RMSE", "--export-direction", "lower",
                                     "--selection-reason", "Lower CV error", "--export-output", str(self.folder / "exports")])
            self.assertEqual(result, 0)
            engine.run.assert_not_called()
            options = engine.export_candidate.call_args.kwargs
            self.assertEqual(options["source_manifest"], self.manifest)
            self.assertEqual(options["metric"], "CV_RMSE")
            self.assertFalse(options["higher_is_better"])
            self.assertEqual(options["selection_reason"], "Lower CV error")

    def test_stored_export_id_and_default_have_inferred_direction(self):
        for selection in [[], ["--export-pipeline-id", self.spec.pipeline_id]]:
            with self.subTest(selection=selection), patch.object(cli, "EvaluationEngine") as engine_class:
                engine_class.return_value.export_candidate.return_value = self.export
                result = self.call_main(["--export-from", str(self.folder), "--data", str(self.data_path)] + selection)
                self.assertEqual(result, 0)
                options = engine_class.return_value.export_candidate.call_args.kwargs
                self.assertIsNone(options["higher_is_better"])
                self.assertEqual(options["pipeline_id"], self.spec.pipeline_id if selection else None)

    def test_post_run_export_uses_existing_results_once(self):
        results = pd.DataFrame({"Pipeline_ID": [self.spec.pipeline_id]})
        outputs = SimpleNamespace(results=results, results_csv=self.folder / "results.csv",
                                  leaderboard_csv=self.folder / "leaderboard.csv", default_pipeline="example.joblib")
        with patch.object(cli, "EvaluationEngine") as engine_class, patch.object(cli, "create_visualization_suite"):
            engine = engine_class.return_value
            engine.run.return_value = outputs
            engine.export_candidate.return_value = self.export
            status = self.call_main(["--headless-demo", str(self.folder), "--data", str(self.data_path),
                                     "--mode", "exploration", "--models", "PLS Regression",
                                     "--preprocessing", "Raw", "--selectors", "None",
                                     "--export-pipeline-id", self.spec.pipeline_id])
            self.assertEqual(status, 0)
            engine.run.assert_called_once()
            self.assertIs(engine.export_candidate.call_args.args[0], results)
            self.assertNotIn("source_manifest", engine.export_candidate.call_args.kwargs)

    def test_export_failure_returns_error_without_cv(self):
        with patch.object(cli, "EvaluationEngine") as engine_class:
            engine_class.return_value.export_candidate.side_effect = ValueError("not eligible")
            self.assertEqual(self.call_main(["--export-from", str(self.folder), "--data", str(self.data_path)]), 2)
            engine_class.return_value.run.assert_not_called()

    def test_explicit_missing_data_is_not_replaced_with_demo(self):
        with patch.object(cli, "create_demo_data") as create_demo, self.assertRaises(SystemExit):
            self.call_main(["--headless-demo", str(self.folder), "--data", str(self.folder / "missing.csv")])
        create_demo.assert_not_called()

    def test_plotting_excludes_false_csv_flags_and_sorts_numeric_ranks(self):
        frame = pd.DataFrame({"Eligible_For_Ranking": ["False", "True", "1", "0", None],
                              "Selection_Rank": ["1", "10", "2", "0", "0"],
                              "Pipeline_ID": ["bad", "ten", "two", "bad2", "bad3"]})
        self.assertEqual(_eligible(frame)["Pipeline_ID"].tolist(), ["two", "ten"])


if __name__ == "__main__":
    unittest.main()
