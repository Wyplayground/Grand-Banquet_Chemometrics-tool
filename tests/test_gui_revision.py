"""Headless checks for GUI state, source-bound export and new AI call recording."""
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from gb_core.contracts import PipelineSpec, RunConfig, TaskType
from gb_core.data import load_dataset, split_dataset
from gb_core.gui import GBApplication
from gb_core.reporting import DEEPSEEK_MODELS


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ImmediateThread:
    def __init__(self, target, daemon=True):
        self.target = target

    def start(self):
        self.target()


class GUIRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.app = GBApplication.__new__(GBApplication)
        self.app.task = TaskType.REGRESSION
        self.app.worker_queue = queue.Queue()
        self.app._run_busy = False
        self.app._export_busy = False
        self.app._ai_busy = False
        self.app.root = Mock()
        self.app.stop_button = Mock()
        self.app._require_outputs = Mock(return_value=True)
        self.app.export_mode_var = Variable("id")
        self.app.export_id_var = Variable("id-2")
        self.app.export_metric_var = Variable("CV_RMSE")
        self.app.export_direction_var = Variable("lower")
        self.app.export_reason_var = Variable("prespecified CV criterion")
        self.app.api_key_var = Variable("secret-test-key")
        self.app.deepseek_model_var = Variable("deepseek-flash")
        self.app.base_dir = Mock()
        self.app.base_dir.get.side_effect = AssertionError("Edited controls must not determine export/report origin")
        self.app.outputs = SimpleNamespace(manifest=self.path/"run_manifest.json",
                                           results=pd.DataFrame({"Pipeline_ID": ["id-1", "id-2"], "CV_RMSE": [2., 1.]}))
        self.app.last_run_data_file = self.path/"data.csv"

    def manifest(self):
        rng = np.random.default_rng(12)
        frame = pd.DataFrame({"Sample name": [f"s{i}" for i in range(32)],
                              "Target": rng.normal(size=32), "Group": np.repeat(np.arange(16), 2),
                              "Split": ["calibration"]*24+["holdout"]*8,
                              "f1": rng.normal(size=32), "f2": rng.normal(size=32)})
        frame.to_csv(self.app.last_run_data_file, index=False)
        config = RunConfig(TaskType.REGRESSION, outer_folds=3, inner_folds=2,
                           output_dir=self.path, random_state=71)
        data = load_dataset(self.app.last_run_data_file, self.app.task)
        specs = [PipelineSpec(TaskType.REGRESSION, "Raw", "None", "Linear Regression", seed=71)]
        payload = {"config": config.to_dict(), "pipeline_specs": [s.to_dict() for s in specs],
                   "dataset_fingerprint": data.fingerprint,
                   "calibration_groups": 12, "internal_holdout_groups": 4,
                   "calibration_observations": 24, "internal_holdout_observations": 8,
                   "source_provenance": {"combined_sha256": "supplied-test-hash"}}
        self.app.outputs.manifest.write_text(json.dumps(payload))
        return config, data, payload

    def test_export_context_uses_saved_config_and_original_data(self):
        config, data, manifest = self.manifest()
        self.app._config = Mock(side_effect=AssertionError("must not use edited controls"))
        engine, specs, loaded = self.app._candidate_export_context()
        self.assertEqual(engine.config.outer_folds, 3)
        self.assertEqual(engine.config.random_state, 71)
        self.assertEqual(engine.split.fingerprint, data.fingerprint)
        self.assertEqual(specs[0].seed, 71)
        self.assertEqual(loaded, manifest)
        expected = split_dataset(data, config)
        self.assertEqual(engine.split.sample_cal.tolist(), expected.sample_cal.tolist())

    def test_changed_source_dataset_is_rejected(self):
        self.manifest()
        frame = pd.read_csv(self.app.last_run_data_file)
        frame.loc[0, "f1"] += 3
        frame.to_csv(self.app.last_run_data_file, index=False)
        with self.assertRaisesRegex(ValueError, "dataset has changed"):
            self.app._candidate_export_context()

    def test_corrupt_spec_id_is_rejected(self):
        _, _, manifest = self.manifest()
        manifest["pipeline_specs"][0]["pipeline_id"] = "foreign"
        self.app.outputs.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "definition"):
            self.app._candidate_export_context()

    def test_saved_partition_mismatch_is_rejected(self):
        _, _, manifest = self.manifest()
        manifest["calibration_samples"] = ["foreign"]
        self.app.outputs.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "partition differs"):
            self.app._candidate_export_context()

    def test_id_and_cv_export_use_worker_and_bound_manifest(self):
        engine = Mock()
        engine.export_candidate.return_value = SimpleNamespace(pipeline_id="id-2")
        specs, manifest = ["test-spec"], {"test": "verified-manifest"}
        self.app._candidate_export_context = Mock(return_value=(engine, specs, manifest))
        with patch("gb_core.gui.threading.Thread", ImmediateThread):
            self.app._export_candidate()
        args = engine.export_candidate.call_args
        self.assertEqual(args.kwargs["pipeline_id"], "id-2")
        self.assertIsNone(args.kwargs["metric"])
        self.assertIsNone(args.kwargs["higher_is_better"])
        self.assertIs(args.kwargs["source_manifest"], manifest)
        self.assertEqual(self.app.worker_queue.get()[0], "export_done")
        self.app._export_busy = False
        self.app.export_mode_var.set("metric")
        with patch("gb_core.gui.threading.Thread", ImmediateThread):
            self.app._export_candidate()
        args = engine.export_candidate.call_args
        self.assertEqual(args.kwargs["metric"], "CV_RMSE")
        self.assertFalse(args.kwargs["higher_is_better"])
        self.assertIsNone(args.kwargs["pipeline_id"])

    def test_export_error_event_preserves_active_run_state(self):
        self.app._run_busy = True
        self.app._export_busy = True
        self.app.worker_queue.put(("error", ValueError("export failed"), "test trace", "export"))
        with patch("gb_core.gui.messagebox.showerror") as error, patch("builtins.print"):
            self.app._poll_worker_queue()
        self.assertTrue(self.app._run_busy)
        self.assertFalse(self.app._export_busy)
        error.assert_called_once_with("Candidate export failed", "export failed")
        self.app.stop_button.configure.assert_not_called()

    def test_export_done_clears_busy_flag_and_reports_paths(self):
        self.app._export_busy = True
        result = SimpleNamespace(pipeline_id="id-2", bundle_path="model.joblib", manifest_path="manifest.json",
                                 validation_status="FAILED_VALIDATION", validation_error="holdout failed")
        self.app.worker_queue.put(("export_done", result))
        with patch("gb_core.gui.messagebox.showinfo") as info, patch("builtins.print"):
            self.app._poll_worker_queue()
        self.assertFalse(self.app._export_busy)
        self.assertIn("manifest.json", info.call_args.args[1])
        self.assertIn("holdout failed", info.call_args.args[1])

    def test_ai_call_records_in_run_output_not_edited_directory(self):
        self.app._report_summary = Mock(return_value={"task": "regression"})
        with patch("gb_core.gui.threading.Thread", ImmediateThread), \
                patch("gb_core.gui.call_deepseek", return_value="supplied report") as call, \
                patch("gb_core.gui.write_report_files", return_value=["report.docx"]):
            self.app._generate_ai_report()
        self.assertEqual(call.call_args.kwargs["record_dir"], self.path/"ai_reports/call_records")
        self.assertEqual(call.call_args.args[2], "deepseek-flash")
        self.assertEqual(self.app.worker_queue.get()[0], "ai_done")

    def test_ai_error_redacts_credentials_in_queued_error(self):
        self.app._report_summary = Mock(return_value={"task": "regression"})
        with patch("gb_core.gui.threading.Thread", ImmediateThread), \
                patch("gb_core.gui.call_deepseek", side_effect=ValueError("failure secret-test-key")):
            self.app._generate_ai_report()
        event = self.app.worker_queue.get()
        self.assertEqual(event[3], "ai")
        self.assertNotIn("secret-test-key", str(event[1])+event[2])

    def test_deepseek_allowlist_is_shared_and_rejects_other_providers(self):
        self.assertEqual(DEEPSEEK_MODELS, ("deepseek-flash", "deepseek-v4-pro"))
        self.app._report_summary = Mock(return_value={"task": "regression"})
        self.app.deepseek_model_var.set("unsupported-provider-model")
        with patch("gb_core.gui.messagebox.showerror") as error, \
                patch("gb_core.gui.threading.Thread") as thread:
            self.app._generate_ai_report()
        error.assert_called_once()
        thread.assert_not_called()
        self.assertFalse(self.app._ai_busy)

    def test_environment_key_is_captured_before_worker_and_redacted(self):
        self.app._report_summary = Mock(return_value={"task": "regression"})
        self.app.api_key_var.set(" ")
        captured = {}

        class DeferredThread:
            def __init__(self, target, daemon=True):
                captured["target"] = target

            def start(self):
                pass

        # Only synthetic environment data is used. The worker starts after the
        # fallback changes, proving that the original effective key is captured.
        with patch("gb_core.gui.os.getenv", return_value="fallback-test-key"), \
                patch("gb_core.gui.threading.Thread", DeferredThread):
            self.app._generate_ai_report()
        with patch("gb_core.gui.os.getenv", return_value="different-test-key"), \
                patch("gb_core.gui.call_deepseek", side_effect=ValueError("failure fallback-test-key")) as call:
            captured["target"]()
        self.assertEqual(call.call_args.args[1], "fallback-test-key")
        event = self.app.worker_queue.get()
        self.assertNotIn("fallback-test-key", str(event[1])+event[2])
        self.assertIn("[REDACTED]", str(event[1]))

    def test_explicit_key_takes_precedence_over_environment(self):
        self.app._report_summary = Mock(return_value={"task": "regression"})
        with patch("gb_core.gui.os.getenv") as fallback, \
                patch("gb_core.gui.threading.Thread", ImmediateThread), \
                patch("gb_core.gui.call_deepseek", return_value="report") as call, \
                patch("gb_core.gui.write_report_files", return_value=[]):
            self.app._generate_ai_report()
        fallback.assert_not_called()
        self.assertEqual(call.call_args.args[1], "secret-test-key")

    def test_prompt_uses_deepseek_filename_and_saved_run_directory(self):
        self.app._report_summary = Mock(return_value={"task": "regression"})
        self.app.ai_text = Mock()
        with patch("gb_core.gui.build_deepseek_prompt", return_value="source-bound test prompt") as build:
            self.app._analyze()
        build.assert_called_once_with({"task": "regression"})
        files = list((self.path/"ai_reports").glob("deepseek_prompt_*.txt"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_text(encoding="utf-8"), "source-bound test prompt")

    def test_busy_ai_blocks_new_call_and_reset_until_error_processed(self):
        self.app._ai_busy = True
        with patch("gb_core.gui.messagebox.showwarning") as warning, \
                patch("gb_core.gui.threading.Thread") as thread, \
                patch("gb_core.gui.messagebox.askyesno") as confirmation:
            self.app._generate_ai_report()
            self.app._reset()
        self.assertEqual(warning.call_count, 2)
        thread.assert_not_called()
        confirmation.assert_not_called()
        self.app._run_busy = True
        self.app.worker_queue.put(("error", RuntimeError("AI failed"), "trace", "ai"))
        with patch("gb_core.gui.messagebox.showerror"), patch("builtins.print"):
            self.app._poll_worker_queue()
        self.assertFalse(self.app._ai_busy)
        self.assertTrue(self.app._run_busy)

    def test_report_context_uses_supplied_counts_and_source(self):
        self.manifest()
        summary = self.app._report_summary()
        self.assertEqual(summary["context"]["calibration_groups"], 12)
        self.assertEqual(summary["context"]["internal_holdout_observations"], 8)
        self.assertEqual(summary["context"]["evaluation_config"]["random_state"], 71)
        self.assertEqual(summary["context"]["source_provenance"]["combined_sha256"], "supplied-test-hash")


if __name__ == "__main__":
    unittest.main()
