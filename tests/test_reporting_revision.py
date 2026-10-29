"""Offline evidence and provenance checks; no API calls are made."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from gb_core.contracts import TaskType
from gb_core.reporting import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODELS,
    DEEPSEEK_SYSTEM_PROMPT,
    build_deepseek_prompt,
    call_deepseek,
    structured_summary,
    write_leaderboard_html,
)


def mock_factory(response=None, error=None):
    create = Mock(return_value=response, side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return Mock(return_value=client), create


def response(text="  Report text  "):
    result = SimpleNamespace(
        id="response-123",
        model="deepseek-flash-actual-snapshot",
        _request_id="request-456",
        created=1791417600,
        usage=SimpleNamespace(prompt_tokens=110, completion_tokens=20, total_tokens=130),
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
    )
    result.model_dump = lambda: {
        "id": result.id,
        "model": result.model,
        "created": result.created,
        "usage": {"prompt_tokens": 110, "completion_tokens": 20, "total_tokens": 130},
        "choices": [{"message": {"content": text}, "finish_reason": result.choices[0].finish_reason}],
    }
    return result


class StructuredEvidenceTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame([
            {"Pipeline_ID": "second", "Selection_Rank": 2, "Eligible_For_Ranking": True,
             "Status": "SUCCESS", "Val_Status": "FAILED_VALIDATION", "Val_Error": "predict failed",
             "CV_MCC": .25, "Cal_MCC": .7, "Val_MCC": np.nan, "Fold_Primary_SD": .04},
            {"Pipeline_ID": "first", "Selection_Rank": 1, "Eligible_For_Ranking": True,
             "Status": "SUCCESS", "Val_Status": "SUCCESS", "CV_MCC": .5432123456,
             "Cal_MCC": .8, "Val_MCC": .4, "Fold_Primary_SD": .03, "Seed": 42,
             "N_Calibration": 74, "N_Validation": 20, "Hyperparameters": '{"C":1}',
             "Overfit_Cal_vs_CV": .2567876544},
            {"Pipeline_ID": "failed", "Selection_Rank": np.nan, "Eligible_For_Ranking": False,
             "Status": "FAILED_INNER_CV", "Val_Status": "NOT_EVALUATED"},
        ])

    def test_summary_has_actual_metrics_diagnostics_scope_and_nulls(self):
        summary = structured_summary(self.table(), TaskType.CLASSIFICATION, top_n=2,
                                     context={"calibration_groups": np.int64(37), "groups": np.array([1, 2])})
        self.assertEqual(summary["total_pipeline_rows"], 3)
        self.assertEqual(summary["eligible_pipeline_rows"], 2)
        self.assertEqual(summary["included_pipeline_rows"], 2)
        self.assertEqual(summary["holdout_failed_pipelines"], 1)
        self.assertEqual(summary["failed_pipelines"], 1)
        first, second = summary["top_pipelines"]
        self.assertEqual(first["Pipeline_ID"], "first")
        self.assertEqual(first["CV_MCC"], .5432123456)
        self.assertEqual(first["Cal_MCC"], .8)
        self.assertEqual(first["Val_MCC"], .4)
        self.assertEqual(first["Fold_Primary_SD"], .03)
        self.assertEqual(first["Seed"], 42)
        self.assertEqual(first["N_Calibration"], 74)
        self.assertEqual(first["N_Validation"], 20)
        self.assertEqual(first["Hyperparameters"], '{"C":1}')
        self.assertEqual(first["Overfit_Cal_vs_CV"], .2567876544)
        self.assertIsNone(first["CV_Kappa"])
        self.assertIsNone(second["Val_MCC"])
        self.assertEqual(second["Val_Error"], "predict failed")
        self.assertEqual(summary["context"], {"calibration_groups": 37, "groups": [1, 2]})
        json.dumps(summary, allow_nan=False)

    def test_regression_defaults_and_absent_status_not_invented(self):
        table = pd.DataFrame([{"Eligible_For_Ranking": True, "CV_R2": .31, "CV_RPD": np.inf}]*12)
        summary = structured_summary(table, TaskType.REGRESSION)
        self.assertEqual(summary["included_pipeline_rows"], 10)
        self.assertEqual(summary["requested_top_n"], 10)
        self.assertIsNone(summary["successful_pipelines"])
        self.assertIsNone(summary["holdout_failed_pipelines"])
        self.assertIsNone(summary["top_pipelines"][0]["CV_RPD"])
        self.assertEqual(summary["top_pipelines"][0]["CV_R2"], .31)
        self.assertIsNone(summary["top_pipelines"][0]["Val_RMSE"])
        self.assertEqual(structured_summary(table, TaskType.REGRESSION, top_n=0)["top_pipelines"], [])
        for bad in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                structured_summary(table, TaskType.REGRESSION, top_n=bad)

    def test_prompt_defines_scope_and_no_unsupported_claims(self):
        prompt = build_deepseek_prompt(structured_summary(self.table(), TaskType.CLASSIFICATION))
        self.assertIn("internal holdout", prompt)
        self.assertIn("Do not assume the input includes all 990", prompt)
        self.assertIn("does not independently assess the whole 990", prompt)
        self.assertIn("null means unavailable", prompt)
        self.assertIn("never claim analyst bias is eliminated", prompt)
        self.assertIn("implementation assumptions", prompt)
        self.assertNotIn("Independent Validation", prompt)
        payload = json.loads(prompt.split("\n\n", 1)[1])
        self.assertEqual(payload["top_pipelines"][0]["CV_MCC"], .5432123456)

    def test_html_note_has_internal_holdout_and_selection_boundary(self):
        with TemporaryDirectory() as temporary:
            path = write_leaderboard_html(self.table(), Path(temporary)/"table.html", TaskType.CLASSIFICATION)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Internal holdout metric values", content)
            self.assertIn("not an unbiased assessment of whole-library selection", content)
            self.assertNotIn("Independent validation", content)


class CallProvenanceTests(unittest.TestCase):
    def summary(self):
        return {"task": "regression", "top_pipelines": [{"CV_R2": .3, "Val_R2": None}]}

    def read_record(self, directory):
        paths = list(Path(directory).glob("deepseek_call_*.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_actual_call_parameters_payload_and_response_archived(self):
        factory, create = mock_factory(response())
        with TemporaryDirectory() as temporary:
            result = call_deepseek(self.summary(), "test-secret-abc", "deepseek-v4-pro", factory,
                               record_dir=temporary, max_tokens=900)
            record = self.read_record(temporary)
            self.assertEqual(result, "Report text")
            self.assertEqual(record["status"], "SUCCESS")
            self.assertEqual(record["api_request_attempts"], 1)
            self.assertEqual(record["summary"], self.summary())
            self.assertEqual(record["system_prompt"], DEEPSEEK_SYSTEM_PROMPT)
            self.assertEqual(record["user_prompt"], build_deepseek_prompt(self.summary()))
            params = {"model": "deepseek-v4-pro", "temperature": .2, "timeout": 120,
                      "max_tokens": 900, "stream": False, "extra_body": {"thinking": {"type": "disabled"}}}
            self.assertEqual(record["requested_parameters"], params)
            factory.assert_called_once_with(api_key="test-secret-abc", base_url=DEEPSEEK_BASE_URL,
                                            timeout=120, max_retries=0)
            self.assertEqual(record["provider"], "DeepSeek")
            self.assertEqual(record["endpoint"], "https://api.deepseek.com")
            self.assertEqual(record["client_parameters"]["max_retries"], 0)
            self.assertEqual(create.call_args.kwargs, {
                **params, "messages": [{"role":"system","content":DEEPSEEK_SYSTEM_PROMPT},
                                         {"role":"user","content":record["user_prompt"]}]})
            returned = record["response"]
            self.assertEqual(returned["returned_model"], "deepseek-flash-actual-snapshot")
            self.assertEqual(returned["response_id"], "response-123")
            self.assertEqual(returned["request_id"], "request-456")
            self.assertEqual(returned["finish_reason"], "stop")
            self.assertEqual(returned["usage"]["total_tokens"], 130)
            self.assertEqual(returned["raw_payload"]["choices"][0]["message"]["content"], "  Report text  ")
            self.assertTrue(record["started_at_utc"].endswith("+00:00"))
            self.assertTrue(record["completed_at_utc"].endswith("+00:00"))
            self.assertGreaterEqual(record["elapsed_seconds"], 0)
            self.assertNotIn("test-secret-abc", json.dumps(record))

    def test_positional_signature_default_parameters_and_no_unsupported_params(self):
        factory, create = mock_factory(response())
        self.assertEqual(call_deepseek(self.summary(), "secret", "deepseek-flash", factory), "Report text")
        for name in ["top_p", "seed"]:
            self.assertNotIn(name, create.call_args.kwargs)
        self.assertEqual(create.call_args.kwargs["max_tokens"], 4096)
        self.assertEqual(create.call_args.kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertFalse(create.call_args.kwargs["stream"])

    def test_provider_allowlist_and_unsupported_parameters_rejected_before_credential_sent(self):
        self.assertEqual(DEEPSEEK_MODELS, ("deepseek-flash", "deepseek-v4-pro"))
        for options in [{"model": "other-provider-model"}, {"seed": 42}, {"top_p": .8},
                        {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": 2.5}, {"max_tokens": 393217}]:
            with self.subTest(options=options), TemporaryDirectory() as temporary:
                factory, create = mock_factory(response())
                with self.assertRaises(ValueError):
                    call_deepseek(self.summary(), "provider-secret", client_factory=factory,
                                  record_dir=temporary, **options)
                factory.assert_not_called()
                create.assert_not_called()
                record = self.read_record(temporary)
                self.assertEqual(record["status"], "FAILED")
                self.assertEqual(record["api_request_attempts"], 0)
                self.assertNotIn("provider-secret", json.dumps(record))

    def test_only_deepseek_environment_key_is_used(self):
        factory, _ = mock_factory(response())
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "  deepseek-env-secret  "}, clear=True):
            self.assertEqual(call_deepseek(self.summary(), client_factory=factory), "Report text")
        self.assertEqual(factory.call_args.kwargs["api_key"], "deepseek-env-secret")
        factory.reset_mock()
        with patch.dict("os.environ", {"OTHER_PROVIDER_API_KEY": "foreign-secret"}, clear=True):
            with self.assertRaises(ValueError):
                call_deepseek(self.summary(), client_factory=factory)
        factory.assert_not_called()

    def test_failed_call_is_logged_and_key_echo_is_redacted(self):
        secret = "test-secret-failure"
        factory, create = mock_factory(error=RuntimeError("server echoed " + secret))
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "server echoed"):
                call_deepseek(self.summary(), secret, client_factory=factory, record_dir=temporary)
            record = self.read_record(temporary)
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["error"]["type"], "RuntimeError")
            self.assertIn("[REDACTED]", record["error"]["message"])
            self.assertIsNone(record["response"])
            self.assertNotIn(secret, json.dumps(record))
            create.assert_called_once()

    def test_missing_key_logged_without_attempting_network(self):
        factory, create = mock_factory(response())
        with TemporaryDirectory() as temporary, patch.dict("os.environ", {"DEEPSEEK_API_KEY":""}):
            with self.assertRaises(ValueError):
                call_deepseek(self.summary(), client_factory=factory, record_dir=temporary)
            self.assertEqual(self.read_record(temporary)["status"], "FAILED")
            factory.assert_not_called()

    def test_empty_response_preserves_returned_payload_in_failure_record(self):
        factory, _ = mock_factory(response(""))
        with TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                call_deepseek(self.summary(), "secret-empty", client_factory=factory, record_dir=temporary)
            record=self.read_record(temporary)
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["response"]["returned_model"], "deepseek-flash-actual-snapshot")
            self.assertEqual(record["response"]["raw_payload"]["choices"][0]["message"]["content"], "")

    def test_truncated_or_abnormal_response_is_failed_without_retry(self):
        for finish in ["length", "content_filter", "insufficient_system_resource", None]:
            with self.subTest(finish=finish), TemporaryDirectory() as temporary:
                reply = response("Incomplete report")
                reply.choices[0].finish_reason = finish
                factory, create = mock_factory(reply)
                with self.assertRaisesRegex(RuntimeError, "not completed normally"):
                    call_deepseek(self.summary(), "secret", client_factory=factory, record_dir=temporary)
                create.assert_called_once()
                record = self.read_record(temporary)
                self.assertEqual(record["status"], "FAILED")
                self.assertEqual(record["response"]["finish_reason"], finish)
                self.assertEqual(record["response"]["text"], "Incomplete report")

    def test_missing_choice_preserves_payload_in_failed_record(self):
        reply = SimpleNamespace(model="deepseek-flash", choices=[], usage=None, id="no-choice")
        factory, create = mock_factory(reply)
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no report choice"):
                call_deepseek(self.summary(), "secret", client_factory=factory, record_dir=temporary)
            record = self.read_record(temporary)
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["response"]["raw_payload"]["choices"], [])
            create.assert_called_once()

    def test_unique_records_and_credentials_scrubbed_from_payload(self):
        secret = "long-test-secret"
        summary = {**self.summary(), "context": {"api_key": secret, "note": secret}}
        reply = response("The echoed key " + secret)
        factory, _ = mock_factory(reply)
        with TemporaryDirectory() as temporary:
            for _ in range(2):
                call_deepseek(summary, secret, client_factory=factory, record_dir=temporary)
            files=list(Path(temporary).glob("deepseek_call_*.json"))
            self.assertEqual(len(files), 2)
            for path in files:
                text=path.read_text(encoding="utf-8")
                self.assertNotIn(secret, text)
                self.assertIn("[REDACTED]", text)


if __name__ == "__main__":
    unittest.main()
