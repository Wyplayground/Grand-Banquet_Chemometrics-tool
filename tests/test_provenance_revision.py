"""Targeted provenance checks; no historical manifests or model fitting required."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gb_core import artifacts


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "copy"
        self.root.mkdir()

    def source(self, relative, contents=b"VALUE = 1\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_hashes_active_packages_and_excludes_tests_uploads_caches(self):
        included = ["gb_core/engine.py", "classification/app.py", "regression/nested/helper.py"]
        for relative in reversed(included):
            self.source(relative)
        for relative in ("gb_core/tests/helper.py", "gb_core/test_helper.py", "regression/uploads/data.py",
                         "classification/upload/a.py", "gb_core/__pycache__/fake.py", "benchmarks/prepare.py"):
            self.source(relative)
        result = artifacts.source_provenance(self.root)
        self.assertEqual(list(result["source_files"]), sorted(included))
        self.assertEqual(result["source_file_count"], 3)
        expected_file_hash = hashlib.sha256(b"VALUE = 1\n").hexdigest()
        self.assertTrue(all(digest == expected_file_hash for digest in result["source_files"].values()))
        combined_input = "".join(f"{name}\0{expected_file_hash}\n" for name in sorted(included))
        self.assertEqual(result["combined_sha256"], hashlib.sha256(combined_input.encode()).hexdigest())

    def test_same_sources_in_different_locations_have_same_combined_hash(self):
        self.source("gb_core/engine.py")
        second = Path(self.temporary.name) / "other" / "gb_core"
        second.mkdir(parents=True)
        (second / "engine.py").write_bytes(b"VALUE = 1\n")
        first_result = artifacts.source_provenance(self.root)
        second_result = artifacts.source_provenance(second.parent)
        self.assertNotEqual(first_result["source_root"], second_result["source_root"])
        self.assertEqual(first_result["combined_sha256"], second_result["combined_sha256"])

    def test_edit_and_rename_both_change_combined_digest(self):
        file = self.source("gb_core/engine.py")
        first = artifacts.source_provenance(self.root)["combined_sha256"]
        file.write_bytes(b"VALUE = 2\n")
        second = artifacts.source_provenance(self.root)["combined_sha256"]
        file.rename(file.with_name("other.py"))
        third = artifacts.source_provenance(self.root)["combined_sha256"]
        self.assertEqual(len({first, second, third}), 3)

    def test_unrelated_ancestor_git_is_never_queried(self):
        (self.root.parent / ".git").mkdir()
        self.source("gb_core/engine.py")
        with patch.object(artifacts.subprocess, "run") as command:
            result = artifacts.source_provenance(self.root)
        command.assert_not_called()
        self.assertNotIn("git_commit", result)

    def test_local_git_commit_is_optional_and_requires_matching_root(self):
        (self.root / ".git").mkdir()
        with patch.object(artifacts.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, str(self.root) + "\n"),
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n"),
        ]):
            self.assertEqual(artifacts.source_provenance(self.root)["git_commit"], "a" * 40)
        with patch.object(artifacts.subprocess, "run", return_value=
                          subprocess.CompletedProcess([], 0, str(self.root.parent) + "\n")) as command:
            self.assertNotIn("git_commit", artifacts.source_provenance(self.root))
            self.assertEqual(command.call_count, 1)

    def test_git_unavailable_keeps_hashes(self):
        (self.root / ".git").write_text("gitdir: missing", encoding="utf-8")
        self.source("gb_core/engine.py")
        with patch.object(artifacts.subprocess, "run", side_effect=FileNotFoundError("git")):
            result = artifacts.source_provenance(self.root)
        self.assertEqual(result["source_file_count"], 1)
        self.assertNotIn("git_commit", result)

    def test_default_root_is_executing_copy(self):
        result = artifacts.source_provenance()
        self.assertEqual(Path(result["source_root"]), Path(artifacts.__file__).resolve().parent.parent)
        self.assertIn("gb_core/artifacts.py", result["source_files"])

    def test_manifest_adds_provenance_without_mutating_payload_or_old_file(self):
        old_path = self.root / "historical.json"
        old_path.write_text('{"historical": true}', encoding="utf-8")
        original_bytes = old_path.read_bytes()
        payload = {"run_id": "new"}
        provenance = {"combined_sha256": "test-digest"}
        with patch.object(artifacts, "source_provenance", return_value=provenance), \
             patch.object(artifacts, "dependency_versions", return_value={"python": "test"}):
            new_path = artifacts.write_manifest(self.root / "new.json", payload)
        result = json.loads(new_path.read_text(encoding="utf-8"))
        self.assertEqual(result["source_provenance"], provenance)
        self.assertEqual(result["dependencies"], {"python": "test"})
        self.assertEqual(payload, {"run_id": "new"})
        self.assertEqual(old_path.read_bytes(), original_bytes)

    def test_manifest_preserves_explicit_provenance_and_dependencies(self):
        payload = {"source_provenance": {"origin": "provided"}, "dependencies": {"numpy": "provided"}}
        with patch.object(artifacts, "source_provenance") as provenance, \
             patch.object(artifacts, "dependency_versions", return_value={}):
            path = artifacts.write_manifest(self.root / "provided.json", payload)
        provenance.assert_not_called()
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["source_provenance"], payload["source_provenance"])
        self.assertEqual(result["dependencies"], payload["dependencies"])


if __name__ == "__main__":
    unittest.main()
