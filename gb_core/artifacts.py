from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .contracts import PipelineSpec, SCHEMA_VERSION, TaskType


_PIPELINE_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_results (
    run_id TEXT NOT NULL,
    pipeline_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, pipeline_id)
)
"""


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Empty/corrupt leftovers from a crashed multiprocess run cannot be opened usefully.
        if self.path.exists() and self.path.stat().st_size == 0:
            self.path.unlink()
        with closing(self._connect()) as connection:
            with connection:
                self._ensure_schema(connection)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(_PIPELINE_RESULTS_DDL)

    def get(self, run_id: str, pipeline_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            with connection:
                self._ensure_schema(connection)
            row = connection.execute(
                "SELECT result_json FROM pipeline_results WHERE run_id=? AND pipeline_id=?",
                (run_id, pipeline_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, run_id: str, spec: PipelineSpec, result: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                self._ensure_schema(connection)
                connection.execute(
                    """
                    INSERT INTO pipeline_results(run_id, pipeline_id, spec_json, result_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, pipeline_id) DO UPDATE SET
                        spec_json=excluded.spec_json,
                        result_json=excluded.result_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        run_id,
                        spec.pipeline_id,
                        json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True),
                        json.dumps(result, ensure_ascii=False, allow_nan=True),
                        now,
                    ),
                )


@dataclass
class FinalPipelineBundle:
    task: TaskType
    spec: Dict[str, Any]
    preprocessor: Any
    selector: Any
    model: Any
    feature_names: list[str]
    selected_feature_names: list[str]
    dataset_fingerprint: str
    label_encoder: Any = None
    schema_version: str = SCHEMA_VERSION

    def predict_frame(self, features: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features.columns]
        extra = [name for name in features.columns if name not in self.feature_names]
        if missing or extra:
            messages = []
            if missing:
                messages.append("missing=" + ",".join(map(str, missing)))
            if extra:
                messages.append("unexpected=" + ",".join(map(str, extra)))
            raise ValueError("Prediction feature schema mismatch: " + "; ".join(messages))
        X = features.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        transformed = self.preprocessor.transform(X)
        selected = self.selector.transform(transformed)
        prediction = np.asarray(self.model.predict(selected)).ravel()
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform(prediction.astype(int))
        return prediction


def save_bundle(bundle: FinalPipelineBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, path)
    return path


def load_bundle(path: str | Path) -> FinalPipelineBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, FinalPipelineBundle):
        raise TypeError("The selected file is not a Grand Banquet pipeline bundle.")
    return bundle


def dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "xgboost", "PIL"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not-installed"
    return versions


def source_provenance(source_root: str | Path | None = None) -> Dict[str, Any]:
    """Fingerprint Python sources in the executing source copy, not an ancestor repo.

    The combined digest covers sorted, POSIX relative paths and each file's raw-byte
    SHA-256. It excludes the absolute installation path, so identical copies have
    identical combined digests. Tests, upload folders and Python caches are omitted.
    A Git commit is supplementary context only; hashes describe the working files.
    """
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parent.parent
    excluded = {"__pycache__", ".git", "test", "tests", "upload", "uploads"}
    hashes: Dict[str, str] = {}
    for package in ("gb_core", "classification", "regression"):
        package_root = root / package
        if not package_root.is_dir():
            continue
        for path in sorted(package_root.rglob("*.py")):
            relative = path.relative_to(root)
            if any(part.lower() in excluded for part in relative.parts) or path.name.startswith("test_"):
                continue
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = hashlib.sha256()
    for relative, digest in sorted(hashes.items()):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    provenance: Dict[str, Any] = {
        "source_root": str(root),
        "source_files": dict(sorted(hashes.items())),
        "source_file_count": len(hashes),
        "combined_sha256": combined.hexdigest(),
        "hash_contract": "SHA-256 of sorted relative-path UTF-8, NUL, file SHA-256 ASCII, newline; file hashes use raw bytes",
        "scope": "gb_core, classification and regression Python sources; excludes tests, uploads and caches",
    }
    # A manuscript copy may live inside an unrelated Git worktree. Do not let
    # `git -C` walk up to it, or describe that ancestor's commit as this source.
    if (root / ".git").exists():
        try:
            repository_root = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if Path(repository_root).resolve() == root:
                commit = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                    check=True, capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                provenance["git_commit"] = commit
        except (OSError, subprocess.SubprocessError):
            # Exported source packages and environments without Git still have
            # complete per-file and combined hashes.
            pass
    return provenance


def write_manifest(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    data.setdefault("dependencies", dependency_versions())
    if "source_provenance" not in data:
        data["source_provenance"] = source_provenance()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def get_resource_path(relative_path: str) -> str:
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(getattr(sys, "_MEIPASS")) / relative_path)
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent / relative_path,
            here.parent.parent / relative_path,
            here.parent.parent / "assets" / relative_path,
            Path.cwd() / relative_path,
            Path(sys.executable).resolve().parent / relative_path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0] if candidates else Path(relative_path).resolve())
