"""Local filesystem model registry for versioning, metadata logging, and rollback."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
from sklearn.pipeline import Pipeline

from config import METADATA_DIR, MODELS_DIR

APPROVED_FILE = MODELS_DIR / "approved.json"
REGISTRY_INDEX_FILE = METADATA_DIR / "registry_index.json"


def _atomic_write_json(file_path: Path, data: Any) -> None:
    """Atomically write JSON to avoid corrupted reads during concurrent accesses."""
    temp_path = file_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temp_path, file_path)


def generate_version_id() -> str:
    """Generate a clean, sortable version identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"v_{ts}_{short_uuid}"


def save_model_version(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    *,
    make_approved: bool = False,
) -> str:
    """Serialize pipeline and metadata into versioned directory within model registry."""
    version_id = metadata.get("version_id") or generate_version_id()
    metadata["version_id"] = version_id
    metadata["created_at"] = metadata.get("created_at") or datetime.now(timezone.utc).isoformat()
    metadata["status"] = "approved" if make_approved else metadata.get("status", "candidate_only")

    version_dir = MODELS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    # Serialize scikit-learn pipeline safely
    model_path = version_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Save metadata JSON
    meta_path = version_dir / "metadata.json"
    _atomic_write_json(meta_path, metadata)

    # Update approved pointer if approved
    if make_approved:
        _update_approved_pointer(version_id, metadata)

    return version_id


def _update_approved_pointer(version_id: str, metadata: dict[str, Any]) -> None:
    """Update approved.json pointer to the designated version."""
    payload = {
        "approved_version_id": version_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata.get("model_name"),
        "primary_metric": metadata.get("primary_metric"),
        "metrics": metadata.get("metrics"),
        "target_column": metadata.get("target_column"),
        "task_type": metadata.get("task_type"),
    }
    _atomic_write_json(APPROVED_FILE, payload)


def get_approved_model() -> tuple[Pipeline | None, dict[str, Any] | None]:
    """Retrieve the currently approved production model and its metadata."""
    if not APPROVED_FILE.exists():
        return None, None
    try:
        data = json.loads(APPROVED_FILE.read_text(encoding="utf-8"))
        version_id = data.get("approved_version_id")
        if not version_id:
            return None, None
        pipeline, meta = get_model_by_version(version_id)
        meta["approved_version_id"] = version_id
        return pipeline, meta
    except Exception:
        return None, None


def get_model_by_version(version_id: str) -> tuple[Pipeline, dict[str, Any]]:
    """Load model pipeline and metadata for a specific version ID."""
    version_dir = MODELS_DIR / version_id
    model_path = version_dir / "model.joblib"
    meta_path = version_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model version '{version_id}' artifacts not found.")

    pipeline: Pipeline = joblib.load(model_path)
    metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def list_all_versions() -> list[dict[str, Any]]:
    """List all registered candidate and approved model versions sorted newest first."""
    versions = []
    if not MODELS_DIR.exists():
        return versions

    for path in sorted(MODELS_DIR.glob("v_*"), reverse=True):
        meta_file = path / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                versions.append(meta)
            except Exception:
                continue
    return versions


def promote_version(version_id: str) -> bool:
    """Promote an existing candidate or superseded version to approved status."""
    _, meta = get_model_by_version(version_id)
    meta["status"] = "approved"
    meta["approved_version_id"] = version_id
    meta["promoted_at"] = datetime.now(timezone.utc).isoformat()

    # Save updated metadata
    meta_path = MODELS_DIR / version_id / "metadata.json"
    _atomic_write_json(meta_path, meta)

    # Update pointer
    _update_approved_pointer(version_id, meta)
    return True


def rollback_to_version(version_id: str) -> bool:
    """Roll back approved model pointer to an earlier valid version."""
    return promote_version(version_id)
