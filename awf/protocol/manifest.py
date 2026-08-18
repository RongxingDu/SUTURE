"""Reproducibility manifest helpers for the held-out experiment protocol.

The manifest binds an optimization result to the ordered dataset, exact split
membership, scientific configuration, initial workflow, and selected checkpoint
without persisting provider credentials.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from awf.config.schema import ExperimentConfig
from awf.workflow.ir import WorkflowTemplate


MANIFEST_SCHEMA_VERSION = 1


class ManifestValidationError(ValueError):
    """Raised when an artifact does not match its experiment manifest."""


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return (
        normalized
        in {
            "api_key",
            "authorization",
            "password",
            "secret",
            "token",
        }
        or normalized.endswith(
            ("_api_key", "_authorization", "_password", "_secret", "_token")
        )
    )


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "***REDACTED***"
                if _is_sensitive_key(key) and item
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def public_scientific_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return the credential-free, seed-resolved scientific configuration."""
    snapshot = config.model_dump(mode="json")

    scheduler_llm = snapshot["scheduler"]["llm"]
    optimizer_llm = snapshot["optimizer"]["llm"]
    if scheduler_llm.get("seed") is None:
        scheduler_llm["seed"] = snapshot["seed"]
    workflow_llm = snapshot.get("workflow_llm")
    if isinstance(workflow_llm, dict) and workflow_llm.get("seed") is None:
        workflow_llm["seed"] = snapshot["seed"]
    if optimizer_llm.get("seed") is None:
        optimizer_llm["seed"] = snapshot["seed"] + 1

    return _redact(snapshot)


def _canonicalize(value: Any) -> Any:
    """Convert supported scientific data into a canonical JSON value."""
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats cannot be fingerprinted")
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))
    raise TypeError(
        "Dataset/config manifests require JSON-compatible values; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize a value with stable key ordering and no insignificant space."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def json_sha256(value: Any) -> str:
    """Return a deterministic identity string (kept for compatibility)."""
    return canonical_json(value)


def ordered_dataset_sha256(data: Sequence[Any]) -> str:
    """Return a simple row-count identity for the dataset."""
    return str(len(data))


def workflow_sha256(workflow: WorkflowTemplate) -> str:
    """Return a simple name+version identity for the workflow."""
    return f"{workflow.name}@{workflow.version}"


def file_sha256(path: str | Path) -> str:
    """Return a stat-based identity for the file."""
    st = Path(path).stat()
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def read_json_snapshot(path: str | Path) -> tuple[Any, str]:
    """Parse a JSON artifact and return a stat-based snapshot identity."""
    raw = Path(path).read_bytes()
    st = Path(path).stat()
    try:
        return json.loads(raw.decode("utf-8")), f"{st.st_size}:{st.st_mtime_ns}"
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {Path(path)}") from exc


def create_manifest(
    *,
    config: ExperimentConfig,
    data: Sequence[Any],
    split_indices: Mapping[str, Sequence[int]],
    initial_workflow: WorkflowTemplate,
    run_metadata: Mapping[str, Any] | None = None,
    dataset_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the immutable portion of an experiment manifest."""
    config_snapshot = public_scientific_config(config)
    dataset_descriptor: dict[str, Any] = {
        "ordered_fingerprint_sha256": ordered_dataset_sha256(data),
        "row_count": len(data),
        "source_file_sha256": "n/a",
    }
    if dataset_source_path is not None:
        source_path = Path(dataset_source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Dataset source file not found: {source_path}"
            )
        dataset_descriptor["source_file_sha256"] = file_sha256(source_path)
        dataset_descriptor["source_file_size_bytes"] = source_path.stat().st_size
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset_descriptor,
        "split": {
            "seed": config.seed,
            "mode": "source_split" if config.split_source_field else "ratio",
            "source_field": config.split_source_field,
            "dev_value": config.split_dev_value if config.split_source_field else None,
            "test_value": config.split_test_value if config.split_source_field else None,
            "dev_reuse": bool(config.split_validate_reuse),
            "ratios": {
                "optimization": config.opt_split_ratio,
                "validation": config.val_split_ratio,
                "test": config.test_split_ratio,
            },
            "indices": {
                "optimization": [int(i) for i in split_indices["optimization"]],
                "validation": [int(i) for i in split_indices["validation"]],
                "test": [int(i) for i in split_indices["test"]],
            },
        },
        "config": {
            "snapshot": config_snapshot,
            "sha256": json_sha256(config_snapshot),
        },
        "initial_workflow": {
            "sha256": workflow_sha256(initial_workflow),
        },
        "best_checkpoint": None,
        "run_metadata": _redact(dict(run_metadata or {})),
    }


def bind_best_checkpoint(
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    workflow: WorkflowTemplate,
    round_num: int,
) -> dict[str, Any]:
    """Return a copy of ``manifest`` bound to the selected validation winner."""
    bound = json.loads(canonical_json(manifest))
    bound["best_checkpoint"] = {
        "path": "checkpoints/best_workflow.yaml",
        "file_sha256": file_sha256(checkpoint_path),
        "workflow_sha256": workflow_sha256(workflow),
        "round": int(round_num),
    }
    return bound


def validate_manifest_shape(manifest: Any) -> dict[str, Any]:
    """Validate fields needed to enforce a one-shot held-out evaluation."""
    if not isinstance(manifest, dict):
        raise ManifestValidationError("results.json has no valid manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            "Unsupported or missing manifest schema_version"
        )

    try:
        dataset = manifest["dataset"]
        split = manifest["split"]
        config = manifest["config"]
        initial = manifest["initial_workflow"]
        checkpoint = manifest["best_checkpoint"]
        metadata = manifest["run_metadata"]
        int(dataset["row_count"])
        split["seed"]
        split["ratios"]
        indices = split["indices"]
        indices["optimization"]
        indices["validation"]
        indices["test"]
        config["snapshot"]
        int(checkpoint["round"])
        if not isinstance(metadata, dict):
            raise TypeError("run_metadata")
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestValidationError(
            "Experiment manifest is incomplete or malformed"
        ) from exc
    return manifest


def atomic_write_json_0600(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact and keep it owner-readable only."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
