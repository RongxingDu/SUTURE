#!/usr/bin/env python3
"""Build a hash-bound Vanilla/AFlow/S-CWU comparison specification.

This command performs only offline artifact validation.  It does not import
adapter factories, initialize provider clients, optimize a workflow, or touch
the held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import _interpolate_env_vars, _load_raw_config
from awf.config.schema import ExperimentConfig
from awf.protocol.manifest import (
    ManifestValidationError,
    file_sha256,
    json_sha256,
    validate_manifest_shape,
    workflow_sha256,
)
from awf.protocol.heldout import heldout_test_ledger_path
from awf.protocol.output import OutputReservation
from awf.workflow.serializer import load_workflow
from experiments.comparison.adapters import validate_configured_specs_ready
from experiments.comparison.dataset_access import (
    validate_bound_dataset_source,
)
from experiments.comparison.models import PhaseTelemetry
from experiments.comparison.source_policy import (
    validate_generated_aflow_directory,
)
from experiments.scripts.run_method_comparison import _reject_sensitive_values
from experiments.scripts.run_test import (
    _official_test_already_recorded,
    _validate_checkpoint,
    _validate_config,
    _validate_manifest_indices,
    _validate_run_metadata,
)


_AWF_FACTORY = (
    "experiments.comparison.builtin_adapters:"
    "build_awf_artifact_adapter"
)
_AFLOW_FACTORY = (
    "experiments.comparison.builtin_adapters:"
    "build_aflow_artifact_adapter"
)
_BENCHMARKS = ("gpqa", "mmlu", "scicode")
_AFLOW_ENVELOPE_V1_FIELDS = {
    "schema_version",
    "method",
    "phase",
    "benchmark",
    "round",
    "start_time",
    "end_time",
    "phase_wall_seconds",
    "returncode",
    "dataset_ordered_fingerprint_sha256",
    "optimization_indices_sha256",
    "artifact",
    "total",
    "by_role",
}
_AFLOW_ENVELOPE_V2_FIELDS = _AFLOW_ENVELOPE_V1_FIELDS | {"provenance"}
_AFLOW_ARTIFACT_V1_FIELDS = {"kind", "relative_path", "sha256"}
_AFLOW_ARTIFACT_V2_FIELDS = _AFLOW_ARTIFACT_V1_FIELDS | {"base"}
_AFLOW_PROVENANCE_FIELDS = {"config", "runtime", "call_telemetry"}
_AFLOW_CONFIG_PROVENANCE_FIELDS = {
    "config_file_sha256",
    "execution_model",
    "optimizer_model",
    "execution_role_config_sha256",
    "optimizer_role_config_sha256",
    "execution_thinking",
    "optimizer_thinking",
    "sdk_max_retries",
}
_AFLOW_RUNTIME_PROVENANCE_FIELDS = {
    "aflow_git_commit",
    "runtime_tree_sha256",
}
_AFLOW_CALL_PROVENANCE_FIELDS = {
    "relative_path",
    "sha256",
    "logical_call_count",
    "counting_semantics",
}
_AFLOW_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "call_count",
    "duration_seconds",
}
_ENV_CREDENTIAL = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an offline-validated Vanilla/AFlow/S-CWU comparison "
            "specification"
        )
    )
    parser.add_argument("--benchmark", choices=_BENCHMARKS, required=True)
    parser.add_argument("--awf-results", required=True)
    parser.add_argument("--initial-workflow", required=True)
    parser.add_argument("--aflow-graph", required=True)
    parser.add_argument("--aflow-search-telemetry", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--aflow-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scicode-hdf5",
        help="Official SciCode test_data.h5 (required only for SciCode)",
    )
    parser.add_argument(
        "--scicode-protocol",
        choices=["first_subproblem", "independent_subproblems"],
        default="first_subproblem",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_path = _path(args.output)
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    try:
        with OutputReservation(output_path) as output_reservation:
            spec = prepare_comparison_spec(
                benchmark=args.benchmark,
                awf_results_path=_path(args.awf_results),
                initial_workflow_path=_path(args.initial_workflow),
                aflow_graph_path=_path(args.aflow_graph),
                aflow_search_telemetry_path=_path(
                    args.aflow_search_telemetry
                ),
                config_path=_path(args.config),
                data_path=_path(args.data),
                aflow_root=_path(args.aflow_root),
                scicode_hdf5_path=(
                    _path(args.scicode_hdf5)
                    if args.scicode_hdf5
                    else None
                ),
                scicode_protocol=args.scicode_protocol,
            )
            output_reservation.commit_json(spec)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output: {output_path}")
    except (
        FileNotFoundError,
        ManifestValidationError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(1, f"comparison preparation failed: {exc}\n")
    print(f"comparison spec: {output_path}")
    print(
        "next: run validation selection first; do not access test until the "
        "roster is frozen"
    )


def prepare_comparison_spec(
    *,
    benchmark: str,
    awf_results_path: Path,
    initial_workflow_path: Path,
    aflow_graph_path: Path,
    aflow_search_telemetry_path: Path,
    config_path: Path,
    data_path: Path,
    aflow_root: Path,
    scicode_hdf5_path: Path | None = None,
    scicode_protocol: str = "first_subproblem",
) -> dict[str, Any]:
    """Validate all inputs and return a directly consumable comparison spec."""
    benchmark = str(benchmark).strip().lower()
    if benchmark not in _BENCHMARKS:
        raise ValueError(
            f"benchmark must be one of {', '.join(_BENCHMARKS)}"
        )
    if benchmark != "gpqa":
        raise ValueError(
            "The current local AFlow extension and one-update search pipeline "
            "are implemented only for GPQA. MMLU and SciCode remain valid "
            "AWF/Vanilla benchmarks, but cannot be labelled as an AFlow "
            "baseline until benchmark-specific AFlow search support exists."
        )
    paths = {
        "AWF results": awf_results_path,
        "initial workflow": initial_workflow_path,
        "AFlow search telemetry": aflow_search_telemetry_path,
        "config": config_path,
        "dataset": data_path,
        "AFlow root": aflow_root,
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not awf_results_path.is_file():
        raise ValueError("AWF results must be a file")
    if not initial_workflow_path.is_file():
        raise ValueError("initial workflow must be a file")
    if not aflow_search_telemetry_path.is_file():
        raise ValueError("AFlow search telemetry must be a file")
    if not config_path.is_file() or not data_path.is_file():
        raise ValueError("config and dataset inputs must be files")
    if not aflow_root.is_dir():
        raise ValueError("AFlow root must be a directory")
    if not (aflow_root / "scripts" / "async_llm.py").is_file():
        raise ValueError("AFlow root has no scripts/async_llm.py")
    if benchmark == "scicode":
        if scicode_hdf5_path is None or not scicode_hdf5_path.is_file():
            raise FileNotFoundError(
                "SciCode preparation requires the official test_data.h5"
            )
    elif scicode_hdf5_path is not None:
        raise ValueError("--scicode-hdf5 applies only to SciCode")

    results = _read_json_object(awf_results_path, "AWF results")
    if _official_test_already_recorded(results):
        raise ValueError(
            "AWF results already contain held-out test outcomes; method "
            "selection must be prepared before any test access"
        )
    comparison_ledger = heldout_test_ledger_path(awf_results_path)
    if comparison_ledger.exists():
        raise ValueError(
            "A held-out comparison attempt is already sealed; refusing to "
            "prepare a new method roster"
        )
    manifest = validate_manifest_shape(results.get("manifest"))
    if manifest["run_metadata"].get("benchmark") != benchmark:
        raise ManifestValidationError(
            "AWF manifest benchmark does not match --benchmark"
        )

    config = _load_config_without_credentials(config_path)
    _validate_config(config, manifest)
    use_scicode_sandbox = benchmark == "scicode"
    _validate_run_metadata(
        manifest,
        benchmark,
        False,
        False,
        use_bubblewrap_code_sandbox=use_scicode_sandbox,
        scicode_hdf5_path=scicode_hdf5_path,
        scicode_protocol=scicode_protocol,
    )
    validate_bound_dataset_source(data_path, manifest, require_seal=True)
    _validate_manifest_indices(manifest)
    if benchmark == "scicode" and scicode_protocol != "first_subproblem":
        raise ValueError(
            "Frozen method comparison currently requires SciCode "
            "first_subproblem so one JSONL row maps to one manifest sample"
        )

    initial_workflow = load_workflow(initial_workflow_path)
    initial_semantic_sha = workflow_sha256(initial_workflow)
    checkpoint_workflow = _validate_checkpoint(
        awf_results_path.parent,
        results,
        manifest,
    )
    if not config.optimizer.selective_update_enabled:
        raise ValueError(
            "AWF config did not enable selective_update; it cannot be "
            "labelled S-CWU"
        )
    if checkpoint_workflow.selective_update is None:
        raise ValueError(
            "Selected AWF checkpoint has no deployed selective update; it "
            "cannot be labelled S-CWU"
        )
    checkpoint_path = (
        awf_results_path.parent / "checkpoints" / "best_workflow.yaml"
    ).resolve()

    awf_search = _awf_search_telemetry(results)
    aflow_envelope = _read_json_object(
        aflow_search_telemetry_path,
        "AFlow search telemetry",
    )
    _reject_sensitive_values(aflow_envelope, "aflow_search_telemetry")
    round_dir, graph_path = _resolve_aflow_round_two(
        aflow_graph_path,
        aflow_root,
        envelope=aflow_envelope,
        envelope_path=aflow_search_telemetry_path,
    )
    aflow_search = _validate_aflow_search_envelope(
        aflow_envelope,
        benchmark=benchmark,
        manifest=manifest,
        aflow_root=aflow_root,
        round_dir=round_dir,
        envelope_path=aflow_search_telemetry_path,
        expected_execution_model=config.scheduler.llm.model,
        expected_optimizer_model=config.optimizer.llm.model,
        expected_sdk_max_retries=config.scheduler.llm.max_retries,
    )

    optimization_indices = list(
        manifest["split"]["indices"]["optimization"]
    )
    common_runtime = {
        "config_path": str(config_path.resolve()),
        "benchmark": benchmark,
    }
    zero = PhaseTelemetry().to_dict()
    spec = {
        "benchmark": benchmark,
        "data": str(data_path.resolve()),
        "protocol_results": str(awf_results_path.resolve()),
        "reference_method": "vanilla",
        "methods": [
            {
                "name": "vanilla",
                "kind": "vanilla",
                "factory": _AWF_FACTORY,
                "artifact_path": str(initial_workflow_path.resolve()),
                "search_telemetry": zero,
                "identity_metadata": {
                    "runtime": dict(common_runtime),
                    "provenance": {
                        "role": "unoptimized_initial_workflow",
                        "workflow_sha256": initial_semantic_sha,
                        "workflow_file_sha256": file_sha256(
                            initial_workflow_path
                        ),
                        "search_cost_policy": "exact_zero",
                    },
                },
            },
            {
                "name": "aflow",
                "kind": "aflow",
                "factory": _AFLOW_FACTORY,
                "artifact_path": str(round_dir),
                "search_telemetry": aflow_search.to_dict(),
                "identity_metadata": {
                    "runtime": {
                        **common_runtime,
                        "aflow_root": str(aflow_root.resolve()),
                    },
                    "provenance": {
                        "role": "one_real_workflow_update",
                        "round": 2,
                        "artifact_sha256": _artifact_sha256(round_dir),
                        "graph_file_sha256": file_sha256(graph_path),
                        "search_telemetry_file_sha256": file_sha256(
                            aflow_search_telemetry_path
                        ),
                        "search_envelope_schema_version": aflow_envelope[
                            "schema_version"
                        ],
                        **(
                            {
                                "search_provenance": aflow_envelope[
                                    "provenance"
                                ]
                            }
                            if aflow_envelope["schema_version"] >= 2
                            else {}
                        ),
                        "dataset_ordered_fingerprint_sha256": manifest[
                            "dataset"
                        ]["ordered_fingerprint_sha256"],
                        "optimization_indices_sha256": json_sha256(
                            optimization_indices
                        ),
                    },
                },
            },
            {
                "name": "scwu",
                "kind": "scwu",
                "factory": _AWF_FACTORY,
                "artifact_path": str(checkpoint_path),
                "search_telemetry": awf_search.to_dict(),
                "identity_metadata": {
                    "runtime": dict(common_runtime),
                    "provenance": {
                        "role": "validation_selected_scwu_checkpoint",
                        "results_file_sha256": file_sha256(
                            awf_results_path
                        ),
                        "manifest_sha256": json_sha256(manifest),
                        "checkpoint_round": manifest["best_checkpoint"][
                            "round"
                        ],
                        "checkpoint_file_sha256": manifest[
                            "best_checkpoint"
                        ]["file_sha256"],
                        "checkpoint_workflow_sha256": manifest[
                            "best_checkpoint"
                        ]["workflow_sha256"],
                    },
                },
            },
        ],
    }
    _reject_sensitive_values(spec)
    validate_configured_specs_ready(spec["methods"])
    return spec


def _load_config_without_credentials(path: Path) -> ExperimentConfig:
    """Load config semantics while refusing embedded credential material."""
    raw = _load_raw_config(path)

    def sanitize(value: Any, location: str) -> Any:
        if isinstance(value, Mapping):
            clean: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                normalized = key.lower().replace("-", "_")
                child_location = f"{location}.{key}"
                if _is_sensitive_key(normalized):
                    if (
                        not isinstance(item, str)
                        or _ENV_CREDENTIAL.fullmatch(item) is None
                    ):
                        raise ValueError(
                            "Credentials must use an environment-only "
                            f"placeholder, not an inline value: {child_location}"
                        )
                    clean[key] = "offline-comparison-placeholder"
                else:
                    clean[key] = sanitize(item, child_location)
            return clean
        if isinstance(value, list):
            return [
                sanitize(item, f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        return value

    sanitized = sanitize(raw, "config")
    interpolated = _interpolate_env_vars(sanitized)
    return ExperimentConfig(**interpolated)


def _awf_search_telemetry(results: Mapping[str, Any]) -> PhaseTelemetry:
    usage = results.get("backend_usage")
    if not isinstance(usage, Mapping):
        raise ValueError("AWF results have no backend_usage")
    total = usage.get("total")
    if not isinstance(total, Mapping):
        raise ValueError("AWF results have no backend_usage.total")
    if total.get("available") is not True:
        raise ValueError("AWF backend search telemetry is unavailable")
    prompt = _nonnegative_int(total, "total_prompt_tokens")
    completion = _nonnegative_int(total, "total_completion_tokens")
    declared_total = _nonnegative_int(total, "total_tokens")
    calls = _nonnegative_int(total, "num_calls")
    latency = _nonnegative_number(total, "total_latency_seconds")
    if declared_total != prompt + completion:
        raise ValueError("AWF backend total_tokens is internally inconsistent")
    if calls <= 0:
        raise ValueError("S-CWU search telemetry contains no LLM calls")
    wall = _elapsed_seconds(
        results.get("start_time"),
        results.get("end_time"),
        label="AWF optimization",
    )
    return PhaseTelemetry(
        prompt_tokens=prompt,
        completion_tokens=completion,
        llm_calls=calls,
        llm_latency_seconds=latency,
        wall_latency_seconds=wall,
    )


def _resolve_aflow_round_two(
    supplied: Path,
    aflow_root: Path,
    *,
    envelope: Mapping[str, Any],
    envelope_path: Path,
) -> tuple[Path, Path]:
    supplied = supplied.resolve()
    aflow_root = aflow_root.resolve()
    round_dir = supplied if supplied.is_dir() else supplied.parent
    graph_path = round_dir / "graph.py"
    if round_dir.name != "round_2":
        raise ValueError(
            "AFlow baseline must be a real round_2 directory; round_1 is "
            "the unsearched initial workflow"
        )
    schema_version = envelope.get("schema_version")
    artifact = envelope.get("artifact")
    if not isinstance(artifact, Mapping):
        raise TypeError("AFlow telemetry artifact must be an object")
    relative_value = artifact.get("relative_path")
    if (
        not isinstance(relative_value, str)
        or not relative_value.strip()
        or Path(relative_value).is_absolute()
        or ".." in Path(relative_value).parts
    ):
        raise ValueError("AFlow artifact relative_path is unsafe")
    if schema_version == 1:
        try:
            round_dir.relative_to(aflow_root)
        except ValueError as exc:
            raise ValueError(
                "AFlow round_2 artifact is outside --aflow-root"
            ) from exc
        expected_round_dir = (aflow_root / relative_value).resolve()
    elif schema_version == 2:
        if artifact.get("base") != "search_run":
            raise ValueError(
                "AFlow schema-v2 artifact base must be search_run"
            )
        expected_round_dir = (
            envelope_path.resolve().parent / relative_value
        ).resolve()
    else:
        raise ValueError("Unsupported AFlow telemetry schema_version")
    if round_dir != expected_round_dir:
        raise ValueError(
            "Supplied AFlow round_2 is not the artifact bound by search telemetry"
        )
    if not graph_path.is_file() or graph_path.stat().st_size <= 0:
        raise FileNotFoundError(f"AFlow round_2 graph.py not found: {graph_path}")
    prompt_path = round_dir / "prompt.py"
    if not prompt_path.is_file() or prompt_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"AFlow round_2 prompt.py not found: {prompt_path}"
        )
    validate_generated_aflow_directory(round_dir)
    graph_text = graph_path.read_text(encoding="utf-8")
    if "round_1" in graph_text:
        raise ValueError("AFlow round_2 graph still imports round_1")
    if "round_2" not in graph_text:
        raise ValueError(
            "AFlow graph has no round_2 provenance in its generated imports"
        )
    return round_dir, graph_path


def _validate_aflow_search_envelope(
    value: Mapping[str, Any],
    *,
    benchmark: str,
    manifest: Mapping[str, Any],
    aflow_root: Path,
    round_dir: Path,
    envelope_path: Path,
    expected_execution_model: str,
    expected_optimizer_model: str,
    expected_sdk_max_retries: int,
) -> PhaseTelemetry:
    schema_version = value.get("schema_version")
    expected_fields = (
        _AFLOW_ENVELOPE_V1_FIELDS
        if schema_version == 1
        else _AFLOW_ENVELOPE_V2_FIELDS
        if schema_version == 2
        else None
    )
    if expected_fields is None:
        raise ValueError("Unsupported AFlow telemetry schema_version")
    _require_exact_fields(
        value,
        expected_fields,
        "AFlow telemetry envelope",
    )
    if value["method"] != "aflow" or value["phase"] != "search":
        raise ValueError("AFlow telemetry must describe method=aflow search")
    if str(value["benchmark"]).lower() != benchmark:
        raise ValueError("AFlow telemetry benchmark mismatch")
    if value["round"] != 2 or isinstance(value["round"], bool):
        raise ValueError("AFlow telemetry must bind one real update at round 2")
    if value["returncode"] != 0 or isinstance(value["returncode"], bool):
        raise ValueError("AFlow search did not complete successfully")

    expected_dataset = manifest["dataset"][
        "ordered_fingerprint_sha256"
    ]
    del expected_dataset
    expected_indices_sha = json_sha256(
        list(manifest["split"]["indices"]["optimization"])
    )
    del expected_indices_sha

    artifact = value["artifact"]
    if not isinstance(artifact, Mapping):
        raise TypeError("AFlow telemetry artifact must be an object")
    _require_exact_fields(
        artifact,
        (
            _AFLOW_ARTIFACT_V1_FIELDS
            if schema_version == 1
            else _AFLOW_ARTIFACT_V2_FIELDS
        ),
        "AFlow telemetry artifact",
    )
    if artifact["kind"] != "directory":
        raise ValueError("AFlow artifact kind must be directory")
    if schema_version == 1:
        relative_path = round_dir.relative_to(aflow_root.resolve()).as_posix()
    else:
        if artifact["base"] != "search_run":
            raise ValueError(
                "AFlow schema-v2 artifact base must be search_run"
            )
        relative_path = round_dir.relative_to(
            envelope_path.resolve().parent
        ).as_posix()
    if artifact["relative_path"] != relative_path:
        raise ValueError("AFlow artifact relative_path mismatch")
    _ = artifact.get("sha256")

    wall = _elapsed_seconds(
        value["start_time"],
        value["end_time"],
        label="AFlow search",
        require_timezone=True,
    )
    declared_wall = _finite_nonnegative(
        value["phase_wall_seconds"],
        "phase_wall_seconds",
    )
    if not math.isclose(
        wall,
        declared_wall,
        rel_tol=1e-6,
        abs_tol=1e-3,
    ):
        raise ValueError(
            "AFlow phase_wall_seconds does not match start/end time"
        )

    total = _validate_aflow_usage(value["total"], "AFlow total")
    by_role = value["by_role"]
    if not isinstance(by_role, Mapping) or not by_role:
        raise ValueError("AFlow by_role telemetry must be a non-empty object")
    role_totals = {
        str(role): _validate_aflow_usage(
            usage,
            f"AFlow role {role!r}",
        )
        for role, usage in by_role.items()
    }
    for required_role in ("workflow_optimizer", "execution_round_2"):
        if (
            required_role not in role_totals
            or role_totals[required_role]["call_count"] <= 0
        ):
            raise ValueError(
                f"AFlow telemetry lacks successful {required_role} calls; "
                "round_1 cannot be presented as a searched baseline"
            )
    for field in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "call_count",
    ):
        if total[field] != sum(role[field] for role in role_totals.values()):
            raise ValueError(f"AFlow total.{field} does not equal by_role sum")
    if not math.isclose(
        total["duration_seconds"],
        sum(role["duration_seconds"] for role in role_totals.values()),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "AFlow total.duration_seconds does not equal by_role sum"
        )
    if total["call_count"] <= 0:
        raise ValueError("AFlow search telemetry contains no LLM calls")
    if schema_version == 2:
        _validate_aflow_search_provenance(
            value["provenance"],
            envelope_path=envelope_path,
            expected_execution_model=expected_execution_model,
            expected_optimizer_model=expected_optimizer_model,
            expected_sdk_max_retries=expected_sdk_max_retries,
            expected_logical_call_count=int(total["call_count"]),
        )
    return PhaseTelemetry(
        prompt_tokens=total["input_tokens"],
        completion_tokens=total["output_tokens"],
        llm_calls=total["call_count"],
        llm_latency_seconds=total["duration_seconds"],
        wall_latency_seconds=wall,
    )


def _validate_aflow_search_provenance(
    value: Any,
    *,
    envelope_path: Path,
    expected_execution_model: str,
    expected_optimizer_model: str,
    expected_sdk_max_retries: int,
    expected_logical_call_count: int,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("AFlow provenance must be an object")
    _require_exact_fields(value, _AFLOW_PROVENANCE_FIELDS, "AFlow provenance")

    config = value["config"]
    if not isinstance(config, Mapping):
        raise TypeError("AFlow provenance.config must be an object")
    _require_exact_fields(
        config,
        _AFLOW_CONFIG_PROVENANCE_FIELDS,
        "AFlow provenance.config",
    )
    for field in (
        "config_file_sha256",
        "execution_role_config_sha256",
        "optimizer_role_config_sha256",
    ):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"AFlow provenance.config.{field} is missing")
    if config["execution_model"] != expected_execution_model:
        raise ValueError("AFlow search execution model mismatch")
    if config["optimizer_model"] != expected_optimizer_model:
        raise ValueError("AFlow search optimizer model mismatch")
    if config["execution_thinking"] != "disabled":
        raise ValueError("AFlow search execution thinking must be disabled")
    if config["optimizer_thinking"] != "enabled":
        raise ValueError("AFlow search optimizer thinking must be enabled")
    if (
        config["sdk_max_retries"] != expected_sdk_max_retries
        or isinstance(config["sdk_max_retries"], bool)
    ):
        raise ValueError("AFlow search SDK retry budget mismatch")

    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise TypeError("AFlow provenance.runtime must be an object")
    _require_exact_fields(
        runtime,
        _AFLOW_RUNTIME_PROVENANCE_FIELDS,
        "AFlow provenance.runtime",
    )
    commit = runtime["aflow_git_commit"]
    if commit is not None and (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError("AFlow provenance has an invalid git commit")
    if (
        not isinstance(runtime["runtime_tree_sha256"], str)
        or not runtime["runtime_tree_sha256"]
    ):
        raise ValueError("AFlow provenance runtime tree fingerprint is missing")

    calls = value["call_telemetry"]
    if not isinstance(calls, Mapping):
        raise TypeError("AFlow provenance.call_telemetry must be an object")
    _require_exact_fields(
        calls,
        _AFLOW_CALL_PROVENANCE_FIELDS,
        "AFlow provenance.call_telemetry",
    )
    relative_value = calls["relative_path"]
    if (
        not isinstance(relative_value, str)
        or not relative_value.strip()
        or Path(relative_value).is_absolute()
        or ".." in Path(relative_value).parts
    ):
        raise ValueError("AFlow call telemetry relative_path is unsafe")
    call_path = (envelope_path.resolve().parent / relative_value).resolve()
    try:
        call_path.relative_to(envelope_path.resolve().parent)
    except ValueError as exc:
        raise ValueError("AFlow call telemetry escapes the search run") from exc
    if not call_path.is_file():
        raise FileNotFoundError(f"AFlow call telemetry not found: {call_path}")
    if (
        not isinstance(calls["sha256"], str)
        or not calls["sha256"]
    ):
        raise ValueError("AFlow call telemetry fingerprint is missing")
    logical_count = calls["logical_call_count"]
    if (
        not isinstance(logical_count, int)
        or isinstance(logical_count, bool)
        or logical_count != expected_logical_call_count
    ):
        raise ValueError("AFlow logical call count mismatch")
    if calls["counting_semantics"] != (
        "one event per OpenAI SDK logical call; transport retries "
        "are not separately observable"
    ):
        raise ValueError("AFlow call telemetry semantics mismatch")
    line_count = sum(
        1
        for line in call_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if line_count != logical_count:
        raise ValueError("AFlow call telemetry line count mismatch")


def _validate_aflow_usage(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    _require_exact_fields(value, _AFLOW_USAGE_FIELDS, label)
    result: dict[str, int | float] = {
        "input_tokens": _nonnegative_int(value, "input_tokens"),
        "output_tokens": _nonnegative_int(value, "output_tokens"),
        "total_tokens": _nonnegative_int(value, "total_tokens"),
        "call_count": _nonnegative_int(value, "call_count"),
        "duration_seconds": _nonnegative_number(
            value,
            "duration_seconds",
        ),
    }
    if result["total_tokens"] != (
        result["input_tokens"] + result["output_tokens"]
    ):
        raise ValueError(f"{label}.total_tokens is inconsistent")
    return result


def _artifact_sha256(path: Path) -> str:
    """Mirror the directory identity used by comparison adapters."""
    if path.is_file():
        return file_sha256(path)
    files = sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and "__pycache__" not in child.parts
        and child.suffix != ".pyc"
    )
    if not files:
        raise ValueError(f"Artifact directory has no files: {path}")
    return json_sha256(
        [
            {
                "relative_path": child.relative_to(path).as_posix(),
                "sha256": file_sha256(child),
            }
            for child in files
        ]
    )


def _elapsed_seconds(
    start_value: Any,
    end_value: Any,
    *,
    label: str,
    require_timezone: bool = False,
) -> float:
    start = _parse_datetime(start_value, f"{label} start_time")
    end = _parse_datetime(end_value, f"{label} end_time")
    if require_timezone and (
        start.tzinfo is None or end.tzinfo is None
    ):
        raise ValueError(f"{label} timestamps must include a UTC offset")
    if (start.tzinfo is None) != (end.tzinfo is None):
        raise ValueError(f"{label} timestamps mix naive and aware values")
    seconds = (end - start).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{label} end_time must be after start_time")
    return seconds


def _parse_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO8601 string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO8601") from exc


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(
            f"{label} has unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise ValueError(
            f"{label} is missing fields: {', '.join(sorted(missing))}"
        )


def _nonnegative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _nonnegative_number(mapping: Mapping[str, Any], key: str) -> float:
    return _finite_nonnegative(mapping.get(key), key)


def _finite_nonnegative(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _is_sensitive_key(normalized: str) -> bool:
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
            (
                "_api_key",
                "_authorization",
                "_password",
                "_secret",
                "_token",
            )
        )
    )


if __name__ == "__main__":
    main()
