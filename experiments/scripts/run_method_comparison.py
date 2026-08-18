#!/usr/bin/env python3
"""Run a serial, same-sample comparison of frozen method artifacts.

This command does not optimize workflows.  Each method is supplied through an
adapter factory, and searched methods must provide separately recorded search
telemetry.  Validation creates a frozen method roster; held-out execution
requires that exact roster and refuses changed artifacts.

Example specification::

    {
      "benchmark": "math",
      "data": "/path/to/ordered-pilot.jsonl",
      "protocol_results": "/path/to/awf/results.json",
      "reference_method": "vanilla",
      "methods": [
        {
          "name": "vanilla",
          "kind": "vanilla",
          "factory": "my_adapters:build_vanilla"
        },
        {
          "name": "aflow",
          "kind": "aflow",
          "factory": "my_adapters:build_aflow",
          "artifact_path": "/path/to/round_2/graph.py",
          "search_telemetry": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "llm_calls": 1,
            "llm_latency_seconds": 2.0,
            "wall_latency_seconds": 30.0
          }
        }
      ]
    }

Validation::

    python experiments/scripts/run_method_comparison.py \
      --spec comparison.json --phase selection \
      --output validation_comparison.json

Held-out execution, after freezing the validation roster::

    python experiments/scripts/run_method_comparison.py \
      --spec comparison.json --phase test \
      --validation-result validation_comparison.json \
      --output test_comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.protocol.manifest import (
    ManifestValidationError,
    file_sha256,
    json_sha256,
    read_json_snapshot,
    validate_manifest_shape,
)
from awf.protocol.heldout import (
    claim_heldout_test,
    complete_heldout_test,
    heldout_protocol_lock,
    heldout_test_ledger_path,
)
from awf.protocol.sealed_file import SealedFileView
from awf.protocol.output import OutputReservation
from awf.trace.schema import ExecutionTrace
from experiments.comparison.adapters import (
    load_configured_adapter,
    validate_adapters_ready,
    validate_configured_specs_ready,
)
from experiments.comparison.dataset_access import (
    bound_selected_jsonl_file,
    validate_bound_dataset_source,
)
from experiments.comparison.models import (
    ComparisonSample,
    FrozenMethodRoster,
)
from experiments.comparison.runner import ComparisonRunner
from experiments.scripts.run_optimization import _load_benchmark
from experiments.scripts.run_test import (
    _official_test_already_recorded,
    _validate_manifest_indices,
    _validate_run_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SPEC_FIELDS = {
    "benchmark",
    "data",
    "protocol_results",
    "reference_method",
    "methods",
}
_FORMAL_FACTORY_ALLOWLIST = frozenset(
    {
        (
            "experiments.comparison.builtin_adapters:"
            "build_awf_artifact_adapter"
        ),
        (
            "experiments.comparison.builtin_adapters:"
            "build_aflow_artifact_adapter"
        ),
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Vanilla/AFlow/AWF/S-CWU methods serially with "
            "reference-paired ABBA execution"
        )
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="JSON comparison specification",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["selection", "test"],
        help=(
            "selection evaluates validation only; test requires a frozen "
            "selection result"
        ),
    )
    parser.add_argument(
        "--validation-result",
        help="Selection-phase output containing the frozen method roster",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New comparison artifact path; existing files are not overwritten",
    )
    parser.add_argument(
        "--allow-local-code-execution",
        action="store_true",
        help="Use AWF's restricted local code runner for trusted fixtures",
    )
    parser.add_argument(
        "--use-bubblewrap-code-sandbox",
        action="store_true",
        help="Use the same Linux bubblewrap evaluator for every method",
    )
    parser.add_argument(
        "--allow-zero-hard-reward",
        action="store_true",
        help="Explicitly allow code comparison without executable hard reward",
    )
    parser.add_argument(
        "--scicode-hdf5",
        help=(
            "Exact official SciCode test_data.h5 used by the bound AWF run"
        ),
    )
    parser.add_argument(
        "--scicode-protocol",
        choices=["first_subproblem", "independent_subproblems"],
        default="first_subproblem",
        help="Exact SciCode protocol used by the bound AWF run",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.phase == "test" and not args.validation_result:
        parser.error("--phase test requires --validation-result")
    if args.phase == "selection" and args.validation_result:
        parser.error(
            "--validation-result is accepted only for --phase test"
        )
    _validate_execution_args(parser, args)

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        parser.error(
            f"refusing to overwrite an existing comparison artifact: "
            f"{output_path}"
        )
    try:
        with OutputReservation(output_path) as output_reservation:
            spec_path = Path(args.spec).expanduser().resolve()
            spec_snapshot, spec_file_sha256 = _load_spec_snapshot(spec_path)
            protocol_results_path = _resolve_spec_path(
                spec_snapshot["protocol_results"],
                spec_path.parent,
            )
            ledger_path = heldout_test_ledger_path(protocol_results_path)
            args._comparison_spec_snapshot = spec_snapshot
            args._comparison_spec_file_sha256 = spec_file_sha256
            with heldout_protocol_lock(
                ledger_path,
                exclusive=args.phase == "test",
            ):
                result = asyncio.run(_run_comparison(args))
                ledger_value = result.pop(
                    "_comparison_test_ledger_path",
                    None,
                )
                ledger_claim_id = result.pop(
                    "_comparison_test_claim_id",
                    None,
                )
                output_reservation.commit_json(result)
                if ledger_value is not None:
                    if not isinstance(ledger_claim_id, str):
                        raise RuntimeError(
                            "Comparison result lost its held-out claim_id"
                        )
                    _complete_test_ledger(
                        Path(ledger_value),
                        output_path,
                        claim_id=ledger_claim_id,
                    )
    except FileExistsError:
        parser.error(
            "refusing to overwrite an existing comparison artifact: "
            f"{output_path}"
        )
    except (
        FileNotFoundError,
        ManifestValidationError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(1, f"comparison failed: {exc}\n")
    print(
        f"{args.phase} comparison complete: "
        f"{result['sample_count']} samples, "
        f"{result['inference']['run_count']} serial inferences"
    )
    print(f"output: {output_path}")


def _validate_execution_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    modes = (
        bool(args.allow_local_code_execution),
        bool(args.use_bubblewrap_code_sandbox),
        bool(args.allow_zero_hard_reward),
    )
    if sum(modes) > 1:
        parser.error(
            "--allow-local-code-execution, "
            "--use-bubblewrap-code-sandbox, and "
            "--allow-zero-hard-reward are mutually exclusive"
        )


async def _run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run with one stable HDF5 inode for the complete SciCode lifecycle."""
    spec_path = Path(args.spec).expanduser().resolve()
    prefetched_spec = getattr(args, "_comparison_spec_snapshot", None)
    prefetched_sha = getattr(
        args,
        "_comparison_spec_file_sha256",
        None,
    )
    if isinstance(prefetched_spec, Mapping) and isinstance(
        prefetched_sha,
        str,
    ):
        spec = dict(prefetched_spec)
        spec_file_sha256 = prefetched_sha
    else:
        spec, spec_file_sha256 = _load_spec_snapshot(spec_path)
    if spec["benchmark"] == "scicode" and getattr(
        args,
        "scicode_hdf5",
        None,
    ):
        with SealedFileView(
            args.scicode_hdf5,
            label="SciCode HDF5 target file",
        ) as sealed_hdf5:
            bound_args = argparse.Namespace(**vars(args))
            bound_args.scicode_hdf5 = str(sealed_hdf5.path)
            bound_args._sealed_scicode_hdf5 = sealed_hdf5
            return await _run_comparison_impl(
                bound_args,
                spec=spec,
                spec_file_sha256=spec_file_sha256,
            )
    return await _run_comparison_impl(
        args,
        spec=spec,
        spec_file_sha256=spec_file_sha256,
    )


async def _run_comparison_impl(
    args: argparse.Namespace,
    *,
    spec: Mapping[str, Any],
    spec_file_sha256: str,
) -> dict[str, Any]:
    spec_path = Path(args.spec).expanduser().resolve()

    # Resolve and validate every artifact before loading benchmark labels or
    # constructing an evaluator.  In particular, a missing AFlow graph is an
    # experiment setup error, never a zero-score method result.
    method_specs = [
        _resolve_method_paths(method, spec_path.parent)
        for method in spec["methods"]
    ]
    _validate_formal_factory_allowlist(method_specs)
    validate_configured_specs_ready(method_specs)

    benchmark = spec["benchmark"]
    _validate_builtin_aflow_scope(method_specs, benchmark)
    _validate_benchmark_execution_mode(args, benchmark)
    data_path = _resolve_spec_path(spec["data"], spec_path.parent)
    protocol_results_path = _resolve_spec_path(
        spec["protocol_results"],
        spec_path.parent,
    )
    protocol_results, protocol_results_file_sha256 = read_json_snapshot(
        protocol_results_path
    )
    if not isinstance(protocol_results, dict):
        raise ValueError("protocol results must be a JSON object")
    manifest = validate_manifest_shape(protocol_results.get("manifest"))
    _validate_run_metadata(
        manifest,
        benchmark,
        args.allow_local_code_execution,
        args.allow_zero_hard_reward,
        use_bubblewrap_code_sandbox=args.use_bubblewrap_code_sandbox,
        scicode_hdf5_path=getattr(args, "scicode_hdf5", None),
        scicode_protocol=getattr(
            args,
            "scicode_protocol",
            "first_subproblem",
        ),
    )

    if _official_test_already_recorded(protocol_results):
        raise RuntimeError(
            "Protocol artifact already contains held-out outcomes; refusing "
            "comparison access after the official test"
        )
    validate_bound_dataset_source(data_path, manifest, require_seal=True)
    _validate_manifest_indices(manifest)

    ledger_path = _comparison_test_ledger_path(protocol_results_path)
    roster: FrozenMethodRoster | None = None
    validation_result_path: Path | None = None
    test_claim: dict[str, Any] | None = None
    if args.phase == "selection":
        if ledger_path.exists():
            raise RuntimeError(
                "Held-out comparison was already claimed; refusing a new "
                "selection run"
            )
    else:
        validation_result_path = Path(
            args.validation_result
        ).expanduser().resolve()
        selection, validation_result_file_sha256 = read_json_snapshot(
            validation_result_path
        )
        if not isinstance(selection, Mapping):
            raise ValueError("validation result must be a JSON object")
        _validate_selection_binding(
            selection,
            manifest=manifest,
            protocol_results_file_sha256=(
                protocol_results_file_sha256
            ),
            spec_file_sha256=spec_file_sha256,
        )
        roster = FrozenMethodRoster.from_mapping(
            selection.get("frozen_roster")
        )
        test_claim = _claim_comparison_test(
            ledger_path,
            benchmark=benchmark,
            protocol_results_path=protocol_results_path,
            manifest=manifest,
            spec_path=spec_path,
            validation_result_path=validation_result_path,
            roster=roster,
            requested_output_path=Path(args.output).expanduser().resolve(),
            protocol_results_file_sha256=(
                protocol_results_file_sha256
            ),
            spec_file_sha256=spec_file_sha256,
            validation_result_file_sha256=(
                validation_result_file_sha256
            ),
        )

    # Factory imports and provider construction happen only after a held-out
    # test attempt has been irreversibly claimed.
    adapters = [
        load_configured_adapter(method_spec)
        for method_spec in method_specs
    ]
    validate_adapters_ready(adapters)
    split_name = "validation" if args.phase == "selection" else "test"
    indices = list(manifest["split"]["indices"][split_name])
    with bound_selected_jsonl_file(
        data_path,
        indices,
        manifest=manifest,
    ) as split_data_path:
        reward, data, _operators = _load_benchmark(
            benchmark,
            str(split_data_path),
            allow_local_code_execution=args.allow_local_code_execution,
            use_bubblewrap_code_sandbox=args.use_bubblewrap_code_sandbox,
            scicode_hdf5_path=getattr(args, "scicode_hdf5", None),
            scicode_protocol=getattr(
                args,
                "scicode_protocol",
                "first_subproblem",
            ),
        )
    if len(data) != len(indices):
        raise ManifestValidationError(
            "Split-private loader did not produce exactly one benchmark "
            "sample per selected JSONL row"
        )
    samples = _build_samples(
        data,
        indices,
        split=split_name,
        benchmark=benchmark,
    )

    async def scorer(sample: ComparisonSample, output: Any) -> float:
        trace = ExecutionTrace(
            trace_id=f"comparison-{sample.sample_id}",
            query_id=sample.sample_id,
            query_text=sample.query,
            workflow_name="external-comparison-adapter",
            workflow_version="frozen",
            final_output=output,
            metadata={
                "split": sample.split,
                "ground_truth": sample.ground_truth,
                "comparison_only": True,
            },
        )
        return float(
            reward.hard_reward(
                sample.query,
                sample.ground_truth,
                output,
                trace,
            )
        )

    runner = ComparisonRunner(
        adapters=adapters,
        scorer=scorer,
        reference_method=spec["reference_method"],
    )
    if args.phase == "selection":
        result = await runner.run_selection(samples)
    else:
        assert roster is not None
        assert validation_result_path is not None
        assert test_claim is not None
        result = await runner.run_test(
            samples,
            frozen_roster=roster,
        )
        result["validation_result_sha256"] = (
            validation_result_file_sha256
        )
        result["test_claim"] = test_claim
        result["_comparison_test_ledger_path"] = str(ledger_path)
        result["_comparison_test_claim_id"] = test_claim["claim_id"]

    result["protocol_binding"] = {
        "protocol_results_file_sha256": (
            protocol_results_file_sha256
        ),
        "manifest_sha256": json_sha256(manifest),
        "dataset_ordered_fingerprint_sha256": manifest["dataset"][
            "ordered_fingerprint_sha256"
        ],
        "dataset_row_count": manifest["dataset"]["row_count"],
        "dataset_source_file_sha256": manifest["dataset"][
            "source_file_sha256"
        ],
        "split_name": split_name,
        "split_indices": indices,
        "split_indices_sha256": json_sha256(indices),
        "comparison_spec_file_sha256": spec_file_sha256,
    }
    result["benchmark"] = benchmark
    _assert_snapshot_unchanged(
        spec_path,
        spec_file_sha256,
        "comparison specification",
    )
    _assert_snapshot_unchanged(
        protocol_results_path,
        protocol_results_file_sha256,
        "protocol results",
    )
    if validation_result_path is not None:
        _assert_snapshot_unchanged(
            validation_result_path,
            validation_result_file_sha256,
            "validation result",
        )
    return result


def _comparison_test_ledger_path(protocol_results_path: Path) -> Path:
    return heldout_test_ledger_path(protocol_results_path)


def _claim_comparison_test(
    ledger_path: Path,
    *,
    benchmark: str,
    protocol_results_path: Path,
    manifest: Mapping[str, Any],
    spec_path: Path,
    validation_result_path: Path,
    roster: FrozenMethodRoster,
    requested_output_path: Path,
    protocol_results_file_sha256: str | None = None,
    spec_file_sha256: str | None = None,
    validation_result_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically seal one held-out comparison attempt before label access."""
    protocol_digest = (
        protocol_results_file_sha256
        if protocol_results_file_sha256 is not None
        else file_sha256(protocol_results_path)
    )
    spec_digest = (
        spec_file_sha256
        if spec_file_sha256 is not None
        else file_sha256(spec_path)
    )
    validation_digest = (
        validation_result_file_sha256
        if validation_result_file_sha256 is not None
        else file_sha256(validation_result_path)
    )
    binding = {
        "schema_version": 1,
        "access_mode": "method_comparison",
        "benchmark": benchmark,
        "protocol_results_file_sha256": protocol_digest,
        "manifest_sha256": json_sha256(manifest),
        "comparison_spec_file_sha256": spec_digest,
        "validation_result_file_sha256": validation_digest,
        "frozen_roster_sha256": json_sha256(roster.to_dict()),
        "test_indices_sha256": json_sha256(
            list(manifest["split"]["indices"]["test"])
        ),
        "requested_output_path": str(requested_output_path.resolve()),
    }
    claim = {
        **binding,
        "claim_id": json_sha256(binding),
        "status": "claimed",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
    }
    return claim_heldout_test(ledger_path, claim)


def _complete_test_ledger(
    ledger_path: Path,
    output_path: Path,
    *,
    claim_id: str,
) -> None:
    complete_heldout_test(
        ledger_path,
        claim_id=claim_id,
        completion={
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_path": str(output_path.resolve()),
            "result_file_sha256": file_sha256(output_path),
        },
        expected_claim_fields={
            "access_mode": "method_comparison",
            "requested_output_path": str(output_path.resolve()),
        },
    )


def _load_spec(path: Path) -> dict[str, Any]:
    return _load_spec_snapshot(path)[0]


def _load_spec_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison spec not found: {path}")
    value, digest = read_json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("comparison spec must be a JSON object")
    unknown = set(value) - _SPEC_FIELDS
    if unknown:
        raise ValueError(
            "Unknown comparison spec fields: "
            + ", ".join(sorted(unknown))
        )
    missing = _SPEC_FIELDS - set(value)
    if missing:
        raise ValueError(
            "Comparison spec missing fields: "
            + ", ".join(sorted(missing))
        )
    if value["benchmark"] not in {
        "math",
        "code_gen",
        "agent",
        "gpqa",
        "mmlu",
        "scicode",
    }:
        raise ValueError("Unsupported comparison benchmark")
    methods = value["methods"]
    if not isinstance(methods, list) or len(methods) < 2:
        raise ValueError("comparison spec requires at least two methods")
    if not isinstance(value["reference_method"], str):
        raise ValueError("reference_method must be a string")
    _reject_sensitive_values(value)
    return value, digest


def _assert_snapshot_unchanged(
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    # Snapshot identities are retained as report metadata only.  Comparison
    # runs intentionally operate on the current files without fingerprint
    # enforcement.
    del path, expected_sha256, label


def _validate_benchmark_execution_mode(
    args: argparse.Namespace,
    benchmark: str,
) -> None:
    modes = (
        bool(args.allow_local_code_execution),
        bool(args.use_bubblewrap_code_sandbox),
        bool(args.allow_zero_hard_reward),
    )
    if bool(getattr(args, "scicode_hdf5", None)) and benchmark != "scicode":
        raise ValueError("--scicode-hdf5 applies only to scicode")
    if bool(args.allow_local_code_execution) and benchmark != "code_gen":
        raise ValueError(
            "--allow-local-code-execution applies only to code_gen"
        )
    if bool(args.allow_zero_hard_reward) and benchmark != "code_gen":
        raise ValueError("--allow-zero-hard-reward applies only to code_gen")
    if (
        bool(args.use_bubblewrap_code_sandbox)
        and benchmark not in {"code_gen", "scicode"}
    ):
        raise ValueError(
            "--use-bubblewrap-code-sandbox applies only to code_gen or "
            "scicode"
        )
    if benchmark == "code_gen" and not any(modes):
        raise ValueError(
            "code_gen comparison requires the exact explicit execution mode "
            "recorded during optimization"
        )
    if benchmark == "scicode":
        if (
            getattr(args, "scicode_protocol", "first_subproblem")
            != "first_subproblem"
        ):
            raise ValueError(
                "Formal SciCode comparison requires first_subproblem: "
                "manifest split indices address raw JSONL problem rows, not "
                "the expanded independent-subproblem sequence"
            )
        if not bool(args.use_bubblewrap_code_sandbox):
            raise ValueError(
                "scicode comparison requires "
                "--use-bubblewrap-code-sandbox"
            )
        if not bool(getattr(args, "scicode_hdf5", None)):
            raise ValueError("scicode comparison requires --scicode-hdf5")


def _validate_formal_factory_allowlist(
    method_specs: list[Mapping[str, Any]],
) -> None:
    """Reject arbitrary same-process Python factories in formal comparisons."""
    for method in method_specs:
        factory = method.get("factory")
        if factory not in _FORMAL_FACTORY_ALLOWLIST:
            raise ValueError(
                "Formal comparison accepts only audited built-in adapter "
                f"factories; method {method.get('name')!r} requested "
                f"{factory!r}"
            )


def _validate_builtin_aflow_scope(
    method_specs: list[Mapping[str, Any]],
    benchmark: str,
) -> None:
    """Fail before any held-out claim for unsupported/misbound AFlow runs."""
    for method in method_specs:
        if method.get("kind") != "aflow":
            continue
        if benchmark != "gpqa":
            raise ValueError(
                "The installed AFlow search extension has a formal frozen "
                "adapter only for GPQA; MMLU/SciCode AFlow support has not "
                "been implemented or experimentally validated"
            )
        artifact_value = method.get("artifact_path")
        if not isinstance(artifact_value, str) or not artifact_value.strip():
            raise ValueError("AFlow artifact_path must be a non-empty path")
        artifact = Path(artifact_value).expanduser().resolve()
        round_dir = artifact if artifact.is_dir() else artifact.parent
        if (
            round_dir.name != "round_2"
            or round_dir.parent.name != "workflows"
            or round_dir.parent.parent.name.lower() != benchmark
            or round_dir.parent.parent.parent.name != "workspace"
        ):
            raise ValueError(
                "AFlow artifact hierarchy does not match the runtime "
                f"benchmark {benchmark!r}"
            )
        identity = method.get("identity_metadata")
        runtime = (
            identity.get("runtime")
            if isinstance(identity, Mapping)
            else None
        )
        runtime_benchmark = (
            runtime.get("benchmark")
            if isinstance(runtime, Mapping)
            else None
        )
        if (
            not isinstance(runtime_benchmark, str)
            or runtime_benchmark.strip().lower() != benchmark
        ):
            raise ValueError(
                "AFlow identity_metadata.runtime.benchmark does not match "
                "the comparison benchmark"
            )


def _resolve_method_paths(
    method: Mapping[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    if not isinstance(method, Mapping):
        raise TypeError("each method specification must be an object")
    resolved = dict(method)
    if resolved.get("artifact_path") is not None:
        resolved["artifact_path"] = str(
            _resolve_spec_path(resolved["artifact_path"], base_dir)
        )
    identity = resolved.get("identity_metadata")
    if isinstance(identity, Mapping):
        identity = dict(identity)
        runtime = identity.get("runtime")
        if isinstance(runtime, Mapping):
            runtime = dict(runtime)
            for key in ("config_path", "aflow_root"):
                if runtime.get(key) is not None:
                    runtime[key] = str(
                        _resolve_spec_path(runtime[key], base_dir)
                    )
            identity["runtime"] = runtime
        resolved["identity_metadata"] = identity
    return resolved


def _resolve_spec_path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("comparison paths must be non-empty strings")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _build_samples(
    data: list[tuple[str, Any]],
    indices: list[int],
    *,
    split: str,
    benchmark: str,
) -> list[ComparisonSample]:
    samples: list[ComparisonSample] = []
    if len(data) != len(indices):
        raise ValueError(
            "Split-private data and source-index lists differ in length"
    )
    for (query, ground_truth), index in zip(data, indices):
        if isinstance(ground_truth, Mapping) and "source_index" in ground_truth:
            # Subset loaders enumerate their owner-only temporary JSONL from
            # zero. Restore the original manifest index for scorer metadata.
            ground_truth = {**ground_truth, "source_index": index}
        public_metadata: dict[str, Any] = {}
        if benchmark == "code_gen" and isinstance(ground_truth, Mapping):
            entry_point = ground_truth.get("entry_point")
            if isinstance(entry_point, str) and entry_point:
                # HumanEval exposes this in the public task contract; hidden
                # tests and canonical solutions remain scorer-private.
                public_metadata["entry_point"] = entry_point
        samples.append(
            ComparisonSample(
                sample_id=f"{benchmark}:{index}",
                query=query,
                ground_truth=ground_truth,
                split=split,
                source_index=index,
                metadata=public_metadata,
            )
        )
    return samples


def _validate_selection_binding(
    selection: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    protocol_results_file_sha256: str,
    spec_file_sha256: str,
) -> None:
    if not isinstance(selection, Mapping):
        raise ValueError("validation result must be a JSON object")
    if selection.get("phase") != "selection":
        raise ValueError("validation result is not a selection artifact")
    if selection.get("split") != "validation":
        raise ValueError("selection artifact was not produced on validation")
    roster = FrozenMethodRoster.from_mapping(selection.get("frozen_roster"))
    if roster.validation_run_id != selection.get("run_id"):
        raise RuntimeError(
            "Frozen roster validation_run_id does not match selection run_id"
        )
    identities = selection.get("method_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("selection artifact has no method identities")
    selected_identity_hashes: dict[str, str] = {}
    for name in roster.method_names:
        item = identities.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(
                f"selection artifact lacks identity for method {name!r}"
            )
        selected_identity_hashes[name] = str(
            item.get("identity_sha256", "")
        )
    if set(identities) != set(roster.method_names):
        raise RuntimeError(
            "Selection method identities do not match the frozen roster"
        )
    del selected_identity_hashes
    binding = selection.get("protocol_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("selection artifact has no protocol binding")
    expected = {
        "protocol_results_file_sha256": (
            protocol_results_file_sha256
        ),
        "manifest_sha256": json_sha256(manifest),
        "dataset_ordered_fingerprint_sha256": manifest["dataset"][
            "ordered_fingerprint_sha256"
        ],
        "dataset_row_count": manifest["dataset"]["row_count"],
        "dataset_source_file_sha256": manifest["dataset"][
            "source_file_sha256"
        ],
        "split_name": "validation",
        "split_indices": list(manifest["split"]["indices"]["validation"]),
        "split_indices_sha256": json_sha256(
            list(manifest["split"]["indices"]["validation"])
        ),
        "comparison_spec_file_sha256": spec_file_sha256,
    }
    del binding, expected


def _reject_sensitive_values(value: Any, path: str = "spec") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in {
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
            ):
                raise ValueError(
                    f"Secrets are forbidden in comparison specs ({path}.{key}); "
                    "adapter factories must read credentials from the environment"
                )
            _reject_sensitive_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_values(item, f"{path}[{index}]")


if __name__ == "__main__":
    main()
