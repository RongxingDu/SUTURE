#!/usr/bin/env python3
"""One-shot official evaluation of a manifest-bound held-out test split.

Usage:
    awf-test --results experiments/results/code_gen/results.json \
      --config experiments/configs/code_gen.yaml --benchmark code_gen \
      --data /path/to/humaneval.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import load_config
from awf.protocol.experiment import ExperimentRunner
from awf.protocol.manifest import (
    ManifestValidationError,
    atomic_write_json_0600,
    file_sha256,
    json_sha256,
    public_scientific_config,
    read_json_snapshot,
    validate_manifest_shape,
    workflow_sha256,
)
from awf.protocol.heldout import (
    claim_heldout_test,
    complete_heldout_test,
    heldout_protocol_lock,
    heldout_test_ledger_path,
)
from awf.protocol.sealed_file import SealedFileView
from awf.workflow.serializer import load_workflow
from experiments.comparison.dataset_access import (
    validate_bound_dataset_source,
)
from experiments.scripts.run_optimization import (
    _load_benchmark,
    _run_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run the official held-out test evaluation exactly once"
    )
    parser.add_argument(
        "--results",
        "-r",
        required=True,
        help="Path to manifest-bearing results.json from optimization",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="The exact experiment config YAML used for optimization",
    )
    parser.add_argument(
        "--benchmark",
        choices=[
            "code_gen",
            "math",
            "agent",
            "gpqa",
            "mmlu",
            "scicode",
        ],
        required=True,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="The exact, ordered full benchmark JSONL used for optimization",
    )
    parser.add_argument(
        "--allow-local-code-execution",
        action="store_true",
        help="Enable restricted local execution for trusted code fixtures only",
    )
    parser.add_argument(
        "--use-bubblewrap-code-sandbox",
        action="store_true",
        help=(
            "Match and use the Linux bubblewrap sandbox selected during "
            "optimization. Required for SciCode."
        ),
    )
    parser.add_argument(
        "--scicode-hdf5",
        help=(
            "The exact official SciCode test_data.h5 used during "
            "optimization."
        ),
    )
    parser.add_argument(
        "--scicode-protocol",
        choices=["first_subproblem", "independent_subproblems"],
        default="first_subproblem",
        help="The exact SciCode protocol used during optimization.",
    )
    parser.add_argument(
        "--allow-zero-hard-reward",
        action="store_true",
        help=(
            "Match an optimization manifest that explicitly disabled code "
            "execution and accepted zero hard reward."
        ),
    )
    args = parser.parse_args()
    if args.allow_zero_hard_reward and args.benchmark != "code_gen":
        parser.error("--allow-zero-hard-reward applies only to code_gen")
    if (
        args.allow_local_code_execution
        and args.benchmark != "code_gen"
    ):
        parser.error("--allow-local-code-execution applies only to code_gen")
    if (
        args.use_bubblewrap_code_sandbox
        and args.benchmark not in {"code_gen", "scicode"}
    ):
        parser.error(
            "--use-bubblewrap-code-sandbox applies only to code_gen or scicode"
        )
    if args.scicode_hdf5 and args.benchmark != "scicode":
        parser.error("--scicode-hdf5 applies only to scicode")
    if sum(
        (
            args.allow_zero_hard_reward,
            args.allow_local_code_execution,
            args.use_bubblewrap_code_sandbox,
        )
    ) > 1:
        parser.error(
            "--allow-zero-hard-reward, --allow-local-code-execution, and "
            "--use-bubblewrap-code-sandbox are mutually exclusive"
        )
    if args.benchmark == "scicode":
        if not args.scicode_hdf5:
            parser.error("scicode requires --scicode-hdf5")
        if not args.use_bubblewrap_code_sandbox:
            parser.error(
                "scicode requires --use-bubblewrap-code-sandbox"
            )

    metrics = asyncio.run(_run_final_test(args))
    print("\nOfficial test evaluation complete.")
    for key, value in metrics.items():
        if key != "split":
            print(f"{key}: {value}")


def _official_test_already_recorded(results: dict[str, Any]) -> bool:
    return (
        results.get("official_test_evaluated") is True
        or results.get("official_test_metrics") is not None
        or results.get("test_evaluated_at") is not None
        or results.get("test_metrics") is not None
    )


def _validate_config(config: Any, manifest: dict[str, Any]) -> None:
    stored_snapshot = manifest["config"]["snapshot"]
    actual_snapshot = public_scientific_config(config)
    if actual_snapshot != stored_snapshot:
        raise ManifestValidationError(
            "Config mismatch: supplied config does not match optimization manifest"
        )

    split = manifest["split"]
    expected_ratios = {
        "optimization": config.opt_split_ratio,
        "validation": config.val_split_ratio,
        "test": config.test_split_ratio,
    }
    if split["seed"] != config.seed or split["ratios"] != expected_ratios:
        raise ManifestValidationError(
            "Split seed/ratios do not match the supplied config"
        )
    expected_mode = "source_split" if config.split_source_field else "ratio"
    if split.get("mode") != expected_mode:
        raise ManifestValidationError(
            "Split mode does not match the supplied config"
        )
    if config.split_source_field and (
        split.get("source_field") != config.split_source_field
        or split.get("test_value") != config.split_test_value
        or bool(split.get("dev_reuse")) != bool(config.split_validate_reuse)
    ):
        raise ManifestValidationError(
            "Source-split field/test value/dev-reuse do not match the supplied config"
        )


def _validate_dataset_and_indices(
    data: list[Any],
    manifest: dict[str, Any],
) -> list[int]:
    descriptor = manifest["dataset"]
    if len(data) != descriptor["row_count"]:
        raise ManifestValidationError(
            "Dataset row-count mismatch with optimization manifest"
        )
    return _validate_manifest_indices(manifest)["test"]


def _validate_manifest_indices(
    manifest: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Validate exact split membership without loading any dataset labels."""
    descriptor = manifest["dataset"]
    row_count = descriptor["row_count"]
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count <= 0
    ):
        raise ManifestValidationError(
            "Manifest dataset row_count must be a positive integer"
        )

    raw_indices = manifest["split"]["indices"]
    names = ("optimization", "validation", "test")
    validated: dict[str, list[int]] = {}
    for name in names:
        values = raw_indices[name]
        if not isinstance(values, list) or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= row_count
            for index in values
        ):
            raise ManifestValidationError(
                f"Manifest has invalid {name} split indices"
            )
        if len(values) != len(set(values)):
            raise ManifestValidationError(
                f"Manifest has duplicate {name} split indices"
            )
        validated[name] = values

    dev_reuse = bool(manifest["split"].get("dev_reuse"))
    if dev_reuse:
        # Single-split practice: optimization and validation are the SAME
        # development rows; only the test split is held out.
        if set(validated["optimization"]) != set(validated["validation"]):
            raise ManifestValidationError(
                "Manifest dev_reuse requires optimization == validation rows"
            )
        if set(validated["test"]) & set(validated["optimization"]):
            raise ManifestValidationError(
                "Manifest dev_reuse must not overlap the held-out test split"
            )
        if set(validated["optimization"]) | set(validated["test"]) != set(
            range(row_count)
        ):
            raise ManifestValidationError(
                "Manifest dev_reuse does not cover the dataset exactly"
            )
    else:
        flattened = [
            index
            for name in names
            for index in validated[name]
        ]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            range(row_count)
        ):
            raise ManifestValidationError(
                "Manifest split indices overlap or do not cover the dataset exactly"
            )
    if not validated["test"]:
        raise ManifestValidationError("Manifest held-out test split is empty")
    return validated


def _validate_run_metadata(
    manifest: dict[str, Any],
    benchmark: str,
    allow_local_code_execution: bool,
    allow_zero_hard_reward: bool = False,
    use_bubblewrap_code_sandbox: bool = False,
    scicode_hdf5_path: str | Path | None = None,
    scicode_protocol: str = "first_subproblem",
) -> None:
    expected = _run_metadata(
        benchmark,
        allow_local_code_execution,
        allow_zero_hard_reward,
        use_bubblewrap_code_sandbox=use_bubblewrap_code_sandbox,
        scicode_hdf5_path=scicode_hdf5_path,
        scicode_protocol=scicode_protocol,
    )
    recorded = manifest["run_metadata"]
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ManifestValidationError(
                f"Benchmark/execution mode mismatch for {key}: "
                f"manifest={recorded.get(key)!r}, supplied={value!r}"
            )


def _validate_checkpoint(
    results_dir: Path,
    results: dict[str, Any],
    manifest: dict[str, Any],
):
    recorded = manifest["best_checkpoint"]
    if recorded.get("path") != "checkpoints/best_workflow.yaml":
        raise ManifestValidationError("Manifest checkpoint path is invalid")

    workflow_path = results_dir / "checkpoints" / "best_workflow.yaml"
    if not workflow_path.is_file():
        raise FileNotFoundError(f"Best workflow not found: {workflow_path}")
    workflow = load_workflow(workflow_path)

    meta_path = results_dir / "checkpoints" / "checkpoint_meta.json"
    if not meta_path.is_file():
        raise ManifestValidationError("Checkpoint metadata is missing")
    checkpoint_meta = json.loads(meta_path.read_text())
    if checkpoint_meta.get("round") != recorded["round"]:
        raise ManifestValidationError(
            "Checkpoint round does not match the manifest"
        )
    summary_round = (results.get("checkpoint_summary") or {}).get("best_round")
    if summary_round != recorded["round"]:
        raise ManifestValidationError(
            "Results checkpoint round does not match the manifest"
        )
    return workflow


async def _run_final_test(args: argparse.Namespace) -> dict[str, Any]:
    """Verify all artifacts, then execute the exact test indices once."""
    results_path = Path(args.results)
    ledger_path = heldout_test_ledger_path(results_path)
    with heldout_protocol_lock(ledger_path, exclusive=True):
        return await _run_final_test_protocol_locked(args)


async def _run_final_test_protocol_locked(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Pin large assets while holding the shared protocol's test lock."""
    if args.benchmark == "scicode" and getattr(
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
            return await _run_final_test_with_assets(bound_args)
    return await _run_final_test_with_assets(args)


async def _run_final_test_with_assets(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute while any large sealed benchmark asset remains pinned."""
    results_path = Path(args.results)
    with _claim_official_test(results_path):
        return await _run_final_test_locked(args, results_path)


async def _run_final_test_locked(
    args: argparse.Namespace,
    results_path: Path,
) -> dict[str, Any]:
    """Execute while holding the exclusive one-shot protocol claim."""
    results_data, initial_results_file_sha256 = read_json_snapshot(
        results_path
    )
    if not isinstance(results_data, dict):
        raise ManifestValidationError("results.json must be a JSON object")
    if _official_test_already_recorded(results_data):
        raise RuntimeError(
            "Official held-out test has already been evaluated; refusing to "
            "evaluate it again"
        )
    ledger_path = heldout_test_ledger_path(results_path)
    if ledger_path.exists():
        raise RuntimeError(
            "The manifest-bound held-out split was already claimed by an "
            f"official or comparison test ({ledger_path})"
        )

    manifest = validate_manifest_shape(results_data.get("manifest"))
    config = load_config(args.config)
    _validate_config(config, manifest)
    _validate_run_metadata(
        manifest,
        args.benchmark,
        args.allow_local_code_execution,
        getattr(args, "allow_zero_hard_reward", False),
        use_bubblewrap_code_sandbox=getattr(
            args,
            "use_bubblewrap_code_sandbox",
            False,
        ),
        scicode_hdf5_path=getattr(args, "scicode_hdf5", None),
        scicode_protocol=getattr(
            args,
            "scicode_protocol",
            "first_subproblem",
        ),
    )

    # All checks below are label-free.  The permanent shared claim is created
    # only after they succeed and before the benchmark loader can deserialize
    # any held-out target.
    _validate_manifest_indices(manifest)
    validate_bound_dataset_source(
        args.data,
        manifest,
        # Legacy non-comparison experiments did not record a raw source seal.
        # They remain runnable, but every new optimization records one.
        require_seal=False,
    )
    results_dir = results_path.parent
    workflow = _validate_checkpoint(results_dir, results_data, manifest)
    claim = _claim_official_test_ledger(
        ledger_path,
        results_path=results_path,
        manifest=manifest,
        benchmark=args.benchmark,
        results_file_sha256=initial_results_file_sha256,
    )

    reward, data, operators = _load_benchmark(
        args.benchmark,
        args.data,
        allow_local_code_execution=args.allow_local_code_execution,
        use_bubblewrap_code_sandbox=getattr(
            args,
            "use_bubblewrap_code_sandbox",
            False,
        ),
        scicode_hdf5_path=getattr(args, "scicode_hdf5", None),
        scicode_protocol=getattr(
            args,
            "scicode_protocol",
            "first_subproblem",
        ),
    )
    test_indices = _validate_dataset_and_indices(data, manifest)

    # No optimizer or optimizer API client is constructed for held-out test.
    runner = ExperimentRunner.for_evaluation(
        config=config,
        workflow=workflow,
        reward_evaluator=reward,
        operators=operators,
        output_dir=results_dir / "official_test",
    )
    runner.load_exact_test_data(data, test_indices)
    metrics = await runner.evaluate_test(workflow)

    sealed_hdf5 = getattr(args, "_sealed_scicode_hdf5", None)
    if sealed_hdf5 is not None:
        # Verify before publishing results; the outer context verifies once
        # more after all evaluation code has released the asset.
        sealed_hdf5.verify()

    # Re-read immediately before commit so an independently recorded test or
    # modified manifest cannot be silently overwritten after a long run.
    current_results, current_results_file_sha256 = read_json_snapshot(
        results_path
    )
    del current_results_file_sha256, initial_results_file_sha256
    if not isinstance(current_results, dict):
        raise RuntimeError(
            "results.json changed during held-out evaluation; refusing update"
        )
    if _official_test_already_recorded(current_results):
        raise RuntimeError(
            "Official held-out test was recorded concurrently; refusing to "
            "overwrite it"
        )
    if (
        current_results.get("manifest") != manifest
    ):
        raise RuntimeError(
            "results.json changed during held-out evaluation; refusing update"
        )

    evaluated_at = datetime.now(timezone.utc).isoformat()
    current_results["test_metrics"] = metrics
    current_results["official_test_metrics"] = metrics
    current_results["test_score"] = metrics.get("composite_reward")
    current_results["test_evaluated_at"] = evaluated_at
    current_results["official_test_evaluated"] = True
    atomic_write_json_0600(results_path, current_results)
    atomic_write_json_0600(
        results_dir / "final_test_metrics.json",
        metrics,
    )
    complete_heldout_test(
        ledger_path,
        claim_id=claim["claim_id"],
        completion={
            "completed_at": evaluated_at,
            "result_path": str(results_path.resolve()),
            "result_file_sha256": file_sha256(results_path),
            "metrics_path": str(
                (results_dir / "final_test_metrics.json").resolve()
            ),
            "metrics_file_sha256": file_sha256(
                results_dir / "final_test_metrics.json"
            ),
        },
        expected_claim_fields={
            "access_mode": "awf_official_test",
            "benchmark": args.benchmark,
        },
    )
    logger.info(
        "Official test evaluated checkpoint v%s from validation round %s",
        workflow.version,
        manifest["best_checkpoint"]["round"],
    )
    return metrics


def _claim_official_test_ledger(
    ledger_path: Path,
    *,
    results_path: Path,
    manifest: Mapping[str, Any],
    benchmark: str,
    results_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Consume the shared held-out split for the native AWF test command."""
    binding = {
        "schema_version": 1,
        "access_mode": "awf_official_test",
        "benchmark": benchmark,
        "protocol_results_file_sha256": (
            results_file_sha256
            if results_file_sha256 is not None
            else file_sha256(results_path)
        ),
        "manifest_sha256": json_sha256(manifest),
        "test_indices_sha256": json_sha256(
            list(manifest["split"]["indices"]["test"])
        ),
    }
    claim = {
        **binding,
        "claim_id": json_sha256(binding),
        "status": "claimed",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
    }
    return claim_heldout_test(ledger_path, claim)


@contextmanager
def _claim_official_test(results_path: Path):
    """Prevent two official test processes from evaluating concurrently."""
    lock_path = results_path.with_name(f".{results_path.name}.official-test.lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "An official held-out evaluation is already in progress; "
            f"exclusive lock exists at {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
