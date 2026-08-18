#!/usr/bin/env python3
"""Paired validation-only evaluation of workflows and schedulers.

This script evaluates the Cartesian product

    {initial workflow, validation-selected workflow}
        x {fixed scheduler, LLM scheduler}

on the exact validation indices recorded by an optimization manifest.  It
never loads the manifest test indices into an evaluation runner and refuses
artifacts that already contain official-test results, preventing held-out
outcomes from influencing scheduler analysis.

Usage:
    python experiments/scripts/run_scheduler_ablation.py \
      --results experiments/results/.../results.json \
      --fixed-config experiments/configs/math_deepseek_pilot.yaml \
      --llm-config experiments/configs/math_deepseek_scheduler_ablation.yaml \
      --benchmark math \
      --data /path/to/math.jsonl \
      --initial-workflow experiments/workflows/math/default_workflow.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import load_config
from awf.config.schema import ExperimentConfig
from awf.protocol.experiment import ExperimentRunner
from awf.protocol.manifest import (
    ManifestValidationError,
    atomic_write_json_0600,
    file_sha256,
    json_sha256,
    public_scientific_config,
    validate_manifest_shape,
    workflow_sha256,
)
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.serializer import load_workflow
from experiments.scripts.run_optimization import _load_benchmark
from experiments.scripts.run_test import (
    _official_test_already_recorded,
    _validate_checkpoint,
    _validate_config,
    _validate_dataset_and_indices,
    _validate_run_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SCHEDULER_TYPES = ("fixed", "llm")
_WORKFLOW_TYPES = ("initial", "optimized")
_DELTA_METRICS = (
    "hard_reward",
    "process_reward",
    "composite_reward",
    "runtime_utility",
    "total_tokens",
    "llm_call_count",
    "latency_seconds",
    "known_partial_api_cost_usd",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate initial/optimized workflows with fixed/LLM schedulers "
            "on the manifest validation split only"
        )
    )
    parser.add_argument(
        "--results",
        "-r",
        required=True,
        help="Manifest-bearing results.json from optimization",
    )
    parser.add_argument(
        "--fixed-config",
        required=True,
        help="The exact fixed-scheduler config used for optimization",
    )
    parser.add_argument(
        "--llm-config",
        required=True,
        help=(
            "A scheduler-only ablation config matching the fixed config in "
            "all other scientific fields"
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=["code_gen", "math", "agent"],
        required=True,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="The exact ordered full benchmark JSONL used for optimization",
    )
    parser.add_argument(
        "--initial-workflow",
        required=True,
        help="The exact initial workflow bound by the optimization manifest",
    )
    parser.add_argument(
        "--allow-local-code-execution",
        action="store_true",
        help="Match trusted restricted-local execution used during optimization",
    )
    parser.add_argument(
        "--use-bubblewrap-code-sandbox",
        action="store_true",
        help="Match the Linux bubblewrap sandbox used during optimization",
    )
    parser.add_argument(
        "--allow-zero-hard-reward",
        action="store_true",
        help="Match a code experiment that explicitly disabled execution",
    )
    return parser


def _validate_execution_mode_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    modes = (
        bool(args.allow_zero_hard_reward),
        bool(args.allow_local_code_execution),
        bool(args.use_bubblewrap_code_sandbox),
    )
    if args.benchmark != "code_gen" and any(modes):
        parser.error(
            "code execution mode flags apply only to the code_gen benchmark"
        )
    if sum(modes) > 1:
        parser.error(
            "--allow-zero-hard-reward, --allow-local-code-execution, and "
            "--use-bubblewrap-code-sandbox are mutually exclusive"
        )
    if args.benchmark == "code_gen" and not any(modes):
        parser.error(
            "code_gen requires the same explicit execution mode used during "
            "optimization; prefer --use-bubblewrap-code-sandbox"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_execution_mode_args(parser, args)
    artifact = asyncio.run(_run_scheduler_ablation(args))
    output_path = Path(args.results).parent / "scheduler_ablation.json"

    print("\nScheduler ablation complete (validation split only).")
    print(f"output: {output_path}")
    print(f"validation_examples: {artifact['split']['validation_count']}")
    for workflow_name in _WORKFLOW_TYPES:
        for scheduler_name in _SCHEDULER_TYPES:
            metrics = artifact["cells"][workflow_name][scheduler_name][
                "metrics"
            ]
            print(
                f"{workflow_name}/{scheduler_name}: "
                f"reward={metrics['composite_reward']:.6f}, "
                f"tokens={metrics['total_tokens']}, "
                f"latency={metrics['latency_seconds']:.6f}s"
            )


def _normalized_ablation_snapshot(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Remove only fields intentionally varied by this paired ablation."""
    snapshot = public_scientific_config(config)
    normalized = json.loads(json.dumps(snapshot))
    normalized.pop("name", None)
    scheduler = normalized.get("scheduler", {})
    scheduler.pop("scheduler_type", None)
    scheduler.pop("allow_deviation", None)
    return normalized


def _validate_ablation_configs(
    fixed_config: ExperimentConfig,
    llm_config: ExperimentConfig,
    manifest: dict[str, Any],
) -> None:
    """Bind the baseline config and constrain the scheduler-only variant."""
    _validate_config(fixed_config, manifest)
    if fixed_config.scheduler.scheduler_type != "fixed":
        raise ManifestValidationError(
            "The optimization/baseline config must use scheduler_type='fixed'"
        )
    if llm_config.scheduler.scheduler_type not in {"graph"}:
        raise ManifestValidationError(
            "The scheduler ablation config must use scheduler_type='graph'"
        )
    if not fixed_config.executor.trace_enabled:
        raise ManifestValidationError(
            "Scheduler ablation requires executor.trace_enabled=true so each "
            "paired cell has an auditable trace artifact"
        )
    if (
        _normalized_ablation_snapshot(fixed_config)
        != _normalized_ablation_snapshot(llm_config)
    ):
        raise ManifestValidationError(
            "LLM scheduler config mismatch: only experiment name, "
            "scheduler_type, and allow_deviation may differ from the "
            "manifest-bound fixed config"
        )


def _validate_validation_indices(
    data: list[Any],
    manifest: dict[str, Any],
) -> list[int]:
    """Validate the full partition, then return validation membership only."""
    test_indices = _validate_dataset_and_indices(data, manifest)
    validation_indices = manifest["split"]["indices"]["validation"]
    if not validation_indices:
        raise ManifestValidationError("Manifest validation split is empty")
    if set(validation_indices).intersection(test_indices):
        raise ManifestValidationError(
            "Manifest validation indices overlap the held-out test split"
        )
    return list(validation_indices)


def _validate_initial_workflow(
    workflow_path: str | Path,
    manifest: dict[str, Any],
) -> WorkflowTemplate:
    path = Path(workflow_path)
    if not path.is_file():
        raise FileNotFoundError(f"Initial workflow not found: {path}")
    return load_workflow(path)


def _usage_by_execution_role(
    usage_delta: dict[str, Any],
) -> dict[str, Any]:
    """Expose backend counters under experiment-facing role names."""
    return {
        "workflow": usage_delta["workflow_client"],
        "scheduler": usage_delta["scheduler_client"],
        "total": usage_delta["total"],
    }


async def _evaluate_cell(
    *,
    config: ExperimentConfig,
    workflow: WorkflowTemplate,
    reward_evaluator: Any,
    operators: dict[str, Any],
    validation_data: list[Any],
    output_dir: Path,
    workflow_name: str,
    scheduler_name: str,
) -> dict[str, Any]:
    """Execute one grid cell with an isolated evaluation-only runner."""
    runner = ExperimentRunner.for_evaluation(
        config=config,
        workflow=workflow,
        reward_evaluator=reward_evaluator,
        operators=operators,
        output_dir=output_dir,
    )
    runner._current_round = (
        f"scheduler_ablation:{workflow_name}:{scheduler_name}"
    )
    usage_before = runner._backend_usage_snapshot()
    wall_start = time.perf_counter()
    metrics = await runner._evaluate_dataset(
        workflow,
        validation_data,
        split_name="validation",
    )
    wall_clock_seconds = time.perf_counter() - wall_start
    usage_delta = runner._backend_usage_delta(
        usage_before,
        runner._backend_usage_snapshot(),
    )

    trace_path = output_dir / "traces" / "validation.jsonl"
    return {
        "workflow": workflow_name,
        "scheduler": scheduler_name,
        "workflow_sha256": workflow_sha256(workflow),
        "metrics": metrics,
        "backend_usage": _usage_by_execution_role(usage_delta),
        "latency": {
            "wall_clock_seconds": wall_clock_seconds,
            "trace_seconds": metrics["latency_seconds"],
            "workflow_llm_seconds": metrics[
                "workflow_llm_latency_seconds"
            ],
            "scheduler_llm_seconds": metrics[
                "scheduler_llm_latency_seconds"
            ],
        },
        "execution": {
            "action_count": metrics["action_count"],
            "action_counts": metrics["action_counts"],
            "decision_node_count": metrics["decision_node_count"],
            "decision_node_counts": metrics["decision_node_counts"],
            "node_execution_count": metrics["node_execution_count"],
            "executed_node_counts": metrics["executed_node_counts"],
            "trace_artifact": (
                str(trace_path.relative_to(output_dir.parent.parent))
                if trace_path.is_file()
                else None
            ),
        },
    }


def _metric_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, int | float | None]:
    """Return ``right - left`` for stable headline metrics."""
    deltas: dict[str, int | float | None] = {}
    for key in _DELTA_METRICS:
        left_value = left.get(key)
        right_value = right.get(key)
        if (
            isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
        ):
            deltas[key] = right_value - left_value
        else:
            deltas[key] = None
    return deltas


def _build_comparisons(
    cells: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compute paired effects without selecting or tuning on test outcomes."""
    return {
        "scheduler_effect": {
            workflow_name: _metric_delta(
                cells[workflow_name]["fixed"]["metrics"],
                cells[workflow_name]["llm"]["metrics"],
            )
            for workflow_name in _WORKFLOW_TYPES
        },
        "workflow_effect": {
            scheduler_name: _metric_delta(
                cells["initial"][scheduler_name]["metrics"],
                cells["optimized"][scheduler_name]["metrics"],
            )
            for scheduler_name in _SCHEDULER_TYPES
        },
    }


async def _run_scheduler_ablation(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate all artifacts, then execute the paired validation grid."""
    results_path = Path(args.results)
    if not results_path.is_file():
        raise FileNotFoundError(f"Optimization results not found: {results_path}")
    results_data = json.loads(results_path.read_text())
    if _official_test_already_recorded(results_data):
        raise RuntimeError(
            "Refusing scheduler ablation after official held-out test results "
            "were recorded; run validation-only analysis before official test"
        )

    manifest = validate_manifest_shape(results_data.get("manifest"))
    fixed_config = load_config(args.fixed_config)
    llm_config = load_config(args.llm_config)
    _validate_ablation_configs(fixed_config, llm_config, manifest)
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
    )

    results_dir = results_path.parent
    initial_workflow = _validate_initial_workflow(
        args.initial_workflow,
        manifest,
    )
    optimized_workflow = _validate_checkpoint(
        results_dir,
        results_data,
        manifest,
    )
    reward_evaluator, data, operators = _load_benchmark(
        args.benchmark,
        args.data,
        allow_local_code_execution=args.allow_local_code_execution,
        use_bubblewrap_code_sandbox=getattr(
            args,
            "use_bubblewrap_code_sandbox",
            False,
        ),
    )
    validation_indices = _validate_validation_indices(data, manifest)
    test_indices = set(manifest["split"]["indices"]["test"])
    if any(index in test_indices for index in validation_indices):
        # Keep this check adjacent to materialization as defense in depth.
        raise ManifestValidationError(
            "Refusing to materialize validation data with held-out overlap"
        )
    validation_data = [data[index] for index in validation_indices]

    workflows = {
        "initial": initial_workflow,
        "optimized": optimized_workflow,
    }
    configs = {
        "fixed": fixed_config,
        "llm": llm_config,
    }
    cells: dict[str, dict[str, dict[str, Any]]] = {
        workflow_name: {} for workflow_name in _WORKFLOW_TYPES
    }
    ablation_dir = results_dir / "scheduler_ablation"
    for workflow_name in _WORKFLOW_TYPES:
        for scheduler_name in _SCHEDULER_TYPES:
            logger.info(
                "Evaluating validation cell workflow=%s scheduler=%s",
                workflow_name,
                scheduler_name,
            )
            cell_output_dir = (
                ablation_dir / f"{workflow_name}__{scheduler_name}"
            )
            cells[workflow_name][scheduler_name] = await _evaluate_cell(
                config=configs[scheduler_name],
                workflow=workflows[workflow_name],
                reward_evaluator=reward_evaluator,
                operators=operators,
                validation_data=validation_data,
                output_dir=cell_output_dir,
                workflow_name=workflow_name,
                scheduler_name=scheduler_name,
            )

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "paired_validation_only_scheduler_ablation",
        "source": {
            "results_path": str(results_path),
            "results_file_sha256": file_sha256(results_path),
            "manifest_sha256": json_sha256(manifest),
            "dataset_ordered_fingerprint_sha256": manifest["dataset"][
                "ordered_fingerprint_sha256"
            ],
            "initial_workflow_sha256": manifest["initial_workflow"]["sha256"],
            "optimized_workflow_sha256": manifest["best_checkpoint"][
                "workflow_sha256"
            ],
            "fixed_config_sha256": manifest["config"]["sha256"],
            "llm_ablation_config_sha256": json_sha256(
                public_scientific_config(llm_config)
            ),
        },
        "split": {
            "name": "validation",
            "validation_count": len(validation_indices),
            "validation_indices": validation_indices,
            "validation_indices_sha256": json_sha256(validation_indices),
            "validation_test_overlap_count": 0,
            "heldout_test_evaluated": False,
        },
        "cells": cells,
        "comparisons": _build_comparisons(cells),
    }
    atomic_write_json_0600(
        results_dir / "scheduler_ablation.json",
        artifact,
    )
    return artifact


if __name__ == "__main__":
    main()
