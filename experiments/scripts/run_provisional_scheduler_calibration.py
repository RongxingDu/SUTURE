#!/usr/bin/env python3
"""Calibrate LAS on an accepted provisional inner-workflow update.

This research helper reconstructs the actually selected workflow patch from a
completed optimization result, freezes that workflow, and runs only the outer
scheduler calibration/audit phase. It is intentionally separate from the
workflow optimizer so the two action spaces cannot update jointly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import load_config
from awf.optimizer.candidate_generator import CandidateGenerator
from awf.protocol.experiment import ExperimentRunner
from awf.scheduler.calibration import FrozenWorkflowGateCalibrator
from awf.scheduler.cascade_scheduler import CascadeScheduler
from awf.trace.schema import ExecutionTrace
from awf.workflow.serializer import dump_workflow, load_workflow
from experiments.scripts.run_optimization import _load_benchmark


class _NoLLM:
    """Candidate materialization below is deterministic and makes no call."""


def _selected_update(results: dict) -> tuple[dict, dict]:
    selected: list[tuple[dict, dict]] = []
    for round_row in results.get("rounds", []):
        for update in round_row.get("deferred_updates", []):
            if not update.get("accepted", False):
                continue
            summary = update.get("summary", {})
            for candidate in summary.get("candidate_evaluations", []):
                if candidate.get("final_selected") is True or candidate.get(
                    "status"
                ) == "selected":
                    selected.append((update, candidate))
    if not selected:
        raise ValueError("result contains no accepted selected workflow update")
    return selected[-1]


def reconstruct_workflow(results_path: Path, initial_workflow_path: Path):
    results = json.loads(results_path.read_text(encoding="utf-8"))
    update, selected = _selected_update(results)
    workflow = load_workflow(initial_workflow_path)
    generator = CandidateGenerator(
        _NoLLM(),
        max_candidates=1,
        max_edit_distance=None,
        allowed_scopes=["prompt", "operator", "block", "multi"],
        aggressive_generation=True,
        allow_cross_block_graph_updates=True,
    )
    raw = {
        "candidates": [
            {
                "scope": selected["scope"],
                "node_id": selected["node_id"],
                "description": selected.get("description", "selected update"),
                "changes": selected["changes"],
            }
        ]
    }
    candidates = generator._parse_candidates(
        json.dumps(raw),
        workflow,
        allowed_anchor_ids={str(selected["node_id"])},
        requested_scope=str(selected["scope"]),
        limit=1,
    )
    if len(candidates) != 1 or candidates[0].modified_workflow is None:
        raise ValueError(
            "selected patch can no longer be materialized under current rules: "
            f"{generator.last_generation_report}"
        )
    provisional = candidates[0].modified_workflow
    provisional.version = str(update.get("workflow_version_after", "1.1"))
    provisional.metadata["provisional_update_source"] = {
        "results": str(results_path.resolve()),
        "update_index": update.get("update_index"),
        "cluster_key": update.get("cluster_key"),
        "candidate_scope": selected.get("scope"),
        "candidate_node_id": selected.get("node_id"),
        "candidate_description": selected.get("description"),
    }
    return provisional


def _load_traces(path: Path) -> list[ExecutionTrace]:
    return [
        ExecutionTrace.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stabilize_recorded_baseline(
    trace_dir: Path,
    *,
    threshold: float,
    min_failures: int,
) -> list[ExecutionTrace]:
    initial = _load_traces(trace_dir / "scheduler_calibration_baseline.jsonl")
    repeat_one = _load_traces(
        trace_dir / "scheduler_calibration_baseline_error_confirmation_1.jsonl"
    )
    repeat_two = _load_traces(
        trace_dir / "scheduler_calibration_baseline_error_confirmation_2.jsonl"
    )
    failure_positions = [
        index
        for index, trace in enumerate(initial)
        if float(trace.hard_reward or 0.0) < threshold
    ]
    if len(repeat_one) != len(failure_positions) or len(repeat_two) != len(
        failure_positions
    ):
        raise ValueError("recorded baseline confirmation traces are incomplete")
    stable = list(initial)
    for position, first, second in zip(
        failure_positions,
        repeat_one,
        repeat_two,
    ):
        observations = [initial[position], first, second]
        failed = [
            trace
            for trace in observations
            if float(trace.hard_reward or 0.0) < threshold
        ]
        successful = [
            trace
            for trace in observations
            if float(trace.hard_reward or 0.0) >= threshold
        ]
        stable[position] = (
            failed[-1]
            if len(failed) >= min_failures or not successful
            else successful[-1]
        )
    return stable


async def _run_reused_baseline_audit(
    runner: ExperimentRunner,
    provisional,
    reuse_dir: Path,
) -> dict:
    old_payload = json.loads(
        (reuse_dir / "scheduler_calibration_results.json").read_text(
            encoding="utf-8"
        )
    )["scheduler_calibration"]
    stable_baseline = _stabilize_recorded_baseline(
        reuse_dir / "traces",
        threshold=float(runner.config.optimizer.hard_success_threshold),
        min_failures=int(
            runner.config.optimizer.failure_confirmation_min_failures
        ),
    )
    stable_baseline_metrics = runner._scheduler_trace_metrics(stable_baseline)
    calibrator = FrozenWorkflowGateCalibrator(
        runner.config.scheduler,
        provisional,
        runner.reward_evaluator,
    )
    calibration = calibrator.fit(stable_baseline)
    candidate_config = calibrator.apply(calibration)
    runner.scheduler = CascadeScheduler(candidate_config)
    audit_traces, audit_metrics = await runner._execute_dataset(
        provisional,
        runner.val_data,
        split_name="scheduler_calibration_audit",
    )
    stable_audit, audit_confirmation = (
        await runner._confirm_scheduler_error_traces(
            provisional,
            audit_traces,
            split_prefix="scheduler_calibration_audit_error_confirmation",
        )
    )
    stable_audit_metrics = runner._scheduler_trace_metrics(stable_audit)
    hard_delta = (
        stable_audit_metrics["hard_reward"]
        - stable_baseline_metrics["hard_reward"]
    )
    token_delta = (
        stable_audit_metrics["total_tokens"]
        - stable_baseline_metrics["total_tokens"]
    )
    latency_delta = (
        stable_audit_metrics["latency_seconds"]
        - stable_baseline_metrics["latency_seconds"]
    )
    hard_safe = hard_delta >= -(
        runner.config.scheduler.calibration_max_hard_regression + 1e-12
    )
    efficiency_safe = (
        token_delta <= 0
        and latency_delta <= 1e-12
        and (token_delta < 0 or latency_delta < -1e-12)
    )
    deployed = hard_safe and efficiency_safe
    deployed_config = candidate_config.model_copy(deep=True)
    if not deployed:
        deployed_config.gate_enabled = False
        deployed_config.early_exit_enabled = False
        deployed_config.gate_direct_early_exit = False
        deployed_config.allow_deviation = False
    runner.scheduler = CascadeScheduler(deployed_config)
    provisional.metadata["scheduler_calibration"] = deployed_config.model_dump(
        mode="json"
    )
    return {
        **calibration.as_dict(),
        "workflow_frozen": True,
        "frozen_workflow_version": provisional.version,
        "baseline_reused": True,
        "baseline_reuse_source": str(reuse_dir.resolve()),
        "baseline_validation_metrics": old_payload[
            "baseline_validation_metrics"
        ],
        "stabilized_baseline_metrics": stable_baseline_metrics,
        "baseline_error_confirmation": old_payload[
            "baseline_error_confirmation"
        ],
        "audit_validation_metrics": audit_metrics,
        "stabilized_audit_metrics": stable_audit_metrics,
        "audit_error_confirmation": audit_confirmation,
        "actual_hard_reward_delta": hard_delta,
        "actual_token_delta": token_delta,
        "actual_latency_delta_seconds": latency_delta,
        "hard_non_regression_passed": hard_safe,
        "efficiency_improvement_passed": efficiency_safe,
        "deployed": deployed,
        "deployment_reason": (
            "validation_non_regression_and_efficiency_improvement"
            if deployed
            else (
                "actual_validation_hard_regression"
                if not hard_safe
                else "actual_validation_efficiency_not_improved"
            )
        ),
        "candidate_scheduler_config": candidate_config.model_dump(mode="json"),
        "deployed_scheduler_config": deployed_config.model_dump(mode="json"),
    }


async def _run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    config.name = args.name
    config.output_dir = str(args.output_dir)
    config.scheduler.scheduler_type = "cascade"
    config.scheduler.calibration_enabled = True
    config.scheduler.gate_formula = "las"
    config.scheduler.allow_deviation = True
    config.scheduler.gate_enabled = True
    config.scheduler.early_exit_enabled = True
    config.scheduler.gate_direct_early_exit = True
    config.scheduler.gate_node_allowlist = ["verify_repair"]
    config.scheduler.calibration_max_hard_regression = 0.0
    # A denser, still deterministic grid is useful for the small MATH study.
    config.scheduler.calibration_thresholds = [
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
    ]
    config.scheduler.calibration_weight_candidates = [
        [0.10, 0.70, 0.10, 0.10],
        [0.15, 0.60, 0.15, 0.10],
        [0.20, 0.55, 0.15, 0.10],
        [0.10, 0.55, 0.25, 0.10],
        [0.20, 0.65, 0.10, 0.05],
    ]
    # This script invokes the outer phase directly rather than pretending it
    # is another optimizer round.
    config.scheduler_calibration_round = None
    provisional = reconstruct_workflow(args.results, args.workflow)
    reward, data, operators = _load_benchmark("math", str(args.data))
    runner = ExperimentRunner(
        config=config,
        workflow=provisional,
        reward_evaluator=reward,
        operators=operators,
        run_metadata={"benchmark": "math", "phase": "scheduler_calibration"},
    )
    runner.load_data(data, dataset_source_path=args.data)
    runner._current_round = "scheduler_calibration"
    workflow_path = runner.output_dir / "provisional_workflow.yaml"
    dump_workflow(provisional, workflow_path)
    if args.reuse_baseline_from is not None:
        calibration_payload = await _run_reused_baseline_audit(
            runner,
            provisional,
            args.reuse_baseline_from,
        )
    else:
        summary = await runner._run_scheduler_calibration_round(3)
        calibration_payload = summary["scheduler_calibration"]
    dump_workflow(provisional, workflow_path)
    payload = {
        "provisional_workflow": str(workflow_path),
        "validation_indices": runner.split_indices.get("validation", []),
        "scheduler_calibration": calibration_payload,
    }
    output_path = runner.output_dir / "scheduler_calibration_results.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/aliyun_qwen35"),
    )
    parser.add_argument(
        "--name",
        default="math_failure_mined_provisional_scheduler_calibration",
    )
    parser.add_argument(
        "--reuse-baseline-from",
        type=Path,
        help=(
            "Reuse a completed passive baseline and its error-only "
            "confirmations; only the new node-scoped scheduler audit runs."
        ),
    )
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    print(json.dumps(payload["scheduler_calibration"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
