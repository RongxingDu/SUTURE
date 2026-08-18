from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from awf.config.schema import SchedulerConfig
from awf.optimizer.candidate_generator import CandidateGenerator
from awf.optimizer.failure_buffer import FailureBuffer
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.protocol.experiment import ExperimentRunner
from awf.reward.base import RewardEvaluator
from awf.scheduler.calibration import FrozenWorkflowGateCalibrator
from awf.scheduler.cascade_scheduler import CascadeGate
from awf.executor.context import ExecutionContext
from awf.trace.schema import ExecutionTrace, TraceStep
from awf.workflow.serializer import load_workflow


class _NoLLM:
    pass


class _ExactReward(RewardEvaluator):
    def hard_reward(self, query, ground_truth, output, trace):
        return float(str(output) == str(ground_truth))

    def process_reward(self, query, ground_truth, output, trace):
        return 0.0


def _math_workflow():
    return load_workflow("experiments/workflows/math/default_workflow.yaml")


def test_all_deterministic_math_blocks_materialize() -> None:
    names = [
        "self_refine",
        "dual_solve_judge",
        "verify_repair",
        "conditional_debate",
        "format_repair",
    ]
    workflow = _math_workflow()
    generator = CandidateGenerator(
        _NoLLM(),
        max_candidates=12,
        max_edit_distance=None,
        allowed_scopes=["prompt", "operator", "block", "multi"],
        allow_cross_block_graph_updates=True,
        deterministic_block_candidates=names,
    )
    raw = generator._deterministic_math_block_candidates(workflow, "solve")
    candidates = generator._parse_candidates(
        json.dumps({"candidates": raw}),
        workflow,
        allowed_anchor_ids={"solve"},
        requested_scope="block",
        limit=12,
    )
    assert len(raw) == len(candidates) == 5
    added = [set(item.modified_workflow.nodes) - set(workflow.nodes) for item in candidates]
    assert {"self_refine"} in added
    assert {"dual_solve", "dual_solve_judge"} in added
    assert {"verify_repair"} in added
    assert {"conditional_debate_gate", "conditional_debate"} in added
    assert {"format_repair"} in added


def test_math_failure_subcluster_uses_domain_and_repair_level() -> None:
    trace = ExecutionTrace(
        trace_id="failure",
        metadata={
            "ground_truth": {"domain": "Number Theory"},
            "math_evaluation": {
                "output_present": True,
                "final_answer_extractable": True,
                "solve_answer_extractable": True,
                "final_answer_correct": False,
                "solve_answer_correct": True,
                "solve_final_consistent": False,
            },
        },
    )
    assert ExperimentRunner._failure_cluster_key(trace) == (
        "math",
        "Number Theory",
        "finalizer_corruption",
        "finalize",
    )


def test_any_repair_accepts_full_run_for_one_cluster_member() -> None:
    optimizer = LLMWorkflowOptimizer.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        hard_success_threshold=1.0,
        allow_failure_full_rerun=True,
        failure_repair_acceptance="any_repair",
    )
    gate = optimizer._failure_suffix_replay_gate(
        {
            "per_query_results": [
                {
                    "trace_id": "a",
                    "candidate_success": True,
                    "candidate_hard": 1.0,
                    "candidate_suffix_replay_used": False,
                },
                {
                    "trace_id": "b",
                    "candidate_success": True,
                    "candidate_hard": 0.0,
                    "candidate_suffix_replay_used": False,
                },
            ]
        },
        {"a", "b"},
        min_coverage=1.0,
    )
    assert gate["passed"] is True
    assert gate["repaired_failure_traces"] == ["a"]
    assert gate["repaired_coverage"] == 0.5


def test_unstable_success_guard_never_becomes_target_failure() -> None:
    target_failure = ExecutionTrace(
        trace_id="cluster-failure",
        workflow_name="math",
        workflow_version="1.0",
        success=False,
        hard_reward=0.0,
    )
    stable_guard = ExecutionTrace(
        trace_id="stable-guard",
        workflow_name="math",
        workflow_version="1.0",
        success=True,
        hard_reward=1.0,
    )
    unstable_guard = ExecutionTrace(
        trace_id="unstable-guard",
        workflow_name="math",
        workflow_version="1.0",
        success=False,
        hard_reward=0.0,
    )

    stable, unstable = ExperimentRunner._partition_replayed_guards(
        [stable_guard, unstable_guard],
        1.0,
    )
    assert [trace.trace_id for trace in stable] == ["stable-guard"]
    assert [trace.trace_id for trace in unstable] == ["unstable-guard"]

    buffer = FailureBuffer(capacity=10)
    buffer.add(target_failure)
    buffer.extend(stable)
    batch = buffer.consume_optimization_round(
        "math",
        "1.0",
        success_fraction=0.0,
        min_success_guards=1,
    )
    assert [trace.trace_id for trace in batch.failures] == ["cluster-failure"]
    assert [trace.trace_id for trace in batch.success_guards] == ["stable-guard"]


def test_frozen_scheduler_calibration_preserves_accuracy_and_saves_cost() -> None:
    workflow = _math_workflow()
    trace = ExecutionTrace(
        trace_id="validation",
        query_text="answer",
        hard_reward=1.0,
        total_prompt_tokens=80,
        total_completion_tokens=20,
        total_latency_seconds=10.0,
        metadata={"ground_truth": "42"},
        steps=[
            TraceStep(
                step_id="start",
                step_index=0,
                node_id="start",
                input_tokens=1,
                output_tokens=1,
                duration_seconds=0.1,
            ),
            TraceStep(
                step_id="solve",
                step_index=1,
                node_id="solve",
                input_tokens=10,
                output_tokens=10,
                duration_seconds=1.0,
            ),
            TraceStep(
                step_id="verify",
                step_index=2,
                node_id="verify",
                state_before={"outputs": {"solve": "42"}},
                metadata={
                    "scheduler_params": {
                        "_awf_gate": {
                            "artifact_node": "solve",
                            "spec_score": 1.0,
                            "lite_score": 1.0,
                            "agreement_score": 1.0,
                            "history_reliability": 0.0,
                        }
                    }
                },
                input_tokens=30,
                output_tokens=30,
                duration_seconds=4.0,
            ),
        ],
    )
    config = SchedulerConfig(
        scheduler_type="cascade",
        calibration_enabled=True,
        calibration_thresholds=[0.5],
        calibration_weight_candidates=[[0.0, 1.0, 0.0, 0.0]],
    )
    result = FrozenWorkflowGateCalibrator(
        config,
        workflow,
        _ExactReward(),
    ).fit([trace])
    assert result.simulated_accuracy == result.baseline_accuracy == 1.0
    assert result.simulated_tokens < result.baseline_tokens
    assert result.simulated_latency_seconds < result.baseline_latency_seconds
    assert result.simulated_early_exits == 1
    deployed = FrozenWorkflowGateCalibrator(
        config,
        workflow,
        _ExactReward(),
    ).apply(result)
    assert deployed.gate_schedule_threshold < deployed.gate_early_exit_threshold


def test_scheduler_calibration_explicitly_disables_infeasible_early_exit() -> None:
    workflow = _math_workflow()
    trace = ExecutionTrace(
        trace_id="regression",
        query_text="answer",
        hard_reward=1.0,
        total_prompt_tokens=80,
        total_completion_tokens=20,
        total_latency_seconds=10.0,
        metadata={"ground_truth": "42"},
        steps=[
            TraceStep(
                step_id="verify",
                step_index=0,
                node_id="verify",
                state_before={"outputs": {"solve": "wrong"}},
                metadata={
                    "scheduler_params": {
                        "_awf_gate": {
                            "artifact_node": "solve",
                            "spec_score": 1.0,
                            "lite_score": 1.0,
                            "agreement_score": 1.0,
                            "history_reliability": 0.0,
                        }
                    }
                },
                input_tokens=30,
                output_tokens=30,
                duration_seconds=4.0,
            ),
        ],
    )
    config = SchedulerConfig(
        scheduler_type="cascade",
        calibration_enabled=True,
        calibration_thresholds=[1.0],
        calibration_weight_candidates=[[0.0, 1.0, 0.0, 0.0]],
        calibration_max_hard_regression=0.0,
    )
    calibrator = FrozenWorkflowGateCalibrator(config, workflow, _ExactReward())

    result = calibrator.fit([trace])
    deployed = calibrator.apply(result)

    assert result.feasible_candidates == 0
    assert result.early_exit_enabled is False
    assert result.simulated_accuracy == result.baseline_accuracy == 1.0
    assert result.simulated_early_exits == 0
    assert deployed.gate_enabled is False
    assert deployed.early_exit_enabled is False
    assert deployed.gate_direct_early_exit is False


def _scheduler_trace(trace_id: str, hard: float, tokens: int = 10):
    return ExecutionTrace(
        trace_id=trace_id,
        query_text=trace_id,
        hard_reward=hard,
        success=True,
        total_prompt_tokens=tokens,
        total_completion_tokens=0,
        total_latency_seconds=1.0,
    )


async def test_scheduler_confirmation_repeats_only_initial_errors() -> None:
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            hard_success_threshold=1.0,
            failure_confirmation_repeats=2,
            failure_confirmation_min_failures=2,
        )
    )
    runner.val_data = [("correct", 1), ("wrong", 1)]
    runner._execute_dataset = AsyncMock(
        side_effect=[
            (
                [_scheduler_trace("wrong-r1", 1.0)],
                {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "total_tokens": 10,
                    "latency_seconds": 1.0,
                },
            ),
            (
                [_scheduler_trace("wrong-r2", 0.0)],
                {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "total_tokens": 10,
                    "latency_seconds": 1.0,
                },
            ),
        ]
    )
    initial = [
        _scheduler_trace("correct", 1.0),
        _scheduler_trace("wrong", 0.0),
    ]

    stable, report = await runner._confirm_scheduler_error_traces(
        _math_workflow(),
        initial,
        split_prefix="confirm",
    )

    assert [trace.hard_reward for trace in stable] == [1.0, 0.0]
    assert report["initial_failures"] == 1
    assert report["additional_executions"] == 2
    assert runner._execute_dataset.await_count == 2
    for call in runner._execute_dataset.await_args_list:
        assert call.args[1] == [("wrong", 1)]


def test_cascade_gate_only_schedules_the_workflow_update_node() -> None:
    workflow = _math_workflow()
    config = SchedulerConfig(
        scheduler_type="cascade",
        gate_node_allowlist=["verify"],
    )
    gate = CascadeGate(config)
    context = ExecutionContext("question", workflow.name)
    context.current_node_id = "solve"
    context.history = ["analyze"]
    context.outputs["analyze"] = "A complete-looking answer: 42."

    outside = gate.evaluate(workflow, context)

    assert outside.route == "continue"
    assert outside.reason == "outside_intervention_node_allowlist"
