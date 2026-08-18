"""Regression tests for incumbent promotion and stability gates."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from awf.config.schema import ExperimentConfig
from awf.protocol.checkpoint import CheckpointManager
from awf.protocol.experiment import ExperimentRunner
from awf.reward.base import RewardEvaluator
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _emit(value: str) -> str:
    return value


def _workflow(output: str, version: str = "1.0") -> WorkflowTemplate:
    return WorkflowTemplate(
        name="stability-gate-workflow",
        version=version,
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "answer": Node(
                node_id="answer",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name="emit",
                    tool_args={"value": output},
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "answer"), ("answer", "end")],
    )


class OutputReward(RewardEvaluator):
    def __init__(
        self,
        hard_by_output: dict[str, float],
        process_by_output: dict[str, float],
    ):
        self.hard_by_output = hard_by_output
        self.process_by_output = process_by_output
        self.observations: list[tuple[str, str, str]] = []

    def hard_reward(self, query, ground_truth, output, trace):
        self.observations.append(
            (query, str(trace.metadata["split"]), str(output))
        )
        return self.hard_by_output[str(output)]

    def process_reward(self, query, ground_truth, output, trace):
        return self.process_by_output[str(output)]


def _runner(
    tmp_path,
    reward: RewardEvaluator,
    *,
    max_rounds: int = 1,
    validation_min_delta: float = 0.0,
    hard_tolerance: float = 0.0,
    confirm_on_opt: bool = False,
    confirm_repeats: int = 1,
    early_stopping_patience: int | None = None,
) -> ExperimentRunner:
    config = ExperimentConfig(
        name="stability",
        output_dir=str(tmp_path),
        seed=31,
        scheduler={
            "scheduler_type": "fixed",
            "llm": {"api_key": "test-only-key"},
        },
        optimizer={
            "max_rounds": max_rounds,
            "lambda_cost": 0.0,
            "rho_omega": 0.0,
            "llm": {"api_key": "test-only-key"},
        },
        reward={"alpha_process": 1.0},
        executor={"trace_enabled": False},
        validation_min_delta=validation_min_delta,
        validation_hard_regression_tolerance=hard_tolerance,
        confirm_on_opt=confirm_on_opt,
        confirm_repeats=confirm_repeats,
        early_stopping_patience=early_stopping_patience,
    )
    runner = ExperimentRunner(
        config=config,
        workflow=_workflow("baseline"),
        reward_evaluator=reward,
        operators={"emit": _emit},
        run_metadata={
            "benchmark": "math",
            "code_execution_mode": "not_applicable",
        },
    )
    runner.load_data(
        [(f"query-{index}", f"truth-{index}") for index in range(10)]
    )
    return runner


def test_confirmation_config_defaults_and_enabled_repeat_validation():
    config = ExperimentConfig()
    assert config.validation_min_delta == 0.0
    assert config.validation_hard_regression_tolerance == 0.0
    assert config.confirm_on_opt is False
    assert config.confirm_repeats == 1

    # Zero repeats is harmless when confirmation is disabled, but must fail
    # closed as soon as the confirmation protocol is enabled.
    assert ExperimentConfig(confirm_repeats=0).confirm_repeats == 0
    with pytest.raises(ValidationError, match="confirm_repeats"):
        ExperimentConfig(confirm_on_opt=True, confirm_repeats=0)


def test_checkpoint_min_delta_rejects_tiny_or_equal_improvements(tmp_path):
    workflow = _workflow("baseline")
    manager = CheckpointManager(tmp_path, min_delta=0.01)

    assert manager.update(workflow, score=0.5, round_num=0)
    assert not manager.update(workflow, score=0.505, round_num=1)
    assert not manager.update(workflow, score=0.51, round_num=2)
    assert manager.update(workflow, score=0.511, round_num=3)

    summary = manager.get_summary()
    assert summary["min_delta"] == 0.01
    assert summary["best_round"] == 3
    assert [entry["accepted"] for entry in summary["history"]] == [
        True,
        False,
        False,
        True,
    ]

    with pytest.raises(ValueError, match="non-negative"):
        CheckpointManager(tmp_path / "bad", min_delta=-0.1)


@pytest.mark.asyncio
async def test_optimizer_rejection_reuses_incumbent_validation_and_never_test(
    tmp_path,
):
    reward = OutputReward({"baseline": 1.0}, {"baseline": 0.0})
    runner = _runner(tmp_path, reward, max_rounds=2)
    heldout_queries = {query for query, _ in runner.test_data}

    async def reject(workflow, *args, **kwargs):
        return workflow, {"accepted": False, "gain": 0.0}

    runner.optimizer.optimize_round = reject
    results = await runner.run()
    split_counts = Counter(split for _, split, _ in reward.observations)

    assert split_counts["validation_baseline"] == len(runner.val_data)
    assert split_counts["validation"] == 0
    assert split_counts["optimization"] == 2 * len(runner.opt_data)
    assert heldout_queries.isdisjoint(
        query for query, _, _ in reward.observations
    )
    assert results["checkpoint_summary"]["num_evaluations"] == 1
    for summary in results["rounds"]:
        assert summary["counterfactual_accepted"] is False
        assert summary["accepted"] is False
        assert summary["validation_candidate_evaluated"] is False
        assert summary["validation_reused"] is True
        assert (
            summary["validation_reuse_reason"]
            == "counterfactual_rejected"
        )


@pytest.mark.asyncio
async def test_validation_hard_gate_rejects_and_keeps_incumbent_next_round(
    tmp_path,
):
    reward = OutputReward(
        {"baseline": 1.0, "candidate": 0.0},
        {"baseline": 0.0, "candidate": 2.0},
    )
    runner = _runner(tmp_path, reward, max_rounds=2)
    candidate = _workflow("candidate", version="1.1")
    optimizer_inputs: list[str] = []

    async def accept(workflow, *args, **kwargs):
        optimizer_inputs.append(workflow.version)
        return candidate.model_copy(deep=True), {
            "accepted": True,
            "gain": 1.0,
        }

    runner.optimizer.optimize_round = accept
    results = await runner.run()

    assert optimizer_inputs == ["1.0", "1.0"]
    assert runner.workflow.version == "1.0"
    assert results["checkpoint_summary"]["best_round"] == 0
    for summary in results["rounds"]:
        assert summary["counterfactual_accepted"] is True
        assert summary["validation_candidate_evaluated"] is True
        assert summary["validation_gate_passed"] is False
        assert summary["validation_gate"][
            "hard_non_regression_passed"
        ] is False
        assert summary["validation_gate"]["runtime_utility_passed"] is False
        assert summary["accepted"] is False
        assert summary["incumbent_workflow_version_after"] == "1.0"


@pytest.mark.asyncio
async def test_validation_min_delta_rejects_tiny_utility_gain(tmp_path):
    reward = OutputReward(
        {"baseline": 1.0, "candidate": 1.0},
        {"baseline": 0.0, "candidate": 0.005},
    )
    runner = _runner(
        tmp_path,
        reward,
        validation_min_delta=0.01,
    )
    candidate = _workflow("candidate", version="1.1")

    async def accept(*args, **kwargs):
        return candidate, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = await runner.run()
    summary = results["rounds"][0]

    assert summary["counterfactual_accepted"] is True
    assert summary["validation_gate"]["hard_non_regression_passed"] is True
    # Process reward is not part of utility, so equal hard reward means zero
    # utility effect even though the process reward improved.
    assert summary["validation_gate"]["runtime_utility_effect"] == (
        pytest.approx(0.0)
    )
    assert summary["validation_gate"]["runtime_utility_passed"] is False
    assert summary["validation_gate_passed"] is False
    assert summary["accepted"] is False
    assert results["checkpoint_summary"]["best_round"] == 0


@pytest.mark.asyncio
async def test_early_stopping_uses_the_same_checkpoint_promotion_gate(tmp_path):
    reward = OutputReward(
        {"baseline": 1.0, "candidate": 1.0},
        {"baseline": 0.0, "candidate": 0.05},
    )
    runner = _runner(
        tmp_path,
        reward,
        max_rounds=3,
        validation_min_delta=0.1,
        early_stopping_patience=1,
    )
    candidate = _workflow("candidate", version="1.1")

    async def accept(*args, **kwargs):
        return candidate, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = await runner.run()

    assert len(results["rounds"]) == 1
    assert results["rounds"][0]["validation_gate_passed"] is False
    assert results["rounds"][0]["is_best"] is False
    assert results["stopped_early"] is True
    assert results["early_stop_round"] == 1
    assert results["checkpoint_summary"]["best_round"] == 0


@pytest.mark.asyncio
async def test_last_configured_round_is_not_reported_as_early_stop(tmp_path):
    reward = OutputReward({"baseline": 1.0}, {"baseline": 0.0})
    runner = _runner(
        tmp_path,
        reward,
        max_rounds=1,
        early_stopping_patience=1,
    )

    async def reject(workflow, *args, **kwargs):
        return workflow, {"accepted": False, "gain": 0.0}

    runner.optimizer.optimize_round = reject
    results = await runner.run()

    assert len(results["rounds"]) == 1
    assert results["stopped_early"] is False
    assert results["early_stop_round"] is None


@pytest.mark.asyncio
async def test_opt_confirmation_rejects_hard_regression_without_validation(
    tmp_path,
):
    reward = OutputReward(
        {"baseline": 1.0, "candidate": 0.0},
        {"baseline": 0.0, "candidate": 2.0},
    )
    runner = _runner(
        tmp_path,
        reward,
        confirm_on_opt=True,
        confirm_repeats=2,
    )
    candidate = _workflow("candidate", version="1.1")

    async def accept(*args, **kwargs):
        return candidate, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = await runner.run()
    summary = results["rounds"][0]
    split_counts = Counter(split for _, split, _ in reward.observations)

    assert split_counts["optimization_confirmation_1"] == len(
        runner.opt_data
    )
    assert split_counts["optimization_confirmation_2"] == len(
        runner.opt_data
    )
    assert split_counts["validation"] == 0
    confirmation = summary["optimization_confirmation"]
    assert confirmation["performed"] is True
    assert confirmation["completed_repeats"] == 2
    assert confirmation["hard_non_regression_passed"] is False
    assert confirmation["mean_utility_effect_passed"] is False
    assert confirmation["decision"] == "rejected"
    assert summary["counterfactual_accepted"] is True
    assert summary["accepted"] is False
    assert summary["validation_reused"] is True
    assert (
        summary["validation_reuse_reason"]
        == "optimization_confirmation_failed"
    )
    assert runner.workflow.version == "1.0"


@pytest.mark.asyncio
async def test_confirmation_and_validation_promote_only_stable_candidate(
    tmp_path,
):
    reward = OutputReward(
        {"baseline": 0.8, "candidate": 1.0},
        {"baseline": 0.0, "candidate": 0.5},
    )
    runner = _runner(
        tmp_path,
        reward,
        validation_min_delta=0.1,
        confirm_on_opt=True,
        confirm_repeats=2,
    )
    candidate = _workflow("candidate", version="1.1")

    async def accept(*args, **kwargs):
        return candidate, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = await runner.run()
    summary = results["rounds"][0]

    assert summary["optimization_confirmation"]["decision"] == "accepted"
    assert summary["counterfactual_accepted"] is True
    assert summary["validation_gate_passed"] is True
    assert summary["accepted"] is True
    assert summary["is_best"] is True
    assert summary["incumbent_workflow_version_after"] == "1.1"
    assert runner.workflow.version == "1.1"
    assert results["checkpoint_summary"]["best_round"] == 1
