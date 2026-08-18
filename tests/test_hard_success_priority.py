"""Temp sanity test for hard_success_priority promotion (removed after verify)."""

from __future__ import annotations

import asyncio

from awf.config.schema import ExperimentConfig
from awf.protocol.experiment import ExperimentRunner
from awf.reward.base import RewardEvaluator
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _emit(value: str) -> str:
    return value


def _wf(output: str, version: str = "1.0") -> WorkflowTemplate:
    return WorkflowTemplate(
        name="w",
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


class R(RewardEvaluator):
    def __init__(self, hard, proc):
        self.hard, self.proc = hard, proc

    def hard_reward(self, q, gt, out, trace):
        return self.hard[str(out)]

    def process_reward(self, q, gt, out, trace):
        return self.proc[str(out)]


def test_priority_promotes_accuracy_gain(tmp_path):
    reward = R(
        {"baseline": 0.6, "candidate": 0.8},
        {"baseline": 0.1, "candidate": 0.0},
    )
    config = ExperimentConfig(
        name="p",
        output_dir=str(tmp_path),
        seed=31,
        scheduler={"scheduler_type": "fixed", "llm": {"api_key": "k"}},
        optimizer={
            "max_rounds": 1,
            "llm": {"api_key": "k"},
            "lambda_cost": 0.0,
            "rho_omega": 0.0,
        },
        reward={"alpha_process": 0.4},
        executor={"trace_enabled": False},
        hard_success_priority=True,
        validation_min_delta=0.01,
        validation_hard_regression_tolerance=0.1,
    )
    runner = ExperimentRunner(
        config=config,
        workflow=_wf("baseline"),
        reward_evaluator=reward,
        operators={"emit": _emit},
        run_metadata={"benchmark": "math", "code_execution_mode": "n/a"},
    )
    runner.load_data([(f"q{i}", f"t{i}") for i in range(10)])
    cand = _wf("candidate", version="1.1")

    async def accept(workflow, *a, **k):
        return cand, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = asyncio.run(runner.run())
    s = results["rounds"][0]
    assert s["validation_gate"]["promotion_mode"] == "hard_priority"
    assert s["validation_gate"]["hard_non_regression_passed"] is True
    assert s["validation_gate"]["runtime_utility_passed"] is True
    assert s["validation_gate_passed"] is True
    assert s["is_best"] is True
    assert runner.workflow.version == "1.1"
    assert results["checkpoint_summary"]["best_round"] == 1


def test_default_mode_uses_hard_reward_utility_not_process_reward(tmp_path):
    # Candidate hard reward improves even though its process reward decreases.
    # Reward-only utility must promote it in the default mode.
    reward = R(
        {"baseline": 0.6, "candidate": 0.8},
        {"baseline": 0.1, "candidate": -0.5},
    )
    config = ExperimentConfig(
        name="d",
        output_dir=str(tmp_path),
        seed=31,
        scheduler={"scheduler_type": "fixed", "llm": {"api_key": "k"}},
        optimizer={
            "max_rounds": 1,
            "llm": {"api_key": "k"},
            "lambda_cost": 0.0,
            "rho_omega": 0.0,
        },
        reward={"alpha_process": 0.4},
        executor={"trace_enabled": False},
        hard_success_priority=False,
        validation_min_delta=0.01,
    )
    runner = ExperimentRunner(
        config=config,
        workflow=_wf("baseline"),
        reward_evaluator=reward,
        operators={"emit": _emit},
        run_metadata={"benchmark": "math", "code_execution_mode": "n/a"},
    )
    runner.load_data([(f"q{i}", f"t{i}") for i in range(10)])
    cand = _wf("candidate", version="1.1")

    async def accept(workflow, *a, **k):
        return cand, {"accepted": True, "gain": 1.0}

    runner.optimizer.optimize_round = accept
    results = asyncio.run(runner.run())
    s = results["rounds"][0]
    assert s["validation_gate"]["promotion_mode"] == "utility_delta"
    assert s["validation_gate_passed"] is True
    assert runner.workflow.version == "1.1"
    assert results["checkpoint_summary"]["best_round"] == 1
