"""Regression tests for the inner-only workflow optimization mode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from awf.config.schema import OptimizerConfig
from awf.config.schema import ExperimentConfig
from awf.protocol.experiment import ExperimentRunner
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.optimizer.candidate_generator import WorkflowCandidate
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


class _Reward(RewardEvaluator):
    def hard_reward(
        self,
        query: str,
        ground_truth: object,
        output: object,
        trace: ExecutionTrace,
    ) -> float:
        return 0.0

    def process_reward(
        self,
        query: str,
        ground_truth: object,
        output: object,
        trace: ExecutionTrace,
    ) -> float:
        return 0.0


def _workflow() -> WorkflowTemplate:
    return WorkflowTemplate(
        name="inner-only",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "answer": Node(
                node_id="answer",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    system_prompt="answer carefully",
                    prompt_template="{query}",
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "answer"), ("answer", "end")],
    )


def test_workflow_content_only_keeps_inner_graph_path_candidates() -> None:
    config = OptimizerConfig(
        workflow_content_only=True,
        selective_update_enabled=True,
        efficiency_optimization_enabled=True,
        use_suffix_replay=True,
        llm={"api_key": "test"},
    )
    optimizer = LLMWorkflowOptimizer(config, _Reward())

    assert optimizer.workflow_content_only is True
    assert optimizer.candidate_generator.allowed_scopes == {
        "prompt",
        "operator",
        "block",
    }
    assert optimizer.selective_gate_searcher is None
    assert optimizer.counterfactual.suffix_replay is not None
    assert optimizer._require_failure_suffix_replay is True


def test_aggressive_cluster_mode_enables_all_scopes_without_edit_cap() -> None:
    config = OptimizerConfig(
        workflow_content_only=True,
        failure_cluster_enabled=True,
        multi_level_updates=True,
        max_edit_distance=None,
        mu_edit=0.0,
        candidates_per_round=4,
        llm={"api_key": "test"},
    )
    optimizer = LLMWorkflowOptimizer(config, _Reward())
    assert optimizer.candidate_generator.allowed_scopes == {
        "prompt",
        "operator",
        "block",
        "multi",
    }
    assert optimizer.candidate_generator.max_edit_distance is None

    lighter = WorkflowCandidate(
        scope="operator",
        node_id="answer",
        description="local",
        changes={"temperature": 0.2},
        edit_distance=0.1,
    )
    multi = WorkflowCandidate(
        scope="multi",
        node_id="answer",
        description="multi-level",
        changes={"patches": []},
        edit_distance=0.9,
    )
    selected = optimizer._select_best_candidate(
        [(lighter, 0.1), (multi, 0.8)],
        selective=False,
        failure_repair=True,
    )
    assert selected is not None and selected[0] is multi


def test_failure_repair_requires_successful_suffix_replay_for_every_failure() -> None:
    """A full-rerun success must not be promoted as an inner repair."""
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(hard_success_threshold=1.0)

    repaired = optimizer._failure_suffix_replay_gate(
        {
            "per_query_results": [
                {
                    "trace_id": "failure-1",
                    "candidate_success": True,
                    "candidate_hard": 1.0,
                    "candidate_suffix_replay_used": True,
                }
            ]
        },
        {"failure-1"},
    )
    assert repaired["passed"] is True

    full_rerun = optimizer._failure_suffix_replay_gate(
        {
            "per_query_results": [
                {
                    "trace_id": "failure-1",
                    "candidate_success": True,
                    "candidate_hard": 1.0,
                    "candidate_suffix_replay_used": False,
                }
            ]
        },
        {"failure-1"},
    )
    assert full_rerun["passed"] is False
    assert full_rerun["non_repaired_failure_traces"] == ["failure-1"]


def test_failure_repair_prefers_lightest_edit_after_gate() -> None:
    """Edit distance is only a preference among already-valid repairs."""
    optimizer = object.__new__(LLMWorkflowOptimizer)
    lighter = WorkflowCandidate(
        scope="operator",
        node_id="answer",
        description="small operator adjustment",
        changes={"temperature": 0.2},
        edit_distance=0.1,
    )
    heavier = WorkflowCandidate(
        scope="block",
        node_id="answer",
        description="larger graph-path repair",
        changes={"graph_patch": {"add_nodes": []}},
        edit_distance=0.4,
    )

    selected = optimizer._select_best_candidate(
        [(heavier, 0.95), (lighter, 0.10)],
        selective=False,
        failure_repair=True,
    )
    assert selected is not None
    assert selected[0] is lighter


def test_graph_path_repair_uses_surviving_suffix_node_when_anchor_removed() -> None:
    incumbent = _workflow()
    incumbent.nodes["verify"] = Node(
        node_id="verify",
        node_type=NodeType.LLM,
        config=NodeConfig(
            node_type=NodeType.LLM,
            system_prompt="verify",
            prompt_template="{query}",
        ),
    )
    incumbent.edges = [
        ("start", "answer"),
        ("answer", "verify"),
        ("verify", "end"),
    ]
    modified = incumbent.model_copy(deep=True)
    modified.nodes.pop("answer")
    modified.edges = [("start", "verify"), ("verify", "end")]
    candidate = WorkflowCandidate(
        scope="block",
        node_id="answer",
        description="remove the failed answer node",
        changes={"remove_node": "answer"},
        modified_workflow=modified,
        changed_units=["start", "answer", "end"],
    )

    replay_node = LLMWorkflowOptimizer._counterfactual_edit_node_id(
        candidate,
        incumbent,
    )
    assert replay_node == "verify"


def test_graph_path_insertion_before_replay_boundary_forces_full_rerun() -> None:
    incumbent = _workflow()
    modified = incumbent.model_copy(deep=True)
    modified.nodes["diagnose"] = Node(
        node_id="diagnose",
        node_type=NodeType.LLM,
        config=NodeConfig(
            node_type=NodeType.LLM,
            system_prompt="diagnose",
            prompt_template="{query}",
        ),
    )
    modified.edges = [
        ("start", "diagnose"),
        ("diagnose", "answer"),
        ("answer", "end"),
    ]
    candidate = WorkflowCandidate(
        scope="block",
        node_id="answer",
        description="insert a prefix diagnosis node",
        changes={"add_node": {"node_id": "diagnose"}},
        modified_workflow=modified,
        changed_units=["start", "diagnose", "answer"],
    )
    assert LLMWorkflowOptimizer._counterfactual_edit_node_id(
        candidate,
        incumbent,
    ) is None


@pytest.mark.asyncio
async def test_inner_only_round_reports_outer_layer_as_dormant() -> None:
    config = OptimizerConfig(
        workflow_content_only=True,
        efficiency_optimization_enabled=True,
        use_suffix_replay=True,
        llm={"api_key": "test"},
    )
    optimizer = LLMWorkflowOptimizer(config, _Reward())
    updated, summary = await optimizer.optimize_round(
        _workflow(),
        executor=object(),
        scheduler=object(),
    )

    assert updated == _workflow()
    assert summary["workflow_content_only"] is True
    assert summary["outer_execution_optimization_enabled"] is False
    assert summary["suffix_replay_requested"] is True
    assert summary["num_efficiency_anchors"] == 0


def test_runner_disables_scheduler_deviation_in_inner_only_mode(tmp_path) -> None:
    config = ExperimentConfig(
        output_dir=str(tmp_path),
        scheduler={
            "scheduler_type": "graph",
            "allow_deviation": True,
            "llm": {"api_key": "test"},
        },
        optimizer={"llm": {"api_key": "test"}},
    )
    runner = ExperimentRunner(config, _workflow(), _Reward())

    assert config.scheduler.allow_deviation is False
    assert runner.scheduler.config.allow_deviation is False
