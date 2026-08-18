"""Offline S-CWU composition, hard constraints, and optimizer integration."""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from awf.config.schema import OptimizerConfig
from awf.optimizer.candidate_archive import CandidateArchive
from awf.optimizer.candidate_generator import CandidateGenerator, WorkflowCandidate
from awf.optimizer.counterfactual import CounterfactualEvaluator
from awf.optimizer.selective_gate import SelectiveGateSearcher
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.workflow.gates import GateSpec
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


class _Reward(RewardEvaluator):
    def hard_reward(self, query, ground_truth, output, trace=None):
        return 0.0

    def process_reward(self, query, ground_truth, output, trace=None):
        return 0.0


def _row(
    trace_id: str,
    words: float,
    *,
    original_hard: float,
    candidate_hard: float,
    original_tokens: float = 100.0,
    candidate_tokens: float = 50.0,
    original_process: float = 0.0,
    candidate_process: float = 0.0,
) -> dict:
    delta_u = candidate_hard - original_hard
    return {
        "query": trace_id,
        "trace_id": trace_id,
        "candidate_trace_id": f"candidate-{trace_id}",
        "original_hard": original_hard,
        "candidate_hard": candidate_hard,
        "original_process": original_process,
        "candidate_process": candidate_process,
        "original_reward": original_hard,
        "candidate_reward": candidate_hard,
        "original_u": original_hard,
        "candidate_u": candidate_hard,
        "delta_u": delta_u,
        "original_total_tokens": original_tokens,
        "candidate_total_tokens": candidate_tokens,
        "original_latency_seconds": 1.0,
        "candidate_latency_seconds": 0.5,
        "original_path": ["work"],
        "candidate_path": ["work"],
        "gate_features": {
            "query_chars": words * 5.0,
            "query_words": words,
            "query_lines": 1.0,
            "numeric_literals": 0.0,
        },
    }


def _result(rows: list[dict]) -> dict:
    return {
        "delta_u": sum(row["delta_u"] for row in rows) / len(rows),
        "original_u": sum(row["original_u"] for row in rows) / len(rows),
        "candidate_u": sum(row["candidate_u"] for row in rows) / len(rows),
        "per_query_results": rows,
        "num_queries": len(rows),
        "evaluation_mode": "full_rerun",
        "suffix_replay_requested": False,
        "suffix_replay_used": False,
        "cache_hit": False,
    }


def _heterogeneous_rows() -> list[dict]:
    return [
        _row("failure-low", 1.0, original_hard=0.0, candidate_hard=1.0),
        _row("success-low", 2.0, original_hard=1.0, candidate_hard=1.0),
        _row("success-high-1", 10.0, original_hard=1.0, candidate_hard=0.0),
        _row("success-high-2", 11.0, original_hard=1.0, candidate_hard=0.0),
    ]


def test_offline_composition_uses_parent_row_when_gate_is_false() -> None:
    raw = _result(_heterogeneous_rows())
    original = copy.deepcopy(raw)
    gate = GateSpec(
        kind="threshold",
        feature="query_words",
        operator="le",
        threshold=5.0,
    )

    composed = CounterfactualEvaluator.compose_with_gate(raw, gate)

    by_id = {
        row["trace_id"]: row for row in composed["per_query_results"]
    }
    assert by_id["failure-low"]["candidate_hard"] == 1.0
    assert by_id["failure-low"]["gate_applied"] is True
    assert by_id["success-high-1"]["candidate_hard"] == 1.0
    assert by_id["success-high-1"]["candidate_total_tokens"] == 100.0
    assert by_id["success-high-1"]["delta_u"] == 0.0
    assert by_id["success-high-1"]["gate_applied"] is False
    assert composed["gate_coverage"] == 0.5
    assert raw == original


def test_hard_constrained_search_finds_safe_stump_when_always_regresses() -> None:
    rows = _heterogeneous_rows()
    searcher = SelectiveGateSearcher(
        features=["query_words"],
        min_leaf_support=2,
        min_effect=1e-6,
    )

    selected = searcher.search(
        _result(rows),
        fit_trace_ids={row["trace_id"] for row in rows},
        protected_trace_ids={
            "success-low",
            "success-high-1",
            "success-high-2",
        },
        edit_distance=0.1,
        failure_mode=True,
    )

    assert selected.positive is True
    assert selected.gate.kind == "threshold"
    assert selected.gate.feature == "query_words"
    assert selected.gate.operator == "le"
    assert selected.metrics["protected_regressions"] == 0
    assert selected.metrics["mean_hard_delta"] > 0.0


def test_process_or_cost_cannot_compensate_for_unavoidable_hard_regression() -> None:
    rows = [
        _row(
            "a",
            1.0,
            original_hard=1.0,
            candidate_hard=0.0,
            candidate_process=1.0,
            candidate_tokens=1.0,
        ),
        _row(
            "b",
            2.0,
            original_hard=1.0,
            candidate_hard=0.0,
            candidate_process=1.0,
            candidate_tokens=1.0,
        ),
    ]
    selected = SelectiveGateSearcher(
        features=["query_words"],
        min_leaf_support=2,
        min_effect=1e-6,
    ).search(
        _result(rows),
        fit_trace_ids={"a", "b"},
        protected_trace_ids={"a", "b"},
        edit_distance=0.1,
        failure_mode=True,
    )

    assert selected.gate.kind == "never"
    assert selected.positive is False


def test_gate_search_fails_closed_on_missing_protected_measurement() -> None:
    with pytest.raises(ValueError, match="protected success"):
        SelectiveGateSearcher(
            features=["query_words"],
            min_leaf_support=1,
        ).search(
            _result([_row("a", 1.0, original_hard=1.0, candidate_hard=1.0)]),
            fit_trace_ids={"a"},
            protected_trace_ids={"a", "missing"},
            edit_distance=0.1,
            failure_mode=False,
        )


def _workflow() -> WorkflowTemplate:
    return WorkflowTemplate(
        name="optimizer-selective",
        version="1.0",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "work": Node(
                node_id="work",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    prompt_template="{query}",
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "work"), ("work", "end")],
    )


def _trace(trace_id: str, query: str, hard: float) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        query_text=query,
        workflow_name="optimizer-selective",
        workflow_version="1.0",
        hard_reward=hard,
        process_reward=0.0,
        success=hard >= 1.0,
        metadata={"split": "optimization"},
    )


@pytest.mark.asyncio
async def test_optimizer_selects_joint_patch_gate_and_uses_representative_rows():
    workflow = _workflow()
    modified = copy.deepcopy(workflow)
    modified.nodes["work"].config.prompt_template = "short {query}"
    candidate = WorkflowCandidate(
        scope="prompt",
        node_id="work",
        anchor_id="work",
        description="shorten only where safe",
        changes={"user_template": "short {query}"},
        modified_workflow=modified,
        edit_distance=0.1,
        changed_units=["work"],
        metadata={"patch_fingerprint": "c" * 64},
    )
    config = OptimizerConfig(
        workflow_content_only=False,
        candidates_per_round=3,
        min_success_guards=1,
        selective_update_enabled=True,
        gate_features=["query_words"],
        gate_min_leaf_support=1,
        gate_fit_max_traces=4,
        epsilon_stat=1e-6,
        candidate_archive_size=2,
        llm={"api_key": "test"},
    )
    optimizer = LLMWorkflowOptimizer(config, _Reward())
    optimizer.anchor_localizer.localize_batch = AsyncMock(
        return_value=[
            {
                "node_id": "work",
                "unit_id": "work",
                "rank": 1,
                "score": 1.0,
                "reason": "failure",
            }
        ]
    )

    async def generate_for_anchor(
        workflow,
        anchor,
        scope,
        failure_context,
        *,
        limit,
        experience_context="",
    ):
        del workflow, anchor, failure_context, limit, experience_context
        return [candidate] if scope == "prompt" else []

    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    rows = _heterogeneous_rows()
    optimizer.counterfactual.prepare_baseline = MagicMock(return_value={})
    optimizer.counterfactual.evaluate = AsyncMock(
        return_value=_result(rows)
    )
    optimizer.failure_buffer.extend(
        [
            _trace("failure-low", "x", 0.0),
            _trace("success-low", "x y", 1.0),
            _trace(
                "success-high-1",
                " ".join(["x"] * 10),
                1.0,
            ),
            _trace(
                "success-high-2",
                " ".join(["x"] * 11),
                1.0,
            ),
        ]
    )

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert summary["accepted"] is True
    assert summary["counterfactual_batch_size"] == 2
    assert summary["candidate_evaluation_batch_size"] == 4
    assert summary["gate_fit_batch_size"] == 4
    assert summary["selected_gate"]["kind"] == "threshold"
    assert summary["candidate_joint_fingerprint"]
    assert updated.selective_update is not None
    assert updated.version == "1.1"
    assert optimizer.counterfactual.evaluate.await_args.args[1]
    assert len(optimizer.counterfactual.evaluate.await_args.args[1]) == 4
    assert summary["candidate_archive"][0]["joint_fingerprint"] == (
        summary["candidate_joint_fingerprint"]
    )

    unchanged, nested_summary = await optimizer.optimize_round(
        updated,
        object(),
        object(),
    )
    assert unchanged == updated
    assert nested_summary["accepted"] is False
    assert nested_summary["selective_update_rejection_reason"] == (
        "nested_policy_not_supported"
    )


def test_archive_distinguishes_two_gates_for_the_same_patch() -> None:
    archive = CandidateArchive(capacity=2)
    rows = [
        _row("a", 1.0, original_hard=1.0, candidate_hard=1.0)
    ]
    base_record = {
        "patch_fingerprint": "d" * 64,
        "anchor_id": "work",
        "scope": "prompt",
        "gain": 0.1,
        "edit_distance": 0.1,
        "per_query_results": rows,
        "gate_coverage": 1.0,
    }
    first = {
        **base_record,
        "joint_fingerprint": "e" * 64,
        "gate": {"kind": "always"},
    }
    second = {
        **base_record,
        "joint_fingerprint": "f" * 64,
        "gate": {
            "kind": "threshold",
            "feature": "query_words",
            "operator": "le",
            "threshold": 2.0,
        },
    }

    assert archive.add(
        first,
        round_num=1,
        split="optimization",
        comparison_key="1" * 64,
        expected_trace_ids={"a"},
    )
    assert archive.add(
        second,
        round_num=1,
        split="optimization",
        comparison_key="1" * 64,
        expected_trace_ids={"a"},
    )
    assert {
        item["joint_fingerprint"] for item in archive.snapshot()
    } == {"e" * 64, "f" * 64}


def test_condition_operator_update_reuses_safe_ast_and_dominance_validation():
    workflow = WorkflowTemplate(
        name="condition-edit",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "gate": Node(
                node_id="gate",
                node_type=NodeType.CONDITION,
                config=NodeConfig(
                    node_type=NodeType.CONDITION,
                    condition_expr="True",
                ),
            ),
            "yes": Node(node_id="yes", node_type=NodeType.LLM),
            "no": Node(node_id="no", node_type=NodeType.LLM),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[
            ("start", "gate"),
            ("gate", "yes"),
            ("gate", "no"),
            ("yes", "end"),
            ("no", "end"),
        ],
    )
    generator = CandidateGenerator(MagicMock())
    unsafe = WorkflowCandidate(
        scope="operator",
        node_id="gate",
        description="unsafe",
        changes={"condition_expr": '__import__("os")'},
    )
    downstream = WorkflowCandidate(
        scope="operator",
        node_id="gate",
        description="future leakage",
        changes={"condition_expr": 'outputs["yes"] == "ok"'},
    )

    assert generator._apply_to_workflow(unsafe, workflow) is None
    assert generator._apply_to_workflow(downstream, workflow) is None
