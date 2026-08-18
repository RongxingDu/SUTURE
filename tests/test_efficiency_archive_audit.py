"""Regression tests for audited efficiency-search safety boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from awf.config.schema import OptimizerConfig
from awf.optimizer.acceptance import AcceptanceCriterion
from awf.optimizer.candidate_archive import CandidateArchive
from awf.optimizer.candidate_generator import WorkflowCandidate
from awf.optimizer.failure_buffer import FailureBuffer
from awf.optimizer.scorer import CandidateScorer
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.trace.schema import ExecutionTrace, LLMCallRecord, TraceStep
from awf.utility.compute import UtilityComputer
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _workflow(*, two_nodes: bool = False) -> WorkflowTemplate:
    nodes = {
        "start": Node(node_id="start", node_type=NodeType.START),
        "a": Node(
            node_id="a",
            node_type=NodeType.LLM,
            config=NodeConfig(
                node_type=NodeType.LLM,
                system_prompt="baseline-a",
                prompt_template="{query}",
            ),
        ),
        "end": Node(node_id="end", node_type=NodeType.END),
    }
    edges = [("start", "a"), ("a", "end")]
    if two_nodes:
        nodes["b"] = Node(
            node_id="b",
            node_type=NodeType.LLM,
            config=NodeConfig(
                node_type=NodeType.LLM,
                system_prompt="baseline-b",
                prompt_template="{query}",
            ),
        )
        edges = [("start", "a"), ("a", "b"), ("b", "end")]
    return WorkflowTemplate(
        name="wf",
        version="1.0",
        entry_node="start",
        nodes=nodes,
        edges=edges,
    )


def _trace(
    trace_id: str,
    *,
    hard: float,
    tokens: int = 100,
    steps: list[TraceStep] | None = None,
) -> ExecutionTrace:
    if steps is None:
        steps = [
            TraceStep(
                step_id=f"{trace_id}-step",
                step_index=0,
                node_id="a",
                node_type="llm",
                action="execute",
                llm_calls=[
                    LLMCallRecord(
                        call_id=f"{trace_id}-call",
                        call_type="workflow",
                        prompt_tokens=tokens,
                    )
                ],
            )
        ]
    return ExecutionTrace(
        trace_id=trace_id,
        query_id=trace_id,
        query_text=f"optimization-query-{trace_id}",
        workflow_name="wf",
        workflow_version="1.0",
        final_output="answer",
        success=hard >= 0.5,
        hard_reward=hard,
        process_reward=hard,
        total_prompt_tokens=tokens,
        total_llm_calls=sum(len(step.llm_calls) for step in steps),
        metadata={"ground_truth": "answer", "split": "optimization"},
        steps=steps,
    )


def _candidate(
    workflow: WorkflowTemplate,
    *,
    scope: str,
    prompt: str,
    edit_distance: float,
    fingerprint_character: str,
) -> WorkflowCandidate:
    modified = workflow.model_copy(deep=True)
    modified.nodes["a"].config.system_prompt = prompt
    candidate = WorkflowCandidate(
        scope=scope,
        node_id="a",
        anchor_id="a",
        description=f"{scope} candidate",
        changes={"system_prompt": prompt},
        modified_workflow=modified,
        edit_distance=edit_distance,
        changed_units=["a"],
    )
    candidate.metadata["patch_fingerprint"] = fingerprint_character * 64
    return candidate


def _row(
    trace_id: str,
    *,
    original_hard: float,
    candidate_hard: float,
    delta_u: float,
    original_tokens: float = 100.0,
    candidate_tokens: float = 80.0,
    original_latency: float = 1.0,
    candidate_latency: float = 0.8,
) -> dict:
    return {
        "trace_id": trace_id,
        "original_hard": original_hard,
        "candidate_hard": candidate_hard,
        "delta_u": delta_u,
        "original_total_tokens": original_tokens,
        "candidate_total_tokens": candidate_tokens,
        "original_latency_seconds": original_latency,
        "candidate_latency_seconds": candidate_latency,
    }


def _result(rows: list[dict], *, delta_u: float) -> dict:
    return {
        "delta_u": delta_u,
        "evaluation_mode": "full_rerun",
        "cache_hit": False,
        "per_query_results": rows,
    }


def _archive_record(
    fingerprint: str,
    rows: list[dict],
    *,
    gain: float = 0.5,
    edit_distance: float = 0.1,
    description: str = "aggregate-only candidate",
) -> dict:
    return {
        "patch_fingerprint": fingerprint,
        "anchor_id": "a",
        "scope": "prompt",
        "description": description,
        "gain": gain,
        "edit_distance": edit_distance,
        "per_query_results": rows,
    }


def _optimizer_for_round(
    workflow: WorkflowTemplate,
    *,
    efficiency_enabled: bool,
    exploration_budget: int,
    counterfactual_results: list[dict],
) -> LLMWorkflowOptimizer:
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        candidates_per_round=1,
        success_guard_fraction=0.2,
        min_success_guards=1 if efficiency_enabled else 0,
        efficiency_optimization_enabled=efficiency_enabled,
        efficiency_anchor_fraction=0.2,
        efficiency_min_relative_cost=2.0,
        efficiency_max_anchors=1,
        exploration_budget=exploration_budget,
    )
    optimizer.failure_buffer = FailureBuffer(capacity=20)
    optimizer.utility_computer = UtilityComputer(
        lambda_cost=1e-3,
        lambda_latency=0.0,
        rho_omega=0.0,
    )
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {
            "node_id": "a",
            "rank": 1,
            "score": 1.0,
            "reason": "failed output",
        }
    ]
    optimizer.candidate_generator = MagicMock()
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {}
    counterfactual.evaluate = AsyncMock(side_effect=counterfactual_results)
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.candidate_archive = CandidateArchive(capacity=3)
    optimizer.round_history = []
    optimizer._candidate_outcomes = {}
    return optimizer


def test_missing_expected_measurement_is_not_archived():
    archive = CandidateArchive(capacity=2)
    record = _archive_record(
        "1" * 64,
        [
            _row(
                "present",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.5,
            )
        ],
    )

    retained = archive.add(
        record,
        round_num=1,
        split="optimization",
        comparison_key="a" * 64,
        expected_trace_ids={"present", "missing"},
    )

    assert retained is False
    assert archive.snapshot() == []
    assert archive.experience_context() == ""


@pytest.mark.asyncio
async def test_efficiency_mode_guards_success_without_efficiency_anchor():
    workflow = _workflow()
    candidate = _candidate(
        workflow,
        scope="prompt",
        prompt="unsafe",
        edit_distance=0.1,
        fingerprint_character="2",
    )
    optimizer = _optimizer_for_round(
        workflow,
        efficiency_enabled=True,
        exploration_budget=0,
        counterfactual_results=[
            _result(
                [
                    _row(
                        "failure",
                        original_hard=0.0,
                        candidate_hard=1.0,
                        delta_u=1.0,
                    ),
                    _row(
                        "ordinary-success",
                        original_hard=1.0,
                        candidate_hard=0.0,
                        delta_u=10.0,
                    ),
                ],
                delta_u=5.5,
            )
        ],
    )

    async def generate_for_anchor(
        workflow,
        anchor,
        scope,
        context,
        *,
        limit,
        experience_context="",
    ):
        del workflow, anchor, context, limit, experience_context
        return [candidate] if scope == "prompt" else []

    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    optimizer.failure_buffer.extend(
        [
            _trace("failure", hard=0.0),
            _trace("ordinary-success", hard=1.0),
        ]
    )

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert summary["num_efficiency_anchors"] == 0
    assert summary["num_success_guards"] == 1
    assert updated.version == "1.0"
    assert summary["accepted"] is False
    record = summary["candidate_evaluations"][0]
    assert record["status"] == "hard_success_regression"
    assert record["hard_success_guard"]["required"] is True
    assert record["hard_success_guard"]["regressions"] == 1
    assert summary["candidate_archive"] == []


def test_partial_hard_reward_is_failure_at_default_threshold():
    assert OptimizerConfig().hard_success_threshold == 1.0
    buffer = FailureBuffer()
    partial = _trace("partial", hard=0.5)

    buffer.add(partial)

    assert buffer.successes == []
    assert buffer.failures == [partial]


def test_candidate_description_never_enters_archive_or_prompt_context():
    sentinel = "RAW_QUERY_OR_OUTPUT_SENTINEL_MUST_NOT_SURVIVE"
    archive = CandidateArchive(capacity=2)
    record = _archive_record(
        "3" * 64,
        [
            _row(
                "trace-1",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.2,
            )
        ],
        description=sentinel,
    )

    assert archive.add(
        record,
        round_num=1,
        split="optimization",
        comparison_key="b" * 64,
        expected_trace_ids={"trace-1"},
    )

    snapshot_text = repr(archive.snapshot())
    context = archive.experience_context()
    assert sentinel not in snapshot_text
    assert sentinel not in context
    assert "description" not in archive.snapshot()[0]


def test_different_comparison_batches_do_not_dominate_and_are_marked_approximate():
    archive = CandidateArchive(capacity=2)
    dominant_metrics = _archive_record(
        "4" * 64,
        [
            _row(
                "trace-1",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=1.0,
                candidate_tokens=20.0,
                candidate_latency=0.1,
            )
        ],
        gain=1.0,
        edit_distance=0.1,
    )
    otherwise_dominated = _archive_record(
        "5" * 64,
        [
            _row(
                "trace-1",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.0,
                candidate_tokens=100.0,
                candidate_latency=1.0,
            )
        ],
        gain=0.0,
        edit_distance=0.5,
    )

    assert archive.add(
        dominant_metrics,
        round_num=1,
        split="optimization",
        comparison_key="c" * 64,
        expected_trace_ids={"trace-1"},
    )
    assert archive.add(
        otherwise_dominated,
        round_num=2,
        split="optimization",
        comparison_key="d" * 64,
        expected_trace_ids={"trace-1"},
    )

    snapshot = archive.snapshot()
    assert len(snapshot) == 2
    assert all(entry["pareto"] for entry in snapshot)
    assert {
        entry["pareto_scope"] for entry in snapshot
    } == {"bounded_within_comparison_batch"}
    context = archive.experience_context()
    assert "approximate within-batch Pareto" in context
    assert context.count("approx_pareto_within_batch=True") == 2


@pytest.mark.asyncio
async def test_two_exploration_evaluations_are_reserved_for_operator_and_block():
    workflow = _workflow()
    candidates = {
        "prompt": _candidate(
            workflow,
            scope="prompt",
            prompt="prompt-plan",
            edit_distance=0.1,
            fingerprint_character="6",
        ),
        "operator": _candidate(
            workflow,
            scope="operator",
            prompt="operator-plan",
            edit_distance=0.2,
            fingerprint_character="7",
        ),
        "block": _candidate(
            workflow,
            scope="block",
            prompt="block-plan",
            edit_distance=0.3,
            fingerprint_character="8",
        ),
    }
    results = [
        _result(
            [
                _row(
                    "failure",
                    original_hard=0.0,
                    candidate_hard=1.0,
                    delta_u=0.5,
                )
            ],
            delta_u=0.5,
        ),
        _result(
            [
                _row(
                    "failure",
                    original_hard=0.0,
                    candidate_hard=1.0,
                    delta_u=0.7,
                )
            ],
            delta_u=0.7,
        ),
        _result(
            [
                _row(
                    "failure",
                    original_hard=0.0,
                    candidate_hard=1.0,
                    delta_u=0.9,
                )
            ],
            delta_u=0.9,
        ),
    ]
    optimizer = _optimizer_for_round(
        workflow,
        efficiency_enabled=False,
        exploration_budget=2,
        counterfactual_results=results,
    )
    requested_limits: list[tuple[str, int]] = []

    async def generate_for_anchor(
        workflow,
        anchor,
        scope,
        context,
        *,
        limit,
        experience_context="",
    ):
        del workflow, anchor, context, experience_context
        requested_limits.append((scope, limit))
        return [candidates[scope]]

    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    optimizer.failure_buffer.add(_trace("failure", hard=0.0))

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert requested_limits == [
        ("prompt", 1),
        ("operator", 1),
        ("block", 1),
    ]
    assert summary["evaluated_candidates"] == 3
    assert summary["exploration_candidates_evaluated"] == 2
    assert [
        (attempt["scope"], attempt["evaluation_budget"], attempt["evaluated"])
        for attempt in summary["scope_attempts"]
    ] == [
        ("prompt", "primary", 1),
        ("operator", "exploration", 1),
        ("block", "exploration", 1),
    ]
    assert summary["candidate_scope"] == "block"
    assert updated.nodes["a"].config.system_prompt == "block-plan"


def test_efficiency_localizer_excludes_scheduler_call_cost_from_node_share():
    workflow = _workflow(two_nodes=True)
    steps = [
        TraceStep(
            step_id="step-a",
            step_index=0,
            node_id="a",
            node_type="llm",
            action="execute",
            llm_calls=[
                LLMCallRecord(
                    call_id="scheduler-a",
                    call_type="scheduler",
                    prompt_tokens=1000,
                ),
                LLMCallRecord(
                    call_id="workflow-a",
                    call_type="workflow",
                    prompt_tokens=10,
                ),
            ],
        ),
        TraceStep(
            step_id="step-b",
            step_index=1,
            node_id="b",
            node_type="llm",
            action="execute",
            llm_calls=[
                LLMCallRecord(
                    call_id="workflow-b",
                    call_type="workflow",
                    prompt_tokens=90,
                )
            ],
        ),
    ]
    trace = _trace("expensive-success", hard=1.0, tokens=1100, steps=steps)
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.utility_computer = UtilityComputer(
        lambda_cost=1.0,
        lambda_latency=0.0,
        lambda_api_cost=0.0,
        rho_omega=0.0,
    )

    anchors = optimizer._localize_efficiency_anchors(
        [trace],
        workflow,
        top_m=2,
    )

    assert [anchor["node_id"] for anchor in anchors] == ["b", "a"]
    scores = {anchor["node_id"]: anchor["score"] for anchor in anchors}
    assert scores == {
        "a": pytest.approx(0.1),
        "b": pytest.approx(0.9),
    }
