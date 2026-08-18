"""Offline tests for efficiency triggers, hard guards, and search diversity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

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


def _workflow() -> WorkflowTemplate:
    return WorkflowTemplate(
        name="wf",
        version="1.0",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "a": Node(
                node_id="a",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    system_prompt="baseline",
                    prompt_template="{query}",
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "a"), ("a", "end")],
    )


def _trace(
    trace_id: str,
    *,
    hard: float,
    tokens: int,
    latency: float = 1.0,
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        query_id=trace_id,
        query_text=f"optimization-query-{trace_id}",
        workflow_name="wf",
        workflow_version="1.0",
        final_output="answer",
        success=hard >= 0.5,
        hard_reward=hard,
        process_reward=1.0 if hard >= 0.5 else 0.0,
        total_prompt_tokens=tokens,
        total_latency_seconds=latency,
        total_llm_calls=1,
        metadata={"ground_truth": "answer", "split": "optimization"},
        steps=[
            TraceStep(
                step_id=f"{trace_id}-step",
                step_index=0,
                node_id="a",
                node_type="llm",
                action="execute",
                success=True,
                llm_calls=[
                    LLMCallRecord(
                        call_id=f"{trace_id}-call",
                        prompt_tokens=tokens,
                        latency_seconds=latency,
                    )
                ],
            )
        ],
    )


def _candidate(
    workflow: WorkflowTemplate,
    *,
    description: str,
    prompt: str,
    edit_distance: float,
    scope: str = "prompt",
) -> WorkflowCandidate:
    modified = workflow.model_copy(deep=True)
    modified.nodes["a"].config.system_prompt = prompt
    return WorkflowCandidate(
        scope=scope,
        node_id="a",
        anchor_id="a",
        description=description,
        changes={"system_prompt": prompt},
        modified_workflow=modified,
        edit_distance=edit_distance,
        changed_units=["a"],
    )


def _result(
    rows: list[dict],
    *,
    delta_u: float,
) -> dict:
    return {
        "delta_u": delta_u,
        "evaluation_mode": "full_rerun",
        "cache_hit": False,
        "per_query_results": rows,
    }


def _row(
    trace_id: str,
    *,
    original_hard: float,
    candidate_hard: float,
    delta_u: float,
    original_tokens: int,
    candidate_tokens: int,
    original_latency: float = 1.0,
    candidate_latency: float = 0.5,
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


def _optimizer(
    workflow: WorkflowTemplate,
    result: dict,
    *,
    exploration_budget: int = 0,
) -> LLMWorkflowOptimizer:
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        candidates_per_round=1,
        success_guard_fraction=0.2,
        min_success_guards=1,
        efficiency_optimization_enabled=True,
        efficiency_anchor_fraction=0.34,
        efficiency_min_relative_cost=1.25,
        efficiency_max_anchors=1,
        exploration_budget=exploration_budget,
    )
    optimizer.failure_buffer = FailureBuffer(capacity=20)
    optimizer.utility_computer = UtilityComputer(
        lambda_cost=1e-3,
        lambda_latency=0.01,
        rho_omega=0.0,
    )
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = []
    optimizer.candidate_generator = AsyncMock()
    optimizer.candidate_generator.generate_for_anchor.return_value = [
        _candidate(
            workflow,
            description="make successful path concise",
            prompt="concise",
            edit_distance=0.1,
        )
    ]
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {"baseline": True}
    counterfactual.evaluate = AsyncMock(return_value=result)
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.candidate_archive = CandidateArchive(capacity=3)
    optimizer.round_history = []
    optimizer._candidate_outcomes = {}
    return optimizer


def test_efficiency_config_is_opt_in_and_strictly_validated():
    defaults = OptimizerConfig()
    assert defaults.efficiency_optimization_enabled is False
    assert defaults.candidate_archive_size == 0
    assert defaults.exploration_budget == 0

    configured = OptimizerConfig(
        workflow_content_only=False,
        efficiency_optimization_enabled=True,
        min_success_guards=1,
        efficiency_anchor_fraction=0.3,
        efficiency_min_relative_cost=1.1,
        efficiency_max_anchors=2,
        candidate_archive_size=4,
        exploration_budget=2,
    )
    assert configured.exploration_budget == 2

    with pytest.raises(ValidationError, match="runtime-cost coefficient"):
        OptimizerConfig(
            workflow_content_only=False,
            efficiency_optimization_enabled=True,
            min_success_guards=1,
            lambda_cost=0.0,
            lambda_latency=0.0,
            lambda_api_cost=0.0,
            rho_omega=0.0,
        )
    with pytest.raises(ValidationError, match="candidate_archive_size"):
        OptimizerConfig(workflow_content_only=False, exploration_budget=1)
    with pytest.raises(ValidationError, match="min_success_guards"):
        OptimizerConfig(
            workflow_content_only=False,
            efficiency_optimization_enabled=True,
        )
    with pytest.raises(ValidationError):
        OptimizerConfig(efficiency_min_relative_cost=0.9)


def test_failure_buffer_dual_trigger_selects_only_relative_high_cost_success():
    buffer = FailureBuffer(capacity=20)
    for trace_id, tokens in (
        ("low-1", 100),
        ("low-2", 100),
        ("expensive", 1000),
    ):
        buffer.add(_trace(trace_id, hard=1.0, tokens=tokens))

    batch = buffer.consume_optimization_round(
        "wf",
        "1.0",
        success_fraction=0.2,
        min_success_guards=1,
        efficiency_enabled=True,
        efficiency_fraction=0.34,
        efficiency_min_relative_cost=1.25,
        max_efficiency_anchors=1,
        efficiency_cost=lambda trace: trace.total_tokens,
    )

    assert batch.failures == []
    assert [trace.trace_id for trace in batch.efficiency_anchors] == [
        "expensive"
    ]
    assert [trace.trace_id for trace in batch.success_guards] == ["low-2"]
    assert batch.counterfactual_batch == (
        batch.efficiency_anchors + batch.success_guards
    )


@pytest.mark.asyncio
async def test_efficiency_only_round_accepts_cost_reduction_with_hard_guard():
    workflow = _workflow()
    result = _result(
        [
            _row(
                "expensive",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.5,
                original_tokens=1000,
                candidate_tokens=400,
            ),
            _row(
                "low-2",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.05,
                original_tokens=100,
                candidate_tokens=90,
            ),
        ],
        delta_u=0.275,
    )
    optimizer = _optimizer(workflow, result)
    optimizer.failure_buffer.extend(
        [
            _trace("low-1", hard=1.0, tokens=100),
            _trace("low-2", hard=1.0, tokens=100),
            _trace("expensive", hard=1.0, tokens=1000),
        ]
    )

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert updated.version == "1.1"
    assert summary["accepted"] is True
    assert summary["num_failures"] == 0
    assert summary["num_efficiency_anchors"] == 1
    assert summary["trigger_types"] == ["high_cost_success"]
    assert summary["candidate_evaluations"][0]["hard_success_guard"] == {
        "required": True,
        "passed": True,
        "protected_traces": 2,
        "regressions": 0,
        "missing_measurements": 0,
    }
    assert summary["candidate_archive"]
    optimizer.anchor_localizer.localize_batch.assert_awaited_once_with(
        [],
        workflow,
        top_m=1,
    )


@pytest.mark.asyncio
async def test_efficiency_candidate_cannot_trade_hard_reward_for_large_saving():
    workflow = _workflow()
    result = _result(
        [
            _row(
                "expensive",
                original_hard=1.0,
                candidate_hard=0.0,
                delta_u=100.0,
                original_tokens=1000,
                candidate_tokens=1,
            ),
            _row(
                "low-2",
                original_hard=1.0,
                candidate_hard=1.0,
                delta_u=0.1,
                original_tokens=100,
                candidate_tokens=50,
            ),
        ],
        delta_u=50.0,
    )
    optimizer = _optimizer(workflow, result)
    optimizer.failure_buffer.extend(
        [
            _trace("low-1", hard=1.0, tokens=100),
            _trace("low-2", hard=1.0, tokens=100),
            _trace("expensive", hard=1.0, tokens=1000),
        ]
    )

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert updated.version == "1.0"
    assert summary["accepted"] is False
    record = summary["candidate_evaluations"][0]
    assert record["status"] == "hard_success_regression"
    assert record["hard_success_guard"]["passed"] is False
    assert record["hard_success_guard"]["regressions"] == 1
    assert summary["candidate_archive"] == []


def _archive_record(
    fingerprint: str,
    *,
    gain: float,
    hard_delta: float,
    utility_delta: float,
    token_delta: float,
    latency_delta: float,
    edit_distance: float,
) -> dict:
    return {
        "patch_fingerprint": fingerprint,
        "anchor_id": "a",
        "scope": "prompt",
        "description": "safe aggregate description",
        "gain": gain,
        "edit_distance": edit_distance,
        "per_query_results": [
            {
                "trace_id": "archive-trace",
                "query": "RAW_OPTIMIZATION_QUERY_MUST_NOT_BE_ARCHIVED",
                "candidate_output": "RAW_OUTPUT_MUST_NOT_BE_ARCHIVED",
                "original_hard": 1.0,
                "candidate_hard": 1.0 + hard_delta,
                "delta_u": utility_delta,
                "original_total_tokens": 100.0,
                "candidate_total_tokens": 100.0 + token_delta,
                "original_latency_seconds": 2.0,
                "candidate_latency_seconds": 2.0 + latency_delta,
            }
        ],
    }


def _add_archive(
    archive: CandidateArchive,
    record: dict,
    *,
    round_num: int,
) -> bool:
    return archive.add(
        record,
        round_num=round_num,
        split="optimization",
        comparison_key="a" * 64,
        expected_trace_ids={"archive-trace"},
    )


def test_candidate_archive_is_bounded_pareto_first_and_aggregate_only():
    archive = CandidateArchive(capacity=2)
    assert _add_archive(
        archive,
        _archive_record(
            "a",
            gain=0.5,
            hard_delta=0.0,
            utility_delta=0.5,
            token_delta=-50.0,
            latency_delta=0.2,
            edit_distance=0.1,
        ),
        round_num=1,
    )
    assert _add_archive(
        archive,
        _archive_record(
            "b",
            gain=0.6,
            hard_delta=0.0,
            utility_delta=0.6,
            token_delta=-10.0,
            latency_delta=-1.0,
            edit_distance=0.3,
        ),
        round_num=1,
    )
    assert not _add_archive(
        archive,
        _archive_record(
            "dominated",
            gain=0.1,
            hard_delta=0.0,
            utility_delta=0.1,
            token_delta=10.0,
            latency_delta=1.0,
            edit_distance=0.4,
        ),
        round_num=2,
    )
    assert not _add_archive(
        archive,
        _archive_record(
            "hard-regression",
            gain=100.0,
            hard_delta=-1.0,
            utility_delta=100.0,
            token_delta=-99.0,
            latency_delta=-1.9,
            edit_distance=0.01,
        ),
        round_num=2,
    )

    snapshot = archive.snapshot()
    assert {entry["patch_fingerprint"] for entry in snapshot} == {"a", "b"}
    assert all(entry["pareto"] for entry in snapshot)
    serialized = repr(snapshot)
    assert "RAW_OPTIMIZATION_QUERY" not in serialized
    assert "RAW_OUTPUT" not in serialized


@pytest.mark.asyncio
async def test_exploration_budget_evaluates_broader_scope_and_archives_tradeoff():
    workflow = _workflow()
    prompt_candidate = _candidate(
        workflow,
        description="small prompt edit",
        prompt="small",
        edit_distance=0.1,
    )
    operator_candidate = _candidate(
        workflow,
        description="broader operator edit",
        prompt="broader",
        edit_distance=0.3,
        scope="operator",
    )

    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        candidates_per_round=1,
        success_guard_fraction=0.2,
        min_success_guards=0,
        efficiency_optimization_enabled=False,
        efficiency_anchor_fraction=0.2,
        efficiency_min_relative_cost=1.25,
        efficiency_max_anchors=1,
        exploration_budget=1,
    )
    optimizer.failure_buffer = FailureBuffer(capacity=10)
    optimizer.failure_buffer.add(_trace("failure", hard=0.0, tokens=100))
    optimizer.utility_computer = UtilityComputer(
        lambda_cost=1e-3,
        rho_omega=0.0,
    )
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {
            "node_id": "a",
            "rank": 1,
            "score": 1.0,
            "reason": "wrong result",
        }
    ]

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
        if scope == "prompt":
            return [prompt_candidate]
        if scope == "operator":
            return [operator_candidate]
        return []

    optimizer.candidate_generator = MagicMock()
    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {}
    counterfactual.evaluate = AsyncMock(
        side_effect=[
            _result(
                [
                    _row(
                        "failure",
                        original_hard=0.0,
                        candidate_hard=0.0,
                        delta_u=0.5,
                        original_tokens=100,
                        candidate_tokens=50,
                        candidate_latency=1.2,
                    )
                ],
                delta_u=0.5,
            ),
            _result(
                [
                    _row(
                        "failure",
                        original_hard=0.0,
                        candidate_hard=0.0,
                        delta_u=0.7,
                        original_tokens=100,
                        candidate_tokens=90,
                        candidate_latency=0.2,
                    )
                ],
                delta_u=0.7,
            ),
        ]
    )
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.candidate_archive = CandidateArchive(capacity=2)
    optimizer.round_history = []
    optimizer._candidate_outcomes = {}

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert updated.nodes["a"].config.system_prompt == "broader"
    assert summary["candidate_scope"] == "operator"
    assert summary["evaluated_candidates"] == 2
    assert summary["exploration_candidates_evaluated"] == 1
    assert [
        attempt["evaluation_budget"]
        for attempt in summary["scope_attempts"]
    ] == ["primary", "exploration"]
    assert len(summary["candidate_archive"]) == 2
