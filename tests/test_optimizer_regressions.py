"""Regression tests for workflow optimizer correctness boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from awf.optimizer.acceptance import AcceptanceCriterion
from awf.optimizer.anchor_localizer import AnchorLocalizer
from awf.optimizer.candidate_generator import (
    CandidateGenerator,
    WorkflowCandidate,
)
from awf.optimizer.counterfactual import CounterfactualEvaluator
from awf.optimizer.failure_buffer import FailureBuffer
from awf.optimizer.scorer import CandidateScorer
from awf.optimizer.suffix_replay import SuffixReplayEngine
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.executor.context import ExecutionContext
from awf.config.schema import OptimizerConfig
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace, TraceStep
from awf.utility.compute import UtilityComputer
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.params import OperatorParams


def _trace_with_prefix_failure(
    *,
    failed_prefix: bool = False,
    include_edit_node: bool = True,
) -> ExecutionTrace:
    """Build a trace whose snapshots are sufficient for strict suffix tests."""
    trace = ExecutionTrace(
        trace_id="suffix",
        query_id="suffix",
        query_text="query-suffix",
        workflow_name="wf",
        workflow_version="1.0",
        final_output="wrong",
        success=False,
        metadata={"ground_truth": "truth", "split": "optimization"},
    )
    prefix = ExecutionContext("query-suffix", "wf")
    prefix.current_node_id = "a"
    prefix.record_output("a", "prefix")
    first_snapshot = prefix.get_state_snapshot()
    trace.steps = [
        TraceStep(
            step_id="suffix-a",
            step_index=0,
            node_id="a",
            node_type="llm",
            state_before={},
            state_after=first_snapshot,
            success=not failed_prefix,
            error_message="prefix failed" if failed_prefix else None,
        )
    ]
    if include_edit_node:
        prefix.current_node_id = "b"
        prefix.record_output("b", "wrong")
        trace.steps.append(
            TraceStep(
                step_id="suffix-b",
                step_index=1,
                node_id="b",
                node_type="llm",
                state_before=first_snapshot,
                state_after=prefix.get_state_snapshot(),
                success=False,
                error_message="answer wrong",
            )
        )
    return trace


def make_workflow(version: str = "1.0") -> WorkflowTemplate:
    return WorkflowTemplate(
        name="wf",
        version=version,
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "a": Node(
                node_id="a",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    system_prompt="old system",
                    prompt_template="old {query}",
                ),
            ),
            "b": Node(
                node_id="b",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    system_prompt="check",
                    prompt_template="{query}",
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[
            ("start", "a"),
            ("a", "b"),
            ("b", "end"),
        ],
    )


def make_trace(
    trace_id: str,
    *,
    version: str = "1.0",
    hard: float = 0.0,
    process: float = 0.0,
    output: str = "wrong",
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id,
        query_id=trace_id,
        query_text=f"query-{trace_id}",
        workflow_name="wf",
        workflow_version=version,
        final_output=output,
        success=hard >= 0.5,
        hard_reward=hard,
        process_reward=process,
        metadata={"ground_truth": "truth", "split": "optimization"},
        steps=[
            TraceStep(
                step_id=f"{trace_id}-0",
                step_index=0,
                node_id="a",
                node_type="llm",
                action="execute",
                state_before={"query": trace_id},
                state_after={"outputs": {"a": output}},
                success=hard >= 0.5,
            ),
            TraceStep(
                step_id=f"{trace_id}-1",
                step_index=1,
                node_id="b",
                node_type="llm",
                action="execute",
                state_before={"outputs": {"a": output}},
                state_after={"outputs": {"b": output}},
                success=hard >= 0.5,
            ),
        ],
    )


def mock_generator(
    response: str,
    *,
    max_edit_distance: float = 1.0,
    allowed_execution_models: list[str] | None = None,
) -> CandidateGenerator:
    llm = AsyncMock()
    llm.generate_json.return_value = (response, {})
    return CandidateGenerator(
        llm,
        max_candidates=10,
        max_edit_distance=max_edit_distance,
        allowed_execution_models=allowed_execution_models,
    )


def test_failure_buffer_is_round_local_and_version_aware():
    buffer = FailureBuffer(capacity=20)
    for index in range(5):
        buffer.add(make_trace(f"failure-{index}"))
    buffer.add(make_trace("success-1", hard=1.0))
    buffer.add(make_trace("success-2", hard=1.0))
    buffer.add(make_trace("old", version="0.9"))

    failures, successes, batch = buffer.consume_round("wf", "1.0")

    assert len(failures) == 5
    assert len(successes) == 1
    assert batch == failures + successes
    assert all(trace.workflow_version == "1.0" for trace in batch)
    assert buffer.total_traces == 0

    # Even a no-op optimizer round cannot reuse the preceding baseline.
    assert buffer.consume_round("wf", "1.0") == ([], [], [])


def test_strict_suffix_replay_rejects_a_failed_prefix() -> None:
    workflow = make_workflow()
    checkpoint, reason, step_index = CounterfactualEvaluator._build_suffix_checkpoint(
        _trace_with_prefix_failure(failed_prefix=True),
        "query-suffix",
        workflow,
        "b",
    )
    assert checkpoint is None
    assert reason == "prefix_step_failed:a"
    assert step_index == 1


def test_strict_suffix_replay_requires_the_edit_node_on_the_actual_path() -> None:
    workflow = make_workflow()
    checkpoint, reason, step_index = CounterfactualEvaluator._build_suffix_checkpoint(
        _trace_with_prefix_failure(include_edit_node=False),
        "query-suffix",
        workflow,
        "b",
    )
    assert checkpoint is None
    assert reason == "edit_node_not_observed"
    assert step_index is None


def test_strict_suffix_replay_restores_a_successful_nonterminal_prefix() -> None:
    workflow = make_workflow()
    checkpoint, reason, step_index = CounterfactualEvaluator._build_suffix_checkpoint(
        _trace_with_prefix_failure(),
        "query-suffix",
        workflow,
        "b",
    )
    assert checkpoint is not None
    assert reason is None
    assert step_index == 1
    assert checkpoint.history == ["a"]
    assert checkpoint.get_last_output() == "prefix"


@pytest.mark.asyncio
async def test_anchor_judge_uses_rank_weights_and_validates_executed_units():
    workflow = make_workflow()
    first = make_trace("first", output="bad-one")
    second = make_trace("second", output="bad-two")
    llm = AsyncMock()
    llm.generate_json.side_effect = [
        (
            '{"anchors": ['
            '{"unit_id": "b", "rank": 1, "reason": "bad verification"},'
            '{"unit_id": "a", "rank": 2, "reason": "bad reasoning"},'
            '{"unit_id": "missing", "rank": 3, "reason": "invented"}'
            "]}",
            {},
        ),
        (
            '{"anchors": ['
            '{"unit_id": "a", "rank": 1, "reason": "root cause"},'
            '{"unit_id": "b", "rank": 3, "reason": "secondary"}'
            "]}",
            {},
        ),
    ]
    localizer = AnchorLocalizer(llm)

    anchors = await localizer.localize_batch([first, second], workflow)

    assert [anchor["node_id"] for anchor in anchors] == ["a", "b"]
    assert anchors[0]["score"] == pytest.approx(1.5)
    assert anchors[1]["score"] == pytest.approx(1.25)
    system_prompt = llm.generate_json.call_args_list[0].kwargs["system_prompt"]
    user_prompt = llm.generate_json.call_args_list[0].kwargs["user_prompt"]
    assert "Do not choose an edit scope" in system_prompt
    assert "Final output" in user_prompt
    assert "bad-one" in user_prompt
    assert "output_state" in user_prompt
    assert "Workflow hierarchy" in user_prompt


@pytest.mark.asyncio
async def test_candidate_generation_enforces_anchor_scope_and_metadata():
    generator = mock_generator(
        '{"candidates": ['
        '{"scope": "prompt", "node_id": "b", "description": "unrelated",'
        ' "changes": {"system_prompt": "bad"}},'
        '{"scope": "operator", "node_id": "a", "description": "wrong scope",'
        ' "changes": {"temperature": 0.2}},'
        '{"scope": "prompt", "node_id": "a", "description": "local fix",'
        ' "changes": {"system_prompt": "new system"}}'
        "]}"
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        {"node_id": "a"},
        "prompt",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.anchor_id == "a"
    assert candidate.changed_units == ["a"]
    assert candidate.metadata["changed_units"] == ["a"]
    assert candidate.modified_workflow.nodes["a"].config.system_prompt == (
        "new system"
    )
    assert candidate.modified_workflow.nodes["b"].config.system_prompt == "check"


@pytest.mark.asyncio
async def test_candidate_rejects_unrenderable_prompt_format_fields():
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "prompt",
                        "node_id": "a",
                        "description": "unescaped math braces",
                        "changes": {
                            "user_template": (
                                r"Solve {query}; answer in \boxed{}."
                            )
                        },
                    },
                    {
                        "scope": "prompt",
                        "node_id": "a",
                        "description": "unknown runtime field",
                        "changes": {"user_template": "{missing_value}"},
                    },
                    {
                        "scope": "prompt",
                        "node_id": "a",
                        "description": "escaped literal braces",
                        "changes": {
                            "user_template": (
                                r"Solve {query}; answer in \boxed{{}}."
                            )
                        },
                    },
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "a",
        "prompt",
    )

    assert len(candidates) == 1
    assert candidates[0].description == "escaped literal braces"


@pytest.mark.asyncio
async def test_max_edit_distance_filters_candidates():
    generator = mock_generator(
        '{"candidates": [{'
        '"scope": "prompt", "node_id": "a", "description": "large rewrite",'
        '"changes": {"system_prompt": "completely different instructions"}'
        "}]}",
        max_edit_distance=0.1,
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "a",
        "prompt",
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_candidate_generation_enforces_operator_model_allowlist():
    generator = mock_generator(
        '{"candidates": ['
        '{"scope": "operator", "node_id": "a", "description": "forbidden",'
        ' "changes": {"model": "other-model"}},'
        '{"scope": "operator", "node_id": "a", "description": "allowed",'
        ' "changes": {"model": "deepseek-v4-flash"}}'
        "]}",
        allowed_execution_models=["deepseek-v4-flash"],
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "a",
        "operator",
    )

    assert len(candidates) == 1
    assert candidates[0].changes["model"] == "deepseek-v4-flash"
    assert (
        candidates[0].modified_workflow.nodes["a"].config.model
        == "deepseek-v4-flash"
    )
    system_prompt = generator.llm.generate_json.call_args.kwargs[
        "system_prompt"
    ]
    assert "deepseek-v4-flash" in system_prompt
    assert "add_node.config.model" in system_prompt
    assert "MUST exactly match" in system_prompt


@pytest.mark.asyncio
async def test_candidate_generation_enforces_added_node_model_allowlist():
    def add_node(model: str) -> dict:
        return {
            "scope": "block",
            "node_id": "a",
            "description": f"add {model}",
            "changes": {
                "add_node": {
                    "node_id": f"repair-{model}",
                    "node_type": "llm",
                    "after": "a",
                    "before": "b",
                    "config": {
                        "model": model,
                        "prompt_template": "repair {query}",
                    },
                }
            },
        }

    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    add_node("other-model"),
                    add_node("deepseek-v4-flash"),
                ]
            }
        ),
        allowed_execution_models=["deepseek-v4-flash"],
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "a",
        "block",
    )

    assert len(candidates) == 1
    added = candidates[0].modified_workflow.nodes[
        "repair-deepseek-v4-flash"
    ]
    assert added.config.model == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_empty_execution_model_allowlist_preserves_unrestricted_edits():
    generator = mock_generator(
        '{"candidates": [{'
        '"scope": "operator", "node_id": "a", "description": "switch",'
        '"changes": {"model": "any-model"}'
        "}]}"
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "a",
        "operator",
    )

    assert len(candidates) == 1
    assert candidates[0].modified_workflow.nodes["a"].config.model == (
        "any-model"
    )


def test_optimizer_passes_execution_model_allowlist_to_generator():
    optimizer = LLMWorkflowOptimizer(
        config=OptimizerConfig(
            allowed_execution_models=["deepseek-v4-flash"],
            lambda_latency=0.25,
            lambda_api_cost=3.0,
            llm={"api_key": "test"},
        ),
        reward_evaluator=CountingReward(),
    )

    assert optimizer.candidate_generator.allowed_execution_models == {
        "deepseek-v4-flash"
    }
    assert optimizer.utility_computer.lambda_latency == 0.25
    assert optimizer.utility_computer.lambda_api_cost == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "expected_edges", "missing_node"),
    [
        (
            {"remove_node": "b"},
            {("start", "a"), ("a", "end")},
            "b",
        ),
        (
            {
                "add_node": {
                    "node_id": "repair",
                    "node_type": "llm",
                    "after": "a",
                    "before": "b",
                    "config": {"prompt_template": "repair {query}"},
                }
            },
            {
                ("start", "a"),
                ("a", "repair"),
                ("repair", "b"),
                ("b", "end"),
            },
            None,
        ),
        (
            {"reorder": ["b", "a"]},
            {("start", "b"), ("b", "a"), ("a", "end")},
            None,
        ),
    ],
)
async def test_block_graph_edits_reconnect_and_validate(
    changes,
    expected_edges,
    missing_node,
):
    anchor = "b" if "remove_node" in changes else "a"
    generator = mock_generator(
        '{"candidates": [{'
        f'"scope": "block", "node_id": "{anchor}", '
        '"description": "local graph repair", '
        f'"changes": {json.dumps(changes)}'
        "}]}"
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        anchor,
        "block",
    )

    assert len(candidates) == 1
    modified = candidates[0].modified_workflow
    assert set(modified.edges) == expected_edges
    if missing_node:
        assert missing_node not in modified.nodes
    assert modified.entry_node == "start"
    assert set(modified.get_node_order()) == set(modified.nodes)


@pytest.mark.asyncio
async def test_atomic_graph_patch_builds_multi_node_repair_chain():
    patch = {
        "add_nodes": [
            {
                "node_id": "diagnose",
                "node_type": "llm",
                "config": {
                    "system_prompt": "Diagnose the failed verification.",
                    "prompt_template": "{b_output}",
                },
            },
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {
                    "system_prompt": "Repair the diagnosed defect.",
                    "prompt_template": "{diagnose_output}",
                },
            },
            {
                "node_id": "reverify",
                "node_type": "llm",
                "config": {
                    "system_prompt": "Verify the repaired answer.",
                    "prompt_template": "{repair_output}",
                },
            },
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [
            ["b", "diagnose"],
            ["diagnose", "repair"],
            ["repair", "reverify"],
            ["reverify", "end"],
        ],
    }
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "block",
                        "node_id": "b",
                        "description": "diagnose, repair, then reverify",
                        "changes": {"graph_patch": patch},
                    }
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "b",
        "block",
    )

    assert len(candidates) == 1
    modified = candidates[0].modified_workflow
    assert set(modified.nodes) == {
        "start",
        "a",
        "b",
        "diagnose",
        "repair",
        "reverify",
        "end",
    }
    assert set(modified.edges) == {
        ("start", "a"),
        ("a", "b"),
        ("b", "diagnose"),
        ("diagnose", "repair"),
        ("repair", "reverify"),
        ("reverify", "end"),
    }
    assert candidates[0].changed_units == [
        "b",
        "diagnose",
        "end",
        "repair",
        "reverify",
    ]
    assert candidates[0].edit_distance > 0.4
    assert "graph_patch" in generator.llm.generate_json.call_args.kwargs[
        "system_prompt"
    ]


@pytest.mark.asyncio
async def test_atomic_graph_patch_rejects_whole_invalid_transaction():
    workflow = make_workflow()
    before = workflow.model_dump(mode="python")
    patch = {
        "add_nodes": [
            {
                "node_id": "gate",
                "node_type": "condition",
                "config": {"condition_expr": "success"},
            }
        ],
        "remove_edges": [["b", "end"]],
        # A condition requires two successors. The final IR validation rejects
        # the complete transaction instead of retaining the added node.
        "add_edges": [["b", "gate"], ["gate", "end"]],
    }
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "block",
                        "node_id": "b",
                        "description": "incomplete conditional patch",
                        "changes": {"graph_patch": patch},
                    }
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        workflow,
        "b",
        "block",
    )

    assert candidates == []
    assert workflow.model_dump(mode="python") == before
    assert "gate" not in workflow.nodes
    assert ("b", "end") in workflow.edges


@pytest.mark.asyncio
async def test_atomic_graph_patch_rejects_unavailable_input_dependency():
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                # END is downstream, so this variable cannot exist when the
                # repair node renders its prompt.
                "config": {"prompt_template": "{end_output}"},
            }
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [["b", "repair"], ["repair", "end"]],
    }
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "block",
                        "node_id": "b",
                        "description": "read a downstream output too early",
                        "changes": {"graph_patch": patch},
                    }
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        make_workflow(),
        "b",
        "block",
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_atomic_graph_patch_preserves_declared_output_contract():
    workflow = make_workflow()
    workflow.metadata["output_contract"] = {
        "terminal_producers": [
            {"node_id": "b", "node_type": "llm"},
        ]
    }
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {"prompt_template": "{b_output}"},
            }
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [["b", "repair"], ["repair", "end"]],
    }
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "block",
                        "node_id": "b",
                        "description": "replace the declared final producer",
                        "changes": {"graph_patch": patch},
                    }
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        workflow,
        "b",
        "block",
    )

    assert candidates == []
    assert workflow.get_successors("b") == ["end"]


@pytest.mark.asyncio
async def test_atomic_graph_patch_rejects_cross_block_edge_edit():
    workflow = make_workflow()
    workflow.parameters.set_operator(
        "stage",
        "local",
        OperatorParams(node_id="a"),
    )
    workflow.parameters.set_operator(
        "stage",
        "local",
        OperatorParams(node_id="b"),
    )
    generator = mock_generator(
        json.dumps(
            {
                "candidates": [
                    {
                        "scope": "block",
                        "node_id": "b",
                        "description": "mutate an edge outside the local block",
                        "changes": {
                            "graph_patch": {
                                "remove_edges": [["b", "end"]],
                                "add_edges": [["a", "end"]],
                            }
                        },
                    }
                ]
            }
        )
    )

    candidates = await generator.generate_for_anchor(
        workflow,
        "b",
        "block",
    )

    assert candidates == []
    assert workflow.edges == [
        ("start", "a"),
        ("a", "b"),
        ("b", "end"),
    ]


class CountingReward(RewardEvaluator):
    def __init__(self):
        self.hard_calls = 0
        self.process_calls = 0

    def hard_reward(self, query, ground_truth, output, trace):
        self.hard_calls += 1
        return 1.0 if output == "candidate" else 0.0

    def process_reward(self, query, ground_truth, output, trace):
        self.process_calls += 1
        return 0.25


class RecordingExecutor:
    def __init__(self):
        self.calls = 0
        self.clients = []
        self.counterfactual_modes = []

    async def execute(
        self,
        workflow,
        scheduler,
        query,
        llm_client=None,
        counterfactual=False,
    ):
        self.calls += 1
        self.clients.append(llm_client)
        self.counterfactual_modes.append(counterfactual)
        trace = ExecutionTrace(
            trace_id=f"new-{self.calls}",
            query_text=query,
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            success=True,
            final_output="candidate",
            total_prompt_tokens=3,
            total_completion_tokens=2,
            total_llm_calls=1,
            total_latency_seconds=0.25,
            steps=[
                TraceStep(
                    step_id="candidate-0",
                    step_index=0,
                    node_id="b",
                    action="repair",
                    metadata={"node_executed": True},
                )
            ],
        )
        return "candidate", SimpleNamespace(), SimpleNamespace(trace=trace)


@pytest.mark.asyncio
async def test_counterfactual_passes_llm_client_reuses_baseline_and_cache():
    reward = CountingReward()
    evaluator = CounterfactualEvaluator(
        reward,
        UtilityComputer(lambda_cost=0.0, rho_omega=0.0),
        reward_alpha=0.8,
    )
    original = make_trace("baseline", process=0.5)
    original.total_prompt_tokens = 10
    original.total_completion_tokens = 4
    original.total_llm_calls = 2
    original.total_latency_seconds = 0.75
    baseline = evaluator.prepare_baseline([original])
    executor = RecordingExecutor()
    scheduler = object()
    llm_client = object()

    first = await evaluator.evaluate(
        make_workflow(),
        [original],
        executor,
        scheduler,
        llm_client=llm_client,
        baseline=baseline,
    )
    second = await evaluator.evaluate(
        make_workflow(),
        [original],
        executor,
        scheduler,
        llm_client=llm_client,
        baseline=baseline,
    )

    # Utility compares hard reward only: original hard=0, candidate hard=1.
    # Process/composite reward remains available in the per-query diagnostics.
    assert first["delta_u"] == pytest.approx(1.0)
    assert executor.clients == [llm_client]
    assert executor.counterfactual_modes == [True]
    assert executor.calls == 1
    assert reward.hard_calls == 1
    assert reward.process_calls == 1
    assert second["cache_hit"] is True
    query_result = first["per_query_results"][0]
    assert query_result["original_total_tokens"] == 14
    assert query_result["candidate_total_tokens"] == 5
    assert query_result["original_llm_calls"] == 2
    assert query_result["candidate_llm_calls"] == 1
    assert query_result["original_latency_seconds"] == pytest.approx(0.75)
    assert query_result["candidate_latency_seconds"] == pytest.approx(0.25)
    assert query_result["original_path"] == ["a", "b"]
    assert query_result["candidate_path"] == ["b"]
    assert query_result["original_actions"] == ["execute", "execute"]
    assert query_result["candidate_actions"] == ["repair"]
    assert query_result["original_error"] is None
    assert query_result["candidate_error"] is None


@pytest.mark.asyncio
async def test_suffix_replay_request_is_explicit_full_rerun_fallback():
    evaluator = CounterfactualEvaluator(
        CountingReward(),
        UtilityComputer(lambda_cost=0.0, rho_omega=0.0),
        suffix_replay=SuffixReplayEngine(),
    )
    trace = make_trace("baseline", process=0.5)
    result = await evaluator.evaluate(
        make_workflow(),
        [trace],
        RecordingExecutor(),
        object(),
        baseline=evaluator.prepare_baseline([trace]),
    )

    assert result["evaluation_mode"] == "full_rerun"
    assert result["suffix_replay_requested"] is True
    assert result["suffix_replay_used"] is False
    assert "could not be used" in result["fallback_reason"]


def test_counterfactual_cache_key_includes_replay_edit_node() -> None:
    workflow = make_workflow()
    key_a = CounterfactualEvaluator._cache_key(
        workflow,
        "batch",
        object(),
        object(),
        edit_node_id="a",
    )
    key_b = CounterfactualEvaluator._cache_key(
        workflow,
        "batch",
        object(),
        object(),
        edit_node_id="b",
    )
    assert key_a != key_b


@pytest.mark.asyncio
async def test_optimizer_consumes_current_round_and_stops_after_prompt_success():
    workflow = make_workflow()
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(candidates_per_round=5)
    optimizer.failure_buffer = FailureBuffer(capacity=20)
    for index in range(5):
        optimizer.failure_buffer.add(make_trace(f"failure-{index}"))
    optimizer.failure_buffer.add(make_trace("guard", hard=1.0))
    optimizer.failure_buffer.add(make_trace("stale", version="0.9"))

    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {"node_id": "a", "rank": 1, "score": 1.0}
    ]
    modified = workflow.model_copy(deep=True)
    modified.nodes["a"].config.system_prompt = "fixed"
    prompt_candidate = WorkflowCandidate(
        scope="prompt",
        node_id="a",
        anchor_id="a",
        description="fix prompt",
        changes={"system_prompt": "fixed"},
        modified_workflow=modified,
        edit_distance=0.1,
        changed_units=["a"],
    )
    optimizer.candidate_generator = AsyncMock()
    optimizer.candidate_generator.generate_for_anchor.return_value = [
        prompt_candidate
    ]

    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {"baseline": True}
    counterfactual.evaluate = AsyncMock(
        return_value={
            "delta_u": 0.5,
            "evaluation_mode": "full_rerun",
            "cache_hit": False,
            "per_query_results": [
                {
                    "trace_id": "failure-0",
                    "delta_u": 0.5,
                }
            ],
        }
    )
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.round_history = []
    runtime_client = object()

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
        llm_client=runtime_client,
    )

    assert updated.version == "1.1"
    assert updated.nodes["a"].config.system_prompt == "fixed"
    assert summary["accepted"] is True
    assert summary["num_failures"] == 5
    assert summary["num_successes"] == 1
    assert summary["num_success_guards"] == 1
    assert summary["counterfactual_batch_size"] == 6
    assert [item["scope"] for item in summary["scope_attempts"]] == ["prompt"]
    assert optimizer.failure_buffer.total_traces == 0
    assert counterfactual.evaluate.call_args.kwargs["llm_client"] is runtime_client
    assert summary["candidate_evaluations"] == [
        {
            "candidate_index": 0,
            "anchor_id": "a",
            "scope": "prompt",
            "node_id": "a",
            "description": "fix prompt",
            "changes": {"system_prompt": "fixed"},
            "patch_fingerprint": (
                CandidateGenerator.candidate_patch_fingerprint(
                    prompt_candidate
                )
            ),
            "changed_units": ["a"],
            "edit_distance": 0.1,
            "delta_u": 0.5,
            "token_delta": 0.0,
            "token_penalty": 0.0,
            "edit_penalty": pytest.approx(0.01),
            "gain": pytest.approx(0.49),
            "status": "selected",
            "per_query_results": [
                {
                    "trace_id": "failure-0",
                    "delta_u": 0.5,
                }
            ],
            "final_selected": True,
            "evaluation_mode": "full_rerun",
            "cache_hit": False,
        }
    ]


@pytest.mark.asyncio
async def test_failure_repair_searches_all_prompts_before_operator_or_block():
    workflow = make_workflow()
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        candidates_per_round=8,
        failure_cluster_enabled=True,
        failure_cluster_min_coverage=0.0,
        failure_repair_acceptance="any_repair",
        allow_failure_full_rerun=True,
        failure_repair_confirmation_enabled=False,
        exploration_acceptance_floor=-1.0,
        hard_success_threshold=1.0,
    )
    optimizer.workflow_content_only = True
    optimizer._require_failure_suffix_replay = True
    optimizer._candidate_scopes = ("prompt", "operator", "block", "multi")
    optimizer.failure_buffer = FailureBuffer(capacity=10)
    optimizer.failure_buffer.add(make_trace("failure"))
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {"node_id": "a", "rank": 1, "score": 1.0},
        {"node_id": "b", "rank": 2, "score": 0.8},
    ]

    modified = workflow.model_copy(deep=True)
    modified.nodes["b"].config.system_prompt = "repair the failure"
    prompt_candidate = WorkflowCandidate(
        scope="prompt",
        node_id="b",
        anchor_id="b",
        description="prompt repair at the second anchor",
        changes={"system_prompt": "repair the failure"},
        modified_workflow=modified,
        edit_distance=0.1,
        changed_units=["b"],
    )
    generated: list[tuple[str, str]] = []

    async def generate_for_anchor(
        workflow,
        anchor,
        scope,
        context,
        *,
        limit,
        **kwargs,
    ):
        del workflow, context, limit, kwargs
        generated.append((anchor["node_id"], scope))
        if anchor["node_id"] == "b" and scope == "prompt":
            return [prompt_candidate]
        return []

    optimizer.candidate_generator = MagicMock()
    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "suffix_replay"
    counterfactual.suffix_replay = object()
    counterfactual.prepare_baseline.return_value = {}
    counterfactual.evaluate = AsyncMock(
        return_value={
            "delta_u": 1.0,
            "evaluation_mode": "suffix_replay",
            "suffix_replay_used": True,
            "cache_hit": False,
            "per_query_results": [
                {
                    "trace_id": "failure",
                    "original_u": 0.0,
                    "candidate_u": 1.0,
                    "delta_u": 1.0,
                    "candidate_hard": 1.0,
                    "candidate_success": True,
                    "candidate_suffix_replay_used": True,
                }
            ],
        }
    )
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.0)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.0)
    optimizer.round_history = []

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert updated.nodes["b"].config.system_prompt == "repair the failure"
    assert generated == [("a", "prompt"), ("b", "prompt")]
    assert summary["candidate_scope"] == "prompt"
    assert summary["scope_first_stop"] == {
        "stopped_before_scope": "operator",
        "selected_scope": "prompt",
        "reason": "lower_level_failure_repair_found",
    }


@pytest.mark.asyncio
async def test_candidate_repair_confirmation_repeats_only_failure_targets():
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        failure_repair_confirmation_enabled=True,
        failure_repair_confirmation_repeats=2,
        failure_repair_confirmation_min_successes=2,
        hard_success_threshold=1.0,
    )
    failure = make_trace("failure")
    result = {
        "delta_u": 1.0,
        "candidate_u": 1.0,
        "per_query_results": [
            {
                "trace_id": "failure",
                "original_u": 0.0,
                "candidate_u": 1.0,
                "delta_u": 1.0,
                "candidate_hard": 1.0,
                "candidate_success": True,
                "candidate_evaluation_mode": "suffix_replay",
            },
            {
                "trace_id": "guard",
                "original_u": 1.0,
                "candidate_u": 1.0,
                "delta_u": 0.0,
                "candidate_hard": 1.0,
                "candidate_success": True,
            },
        ],
    }
    optimizer.counterfactual = MagicMock()
    optimizer.counterfactual.prepare_baseline.return_value = {"baseline": True}
    optimizer.counterfactual.evaluate = AsyncMock(
        side_effect=[
            {
                "per_query_results": [
                    {
                        "trace_id": "failure",
                        "candidate_hard": 0.0,
                        "candidate_success": True,
                        "candidate_evaluation_mode": "full_rerun",
                    }
                ]
            },
            {
                "per_query_results": [
                    {
                        "trace_id": "failure",
                        "candidate_hard": 1.0,
                        "candidate_success": True,
                        "candidate_evaluation_mode": "full_rerun",
                    }
                ]
            },
        ]
    )

    confirmed = await optimizer._confirm_candidate_failure_repairs(
        result,
        make_workflow(),
        [failure],
        object(),
        object(),
        llm_client=object(),
        edit_node_id="b",
    )

    failure_row, guard_row = confirmed["per_query_results"]
    assert failure_row["candidate_hard"] == 1.0
    assert failure_row["repair_confirmation"]["successes"] == 2
    assert failure_row["repair_confirmation"]["total_runs"] == 3
    assert guard_row.get("repair_confirmation") is None
    assert optimizer.counterfactual.evaluate.await_count == 2
    for call in optimizer.counterfactual.evaluate.await_args_list:
        assert call.args[1] == [failure]
        assert call.kwargs["use_cache"] is False


@pytest.mark.asyncio
async def test_optimizer_escalates_scope_only_after_smaller_edit_fails():
    workflow = make_workflow()
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(candidates_per_round=5)
    optimizer.failure_buffer = FailureBuffer(capacity=10)
    optimizer.failure_buffer.add(make_trace("failure"))
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {"node_id": "a", "rank": 1, "score": 1.0}
    ]

    prompt_workflow = workflow.model_copy(deep=True)
    prompt_workflow.nodes["a"].config.system_prompt = "weak fix"
    operator_workflow = workflow.model_copy(deep=True)
    operator_workflow.nodes["a"].config.temperature = 0.1
    candidates = {
        "prompt": [
            WorkflowCandidate(
                scope="prompt",
                node_id="a",
                description="weak",
                changes={"system_prompt": "weak fix"},
                modified_workflow=prompt_workflow,
                edit_distance=0.1,
                changed_units=["a"],
            )
        ],
        "operator": [
            WorkflowCandidate(
                scope="operator",
                node_id="a",
                description="strong",
                changes={"temperature": 0.1},
                modified_workflow=operator_workflow,
                edit_distance=0.3,
                changed_units=["a"],
            )
        ],
    }

    async def generate_for_anchor(workflow, anchor, scope, context, limit):
        return candidates.get(scope, [])

    optimizer.candidate_generator = MagicMock()
    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {}
    counterfactual.evaluate = AsyncMock(
        side_effect=[
            {"delta_u": 0.0, "evaluation_mode": "full_rerun", "cache_hit": False},
            {"delta_u": 0.5, "evaluation_mode": "full_rerun", "cache_hit": False},
        ]
    )
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.round_history = []

    updated, summary = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert updated.nodes["a"].config.temperature == 0.1
    assert summary["candidate_scope"] == "operator"
    assert [attempt["scope"] for attempt in summary["scope_attempts"]] == [
        "prompt",
        "operator",
    ]


def test_failure_context_does_not_assume_trace_level_error_message():
    optimizer = object.__new__(LLMWorkflowOptimizer)
    trace = make_trace("failed")
    trace.steps[-1].error_message = "node failed"

    context = optimizer._build_failure_context([trace])

    assert "node failed" in context
    assert "Final output" in context


def test_failure_buffer_fraction_does_not_force_a_fifty_fifty_tiny_batch():
    buffer = FailureBuffer(capacity=10)
    buffer.add(make_trace("failure"))
    for index in range(3):
        buffer.add(make_trace(f"success-{index}", hard=1.0))

    failures, guards, batch = buffer.consume_round(
        "wf",
        "1.0",
        success_fraction=0.2,
    )

    assert len(failures) == 1
    assert guards == []
    assert batch == failures

    explicit = FailureBuffer(capacity=10)
    explicit.add(make_trace("failure"))
    for index in range(3):
        explicit.add(make_trace(f"success-{index}", hard=1.0))
    failures, guards, batch = explicit.consume_round(
        "wf",
        "1.0",
        success_fraction=0.2,
        min_success_guards=1,
    )

    assert len(failures) == 1
    assert len(guards) == 1
    assert batch == failures + guards


def test_failure_context_includes_safe_code_diagnostics_without_raw_tests():
    optimizer = object.__new__(LLMWorkflowOptimizer)
    trace = make_trace("code-failure")
    trace.metadata.update(
        {
            "split": "optimization",
            "ground_truth": {
                "task_id": "HumanEval/7",
                "entry_point": "is_nested",
                "canonical_solution": "REFERENCE_SOLUTION_MUST_NOT_LEAK",
                "test": "RAW_HUMANEVAL_TEST_SOURCE_MUST_NOT_LEAK",
            },
            "code_evaluation": {
                "passed_tests": 0,
                "total_tests": 1,
                "tests_executed": 1,
                "pass_rate": 0.0,
                "summary": (
                    "0/1 tests passed: AssertionError in edge-case check\n"
                    "RAW_HUMANEVAL_TEST_SOURCE_MUST_NOT_LEAK"
                ),
            },
        }
    )

    context = optimizer._build_failure_context([trace])

    assert "task_failure" in context
    assert "AssertionError in edge-case check" in context
    assert "HumanEval/7" in context
    assert "is_nested" in context
    assert "withheld_test_suite" in context
    assert "reference_solution" in context
    assert "RAW_HUMANEVAL_TEST_SOURCE_MUST_NOT_LEAK" not in context
    assert "REFERENCE_SOLUTION_MUST_NOT_LEAK" not in context


def test_failure_context_explains_invalid_code_output_contract():
    optimizer = object.__new__(LLMWorkflowOptimizer)
    trace = make_trace("invalid-code")
    trace.metadata.update(
        {
            "ground_truth": {
                "task_id": "HumanEval/3",
                "entry_point": "below_zero",
                "test": "WITHHELD_TEST_SOURCE",
            },
            "code_output_contract": {
                "source": "final_output",
                "entry_point": "below_zero",
                "parseable": True,
                "entry_point_present": False,
                "valid": False,
                "reason": (
                    "final code does not define entry point 'below_zero'"
                ),
            },
        }
    )

    context = optimizer._build_failure_context([trace])

    assert "output_contract_failure" in context
    assert "entry_point_present" in context
    assert "final code does not define entry point" in context
    assert "WITHHELD_TEST_SOURCE" not in context


@pytest.mark.parametrize(
    "split",
    [None, "validation", "validation_baseline", "test", "official_test"],
)
def test_failure_context_rejects_non_optimization_split(split):
    optimizer = object.__new__(LLMWorkflowOptimizer)
    trace = make_trace("held-out")
    if split is None:
        trace.metadata.pop("split")
    else:
        trace.metadata["split"] = split
    trace.query_text = "HELD_OUT_QUERY_MUST_NOT_REACH_OPTIMIZER"

    with pytest.raises(ValueError, match="only optimization-split"):
        optimizer._build_failure_context([trace])


@pytest.mark.asyncio
async def test_candidate_prompt_contains_anchor_evidence_and_effective_defaults():
    llm = AsyncMock()
    llm.generate_json.return_value = (
        '{"candidates": [{'
        '"scope": "operator", "node_id": "a",'
        '"description": "raise temperature for exploration",'
        '"changes": {"temperature": 0.2}'
        "}]}",
        {},
    )
    generator = CandidateGenerator(
        llm,
        execution_llm_defaults={
            "model": "deepseek-v4-flash",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    )

    await generator.generate_for_anchor(
        make_workflow(),
        {
            "node_id": "a",
            "rank": 1,
            "score": 1.5,
            "evidence_count": 2,
            "reason": "generation confused subsequence with substring",
        },
        "operator",
    )

    call = llm.generate_json.call_args.kwargs
    user_prompt = call["user_prompt"]
    assert "deepseek-v4-flash" in user_prompt
    assert "temperature=0.0" in user_prompt
    assert "max_tokens=2048" in user_prompt
    assert "generation confused subsequence with substring" in user_prompt
    assert '"evidence_count": 2' in user_prompt
    assert '"score": 1.5' in user_prompt
    assert "never describe 0.2 as lowering" in call["system_prompt"]


def test_experience_context_uses_only_optimization_outcome_aggregates():
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.round_history = [
        {
            "round": 1,
            "is_best": False,
            "optimization_confirmation": {
                "performed": True,
                "passed": False,
                "decision": "rejected",
                "reason": "mean_utility_not_improved",
                "completed_repeats": 2,
                "hard_reward_effect": 0.0,
                "mean_utility_effect": -0.05,
                "repeat_metrics": [
                    {
                        "private_rows": (
                            "OPT_CONFIRMATION_ROWS_MUST_NOT_LEAK"
                        )
                    }
                ],
            },
            "candidate_evaluations": [
                {
                    "status": "selected",
                    "patch_fingerprint": "a" * 64,
                    "anchor_id": "generate",
                    "scope": "prompt",
                    "gain": -0.1,
                    "description": "generic edge-case reminder",
                    "per_query_results": [
                        {
                            "query": "VALIDATION_OR_TEST_QUERY_MUST_NOT_LEAK",
                            "candidate_output": (
                                "VALIDATION_OR_TEST_OUTPUT_MUST_NOT_LEAK"
                            ),
                            "original_hard": 0.0,
                            "candidate_hard": 1.0,
                            "original_total_tokens": 100,
                            "candidate_total_tokens": 120,
                            "original_latency_seconds": 1.0,
                            "candidate_latency_seconds": 1.5,
                        }
                    ],
                }
            ],
            "validation_metrics": {
                "num_examples": 4,
                "hard_reward": 0.75,
                "runtime_utility": 0.7,
                "total_tokens": 1234,
                "private_rows": "VALIDATION_ROWS_MUST_NOT_LEAK",
            },
        }
    ]

    context = optimizer._build_experience_context()

    assert "generic edge-case reminder" in context
    assert "negative_full_opt_confirmation" in context
    assert "mean_utility_not_improved" in context
    assert "hard_fixes" in context
    assert "mean_token_delta" in context
    assert "Previous aggregate validation outcome" not in context
    assert "runtime_utility" not in context
    assert "1234" not in context
    assert "VALIDATION_OR_TEST_QUERY_MUST_NOT_LEAK" not in context
    assert "VALIDATION_OR_TEST_OUTPUT_MUST_NOT_LEAK" not in context
    assert "VALIDATION_ROWS_MUST_NOT_LEAK" not in context
    assert "OPT_CONFIRMATION_ROWS_MUST_NOT_LEAK" not in context


@pytest.mark.asyncio
async def test_optimizer_deduplicates_exact_patch_across_rounds():
    workflow = make_workflow()
    optimizer = object.__new__(LLMWorkflowOptimizer)
    optimizer.config = SimpleNamespace(
        candidates_per_round=1,
        success_guard_fraction=0.2,
        min_success_guards=0,
    )
    optimizer.failure_buffer = FailureBuffer(capacity=10)
    optimizer.failure_buffer.add(make_trace("round-1-failure"))
    optimizer.anchor_localizer = AsyncMock()
    optimizer.anchor_localizer.localize_batch.return_value = [
        {
            "node_id": "a",
            "rank": 1,
            "score": 1.0,
            "reason": "bad answer",
        }
    ]
    modified = workflow.model_copy(deep=True)
    modified.nodes["a"].config.system_prompt = "same rejected patch"
    candidate = WorkflowCandidate(
        scope="prompt",
        node_id="a",
        description="same rejected patch",
        changes={"system_prompt": "same rejected patch"},
        modified_workflow=modified,
        edit_distance=0.1,
        changed_units=["a"],
    )
    experience_prompts = []

    async def generate_for_anchor(
        workflow,
        anchor,
        scope,
        context,
        *,
        limit,
        experience_context="",
    ):
        del workflow, anchor, context, limit
        experience_prompts.append(experience_context)
        return [candidate] if scope == "prompt" else []

    optimizer.candidate_generator = MagicMock()
    optimizer.candidate_generator.generate_for_anchor = generate_for_anchor
    counterfactual = MagicMock()
    counterfactual.evaluation_mode = "full_rerun"
    counterfactual.suffix_replay = None
    counterfactual.prepare_baseline.return_value = {}
    counterfactual.evaluate = AsyncMock(
        return_value={
            "delta_u": 0.0,
            "evaluation_mode": "full_rerun",
            "cache_hit": False,
            "per_query_results": [],
        }
    )
    optimizer.counterfactual = counterfactual
    optimizer.scorer = CandidateScorer(mu_edit=0.1)
    optimizer.acceptance = AcceptanceCriterion(epsilon_stat=0.01)
    optimizer.round_history = []
    optimizer._candidate_outcomes = {}

    _, first = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )
    optimizer.failure_buffer.add(make_trace("round-2-failure"))
    _, second = await optimizer.optimize_round(
        workflow,
        object(),
        object(),
    )

    assert first["evaluated_candidates"] == 1
    assert second["evaluated_candidates"] == 0
    assert second["duplicate_candidates"] == 1
    assert second["candidate_evaluations"][0]["status"] == (
        "duplicate_history"
    )
    assert second["candidate_evaluations"][0]["duplicate_of_round"] == 1
    assert counterfactual.evaluate.await_count == 1
    assert any("negative_or_failed" in prompt for prompt in experience_prompts)
