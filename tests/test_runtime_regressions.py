"""Focused regressions for runtime scheduling, tracing, and cost semantics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from awf.config.schema import ExecutorConfig, LLMConfig, LLMProvider, SchedulerConfig
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.llm.client import AsyncLLMClient
from awf.llm.cost_tracker import (
    TokenUsageTracker,
    estimate_cost_usd,
    has_known_pricing,
)
from awf.reward.base import RewardEvaluator
from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.trace.schema import ExecutionTrace, LLMCallRecord, TraceStep
from awf.utility.compute import UtilityComputer
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


class SequenceScheduler(BaseScheduler):
    def __init__(self, *actions, max_actions=None):
        self.actions = list(actions)
        self.index = 0
        if max_actions is not None:
            self.config = SimpleNamespace(max_actions_per_query=max_actions)

    async def initialize(self, workflow, query):
        self.index = 0

    async def select_action(self, workflow, context):
        if self.index >= len(self.actions):
            return SchedulerAction.CONTINUE, {}
        value = self.actions[self.index]
        self.index += 1
        if isinstance(value, tuple):
            return value
        return value, {}


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.config = SimpleNamespace(model="base-model", seed=17)

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return response, {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        }


def node(node_id, node_type, **config):
    return Node(
        node_id=node_id,
        node_type=node_type,
        config=NodeConfig(node_type=node_type, **config),
    )


@pytest.mark.asyncio
async def test_intermediate_outputs_and_per_node_model_reach_api_and_trace():
    workflow = WorkflowTemplate(
        name="chain",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "first": node(
                "first",
                NodeType.LLM,
                prompt_template="{query}",
                model="special-model",
            ),
            "second": node(
                "second",
                NodeType.LLM,
                prompt_template="prior={first_output}",
            ),
            "end": node("end", NodeType.END),
        },
        edges=[
            ("start", "first"),
            ("first", "second"),
            ("second", "end"),
        ],
    )
    client = RecordingClient(["intermediate", "final"])

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=10)
    ).execute(workflow, FixedScheduler(), "question", llm_client=client)

    assert output == "final"
    assert context.variables["first_output"] == "intermediate"
    assert client.calls[0]["model"] == "special-model"
    assert "prior=intermediate" == client.calls[1]["user_prompt"]
    llm_calls = [
        call for step in recorder.trace.steps for call in step.llm_calls
    ]
    assert [call.model for call in llm_calls] == [
        "special-model",
        "base-model",
    ]
    assert llm_calls[0].metadata["usage"]["prompt_tokens"] == 4


@pytest.mark.asyncio
async def test_llm_node_without_client_is_an_explicit_failure():
    workflow = WorkflowTemplate(
        name="missing-client",
        entry_node="answer",
        nodes={"answer": node("answer", NodeType.LLM)},
    )

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=2)
    ).execute(workflow, FixedScheduler(), "q")

    assert output is None
    assert context.success is False
    assert "requires an llm_client" in context.error_message
    assert recorder.trace.success is False
    assert recorder.trace.error_message == context.error_message


@pytest.mark.asyncio
async def test_tools_use_tool_name_templated_args_and_sync_or_async_functions():
    workflow = WorkflowTemplate(
        name="tools",
        entry_node="add_node",
        nodes={
            "add_node": node(
                "add_node",
                NodeType.TOOL,
                tool_name="add",
                tool_args={"x": 2, "y": 3},
            ),
            "double_node": node(
                "double_node",
                NodeType.TOOL,
                tool_name="double",
                tool_args={"value": "{add_node_output}"},
            ),
            "end": node("end", NodeType.END),
        },
        edges=[("add_node", "double_node"), ("double_node", "end")],
    )

    def add(x, y):
        return x + y

    async def double(value):
        await asyncio.sleep(0)
        return value * 2

    output, _, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5),
        operators={"add": add, "double": double},
    ).execute(workflow, FixedScheduler(), "q")

    assert output == 10
    tool_calls = [
        call for step in recorder.trace.steps for call in step.tool_calls
    ]
    assert [call.tool_name for call in tool_calls] == ["add", "double"]
    assert tool_calls[1].tool_args == {"value": 5}
    assert recorder.trace.total_tool_calls == 2


@pytest.mark.asyncio
async def test_timeout_and_action_limit_are_enforced_and_limit_keeps_output():
    workflow = WorkflowTemplate(
        name="loop",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "work": node("work", NodeType.LLM),
        },
        edges=[("start", "work"), ("work", "work")],
    )

    async def slow(context):
        await asyncio.sleep(0.05)
        return "too late"

    _, timed_out, _ = await RuntimeExecutor(
        ExecutorConfig(max_steps=3, timeout_per_step=0.005),
        operators={"work": slow},
    ).execute(
        workflow,
        SequenceScheduler(SchedulerAction.CONTINUE),
        "q",
    )
    assert timed_out.success is False
    assert "timed out" in timed_out.error_message

    async def numbered(context):
        return f"value-{len(context.history) + 1}"

    output, limited, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=10),
        operators={"work": numbered},
    ).execute(
        workflow,
        SequenceScheduler(
            SchedulerAction.CONTINUE,
            SchedulerAction.CONTINUE,
            max_actions=2,
        ),
        "q",
    )
    assert output == "value-2"
    assert limited.success is False
    assert limited.error_message == "Max actions exceeded"
    assert recorder.trace.final_output == "value-2"


@pytest.mark.asyncio
async def test_trace_persistence_flag_does_not_change_in_memory_semantics():
    workflow = WorkflowTemplate(
        name="stop",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "work": node("work", NodeType.LLM),
        },
        edges=[("start", "work"), ("work", "work")],
    )
    client = RecordingClient(["answer"])
    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5, trace_enabled=False)
    ).execute(
        workflow,
        SequenceScheduler(
            SchedulerAction.EXECUTE,
            SchedulerAction.EXECUTE,
            SchedulerAction.STOP,
        ),
        "q",
        llm_client=client,
    )

    assert output == "answer"
    assert context.success is True
    assert [step.action for step in recorder.trace.steps] == [
        "execute",
        "execute",
        "stop",
    ]
    assert recorder.trace.total_llm_calls == 1
    assert recorder.trace.total_prompt_tokens == 4


@pytest.mark.asyncio
async def test_fixed_scheduler_visits_sibling_branches_once_even_with_cycle():
    workflow = WorkflowTemplate(
        name="branched-cycle",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "left": node("left", NodeType.LLM),
            "right": node("right", NodeType.LLM),
            "end": node("end", NodeType.END),
        },
        edges=[
            ("start", "left"),
            ("start", "right"),
            ("left", "right"),
            ("right", "left"),
            ("right", "end"),
        ],
    )

    output, context, _ = await RuntimeExecutor(
        ExecutorConfig(max_steps=10),
        operators={
            "left": lambda context: "left",
            "right": lambda context: "right",
        },
    ).execute(workflow, FixedScheduler(), "q")

    assert output == "right"
    assert context.history == ["start", "left", "right"]


@pytest.mark.asyncio
async def test_targeted_action_trace_uses_actual_execution_input_state():
    workflow = WorkflowTemplate(
        name="targeted",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "verify": node(
                "verify",
                NodeType.TOOL,
                tool_name="verify_tool",
            ),
            "end": node("end", NodeType.END),
        },
        edges=[("start", "verify"), ("verify", "end")],
    )
    scheduler = SequenceScheduler(
        (
            SchedulerAction.VERIFY,
            {"next_unit_id": "verify"},
        )
    )

    _, _, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5),
        operators={"verify_tool": lambda: "verified"},
    ).execute(workflow, scheduler, "q")

    step = recorder.trace.steps[0]
    assert step.node_id == "verify"
    assert step.state_before["current_node_id"] == "verify"
    assert step.metadata["decision_node_id"] == "start"
    assert step.metadata["scheduler_action_validated"] is True


@pytest.mark.asyncio
async def test_continue_ignores_hidden_target_and_follows_normal_edge():
    calls: list[str] = []
    workflow = WorkflowTemplate(
        name="bounded",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "work": node("work", NodeType.TOOL, tool_name="work"),
        },
        edges=[("start", "work")],
    )
    scheduler = SequenceScheduler(
        (
            SchedulerAction.CONTINUE,
            {"next_unit_id": "work"},
        )
    )
    scheduler.config = SimpleNamespace(
        max_actions_per_query=5,
        allow_deviation=False,
    )

    _, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5),
        operators={"work": lambda: calls.append("ran")},
    ).execute(workflow, scheduler, "q")

    assert context.success is True
    assert context.error_message is None
    assert calls == ["ran"]
    assert recorder.trace.steps[0].action == "continue"
    assert recorder.trace.steps[0].node_id == "start"
    assert recorder.trace.steps[0].metadata["node_executed"] is True


def test_reward_cost_and_structural_omega_follow_specification():
    class Evaluator(RewardEvaluator):
        def hard_reward(self, query, ground_truth, output, trace):
            return 1.0

        def process_reward(self, query, ground_truth, output, trace):
            return 0.25

    trace = ExecutionTrace(
        trace_id="t",
        total_prompt_tokens=10,
        total_completion_tokens=5,
        steps=[
            TraceStep(
                step_id="0",
                step_index=0,
                node_id="same",
                action="repair",
            ),
            TraceStep(
                step_id="1",
                step_index=1,
                node_id="same",
                action="reroute",
            ),
            TraceStep(
                step_id="2",
                step_index=2,
                node_id="other",
                action="fallback",
            ),
        ],
    )
    evaluator = Evaluator()
    utility = UtilityComputer()

    assert evaluator.combined_reward("", None, None, trace, alpha=0.4) == 1.1
    assert utility.compute_execution_cost(trace) == 15
    assert utility.compute_runtime_complexity(trace) == 4


def test_context_snapshots_are_deep_copies():
    context = ExecutionContext("q")
    context.record_output("n", {"nested": []})
    snapshot = context.get_state_snapshot()
    context.outputs["n"]["nested"].append("later")
    assert snapshot["outputs"]["n"]["nested"] == []
    context.record_output("n", None)
    assert context.get_last_output() == {"nested": ["later"]}


@pytest.mark.asyncio
async def test_client_seed_model_override_and_provider_validation():
    client = AsyncLLMClient(
        LLMConfig(
            api_key="test",
            model="default",
            seed=77,
            extra_kwargs={"top_p": 0.9},
        )
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
    )
    client._call_with_retry = AsyncMock(return_value=response)

    await client.generate(user_prompt="q", model="override")

    request = client._call_with_retry.await_args.kwargs
    assert request["model"] == "override"
    assert request["seed"] == 77
    assert request["top_p"] == 0.9
    with pytest.raises(ValueError, match="not implemented"):
        LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test",
        )
    with pytest.raises(ValueError, match="api_base"):
        LLMConfig(
            provider=LLMProvider.CUSTOM,
            api_key="test",
        )


@pytest.mark.asyncio
async def test_client_honors_zero_retry_budget():
    client = AsyncLLMClient(
        LLMConfig(api_key="test", max_retries=0)
    )
    request = AsyncMock(side_effect=RuntimeError("connection failed"))
    client._client.chat.completions.create = request

    with pytest.raises(Exception, match="connection failed"):
        await client._call_with_retry(model="m", messages=[])

    assert request.await_count == 1


def test_pricing_recognizes_versioned_models_and_flags_unknown_models():
    assert has_known_pricing("gpt-4o-2024-08-06") is True
    assert has_known_pricing("deepseek-v4-flash") is True
    assert has_known_pricing("deepseek-v4-pro") is True
    assert has_known_pricing("unknown-provider-model") is False


def test_deepseek_v4_cache_aware_pricing():
    tracker = TokenUsageTracker(model="deepseek-v4-flash")
    flash = tracker.record(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_cache_hit_tokens=60,
        prompt_cache_miss_tokens=40,
        reasoning_tokens=12,
    )
    assert flash.cost_usd == pytest.approx(0.000011368)
    assert flash.reasoning_tokens == 12
    assert tracker.snapshot()["prompt_cache_hit_tokens"] == 60
    assert tracker.snapshot()["prompt_cache_miss_tokens"] == 40

    pro = estimate_cost_usd(
        "deepseek-v4-pro",
        prompt_tokens=1000,
        completion_tokens=100,
        prompt_cache_hit_tokens=600,
        prompt_cache_miss_tokens=400,
    )
    assert pro == pytest.approx(0.000263175)
    assert estimate_cost_usd(
        "deepseek-v4-flash",
        prompt_tokens=1000,
        completion_tokens=100,
    ) == pytest.approx(0.000168)


@pytest.mark.asyncio
async def test_deepseek_omits_seed_and_extracts_extended_usage():
    client = AsyncLLMClient(
        LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            api_key="test",
            api_base="https://api.deepseek.com",
            model="deepseek-v4-flash",
            seed=77,
        )
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_cache_hit_tokens=60,
            prompt_cache_miss_tokens=40,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=12,
            ),
        ),
    )
    client._call_with_retry = AsyncMock(return_value=response)

    _, usage = await client.generate(user_prompt="q", seed=99)

    request = client._call_with_retry.await_args.kwargs
    assert "seed" not in request
    assert usage["prompt_cache_hit_tokens"] == 60
    assert usage["prompt_cache_miss_tokens"] == 40
    assert usage["reasoning_tokens"] == 12
    assert usage["cost_usd"] == pytest.approx(0.000011368)
    assert client.get_usage_summary()["reasoning_tokens"] == 12

    workflow = WorkflowTemplate(
        name="deepseek-usage",
        entry_node="answer",
        nodes={"answer": node("answer", NodeType.LLM)},
    )
    _, _, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=2)
    ).execute(workflow, FixedScheduler(), "q", llm_client=client)
    recorded_usage = recorder.trace.steps[0].llm_calls[0].metadata["usage"]
    assert recorded_usage["prompt_cache_hit_tokens"] == 60
    assert recorded_usage["prompt_cache_miss_tokens"] == 40
    assert recorded_usage["reasoning_tokens"] == 12
