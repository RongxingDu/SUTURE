from __future__ import annotations

import pytest

from awf.config.schema import ExecutorConfig
from awf.executor.conditions import (
    ConditionExpressionError,
    evaluate_condition_expression,
)
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.executor.safety import counterfactual_safe
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _condition_workflow(expression: str) -> WorkflowTemplate:
    return WorkflowTemplate(
        name="condition-test",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "condition": Node(
                node_id="condition",
                node_type=NodeType.CONDITION,
                config=NodeConfig(
                    node_type=NodeType.CONDITION,
                    condition_expr=expression,
                ),
            ),
            "true_path": Node(
                node_id="true_path",
                node_type=NodeType.JOIN,
            ),
            "false_path": Node(
                node_id="false_path",
                node_type=NodeType.JOIN,
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[
            ("start", "condition"),
            ("condition", "true_path"),
            ("condition", "false_path"),
            ("true_path", "end"),
            ("false_path", "end"),
        ],
    )


def test_restricted_condition_supports_state_lookups():
    context = ExecutionContext("query")
    context.record_output("start", {"status": "started"})

    assert evaluate_condition_expression(
        'outputs.get("start", {}).get("status") == "started" '
        "and context.step_count == 0",
        context,
    )


def test_restricted_condition_rejects_python_object_traversal():
    context = ExecutionContext("query")

    with pytest.raises(ConditionExpressionError):
        evaluate_condition_expression(
            "().__class__.__base__.__subclasses__()",
            context,
        )


@pytest.mark.asyncio
async def test_invalid_condition_fails_closed_in_executor():
    workflow = _condition_workflow(
        "().__class__.__base__.__subclasses__()"
    )
    executor = RuntimeExecutor(ExecutorConfig(max_steps=10))

    output, context, recorder = await executor.execute(
        workflow,
        FixedScheduler(),
        "query",
    )

    assert output is None
    assert context.success is False
    assert "Invalid condition expression" in (context.error_message or "")
    assert recorder.trace.success is False


def test_trace_step_exposes_step_level_token_and_cost_totals():
    from awf.trace.recorder import TraceRecorder
    from awf.trace.schema import LLMCallRecord, ToolCallRecord

    recorder = TraceRecorder()
    step_id = recorder.record_step_start("node")
    recorder.record_llm_call(
        step_id,
        LLMCallRecord(
            call_id="llm",
            prompt_tokens=3,
            completion_tokens=4,
            cost_usd=0.25,
        ),
    )
    recorder.record_tool_call(
        step_id,
        ToolCallRecord(
            tool_call_id="tool",
            tool_name="lookup",
            cost_usd=0.5,
        ),
    )

    step = recorder.trace.steps[0]
    assert step.input_tokens == 3
    assert step.output_tokens == 4
    assert step.cost_usd == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_counterfactual_execution_blocks_unapproved_side_effect_tools():
    calls: list[str] = []

    def side_effect():
        calls.append("called")
        return "done"

    workflow = WorkflowTemplate(
        name="tool-safety",
        entry_node="tool",
        nodes={
            "tool": Node(
                node_id="tool",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name="side_effect",
                ),
            ),
        },
    )
    executor = RuntimeExecutor(
        ExecutorConfig(max_steps=2),
        operators={"side_effect": side_effect},
    )

    output, context, trace = await executor.execute(
        workflow,
        FixedScheduler(),
        "query",
        counterfactual=True,
    )

    assert output is None
    assert context.success is False
    assert calls == []
    assert "Counterfactual execution blocked" in (
        trace.trace.error_message or ""
    )


@pytest.mark.asyncio
async def test_explicitly_safe_tool_can_run_in_counterfactual_mode():
    @counterfactual_safe
    def pure_tool():
        return "safe"

    workflow = WorkflowTemplate(
        name="safe-tool",
        entry_node="tool",
        nodes={
            "tool": Node(
                node_id="tool",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name="pure_tool",
                ),
            ),
        },
    )

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=2),
        operators={"pure_tool": pure_tool},
    ).execute(
        workflow,
        FixedScheduler(),
        "query",
        counterfactual=True,
    )

    assert output == "safe"
    assert context.success is True
    assert recorder.trace.metadata["execution_mode"] == "counterfactual"


@pytest.mark.asyncio
async def test_non_serializable_tool_output_cannot_break_trace_finalization():
    import threading

    workflow = WorkflowTemplate(
        name="opaque-tool-output",
        entry_node="tool",
        nodes={
            "tool": Node(
                node_id="tool",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name="make_lock",
                ),
            ),
        },
    )

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=2),
        operators={"make_lock": threading.Lock},
    ).execute(workflow, FixedScheduler(), "query")

    assert output is not None
    assert context.success is True
    recorded = recorder.trace.steps[0].tool_calls[0].tool_result
    assert recorded["__type__"].endswith(".lock")
    assert recorder.trace.final_output["__type__"].endswith(".lock")
    # The persisted model itself remains JSON serializable.
    assert "opaque-tool-output" in recorder.trace.model_dump_json()
