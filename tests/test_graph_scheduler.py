"""Regression tests for branch-faithful, zero-LLM graph scheduling."""

from __future__ import annotations

import pytest

from awf.config.schema import ExecutorConfig, SchedulerConfig
from awf.executor.runtime import RuntimeExecutor
from awf.protocol.experiment import ExperimentRunner
from awf.scheduler import GraphScheduler
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _node(
    node_id: str,
    node_type: NodeType,
    *,
    condition_expr: str | None = None,
) -> Node:
    return Node(
        node_id=node_id,
        node_type=node_type,
        config=NodeConfig(
            node_type=node_type,
            condition_expr=condition_expr,
        ),
    )


def _conditional_workflow() -> WorkflowTemplate:
    return WorkflowTemplate(
        name="conditional",
        entry_node="start",
        nodes={
            "start": _node("start", NodeType.START),
            "gate": _node(
                "gate",
                NodeType.CONDITION,
                condition_expr='context.query == "fast"',
            ),
            "fast_path": _node("fast_path", NodeType.LLM),
            "slow_path": _node("slow_path", NodeType.LLM),
            "end": _node("end", NodeType.END),
        },
        # CONDITION successors are ordered as (true, false).
        edges=[
            ("start", "gate"),
            ("gate", "fast_path"),
            ("gate", "slow_path"),
            ("fast_path", "end"),
            ("slow_path", "end"),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "selected", "not_selected"),
    [
        ("fast", "fast_path", "slow_path"),
        ("slow", "slow_path", "fast_path"),
    ],
)
async def test_graph_scheduler_executes_only_selected_condition_branch(
    query: str,
    selected: str,
    not_selected: str,
) -> None:
    calls: list[str] = []

    def run_path(context):
        calls.append(context.current_node_id)
        return context.current_node_id

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=8),
        operators={
            "fast_path": run_path,
            "slow_path": run_path,
        },
    ).execute(
        _conditional_workflow(),
        GraphScheduler(),
        query,
    )

    assert output == selected
    assert calls == [selected]
    assert selected in context.history
    assert not_selected not in context.history
    assert [step.action for step in recorder.trace.steps] == [
        "continue",
        "continue",
        "continue",
        "continue",
    ]
    assert recorder.trace.total_llm_calls == 0


def test_experiment_factory_constructs_graph_scheduler() -> None:
    config = SchedulerConfig(
        scheduler_type="graph",
        allow_deviation=False,
    )

    scheduler = ExperimentRunner._create_scheduler(config)

    assert isinstance(scheduler, GraphScheduler)
    assert scheduler.config is config
