from __future__ import annotations

from types import SimpleNamespace

import pytest

from awf.config.schema import ExecutorConfig, LLMConfig, SchedulerConfig
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.protocol.experiment import ExperimentRunner
from awf.scheduler import CascadeScheduler, SchedulerAction
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


def _workflow() -> WorkflowTemplate:
    def node(node_id: str, node_type: NodeType, role: str = "") -> Node:
        return Node(
            node_id=node_id,
            node_type=node_type,
            label=role,
            config=NodeConfig(
                node_type=node_type,
                metadata={"role": role} if role else {},
            ),
        )

    return WorkflowTemplate(
        name="cascade-test",
        entry_node="start",
        nodes={
            "start": node("start", NodeType.START),
            "answer": node("answer", NodeType.LLM, "answer"),
            "verify": node("verify", NodeType.LLM, "verify"),
            "end": node("end", NodeType.END),
        },
        edges=[
            ("start", "answer"),
            ("answer", "verify"),
            ("verify", "end"),
        ],
    )


class FakeSchedulerLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []
        self.config = SimpleNamespace(model="deepseek-v4-flash")

    async def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response, {
            "prompt_tokens": 61,
            "completion_tokens": 12,
            "total_tokens": 73,
            "latency_seconds": 0.12,
            "cost_estimate_available": False,
        }


def _context(output: str, query: str = "compute 2 + 2") -> ExecutionContext:
    context = ExecutionContext(query=query, workflow_name="cascade-test")
    context.current_node_id = "end"
    context.record_output("answer", output)
    context.current_node_id = "end"
    return context


def _config(**kwargs) -> SchedulerConfig:
    values = {
        "scheduler_type": "cascade",
        "allow_deviation": True,
        "llm": LLMConfig(model="deepseek-v4-flash"),
    }
    values.update(kwargs)
    return SchedulerConfig(**values)


@pytest.mark.asyncio
async def test_gate_early_exits_without_scheduler_generation_call():
    llm = FakeSchedulerLLM('{"action":"continue"}')
    scheduler = CascadeScheduler(_config(), llm_client=llm)
    await scheduler.initialize(_workflow(), "compute 2 + 2")

    action, params = await scheduler.select_action(
        _workflow(), _context("final answer: 4.")
    )

    assert action == SchedulerAction.EARLY_EXIT
    assert params["_awf_gate"]["route"] == "early_exit"
    assert llm.calls == []
    assert scheduler.pop_last_llm_call() is None


@pytest.mark.asyncio
async def test_las_gate_forwards_promising_artifact_when_direct_exit_disabled():
    llm = FakeSchedulerLLM('{"action":"continue"}')
    scheduler = CascadeScheduler(
        _config(
            gate_formula="las",
            gate_direct_early_exit=False,
            gate_schedule_threshold=0.55,
        ),
        llm_client=llm,
    )
    workflow = _workflow()
    await scheduler.initialize(workflow, "compute 2 + 2")

    action, params = await scheduler.select_action(
        workflow,
        _context("final answer: 4."),
    )

    assert action == SchedulerAction.CONTINUE
    assert params["_awf_gate"]["route"] == "invoke_scheduler"
    assert scheduler.gate.config.gate_formula == "las"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_low_quality_gate_follows_template_without_llm():
    llm = FakeSchedulerLLM('{"action":"early_exit"}')
    scheduler = CascadeScheduler(_config(), llm_client=llm)
    await scheduler.initialize(_workflow(), "compute 2 + 2")

    action, params = await scheduler.select_action(
        _workflow(), _context("x")
    )

    assert action == SchedulerAction.CONTINUE
    assert params["_awf_gate"]["route"] == "continue"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_borderline_gate_invokes_deepseek_and_routes_to_verifier():
    llm = FakeSchedulerLLM(
        '{"action":"verify","target_node":"verify"}'
    )
    scheduler = CascadeScheduler(_config(), llm_client=llm)
    await scheduler.initialize(_workflow(), "compute 2 + 2")

    action, params = await scheduler.select_action(
        _workflow(), _context("answer: maybe")
    )

    assert action == SchedulerAction.VERIFY
    assert params["target_node"] == "verify"
    assert len(llm.calls) == 1
    call = scheduler.pop_last_llm_call()
    assert call is not None
    assert call.model == "deepseek-v4-flash"
    assert call.call_type == "scheduler"
    assert call.total_tokens == 73


@pytest.mark.asyncio
async def test_high_risk_early_exit_response_is_overridden():
    llm = FakeSchedulerLLM('{"action":"early_exit"}')
    scheduler = CascadeScheduler(_config(), llm_client=llm)
    query = "prove " + " 2 + 3" * 5
    await scheduler.initialize(_workflow(), query)

    action, params = await scheduler.select_action(
        _workflow(), _context("final answer: 5.", query=query)
    )

    assert action == SchedulerAction.CONTINUE
    assert params["risk_override"] == "high_risk_requires_verification"


@pytest.mark.asyncio
async def test_runtime_records_cascade_telemetry_and_scheduler_tokens():
    llm = FakeSchedulerLLM('{"action":"continue"}')
    config = _config()
    scheduler = CascadeScheduler(config, llm_client=llm)

    def answer(context):
        return "answer: maybe"

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=8),
        operators={"answer": answer, "verify": answer},
    ).execute(_workflow(), scheduler, "compute 2 + 2")

    assert context.finished is True
    assert recorder.trace.metadata["scheduler_telemetry"]["gate_invocations"]
    assert recorder.trace.metadata["scheduler_telemetry"]["scheduler_invocations"] >= 1
    assert recorder.trace.total_prompt_tokens == 122
    assert recorder.trace.total_completion_tokens == 24
    assert any(
        call.call_type == "scheduler"
        for step in recorder.trace.steps
        for call in step.llm_calls
    )


def test_experiment_factory_constructs_cascade_scheduler():
    config = _config(llm=LLMConfig(model="deepseek-v4-flash", api_key="test"))
    scheduler = ExperimentRunner._create_scheduler(config)
    assert isinstance(scheduler, CascadeScheduler)
    assert scheduler.config is config
