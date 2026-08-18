"""Regression tests for observation-only experiment telemetry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from awf.config.schema import LLMConfig
from awf.llm.client import AsyncLLMClient
from awf.llm.cost_tracker import TokenUsageTracker
from awf.protocol.experiment import ExperimentRunner
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace, LLMCallRecord, TraceStep
from awf.utility.compute import UtilityComputer


class _ConstantReward(RewardEvaluator):
    def hard_reward(self, query, ground_truth, output, trace):
        return 1.0

    def process_reward(self, query, ground_truth, output, trace):
        return 0.0


class _UsageClient:
    def __init__(self, model: str):
        self.config = SimpleNamespace(model=model)
        self.tracker = TokenUsageTracker(model=model)

    def get_usage_summary(self):
        return self.tracker.snapshot()


@pytest.mark.asyncio
async def test_success_without_provider_usage_still_counts_call_and_latency():
    client = AsyncLLMClient(LLMConfig(api_key="test", model="gpt-4o"))
    client._call_with_retry = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ],
            usage=None,
        )
    )

    _, usage = await client.generate(user_prompt="question")

    summary = client.get_usage_summary()
    assert summary["num_calls"] == 1
    assert summary["total_tokens"] == 0
    assert summary["total_latency_seconds"] >= 0.0
    assert summary["unknown_pricing_calls"] == 1
    assert summary["cost_estimate_complete"] is False
    assert usage["latency_seconds"] >= 0.0
    assert usage["cost_estimate_available"] is False


@pytest.mark.asyncio
async def test_provider_usage_preserves_measured_call_latency():
    client = AsyncLLMClient(LLMConfig(api_key="test", model="gpt-4o"))
    client._call_with_retry = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
            ),
        )
    )

    _, usage = await client.generate(user_prompt="question")

    assert usage["latency_seconds"] >= 0.0
    assert usage["total_tokens"] == 5


def test_backend_usage_snapshot_and_delta_aggregate_all_three_clients():
    runner = object.__new__(ExperimentRunner)
    runner.llm_client = _UsageClient("workflow-model")
    runner.scheduler = SimpleNamespace(
        llm=_UsageClient("scheduler-model")
    )
    runner.optimizer = SimpleNamespace(
        llm=_UsageClient("optimizer-model")
    )
    before = runner._backend_usage_snapshot()

    runner.llm_client.tracker.record(
        10,
        5,
        prompt_cache_hit_tokens=4,
        latency_seconds=0.3,
    )
    runner.scheduler.llm.tracker.record(
        6,
        2,
        reasoning_tokens=1,
        latency_seconds=0.2,
    )
    runner.optimizer.llm.tracker.record(8, 3, latency_seconds=0.4)
    runner.optimizer.llm.tracker.record(7, 2, latency_seconds=0.5)

    after = runner._backend_usage_snapshot()
    delta = runner._backend_usage_delta(before, after)

    assert delta["workflow_client"]["num_calls"] == 1
    assert delta["scheduler_client"]["num_calls"] == 1
    assert delta["optimizer_client"]["num_calls"] == 2
    assert delta["total"]["num_calls"] == 4
    assert delta["total"]["total_tokens"] == 43
    assert delta["total"]["prompt_cache_hit_tokens"] == 4
    assert delta["total"]["reasoning_tokens"] == 1
    assert delta["total"]["total_latency_seconds"] == pytest.approx(1.4)


@pytest.mark.asyncio
async def test_execute_dataset_splits_llm_metrics_and_counts_real_path():
    trace = ExecutionTrace(
        trace_id="telemetry-trace",
        success=True,
        final_output="answer",
        total_prompt_tokens=12,
        total_completion_tokens=6,
        total_latency_seconds=1.2,
        total_cost_usd=0.03,
        total_cost_estimate_complete=True,
        total_llm_calls=2,
        steps=[
            TraceStep(
                step_id="0",
                step_index=0,
                node_id="solve",
                action="execute",
                metadata={
                    "decision_node_id": "solve",
                    "node_executed": True,
                },
                llm_calls=[
                    LLMCallRecord(
                        call_id="workflow",
                        model="m",
                        prompt_tokens=8,
                        completion_tokens=5,
                        total_tokens=13,
                        latency_seconds=0.7,
                        cost_usd=0.02,
                        call_type="workflow",
                        metadata={
                            "usage": {
                                "prompt_cache_hit_tokens": 3,
                            }
                        },
                    )
                ],
            ),
            TraceStep(
                step_id="1",
                step_index=1,
                node_id="end",
                action="early_exit",
                metadata={
                    "decision_node_id": "end",
                    "node_executed": False,
                },
                llm_calls=[
                    LLMCallRecord(
                        call_id="scheduler",
                        model="m",
                        prompt_tokens=4,
                        completion_tokens=1,
                        total_tokens=5,
                        latency_seconds=0.2,
                        cost_usd=0.01,
                        call_type="scheduler",
                        metadata={
                            "usage": {
                                "reasoning_tokens": 1,
                            }
                        },
                    )
                ],
            ),
        ],
    )

    class _Executor:
        async def execute(self, *args, **kwargs):
            return "answer", SimpleNamespace(), SimpleNamespace(trace=trace)

    runner = object.__new__(ExperimentRunner)
    runner.executor = _Executor()
    runner.scheduler = object()
    runner.llm_client = object()
    runner.reward_evaluator = _ConstantReward()
    runner.utility_computer = UtilityComputer(
        lambda_cost=0.0,
        rho_omega=0.0,
    )
    runner.config = SimpleNamespace(
        reward=SimpleNamespace(alpha_process=0.0),
    )
    runner._current_round = 1
    runner._save_traces = lambda *args, **kwargs: None

    _, metrics = await runner._execute_dataset(
        SimpleNamespace(),
        [("query", "answer")],
        "optimization",
    )

    assert metrics["workflow_llm"]["call_count"] == 1
    assert metrics["workflow_llm"]["total_tokens"] == 13
    assert metrics["workflow_llm"]["prompt_cache_hit_tokens"] == 3
    assert metrics["workflow_llm"]["latency_seconds"] == pytest.approx(0.7)
    assert metrics["scheduler_llm"]["call_count"] == 1
    assert metrics["scheduler_llm"]["total_tokens"] == 5
    assert metrics["scheduler_llm"]["reasoning_tokens"] == 1
    assert metrics["scheduler_llm"]["latency_seconds"] == pytest.approx(0.2)
    assert metrics["action_counts"] == {
        "early_exit": 1,
        "execute": 1,
    }
    assert metrics["decision_node_counts"] == {"end": 1, "solve": 1}
    assert metrics["executed_node_counts"] == {"solve": 1}
    assert metrics["node_execution_count"] == 1
