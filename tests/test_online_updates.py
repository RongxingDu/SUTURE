"""Tests for streaming failure-triggered workflow updates."""

import logging
from types import SimpleNamespace

import pytest

from awf.protocol.experiment import ExperimentRunner
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.utility.compute import UtilityComputer
from awf.workflow.serializer import load_workflow


class _SequenceReward(RewardEvaluator):
    def hard_reward(self, query, ground_truth, output, trace):
        return 0.0 if query == "bad" else 1.0

    def process_reward(self, query, ground_truth, output, trace):
        return 0.0


class _OutputReward(RewardEvaluator):
    def hard_reward(self, query, ground_truth, output, trace):
        return float(output == "good")

    def process_reward(self, query, ground_truth, output, trace):
        return 0.0


class _Executor:
    async def execute(self, workflow, scheduler, query, **kwargs):
        trace = ExecutionTrace(
            trace_id=f"{query}-{workflow.version}",
            query_text=query,
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            success=True,
            final_output=query,
            total_prompt_tokens=10,
            total_completion_tokens=2,
            total_latency_seconds=0.25,
            total_llm_calls=1,
        )
        return query, SimpleNamespace(), SimpleNamespace(trace=trace)


class _SequenceExecutor:
    """Return a configured output sequence while recording queried items."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.queries = []

    async def execute(self, workflow, scheduler, query, **kwargs):
        del scheduler, kwargs
        output = next(self.outputs)
        self.queries.append(query)
        trace = ExecutionTrace(
            trace_id=f"{query}-{workflow.version}-{len(self.queries)}",
            query_text=query,
            workflow_name=workflow.name,
            workflow_version=workflow.version,
            success=True,
            final_output=output,
            total_prompt_tokens=10,
            total_completion_tokens=2,
            total_latency_seconds=0.25,
            total_llm_calls=1,
        )
        return output, SimpleNamespace(), SimpleNamespace(trace=trace)


class _Buffer:
    def __init__(self):
        self.items = []

    def add(self, trace):
        self.items.append(trace)

    @property
    def failures(self):
        return list(self.items)

    def extend(self, traces):
        for trace in traces:
            self.add(trace)

    def clear(self):
        self.items.clear()


class _OnlineOptimizer:
    def __init__(self):
        self.failure_buffer = _Buffer()
        self.counterfactual = SimpleNamespace(suffix_replay=object())
        self.round_history = []
        self.calls = []

    async def optimize_round(self, workflow, executor, scheduler, **kwargs):
        self.calls.append((workflow.version, list(self.failure_buffer.items)))
        candidate = workflow.model_copy(deep=True)
        candidate.version = f"{workflow.version}.online"
        summary = {
            "accepted": True,
            "gain": 0.5,
            "candidate_description": "repair failing suffix",
            "suffix_replay_used": True,
            "generated_candidates": 1,
            "evaluated_candidates": 1,
        }
        self.round_history.append(summary)
        self.failure_buffer.clear()
        return candidate, summary


@pytest.mark.asyncio
async def test_online_epoch_logs_each_sample_and_updates_on_failure(caplog):
    caplog.set_level(logging.INFO)
    workflow = load_workflow("experiments/workflows/math/default_workflow.yaml")
    optimizer = _OnlineOptimizer()
    runner = object.__new__(ExperimentRunner)
    runner.workflow = workflow
    runner.opt_data = [("bad", "answer"), ("good", "answer")]
    runner.executor = _Executor()
    runner.scheduler = object()
    runner.llm_client = None
    runner.optimizer = optimizer
    runner.reward_evaluator = _SequenceReward()
    runner.utility_computer = UtilityComputer(lambda_cost=0.0, rho_omega=0.0)
    runner.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            online_failure_updates=True,
            hard_success_threshold=1.0,
        ),
        reward=SimpleNamespace(alpha_process=0.0),
    )
    runner._current_round = 1
    runner._save_traces = lambda *args, **kwargs: None
    runner._initialized_trace_splits = set()
    runner._persist_traces = False

    traces, metrics = await runner._run_optimization_epoch()

    assert len(traces) == 2
    assert metrics["num_examples"] == 2
    assert len(optimizer.calls) == 1
    assert optimizer.calls[0][0] == workflow.version
    assert optimizer.calls[0][1][0].query_text == "bad"
    assert runner.workflow.version.endswith(".online")
    assert len(runner._online_round_updates) == 1
    assert runner._online_round_updates[0]["suffix_replay_used"] is True
    assert "success=False" in caplog.text
    assert "success=True" in caplog.text


@pytest.mark.asyncio
async def test_deferred_epoch_buffers_then_updates_failures_after_epoch():
    workflow = load_workflow("experiments/workflows/math/default_workflow.yaml")
    optimizer = _OnlineOptimizer()
    runner = object.__new__(ExperimentRunner)
    runner.workflow = workflow
    runner.opt_data = [("bad", "answer"), ("good", "answer")]
    runner.executor = _Executor()
    runner.scheduler = object()
    runner.llm_client = None
    runner.optimizer = optimizer
    runner.reward_evaluator = _SequenceReward()
    runner.utility_computer = UtilityComputer(lambda_cost=0.0, rho_omega=0.0)
    runner.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            failure_update_mode="deferred_sequential",
            online_failure_updates=False,
            hard_success_threshold=1.0,
        ),
        reward=SimpleNamespace(alpha_process=0.0),
    )
    runner._current_round = 1
    runner._save_traces = lambda *args, **kwargs: None
    runner._initialized_trace_splits = set()
    runner._persist_traces = False

    traces, metrics = await runner._run_optimization_epoch()
    assert len(traces) == 2
    assert metrics["num_examples"] == 2
    assert optimizer.calls == []
    assert [trace.query_text for trace in optimizer.failure_buffer.items] == [
        "bad"
    ]

    updated, summary = await runner._run_deferred_failure_updates(
        1,
        workflow.model_copy(deep=True),
        traces,
    )

    assert len(optimizer.calls) == 1
    assert optimizer.calls[0][1][0].query_text == "bad"
    assert updated.version.endswith(".online")
    assert summary["failure_update_mode"] == "deferred_sequential"
    assert summary["deferred_failure_count"] == 1
    assert summary["deferred_accepted_count"] == 1


@pytest.mark.asyncio
async def test_deferred_confirmation_repeats_only_observed_failures():
    workflow = load_workflow("experiments/workflows/math/default_workflow.yaml")
    optimizer = _OnlineOptimizer()
    runner = object.__new__(ExperimentRunner)
    runner.workflow = workflow
    # The first failed observation is supplied in round_traces. Its two
    # confirmations disagree (one correct, one wrong), but 2/3 total runs are
    # still failures, so it remains eligible. The success row is a guard and
    # is not repeated by failure confirmation.
    runner.executor = _SequenceExecutor(["good", "bad"])
    runner.scheduler = object()
    runner.llm_client = None
    runner.optimizer = optimizer
    runner.reward_evaluator = _OutputReward()
    runner.utility_computer = UtilityComputer(lambda_cost=0.0, rho_omega=0.0)
    runner.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            failure_update_mode="deferred_sequential",
            online_failure_updates=False,
            hard_success_threshold=1.0,
            failure_confirmation_repeats=2,
            failure_confirmation_min_failures=2,
            success_guard_fraction=0.0,
            min_success_guards=0,
        ),
        reward=SimpleNamespace(alpha_process=0.0),
    )
    runner._current_round = 1
    runner._save_traces = lambda *args, **kwargs: None
    runner._deferred_round_updates = []

    failure = ExecutionTrace(
        trace_id="failure-source",
        query_text="bad",
        workflow_name=workflow.name,
        workflow_version=workflow.version,
        success=True,
        final_output="bad",
        hard_reward=0.0,
        process_reward=0.0,
        metadata={"split": "optimization", "ground_truth": "answer"},
    )
    success = ExecutionTrace(
        trace_id="success-source",
        query_text="good",
        workflow_name=workflow.name,
        workflow_version=workflow.version,
        success=True,
        final_output="good",
        hard_reward=1.0,
        process_reward=0.0,
        metadata={"split": "optimization", "ground_truth": "answer"},
    )

    updated, summary = await runner._run_deferred_failure_updates(
        1,
        workflow.model_copy(deep=True),
        [failure, success],
    )

    assert updated.version.endswith(".online")
    assert runner.executor.queries == ["bad", "bad"]
    assert len(optimizer.calls) == 1
    update = runner._deferred_round_updates[0]
    assert update["failure_confirmation_total_runs"] == 3
    assert update["failure_confirmation_failures"] == 2
    assert summary["deferred_accepted_count"] == 1


@pytest.mark.asyncio
async def test_deferred_replay_remains_optimization_trace_for_cwu():
    workflow = load_workflow("experiments/workflows/math/default_workflow.yaml")
    runner = object.__new__(ExperimentRunner)
    runner.workflow = workflow
    runner.scheduler = object()
    runner.llm_client = None
    runner.executor = _Executor()
    runner.reward_evaluator = _SequenceReward()
    runner._current_round = 1

    source = ExecutionTrace(
        trace_id="source-failure",
        query_text="bad",
        workflow_name=workflow.name,
        workflow_version=workflow.version,
        metadata={
            "split": "optimization",
            "ground_truth": "answer",
            "split_example_index": 3,
        },
    )

    replay = await runner._replay_deferred_failure(source, 0)

    # Replay traces are persisted under their own phase, but remain in the
    # optimization split contract consumed by the workflow optimizer.
    assert replay.metadata["split"] == "optimization"
    assert replay.metadata["trace_phase"] == "deferred_failure_replay"
    assert replay.metadata["replay_source_split"] == "optimization"


def test_suffix_replay_utility_includes_cached_prefix_latency():
    from awf.trace.schema import LLMCallRecord, TraceStep

    trace = ExecutionTrace(
        trace_id="suffix-cost",
        total_latency_seconds=0.7,
        metadata={
            "suffix_replay_prefix_llm_latency_seconds": 0.5,
        },
        steps=[
            TraceStep(
                step_id="suffix",
                step_index=0,
                node_id="solve",
                llm_calls=[
                    LLMCallRecord(
                        call_id="suffix-call",
                        latency_seconds=0.2,
                    )
                ],
            )
        ],
    )

    assert UtilityComputer.compute_llm_latency(trace) == pytest.approx(0.7)
