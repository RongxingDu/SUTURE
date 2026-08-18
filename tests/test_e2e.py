"""End-to-end tests with mock LLM.

Tests the full pipeline:
1. Create a workflow -> execute with scheduler -> verify trace
2. Compute reward + utility -> run one optimization round -> verify acceptance
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from awf.config.schema import (
    ExecutorConfig,
    ExperimentConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.optimizer.acceptance import AcceptanceCriterion
from awf.optimizer.anchor_localizer import AnchorLocalizer
from awf.optimizer.candidate_generator import CandidateGenerator, WorkflowCandidate
from awf.optimizer.failure_buffer import FailureBuffer
from awf.optimizer.scorer import CandidateScorer
from awf.reward.base import RewardEvaluator
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.trace.recorder import TraceRecorder
from awf.trace.schema import ExecutionTrace, LLMCallRecord, TraceStep
from awf.utility.compute import UtilityComputer
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType


class MockRewardEvaluator(RewardEvaluator):
    """Simple reward evaluator for testing."""

    def hard_reward(self, query, ground_truth, output, trace):
        if output and "correct" in str(output).lower():
            return 1.0
        return 0.0

    def process_reward(self, query, ground_truth, output, trace):
        return 0.5


def make_test_workflow() -> WorkflowTemplate:
    """Create a simple 3-node workflow for testing."""
    return WorkflowTemplate(
        name="test_workflow",
        version="1.0",
        entry_node="start",
        nodes={
            "start": Node(
                node_id="start", node_type=NodeType.START, label="Start"
            ),
            "process": Node(
                node_id="process",
                node_type=NodeType.LLM,
                config=NodeConfig(
                    node_type=NodeType.LLM,
                    prompt_template="Process: {query}",
                    system_prompt="You are helpful.",
                ),
                label="Process",
            ),
            "end": Node(
                node_id="end", node_type=NodeType.END, label="End"
            ),
        },
        edges=[("start", "process"), ("process", "end")],
    )


class TestEndToEndExecution:
    """Test the full execution pipeline with mock LLM."""

    @pytest.mark.asyncio
    async def test_execute_with_fixed_scheduler(self):
        """Execute a workflow with FixedScheduler and mock LLM."""
        workflow = make_test_workflow()
        scheduler = FixedScheduler()
        executor = RuntimeExecutor(
            config=ExecutorConfig(max_steps=10),
        )

        # Create a mock LLM client
        mock_client = AsyncMock()
        mock_client.generate.return_value = (
            "This is the correct answer.",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_client.config = MagicMock()
        mock_client.config.model = "gpt-4o"

        output, context, recorder = await executor.execute(
            workflow, scheduler, "Test query", llm_client=mock_client
        )

        assert output is not None
        assert context.finished is True
        assert context.success is True
        trace = recorder.trace
        assert trace.total_steps > 0
        assert trace.trace_id is not None

    @pytest.mark.asyncio
    async def test_execute_with_mock_operators(self):
        """Execute with custom operator functions."""
        workflow = make_test_workflow()

        async def custom_process(context):
            return "correct result"

        executor = RuntimeExecutor(
            config=ExecutorConfig(max_steps=10),
            operators={"process": custom_process},
        )
        scheduler = FixedScheduler()

        output, context, recorder = await executor.execute(
            workflow, scheduler, "Test query"
        )

        assert output == "correct result"

    @pytest.mark.asyncio
    async def test_execution_trace_structure(self):
        """Verify trace structure after execution."""
        workflow = make_test_workflow()
        scheduler = FixedScheduler()

        mock_client = AsyncMock()
        mock_client.generate.return_value = (
            "result",
            {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        mock_client.config = MagicMock()
        mock_client.config.model = "gpt-4o"

        executor = RuntimeExecutor(ExecutorConfig(max_steps=10))
        _, _, recorder = await executor.execute(
            workflow, scheduler, "test", llm_client=mock_client
        )

        trace = recorder.trace
        # Should have at least START and PROCESS steps
        assert trace.total_steps >= 2
        for step in trace.steps:
            assert step.step_id.startswith("step_")
            assert step.node_id in workflow.nodes
            assert step.action in ["execute", "stop", "skip", "retry"]


class TestOptimizerPipeline:
    """Test the optimization pipeline components."""

    def _make_trace(self, success: bool, hard_reward: float = 0.0) -> ExecutionTrace:
        trace = ExecutionTrace(
            trace_id="test_trace",
            query_text="test query",
            success=success,
            hard_reward=hard_reward,
            total_prompt_tokens=50,
            total_completion_tokens=25,
        )
        step = TraceStep(
            step_id="step_0",
            step_index=0,
            node_id="process",
            node_type="llm",
            action="execute",
            llm_calls=[
                LLMCallRecord(
                    call_id="c0",
                    model="gpt-4o",
                    system_prompt="You are helpful.",
                    user_prompt="Test",
                    response_text="wrong answer" if not success else "correct",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                )
            ],
            success=success,
        )
        trace.steps.append(step)
        trace.metadata["ground_truth"] = "correct"
        return trace

    def test_failure_buffer(self):
        buffer = FailureBuffer(capacity=10, success_threshold=0.5)

        # Add failure
        buffer.add(self._make_trace(success=False, hard_reward=0.0))
        assert len(buffer.failures) == 1
        assert len(buffer.successes) == 0

        # Add success
        buffer.add(self._make_trace(success=True, hard_reward=1.0))
        assert len(buffer.failures) == 1
        assert len(buffer.successes) == 1
        assert buffer.total_traces == 2
        assert buffer.get_failure_rate() == 0.5

    def test_acceptance_criterion(self):
        criterion = AcceptanceCriterion(epsilon_stat=0.01)
        assert criterion.accept(0.05) is True
        assert criterion.accept(0.005) is False
        assert criterion.accept(-0.1) is False

    def test_candidate_scorer(self):
        scorer = CandidateScorer(mu_edit=0.1)
        candidate = WorkflowCandidate(
            scope="prompt",
            node_id="process",
            description="Fix prompt",
            changes={"system_prompt": "Better prompt"},
            edit_distance=0.15,
        )
        gain = scorer.score(candidate, delta_u=0.1)
        # G = 0.1 - 0.1 * 0.15 = 0.085
        assert gain == pytest.approx(0.085)

    def test_candidate_gain_adds_token_delta_outside_utility(self):
        scorer = CandidateScorer(mu_edit=0.1, lambda_tokens=0.01)
        candidate = WorkflowCandidate(
            scope="prompt",
            node_id="process",
            description="shorter prompt",
            changes={},
            edit_distance=0.15,
        )
        # Candidate uses 10 fewer tokens: .1 - .01*(-10) - .1*.15 = .185.
        gain = scorer.score(candidate, delta_u=0.1, token_delta=-10)
        assert gain == pytest.approx(0.185)

    def test_scorer_batch(self):
        scorer = CandidateScorer(mu_edit=0.1)
        c1 = WorkflowCandidate(
            scope="prompt", node_id="n1", description="c1",
            changes={}, edit_distance=0.1,
        )
        c2 = WorkflowCandidate(
            scope="operator", node_id="n2", description="c2",
            changes={}, edit_distance=0.3,
        )
        scored = scorer.score_batch([c1, c2], [0.2, 0.1])
        # c1: 0.2 - 0.01 = 0.19, c2: 0.1 - 0.03 = 0.07
        assert scored[0][0] == c1  # Higher gain first
        assert scored[0][1] == pytest.approx(0.19)

    def test_utility_computation_on_trace(self):
        computer = UtilityComputer(lambda_cost=1e-4, rho_omega=1e-3)
        trace = self._make_trace(success=False, hard_reward=0.0)
        u = computer.compute(reward=0.3, trace=trace)
        assert u == pytest.approx(0.3)  # Utility is reward-only

    @pytest.mark.asyncio
    async def test_anchor_localizer_with_mock(self):
        """Test anchor localizer with mock LLM responses."""
        workflow = make_test_workflow()
        trace = self._make_trace(success=False, hard_reward=0.0)

        mock_llm = AsyncMock()
        mock_llm.generate_json.return_value = (
            '{"anchors": [{"node_id": "process", "scope": "prompt", '
            '"reasoning": "Bad prompt caused wrong answer", "confidence": 0.9}]}',
            {"prompt_tokens": 50, "completion_tokens": 30},
        )

        localizer = AnchorLocalizer(mock_llm)
        anchors = await localizer.localize(trace, workflow)
        assert len(anchors) == 1
        assert anchors[0]["node_id"] == "process"
        assert anchors[0]["scope"] == "prompt"

    @pytest.mark.asyncio
    async def test_candidate_generator_with_mock(self):
        """Test candidate generator with mock LLM."""
        workflow = make_test_workflow()
        anchors = [
            {"node_id": "process", "scope": "prompt",
             "reasoning": "Bad prompt", "confidence": 0.9}
        ]

        mock_llm = AsyncMock()
        mock_llm.generate_json.return_value = (
            '{"candidates": [{"scope": "prompt", "node_id": "process", '
            '"description": "Fix system prompt", '
            '"changes": {"system_prompt": "Better prompt"}}]}',
            {"prompt_tokens": 50, "completion_tokens": 30},
        )

        generator = CandidateGenerator(mock_llm)
        candidates = await generator.generate(workflow, anchors)

        assert len(candidates) == 1
        assert candidates[0].scope == "prompt"
        assert candidates[0].node_id == "process"
        assert candidates[0].modified_workflow is not None
        # Check the modification was applied
        modified_node = candidates[0].modified_workflow.nodes["process"]
        assert modified_node.config.system_prompt == "Better prompt"

    def test_acceptance_select_best(self):
        criterion = AcceptanceCriterion(epsilon_stat=0.01)
        c1 = WorkflowCandidate(scope="p", node_id="n1", description="", changes={})
        c2 = WorkflowCandidate(scope="p", node_id="n2", description="", changes={})

        scored = [(c1, 0.05), (c2, 0.005)]
        best = criterion.select_best(scored)
        assert best is not None
        assert best[0] == c1

        # No candidate meets threshold
        scored_bad = [(c1, -0.1), (c2, -0.2)]
        best = criterion.select_best(scored_bad)
        assert best is None
