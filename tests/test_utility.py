"""Tests for utility computation."""

import pytest

from awf.optimizer.acceptance import AcceptanceCriterion
from awf.trace.schema import ExecutionTrace, LLMCallRecord, TraceStep
from awf.utility.compute import UtilityComputer


class TestUtilityComputer:
    """Test UtilityComputer."""

    def _make_trace(self, num_steps=3, tokens=100) -> ExecutionTrace:
        trace = ExecutionTrace(
            trace_id="test",
            query_text="test query",
            total_prompt_tokens=tokens,
            total_completion_tokens=tokens // 2,
            total_latency_seconds=10.0,
        )
        for i in range(num_steps):
            step = TraceStep(
                step_id=f"step_{i}",
                step_index=i,
                node_id=f"node_{i}",
                node_type="llm",
                action="execute",
                llm_calls=[
                    LLMCallRecord(
                        call_id=f"call_{i}",
                        model="gpt-4o",
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    )
                ],
                duration_seconds=2.0,
            )
            trace.steps.append(step)
        return trace

    def test_compute_utility(self):
        computer = UtilityComputer(lambda_cost=1e-4, rho_omega=1e-3)
        trace = self._make_trace(num_steps=3, tokens=150)
        u = computer.compute(reward=0.8, trace=trace)
        # Utility is reward-only.  Resource terms are applied by candidate
        # gain scoring, not by the per-execution utility metric.
        assert u == pytest.approx(0.8)
        assert trace.metadata["utility_breakdown"]["token_count"] == 225
        assert trace.metadata["utility_breakdown"]["token_penalty"] == 0.0

    def test_new_penalties_default_to_backward_compatible_zero(self):
        trace = self._make_trace(num_steps=3, tokens=150)
        for step in trace.steps:
            step.llm_calls[0].latency_seconds = 4.0
            step.llm_calls[0].cost_usd = 0.25
        trace.total_cost_usd = 0.75

        default = UtilityComputer().compute(0.8, trace)
        explicit = UtilityComputer(
            lambda_cost=1e-4,
            lambda_latency=0.0,
            lambda_api_cost=0.0,
            rho_omega=1e-3,
        ).compute(0.8, trace)

        assert default == pytest.approx(explicit)
        assert default == pytest.approx(0.8)
        assert trace.metadata["utility_breakdown"]["latency_penalty"] == 0.0
        assert trace.metadata["utility_breakdown"]["api_cost_penalty"] == 0.0
        legacy_positional = UtilityComputer(0.1, 0.2)
        assert legacy_positional.lambda_cost == 0.1
        assert legacy_positional.rho_omega == 0.2
        assert legacy_positional.lambda_latency == 0.0
        assert legacy_positional.lambda_api_cost == 0.0

    def test_compute_delta_u(self):
        computer = UtilityComputer(lambda_cost=1e-4, rho_omega=1e-3)
        trace_before = self._make_trace(num_steps=3, tokens=150)
        trace_after = self._make_trace(num_steps=2, tokens=100)

        delta = computer.compute_delta_u(
            reward_before=0.5,
            trace_before=trace_before,
            reward_after=0.9,
            trace_after=trace_after,
        )
        # Should be positive since reward improved and cost decreased
        assert delta > 0.0

    def test_utility_is_independent_of_token_usage(self):
        computer = UtilityComputer(lambda_cost=1.0, rho_omega=0.0)
        trace_low = self._make_trace(num_steps=1, tokens=10)
        trace_high = self._make_trace(num_steps=10, tokens=1000)

        u_low = computer.compute(reward=0.9, trace=trace_low)
        u_high = computer.compute(reward=0.9, trace=trace_high)
        assert u_low == pytest.approx(u_high)

    def test_token_and_latency_penalties_are_independent(self):
        low_tokens = self._make_trace(num_steps=1, tokens=10)
        high_tokens = self._make_trace(num_steps=1, tokens=1000)
        fast = self._make_trace(num_steps=1, tokens=100)
        slow = self._make_trace(num_steps=1, tokens=100)
        low_tokens.steps[0].llm_calls[0].latency_seconds = 2.0
        high_tokens.steps[0].llm_calls[0].latency_seconds = 2.0
        fast.steps[0].llm_calls[0].latency_seconds = 1.0
        slow.steps[0].llm_calls[0].latency_seconds = 9.0

        latency_only = UtilityComputer(
            lambda_cost=0.0,
            lambda_latency=0.1,
            rho_omega=0.0,
        )
        token_only = UtilityComputer(
            lambda_cost=0.01,
            lambda_latency=0.0,
            rho_omega=0.0,
        )

        assert latency_only.compute(1.0, low_tokens) == pytest.approx(
            latency_only.compute(1.0, high_tokens)
        )
        assert token_only.compute(1.0, fast) == pytest.approx(
            token_only.compute(1.0, slow)
        )

    def test_slow_equal_reward_candidate_is_rejected(self):
        fast = self._make_trace(num_steps=1, tokens=100)
        slow = self._make_trace(num_steps=1, tokens=100)
        fast.steps[0].llm_calls[0].latency_seconds = 1.0
        slow.steps[0].llm_calls[0].latency_seconds = 8.0
        computer = UtilityComputer(
            lambda_cost=0.0,
            lambda_latency=0.1,
            rho_omega=0.0,
        )

        delta_u = computer.compute_delta_u(1.0, fast, 1.0, slow)

        assert delta_u == pytest.approx(0.0)
        assert AcceptanceCriterion(epsilon_stat=0.0).accept(delta_u) is False
        assert slow.metadata["utility_breakdown"][
            "llm_latency_seconds"
        ] == pytest.approx(8.0)

    def test_llm_latency_precedes_wall_clock_and_known_partial_cost_is_used(self):
        trace = self._make_trace(num_steps=2, tokens=100)
        trace.total_latency_seconds = 100.0
        trace.steps[0].llm_calls[0].latency_seconds = 0.75
        trace.steps[1].llm_calls[0].latency_seconds = 1.25
        trace.total_cost_usd = 0.25
        trace.total_cost_estimate_complete = False
        computer = UtilityComputer(
            lambda_cost=0.0,
            lambda_latency=0.1,
            lambda_api_cost=2.0,
            rho_omega=0.0,
        )

        utility = computer.compute(1.0, trace)
        breakdown = trace.metadata["utility_breakdown"]

        assert utility == pytest.approx(1.0)
        assert breakdown["llm_latency_seconds"] == pytest.approx(2.0)
        assert breakdown["latency_penalty"] == 0.0
        assert breakdown["known_api_cost_usd"] == pytest.approx(0.25)
        assert breakdown["api_cost_penalty"] == 0.0
        assert breakdown["api_cost_estimate_complete"] is False

    def test_complexity_counts_only_executed_steps_and_legacy_aliases(self):
        trace = ExecutionTrace(
            trace_id="complexity",
            steps=[
                TraceStep(
                    step_id="execute",
                    step_index=0,
                    node_id="same",
                    action="retry",
                    metadata={"node_executed": True},
                ),
                TraceStep(
                    step_id="stop",
                    step_index=1,
                    node_id="same",
                    action="fallback",
                    metadata={"node_executed": False},
                ),
            ],
        )

        components = UtilityComputer.runtime_complexity_components(trace)

        assert components == {
            "repair": 1,
            "reroute": 0,
            "fallback": 0,
            "loop": 0,
        }
