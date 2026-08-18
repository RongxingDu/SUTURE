"""Reward utility for workflow executions.

The scalar utility is deliberately kept *reward-only*.  Resource usage is an
optimization signal, rather than part of the reward measurement itself:

    U(trace) = R_hard(trace)

The optimizer's candidate gain adds the token delta and edit-distance
penalties around ``delta U``.  Latency, API cost, and runtime-complexity
statistics remain available as trace diagnostics (and can be used by the
separate efficiency-anchor mechanism), but they must not silently change the
utility reported for a workflow execution.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from awf.trace.schema import ExecutionTrace


class UtilityComputer:
    """Computes utility for a workflow execution.

    ``reward`` is the hard task reward.  The parameter name is retained for
    compatibility with existing callers, but a combined hard/process reward
    must not be passed here.  ``lambda_cost`` and the other coefficients are
    retained for configuration and the external efficiency objective; they do
    not alter this scalar utility.
    """

    def __init__(
        self,
        lambda_cost: float = 1e-4,
        rho_omega: float = 1e-3,
        beta_repair: float = 1.0,
        beta_reroute: float = 1.0,
        beta_fallback: float = 1.0,
        beta_loop: float = 1.0,
        lambda_latency: float = 0.0,
        lambda_api_cost: float = 0.0,
    ):
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency
        self.lambda_api_cost = lambda_api_cost
        self.rho_omega = rho_omega
        self.beta_repair = beta_repair
        self.beta_reroute = beta_reroute
        self.beta_fallback = beta_fallback
        self.beta_loop = beta_loop

    def compute(
        self,
        reward: float | None = None,
        trace: ExecutionTrace | None = None,
        operator_count: Optional[int] = None,
        *,
        hard_reward: float | None = None,
    ) -> float:
        """Compute utility for a single execution.

        Args:
            reward: The hard reward value.  This is intentionally reward-only;
                token/edit penalties are applied by candidate gain scoring.
            trace: The execution trace.
            operator_count: Number of operators used (computed from trace if None).
            hard_reward: Explicit alias for ``reward``.  It is useful for
                callers that want the reward-only contract to be self-documenting.

        Returns:
            Utility value.
        """
        if hard_reward is not None:
            if reward is not None:
                raise TypeError("pass either reward or hard_reward, not both")
            reward = hard_reward
        if reward is None:
            raise TypeError("reward (hard reward) is required")
        if trace is None:
            raise TypeError("trace is required")
        reward = float(reward)
        token_count = self.compute_execution_cost(trace)
        llm_latency = self.compute_llm_latency(trace)
        known_api_cost = self.compute_known_api_cost(trace)
        omega = self.compute_runtime_complexity(trace, operator_count)
        # Keep the scalar objective pure.  ``token_count`` and the remaining
        # observations are recorded below so callers can construct the gain
        # penalty without recomputing trace diagnostics, but none of them are
        # subtracted from utility here.
        utility = reward
        trace.metadata["utility_breakdown"] = {
            "reward": reward,
            "hard_reward": reward,
            "utility_definition": "hard_reward",
            "token_count": token_count,
            "token_penalty": 0.0,
            "llm_latency_seconds": llm_latency,
            "latency_penalty": 0.0,
            "known_api_cost_usd": known_api_cost,
            "api_cost_penalty": 0.0,
            "api_cost_estimate_complete": self.api_cost_estimate_complete(trace),
            "runtime_complexity": omega,
            "complexity_penalty": 0.0,
            "excluded_from_utility": [
                "process_reward",
                "tokens",
                "latency",
                "api_cost",
                "runtime_complexity",
            ],
            "utility": utility,
        }
        return utility

    @staticmethod
    def compute_execution_cost(trace: ExecutionTrace) -> float:
        """Return the recorded total token count.

        This compatibility name is used by the external efficiency objective.
        It intentionally uses raw input + output token count and never
        estimates cost from provider prices.  Aggregate counters are preferred
        when present; older traces may only contain call- or step-level
        counters.
        """
        aggregate_tokens = (
            trace.total_prompt_tokens + trace.total_completion_tokens
        )
        if aggregate_tokens:
            return float(aggregate_tokens)
        call_tokens = sum(
            call.prompt_tokens + call.completion_tokens
            for step in trace.steps
            for call in step.llm_calls
        )
        if call_tokens:
            return float(call_tokens)
        return float(
            sum(step.input_tokens + step.output_tokens for step in trace.steps)
        )

    @staticmethod
    def compute_llm_latency(trace: ExecutionTrace) -> float:
        """Return LLM-call latency without also charging encompassing wall time."""
        calls = [
            call
            for step in trace.steps
            for call in step.llm_calls
        ]
        recorded_latency = sum(
            max(float(call.latency_seconds), 0.0)
            for call in calls
        )
        # Suffix-replay traces contain only newly executed steps.  The
        # counterfactual evaluator records the cached prefix's exact LLM
        # latency separately so the candidate utility remains comparable to a
        # full execution of the workflow.
        prefix_latency = trace.metadata.get(
            "suffix_replay_prefix_llm_latency_seconds",
            0.0,
        )
        try:
            prefix_latency = max(float(prefix_latency), 0.0)
        except (TypeError, ValueError, OverflowError):
            prefix_latency = 0.0
        if recorded_latency > 0.0:
            return recorded_latency + prefix_latency
        # Older traces may have only the aggregate. Use it solely as a fallback;
        # summing both would charge the same LLM wait twice.
        if prefix_latency > 0.0:
            return prefix_latency + max(
                float(trace.total_latency_seconds) - prefix_latency,
                0.0,
            )
        return max(float(trace.total_latency_seconds), 0.0)

    @staticmethod
    def compute_known_api_cost(trace: ExecutionTrace) -> float:
        """Return the known cost portion, regardless of estimate completeness."""
        aggregate = max(float(trace.total_cost_usd), 0.0)
        if aggregate > 0.0:
            return aggregate
        return sum(
            max(float(call.cost_usd), 0.0)
            for step in trace.steps
            for call in step.llm_calls
        )

    @staticmethod
    def api_cost_estimate_complete(trace: ExecutionTrace) -> bool:
        """Whether every recorded provider call has a known price estimate."""
        return bool(trace.total_cost_estimate_complete) and all(
            call.cost_estimate_available
            for step in trace.steps
            for call in step.llm_calls
        )

    # Backwards-compatible private name used by early callers.
    def _compute_cost(self, trace: ExecutionTrace) -> float:
        return self.compute_execution_cost(trace)

    def compute_runtime_complexity(
        self,
        trace: ExecutionTrace,
        operator_count: Optional[int] = None,
    ) -> float:
        """Compute structural runtime complexity from actions and revisits."""
        if operator_count is not None:
            return float(operator_count)

        components = self.runtime_complexity_components(trace)
        return float(
            self.beta_repair * components["repair"]
            + self.beta_reroute * components["reroute"]
            + self.beta_fallback * components["fallback"]
            + self.beta_loop * components["loop"]
        )

    @staticmethod
    def runtime_complexity_components(
        trace: ExecutionTrace,
    ) -> dict[str, int]:
        """Return reportable counts using the same semantics as ``Omega``."""
        executed_steps = [
            step
            for step in trace.steps
            if step.metadata.get("node_executed", True)
        ]
        actions = [step.action.lower() for step in executed_steps]
        repair_count = sum(
            action in {"repair", "retry"} for action in actions
        )
        reroute_count = sum(
            action in {"reroute", "branch"} for action in actions
        )
        fallback_count = sum(
            action in {"fallback", "deviate"} for action in actions
        )
        visit_counts = Counter(
            step.node_id
            for step in executed_steps
        )
        repeated_visits = sum(max(count - 1, 0) for count in visit_counts.values())
        return {
            "repair": repair_count,
            "reroute": reroute_count,
            "fallback": fallback_count,
            "loop": repeated_visits,
        }

    # Backwards-compatible private name used by early callers.
    def _compute_omega(
        self,
        trace: ExecutionTrace,
        operator_count: Optional[int] = None,
    ) -> float:
        return self.compute_runtime_complexity(trace, operator_count)

    def compute_delta_u(
        self,
        reward_before: float,
        trace_before: ExecutionTrace,
        reward_after: float,
        trace_after: ExecutionTrace,
    ) -> float:
        """Compute utility delta between two executions.

        Used for counterfactual evaluation.
        """
        u_before = self.compute(reward_before, trace_before)
        u_after = self.compute(reward_after, trace_after)
        return u_after - u_before
