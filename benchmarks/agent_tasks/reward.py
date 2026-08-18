"""Agent-task hard reward and grounded process signals."""

from __future__ import annotations

from typing import Any

from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace, ToolCallRecord
from benchmarks.agent_tasks.evaluator import AgentTaskEvaluator


class AgentReward(RewardEvaluator):
    """Task-success score plus trace-derived tool/process quality."""

    def __init__(self, evaluator: AgentTaskEvaluator | None = None):
        self.evaluator = evaluator or AgentTaskEvaluator()

    def hard_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        if output is None or not isinstance(ground_truth, dict):
            return 0.0
        _, score = self.evaluator.evaluate(output, ground_truth)
        return score

    def process_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        """Reward successful steps, valid tools, observations and grounding."""
        meaningful_steps = [
            step
            for step in trace.steps
            if step.node_type not in {"start", "end"}
            and step.action != "stop"
        ]
        tool_calls = [
            call
            for step in trace.steps
            for call in step.tool_calls
        ]

        if output is None and not meaningful_steps and not tool_calls:
            return 0.0

        score = 0.0
        output_text = "" if output is None else str(output).strip()
        if output_text:
            score += 0.20

        if meaningful_steps:
            successful_steps = sum(step.success for step in meaningful_steps)
            score += 0.30 * (successful_steps / len(meaningful_steps))
            if successful_steps == len(meaningful_steps):
                score += 0.05

        if tool_calls:
            qualities = [_tool_quality(call) for call in tool_calls]
            score += 0.35 * (sum(qualities) / len(qualities))
            if output_text and _is_grounded_in_tools(output_text, tool_calls):
                score += 0.10

        failed_steps = sum(not step.success for step in meaningful_steps)
        if meaningful_steps:
            score -= 0.30 * (failed_steps / len(meaningful_steps))

        retries = sum(
            step.action in {"retry", "repair"}
            for step in trace.steps
        )
        score -= min(0.20, 0.05 * retries)
        return max(0.0, min(1.0, score))


def _tool_quality(call: ToolCallRecord) -> float:
    score = 0.0
    if call.tool_name.strip():
        score += 0.25
    if isinstance(call.tool_args, dict):
        score += 0.25
    if call.success:
        score += 0.25
    if call.tool_result is not None and str(call.tool_result).strip():
        score += 0.25
    return score


def _is_grounded_in_tools(
    output: str,
    tool_calls: list[ToolCallRecord],
) -> bool:
    normalized_output = output.casefold()
    for call in tool_calls:
        if call.tool_result is None:
            continue
        observation = str(call.tool_result).strip().casefold()
        if observation and observation[:100] in normalized_output:
            return True
    return False
