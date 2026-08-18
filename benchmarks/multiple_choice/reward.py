"""Correctness-grounded reward for GPQA and MMLU."""

from __future__ import annotations

from typing import Any

from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from benchmarks.multiple_choice.evaluator import MultipleChoiceEvaluator


class MultipleChoiceReward(RewardEvaluator):
    """Exact accuracy plus bounded, non-semantic process diagnostics.

    Correctness contributes 0.80 of the process reward.  The remaining credit
    comes only from satisfying the public output contract and completing the
    workflow without failed steps.  Consequently, polished but wrong answers
    cannot receive more than 0.20.
    """

    def __init__(
        self,
        evaluator: MultipleChoiceEvaluator | None = None,
    ) -> None:
        self.evaluator = evaluator or MultipleChoiceEvaluator()

    def hard_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        correct, parsed = self.evaluator.evaluate_detailed(
            output,
            ground_truth,
        )
        _record_diagnostics(
            trace,
            {
                "parse_mode": parsed.parse_mode,
                "answer_extractable": parsed.valid,
                "final_answer_correct": correct,
                "hard_reward": 1.0 if correct else 0.0,
            },
        )
        return 1.0 if correct else 0.0

    def process_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        correct, parsed = self.evaluator.evaluate_detailed(
            output,
            ground_truth,
        )
        meaningful_steps = [
            step
            for step in trace.steps
            if step.node_type not in {"start", "end"}
            and step.action != "stop"
        ]
        execution_clean = bool(
            meaningful_steps
            and all(step.success for step in meaningful_steps)
        )

        components = {
            "strict_output_contract": (
                0.10
                if parsed.parse_mode == "strict_final_marker"
                else 0.0
            ),
            "execution_clean": 0.10 if execution_clean else 0.0,
            "final_answer_correct": 0.80 if correct else 0.0,
        }
        raw_score = sum(components.values())
        incorrect_final_cap_applied = not correct and raw_score > 0.20
        score = min(raw_score, 0.20) if not correct else raw_score
        score = max(0.0, min(1.0, score))

        _record_diagnostics(
            trace,
            {
                "parse_mode": parsed.parse_mode,
                "answer_extractable": parsed.valid,
                "ground_truth_available": (
                    self.evaluator.normalize_ground_truth(ground_truth)
                    is not None
                ),
                "final_answer_correct": correct,
                "execution_clean": execution_clean,
                "component_scores": components,
                "raw_process_score": raw_score,
                "incorrect_final_cap_applied": (
                    incorrect_final_cap_applied
                ),
                "process_reward": score,
            },
        )
        return score


def _record_diagnostics(
    trace: ExecutionTrace,
    diagnostic: dict[str, Any],
) -> None:
    """Store outcome categories without raw answers or dataset labels."""
    existing = trace.metadata.get("reward_diagnostics")
    reward_diagnostics = (
        dict(existing)
        if isinstance(existing, dict)
        else {}
    )
    reward_diagnostics["multiple_choice"] = diagnostic
    trace.metadata["reward_diagnostics"] = reward_diagnostics
