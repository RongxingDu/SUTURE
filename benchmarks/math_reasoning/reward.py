"""Math hard reward and dense process signals."""

from __future__ import annotations

import re
from typing import Any

from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from benchmarks.math_reasoning.evaluator import MathEvaluator


class MathReward(RewardEvaluator):
    """Exact-answer reward plus correctness-grounded process diagnostics."""

    def __init__(self, evaluator: MathEvaluator | None = None):
        self.evaluator = evaluator or MathEvaluator()

    def hard_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        if output is None:
            return 0.0
        ground_truth_answer = self.evaluator.normalize_ground_truth(ground_truth)
        return (
            1.0
            if self.evaluator.evaluate(str(output), ground_truth_answer)
            else 0.0
        )

    def process_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        """Score reasoning signals while preventing format-only reward gaming.

        Formatting and execution-health evidence is worth at most ``0.30``.
        Most credit requires either a correct intermediate solve answer or a
        correct final answer. A wrong final answer is explicitly capped at
        ``0.35`` even when an earlier solve step happened to be correct.
        """
        if output is None:
            trace.metadata["math_evaluation"] = {
                "output_present": False,
                "process_reward": 0.0,
                "reason": "missing_final_output",
            }
            return 0.0

        final_text = str(output)
        reasoning_text = _trace_output(trace, "solve") or final_text
        lowered = reasoning_text.lower()
        final_answer = self.evaluator.extract_answer(final_text)
        solve_answer = self.evaluator.extract_answer(reasoning_text)
        ground_truth_answer = self.evaluator.normalize_ground_truth(
            ground_truth
        )

        has_reasoning_structure = bool(re.search(
            r"(?:step\s*\d+|first|second|third|next|finally|because|therefore)",
            lowered,
        ))
        has_visible_calculations = bool(re.search(
            r"(?:=|[+\-*/]\s*\d|compute|calculate|substitut)",
            lowered,
        ))
        has_explicit_final_marker = bool(re.search(
            r"(?:verified\s*:|passed?\s*:|\\boxed\{|####|"
            r"final\s+answer|answer\s*(?:is|=|:))",
            final_text,
            re.IGNORECASE,
        ))
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
        solve_final_consistent = bool(
            final_answer
            and solve_answer
            and self.evaluator.compare_answers(
                final_answer,
                solve_answer,
            )
        )
        final_answer_correct = bool(
            final_answer
            and ground_truth_answer
            and self.evaluator.compare_answers(
                final_answer,
                ground_truth_answer,
            )
        )
        solve_answer_correct = bool(
            solve_answer
            and ground_truth_answer
            and self.evaluator.compare_answers(
                solve_answer,
                ground_truth_answer,
            )
        )

        components = {
            "final_answer_extractable": 0.05 if final_answer else 0.0,
            "explicit_final_marker": (
                0.05 if has_explicit_final_marker else 0.0
            ),
            "reasoning_structure": (
                0.04 if has_reasoning_structure else 0.0
            ),
            "visible_calculations": (
                0.06 if has_visible_calculations else 0.0
            ),
            "execution_clean": 0.05 if execution_clean else 0.0,
            "solve_final_consistency": (
                0.05 if solve_final_consistent else 0.0
            ),
            "solve_answer_correct": (
                0.25 if solve_answer_correct else 0.0
            ),
            "final_answer_correct": (
                0.45 if final_answer_correct else 0.0
            ),
        }
        raw_score = sum(components.values())
        incorrect_final_cap_applied = (
            not final_answer_correct and raw_score > 0.35
        )
        score = min(raw_score, 0.35) if not final_answer_correct else raw_score
        score = max(0.0, min(1.0, score))

        trace.metadata["math_evaluation"] = {
            "output_present": True,
            "ground_truth_available": bool(ground_truth_answer),
            "final_answer_extractable": bool(final_answer),
            "solve_answer_extractable": bool(solve_answer),
            "final_answer_correct": final_answer_correct,
            "solve_answer_correct": solve_answer_correct,
            "solve_final_consistent": solve_final_consistent,
            "reasoning_structure_present": has_reasoning_structure,
            "visible_calculations_present": has_visible_calculations,
            "explicit_final_marker_present": has_explicit_final_marker,
            "execution_clean": execution_clean,
            "component_scores": components,
            "raw_process_score": raw_score,
            "incorrect_final_cap_applied": incorrect_final_cap_applied,
            "process_reward": score,
        }
        return score


def _trace_output(trace: ExecutionTrace, node_id: str) -> Any:
    for step in reversed(trace.steps):
        for state in (step.state_after, step.state_before):
            outputs = state.get("outputs") if isinstance(state, dict) else None
            if isinstance(outputs, dict) and outputs.get(node_id) is not None:
                return outputs[node_id]
    return None
