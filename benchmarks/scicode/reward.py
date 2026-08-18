"""Hard-constrained SciCode reward with failure-only process shaping."""

from __future__ import annotations

from collections import Counter
from typing import Any

from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from benchmarks.scicode.evaluator import (
    SciCodeEvaluationResult,
    SciCodeEvaluator,
)


class SciCodeReward(RewardEvaluator):
    """Official-style binary subproblem success plus grounded diagnostics."""

    def __init__(self, evaluator: SciCodeEvaluator) -> None:
        self.evaluator = evaluator

    def hard_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        result = self.evaluator.evaluate_detailed(output, ground_truth)
        self._record(trace, ground_truth, result)
        return 1.0 if result.all_passed else 0.0

    def process_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        result = self._cached(trace, ground_truth)
        if result is None:
            # Compatibility for direct process_reward calls; the experiment
            # runner invokes hard_reward first, so normal evaluation runs once.
            result = self.evaluator.evaluate_detailed(output, ground_truth)
            self._record(trace, ground_truth, result)

        meaningful = [
            step
            for step in trace.steps
            if step.node_type not in {"start", "end"}
            and step.action != "stop"
        ]
        execution_clean = bool(
            meaningful and all(step.success for step in meaningful)
        )
        policy_rejected = "unsafe_code" in result.outcome_codes
        components = (
            {
                "compiled": 0.0,
                "entry_point_present": 0.0,
                "test_execution_coverage": 0.0,
                "test_pass_rate": 0.0,
                "clean_workflow_execution": 0.0,
            }
            if policy_rejected
            else {
                "compiled": 0.05 if result.compiled else 0.0,
                "entry_point_present": (
                    0.05 if result.entry_point_present else 0.0
                ),
                "test_execution_coverage": 0.15 * result.execution_rate,
                "test_pass_rate": 0.65 * result.pass_rate,
                "clean_workflow_execution": (
                    0.10 if execution_clean else 0.0
                ),
            }
        )
        raw = sum(components.values())
        score = max(0.0, min(1.0, raw))
        diagnostic = trace.metadata["scicode_evaluation"]
        diagnostic.update(
            {
                "execution_clean": execution_clean,
                "policy_rejected": policy_rejected,
                "component_scores": components,
                "process_reward": score,
            }
        )
        return score

    @staticmethod
    def _record(
        trace: ExecutionTrace,
        ground_truth: Any,
        result: SciCodeEvaluationResult,
    ) -> None:
        task_id = (
            str(ground_truth.get("task_id"))
            if isinstance(ground_truth, dict)
            else ""
        )
        trace.metadata["scicode_evaluation"] = {
            "task_id": task_id,
            "integrity_policy": "static_introspection_denylist_v1",
            "passed_tests": result.passed_tests,
            "total_tests": result.total_tests,
            "tests_executed": result.tests_executed,
            "execution_rate": result.execution_rate,
            "pass_rate": result.pass_rate,
            "all_passed": result.all_passed,
            "compiled": result.compiled,
            "entry_point_present": result.entry_point_present,
            "outcome_counts": dict(Counter(result.outcome_codes)),
        }
        trace.metadata["_scicode_result"] = {
            "task_id": task_id,
            "passed_tests": result.passed_tests,
            "total_tests": result.total_tests,
            "tests_executed": result.tests_executed,
            "compiled": result.compiled,
            "entry_point_present": result.entry_point_present,
            "outcome_codes": list(result.outcome_codes),
        }

    @staticmethod
    def _cached(
        trace: ExecutionTrace,
        ground_truth: Any,
    ) -> SciCodeEvaluationResult | None:
        cached = trace.metadata.get("_scicode_result")
        task_id = (
            str(ground_truth.get("task_id"))
            if isinstance(ground_truth, dict)
            else ""
        )
        if not isinstance(cached, dict) or cached.get("task_id") != task_id:
            return None
        return SciCodeEvaluationResult(
            passed_tests=int(cached["passed_tests"]),
            total_tests=int(cached["total_tests"]),
            tests_executed=int(cached["tests_executed"]),
            compiled=bool(cached["compiled"]),
            entry_point_present=bool(cached["entry_point_present"]),
            outcome_codes=tuple(cached["outcome_codes"]),
        )
