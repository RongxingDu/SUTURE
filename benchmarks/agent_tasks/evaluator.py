"""Task-specific success evaluation for general agent tasks."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CustomCriterion = Callable[
    [Any, dict[str, Any], dict[str, Any]],
    bool | float,
]


class AgentTaskEvaluator:
    """Evaluate explicit criteria, with a conservative expected-outcome fallback.

    Custom criteria are deny-by-default.  A criterion with ``type: custom``
    must name a callback registered at construction time.
    """

    def __init__(
        self,
        custom_evaluators: Mapping[str, CustomCriterion] | None = None,
    ):
        self.custom_evaluators = dict(custom_evaluators or {})

    def evaluate(
        self,
        output: Any,
        ground_truth: dict[str, Any],
    ) -> tuple[bool, float]:
        if output is None or not isinstance(ground_truth, dict):
            return False, 0.0

        criteria = ground_truth.get("success_criteria") or []
        if not criteria:
            return _evaluate_expected_outcome(
                output,
                ground_truth.get("expected_outcome"),
            )

        results: list[float] = []
        for criterion in criteria:
            if not isinstance(criterion, dict):
                results.append(0.0)
                continue
            results.append(
                self._evaluate_criterion(output, criterion, ground_truth)
            )

        if not results:
            return False, 0.0
        score = sum(results) / len(results)
        return all(result >= 1.0 for result in results), score

    def _evaluate_criterion(
        self,
        output: Any,
        criterion: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:
        criterion_type = str(criterion.get("type", "contains")).lower()
        target = criterion.get("target")
        output_str = str(output)

        if criterion_type == "exact_match":
            if target is None:
                return 0.0
            return float(_normalized(output_str) == _normalized(str(target)))

        if criterion_type == "contains":
            targets = (
                list(target)
                if isinstance(target, Sequence)
                and not isinstance(target, (str, bytes))
                else [target]
            )
            if not targets or any(
                value is None or not str(value).strip() for value in targets
            ):
                return 0.0
            matches = [
                _normalized(str(value)) in _normalized(output_str)
                for value in targets
            ]
            return sum(matches) / len(matches)

        if criterion_type == "regex":
            if target is None or not str(target):
                return 0.0
            try:
                return float(bool(re.search(str(target), output_str)))
            except re.error:
                return 0.0

        if criterion_type == "custom":
            name = criterion.get("name") or criterion.get("evaluator")
            callback = self.custom_evaluators.get(str(name)) if name else None
            if callback is None:
                return 0.0
            try:
                result = callback(output, criterion, ground_truth)
            except Exception:
                return 0.0
            if isinstance(result, bool):
                return float(result)
            if isinstance(result, (int, float)):
                return max(0.0, min(1.0, float(result)))
            return 0.0

        return 0.0


def _evaluate_expected_outcome(
    output: Any,
    expected: Any,
) -> tuple[bool, float]:
    """Conservatively compare expected values when criteria are absent."""
    if expected is None or expected == "" or expected == {} or expected == []:
        return False, 0.0

    if isinstance(expected, Mapping):
        if isinstance(output, Mapping):
            results = [
                _values_match(output.get(key), value)
                for key, value in expected.items()
            ]
        else:
            leaves = _scalar_leaves(expected)
            results = [
                _normalized(str(value)) in _normalized(str(output))
                for value in leaves
                if str(value).strip()
            ]
    elif isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes)
    ):
        leaves = _scalar_leaves(expected)
        results = [
            _normalized(str(value)) in _normalized(str(output))
            for value in leaves
            if str(value).strip()
        ]
    else:
        results = [_values_match(output, expected)]

    if not results:
        return False, 0.0
    score = sum(results) / len(results)
    return all(results), score


def _scalar_leaves(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        leaves: list[Any] = []
        for nested in value.values():
            leaves.extend(_scalar_leaves(nested))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves = []
        for nested in value:
            leaves.extend(_scalar_leaves(nested))
        return leaves
    return [value]


def _values_match(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(expected, (Mapping, list, tuple)):
        return _normalized(str(actual)) == _normalized(str(expected))
    actual_text = _normalized(str(actual))
    expected_text = _normalized(str(expected))
    return bool(expected_text) and (
        actual_text == expected_text or expected_text in actual_text
    )


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
