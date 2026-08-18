"""Hard-constrained depth-1 gate search for S-CWU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from awf.optimizer.counterfactual import CounterfactualEvaluator
from awf.workflow.gates import GateSpec


@dataclass(frozen=True)
class GateSearchResult:
    """The best joint applicability hypothesis for one workflow patch."""

    gate: GateSpec
    composed_result: dict[str, Any]
    metrics: dict[str, Any]
    rank_key: tuple[float, ...]
    positive: bool
    report_score: float


class SelectiveGateSearcher:
    """Enumerate interpretable stumps and select under hard constraints.

    Correctness is never traded against token or latency savings. Among safe
    gates, comparison is lexicographic:

    hard effect -> equal-hard failure process effect -> tokens -> latency ->
    gate/edit simplicity.
    """

    def __init__(
        self,
        *,
        features: Iterable[str],
        min_leaf_support: int = 2,
        hard_success_threshold: float = 1.0,
        hard_regression_tolerance: float = 0.0,
        min_effect: float = 0.01,
        simplicity_penalty: float = 1e-3,
    ):
        feature_list = tuple(features)
        if not feature_list or len(feature_list) != len(set(feature_list)):
            raise ValueError("gate features must be non-empty and unique")
        if min_leaf_support <= 0:
            raise ValueError("min_leaf_support must be positive")
        if not 0.0 <= hard_success_threshold <= 1.0:
            raise ValueError("hard_success_threshold must be in [0, 1]")
        if hard_regression_tolerance < 0.0:
            raise ValueError("hard_regression_tolerance cannot be negative")
        if min_effect < 0.0 or simplicity_penalty < 0.0:
            raise ValueError("effect and simplicity thresholds cannot be negative")
        self.features = feature_list
        self.min_leaf_support = min_leaf_support
        self.hard_success_threshold = hard_success_threshold
        self.hard_regression_tolerance = hard_regression_tolerance
        self.min_effect = min_effect
        self.simplicity_penalty = simplicity_penalty

    def search(
        self,
        result: dict[str, Any],
        *,
        fit_trace_ids: set[str],
        protected_trace_ids: set[str],
        edit_distance: float,
        failure_mode: bool,
    ) -> GateSearchResult:
        """Return the best safe gate; ``never`` means reject the update."""
        rows = result.get("per_query_results")
        if not isinstance(rows, list) or not rows or not fit_trace_ids:
            raise ValueError("gate search requires measured representative rows")
        row_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("counterfactual rows must be mappings")
            trace_id = row.get("trace_id")
            if (
                not isinstance(trace_id, str)
                or not trace_id
                or trace_id in row_by_id
            ):
                raise ValueError("counterfactual trace ids must be unique")
            row_by_id[trace_id] = row
        if not fit_trace_ids <= set(row_by_id):
            raise ValueError("gate-fit measurements are incomplete")
        if not protected_trace_ids <= set(row_by_id):
            raise ValueError("protected success measurements are incomplete")

        candidates = self._enumerate_gates(
            [row_by_id[trace_id] for trace_id in sorted(fit_trace_ids)]
        )
        evaluated: list[GateSearchResult] = []
        for gate in candidates:
            composed = CounterfactualEvaluator.compose_with_gate(result, gate)
            assessment = self._assess(
                composed,
                gate=gate,
                fit_trace_ids=fit_trace_ids,
                protected_trace_ids=protected_trace_ids,
                edit_distance=edit_distance,
                failure_mode=failure_mode,
            )
            if assessment is not None:
                evaluated.append(assessment)
        if not evaluated:
            raise ValueError("no gate has complete, hard-safe measurements")

        # Deterministic final tie-break: simpler rule, then canonical fingerprint.
        best = max(
            evaluated,
            key=lambda item: (
                item.rank_key,
                -item.gate.complexity,
                item.gate.fingerprint,
            ),
        )
        return best

    def _enumerate_gates(
        self,
        rows: list[dict[str, Any]],
    ) -> list[GateSpec]:
        gates = [GateSpec(kind="never"), GateSpec(kind="always")]
        seen = {gate.fingerprint for gate in gates}
        for feature in self.features:
            values: list[float] = []
            valid = True
            for row in rows:
                feature_map = row.get("gate_features")
                raw = (
                    feature_map.get(feature)
                    if isinstance(feature_map, dict)
                    else None
                )
                if isinstance(raw, bool):
                    valid = False
                    break
                try:
                    value = float(raw)
                except (TypeError, ValueError, OverflowError):
                    valid = False
                    break
                if not math.isfinite(value):
                    valid = False
                    break
                values.append(value)
            if not valid:
                continue
            unique = sorted(set(values))
            for left, right in zip(unique, unique[1:]):
                threshold = left / 2.0 + right / 2.0
                low_support = sum(value <= threshold for value in values)
                high_support = len(values) - low_support
                if (
                    low_support < self.min_leaf_support
                    or high_support < self.min_leaf_support
                ):
                    continue
                for operator in ("le", "gt"):
                    gate = GateSpec(
                        kind="threshold",
                        feature=feature,
                        operator=operator,
                        threshold=threshold,
                    )
                    if gate.fingerprint not in seen:
                        seen.add(gate.fingerprint)
                        gates.append(gate)
        return gates

    def _assess(
        self,
        composed: dict[str, Any],
        *,
        gate: GateSpec,
        fit_trace_ids: set[str],
        protected_trace_ids: set[str],
        edit_distance: float,
        failure_mode: bool,
    ) -> GateSearchResult | None:
        rows = composed["per_query_results"]
        by_id = {str(row["trace_id"]): row for row in rows}
        fit_rows = [by_id[trace_id] for trace_id in sorted(fit_trace_ids)]

        hard_deltas: list[float] = []
        process_effects: list[float] = []
        token_savings: list[float] = []
        latency_savings: list[float] = []
        utility_effects: list[float] = []
        for row in fit_rows:
            original_hard = self._finite(row.get("original_hard"))
            candidate_hard = self._finite(row.get("candidate_hard"))
            original_process = self._finite(row.get("original_process"))
            candidate_process = self._finite(row.get("candidate_process"))
            original_tokens = self._finite(row.get("original_total_tokens"))
            candidate_tokens = self._finite(row.get("candidate_total_tokens"))
            original_latency = self._finite(
                row.get("original_latency_seconds")
            )
            candidate_latency = self._finite(
                row.get("candidate_latency_seconds")
            )
            delta_u = self._finite(row.get("delta_u"))
            measurements = (
                original_hard,
                candidate_hard,
                original_process,
                candidate_process,
                original_tokens,
                candidate_tokens,
                original_latency,
                candidate_latency,
                delta_u,
            )
            if any(value is None for value in measurements):
                return None
            if (
                not 0.0 <= original_hard <= 1.0
                or not 0.0 <= candidate_hard <= 1.0
                or not 0.0 <= original_process <= 1.0
                or not 0.0 <= candidate_process <= 1.0
                or original_tokens < 0.0
                or candidate_tokens < 0.0
                or original_latency < 0.0
                or candidate_latency < 0.0
            ):
                return None
            hard_deltas.append(candidate_hard - original_hard)
            process_effects.append(
                candidate_process - original_process
                if (
                    abs(candidate_hard - original_hard) <= 1e-12
                    and original_hard < self.hard_success_threshold
                )
                else 0.0
            )
            token_savings.append(
                self._bounded_saving_ratio(
                    original_tokens,
                    candidate_tokens,
                    floor=1.0,
                )
            )
            latency_savings.append(
                self._bounded_saving_ratio(
                    original_latency,
                    candidate_latency,
                    floor=1e-9,
                )
            )
            utility_effects.append(delta_u)

        protected_regressions = 0
        for trace_id in protected_trace_ids:
            row = by_id[trace_id]
            original = self._finite(row.get("original_hard"))
            candidate = self._finite(row.get("candidate_hard"))
            if original is None or candidate is None:
                return None
            if candidate < original - 1e-12:
                protected_regressions += 1

        mean_hard = self._mean(hard_deltas)
        if (
            protected_regressions
            or mean_hard < -self.hard_regression_tolerance - 1e-12
        ):
            return None
        mean_process = (
            self._mean(process_effects) if failure_mode else 0.0
        )
        mean_tokens = self._mean(token_savings)
        mean_latency = self._mean(latency_savings)
        mean_utility = self._mean(utility_effects)
        effective_edit_distance = 0.0 if gate.kind == "never" else edit_distance
        simplicity = (
            self.simplicity_penalty
            * (float(gate.complexity) + max(float(effective_edit_distance), 0.0))
        )
        rank_key = (
            self._material_effect(mean_hard),
            self._material_effect(mean_process),
            self._material_effect(mean_tokens),
            self._material_effect(mean_latency),
            -simplicity,
        )
        baseline_rank = (0.0, 0.0, 0.0, 0.0, 0.0)
        positive = rank_key > baseline_rank and gate.kind != "never"
        coverage = self._mean(
            [float(bool(row.get("gate_applied"))) for row in fit_rows]
        )
        metrics = {
            "hard_safe": True,
            "mean_hard_delta": mean_hard,
            "failure_process_delta": mean_process,
            "mean_token_saving_ratio": mean_tokens,
            "mean_latency_saving_ratio": mean_latency,
            "mean_utility_delta": mean_utility,
            "coverage": coverage,
            "applied_support": sum(
                bool(row.get("gate_applied")) for row in fit_rows
            ),
            "fit_support": len(fit_rows),
            "protected_support": len(protected_trace_ids),
            "protected_regressions": protected_regressions,
            "gate_complexity": gate.complexity,
            "edit_distance": effective_edit_distance,
            "rank_key": list(rank_key),
        }
        return GateSearchResult(
            gate=gate,
            composed_result=composed,
            metrics=metrics,
            rank_key=rank_key,
            positive=positive,
            report_score=mean_utility - simplicity,
        )

    def _material_effect(self, value: float) -> float:
        if abs(value) <= self.min_effect:
            return 0.0
        # Bound noisy relative latency ratios so they cannot destabilize trace
        # serialization or later tie-breaks. This never changes higher-priority
        # hard/process dimensions.
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _bounded_saving_ratio(
        original: float,
        candidate: float,
        *,
        floor: float,
    ) -> float:
        if original <= 0.0:
            return 0.0 if candidate <= 0.0 else -1.0
        ratio = (original - candidate) / max(original, floor)
        if not math.isfinite(ratio):
            return 1.0 if original > candidate else -1.0
        return max(-1.0, min(1.0, ratio))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _finite(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None
