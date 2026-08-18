"""Validation calibration for a frozen workflow scheduler action space."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from awf.config.schema import SchedulerConfig
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.workflow.ir import WorkflowTemplate


@dataclass(frozen=True)
class GateCalibrationResult:
    """Selected gate and its label-based validation simulation."""

    weights: tuple[float, float, float, float]
    threshold: float
    baseline_accuracy: float
    simulated_accuracy: float
    baseline_tokens: int
    simulated_tokens: int
    baseline_latency_seconds: float
    simulated_latency_seconds: float
    simulated_early_exits: int
    feasible_candidates: int
    evaluated_candidates: int
    early_exit_enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "threshold": self.threshold,
            "baseline_accuracy": self.baseline_accuracy,
            "simulated_accuracy": self.simulated_accuracy,
            "accuracy_delta": self.simulated_accuracy - self.baseline_accuracy,
            "baseline_tokens": self.baseline_tokens,
            "simulated_tokens": self.simulated_tokens,
            "token_delta": self.simulated_tokens - self.baseline_tokens,
            "baseline_latency_seconds": self.baseline_latency_seconds,
            "simulated_latency_seconds": self.simulated_latency_seconds,
            "latency_delta_seconds": (
                self.simulated_latency_seconds - self.baseline_latency_seconds
            ),
            "simulated_early_exits": self.simulated_early_exits,
            "feasible_candidates": self.feasible_candidates,
            "evaluated_candidates": self.evaluated_candidates,
            "early_exit_enabled": self.early_exit_enabled,
        }


class FrozenWorkflowGateCalibrator:
    """Grid-select LAS gate parameters on complete validation traces.

    The workflow and its available node catalog are immutable.  For each gate
    candidate we replay only the recorded decisions and simulate a direct exit
    at the earliest eligible terminal artifact.  The benchmark reward checks
    whether that artifact would preserve correctness.  Candidates violating
    the configured validation accuracy guard are discarded, then token and
    latency savings choose among the feasible set.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        workflow: WorkflowTemplate,
        reward_evaluator: RewardEvaluator,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.reward_evaluator = reward_evaluator

    def fit(self, traces: list[ExecutionTrace]) -> GateCalibrationResult:
        if not traces:
            raise ValueError("scheduler calibration requires validation traces")
        baseline_accuracy = sum(float(trace.hard_reward or 0.0) for trace in traces) / len(traces)
        baseline_tokens = sum(trace.total_tokens for trace in traces)
        baseline_latency = sum(trace.total_latency_seconds for trace in traces)
        rows: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
        feasible = 0
        for raw_weights in self.config.calibration_weight_candidates:
            weights = tuple(float(item) for item in raw_weights)
            for threshold in self.config.calibration_thresholds:
                metrics = self._simulate(traces, weights, float(threshold))
                accuracy = metrics["accuracy"]
                if accuracy + self.config.calibration_max_hard_regression < baseline_accuracy:
                    continue
                feasible += 1
                token_saving = baseline_tokens - metrics["tokens"]
                latency_saving = baseline_latency - metrics["latency"]
                # Accuracy is primary. Savings are normalized only for stable
                # tie-breaking across datasets with different absolute costs.
                rank = (
                    accuracy,
                    token_saving / max(baseline_tokens, 1),
                    latency_saving / max(baseline_latency, 1e-9),
                    metrics["early_exits"],
                    -float(threshold),
                )
                rows.append((rank, threshold, {**metrics, "weights": weights}))
        early_exit_enabled = bool(rows)
        if not rows:
            # Scores are bounded inclusively: threshold=1.0 is not a valid
            # zero-intervention sentinel because a perfect score still exits.
            # Represent the frozen full workflow explicitly instead.
            weights = tuple(float(item) for item in self.config.calibration_weight_candidates[0])
            metrics = {
                "accuracy": baseline_accuracy,
                "tokens": baseline_tokens,
                "latency": baseline_latency,
                "early_exits": 0,
            }
            selected_threshold = 1.0
        else:
            _, selected_threshold, metrics = max(rows, key=lambda item: item[0])
            weights = metrics["weights"]
        return GateCalibrationResult(
            weights=weights,
            threshold=float(selected_threshold),
            baseline_accuracy=baseline_accuracy,
            simulated_accuracy=float(metrics["accuracy"]),
            baseline_tokens=baseline_tokens,
            simulated_tokens=int(metrics["tokens"]),
            baseline_latency_seconds=baseline_latency,
            simulated_latency_seconds=float(metrics["latency"]),
            simulated_early_exits=int(metrics["early_exits"]),
            feasible_candidates=feasible,
            evaluated_candidates=(
                len(self.config.calibration_weight_candidates)
                * len(self.config.calibration_thresholds)
            ),
            early_exit_enabled=early_exit_enabled,
        )

    def apply(self, result: GateCalibrationResult) -> SchedulerConfig:
        calibrated = self.config.model_copy(deep=True)
        (
            calibrated.gate_spec_weight,
            calibrated.gate_lite_weight,
            calibrated.gate_agreement_weight,
            calibrated.gate_history_weight,
        ) = result.weights
        # Keep a genuine LAS cascade band. Scores above the calibrated
        # threshold may exit directly, scores in the lower intermediate band
        # are sent to the configured scheduler LLM, and low scores follow the
        # frozen graph. Setting all three thresholds to the same value made
        # the scheduler model unreachable in the earlier implementation.
        calibrated.gate_schedule_threshold = min(
            calibrated.gate_schedule_threshold,
            max(result.threshold - 0.15, 0.0),
        )
        calibrated.gate_early_exit_threshold = result.threshold
        calibrated.gate_high_risk_threshold = max(
            calibrated.gate_high_risk_threshold,
            result.threshold,
        )
        calibrated.gate_direct_early_exit = result.early_exit_enabled
        calibrated.early_exit_enabled = result.early_exit_enabled
        calibrated.gate_enabled = result.early_exit_enabled
        return calibrated

    def _simulate(
        self,
        traces: list[ExecutionTrace],
        weights: tuple[float, ...],
        threshold: float,
    ) -> dict[str, float | int]:
        hard_total = 0.0
        token_total = 0
        latency_total = 0.0
        early_exits = 0
        for trace in traces:
            selected = self._first_safe_candidate(trace, weights, threshold)
            if selected is None:
                hard_total += float(trace.hard_reward or 0.0)
                token_total += trace.total_tokens
                latency_total += trace.total_latency_seconds
                continue
            step_index, artifact = selected
            probe_trace = copy.deepcopy(trace)
            hard_total += float(self.reward_evaluator.hard_reward(
                trace.query_text,
                trace.metadata.get("ground_truth"),
                artifact,
                probe_trace,
            ))
            prefix = trace.steps[:step_index]
            token_total += sum(step.input_tokens + step.output_tokens for step in prefix)
            latency_total += sum(step.duration_seconds for step in prefix)
            early_exits += 1
        return {
            "accuracy": hard_total / len(traces),
            "tokens": token_total,
            "latency": latency_total,
            "early_exits": early_exits,
        }

    def _first_safe_candidate(
        self,
        trace: ExecutionTrace,
        weights: tuple[float, ...],
        threshold: float,
    ) -> tuple[int, Any] | None:
        for index, step in enumerate(trace.steps):
            params = step.metadata.get("scheduler_params", {})
            gate = params.get("_awf_gate", {}) if isinstance(params, dict) else {}
            artifact_node = gate.get("artifact_node") if isinstance(gate, dict) else None
            if not artifact_node or not self._terminal_candidate(step.node_id):
                continue
            score = self._score(gate, weights)
            if float(gate.get("spec_score", 0.0) or 0.0) < 1.0 or score < threshold:
                continue
            state = step.state_before if isinstance(step.state_before, dict) else {}
            outputs = state.get("outputs", {}) if isinstance(state, dict) else {}
            if isinstance(outputs, dict) and outputs.get(artifact_node) is not None:
                return index, outputs[artifact_node]
        return None

    def _terminal_candidate(self, current_node_id: str) -> bool:
        if (
            self.config.gate_node_allowlist
            and current_node_id not in self.config.gate_node_allowlist
        ):
            return False
        node = self.workflow.nodes.get(current_node_id)
        if node is None:
            return False
        haystack = f"{current_node_id} {node.label}".lower()
        return any(term in haystack for term in ("verify", "final", "check", "end"))

    def _score(self, gate: dict[str, Any], weights: tuple[float, ...]) -> float:
        spec = float(gate.get("spec_score", 0.0) or 0.0)
        lite = float(gate.get("lite_score", 0.0) or 0.0)
        agreement = float(gate.get("agreement_score", 0.0) or 0.0)
        history = float(gate.get("history_reliability", 0.0) or 0.0)
        denominator = max(sum(weights), 1e-12)
        if self.config.gate_formula == "las":
            return (
                weights[0] * spec
                + weights[1] * lite
                + weights[2] * (agreement - 1.0)
                + weights[3] * (1.0 - history)
            ) / denominator
        return sum(
            value * weight
            for value, weight in zip((spec, lite, agreement, history), weights)
        ) / denominator
