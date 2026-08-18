"""Round-local, workflow-aware success and failure trace buffering."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable

from awf.trace.schema import ExecutionTrace


@dataclass(frozen=True)
class OptimizationTraceBatch:
    """One version-matched, round-local counterfactual trace batch."""

    failures: list[ExecutionTrace]
    efficiency_anchors: list[ExecutionTrace]
    success_guards: list[ExecutionTrace]
    counterfactual_batch: list[ExecutionTrace]
    # The complete current-version optimization epoch, before failure-heavy
    # counterfactual sampling. S-CWU uses a bounded deterministic subset of
    # these rows to fit deployment applicability without changing the default
    # failure-only behavior.
    representative_traces: list[ExecutionTrace]


class FailureBuffer:
    """Collects and organizes execution traces into success and failure bins.

    Failure traces are used by the optimizer to identify and fix issues.
    Success traces can serve as references for what works.
    """

    def __init__(self, capacity: int = 100,
                 success_threshold: float = 1.0):
        """
        Args:
            capacity: Maximum number of traces to keep per bin.
            success_threshold: Hard reward threshold for considering a trace successful.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        self.capacity = capacity
        self.success_threshold = success_threshold
        self._success_traces: list[ExecutionTrace] = []
        self._failure_traces: list[ExecutionTrace] = []

    @property
    def successes(self) -> list[ExecutionTrace]:
        return list(self._success_traces)

    @property
    def failures(self) -> list[ExecutionTrace]:
        return list(self._failure_traces)

    @property
    def total_traces(self) -> int:
        return len(self._success_traces) + len(self._failure_traces)

    def add(self, trace: ExecutionTrace) -> None:
        """Add a trace to the appropriate bin.

        Uses hard_reward if available, otherwise falls back to trace.success.
        """
        reward = trace.hard_reward
        if reward is not None:
            is_success = reward >= self.success_threshold
        else:
            is_success = trace.success

        if is_success:
            self._success_traces.append(trace)
            if len(self._success_traces) > self.capacity:
                self._success_traces.pop(0)
        else:
            self._failure_traces.append(trace)
            if len(self._failure_traces) > self.capacity:
                self._failure_traces.pop(0)

    def extend(self, traces: list[ExecutionTrace]) -> None:
        """Add multiple traces."""
        for trace in traces:
            self.add(trace)

    def get_failure_rate(self) -> float:
        """Return the proportion of failure traces."""
        total = self.total_traces
        if total == 0:
            return 0.0
        return len(self._failure_traces) / total

    def clear(self) -> None:
        """Clear all buffered traces."""
        self._success_traces.clear()
        self._failure_traces.clear()

    def consume_round(
        self,
        workflow_name: str,
        workflow_version: str,
        *,
        success_fraction: float = 0.2,
        min_success_guards: int = 0,
        max_failures: int | None = None,
    ) -> tuple[list[ExecutionTrace], list[ExecutionTrace], list[ExecutionTrace]]:
        """Consume traces for one workflow version and build ``B_cf``.

        Optimizer traces describe the workflow that produced them.  Reusing a
        trace after the workflow has changed creates an invalid baseline, so a
        round consumes the buffer and only returns exact name/version matches.
        The counterfactual batch is failure-heavy, with a small, deterministic
        sample of recent successes to detect regressions.

        Returns:
            ``(failures, sampled_successes, counterfactual_batch)``.
        """
        batch = self.consume_optimization_round(
            workflow_name,
            workflow_version,
            success_fraction=success_fraction,
            min_success_guards=min_success_guards,
            max_failures=max_failures,
        )
        return (
            batch.failures,
            batch.success_guards,
            batch.counterfactual_batch,
        )

    def consume_optimization_round(
        self,
        workflow_name: str,
        workflow_version: str,
        *,
        success_fraction: float = 0.2,
        min_success_guards: int = 0,
        max_failures: int | None = None,
        efficiency_enabled: bool = False,
        efficiency_fraction: float = 0.2,
        efficiency_min_relative_cost: float = 1.25,
        max_efficiency_anchors: int = 5,
        efficiency_cost: Callable[[ExecutionTrace], float] | None = None,
    ) -> OptimizationTraceBatch:
        """Consume one round with hard-failure and optional efficiency triggers.

        Efficiency anchors are explicit hard successes whose configured runtime
        cost is both in the top ``efficiency_fraction`` and at least
        ``efficiency_min_relative_cost`` times the median success cost. They are
        removed from the independently sampled regression-guard set.
        """
        if not 0.0 <= success_fraction < 1.0:
            raise ValueError("success_fraction must be in [0, 1)")
        if min_success_guards < 0:
            raise ValueError("min_success_guards cannot be negative")
        if max_failures is not None and max_failures <= 0:
            raise ValueError("max_failures must be positive when provided")
        if not 0.0 < efficiency_fraction <= 1.0:
            raise ValueError("efficiency_fraction must be in (0, 1]")
        if efficiency_min_relative_cost < 1.0:
            raise ValueError(
                "efficiency_min_relative_cost must be at least 1"
            )
        if max_efficiency_anchors <= 0:
            raise ValueError("max_efficiency_anchors must be positive")
        if efficiency_enabled and efficiency_cost is None:
            raise ValueError(
                "efficiency_cost is required when efficiency is enabled"
            )

        def matches(trace: ExecutionTrace) -> bool:
            return (
                trace.workflow_name == workflow_name
                and trace.workflow_version == workflow_version
            )

        failures = [t for t in self._failure_traces if matches(t)]
        successes = [t for t in self._success_traces if matches(t)]
        representative_traces = failures + successes

        # A buffer represents one optimization epoch.  Clear incompatible and
        # consumed traces alike so a no-op round cannot accidentally compare a
        # later candidate against stale executions of the same workflow version.
        self.clear()

        if max_failures is not None:
            failures = failures[-max_failures:]

        efficiency_anchors: list[ExecutionTrace] = []
        if efficiency_enabled and successes:
            scored: list[tuple[float, int, ExecutionTrace]] = []
            for index, trace in enumerate(successes):
                # ``trace.success`` alone is not sufficient for an efficiency
                # update: performance preservation needs an observed hard label.
                if (
                    trace.hard_reward is None
                    or trace.hard_reward < self.success_threshold
                ):
                    continue
                try:
                    score = float(efficiency_cost(trace))
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(score) and score > 0.0:
                    scored.append((score, index, trace))

            if scored:
                median_cost = statistics.median(
                    score for score, _, _ in scored
                )
                relative_threshold = (
                    median_cost * efficiency_min_relative_cost
                )
                qualifying = [
                    item for item in scored if item[0] >= relative_threshold
                ]
                # Prefer higher cost, then the more recent trace.
                qualifying.sort(
                    key=lambda item: (item[0], item[1]),
                    reverse=True,
                )
                fraction_target = math.ceil(
                    len(scored) * efficiency_fraction
                )
                target = min(
                    fraction_target,
                    max_efficiency_anchors,
                )
                efficiency_anchors = [
                    trace for _, _, trace in qualifying[:target]
                ]

        efficiency_ids = {id(trace) for trace in efficiency_anchors}
        guard_candidates = [
            trace for trace in successes if id(trace) not in efficiency_ids
        ]
        trigger_count = len(failures) + len(efficiency_anchors)
        if not trigger_count or not guard_candidates:
            sampled_successes: list[ExecutionTrace] = []
        else:
            # By default, preserve the advertised fraction exactly. In
            # particular, one failure with a 20% target selects zero successes
            # rather than silently turning the batch into a 50/50 mixture.
            fraction_target = (
                math.floor(
                    trigger_count
                    * success_fraction
                    / (1.0 - success_fraction)
                )
                if success_fraction
                else 0
            )
            # A caller may deliberately override the fraction for tiny batches
            # by configuring a non-zero minimum regression guard count.
            target = max(fraction_target, min_success_guards)
            sampled_successes = (
                guard_candidates[-target:] if target else []
            )

        counterfactual_batch = (
            failures + efficiency_anchors + sampled_successes
        )
        return OptimizationTraceBatch(
            failures=failures,
            efficiency_anchors=efficiency_anchors,
            success_guards=sampled_successes,
            counterfactual_batch=counterfactual_batch,
            representative_traces=representative_traces,
        )
