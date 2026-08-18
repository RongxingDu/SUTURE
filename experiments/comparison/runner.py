"""Serial reference-paired ABBA comparison runner."""

from __future__ import annotations

import inspect
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from experiments.comparison.adapters import (
    MethodAdapter,
    method_identity_sha256,
    validate_adapters_ready,
)
from experiments.comparison.models import (
    ComparisonSample,
    FrozenMethodRoster,
    PhaseTelemetry,
)


ScoreCallable = Callable[
    [ComparisonSample, Any],
    float | Awaitable[float],
]

_PUBLIC_INFERENCE_METADATA = frozenset({"entry_point"})
_PUBLIC_RESULT_METADATA = frozenset(
    {
        "failure_type",
        "runtime",
        "runtime_success",
        "workflow_attempt_count",
        "workflow_retry_max_attempts",
        "workflow_retry_semantics",
        "workflow_retry_wait_seconds",
        "workflow_version",
    }
)


@dataclass(frozen=True, slots=True)
class ScheduledInference:
    sample: ComparisonSample
    pair_id: str
    reference_method: str
    comparator_method: str
    method_name: str
    order_position: int
    replicate: int
    block_pattern: str


def build_reference_abba_schedule(
    samples: Sequence[ComparisonSample],
    method_names: Sequence[str],
    reference_method: str,
) -> list[ScheduledInference]:
    """Return deterministic, counterbalanced A/B replicate blocks."""
    names = tuple(method_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("method_names must contain at least two unique names")
    if reference_method not in names:
        raise ValueError("reference_method is not in method_names")
    schedule: list[ScheduledInference] = []
    comparators = [name for name in names if name != reference_method]
    for sample_position, sample in enumerate(samples):
        for comparator_position, comparator in enumerate(comparators):
            pair_id = f"{reference_method}__vs__{comparator}"
            use_abba = (sample_position + comparator_position) % 2 == 0
            if use_abba:
                pattern = "ABBA"
                block = (
                    (reference_method, 1),
                    (comparator, 1),
                    (comparator, 2),
                    (reference_method, 2),
                )
            else:
                pattern = "BAAB"
                block = (
                    (comparator, 1),
                    (reference_method, 1),
                    (reference_method, 2),
                    (comparator, 2),
                )
            for position, (method_name, replicate) in enumerate(
                block,
                start=1,
            ):
                schedule.append(
                    ScheduledInference(
                        sample=sample,
                        pair_id=pair_id,
                        reference_method=reference_method,
                        comparator_method=comparator,
                        method_name=method_name,
                        order_position=position,
                        replicate=replicate,
                        block_pattern=pattern,
                    )
                )
    return schedule


class ComparisonRunner:
    """Evaluate frozen methods serially on identical examples.

    Selection accepts validation examples only.  Held-out execution requires a
    validation-created ``FrozenMethodRoster`` with exact method identities.
    No method winner is chosen from test outcomes by this class.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        adapters: Sequence[MethodAdapter],
        scorer: ScoreCallable,
        reference_method: str | None = None,
    ) -> None:
        self.adapters = list(adapters)
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self.scorer = scorer
        self.reference_method = (
            reference_method
            if reference_method is not None
            else (self.adapters[0].name if self.adapters else "")
        )

    async def run_selection(
        self,
        samples: Sequence[ComparisonSample],
    ) -> dict[str, Any]:
        """Run validation-only comparison and freeze the method roster."""
        validated = _validate_samples(samples, expected_split="validation")
        result = await self._run(validated, phase="selection")
        roster = FrozenMethodRoster(
            validation_run_id=result["run_id"],
            reference_method=self.reference_method,
            method_names=tuple(adapter.name for adapter in self.adapters),
            method_identity_sha256={
                adapter.name: method_identity_sha256(adapter)
                for adapter in self.adapters
            },
        )
        result["frozen_roster"] = roster.to_dict()
        return result

    async def run_test(
        self,
        samples: Sequence[ComparisonSample],
        *,
        frozen_roster: FrozenMethodRoster,
    ) -> dict[str, Any]:
        """Run a validation-frozen roster on held-out examples."""
        validated = _validate_samples(samples, expected_split="test")
        validate_adapters_ready(self.adapters)
        current_names = tuple(adapter.name for adapter in self.adapters)
        current_identities = {
            adapter.name: method_identity_sha256(adapter)
            for adapter in self.adapters
        }
        if (
            current_names != frozen_roster.method_names
            or self.reference_method != frozen_roster.reference_method
        ):
            raise RuntimeError(
                "Held-out method roster or artifact identity differs from "
                "the validation-frozen roster"
            )
        result = await self._run(validated, phase="test")
        result["frozen_from_validation_run_id"] = (
            frozen_roster.validation_run_id
        )
        return result

    async def _run(
        self,
        samples: Sequence[ComparisonSample],
        *,
        phase: str,
    ) -> dict[str, Any]:
        # This is intentionally before schedule construction/scoring.  A
        # missing AFlow graph cannot be represented as a zero-score run.
        validate_adapters_ready(self.adapters)
        method_names = [adapter.name for adapter in self.adapters]
        if self.reference_method not in method_names:
            raise ValueError("reference_method is not in the adapter roster")
        adapter_by_name = {
            adapter.name: adapter for adapter in self.adapters
        }
        starting_identity_hashes = {
            adapter.name: method_identity_sha256(adapter)
            for adapter in self.adapters
        }
        schedule = build_reference_abba_schedule(
            samples,
            method_names,
            self.reference_method,
        )

        rows: list[dict[str, Any]] = []
        for item in schedule:
            adapter = adapter_by_name[item.method_name]
            # Adapters receive only public inference fields.  Ground truth and
            # scorer metadata stay in the runner's private sample object.
            # Keeping the same dataclass avoids a breaking factory API change
            # while making accidental label access fail closed to ``None``.
            blind_sample = ComparisonSample(
                sample_id=item.sample.sample_id,
                query=item.sample.query,
                ground_truth=None,
                split=item.sample.split,
                source_index=item.sample.source_index,
                metadata={
                    key: value
                    for key, value in item.sample.metadata.items()
                    if key in _PUBLIC_INFERENCE_METADATA
                },
            )
            started = time.perf_counter()
            inference = await adapter.infer(blind_sample)
            harness_wall = time.perf_counter() - started
            score = self.scorer(item.sample, inference.output)
            if inspect.isawaitable(score):
                score = await score
            score = _validate_score(score)
            telemetry = inference.telemetry
            rows.append(
                {
                    "sample_id": item.sample.sample_id,
                    "source_index": item.sample.source_index,
                    "split": item.sample.split,
                    "pair_id": item.pair_id,
                    "reference_method": item.reference_method,
                    "comparator_method": item.comparator_method,
                    "method": item.method_name,
                    "order_position": item.order_position,
                    "replicate": item.replicate,
                    "block_pattern": item.block_pattern,
                    "score": score,
                    "output_sha256": _output_sha256(inference.output),
                    "telemetry": telemetry.to_dict(),
                    "harness_wall_latency_seconds": harness_wall,
                    "adapter_metadata": {
                        key: value
                        for key, value in inference.metadata.items()
                        if key in _PUBLIC_RESULT_METADATA
                        and (
                            value is None
                            or isinstance(value, (bool, int, float, str))
                        )
                    },
                }
            )

        ending_identity_hashes = {
            adapter.name: method_identity_sha256(adapter)
            for adapter in self.adapters
        }
        del ending_identity_hashes
        identities = {
            adapter.name: {
                "identity": dict(adapter.identity()),
                "identity_sha256": starting_identity_hashes[adapter.name],
            }
            for adapter in self.adapters
        }
        return {
            "schema_version": self.schema_version,
            "run_id": uuid.uuid4().hex,
            "phase": phase,
            "split": samples[0].split,
            "execution_protocol": {
                "serial": True,
                "max_concurrency": 1,
                "order": "reference-paired counterbalanced ABBA/BAAB",
                "reference_method": self.reference_method,
                "method_order": method_names,
                "repetitions_per_method_per_pair_sample": 2,
                "pair_summary_is_primary": True,
                "method_totals_cross_method_comparable": (
                    len(method_names) == 2
                ),
            },
            "sample_count": len(samples),
            "sample_ids": [sample.sample_id for sample in samples],
            "method_identities": identities,
            "search_telemetry": {
                adapter.name: adapter.search_telemetry.to_dict()
                for adapter in self.adapters
            },
            "inference": {
                "run_count": len(rows),
                "rows": rows,
                "method_summary": _summarize_methods(rows),
                "pair_summary": _summarize_pairs(rows),
            },
        }


def _validate_samples(
    samples: Sequence[ComparisonSample],
    *,
    expected_split: str,
) -> list[ComparisonSample]:
    values = list(samples)
    if not values:
        raise ValueError(f"{expected_split} comparison sample set is empty")
    if any(not isinstance(sample, ComparisonSample) for sample in values):
        raise TypeError("All comparison rows must be ComparisonSample")
    wrong = [
        sample.sample_id
        for sample in values
        if sample.split != expected_split
    ]
    if wrong:
        raise ValueError(
            f"{expected_split} comparison received rows from another split: "
            + ", ".join(wrong[:5])
        )
    ids = [sample.sample_id for sample in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Comparison sample_id values must be unique")
    return values


def _validate_score(value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("scorer must return a finite number")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("scorer must return a hard score in [0, 1]")
    return score


def _output_sha256(output: Any) -> str:
    from awf.protocol.manifest import json_sha256

    try:
        return json_sha256(output)
    except (TypeError, ValueError):
        return json_sha256(str(output))


def _summarize_methods(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: dict[str, Any] = {}
    for method, method_rows in grouped.items():
        telemetry = _sum_telemetry(method_rows)
        scores = [float(row["score"]) for row in method_rows]
        summary[method] = {
            "num_runs": len(method_rows),
            "num_unique_samples": len(
                {str(row["sample_id"]) for row in method_rows}
            ),
            "mean_hard_score": mean(scores),
            "hard_success_rate": mean(score >= 1.0 for score in scores),
            "inference_telemetry_total": telemetry.to_dict(),
            "inference_telemetry_per_run": _divide_telemetry(
                telemetry,
                len(method_rows),
            ),
            "mean_harness_wall_latency_seconds": mean(
                float(row["harness_wall_latency_seconds"])
                for row in method_rows
            ),
        }
    return summary


def _summarize_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair_sample: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        by_pair_sample[
            (str(row["pair_id"]), str(row["sample_id"]))
        ].append(row)

    paired_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    pair_meta: dict[str, tuple[str, str]] = {}
    for (pair_id, _sample_id), sample_rows in by_pair_sample.items():
        reference = str(sample_rows[0]["reference_method"])
        comparator = str(sample_rows[0]["comparator_method"])
        pair_meta[pair_id] = (reference, comparator)
        reference_rows = [
            row for row in sample_rows if row["method"] == reference
        ]
        comparator_rows = [
            row for row in sample_rows if row["method"] == comparator
        ]
        if len(reference_rows) != 2 or len(comparator_rows) != 2:
            raise RuntimeError("Incomplete ABBA block in comparison results")
        ref = _mean_row_metrics(reference_rows)
        comp = _mean_row_metrics(comparator_rows)
        paired_rows[pair_id].append(
            {
                key: comp[key] - ref[key]
                for key in ref
            }
        )

    summary: dict[str, Any] = {}
    for pair_id, deltas in paired_rows.items():
        reference, comparator = pair_meta[pair_id]
        summary[pair_id] = {
            "reference_method": reference,
            "comparator_method": comparator,
            "num_paired_samples": len(deltas),
            "mean_comparator_minus_reference": {
                key: mean(row[key] for row in deltas)
                for key in deltas[0]
            },
        }
    return summary


def _mean_row_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    return {
        "hard_score": mean(float(row["score"]) for row in rows),
        "prompt_tokens": mean(
            float(row["telemetry"]["prompt_tokens"]) for row in rows
        ),
        "completion_tokens": mean(
            float(row["telemetry"]["completion_tokens"]) for row in rows
        ),
        "total_tokens": mean(
            float(row["telemetry"]["total_tokens"]) for row in rows
        ),
        "llm_calls": mean(
            float(row["telemetry"]["llm_calls"]) for row in rows
        ),
        "llm_latency_seconds": mean(
            float(row["telemetry"]["llm_latency_seconds"]) for row in rows
        ),
        "wall_latency_seconds": mean(
            float(row["telemetry"]["wall_latency_seconds"]) for row in rows
        ),
        "harness_wall_latency_seconds": mean(
            float(row["harness_wall_latency_seconds"]) for row in rows
        ),
    }


def _sum_telemetry(
    rows: Sequence[Mapping[str, Any]],
) -> PhaseTelemetry:
    total = PhaseTelemetry()
    for row in rows:
        total += PhaseTelemetry.from_mapping(row["telemetry"])
    return total


def _divide_telemetry(
    telemetry: PhaseTelemetry,
    count: int,
) -> dict[str, float]:
    return {
        "prompt_tokens": telemetry.prompt_tokens / count,
        "completion_tokens": telemetry.completion_tokens / count,
        "total_tokens": telemetry.total_tokens / count,
        "llm_calls": telemetry.llm_calls / count,
        "llm_latency_seconds": telemetry.llm_latency_seconds / count,
        "wall_latency_seconds": telemetry.wall_latency_seconds / count,
        "latency_seconds": telemetry.wall_latency_seconds / count,
    }
