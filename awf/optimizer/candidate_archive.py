"""Bounded, aggregate-only archive of hard-safe workflow candidates."""

from __future__ import annotations

import copy
import math
from typing import Any


class CandidateArchive:
    """Keep bounded within-batch Pareto experience, then scalar Top-K.

    The archive intentionally stores only aggregate optimization-split effects.
    It does not retain queries, outputs, traces, workflows, validation metrics,
    or test metrics. Candidates with an observed hard-reward regression or an
    incomplete expected batch are ineligible, regardless of savings. Pareto
    dominance is evaluated only for candidates sharing a hashed comparison
    batch; cross-round heterogeneous means are never treated as comparable.
    """

    def __init__(self, capacity: int = 0):
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity
        self._entries: dict[str, dict[str, Any]] = {}
        self._retained: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def add(
        self,
        record: dict[str, Any],
        *,
        round_num: int,
        split: str,
        comparison_key: str,
        expected_trace_ids: set[str],
    ) -> bool:
        """Add one evaluated, hard-safe candidate using aggregate effects only."""
        if not self.enabled:
            return False
        if (
            split != "optimization"
            or not isinstance(comparison_key, str)
            or not comparison_key
            or not expected_trace_ids
            or any(
                not isinstance(trace_id, str) or not trace_id
                for trace_id in expected_trace_ids
            )
        ):
            return False
        entry = self._aggregate(
            record,
            round_num=round_num,
            comparison_key=comparison_key,
            expected_trace_ids=expected_trace_ids,
        )
        if entry is None:
            return False

        fingerprint = entry["joint_fingerprint"]
        previous = self._entries.get(fingerprint)
        if previous is None or self._rank_key(entry) > self._rank_key(previous):
            self._entries[fingerprint] = entry
        self._rebuild()
        return fingerprint in {
            item["joint_fingerprint"] for item in self._retained
        }

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a serialization-safe copy of retained aggregate entries."""
        return copy.deepcopy(self._retained)

    def experience_context(self, limit: int = 8) -> str:
        """Return bounded aggregate evidence suitable for optimizer prompts."""
        if limit <= 0 or not self._retained:
            return ""
        lines = [
            "Bounded hard-safe candidate experience archive "
            "(approximate within-batch Pareto points first, then scalar gain):"
        ]
        for entry in self._retained[:limit]:
            lines.append(
                "- "
                f"approx_pareto_within_batch={entry['pareto']}; "
                f"comparison_batch={entry['comparison_key'][:12]}; "
                f"patch={entry['patch_fingerprint'][:16]}; "
                f"joint={entry['joint_fingerprint'][:16]}; "
                f"gate={entry['gate_kind']}; "
                f"round={entry['round']}; "
                f"anchor={entry['anchor_id']}; scope={entry['scope']}; "
                f"hard_delta={entry['mean_hard_delta']:.6g}; "
                f"utility_delta={entry['mean_utility_delta']:.6g}; "
                f"token_delta={entry['mean_token_delta']:.6g}; "
                f"latency_delta_seconds="
                f"{entry['mean_latency_delta_seconds']:.6g}; "
                f"edit_distance={entry['edit_distance']:.6g}; "
                f"gain={entry['gain']:.6g}"
            )
        return "\n".join(lines)

    @classmethod
    def _aggregate(
        cls,
        record: dict[str, Any],
        *,
        round_num: int,
        comparison_key: str,
        expected_trace_ids: set[str],
    ) -> dict[str, Any] | None:
        fingerprint = record.get("patch_fingerprint")
        joint_fingerprint = record.get("joint_fingerprint", fingerprint)
        rows = record.get("per_query_results")
        gain = cls._finite(record.get("gain"))
        edit_distance = cls._finite(record.get("edit_distance"))
        if (
            not isinstance(fingerprint, str)
            or not fingerprint
            or not isinstance(joint_fingerprint, str)
            or not joint_fingerprint
            or not isinstance(rows, list)
            or not rows
            or gain is None
            or edit_distance is None
        ):
            return None

        hard_deltas: list[float] = []
        utility_deltas: list[float] = []
        token_deltas: list[float] = []
        latency_deltas: list[float] = []
        observed_trace_ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            trace_id = row.get("trace_id")
            if not isinstance(trace_id, str) or not trace_id:
                return None
            observed_trace_ids.append(trace_id)
            hard_pair = cls._pair(row, "candidate_hard", "original_hard")
            utility_delta = cls._finite(row.get("delta_u"))
            token_pair = cls._pair(
                row,
                "candidate_total_tokens",
                "original_total_tokens",
            )
            latency_pair = cls._pair(
                row,
                "candidate_latency_seconds",
                "original_latency_seconds",
            )
            if (
                hard_pair is None
                or utility_delta is None
                or token_pair is None
                or latency_pair is None
            ):
                return None
            hard_deltas.append(hard_pair[0] - hard_pair[1])
            utility_deltas.append(utility_delta)
            token_deltas.append(token_pair[0] - token_pair[1])
            latency_deltas.append(latency_pair[0] - latency_pair[1])

        if (
            len(observed_trace_ids) != len(set(observed_trace_ids))
            or set(observed_trace_ids) != expected_trace_ids
        ):
            return None

        # Performance is a constraint, not a tradable objective.
        if any(delta < -1e-12 for delta in hard_deltas):
            return None

        gate = record.get("gate")
        gate_kind = "none"
        gate_feature: str | None = None
        gate_operator: str | None = None
        gate_threshold: float | None = None
        if gate is not None:
            if not isinstance(gate, dict):
                return None
            gate_kind = str(gate.get("kind", ""))[:20]
            if gate_kind not in {"never", "always", "threshold"}:
                return None
            raw_feature = gate.get("feature")
            raw_operator = gate.get("operator")
            raw_threshold = gate.get("threshold")
            gate_feature = (
                str(raw_feature)[:50] if raw_feature is not None else None
            )
            gate_operator = (
                str(raw_operator)[:8] if raw_operator is not None else None
            )
            if raw_threshold is not None:
                gate_threshold = cls._finite(raw_threshold)
                if gate_threshold is None:
                    return None

        gate_coverage = cls._finite(record.get("gate_coverage"))
        if gate_coverage is not None and not 0.0 <= gate_coverage <= 1.0:
            return None

        return {
            "patch_fingerprint": fingerprint,
            "joint_fingerprint": joint_fingerprint,
            "comparison_key": comparison_key,
            "pareto_scope": "bounded_within_comparison_batch",
            "round": int(round_num),
            "anchor_id": str(record.get("anchor_id", ""))[:200],
            "scope": str(record.get("scope", ""))[:50],
            "gain": gain,
            "edit_distance": edit_distance,
            "num_queries": len(rows),
            "mean_hard_delta": sum(hard_deltas) / len(hard_deltas),
            "mean_utility_delta": (
                sum(utility_deltas) / len(utility_deltas)
            ),
            "mean_token_delta": sum(token_deltas) / len(token_deltas),
            "mean_latency_delta_seconds": (
                sum(latency_deltas) / len(latency_deltas)
            ),
            "gate_kind": gate_kind,
            "gate_feature": gate_feature,
            "gate_operator": gate_operator,
            "gate_threshold": gate_threshold,
            "gate_coverage": gate_coverage,
            "pareto": False,
        }

    def _rebuild(self) -> None:
        entries = list(self._entries.values())
        frontier = [
            entry
            for entry in entries
            if not any(
                other is not entry and self._dominates(other, entry)
                for other in entries
            )
        ]
        frontier.sort(key=self._rank_key, reverse=True)
        frontier_ids = {
            entry["joint_fingerprint"] for entry in frontier
        }
        dominated = [
            entry
            for entry in entries
            if entry["joint_fingerprint"] not in frontier_ids
        ]
        dominated.sort(key=self._rank_key, reverse=True)
        retained = (frontier + dominated)[: self.capacity]
        self._retained = []
        for entry in retained:
            copied = copy.deepcopy(entry)
            copied["pareto"] = entry["joint_fingerprint"] in frontier_ids
            self._retained.append(copied)
        # Bound the backing store as well as the public view. Evicted dominated
        # candidates cannot later make memory grow with the number of rounds.
        self._entries = {
            entry["joint_fingerprint"]: {
                **copy.deepcopy(entry),
                "pareto": False,
            }
            for entry in self._retained
        }

    @staticmethod
    def _dominates(
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        """Return whether ``left`` weakly beats ``right`` on every objective."""
        if left["comparison_key"] != right["comparison_key"]:
            return False
        left_values = (
            left["mean_hard_delta"],
            left["mean_utility_delta"],
            -left["mean_token_delta"],
            -left["mean_latency_delta_seconds"],
            -left["edit_distance"],
        )
        right_values = (
            right["mean_hard_delta"],
            right["mean_utility_delta"],
            -right["mean_token_delta"],
            -right["mean_latency_delta_seconds"],
            -right["edit_distance"],
        )
        return (
            all(a >= b - 1e-12 for a, b in zip(left_values, right_values))
            and any(a > b + 1e-12 for a, b in zip(left_values, right_values))
        )

    @staticmethod
    def _rank_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        return (
            entry["gain"],
            entry["mean_hard_delta"],
            entry["mean_utility_delta"],
            -entry["mean_token_delta"],
            -entry["mean_latency_delta_seconds"],
            -entry["edit_distance"],
            entry["joint_fingerprint"],
        )

    @staticmethod
    def _finite(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _pair(
        cls,
        row: dict[str, Any],
        left: str,
        right: str,
    ) -> tuple[float, float] | None:
        left_value = cls._finite(row.get(left))
        right_value = cls._finite(row.get(right))
        if left_value is None or right_value is None:
            return None
        return left_value, right_value
