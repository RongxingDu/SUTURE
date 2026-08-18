"""Data models shared by comparison adapters and the paired runner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


_METHOD_KINDS = frozenset({"vanilla", "aflow", "awf", "scwu", "mock"})
_SPLITS = frozenset({"optimization", "validation", "test"})


def validate_method_kind(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _METHOD_KINDS:
        raise ValueError(
            f"Unsupported method kind {value!r}; "
            f"expected one of {sorted(_METHOD_KINDS)}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ComparisonSample:
    """One benchmark row with an explicit protocol split.

    Queries and ground truth remain in the runner.  Before calling an adapter,
    the runner constructs a blind view with ``ground_truth=None`` and only
    allowlisted public metadata (currently a code task's public
    ``entry_point``).  The public artifact records only identifiers and
    aggregate scores, so held-out answers and model outputs are not copied
    into comparison telemetry.
    """

    sample_id: str
    query: str
    ground_truth: Any
    split: str
    source_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        split = str(self.split).strip().lower()
        if split not in _SPLITS:
            raise ValueError(
                f"Unsupported split {self.split!r}; "
                f"expected one of {sorted(_SPLITS)}"
            )
        if (
            self.source_index is not None
            and (
                not isinstance(self.source_index, int)
                or isinstance(self.source_index, bool)
                or self.source_index < 0
            )
        ):
            raise ValueError("source_index must be a non-negative integer")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "split", split)


@dataclass(frozen=True, slots=True)
class PhaseTelemetry:
    """Provider and end-to-end telemetry for one research phase.

    ``llm_latency_seconds`` is the sum of provider-call latency.  It may exceed
    end-to-end wall latency when a workflow makes parallel calls.
    ``wall_latency_seconds`` is the externally observed phase duration.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    llm_latency_seconds: float = 0.0
    wall_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "llm_calls"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("llm_latency_seconds", "wall_latency_seconds"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: object) -> "PhaseTelemetry":
        if not isinstance(other, PhaseTelemetry):
            return NotImplemented
        return PhaseTelemetry(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=(
                self.completion_tokens + other.completion_tokens
            ),
            llm_calls=self.llm_calls + other.llm_calls,
            llm_latency_seconds=(
                self.llm_latency_seconds + other.llm_latency_seconds
            ),
            wall_latency_seconds=(
                self.wall_latency_seconds + other.wall_latency_seconds
            ),
        )

    def with_wall_latency(self, seconds: float) -> "PhaseTelemetry":
        return replace(self, wall_latency_seconds=float(seconds))

    def to_dict(self) -> dict[str, int | float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "llm_latency_seconds": float(self.llm_latency_seconds),
            "wall_latency_seconds": float(self.wall_latency_seconds),
            # Stable convenience alias for consumers that need one latency.
            "latency_seconds": float(self.wall_latency_seconds),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhaseTelemetry":
        if not isinstance(value, Mapping):
            raise TypeError("telemetry must be a mapping")
        allowed = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_calls",
            "llm_latency_seconds",
            "wall_latency_seconds",
            "latency_seconds",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown telemetry fields: " + ", ".join(sorted(unknown))
            )
        prompt = value.get("prompt_tokens", 0)
        completion = value.get("completion_tokens", 0)
        declared_total = value.get("total_tokens")
        if (
            declared_total is not None
            and declared_total != prompt + completion
        ):
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )
        wall = value.get(
            "wall_latency_seconds",
            value.get("latency_seconds", 0.0),
        )
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            llm_calls=value.get("llm_calls", 0),
            llm_latency_seconds=value.get("llm_latency_seconds", 0.0),
            wall_latency_seconds=wall,
        )


@dataclass(frozen=True, slots=True)
class AdapterInference:
    """One method output plus telemetry measured by its adapter."""

    output: Any
    telemetry: PhaseTelemetry
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.telemetry, PhaseTelemetry):
            raise TypeError("telemetry must be PhaseTelemetry")


@dataclass(frozen=True, slots=True)
class FrozenMethodRoster:
    """Validation-frozen method identities required for held-out execution."""

    validation_run_id: str
    reference_method: str
    method_names: tuple[str, ...]
    method_identity_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not str(self.validation_run_id).strip():
            raise ValueError("validation_run_id must be non-empty")
        if len(self.method_names) < 2:
            raise ValueError("A frozen roster requires at least two methods")
        if len(set(self.method_names)) != len(self.method_names):
            raise ValueError("Frozen method names must be unique")
        if self.reference_method not in self.method_names:
            raise ValueError("reference_method is not in the frozen roster")
        if set(self.method_identity_sha256) != set(self.method_names):
            raise ValueError(
                "Frozen method identity keys do not match method_names"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_run_id": self.validation_run_id,
            "reference_method": self.reference_method,
            "method_names": list(self.method_names),
            "method_identity_sha256": dict(self.method_identity_sha256),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenMethodRoster":
        if not isinstance(value, Mapping):
            raise TypeError("frozen roster must be a mapping")
        try:
            names = tuple(value["method_names"])
            identities = dict(value["method_identity_sha256"])
            return cls(
                validation_run_id=str(value["validation_run_id"]),
                reference_method=str(value["reference_method"]),
                method_names=names,
                method_identity_sha256=identities,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed frozen method roster") from exc
