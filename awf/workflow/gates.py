"""Typed, pre-execution gates for selective workflow updates.

The gate language is deliberately smaller than the workflow CONDITION
expression language.  It cannot read traces, rewards, labels, outputs, or
arbitrary metadata: every feature is deterministically derived from the query
that is already available before either workflow variant executes.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from awf.workflow.ir import WorkflowTemplate


GateFeature = Literal[
    "query_chars",
    "query_words",
    "query_lines",
    "numeric_literals",
]
GateOperator = Literal["le", "gt"]
GateKind = Literal["never", "always", "threshold"]

SUPPORTED_GATE_FEATURES: tuple[str, ...] = (
    "query_chars",
    "query_words",
    "query_lines",
    "numeric_literals",
)


class _StrictGateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateSpec(_StrictGateModel):
    """One interpretable depth-1 applicability rule."""

    kind: GateKind
    feature: GateFeature | None = None
    operator: GateOperator | None = None
    threshold: float | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "GateSpec":
        if self.kind in {"never", "always"}:
            if any(
                value is not None
                for value in (self.feature, self.operator, self.threshold)
            ):
                raise ValueError(
                    f"{self.kind} gate cannot define feature/operator/threshold"
                )
            return self
        if (
            self.feature is None
            or self.operator is None
            or self.threshold is None
            or not math.isfinite(float(self.threshold))
        ):
            raise ValueError(
                "threshold gate requires a finite feature/operator/threshold"
            )
        return self

    @property
    def complexity(self) -> int:
        return 0 if self.kind in {"never", "always"} else 1

    @property
    def fingerprint(self) -> str:
        return _json_sha256(self.model_dump(mode="json"))


class SelectiveUpdateSpec(_StrictGateModel):
    """A single candidate workflow and its bound applicability gate.

    ``candidate_workflow`` is a validated canonical WorkflowTemplate payload.
    It is kept as a mapping rather than a recursively typed WorkflowTemplate so
    checkpoints remain ordinary workflow YAML and nested policies can be
    rejected explicitly.
    """

    schema_version: Literal[1] = 1
    gate: GateSpec
    # Legacy identity fields are retained for loading old experiment records,
    # but they are informational only and are never used for acceptance or
    # runtime validation.
    base_workflow_fingerprint: str = ""
    candidate_workflow_fingerprint: str = ""
    patch_fingerprint: str = ""
    joint_fingerprint: str = ""
    changed_units: list[str] = Field(min_length=1, max_length=32)
    candidate_workflow: dict[str, Any]

    @field_validator("changed_units")
    @classmethod
    def _validate_changed_units(cls, values: list[str]) -> list[str]:
        if (
            any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("changed_units must be unique non-empty strings")
        return values

    @model_validator(mode="after")
    def _reject_non_deployable_policy(self) -> "SelectiveUpdateSpec":
        if self.gate.kind == "never":
            raise ValueError("a rejected/never update cannot be deployed")
        if not isinstance(self.candidate_workflow, dict) or not self.candidate_workflow:
            raise ValueError("candidate_workflow must be a non-empty mapping")
        if self.candidate_workflow.get("selective_update") is not None:
            raise ValueError("nested selective updates are not supported")
        return self


class GateDecision(_StrictGateModel):
    """Sanitized runtime telemetry for one policy realization."""

    configured: bool = True
    applied: bool = False
    fail_closed: bool = False
    reason: str
    gate_kind: GateKind
    gate_fingerprint: str
    joint_fingerprint: str
    feature: GateFeature | None = None
    feature_value: float | None = None


def extract_pre_execution_features(query: str) -> dict[str, float]:
    """Return the complete allowlisted feature vector for ``query`` only."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return {
        "query_chars": float(len(query)),
        "query_words": float(len(re.findall(r"\S+", query))),
        "query_lines": float(query.count("\n") + 1 if query else 0),
        "numeric_literals": float(
            len(re.findall(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)", query))
        ),
    }


def evaluate_gate(gate: GateSpec, features: dict[str, float]) -> bool:
    """Evaluate a typed gate over a complete pre-execution feature vector."""
    if gate.kind == "never":
        return False
    if gate.kind == "always":
        return True
    feature = gate.feature
    if feature is None or feature not in features:
        raise ValueError("gate feature is unavailable")
    raw_value = features[feature]
    if isinstance(raw_value, bool):
        raise ValueError("gate feature must be numeric")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError("gate feature must be finite")
    threshold = float(gate.threshold)
    if gate.operator == "le":
        return value <= threshold
    if gate.operator == "gt":
        return value > threshold
    raise ValueError("unsupported gate operator")


def workflow_structure_fingerprint(workflow: "WorkflowTemplate") -> str:
    """Fingerprint executable content while ignoring version and outer policy."""
    payload = workflow.model_dump(mode="json")
    payload.pop("version", None)
    payload.pop("selective_update", None)
    return _json_sha256(payload)


def joint_update_fingerprint(
    patch_fingerprint: str,
    gate: GateSpec,
) -> str:
    """Return the stable identity of the joint hypothesis ``(delta, gate)``."""
    return _json_sha256(
        {
            "patch_fingerprint": patch_fingerprint,
            "gate": gate.model_dump(mode="json"),
        }
    )


def attach_selective_update(
    base_workflow: "WorkflowTemplate",
    candidate_workflow: "WorkflowTemplate",
    gate: GateSpec,
    *,
    patch_fingerprint: str,
    changed_units: list[str],
) -> "WorkflowTemplate":
    """Attach one validated candidate without mutating either input workflow."""
    from awf.workflow.ir import WorkflowTemplate

    if base_workflow.selective_update is not None:
        raise ValueError("nested selective updates are not supported")
    if candidate_workflow.selective_update is not None:
        raise ValueError("candidate workflow already contains a selective update")
    if gate.kind == "never":
        raise ValueError("cannot attach a never gate")
    if base_workflow.name != candidate_workflow.name:
        raise ValueError("base and candidate workflow names must match")
    candidate_payload = candidate_workflow.model_dump(mode="json")
    spec = SelectiveUpdateSpec(
        gate=gate,
        base_workflow_fingerprint="",
        candidate_workflow_fingerprint="",
        patch_fingerprint=patch_fingerprint,
        joint_fingerprint="",
        changed_units=list(changed_units),
        candidate_workflow=candidate_payload,
    )
    payload = base_workflow.model_dump(mode="json")
    payload["selective_update"] = spec.model_dump(mode="json")
    return WorkflowTemplate.model_validate(payload)


def validate_selective_update_attachment(workflow: "WorkflowTemplate") -> None:
    """Validate only the executable shape of a selective policy."""
    from awf.workflow.ir import WorkflowTemplate

    spec = workflow.selective_update
    if spec is None:
        return
    candidate = WorkflowTemplate.model_validate(spec.candidate_workflow)
    if candidate.selective_update is not None:
        raise ValueError("nested selective updates are not supported")
    if candidate.name != workflow.name:
        raise ValueError("selective candidate workflow name mismatch")


def realize_selective_update(
    workflow: "WorkflowTemplate",
    query: str,
) -> tuple["WorkflowTemplate", GateDecision | None]:
    """Choose one workflow variant before execution, failing closed to base."""
    from awf.workflow.ir import WorkflowTemplate

    spec = workflow.selective_update
    if spec is None:
        return workflow, None

    def base_copy() -> WorkflowTemplate:
        payload = workflow.model_dump(mode="json")
        payload["selective_update"] = None
        return WorkflowTemplate.model_validate(payload)

    feature_value: float | None = None
    try:
        validate_selective_update_attachment(workflow)
        features = extract_pre_execution_features(query)
        if spec.gate.feature is not None:
            feature_value = features[spec.gate.feature]
        applied = evaluate_gate(spec.gate, features)
        if applied:
            realized = WorkflowTemplate.model_validate(
                copy.deepcopy(spec.candidate_workflow)
            )
            # The outer policy is the promoted workflow version. Preserve it in
            # runtime-facing state and traces whichever variant is realized.
            realized.version = workflow.version
            reason = "gate_applied"
        else:
            realized = base_copy()
            reason = "gate_not_applied"
        return realized, GateDecision(
            applied=applied,
            fail_closed=False,
            reason=reason,
            gate_kind=spec.gate.kind,
            gate_fingerprint=spec.gate.fingerprint,
            joint_fingerprint=spec.joint_fingerprint,
            feature=spec.gate.feature,
            feature_value=feature_value,
        )
    except Exception:
        # Do not expose candidate contents or exception text in deployment
        # telemetry. Invalid/tampered policy state deterministically falls back.
        return base_copy(), GateDecision(
            applied=False,
            fail_closed=True,
            reason="invalid_policy_fallback",
            gate_kind=spec.gate.kind,
            gate_fingerprint=spec.gate.fingerprint,
            joint_fingerprint=spec.joint_fingerprint,
            feature=spec.gate.feature,
            feature_value=feature_value,
        )


def _json_sha256(payload: Any) -> str:
    """Simple JSON identity for workflow component comparison."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
