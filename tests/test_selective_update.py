"""Runtime and serialization semantics for one-layer selective updates."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from awf.config.schema import ExecutorConfig, ExperimentConfig
from awf.executor.runtime import RuntimeExecutor
from awf.scheduler.graph_scheduler import GraphScheduler
from awf.workflow.gates import (
    GateSpec,
    attach_selective_update,
    extract_pre_execution_features,
)
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.serializer import dump_workflow, load_workflow


def _workflow(tool_name: str) -> WorkflowTemplate:
    return WorkflowTemplate(
        name="selective-runtime",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "work": Node(
                node_id="work",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name=tool_name,
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "work"), ("work", "end")],
    )


def _policy() -> WorkflowTemplate:
    return attach_selective_update(
        _workflow("base"),
        _workflow("candidate"),
        GateSpec(
            kind="threshold",
            feature="query_words",
            operator="le",
            threshold=1.0,
        ),
        patch_fingerprint="a" * 64,
        changed_units=["work"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected", "applied"),
    [
        ("short", "candidate", True),
        ("two words", "base", False),
    ],
)
async def test_runtime_realizes_exactly_one_variant_and_records_gate(
    query: str,
    expected: str,
    applied: bool,
) -> None:
    calls: list[str] = []

    def invoke(name: str):
        def operator():
            calls.append(name)
            return name

        return operator

    output, _, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5),
        operators={
            "base": invoke("base"),
            "candidate": invoke("candidate"),
        },
    ).execute(_policy(), GraphScheduler(), query)

    assert output == expected
    assert calls == [expected]
    decision = recorder.trace.metadata["selective_update"]
    assert decision["applied"] is applied
    assert decision["fail_closed"] is False
    assert decision["feature"] == "query_words"
    assert decision["feature_value"] == float(len(query.split()))
    assert recorder.trace.total_llm_calls == 0


def test_selective_policy_round_trips_and_rejects_nested_policy(tmp_path) -> None:
    policy = _policy()
    path = tmp_path / "policy.yaml"

    dump_workflow(policy, path)
    restored = load_workflow(path)

    assert restored == policy
    with pytest.raises(ValueError, match="nested selective"):
        attach_selective_update(
            policy,
            _workflow("candidate"),
            GateSpec(kind="always"),
            patch_fingerprint="b" * 64,
            changed_units=["work"],
        )


def test_default_workflow_serialization_does_not_add_null_policy_field(
    tmp_path,
) -> None:
    path = tmp_path / "base.yaml"

    dump_workflow(_workflow("base"), path)

    assert "selective_update" not in path.read_text()


@pytest.mark.asyncio
async def test_tampered_policy_fails_closed_to_base_without_candidate_call() -> None:
    policy = _policy()
    assert policy.selective_update is not None
    policy.selective_update.base_workflow_fingerprint = "0" * 64
    calls: list[str] = []

    def base():
        calls.append("base")
        return "base"

    def candidate():
        calls.append("candidate")
        return "candidate"

    output, _, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=5),
        operators={"base": base, "candidate": candidate},
    ).execute(policy, GraphScheduler(), "short")

    assert output == "base"
    assert calls == ["base"]
    decision = recorder.trace.metadata["selective_update"]
    assert decision["applied"] is False
    assert decision["fail_closed"] is True
    assert decision["reason"] == "invalid_policy_fallback"


def test_gate_schema_and_features_cannot_encode_labels_or_rewards() -> None:
    assert set(extract_pre_execution_features("Question 12?")) == {
        "query_chars",
        "query_words",
        "query_lines",
        "numeric_literals",
    }
    with pytest.raises(ValidationError):
        GateSpec.model_validate(
            {
                "kind": "threshold",
                "feature": "hard_reward",
                "operator": "gt",
                "threshold": 0.5,
            }
        )
    with pytest.raises(ValidationError):
        GateSpec(
            kind="threshold",
            feature="query_words",
            operator="le",
            threshold=float("nan"),
        )


def test_selective_config_is_default_off_and_requires_graph_scheduler() -> None:
    assert ExperimentConfig().optimizer.selective_update_enabled is False
    with pytest.raises(ValidationError, match="scheduler_type='graph'"):
        ExperimentConfig(
            optimizer={
                "workflow_content_only": False,
                "selective_update_enabled": True,
                "gate_min_leaf_support": 1,
                "gate_fit_max_traces": 2,
            }
        )

    configured = ExperimentConfig(
        scheduler={"scheduler_type": "graph"},
        optimizer={
            "workflow_content_only": False,
            "selective_update_enabled": True,
            "gate_min_leaf_support": 1,
            "gate_fit_max_traces": 2,
        },
    )
    assert configured.optimizer.selective_update_enabled is True
