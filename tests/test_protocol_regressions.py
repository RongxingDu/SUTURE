"""Regression tests for configuration, protocol safety, and graph validation."""

from pathlib import Path
import json

import pytest

from awf.config.loader import load_config
from awf.protocol.checkpoint import CheckpointManager
from awf.protocol.experiment import ExperimentRunner
from awf.protocol.split_manager import SplitManager
from awf.trace.schema import ExecutionTrace
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeType


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nested_config_inheritance_preserves_llm_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    config = load_config(PROJECT_ROOT / "experiments/configs/code_gen.yaml")

    assert config.scheduler.llm.model == "gpt-4o"
    assert config.scheduler.llm.api_key == "test-secret"
    assert config.optimizer.llm.api_key == "test-secret"
    assert config.scheduler.max_actions_per_query == 30


def test_public_experiment_config_redacts_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    config = load_config(PROJECT_ROOT / "experiments/configs/default.yaml")
    runner = object.__new__(ExperimentRunner)
    runner.config = config

    public = runner._public_config()

    assert public["scheduler"]["llm"]["api_key"] == "***REDACTED***"
    assert public["optimizer"]["llm"]["api_key"] == "***REDACTED***"
    assert "must-not-leak" not in str(public)


def test_public_config_redacts_nested_provider_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ordinary-key")
    config = load_config(PROJECT_ROOT / "experiments/configs/default.yaml")
    config.scheduler.llm.extra_kwargs = {
        "headers": {
            "Authorization": "Bearer nested-secret",
            "x-access-token": "nested-token",
        }
    }
    runner = object.__new__(ExperimentRunner)
    runner.config = config

    public = runner._public_config()

    assert "nested-secret" not in str(public)
    assert "nested-token" not in str(public)


def test_runner_passes_effective_execution_llm_defaults_to_optimizer(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    captured = {}

    class CapturingOptimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "awf.protocol.experiment.LLMWorkflowOptimizer",
        CapturingOptimizer,
    )
    config = load_config(PROJECT_ROOT / "experiments/configs/default.yaml")
    config.name = "execution-defaults"
    config.output_dir = str(tmp_path)
    config.scheduler.scheduler_type = "fixed"
    config.scheduler.llm.model = "deepseek-v4-flash"
    config.scheduler.llm.temperature = 0.0
    config.scheduler.llm.max_tokens = 2048
    workflow = WorkflowTemplate(
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "end")],
    )

    ExperimentRunner(
        config=config,
        workflow=workflow,
        reward_evaluator=object(),
    )

    assert captured["execution_llm_defaults"] == {
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
    }


def test_results_are_written_private_and_without_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    config = load_config(PROJECT_ROOT / "experiments/configs/default.yaml")
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.output_dir = tmp_path

    runner._save_results({"config": runner._public_config()})

    path = tmp_path / "results.json"
    assert path.stat().st_mode & 0o777 == 0o600
    saved = json.loads(path.read_text())
    assert "must-not-leak" not in str(saved)


def test_checkpoint_is_atomic_private_loadable_and_counts_updates(tmp_path):
    workflow = WorkflowTemplate(
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "end")],
    )
    manager = CheckpointManager(tmp_path)

    assert manager.update(workflow, score=0.5, round_num=0)
    assert not manager.update(workflow, score=0.4, round_num=1)
    assert manager.update(workflow, score=0.6, round_num=2)

    workflow_path = tmp_path / "best_workflow.yaml"
    metadata_path = tmp_path / "checkpoint_meta.json"
    assert workflow_path.stat().st_mode & 0o777 == 0o600
    assert metadata_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".best_workflow.*"))

    summary = manager.get_summary()
    assert summary["num_checkpoints"] == 1
    assert summary["num_checkpoint_updates"] == 2
    assert summary["num_evaluations"] == 3

    reloaded = CheckpointManager(tmp_path)
    assert reloaded.load_best() == workflow
    assert reloaded.best_score == 0.6
    assert reloaded.best_round == 2


def test_trace_store_is_private_robust_and_truncated_per_runner(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    config = load_config(PROJECT_ROOT / "experiments/configs/default.yaml")

    class CustomOutput:
        def __str__(self):
            return "custom-output"

    trace = ExecutionTrace(
        trace_id="trace",
        final_output=CustomOutput(),
    )
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.output_dir = tmp_path
    runner._initialized_trace_splits = set()

    runner._save_traces("optimization", [trace])
    runner._save_traces("optimization", [trace])

    path = tmp_path / "traces" / "optimization.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    assert len(path.read_text().splitlines()) == 2
    assert "custom-output" in path.read_text()

    fresh_runner = object.__new__(ExperimentRunner)
    fresh_runner.config = config
    fresh_runner.output_dir = tmp_path
    fresh_runner._initialized_trace_splits = set()
    fresh_runner._save_traces("optimization", [trace])
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("ratio", [-0.1, 0.0])
def test_split_manager_rejects_non_positive_ratios(ratio):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        SplitManager(opt_ratio=ratio, val_ratio=0.5, test_ratio=0.5 - ratio)


def test_workflow_rejects_dangling_edge():
    with pytest.raises(ValueError, match="Dangling workflow edge"):
        WorkflowTemplate(
            entry_node="start",
            nodes={
                "start": Node(node_id="start", node_type=NodeType.START),
            },
            edges=[("start", "missing")],
        )


def test_workflow_rejects_node_key_mismatch():
    with pytest.raises(ValueError, match="does not match node_id"):
        WorkflowTemplate(
            entry_node="key",
            nodes={
                "key": Node(node_id="different", node_type=NodeType.START),
            },
        )


def test_workflow_and_hierarchy_reject_unknown_or_inconsistent_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="prompt_temlate"):
        from awf.workflow.nodes import NodeConfig

        NodeConfig(prompt_temlate="{query}")

    with pytest.raises(ValueError, match="Hierarchy user_template"):
        WorkflowTemplate(
            entry_node="node",
            nodes={
                "node": Node(
                    node_id="node",
                    node_type=NodeType.LLM,
                    config={"prompt_template": "actual"},
                ),
            },
            parameters={
                "stages": {
                    "stage": {
                        "stage_id": "stage",
                        "blocks": {
                            "block": {
                                "block_id": "block",
                                "operators": {
                                    "node": {
                                        "node_id": "node",
                                        "prompt": {
                                            "user_template": "stale",
                                        },
                                    }
                                },
                            }
                        },
                    }
                }
            },
        )


def test_workflow_rejects_incoming_entry_and_outgoing_end():
    with pytest.raises(ValueError, match="entry node.*incoming"):
        WorkflowTemplate(
            entry_node="start",
            nodes={
                "start": Node(node_id="start", node_type=NodeType.START),
                "loop": Node(node_id="loop", node_type=NodeType.LLM),
            },
            edges=[("start", "loop"), ("loop", "start")],
        )

    with pytest.raises(ValueError, match="END node.*outgoing"):
        WorkflowTemplate(
            entry_node="start",
            nodes={
                "start": Node(node_id="start", node_type=NodeType.START),
                "end": Node(node_id="end", node_type=NodeType.END),
                "after": Node(node_id="after", node_type=NodeType.LLM),
            },
            edges=[("start", "end"), ("end", "after")],
        )
