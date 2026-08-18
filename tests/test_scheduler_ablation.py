"""Regression tests for the validation-only scheduler ablation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from awf.config.schema import ExperimentConfig
from awf.protocol.manifest import (
    ManifestValidationError,
    atomic_write_json_0600,
    bind_best_checkpoint,
    create_manifest,
)
from awf.reward.base import RewardEvaluator
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.serializer import dump_workflow
from experiments.scripts import run_scheduler_ablation


class RecordingReward(RewardEvaluator):
    def __init__(self):
        self.queries: list[str] = []

    def hard_reward(self, query, ground_truth, output, trace):
        self.queries.append(query)
        return float(output == ground_truth)

    def process_reward(self, query, ground_truth, output, trace):
        return 0.0


def _identity(value: str) -> str:
    return value


def _tool_workflow() -> WorkflowTemplate:
    return WorkflowTemplate(
        name="scheduler-ablation-workflow",
        version="1.0",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "answer": Node(
                node_id="answer",
                node_type=NodeType.TOOL,
                config=NodeConfig(
                    node_type=NodeType.TOOL,
                    tool_name="identity",
                    tool_args={"value": "{query}"},
                ),
            ),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[("start", "answer"), ("answer", "end")],
    )


@pytest.fixture
def ablation_artifacts(tmp_path):
    config = ExperimentConfig(
        name="paired",
        output_dir=str(tmp_path),
        seed=23,
        scheduler={
            "scheduler_type": "fixed",
            "allow_deviation": False,
            "llm": {"api_key": "test-only-key"},
        },
        optimizer={
            "max_rounds": 1,
            "llm": {"api_key": "test-only-key"},
        },
        executor={"trace_enabled": True},
        early_stopping_patience=None,
    )
    llm_config = config.model_copy(deep=True)
    llm_config.name = "paired_scheduler_ablation"
    llm_config.scheduler.scheduler_type = "graph"
    llm_config.scheduler.allow_deviation = False

    workflow = _tool_workflow()
    data = [(f"row-{index}", f"row-{index}") for index in range(10)]
    split_indices = {
        "optimization": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7],
        "test": [8, 9],
    }
    manifest = create_manifest(
        config=config,
        data=data,
        split_indices=split_indices,
        initial_workflow=workflow,
        run_metadata={
            "benchmark": "math",
            "code_execution_mode": "not_applicable",
        },
    )

    results_dir = tmp_path / "paired"
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best_workflow.yaml"
    dump_workflow(workflow, checkpoint_path)
    atomic_write_json_0600(
        checkpoint_dir / "checkpoint_meta.json",
        {"round": 0, "score": 1.0, "metadata": {}},
    )
    manifest = bind_best_checkpoint(
        manifest,
        checkpoint_path=checkpoint_path,
        workflow=workflow,
        round_num=0,
    )
    results = {
        "manifest": manifest,
        "checkpoint_summary": {"best_round": 0},
        "test_metrics": None,
        "official_test_metrics": None,
        "test_evaluated_at": None,
        "official_test_evaluated": False,
    }
    results_path = results_dir / "results.json"
    atomic_write_json_0600(results_path, results)

    initial_workflow_path = tmp_path / "initial_workflow.yaml"
    dump_workflow(workflow, initial_workflow_path)
    fixed_config_path = tmp_path / "fixed.yaml"
    fixed_config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    llm_config_path = tmp_path / "llm.yaml"
    llm_config_path.write_text(
        yaml.safe_dump(llm_config.model_dump(mode="json"), sort_keys=False)
    )
    return {
        "config": config,
        "llm_config": llm_config,
        "workflow": workflow,
        "data": data,
        "results": results,
        "results_path": results_path,
        "fixed_config_path": fixed_config_path,
        "llm_config_path": llm_config_path,
        "initial_workflow_path": initial_workflow_path,
    }


def _args(artifact) -> argparse.Namespace:
    return argparse.Namespace(
        results=str(artifact["results_path"]),
        fixed_config=str(artifact["fixed_config_path"]),
        llm_config=str(artifact["llm_config_path"]),
        benchmark="math",
        data="unused.jsonl",
        initial_workflow=str(artifact["initial_workflow_path"]),
        allow_local_code_execution=False,
        use_bubblewrap_code_sandbox=False,
        allow_zero_hard_reward=False,
    )


@pytest.mark.asyncio
async def test_paired_ablation_evaluates_only_exact_validation_indices(
    ablation_artifacts,
    monkeypatch,
):
    artifact = ablation_artifacts
    reward = RecordingReward()

    monkeypatch.setattr(
        run_scheduler_ablation,
        "_load_benchmark",
        lambda *args, **kwargs: (
            reward,
            artifact["data"],
            {"identity": _identity},
        ),
    )
    # The protocol grid and split handling are under test, not network I/O.
    # Make both scheduler variants deterministic while retaining distinct
    # config labels and isolated runners.
    monkeypatch.setattr(
        run_scheduler_ablation.ExperimentRunner,
        "_create_scheduler",
        staticmethod(lambda config: FixedScheduler()),
    )

    output = await run_scheduler_ablation._run_scheduler_ablation(
        _args(artifact)
    )
    validation_queries = ["row-6", "row-7"]
    test_queries = {"row-8", "row-9"}

    assert reward.queries == validation_queries * 4
    assert test_queries.isdisjoint(reward.queries)
    assert output["split"] == {
        "name": "validation",
        "validation_count": 2,
        "validation_indices": [6, 7],
        "validation_indices_sha256": output["split"][
            "validation_indices_sha256"
        ],
        "validation_test_overlap_count": 0,
        "heldout_test_evaluated": False,
    }
    assert set(output["cells"]) == {"initial", "optimized"}
    for workflow_cells in output["cells"].values():
        assert set(workflow_cells) == {"fixed", "llm"}
        for cell in workflow_cells.values():
            assert cell["metrics"]["split"] == "validation"
            assert cell["metrics"]["num_examples"] == 2
            assert set(cell["backend_usage"]) == {
                "workflow",
                "scheduler",
                "total",
            }
            assert cell["execution"]["action_count"] > 0
            assert cell["execution"]["node_execution_count"] > 0
            assert cell["execution"]["trace_artifact"]

    output_path = artifact["results_path"].parent / "scheduler_ablation.json"
    assert json.loads(output_path.read_text()) == output
    assert output_path.stat().st_mode & 0o777 == 0o600


def test_scheduler_variant_cannot_change_other_scientific_config(
    ablation_artifacts,
):
    artifact = ablation_artifacts
    changed = artifact["llm_config"].model_copy(deep=True)
    changed.reward.alpha_process = 0.123

    with pytest.raises(
        ManifestValidationError,
        match="only experiment name, scheduler_type",
    ):
        run_scheduler_ablation._validate_ablation_configs(
            artifact["config"],
            changed,
            artifact["results"]["manifest"],
        )


def test_dataset_manifest_and_initial_workflow_are_bound(
    ablation_artifacts,
    tmp_path,
):
    artifact = ablation_artifacts
    manifest = artifact["results"]["manifest"]
    truncated = artifact["data"][:-1]
    with pytest.raises(ManifestValidationError, match="Dataset row-count mismatch"):
        run_scheduler_ablation._validate_validation_indices(
            truncated,
            manifest,
        )

    changed_workflow = artifact["workflow"].model_copy(deep=True)
    changed_workflow.name = "different-workflow-name"
    changed_path = tmp_path / "changed.yaml"
    dump_workflow(changed_workflow, changed_path)
    with pytest.raises(ManifestValidationError, match="Initial workflow mismatch"):
        run_scheduler_ablation._validate_initial_workflow(
            changed_path,
            manifest,
        )


@pytest.mark.asyncio
async def test_ablation_refuses_results_containing_official_test_outcomes(
    ablation_artifacts,
    monkeypatch,
):
    artifact = ablation_artifacts
    results = json.loads(artifact["results_path"].read_text())
    results["official_test_evaluated"] = True
    results["official_test_metrics"] = {"composite_reward": 1.0}
    atomic_write_json_0600(artifact["results_path"], results)

    def should_not_load(*args, **kwargs):
        raise AssertionError("benchmark must not load after test disclosure")

    monkeypatch.setattr(
        run_scheduler_ablation,
        "_load_benchmark",
        should_not_load,
    )
    with pytest.raises(RuntimeError, match="after official held-out test"):
        await run_scheduler_ablation._run_scheduler_ablation(_args(artifact))


@pytest.mark.parametrize(
    ("benchmark", "local", "bubblewrap", "zero"),
    [
        ("math", False, True, False),
        ("code_gen", True, True, False),
        ("code_gen", False, False, False),
    ],
)
def test_code_execution_cli_modes_are_constrained(
    benchmark,
    local,
    bubblewrap,
    zero,
):
    parser = run_scheduler_ablation._build_parser()
    args = argparse.Namespace(
        benchmark=benchmark,
        allow_local_code_execution=local,
        use_bubblewrap_code_sandbox=bubblewrap,
        allow_zero_hard_reward=zero,
    )
    with pytest.raises(SystemExit):
        run_scheduler_ablation._validate_execution_mode_args(parser, args)
