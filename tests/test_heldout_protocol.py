"""Regression tests for the strict, one-shot held-out test protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from awf.config.schema import ExperimentConfig
from awf.protocol.experiment import ExperimentRunner
from awf.protocol.manifest import ManifestValidationError
from awf.reward.base import RewardEvaluator
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from experiments.scripts import run_test


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
        name="heldout-test-workflow",
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
async def protocol_artifacts(tmp_path):
    config = ExperimentConfig(
        name="strict",
        output_dir=str(tmp_path),
        seed=19,
        scheduler={
            "scheduler_type": "fixed",
            "llm": {"api_key": "test-only-key"},
        },
        optimizer={
            "max_rounds": 1,
            "llm": {"api_key": "test-only-key"},
        },
        executor={"trace_enabled": False},
        early_stopping_patience=None,
    )
    workflow = _tool_workflow()
    reward = RecordingReward()
    runner = ExperimentRunner(
        config=config,
        workflow=workflow,
        reward_evaluator=reward,
        operators={"identity": _identity},
        run_metadata={
            "benchmark": "math",
            "code_execution_mode": "not_applicable",
        },
    )
    data = [(f"row-{index}", f"row-{index}") for index in range(10)]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "".join(
            json.dumps({"query": query, "answer": answer}) + "\n"
            for query, answer in data
        ),
        encoding="utf-8",
    )
    runner.load_data(data, dataset_source_path=data_path)

    async def no_update(workflow, executor, scheduler, llm_client=None):
        return workflow, {"accepted": False, "gain": 0.0}

    runner.optimizer.optimize_round = no_update
    results = await runner.run()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    return {
        "config": config,
        "config_path": config_path,
        "data": data,
        "data_path": data_path,
        "reward": reward,
        "runner": runner,
        "results": results,
        "results_path": tmp_path / "strict" / "results.json",
    }


@pytest.mark.asyncio
async def test_optimization_never_evaluates_test_and_persists_manifest(
    protocol_artifacts,
):
    artifact = protocol_artifacts
    runner = artifact["runner"]
    test_queries = {query for query, _ in runner.test_data}

    assert test_queries.isdisjoint(artifact["reward"].queries)
    assert artifact["results"]["test_metrics"] is None
    assert artifact["results"]["test_evaluated_at"] is None
    assert artifact["results"]["backend_usage"]["total"]["num_calls"] == 0
    assert (
        artifact["results"]["rounds"][0]["backend_usage_delta"]["total"][
            "num_calls"
        ]
        == 0
    )

    saved = json.loads(artifact["results_path"].read_text())
    manifest = saved["manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["dataset"]["row_count"] == len(artifact["data"])
    assert manifest["dataset"]["source_file_sha256"]
    assert manifest["dataset"]["source_file_size_bytes"] > 0
    assert manifest["split"]["indices"]["test"] == runner.split_indices["test"]
    assert manifest["best_checkpoint"]["round"] == 0
    assert manifest["best_checkpoint"]["file_sha256"]


@pytest.mark.asyncio
async def test_dataset_row_count_mismatch_is_rejected(protocol_artifacts):
    manifest = protocol_artifacts["results"]["manifest"]
    truncated = protocol_artifacts["data"][:-1]

    with pytest.raises(ManifestValidationError, match="row-count mismatch"):
        run_test._validate_dataset_and_indices(truncated, manifest)


@pytest.mark.asyncio
async def test_config_mismatch_is_rejected(protocol_artifacts):
    manifest = protocol_artifacts["results"]["manifest"]
    changed = protocol_artifacts["config"].model_copy(deep=True)
    changed.reward.alpha_process = 0.123

    with pytest.raises(ManifestValidationError, match="Config mismatch"):
        run_test._validate_config(changed, manifest)


@pytest.mark.asyncio
async def test_checkpoint_mismatch_is_rejected(protocol_artifacts):
    results_path = protocol_artifacts["results_path"]
    workflow_path = results_path.parent / "checkpoints" / "best_workflow.yaml"
    workflow_path.write_text(workflow_path.read_text() + "\n# tampered\n")

    with pytest.raises(ManifestValidationError, match="Checkpoint mismatch"):
        run_test._validate_checkpoint(
            results_path.parent,
            protocol_artifacts["results"],
            protocol_artifacts["results"]["manifest"],
        )


@pytest.mark.asyncio
async def test_official_test_uses_exact_indices_and_refuses_repeat(
    protocol_artifacts,
    monkeypatch,
):
    artifact = protocol_artifacts
    evaluation_reward = RecordingReward()

    def fake_load_benchmark(*args, **kwargs):
        return evaluation_reward, artifact["data"], {"identity": _identity}

    monkeypatch.setattr(run_test, "_load_benchmark", fake_load_benchmark)
    args = argparse.Namespace(
        results=str(artifact["results_path"]),
        config=str(artifact["config_path"]),
        benchmark="math",
        data=str(artifact["data_path"]),
        allow_local_code_execution=False,
        allow_zero_hard_reward=False,
    )

    metrics = await run_test._run_final_test(args)
    expected_queries = [
        artifact["data"][index][0]
        for index in artifact["results"]["manifest"]["split"]["indices"]["test"]
    ]
    assert evaluation_reward.queries == expected_queries
    assert metrics["num_examples"] == len(expected_queries)

    saved = json.loads(artifact["results_path"].read_text())
    assert saved["official_test_evaluated"] is True
    assert saved["test_evaluated_at"]
    assert saved["test_metrics"] == metrics
    assert artifact["results_path"].stat().st_mode & 0o777 == 0o600

    with pytest.raises(RuntimeError, match="already been evaluated"):
        await run_test._run_final_test(args)


def test_official_test_uses_an_exclusive_process_claim(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text("{}")

    with run_test._claim_official_test(results_path):
        with pytest.raises(RuntimeError, match="already in progress"):
            with run_test._claim_official_test(results_path):
                pass

    assert not (
        tmp_path / ".results.json.official-test.lock"
    ).exists()
