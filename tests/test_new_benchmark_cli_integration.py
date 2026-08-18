"""CLI and minimal-workflow integration for GPQA, MMLU, and SciCode."""

from __future__ import annotations

import json

import pytest

from awf.config.loader import load_config
from awf.protocol.manifest import file_sha256
from awf.config.schema import ExecutorConfig, LLMConfig, SchedulerConfig
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.executor.safety import is_counterfactual_safe
from awf.scheduler.graph_scheduler import GraphScheduler
from awf.workflow.nodes import NodeType
from awf.workflow.serializer import load_workflow
from benchmarks.multiple_choice import MultipleChoiceReward
from benchmarks.scicode import SciCodeEvaluator, SciCodeReward
from experiments.scripts.run_optimization import (
    _default_workflow_path,
    _load_benchmark,
    _run_metadata,
)
from experiments.scripts.run_test import _validate_run_metadata
from experiments.workflows.scicode.operators import extract_scicode_code


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("benchmark", "row", "expected_dataset"),
    [
        (
            "gpqa",
            {
                "Record ID": "gpqa-fixture",
                "Question": "Pick the even number.",
                "Correct Answer": "2",
                "Incorrect Answer 1": "1",
                "Incorrect Answer 2": "3",
                "Incorrect Answer 3": "5",
            },
            "gpqa",
        ),
        (
            "mmlu",
            {
                "sample_id": "mmlu-fixture",
                "subject": "arithmetic",
                "question": "What is 1 + 1?",
                "choices": ["1", "2", "3", "4"],
                "answer": 1,
            },
            "mmlu",
        ),
    ],
)
def test_cli_loads_multiple_choice_benchmarks(
    tmp_path,
    benchmark,
    row,
    expected_dataset,
):
    path = _write_jsonl(tmp_path / f"{benchmark}.jsonl", [row])

    reward, data, operators = _load_benchmark(benchmark, str(path))

    assert isinstance(reward, MultipleChoiceReward)
    assert data[0][1]["dataset"] == expected_dataset
    assert "The final answer is: X" in data[0][0]
    assert operators == {}


def test_cli_loads_scicode_with_private_specs_and_safe_extractor(
    tmp_path,
    monkeypatch,
):
    import benchmarks.scicode as scicode_package

    class _StubScientificRunner:
        pass

    monkeypatch.setattr(
        scicode_package,
        "ScientificBubblewrapRunner",
        _StubScientificRunner,
    )
    monkeypatch.setattr(
        SciCodeEvaluator,
        "validate_assets",
        lambda self: {"sha256": "fixture"},
    )
    data_path = _write_jsonl(
        tmp_path / "scicode.jsonl",
        [
            {
                "problem_id": "4",
                "required_dependencies": "import numpy as np",
                "sub_steps": [
                    {
                        "step_number": "4.1",
                        "step_description_prompt": "Implement square.",
                        "function_header": "def square(x):\n    pass",
                        "return_line": "return result",
                        "test_cases": [
                            "assert square(3) == target",
                        ],
                    }
                ],
            }
        ],
    )
    hdf5_path = tmp_path / "test_data.h5"
    hdf5_path.write_bytes(b"h5 fixture")

    reward, data, operators = _load_benchmark(
        "scicode",
        str(data_path),
        use_bubblewrap_code_sandbox=True,
        scicode_hdf5_path=hdf5_path,
    )

    assert isinstance(reward, SciCodeReward)
    assert isinstance(reward.evaluator.runner, _StubScientificRunner)
    assert data[0][1]["protocol"] == "first_subproblem"
    assert "assert square" not in data[0][0]
    assert "assert square" in json.dumps(data[0][1])
    assert operators["extract_scicode_code"] is extract_scicode_code
    assert is_counterfactual_safe(extract_scicode_code)


def test_scicode_loader_rejects_host_execution(tmp_path):
    with pytest.raises(ValueError, match="bubblewrap"):
        _load_benchmark(
            "scicode",
            str(tmp_path / "missing.jsonl"),
            scicode_hdf5_path=tmp_path / "missing.h5",
        )


def test_scicode_manifest_binds_protocol_and_hdf5_content(tmp_path):
    hdf5_path = tmp_path / "test_data.h5"
    hdf5_path.write_bytes(b"official target fixture")

    metadata = _run_metadata(
        "scicode",
        False,
        use_bubblewrap_code_sandbox=True,
        scicode_hdf5_path=hdf5_path,
        scicode_protocol="first_subproblem",
    )

    assert metadata["code_execution_mode"] == "scientific_bubblewrap"
    assert metadata["scicode_hdf5_sha256"] == file_sha256(hdf5_path)
    manifest = {"run_metadata": metadata}
    _validate_run_metadata(
        manifest,
        "scicode",
        False,
        use_bubblewrap_code_sandbox=True,
        scicode_hdf5_path=hdf5_path,
        scicode_protocol="first_subproblem",
    )

    with pytest.raises(ValueError, match="execution mode mismatch"):
        _validate_run_metadata(
            manifest,
            "scicode",
            False,
            use_bubblewrap_code_sandbox=True,
            scicode_hdf5_path=hdf5_path,
            scicode_protocol="independent_subproblems",
        )


def test_minimal_workflows_are_valid_and_one_call():
    multiple_choice = load_workflow(
        "experiments/workflows/multiple_choice/default_workflow.yaml"
    )
    scicode = load_workflow(
        "experiments/workflows/scicode/default_workflow.yaml"
    )

    assert sum(
        node.node_type == NodeType.LLM
        for node in multiple_choice.nodes.values()
    ) == 1
    assert multiple_choice.get_node_order() == ["start", "answer", "end"]
    assert sum(
        node.node_type == NodeType.LLM
        for node in scicode.nodes.values()
    ) == 1
    assert scicode.get_node_order() == [
        "start",
        "generate",
        "finalize",
        "end",
    ]
    assert scicode.nodes["finalize"].config.tool_name == (
        "extract_scicode_code"
    )


@pytest.mark.parametrize("benchmark", ["gpqa", "mmlu"])
def test_multiple_choice_cli_defaults_to_shared_workflow(benchmark):
    assert _default_workflow_path(benchmark).relative_to(
        _default_workflow_path(benchmark).parents[3]
    ).as_posix().endswith(
        "experiments/workflows/multiple_choice/default_workflow.yaml"
    )


class _StubLLM:
    def __init__(self, response):
        self.config = LLMConfig(model="fixture-model")
        self.response = response
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return self.response, {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workflow_path", "response", "operators", "expected"),
    [
        (
            "experiments/workflows/multiple_choice/default_workflow.yaml",
            "Reasoning.\nThe final answer is: B",
            {},
            "Reasoning.\nThe final answer is: B",
        ),
        (
            "experiments/workflows/scicode/default_workflow.yaml",
            "```python\ndef square(x):\n    return x * x\n```",
            {"extract_scicode_code": extract_scicode_code},
            "def square(x):\n    return x * x",
        ),
    ],
)
async def test_minimal_workflows_execute_exactly_one_llm_call(
    workflow_path,
    response,
    operators,
    expected,
):
    workflow = load_workflow(workflow_path)
    scheduler = GraphScheduler(
        SchedulerConfig(
            scheduler_type="graph",
            allow_deviation=False,
            max_actions_per_query=8,
        )
    )
    client = _StubLLM(response)
    executor = RuntimeExecutor(
        ExecutorConfig(max_steps=8),
        operators=operators,
    )

    output, context, recorder = await executor.execute(
        workflow,
        scheduler,
        "public query",
        client,
    )

    assert output == expected
    assert context.success is True
    assert client.calls == 1
    assert recorder.get_cost_summary()["llm_calls"] == 1


def test_scicode_extractor_uses_last_code_block_without_execution():
    context = ExecutionContext(query="fixture")
    context.record_output(
        "generate",
        "draft\n```python\nraise RuntimeError('not executed')\n```\n"
        "final\n```python\ndef answer():\n    return 42\n```",
    )

    assert extract_scicode_code(context) == (
        "def answer():\n    return 42"
    )


@pytest.mark.parametrize(
    ("filename", "seed"),
    [
        ("gpqa_deepseek_pilot.yaml", 20261930),
        ("mmlu_deepseek_pilot.yaml", 20286299),
        ("scicode_deepseek_pilot.yaml", 20260465),
    ],
)
def test_deepseek_pilot_configs_use_manifest_seed_and_graph_scheduler(
    monkeypatch,
    filename,
    seed,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")

    config = load_config(f"experiments/configs/{filename}")

    assert config.seed == seed
    assert config.scheduler.scheduler_type == "graph"
    assert config.scheduler.allow_deviation is False
    assert config.optimizer.selective_update_enabled is True
    assert config.optimizer.efficiency_optimization_enabled is True
    assert config.optimizer.candidate_archive_size == 3
    assert config.optimizer.exploration_budget == 1
