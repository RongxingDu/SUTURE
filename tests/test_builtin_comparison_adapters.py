from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import awf.llm.client as llm_module
from experiments.comparison.adapters import load_configured_adapter
from experiments.comparison.builtin_adapters import (
    _load_aflow_graph,
    build_aflow_artifact_adapter,
    build_awf_artifact_adapter,
)
from experiments.comparison.models import ComparisonSample


def _write_config(path: Path) -> None:
    path.write_text(
        """
name: adapter-test
scheduler:
  scheduler_type: graph
  allow_deviation: false
  llm:
    provider: custom
    model: test-model
    api_key: ${ADAPTER_TEST_API_KEY}
    api_base: https://example.invalid/v1
    temperature: 0.0
    max_tokens: 64
    timeout_seconds: 17
    extra_kwargs:
      top_p: 0.9
      extra_body:
        thinking:
          type: disabled
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _sample() -> ComparisonSample:
    return ComparisonSample(
        sample_id="gpqa:0",
        query="Question\nA. a\nB. b\nC. c\nD. d",
        ground_truth={"answer_index": 0, "answer_letter": "A"},
        split="validation",
        source_index=0,
    )


def test_aflow_snapshot_imports_adjacent_frozen_prompt(tmp_path: Path):
    aflow_root = tmp_path / "AFlow"
    mutable_round = (
        aflow_root / "workspace" / "GPQA" / "workflows" / "round_2"
    )
    mutable_round.mkdir(parents=True)
    (mutable_round / "prompt.py").write_text(
        "MARKER = 'mutable-checkout'\n",
        encoding="utf-8",
    )
    snapshot_round = (
        tmp_path
        / "run"
        / "frozen_artifacts"
        / "workspace"
        / "GPQA"
        / "workflows"
        / "round_2"
    )
    snapshot_round.mkdir(parents=True)
    (snapshot_round / "prompt.py").write_text(
        "MARKER = 'frozen-snapshot'\n",
        encoding="utf-8",
    )
    graph = snapshot_round / "graph.py"
    graph.write_text(
        "import workspace.GPQA.workflows.round_2.prompt as prompt_custom\n"
        "class Workflow:\n"
        "    marker = prompt_custom.MARKER\n",
        encoding="utf-8",
    )

    module = _load_aflow_graph(graph, aflow_root)

    assert module.Workflow.marker == "frozen-snapshot"


@pytest.mark.asyncio
async def test_awf_builtin_adapter_uses_trace_telemetry_and_graph_scheduler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ADAPTER_TEST_API_KEY", "test-secret-value")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    workflow_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "workflows"
        / "multiple_choice"
        / "default_workflow.yaml"
    )

    class FakeClient:
        def __init__(self, config):
            self.config = config

        async def generate(self, **_kwargs):
            return (
                "The final answer is: A",
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                    "latency_seconds": 0.01,
                    "cost_usd": 0.0,
                    "cost_estimate_available": False,
                },
            )

    monkeypatch.setattr(llm_module, "AsyncLLMClient", FakeClient)
    adapter = build_awf_artifact_adapter(
        {
            "name": "vanilla",
            "kind": "vanilla",
            "factory": (
                "experiments.comparison.builtin_adapters:"
                "build_awf_artifact_adapter"
            ),
            "artifact_path": str(workflow_path),
            "identity_metadata": {
                "runtime": {
                    "config_path": str(config_path),
                    "benchmark": "gpqa",
                }
            },
        }
    )

    result = await adapter.infer(_sample())

    assert result.output == "The final answer is: A"
    assert result.telemetry.prompt_tokens == 11
    assert result.telemetry.completion_tokens == 5
    assert result.telemetry.llm_calls == 1
    assert result.telemetry.llm_latency_seconds >= 0.0
    assert "test-secret-value" not in repr(adapter.identity())
    assert adapter.identity()["metadata"]["scheduler"] == "graph"


@pytest.mark.asyncio
async def test_aflow_builtin_adapter_differences_cumulative_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ADAPTER_TEST_API_KEY", "test-secret-value")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    aflow_root = tmp_path / "AFlow"
    (aflow_root / "scripts").mkdir(parents=True)
    (aflow_root / "scripts" / "async_llm.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (aflow_root / "benchmarks").mkdir()
    (aflow_root / "benchmarks" / "__init__.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    template = (
        aflow_root / "workspace" / "GPQA" / "workflows" / "template"
    )
    template.mkdir(parents=True)
    (template / "operator.py").write_text("# fixture\n", encoding="utf-8")
    graph_path = tmp_path / "graph.py"
    graph_path.write_text("# generated AFlow graph\n", encoding="utf-8")

    class FakeCompletions:
        async def create(self, *_args, **_kwargs):
            await asyncio.sleep(0)
            return object()

    class FakeLLM:
        def __init__(self):
            self.prompt = 0
            self.completion = 0
            self.calls = 0
            self.aclient = SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            )

        def get_usage_summary(self):
            return {
                "total_input_tokens": self.prompt,
                "total_output_tokens": self.completion,
                "call_count": self.calls,
                "history": [{} for _ in range(self.calls)],
            }

    class FakeWorkflow:
        received_kwargs = None

        def __init__(self, **kwargs):
            type(self).received_kwargs = kwargs
            self.llm = FakeLLM()
            self.workflow_attempts = 0

        async def __call__(self, _query):
            self.workflow_attempts += 1
            await self.llm.aclient.chat.completions.create()
            self.llm.prompt += 7
            self.llm.completion += 3
            self.llm.calls += 1
            if self.workflow_attempts <= 2:
                raise KeyError("fixture selector failure")
            return "The final answer is: A", 0.0

    observed_sleeps: list[float] = []

    async def fake_sleep(seconds: float):
        observed_sleeps.append(float(seconds))

    monkeypatch.setattr(
        "experiments.comparison.builtin_adapters.asyncio.sleep",
        fake_sleep,
    )
    monkeypatch.setattr(
        "experiments.comparison.builtin_adapters._load_aflow_graph",
        lambda _graph, _root: SimpleNamespace(Workflow=FakeWorkflow),
    )
    adapter = build_aflow_artifact_adapter(
        {
            "name": "aflow",
            "kind": "aflow",
            "factory": (
                "experiments.comparison.builtin_adapters:"
                "build_aflow_artifact_adapter"
            ),
            "artifact_path": str(graph_path),
            "search_telemetry": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "llm_calls": 1,
            },
            "identity_metadata": {
                "runtime": {
                    "config_path": str(config_path),
                    "benchmark": "GPQA",
                    "aflow_root": str(aflow_root),
                }
            },
        }
    )

    first = await adapter.infer(_sample())
    second = await adapter.infer(_sample())

    assert first.output == "The final answer is: A"
    assert first.telemetry.prompt_tokens == 21
    assert first.telemetry.completion_tokens == 9
    assert first.telemetry.llm_calls == 3
    assert first.metadata["workflow_attempt_count"] == 3
    assert first.metadata["runtime_success"] is True
    assert first.metadata["workflow_retry_max_attempts"] == 5
    assert first.metadata["workflow_retry_wait_seconds"] == 1.0
    assert first.metadata["workflow_retry_semantics"] == (
        "AFlow GPQA stop_after_attempt(5)+wait_fixed(1)"
    )
    assert [seconds for seconds in observed_sleeps if seconds > 0] == [1.0, 1.0]
    assert second.telemetry.prompt_tokens == 7
    assert second.telemetry.completion_tokens == 3
    assert second.telemetry.llm_calls == 1
    assert first.telemetry.llm_latency_seconds >= 0.0
    forwarded = FakeWorkflow.received_kwargs["llm_config"]
    assert forwarded["max_tokens"] == 64
    assert forwarded["timeout_seconds"] == 17
    assert forwarded["max_retries"] == 3
    assert forwarded["top_p"] == pytest.approx(0.9)
    assert forwarded["extra_kwargs"] == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert forwarded["telemetry_role"] == "workflow_execution"
    assert adapter.identity()["metadata"]["workflow_retry_policy"] == {
        "max_attempts": 5,
        "wait_seconds": 1.0,
        "retry_on": "Exception",
        "semantics": "AFlow GPQA stop_after_attempt(5)+wait_fixed(1)",
    }


def test_configured_adapter_identity_binds_builtin_delegate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ADAPTER_TEST_API_KEY", "test-secret-value")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    workflow_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "workflows"
        / "multiple_choice"
        / "default_workflow.yaml"
    )

    class FakeClient:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(llm_module, "AsyncLLMClient", FakeClient)
    adapter = load_configured_adapter(
        {
            "name": "vanilla",
            "kind": "vanilla",
            "factory": (
                "experiments.comparison.builtin_adapters:"
                "build_awf_artifact_adapter"
            ),
            "artifact_path": str(workflow_path),
            "identity_metadata": {
                "runtime": {
                    "config_path": str(config_path),
                    "benchmark": "gpqa",
                }
            },
        }
    )

    delegate = adapter.identity()["metadata"]["delegate_identity"]
    assert delegate["metadata"]["implementation"] == "awf_graph_runtime"
    assert delegate["metadata"]["config_sha256"]
    assert "test-secret-value" not in repr(adapter.identity())
