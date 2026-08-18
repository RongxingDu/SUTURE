from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from awf.protocol.manifest import file_sha256
from experiments.comparison.adapters import (
    ArtifactNotReadyError,
    CallableMethodAdapter,
    load_configured_adapter,
    validate_configured_specs_ready,
)
from experiments.comparison.dataset_access import (
    bound_selected_jsonl_file,
    selected_jsonl_file,
    validate_bound_dataset_source,
)
from experiments.comparison.models import (
    AdapterInference,
    ComparisonSample,
    FrozenMethodRoster,
    PhaseTelemetry,
)
from experiments.comparison.runner import (
    ComparisonRunner,
    build_reference_abba_schedule,
)
from experiments.scripts.run_method_comparison import (
    _claim_comparison_test,
    _complete_test_ledger,
    _load_spec,
    _resolve_method_paths,
    _validate_benchmark_execution_mode,
    _validate_builtin_aflow_scope,
    _validate_formal_factory_allowlist,
)
from experiments.scripts.run_test import _claim_official_test_ledger


def _sample(
    sample_id: str,
    *,
    split: str = "validation",
    answer: str = "correct",
) -> ComparisonSample:
    return ComparisonSample(
        sample_id=sample_id,
        query=f"query-{sample_id}",
        ground_truth=answer,
        split=split,
        source_index=int(sample_id.rsplit("-", 1)[-1]),
    )


def _adapter(
    name: str,
    calls: list[tuple[str, str]],
    *,
    kind: str = "mock",
    output: str = "correct",
    inference_telemetry: PhaseTelemetry | None = None,
    search_telemetry: PhaseTelemetry | None = None,
    artifact_path: Path | None = None,
) -> CallableMethodAdapter:
    async def infer(sample: ComparisonSample) -> AdapterInference:
        calls.append((name, sample.sample_id))
        return AdapterInference(
            output=output,
            telemetry=inference_telemetry or PhaseTelemetry(
                prompt_tokens=2,
                completion_tokens=1,
                llm_calls=1,
                llm_latency_seconds=0.1,
                wall_latency_seconds=0.2,
            ),
        )

    return CallableMethodAdapter(
        name=name,
        kind=kind,
        infer=infer,
        search_telemetry=search_telemetry or PhaseTelemetry(),
        artifact_path=artifact_path,
    )


def test_phase_telemetry_is_strict_and_additive():
    first = PhaseTelemetry(
        prompt_tokens=3,
        completion_tokens=2,
        llm_calls=1,
        llm_latency_seconds=0.4,
        wall_latency_seconds=0.5,
    )
    second = PhaseTelemetry.from_mapping(
        {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "total_tokens": 5,
            "llm_calls": 2,
            "llm_latency_seconds": 0.6,
            "latency_seconds": 0.7,
        }
    )

    total = first + second

    assert total.total_tokens == 10
    assert total.llm_calls == 3
    assert total.llm_latency_seconds == pytest.approx(1.0)
    assert total.wall_latency_seconds == pytest.approx(1.2)
    assert total.to_dict()["latency_seconds"] == pytest.approx(1.2)
    with pytest.raises(ValueError, match="total_tokens"):
        PhaseTelemetry.from_mapping(
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 99,
            }
        )
    with pytest.raises(ValueError, match="non-negative"):
        PhaseTelemetry(prompt_tokens=-1)


def test_code_comparison_requires_explicit_shared_execution_mode():
    args = Namespace(
        allow_local_code_execution=False,
        use_bubblewrap_code_sandbox=False,
        allow_zero_hard_reward=False,
        scicode_hdf5=None,
    )

    with pytest.raises(ValueError, match="explicit execution mode"):
        _validate_benchmark_execution_mode(args, "code_gen")

    args.use_bubblewrap_code_sandbox = True
    _validate_benchmark_execution_mode(args, "code_gen")
    with pytest.raises(ValueError, match="only to code_gen"):
        _validate_benchmark_execution_mode(args, "math")


def test_scicode_comparison_requires_hdf5_and_bubblewrap(tmp_path: Path):
    args = Namespace(
        allow_local_code_execution=False,
        use_bubblewrap_code_sandbox=False,
        allow_zero_hard_reward=False,
        scicode_hdf5=None,
    )

    with pytest.raises(ValueError, match="bubblewrap"):
        _validate_benchmark_execution_mode(args, "scicode")

    args.use_bubblewrap_code_sandbox = True
    with pytest.raises(ValueError, match="hdf5"):
        _validate_benchmark_execution_mode(args, "scicode")

    args.scicode_hdf5 = str(tmp_path / "test_data.h5")
    _validate_benchmark_execution_mode(args, "scicode")

    args.scicode_protocol = "independent_subproblems"
    with pytest.raises(ValueError, match="first_subproblem"):
        _validate_benchmark_execution_mode(args, "scicode")


@pytest.mark.parametrize("benchmark", ["gpqa", "mmlu", "scicode"])
def test_comparison_spec_accepts_new_benchmarks(
    benchmark: str,
    tmp_path: Path,
):
    spec = tmp_path / "comparison.json"
    spec.write_text(
        (
            '{"benchmark": "'
            + benchmark
            + '", "data": "data.jsonl", '
            '"protocol_results": "results.json", '
            '"reference_method": "vanilla", "methods": ['
            '{"name": "vanilla", "kind": "vanilla", '
            '"factory": "fixture:vanilla"}, '
            '{"name": "awf", "kind": "awf", '
            '"factory": "fixture:awf", "search_telemetry": {}}]}'
        ),
        encoding="utf-8",
    )

    assert _load_spec(spec)["benchmark"] == benchmark


def test_method_runtime_paths_resolve_relative_to_spec(tmp_path: Path):
    resolved = _resolve_method_paths(
        {
            "name": "aflow",
            "kind": "aflow",
            "factory": "fixture:aflow",
            "artifact_path": "artifacts/round_2",
            "identity_metadata": {
                "runtime": {
                    "config_path": "config.yaml",
                    "aflow_root": "../AFlow",
                    "benchmark": "GPQA",
                }
            },
        },
        tmp_path,
    )

    runtime = resolved["identity_metadata"]["runtime"]
    assert resolved["artifact_path"] == str(
        (tmp_path / "artifacts" / "round_2").resolve()
    )
    assert runtime["config_path"] == str((tmp_path / "config.yaml").resolve())
    assert runtime["aflow_root"] == str((tmp_path / "../AFlow").resolve())


def test_formal_comparison_rejects_arbitrary_same_process_factory():
    with pytest.raises(ValueError, match="audited built-in"):
        _validate_formal_factory_allowlist(
            [
                {
                    "name": "untrusted",
                    "factory": "user_module:read_everything",
                }
            ]
        )


def test_formal_aflow_scope_rejects_unimplemented_benchmark_before_claim(
    tmp_path: Path,
):
    round_dir = (
        tmp_path / "workspace" / "MMLU" / "workflows" / "round_2"
    )
    round_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="only for GPQA"):
        _validate_builtin_aflow_scope(
            [
                {
                    "name": "aflow",
                    "kind": "aflow",
                    "artifact_path": str(round_dir),
                    "identity_metadata": {
                        "runtime": {"benchmark": "mmlu"}
                    },
                }
            ],
            "mmlu",
        )


def test_partial_dataset_source_seal_is_never_treated_as_legacy(
    tmp_path: Path,
):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "dataset": {
            "row_count": 1,
            "source_file_sha256": file_sha256(data_path),
            "source_file_size_bytes": 999999,
        }
    }

    with pytest.raises(ValueError, match="size changed"):
        validate_bound_dataset_source(
            data_path,
            manifest,
            require_seal=False,
        )


def test_reference_abba_schedule_is_stable_for_multiple_methods():
    samples = [_sample("sample-0")]

    schedule = build_reference_abba_schedule(
        samples,
        ["vanilla", "aflow", "scwu"],
        "vanilla",
    )

    assert [item.method_name for item in schedule] == [
        "vanilla",
        "aflow",
        "aflow",
        "vanilla",
        "scwu",
        "vanilla",
        "vanilla",
        "scwu",
    ]
    assert [item.order_position for item in schedule] == [
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
    ]
    assert [item.replicate for item in schedule] == [
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        2,
    ]
    assert [item.block_pattern for item in schedule] == (
        ["ABBA"] * 4 + ["BAAB"] * 4
    )


@pytest.mark.asyncio
async def test_runner_executes_serial_abba_and_separates_search_inference():
    calls: list[tuple[str, str]] = []
    exposed_labels: list[object] = []

    def blind_adapter(
        name: str,
        *,
        kind: str,
        artifact_path: Path | None = None,
    ) -> CallableMethodAdapter:
        async def infer(sample: ComparisonSample) -> AdapterInference:
            calls.append((name, sample.sample_id))
            exposed_labels.append(sample.ground_truth)
            assert dict(sample.metadata) == {}
            return AdapterInference(
                output="correct",
                telemetry=PhaseTelemetry(
                    prompt_tokens=10,
                    completion_tokens=2,
                    llm_calls=1,
                    llm_latency_seconds=0.4,
                    wall_latency_seconds=0.5,
                ),
            )

        return CallableMethodAdapter(
            name=name,
            kind=kind,
            infer=infer,
            search_telemetry=PhaseTelemetry(),
            artifact_path=artifact_path,
        )

    vanilla = blind_adapter(
        "vanilla",
        kind="vanilla",
    )
    aflow_artifact = Path(__file__)
    aflow = _adapter(
        "aflow",
        calls,
        kind="aflow",
        artifact_path=aflow_artifact,
        inference_telemetry=PhaseTelemetry(
            prompt_tokens=20,
            completion_tokens=4,
            llm_calls=2,
            llm_latency_seconds=0.7,
            wall_latency_seconds=0.8,
        ),
        search_telemetry=PhaseTelemetry(
            prompt_tokens=100,
            completion_tokens=50,
            llm_calls=3,
            llm_latency_seconds=4.0,
            wall_latency_seconds=10.0,
        ),
    )
    runner = ComparisonRunner(
        adapters=[vanilla, aflow],
        scorer=lambda sample, output: float(output == sample.ground_truth),
        reference_method="vanilla",
    )

    result = await runner.run_selection(
        [_sample("sample-0"), _sample("sample-1")]
    )

    assert calls == [
        ("vanilla", "sample-0"),
        ("aflow", "sample-0"),
        ("aflow", "sample-0"),
        ("vanilla", "sample-0"),
        ("aflow", "sample-1"),
        ("vanilla", "sample-1"),
        ("vanilla", "sample-1"),
        ("aflow", "sample-1"),
    ]
    assert result["execution_protocol"]["serial"] is True
    assert result["execution_protocol"]["max_concurrency"] == 1
    assert result["search_telemetry"]["vanilla"]["total_tokens"] == 0
    assert result["search_telemetry"]["aflow"]["total_tokens"] == 150
    vanilla_summary = result["inference"]["method_summary"]["vanilla"]
    aflow_summary = result["inference"]["method_summary"]["aflow"]
    assert vanilla_summary["num_runs"] == 4
    assert aflow_summary["num_runs"] == 4
    assert (
        vanilla_summary["inference_telemetry_total"]["total_tokens"]
        == 48
    )
    assert (
        aflow_summary["inference_telemetry_total"]["total_tokens"]
        == 96
    )
    pair = result["inference"]["pair_summary"]["vanilla__vs__aflow"]
    assert pair["num_paired_samples"] == 2
    deltas = pair["mean_comparator_minus_reference"]
    assert deltas["total_tokens"] == pytest.approx(12.0)
    assert deltas["llm_calls"] == pytest.approx(1.0)
    assert result["frozen_roster"]["method_names"] == [
        "vanilla",
        "aflow",
    ]
    assert all(
        "output" not in row
        for row in result["inference"]["rows"]
    )
    assert exposed_labels and set(exposed_labels) == {None}


@pytest.mark.asyncio
async def test_selection_rejects_heldout_rows_before_inference():
    calls: list[tuple[str, str]] = []
    runner = ComparisonRunner(
        adapters=[
            _adapter("vanilla", calls, kind="vanilla"),
            _adapter("awf", calls),
        ],
        scorer=lambda _sample, _output: 1.0,
    )

    with pytest.raises(ValueError, match="another split"):
        await runner.run_selection([_sample("sample-0", split="test")])

    assert calls == []


@pytest.mark.asyncio
async def test_missing_aflow_artifact_fails_before_any_method_runs(
    tmp_path: Path,
):
    calls: list[tuple[str, str]] = []
    runner = ComparisonRunner(
        adapters=[
            _adapter("vanilla", calls, kind="vanilla"),
            _adapter(
                "aflow",
                calls,
                kind="aflow",
                artifact_path=tmp_path / "round_2" / "graph.py",
                search_telemetry=PhaseTelemetry(
                    prompt_tokens=1,
                    llm_calls=1,
                ),
            ),
        ],
        scorer=lambda _sample, _output: 1.0,
    )

    with pytest.raises(ArtifactNotReadyError, match="does not exist"):
        await runner.run_selection([_sample("sample-0")])

    assert calls == []


def test_configured_aflow_adapter_checks_artifact_before_factory_import(
    tmp_path: Path,
):
    with pytest.raises(ArtifactNotReadyError, match="does not exist"):
        load_configured_adapter(
            {
                "name": "aflow",
                "kind": "aflow",
                "factory": "module_that_must_not_be_imported:factory",
                "artifact_path": str(tmp_path / "missing.py"),
                "search_telemetry": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "llm_calls": 1,
                    "llm_latency_seconds": 0.1,
                    "wall_latency_seconds": 0.1,
                },
            }
        )


def test_configured_roster_preflights_all_artifacts_before_factories(
    tmp_path: Path,
):
    specs = [
        {
            "name": "vanilla",
            "kind": "vanilla",
            "factory": "factory_that_would_be_imported:first",
        },
        {
            "name": "aflow",
            "kind": "aflow",
            "factory": "factory_that_would_be_imported:second",
            "artifact_path": str(tmp_path / "round_2" / "graph.py"),
            "search_telemetry": {
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "llm_calls": 1,
                "llm_latency_seconds": 0.1,
                "wall_latency_seconds": 0.1,
            },
        },
    ]

    with pytest.raises(ArtifactNotReadyError, match="does not exist"):
        validate_configured_specs_ready(specs)


@pytest.mark.asyncio
async def test_configured_adapter_factory_can_return_inference_callable(
    monkeypatch: pytest.MonkeyPatch,
):
    module = types.ModuleType("comparison_mock_factory")

    def build(_spec):
        async def infer(_sample):
            return AdapterInference(
                output="correct",
                telemetry=PhaseTelemetry(
                    prompt_tokens=2,
                    completion_tokens=1,
                    llm_calls=1,
                ),
            )

        return infer

    module.build = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = load_configured_adapter(
        {
            "name": "vanilla",
            "kind": "vanilla",
            "factory": "comparison_mock_factory:build",
        }
    )

    result = await adapter.infer(_sample("sample-0"))

    assert result.output == "correct"
    assert result.telemetry.total_tokens == 3


@pytest.mark.asyncio
async def test_heldout_requires_exact_validation_frozen_artifact(
    tmp_path: Path,
):
    calls: list[tuple[str, str]] = []
    artifact = tmp_path / "graph.py"
    artifact.write_text("version = 1\n")
    vanilla = _adapter("vanilla", calls, kind="vanilla")
    aflow = _adapter(
        "aflow",
        calls,
        kind="aflow",
        artifact_path=artifact,
        search_telemetry=PhaseTelemetry(
            prompt_tokens=1,
            llm_calls=1,
        ),
    )
    runner = ComparisonRunner(
        adapters=[vanilla, aflow],
        scorer=lambda _sample, _output: 1.0,
    )
    selection = await runner.run_selection([_sample("sample-0")])
    roster = FrozenMethodRoster.from_mapping(selection["frozen_roster"])
    calls.clear()
    artifact.write_text("version = 2 -- modified content\n")

    with pytest.raises(RuntimeError, match="differs"):
        await runner.run_test(
            [_sample("sample-1", split="test")],
            frozen_roster=roster,
        )

    assert calls == []


@pytest.mark.asyncio
async def test_heldout_with_frozen_roster_runs_only_test_rows(tmp_path: Path):
    calls: list[tuple[str, str]] = []
    artifact = tmp_path / "workflow.yaml"
    artifact.write_text("name: awf\n")
    vanilla = _adapter("vanilla", calls, kind="vanilla")
    awf = _adapter(
        "awf",
        calls,
        kind="awf",
        artifact_path=artifact,
        search_telemetry=PhaseTelemetry(
            prompt_tokens=7,
            completion_tokens=2,
            llm_calls=1,
        ),
    )
    runner = ComparisonRunner(
        adapters=[vanilla, awf],
        scorer=lambda sample, output: float(output == sample.ground_truth),
    )
    selection = await runner.run_selection([_sample("sample-0")])
    roster = FrozenMethodRoster.from_mapping(selection["frozen_roster"])
    calls.clear()

    result = await runner.run_test(
        [_sample("sample-1", split="test")],
        frozen_roster=roster,
    )

    assert result["phase"] == "test"
    assert result["split"] == "test"
    assert result["frozen_from_validation_run_id"] == selection["run_id"]
    assert calls == [
        ("vanilla", "sample-1"),
        ("awf", "sample-1"),
        ("awf", "sample-1"),
        ("vanilla", "sample-1"),
    ]


def test_heldout_claim_is_atomic_and_permanent(tmp_path: Path):
    protocol = tmp_path / "results.json"
    spec = tmp_path / "comparison.json"
    validation = tmp_path / "selection.json"
    output = tmp_path / "test.json"
    for path, value in (
        (protocol, {"manifest": "fixture"}),
        (spec, {"methods": ["fixture"]}),
        (validation, {"run_id": "validation-1"}),
        (output, {"phase": "test"}),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    roster = FrozenMethodRoster(
        validation_run_id="validation-1",
        reference_method="vanilla",
        method_names=("vanilla", "scwu"),
        method_identity_sha256={"vanilla": "a", "scwu": "b"},
    )
    manifest = {"split": {"indices": {"test": [16, 17, 18, 19]}}}
    ledger = tmp_path / "comparison_test_ledger.json"

    claim = _claim_comparison_test(
        ledger,
        benchmark="gpqa",
        protocol_results_path=protocol,
        manifest=manifest,
        spec_path=spec,
        validation_result_path=validation,
        roster=roster,
        requested_output_path=output,
    )

    assert claim["status"] == "claimed"
    assert ledger.stat().st_mode & 0o777 == 0o600
    with pytest.raises(RuntimeError, match="already claimed"):
        _claim_comparison_test(
            ledger,
            benchmark="gpqa",
            protocol_results_path=protocol,
            manifest=manifest,
            spec_path=spec,
            validation_result_path=validation,
            roster=roster,
            requested_output_path=tmp_path / "another.json",
        )

    _complete_test_ledger(
        ledger,
        output,
        claim_id=str(claim["claim_id"]),
    )
    completed = json.loads(ledger.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["result_file_sha256"]


def test_native_and_comparison_test_share_one_irreversible_claim(
    tmp_path: Path,
):
    protocol = tmp_path / "results.json"
    spec = tmp_path / "comparison.json"
    validation = tmp_path / "selection.json"
    output = tmp_path / "test.json"
    for path, value in (
        (protocol, {"manifest": "fixture"}),
        (spec, {"methods": ["fixture"]}),
        (validation, {"run_id": "validation-1"}),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    manifest = {
        "split": {"indices": {"test": [4]}},
    }
    roster = FrozenMethodRoster(
        validation_run_id="validation-1",
        reference_method="vanilla",
        method_names=("vanilla", "scwu"),
        method_identity_sha256={"vanilla": "a", "scwu": "b"},
    )
    ledger = tmp_path / "comparison_test_ledger.json"

    comparison_claim = _claim_comparison_test(
        ledger,
        benchmark="gpqa",
        protocol_results_path=protocol,
        manifest=manifest,
        spec_path=spec,
        validation_result_path=validation,
        roster=roster,
        requested_output_path=output,
    )
    assert comparison_claim["access_mode"] == "method_comparison"

    with pytest.raises(RuntimeError, match="already claimed"):
        _claim_official_test_ledger(
            ledger,
            results_path=protocol,
            manifest=manifest,
            benchmark="gpqa",
        )


def test_split_private_jsonl_access_never_deserializes_unselected_row(
    tmp_path: Path,
):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        '{"question":"validation","answer":"A"}\n'
        'this is intentionally not valid JSON and represents test\n',
        encoding="utf-8",
    )
    manifest = {
        "dataset": {
            "row_count": 2,
            "source_file_sha256": file_sha256(data_path),
            "source_file_size_bytes": data_path.stat().st_size,
        }
    }

    assert validate_bound_dataset_source(data_path, manifest)
    with selected_jsonl_file(
        data_path,
        [0],
        expected_row_count=2,
    ) as selected:
        rows = [
            json.loads(line)
            for line in selected.read_text(encoding="utf-8").splitlines()
        ]
    assert rows == [{"question": "validation", "answer": "A"}]


def test_bound_split_access_rechecks_exact_bytes_after_path_replacement(
    tmp_path: Path,
):
    data_path = tmp_path / "data.jsonl"
    original = (
        '{"question":"validation","answer":"A"}\n'
        '{"question":"heldout","answer":"B"}\n'
    )
    data_path.write_text(original, encoding="utf-8")
    manifest = {
        "dataset": {
            "source_file_sha256": file_sha256(data_path),
            "source_file_size_bytes": data_path.stat().st_size,
            "row_count": 2,
        }
    }
    validate_bound_dataset_source(data_path, manifest)

    replacement = (
        '{"question":"validation","answer":"A"}\n'
        '{"question":"heldout","answer":"C"}\n'
        '{"question":"extra","answer":"D"}\n'
    )
    data_path.write_text(replacement, encoding="utf-8")

    with pytest.raises(ValueError, match="size changed"):
        with bound_selected_jsonl_file(
            data_path,
            [1],
            manifest=manifest,
        ):
            pass
