from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from awf.config.schema import ExperimentConfig
from awf.protocol.manifest import (
    bind_best_checkpoint,
    create_manifest,
    file_sha256,
    json_sha256,
)
from awf.workflow.gates import GateSpec, attach_selective_update
from awf.workflow.serializer import dump_workflow, load_workflow
from benchmarks.multiple_choice.dataset import load_gpqa
from experiments.scripts.prepare_method_comparison import (
    _artifact_sha256,
    main,
    prepare_comparison_spec,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config(api_key: str) -> ExperimentConfig:
    return ExperimentConfig(
        name="gpqa-scwu-fixture",
        output_dir="unused",
        seed=7,
        scheduler={
            "scheduler_type": "graph",
            "llm": {
                "provider": "custom",
                "model": "fixture-model",
                "api_key": api_key,
                "api_base": "https://example.invalid/v1",
            },
        },
        optimizer={
            "max_rounds": 1,
            "selective_update_enabled": True,
            "gate_min_leaf_support": 1,
            "gate_fit_max_traces": 2,
            "llm": {
                "provider": "custom",
                "model": "fixture-model",
                "api_key": api_key,
                "api_base": "https://example.invalid/v1",
            },
        },
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    data_path = tmp_path / "gpqa.jsonl"
    rows = [
        {
            "Record ID": f"row-{index}",
            "Question": f"Which option is correct for item {index}?",
            "Correct Answer": f"correct-{index}",
            "Incorrect Answer 1": f"wrong-a-{index}",
            "Incorrect Answer 2": f"wrong-b-{index}",
            "Incorrect Answer 3": f"wrong-c-{index}",
        }
        for index in range(5)
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    data = load_gpqa(data_path)

    initial_path = tmp_path / "initial.yaml"
    source_workflow = (
        Path(__file__).parents[1]
        / "experiments"
        / "workflows"
        / "multiple_choice"
        / "default_workflow.yaml"
    )
    initial = load_workflow(source_workflow)
    dump_workflow(initial, initial_path)

    candidate = initial.model_copy(deep=True)
    candidate_prompt = "Use the candidate reasoning policy."
    candidate.nodes["answer"].config.system_prompt = candidate_prompt
    candidate.parameters.stages["solve"].blocks["answer"].operators[
        "answer"
    ].prompt.system_prompt = candidate_prompt
    checkpoint = attach_selective_update(
        initial,
        candidate,
        GateSpec(
            kind="threshold",
            feature="query_words",
            operator="gt",
            threshold=4.0,
        ),
        patch_fingerprint="a" * 64,
        changed_units=["answer"],
    )
    checkpoint.version = "1.1"

    config = _config("fixture-secret-never-written")
    config_path = tmp_path / "config.yaml"
    config_payload = config.model_dump(mode="json")
    config_payload["scheduler"]["llm"]["api_key"] = "${FIXTURE_API_KEY}"
    config_payload["optimizer"]["llm"]["api_key"] = "${FIXTURE_API_KEY}"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    split = {
        "optimization": [0, 1, 2],
        "validation": [3],
        "test": [4],
    }
    manifest = create_manifest(
        config=config,
        data=data,
        split_indices=split,
        initial_workflow=initial,
        run_metadata={
            "benchmark": "gpqa",
            "code_execution_mode": "not_applicable",
        },
        dataset_source_path=data_path,
    )
    results_dir = tmp_path / "awf-results"
    checkpoint_path = results_dir / "checkpoints" / "best_workflow.yaml"
    checkpoint_path.parent.mkdir(parents=True)
    dump_workflow(checkpoint, checkpoint_path)
    _write_json(
        checkpoint_path.parent / "checkpoint_meta.json",
        {"round": 1, "score": 0.75},
    )
    manifest = bind_best_checkpoint(
        manifest,
        checkpoint_path=checkpoint_path,
        workflow=checkpoint,
        round_num=1,
    )
    results_path = results_dir / "results.json"
    _write_json(
        results_path,
        {
            "start_time": "2026-07-27T10:00:00",
            "end_time": "2026-07-27T10:02:00",
            "backend_usage": {
                "total": {
                    "available": True,
                    "num_calls": 4,
                    "total_prompt_tokens": 100,
                    "total_completion_tokens": 20,
                    "total_tokens": 120,
                    "total_latency_seconds": 15.0,
                }
            },
            "checkpoint_summary": {"best_round": 1},
            "test_metrics": None,
            "test_evaluated_at": None,
            "manifest": manifest,
        },
    )

    aflow_root = tmp_path / "AFlow"
    (aflow_root / "scripts").mkdir(parents=True)
    (aflow_root / "scripts" / "async_llm.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    round_dir = (
        aflow_root / "workspace" / "GPQA" / "workflows" / "round_2"
    )
    round_dir.mkdir(parents=True)
    (round_dir / "graph.py").write_text(
        "import workspace.GPQA.workflows.round_2.prompt as prompt\n"
        "class Workflow:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (round_dir / "prompt.py").write_text(
        "PROMPT = 'fixture'\n",
        encoding="utf-8",
    )
    telemetry_path = tmp_path / "aflow-search.json"
    _write_json(
        telemetry_path,
        {
            "schema_version": 1,
            "method": "aflow",
            "phase": "search",
            "benchmark": "gpqa",
            "round": 2,
            "start_time": "2026-07-27T02:00:00+00:00",
            "end_time": "2026-07-27T02:00:10+00:00",
            "phase_wall_seconds": 10.0,
            "returncode": 0,
            "dataset_ordered_fingerprint_sha256": manifest["dataset"][
                "ordered_fingerprint_sha256"
            ],
            "optimization_indices_sha256": json_sha256(
                split["optimization"]
            ),
            "artifact": {
                "kind": "directory",
                "relative_path": (
                    "workspace/GPQA/workflows/round_2"
                ),
                "sha256": _artifact_sha256(round_dir),
            },
            "total": {
                "input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 35,
                "call_count": 2,
                "duration_seconds": 3.0,
            },
            "by_role": {
                "workflow_optimizer": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "call_count": 1,
                    "duration_seconds": 1.0,
                },
                "execution_round_2": {
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "total_tokens": 23,
                    "call_count": 1,
                    "duration_seconds": 2.0,
                },
            },
        },
    )
    return {
        "results": results_path,
        "initial": initial_path,
        "round_dir": round_dir,
        "telemetry": telemetry_path,
        "config": config_path,
        "data": data_path,
        "aflow_root": aflow_root,
    }


def _prepare(paths: dict[str, Path]) -> dict:
    return prepare_comparison_spec(
        benchmark="gpqa",
        awf_results_path=paths["results"],
        initial_workflow_path=paths["initial"],
        aflow_graph_path=paths["round_dir"] / "graph.py",
        aflow_search_telemetry_path=paths["telemetry"],
        config_path=paths["config"],
        data_path=paths["data"],
        aflow_root=paths["aflow_root"],
    )


def _with_v2_frozen_aflow_run(paths: dict[str, Path]) -> dict[str, Path]:
    upgraded = dict(paths)
    run_dir = paths["results"].parent.parent / "aflow-run"
    frozen_round = (
        run_dir
        / "frozen_artifacts"
        / "workspace"
        / "GPQA"
        / "workflows"
        / "round_2"
    )
    frozen_round.parent.mkdir(parents=True)
    shutil.copytree(paths["round_dir"], frozen_round)
    calls_path = run_dir / "call_telemetry.jsonl"
    calls_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "role": role,
                    "model": "fixture-model",
                    "status": "ok",
                }
            )
            for role in ("workflow_optimizer", "execution_round_2")
        )
        + "\n",
        encoding="utf-8",
    )
    old = json.loads(paths["telemetry"].read_text(encoding="utf-8"))
    old["schema_version"] = 2
    old["artifact"] = {
        "kind": "directory",
        "base": "search_run",
        "relative_path": frozen_round.relative_to(run_dir).as_posix(),
        "sha256": _artifact_sha256(frozen_round),
    }
    old["provenance"] = {
        "config": {
            "config_file_sha256": "1" * 64,
            "execution_model": "fixture-model",
            "optimizer_model": "fixture-model",
            "execution_role_config_sha256": "2" * 64,
            "optimizer_role_config_sha256": "3" * 64,
            "execution_thinking": "disabled",
            "optimizer_thinking": "enabled",
            "sdk_max_retries": 3,
        },
        "runtime": {
            "aflow_git_commit": None,
            "runtime_tree_sha256": "4" * 64,
        },
        "call_telemetry": {
            "relative_path": calls_path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(calls_path),
            "logical_call_count": 2,
            "counting_semantics": (
                "one event per OpenAI SDK logical call; transport retries "
                "are not separately observable"
            ),
        },
    }
    summary = run_dir / "search_telemetry_summary.json"
    _write_json(summary, old)
    upgraded["round_dir"] = frozen_round
    upgraded["telemetry"] = summary
    upgraded["call_telemetry"] = calls_path
    return upgraded


def test_prepares_hash_bound_builtin_adapter_spec_without_api(
    tmp_path: Path,
):
    paths = _fixture(tmp_path)

    spec = _prepare(paths)

    assert spec["reference_method"] == "vanilla"
    assert [method["kind"] for method in spec["methods"]] == [
        "vanilla",
        "aflow",
        "scwu",
    ]
    vanilla, aflow, scwu = spec["methods"]
    assert vanilla["search_telemetry"]["total_tokens"] == 0
    assert vanilla["search_telemetry"]["wall_latency_seconds"] == 0.0
    assert aflow["search_telemetry"]["total_tokens"] == 35
    assert aflow["search_telemetry"]["wall_latency_seconds"] == 10.0
    assert scwu["search_telemetry"]["total_tokens"] == 120
    assert scwu["search_telemetry"]["wall_latency_seconds"] == 120.0
    assert aflow["artifact_path"].endswith("/round_2")
    assert (
        aflow["identity_metadata"]["provenance"]["artifact_sha256"]
        == _artifact_sha256(paths["round_dir"])
    )
    assert "fixture-secret-never-written" not in json.dumps(spec)
    assert all(
        method["factory"].startswith(
            "experiments.comparison.builtin_adapters:"
        )
        for method in spec["methods"]
    )


def test_prepares_v2_spec_from_frozen_search_run_snapshot(tmp_path: Path):
    paths = _with_v2_frozen_aflow_run(_fixture(tmp_path))

    spec = _prepare(paths)

    aflow = spec["methods"][1]
    assert Path(aflow["artifact_path"]) == paths["round_dir"]
    provenance = aflow["identity_metadata"]["provenance"]
    assert provenance["search_envelope_schema_version"] == 2
    assert (
        provenance["search_provenance"]["config"]["execution_model"]
        == "fixture-model"
    )
    assert (
        provenance["search_provenance"]["call_telemetry"][
            "logical_call_count"
        ]
        == 2
    )


def test_v2_spec_rejects_tampered_frozen_call_log(tmp_path: Path):
    paths = _with_v2_frozen_aflow_run(_fixture(tmp_path))
    paths["call_telemetry"].write_text(
        paths["call_telemetry"].read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="call telemetry"):
        _prepare(paths)


def test_rejects_tampered_aflow_round_artifact(tmp_path: Path):
    paths = _fixture(tmp_path)
    (paths["round_dir"] / "prompt.py").write_text(
        "PROMPT = 'tampered'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _prepare(paths)


def test_rejects_missing_aflow_search_telemetry(tmp_path: Path):
    paths = _fixture(tmp_path)
    telemetry = json.loads(paths["telemetry"].read_text())
    telemetry.pop("by_role")
    _write_json(paths["telemetry"], telemetry)

    with pytest.raises(ValueError, match="missing fields: by_role"):
        _prepare(paths)


def test_rejects_round_one_disguised_as_aflow_search(tmp_path: Path):
    paths = _fixture(tmp_path)
    round_one = paths["round_dir"].parent / "round_1"
    paths["round_dir"].rename(round_one)

    with pytest.raises(ValueError, match="real round_2"):
        prepare_comparison_spec(
            benchmark="gpqa",
            awf_results_path=paths["results"],
            initial_workflow_path=paths["initial"],
            aflow_graph_path=round_one / "graph.py",
            aflow_search_telemetry_path=paths["telemetry"],
            config_path=paths["config"],
            data_path=paths["data"],
            aflow_root=paths["aflow_root"],
        )


def test_rejects_inline_config_key(tmp_path: Path):
    paths = _fixture(tmp_path)
    raw = yaml.safe_load(paths["config"].read_text())
    raw["scheduler"]["llm"]["api_key"] = "inline-secret"
    paths["config"].write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="environment-only"):
        _prepare(paths)


def test_cli_refuses_to_overwrite_before_reading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "comparison.json"
    output.write_text("keep me\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_method_comparison.py",
            "--benchmark",
            "gpqa",
            "--awf-results",
            "missing",
            "--initial-workflow",
            "missing",
            "--aflow-graph",
            "missing",
            "--aflow-search-telemetry",
            "missing",
            "--config",
            "missing",
            "--data",
            "missing",
            "--aflow-root",
            "missing",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep me\n"
