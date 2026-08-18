import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from awf.protocol.manifest import file_sha256
from experiments.comparison.source_policy import (
    validate_generated_aflow_directory,
)
from experiments.scripts.prepare_aflow_gpqa import (
    DEFAULT_PILOT,
    DEFAULT_SELECTION_MANIFEST,
    prepare,
)
from experiments.scripts.run_aflow_gpqa_one_update import (
    _aggregate_telemetry,
    _freeze_round_two,
    _success_envelope,
    _validate_role_configs,
)


AFLOW_DIR = Path("/home/rongxing/AFlow")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_prepare_aflow_gpqa_uses_only_shared_optimization(tmp_path):
    target = tmp_path / "AFlow"
    research_manifest = tmp_path / "aflow_gpqa_manifest.json"
    prepared = prepare(
        pilot_path=DEFAULT_PILOT,
        selection_manifest_path=DEFAULT_SELECTION_MANIFEST,
        aflow_dir=target,
        research_manifest_path=research_manifest,
    )
    validate = _read_jsonl(
        target / "data" / "datasets" / "gpqa_validate.jsonl",
    )

    assert len(validate) == 12
    assert all(row["_source_split"] == "train" for row in validate)
    assert not (target / "data" / "datasets" / "gpqa_test.jsonl").exists()
    assert prepared["outputs"]["fitness"]["rows"] == 12
    assert research_manifest.is_file()


def test_prepare_aflow_gpqa_is_minimal_and_has_strict_public_prompt(tmp_path):
    target = tmp_path / "AFlow"
    prepare(
        pilot_path=DEFAULT_PILOT,
        selection_manifest_path=DEFAULT_SELECTION_MANIFEST,
        aflow_dir=target,
        research_manifest_path=tmp_path / "manifest.json",
    )
    rows = _read_jsonl(
        target / "data" / "datasets" / "gpqa_validate.jsonl",
    )
    allowed = {
        "_pilot_index",
        "_source_split",
        "answer",
        "formatter_version",
        "question",
        "sample_id",
    }
    forbidden_fragments = (
        "canary",
        "explanation",
        "validator",
        "writer's difficulty",
    )
    for row in rows:
        assert set(row) == allowed
        assert row["answer"] in {"A", "B", "C", "D"}
        assert all(f"\n{letter}. " in row["question"] for letter in "ABCD")
        assert row["question"].rstrip().endswith(
            "Replace X with A, B, C, or D.",
        )
        serialized = json.dumps(row).lower()
        assert not any(fragment in serialized for fragment in forbidden_fragments)


def test_aflow_deepseek_role_configs_are_separate_and_safe():
    provenance = _validate_role_configs(
        AFLOW_DIR / "config" / "config2.gpqa.example.yaml",
    )
    text = (
        AFLOW_DIR / "config" / "config2.gpqa.example.yaml"
    ).read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" in text
    assert "sk-" not in text
    assert provenance["execution_model"] == "deepseek-v4-flash"
    assert provenance["optimizer_model"] == "deepseek-v4-flash"
    assert provenance["sdk_max_retries"] == 1
    assert isinstance(provenance["config_file_sha256"], str)
    assert provenance["config_file_sha256"]


def test_generated_aflow_policy_rejects_direct_label_or_file_access(
    tmp_path: Path,
):
    (tmp_path / "prompt.py").write_text("PROMPT = 'safe'\n")
    (tmp_path / "graph.py").write_text(
        "from typing import Literal\nclass Workflow: pass\n"
    )
    validate_generated_aflow_directory(tmp_path)

    (tmp_path / "graph.py").write_text(
        "class Workflow:\n"
        "    def leak(self):\n"
        "        return open('/home/user/gpqa_test.jsonl').read()\n"
    )
    with pytest.raises(ValueError, match="forbidden"):
        validate_generated_aflow_directory(tmp_path)


def test_call_telemetry_aggregation_is_not_cumulative(tmp_path):
    events = tmp_path / "calls.jsonl"
    rows = [
        {
            "role": "execution_round_1",
            "status": "ok",
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "duration_seconds": 0.5,
        },
        {
            "role": "workflow_optimizer",
            "status": "ok",
            "input_tokens": 20,
            "output_tokens": 3,
            "total_tokens": 23,
            "duration_seconds": 1.0,
        },
        {
            "role": "execution_round_2",
            "status": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "duration_seconds": 0.25,
        },
    ]
    events.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _aggregate_telemetry(
        events,
        wall_seconds=1.2,
        returncode=0,
    )
    assert summary["total"]["call_count"] == 3
    assert summary["total"]["total_tokens"] == 35
    assert summary["phase_wall_seconds"] == 1.2
    assert summary["by_role"]["execution_round_1"]["total_tokens"] == 12


def test_success_envelope_binds_dataset_split_and_real_round_two(tmp_path):
    aflow_dir = tmp_path / "AFlow"
    artifact = aflow_dir / "workspace" / "GPQA" / "workflows" / "round_2"
    artifact.mkdir(parents=True)
    (artifact / "graph.py").write_text("class Workflow: pass\n")
    (artifact / "prompt.py").write_text("PROMPT = 'x'\n")
    (artifact / "result.csv").write_text("score\n1\n")
    cache = artifact / "__pycache__"
    cache.mkdir()
    (cache / "graph.pyc").write_bytes(b"cache")
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    frozen = _freeze_round_two(artifact, run_dir)
    call_telemetry = run_dir / "call_telemetry.jsonl"
    call_telemetry.write_text("{}\n{}\n{}\n", encoding="utf-8")
    start = datetime(2026, 7, 27, tzinfo=timezone.utc)
    aggregate = {
        "subprocess_returncode": 0,
        "phase_wall_seconds": 2.0,
        "total": {
            "input_tokens": 30,
            "output_tokens": 5,
            "total_tokens": 35,
            "call_count": 3,
            "duration_seconds": 1.5,
        },
        "by_role": {
            "execution_round_1": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "call_count": 1,
                "duration_seconds": 0.25,
            },
            "workflow_optimizer": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "call_count": 1,
                "duration_seconds": 1.0,
            },
            "execution_round_2": {
                "input_tokens": 10,
                "output_tokens": 1,
                "total_tokens": 11,
                "call_count": 1,
                "duration_seconds": 0.25,
            },
        },
        "_events": [
            {"role": "execution_round_1", "status": "ok"},
            {"role": "workflow_optimizer", "status": "ok"},
            {"role": "execution_round_2", "status": "ok"},
        ],
    }
    optimization_indices = [10, 7, 15, 3, 2, 5, 6, 12, 4, 13, 11, 1]
    envelope = _success_envelope(
        aggregate=aggregate,
        start_time=start,
        end_time=start + timedelta(seconds=2),
        pilot_path=DEFAULT_PILOT,
        optimization_indices=optimization_indices,
        artifact_dir=frozen,
        artifact_relative_path=frozen.relative_to(run_dir).as_posix(),
        provenance={
            "config": {
                "config_file_sha256": "a" * 64,
                "execution_model": "deepseek-v4-flash",
                "optimizer_model": "deepseek-v4-flash",
                "execution_role_config_sha256": "b" * 64,
                "optimizer_role_config_sha256": "c" * 64,
                "execution_thinking": "disabled",
                "optimizer_thinking": "enabled",
                "sdk_max_retries": 1,
            },
            "runtime": {
                "aflow_git_commit": "d" * 40,
                "runtime_tree_sha256": "e" * 64,
            },
            "call_telemetry": {
                "relative_path": "call_telemetry.jsonl",
                "sha256": file_sha256(call_telemetry),
                "logical_call_count": 3,
                "counting_semantics": (
                    "one event per OpenAI SDK logical call; transport retries "
                    "are not separately observable"
                ),
            },
        },
    )

    assert envelope["schema_version"] == 2
    assert envelope["method"] == "aflow"
    assert envelope["phase"] == "search"
    assert envelope["benchmark"] == "gpqa"
    assert envelope["round"] == 2
    assert envelope["artifact"]["relative_path"].endswith("round_2")
    assert envelope["artifact"]["base"] == "search_run"
    assert isinstance(envelope["artifact"]["sha256"], str)
    assert (frozen / "result.csv").is_file()
    assert not (frozen / "__pycache__").exists()
    assert isinstance(envelope["dataset_ordered_fingerprint_sha256"], str)
    assert isinstance(envelope["optimization_indices_sha256"], str)
    assert envelope["total"] == {
        key: sum(role[key] for role in envelope["by_role"].values())
        for key in envelope["total"]
    }
