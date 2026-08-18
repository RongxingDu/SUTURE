#!/usr/bin/env python3
"""Run or dry-smoke the local AFlow GPQA one-update baseline.

Real execution is opt-in via ``--execute``. The API key is read by AFlow from
``DEEPSEEK_API_KEY`` and is never copied into command arguments, config files,
telemetry, or the run summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from awf.protocol.manifest import (
    file_sha256,
    json_sha256,
    ordered_dataset_sha256,
)
from benchmarks.multiple_choice.dataset import load_gpqa
from experiments.scripts.prepare_aflow_gpqa import prepare


DEFAULT_AFLOW_DIR = Path("/home/rongxing/AFlow")
DEFAULT_CONFIG = DEFAULT_AFLOW_DIR / "config" / "config2.gpqa.example.yaml"
DEFAULT_RUNS_DIR = Path("/home/rongxing/Benchmark/AFlow/GPQA/runs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_role_configs(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config.get("models", {})
    execution = models.get("deepseek-v4-flash-execution")
    optimizer = models.get("deepseek-v4-flash-optimizer")
    if not isinstance(execution, dict) or not isinstance(optimizer, dict):
        raise ValueError(
            "AFlow config must define separate execution/optimizer entries"
        )
    for role, value in (("execution", execution), ("optimizer", optimizer)):
        if value.get("model") != "deepseek-v4-flash":
            raise ValueError(f"{role} config must call deepseek-v4-flash")
        if value.get("api_key_env") != "DEEPSEEK_API_KEY":
            raise ValueError(f"{role} config must inject DEEPSEEK_API_KEY")
        if value.get("api_key"):
            raise ValueError(f"{role} config must not persist an API key")
        if (
            not value.get("max_tokens")
            or not value.get("timeout_seconds")
            or value.get("max_retries") != 1
        ):
            raise ValueError(
                f"{role} config lacks matched max_tokens/timeout/retry policy"
            )
    execution_thinking = (
        execution.get("extra_kwargs", {})
        .get("extra_body", {})
        .get("thinking", {})
        .get("type")
    )
    optimizer_thinking = (
        optimizer.get("extra_kwargs", {})
        .get("extra_body", {})
        .get("thinking", {})
        .get("type")
    )
    if execution_thinking != "disabled":
        raise ValueError("DeepSeek workflow execution must disable thinking")
    if optimizer_thinking != "enabled":
        raise ValueError("DeepSeek workflow optimization must enable thinking")
    return {
        "config_file_sha256": file_sha256(config_path),
        "execution_model": str(execution["model"]),
        "optimizer_model": str(optimizer["model"]),
        "execution_role_config_sha256": json_sha256(execution),
        "optimizer_role_config_sha256": json_sha256(optimizer),
        "execution_thinking": str(execution_thinking),
        "optimizer_thinking": str(optimizer_thinking),
        "sdk_max_retries": int(execution["max_retries"]),
    }


def _offline_smoke(*, python: Path, aflow_dir: Path) -> None:
    code = r"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from benchmarks.gpqa import GPQABenchmark
from scripts.async_llm import LLMConfig
from scripts.evaluator import Evaluator
from workspace.GPQA.workflows.round_1.graph import Workflow

async def main():
    evaluator = Evaluator(eval_path="workspace/GPQA/workflows/round_1")
    assert evaluator._get_data_path("GPQA", False) == "data/datasets/gpqa_validate.jsonl"
    assert evaluator._get_data_path("GPQA", True) == "data/datasets/gpqa_test.jsonl"
    assert not Path("data/datasets/gpqa_test.jsonl").exists()
    benchmark = GPQABenchmark(
        name="GPQA",
        file_path="data/datasets/gpqa_validate.jsonl",
        log_path="workspace/GPQA/workflows/round_1",
    )
    rows = await benchmark.load_data()
    assert len(rows) == 12
    assert all(row["answer"] in "ABCD" for row in rows)
    assert all("\nA. " in row["question"] for row in rows)
    assert benchmark.calculate_score(
        "B", "reasoning\nThe final answer is: B"
    )[0] == 1.0
    assert benchmark.calculate_score("B", "B")[0] == 0.0
    assert benchmark.calculate_score(
        "B", "The final answer is: B\nextra"
    )[0] == 0.0

    workflow = Workflow(
        name="GPQA",
        llm_config=LLMConfig({
            "model": "offline-smoke",
            "key": "offline-placeholder",
            "base_url": "https://invalid.local",
            "max_tokens": 16,
            "timeout_seconds": 1,
            "extra_kwargs": {
                "extra_body": {"thinking": {"type": "disabled"}}
            },
        }),
        dataset="GPQA",
    )
    captured = {}
    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="offline response"),
                    )
                ],
            )
    workflow.llm.aclient = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    assert await workflow.llm("offline request") == "offline response"
    assert captured["model"] == "offline-smoke"
    assert captured["max_tokens"] == 16
    assert captured["extra_body"]["thinking"]["type"] == "disabled"
    assert workflow.llm.config.timeout_seconds == 1

    class StubCustom:
        async def __call__(self, input, instruction):
            assert "A. " in input and "The final answer is: X" in instruction
            return {"response": "offline\nThe final answer is: A"}
    workflow.custom = StubCustom()
    output, _ = await workflow(rows[0]["question"])
    assert output.endswith("The final answer is: A")

asyncio.run(main())
"""
    subprocess.run(
        [str(python), "-c", code],
        cwd=aflow_dir,
        check=True,
    )
    _offline_one_update_smoke(python=python, aflow_dir=aflow_dir)


def _offline_one_update_smoke(*, python: Path, aflow_dir: Path) -> None:
    """Exercise the real AFlow CLI/update path with an in-memory fake LLM."""
    fake_graph = """
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(
            input=problem,
            instruction=prompt_custom.GPQA_FINAL_PROMPT,
        )
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
""".strip()
    fake_prompt = """
GPQA_FINAL_PROMPT = '''
Recheck the public choices and end with exactly:
The final answer is: X
'''
""".strip()
    optimizer_response = (
        "<modification>Use a GPQA-specific final-answer reminder.</modification>"
        f"<graph>{fake_graph}</graph>"
        f"<prompt>{fake_prompt}</prompt>"
    )
    sitecustomize = f"""
from scripts.async_llm import AsyncLLM

async def _offline_call(self, prompt):
    if self.config.telemetry_role == "workflow_optimizer":
        return {optimizer_response!r}
    return "offline reasoning\\nThe final answer is: A"

AsyncLLM.__call__ = _offline_call
"""
    with tempfile.TemporaryDirectory(prefix="aflow-gpqa-smoke-") as raw_tmp:
        copied = Path(raw_tmp) / "AFlow"
        shutil.copytree(
            aflow_dir,
            copied,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                "*.pyc",
                "round_2",
            ),
        )
        # The source checkout may legitimately contain a completed real run.
        # A smoke must always exercise one update from a pristine temporary
        # state rather than inheriting its results/experience pointers.
        copied_workflows = (
            copied / "workspace" / "GPQA" / "workflows"
        )
        for relative in (
            Path("results.json"),
            Path("processed_experience.json"),
            Path("round_1/log.json"),
        ):
            (copied_workflows / relative).write_text(
                "[]\n",
                encoding="utf-8",
            )
        for csv_path in (
            copied_workflows / "round_1"
        ).glob("*.csv"):
            csv_path.unlink()
        (copied / "sitecustomize.py").write_text(
            sitecustomize,
            encoding="utf-8",
        )
        env = dict(os.environ)
        env.update(
            {
                "AFLOW_CONFIG": str(
                    copied / "config" / "config2.gpqa.example.yaml",
                ),
                "AFLOW_ROUND_SLEEP_SECONDS": "0",
                "DEEPSEEK_API_KEY": "offline-placeholder",
                "PYTHONPATH": (
                    str(copied)
                    + (
                        os.pathsep + env["PYTHONPATH"]
                        if env.get("PYTHONPATH")
                        else ""
                    )
                ),
            },
        )
        env.pop("AFLOW_TELEMETRY_PATH", None)
        completed = subprocess.run(
            [
                str(python),
                "run.py",
                "--dataset",
                "GPQA",
                "--sample",
                "1",
                "--initial_round",
                "1",
                "--max_rounds",
                "1",
                "--validation_rounds",
                "1",
                "--optimized_path",
                "workspace",
                "--opt_model_name",
                "deepseek-v4-flash-optimizer",
                "--exec_model_name",
                "deepseek-v4-flash-execution",
            ],
            cwd=copied,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Offline AFlow one-update smoke failed:\n"
                + completed.stdout[-4000:]
                + "\n"
                + completed.stderr[-4000:],
            )
        generated = (
            copied / "workspace" / "GPQA" / "workflows" / "round_2"
        )
        for filename in ("graph.py", "prompt.py"):
            path = generated / filename
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"Offline AFlow smoke did not generate {filename}:\n"
                    + completed.stdout[-4000:]
                    + "\n"
                    + completed.stderr[-4000:],
                )
        results = json.loads(
            (
                copied / "workspace" / "GPQA" / "workflows" / "results.json"
            ).read_text(encoding="utf-8"),
        )
        if {int(row["round"]) for row in results} != {1, 2}:
            raise RuntimeError(
                "Offline AFlow smoke did not evaluate both rounds 1 and 2",
            )


def _assert_pristine_one_update_state(aflow_dir: Path) -> None:
    workflows = aflow_dir / "workspace" / "GPQA" / "workflows"
    round_two = workflows / "round_2"
    results = workflows / "results.json"
    round_one_log = workflows / "round_1" / "log.json"
    if round_two.exists():
        raise RuntimeError(
            "GPQA round_2 already exists; use a clean checkout/worktree for "
            "a reproducible one-update run"
        )
    if json.loads(results.read_text(encoding="utf-8")) != []:
        raise RuntimeError("GPQA results.json is not pristine")
    if json.loads(round_one_log.read_text(encoding="utf-8")) != []:
        raise RuntimeError("GPQA round_1/log.json is not pristine")


def _aggregate_telemetry(
    telemetry_path: Path,
    *,
    wall_seconds: float,
    returncode: int,
) -> dict[str, Any]:
    rows = _read_jsonl(telemetry_path)

    def blank() -> dict[str, Any]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "duration_seconds": 0.0,
        }

    total = blank()
    by_role: dict[str, dict[str, Any]] = defaultdict(blank)
    for row in rows:
        role = str(row.get("role", "unclassified"))
        for bucket in (total, by_role[role]):
            bucket["call_count"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                bucket[key] += int(row.get(key, 0))
            bucket["duration_seconds"] += float(
                row.get("duration_seconds", 0.0),
            )
    return {
        "subprocess_returncode": returncode,
        "phase_wall_seconds": float(wall_seconds),
        "total": total,
        "by_role": dict(sorted(by_role.items())),
        "_events": rows,
    }


def _file_content_identity(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _directory_sha256(path: Path) -> str:
    files = sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and "__pycache__" not in child.parts
        and child.suffix != ".pyc"
    )
    return json_sha256(
        [
            {
                "relative_path": child.relative_to(path).as_posix(),
                "sha256": _file_content_identity(child),
            }
            for child in files
        ],
    )


def _content_tree_sha256(base: Path, roots: tuple[Path, ...]) -> str:
    """Hash the AFlow source/runtime files used by the GPQA search."""
    records: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"AFlow runtime identity root missing: {root}")
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for child in files:
            if (
                not child.is_file()
                or "__pycache__" in child.parts
                or child.suffix == ".pyc"
            ):
                continue
            records[child.relative_to(base).as_posix()] = _file_content_identity(child)
    return json_sha256(
        [
            {"relative_path": relative_path, "sha256": digest}
            for relative_path, digest in sorted(records.items())
        ]
    )


def _aflow_runtime_tree_sha256(aflow_dir: Path) -> str:
    workflows = aflow_dir / "workspace" / "GPQA" / "workflows"
    return _content_tree_sha256(
        aflow_dir,
        (
            aflow_dir / "run.py",
            aflow_dir / "scripts",
            aflow_dir / "benchmarks",
            workflows / "template",
            workflows / "round_1" / "__init__.py",
            workflows / "round_1" / "graph.py",
            workflows / "round_1" / "prompt.py",
        ),
    )


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _freeze_round_two(source: Path, run_dir: Path) -> Path:
    """Copy a complete, cache-free round_2 snapshot into one search run."""
    snapshot = (
        run_dir
        / "frozen_artifacts"
        / "workspace"
        / "GPQA"
        / "workflows"
        / "round_2"
    )
    shutil.copytree(
        source,
        snapshot,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        copy_function=shutil.copy2,
    )
    return snapshot


def _success_envelope(
    *,
    aggregate: dict[str, Any],
    start_time: datetime,
    end_time: datetime,
    pilot_path: Path,
    optimization_indices: list[int],
    artifact_dir: Path,
    artifact_relative_path: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if aggregate["subprocess_returncode"] != 0:
        raise ValueError("Cannot create a success envelope for a failed run")
    events = aggregate["_events"]
    if aggregate["total"]["call_count"] <= 0:
        raise ValueError("AFlow search emitted no API call telemetry")
    for required_role in ("workflow_optimizer", "execution_round_2"):
        if not any(
            row.get("role") == required_role
            and row.get("status") == "ok"
            for row in events
        ):
            raise ValueError(
                f"AFlow search lacks a successful {required_role} call",
            )

    for required_file in ("graph.py", "prompt.py"):
        path = artifact_dir / required_file
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"AFlow round_2 lacks non-empty {required_file}")

    pairs = load_gpqa(pilot_path)
    total = aggregate["total"]
    by_role = aggregate["by_role"]
    return {
        "schema_version": 2,
        "method": "aflow",
        "phase": "search",
        "benchmark": "gpqa",
        "round": 2,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "phase_wall_seconds": aggregate["phase_wall_seconds"],
        "returncode": 0,
        "dataset_ordered_fingerprint_sha256": ordered_dataset_sha256(
            pairs,
        ),
        "optimization_indices_sha256": json_sha256(
            optimization_indices
        ),
        "artifact": {
            "kind": "directory",
            "base": "search_run",
            "relative_path": artifact_relative_path,
            "sha256": _directory_sha256(artifact_dir),
        },
        "provenance": provenance,
        "total": total,
        "by_role": by_role,
    }


def run(
    *,
    aflow_dir: Path,
    config_path: Path,
    python: Path,
    execute: bool,
    runs_dir: Path,
) -> Path | None:
    preparation_manifest = prepare(aflow_dir=aflow_dir)
    role_config_provenance = _validate_role_configs(config_path)
    _offline_smoke(python=python, aflow_dir=aflow_dir)
    if not execute:
        return None

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for --execute; inject it through "
            "the environment, never a CLI argument or file"
        )
    _assert_pristine_one_update_state(aflow_dir)
    runtime_tree_sha256 = _aflow_runtime_tree_sha256(aflow_dir)
    aflow_git_commit = _git_head(aflow_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = run_dir / "call_telemetry.jsonl"
    command = [
        str(python),
        "run.py",
        "--dataset",
        "GPQA",
        "--sample",
        "1",
        "--initial_round",
        "1",
        "--max_rounds",
        "1",
        "--validation_rounds",
        "1",
        "--optimized_path",
        "workspace",
        "--opt_model_name",
        "deepseek-v4-flash-optimizer",
        "--exec_model_name",
        "deepseek-v4-flash-execution",
    ]
    env = dict(os.environ)
    env.update(
        {
            "AFLOW_CONFIG": str(config_path.resolve()),
            "AFLOW_TELEMETRY_PATH": str(telemetry_path.resolve()),
            "AFLOW_ROUND_SLEEP_SECONDS": "0",
        },
    )
    start_time = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=aflow_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    end_time = datetime.now(timezone.utc)
    # Derive the declared wall duration from the exact timestamps carried by
    # the envelope, so downstream validation cannot observe clock skew between
    # an independent monotonic timer and UTC timestamps.
    wall_seconds = (end_time - start_time).total_seconds()
    (run_dir / "stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (run_dir / "stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    aggregate = _aggregate_telemetry(
        telemetry_path,
        wall_seconds=wall_seconds,
        returncode=completed.returncode,
    )
    if completed.returncode != 0:
        failure_summary = {
            key: value
            for key, value in aggregate.items()
            if key != "_events"
        }
        (run_dir / "failed_search_telemetry.json").write_text(
            json.dumps(
                failure_summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"AFlow failed with code {completed.returncode}; inspect "
            f"{run_dir / 'stderr.log'}"
        )
    round_two = (
        aflow_dir / "workspace" / "GPQA" / "workflows" / "round_2"
    )
    if not (round_two / "graph.py").is_file() or not (
        round_two / "prompt.py"
    ).is_file():
        raise RuntimeError("AFlow returned success without a round_2 artifact")
    frozen_round_two = _freeze_round_two(round_two, run_dir)
    frozen_relative_path = frozen_round_two.relative_to(run_dir).as_posix()
    events = aggregate["_events"]
    provenance = {
        "config": role_config_provenance,
        "runtime": {
            "aflow_git_commit": aflow_git_commit,
            "runtime_tree_sha256": runtime_tree_sha256,
        },
        "call_telemetry": {
            "relative_path": telemetry_path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(telemetry_path),
            "logical_call_count": len(events),
            "counting_semantics": (
                "one event per OpenAI SDK logical call; transport retries "
                "are not separately observable"
            ),
        },
    }
    summary = _success_envelope(
        aggregate=aggregate,
        start_time=start_time,
        end_time=end_time,
        pilot_path=Path(
            preparation_manifest["source"]["pilot_path"],
        ),
        optimization_indices=preparation_manifest["split_indices"][
            "optimization"
        ],
        artifact_dir=frozen_round_two,
        artifact_relative_path=frozen_relative_path,
        provenance=provenance,
    )
    (run_dir / "search_telemetry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aflow-dir", type=Path, default=DEFAULT_AFLOW_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_AFLOW_DIR / ".venv" / "bin" / "python",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the real one-update API run (default is offline smoke)",
    )
    args = parser.parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(
            f"AFlow Python environment not found: {args.python}",
        )
    run_dir = run(
        aflow_dir=args.aflow_dir,
        config_path=args.config,
        python=args.python,
        execute=args.execute,
        runs_dir=args.runs_dir,
    )
    if run_dir is None:
        print("AFlow GPQA offline smoke passed; no API call was made.")
    else:
        print(f"AFlow GPQA one-update completed: {run_dir}")


if __name__ == "__main__":
    main()
