import json
import sys
import warnings
from pathlib import Path

import pytest

from awf.trace.schema import ExecutionTrace, TraceStep
from benchmarks.code_generation.evaluator import RunnerResult
from benchmarks.scicode.dataset import SciCodeDataset, format_scicode_prompt
from benchmarks.scicode.evaluator import (
    SciCodeEvaluator,
    ScientificBubblewrapRunner,
    _build_test_script,
)
from benchmarks.scicode.reward import SciCodeReward


def _problem(problem_id: str = "4", step_number: str = "4.1"):
    return {
        "problem_id": problem_id,
        "problem_name": "fixture",
        "problem_description_main": "public description",
        "problem_io": "public io",
        "required_dependencies": "import numpy as np",
        "sub_steps": [
            {
                "step_number": step_number,
                "step_description_prompt": "Implement a square function.",
                "function_header": "def square(x):\n    pass",
                "return_line": "return result",
                "test_cases": ["assert square(3) == target"],
            },
            {
                "step_number": f"{problem_id}.2",
                "step_description_prompt": "Implement a cube function.",
                "function_header": "def cube(x):\n    pass",
                "return_line": "return result",
                "test_cases": ["assert cube(2) == target"],
            },
        ],
    }


def _write(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_first_subproblem_loader_keeps_hidden_tests_out_of_pairs(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path, protocol="first_subproblem")
    dataset.load()
    pairs = dataset.to_pairs()
    assert len(pairs) == 1
    query, target = pairs[0]
    assert "assert square" not in query
    assert "assert square" in json.dumps(target)
    assert target["task_id"] == "SciCode/4.1"
    assert target["protocol"] == "first_subproblem"
    assert dataset.private_test_specs[target["task_id"]]["tests"] == [
        "assert square(3) == target"
    ]


def test_independent_protocol_is_explicit_and_includes_all_steps(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(
        path,
        protocol="independent_subproblems",
    )
    dataset.load()
    pairs = dataset.to_pairs()
    assert [target["step_id"] for _, target in pairs] == ["4.1", "4.2"]
    assert all(
        target["protocol"] == "independent_subproblems"
        for _, target in pairs
    )


def test_prompt_has_no_test_or_reference_solution():
    prompt = format_scicode_prompt(_problem(), 0)
    assert "assert square" not in prompt
    assert "ground_truth" not in prompt
    assert "def square" in prompt


def test_official_excluded_step_is_not_emitted(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem("62", "62.1")])
    dataset = SciCodeDataset(
        path,
        protocol="independent_subproblems",
    )
    dataset.load()
    pairs = dataset.to_pairs()
    assert [target["step_id"] for _, target in pairs] == ["62.2"]


class _AlwaysPassRunner:
    def __init__(self):
        self.calls = 0

    def run(self, script: str, timeout_seconds: float) -> RunnerResult:
        self.calls += 1
        marker = next(
            line.split("print(", 1)[1].rsplit(")", 1)[0].strip("'\"")
            for line in script.splitlines()
            if line.startswith("print('__AWF_SCICODE_PASS_")
        )
        return RunnerResult(returncode=0, stdout=marker)


def test_evaluator_fails_closed_without_runner(tmp_path, monkeypatch):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    pair = dataset.to_pairs()[0]
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
        runner=None,
    )
    result = evaluator.evaluate_detailed(
        "```python\ndef square(x):\n    return x*x\n```",
        pair[1],
    )
    assert result.compiled
    assert result.tests_executed == 0
    assert result.outcome_codes == ("execution_disabled",)


def test_private_spec_fingerprint_mismatch_is_rejected(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    query, target = dataset.to_pairs()[0]
    del query
    target = {**target, "test_spec_sha256": "0" * 64}
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        evaluator.evaluate_detailed(
            "def square(x):\n    return x*x",
            target,
        )


def test_reward_uses_binary_hard_reward_and_cached_process(tmp_path, monkeypatch):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    query, target = dataset.to_pairs()[0]
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "fake.h5",
        runner=_AlwaysPassRunner(),
    )
    monkeypatch.setattr(evaluator, "validate_assets", lambda: {})
    monkeypatch.setattr(
        "benchmarks.scicode.evaluator._load_hdf5_targets",
        lambda path, step_id, count: [9],
    )
    reward = SciCodeReward(evaluator)
    trace = ExecutionTrace(
        trace_id="trace",
        query_text=query,
        workflow_name="fixture",
        workflow_version="1.0",
        success=True,
        steps=[
            TraceStep(
                step_id="s1",
                step_index=0,
                node_id="generate",
                node_type="llm",
                success=True,
            )
        ],
    )
    output = "```python\ndef square(x):\n    return x*x\n```"
    assert reward.hard_reward(query, target, output, trace) == 1.0
    calls = evaluator.runner.calls
    assert reward.process_reward(query, target, output, trace) == 1.0
    assert evaluator.runner.calls == calls
    diagnostic = trace.metadata["scicode_evaluation"]
    assert "tests" not in diagnostic
    assert "target" not in diagnostic


def test_scientific_runner_mount_is_inserted_before_command_separator(
    tmp_path,
    monkeypatch,
):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(
        "benchmarks.code_generation.evaluator.shutil.which",
        lambda executable: (
            "/usr/bin/bwrap"
            if executable == "bwrap"
            else "/usr/bin/python3"
        ),
    )
    runner = ScientificBubblewrapRunner(site_packages=site_packages)
    command = runner._build_command("print('probe')")
    separator = command.index("--")

    mount = [
        "--dir",
        "/opt",
        "--ro-bind",
        str(site_packages.resolve()),
        "/opt/awf-scientific-packages",
    ]
    mount_index = next(
        index
        for index in range(separator)
        if command[index : index + len(mount)] == mount
    )
    assert mount_index < separator
    assert "print('probe')" not in command
    assert command[separator + 1].startswith("/usr/bin/python3")
    assert command[separator + 2 : separator + 5] == ["-I", "-S", "-"]
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "HOME",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "MPLBACKEND",
    ):
        assert variable in command[:separator]


def test_scientific_runner_uses_private_per_test_target_mount(
    tmp_path,
    monkeypatch,
):
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(
        "benchmarks.code_generation.evaluator.shutil.which",
        lambda executable: (
            "/usr/bin/bwrap"
            if executable == "bwrap"
            else "/usr/bin/python3"
        ),
    )
    runner = ScientificBubblewrapRunner(site_packages=site_packages)
    captured = {}

    def fake_run(script, timeout_seconds, *, target_mount=None):
        captured["script"] = script
        captured["timeout"] = timeout_seconds
        captured["mount"] = target_mount
        with target_mount[0].open("rb") as handle:
            captured["target"] = __import__("pickle").load(handle)
        return RunnerResult(returncode=0, stdout="done")

    monkeypatch.setattr(runner, "_run_script", fake_run)
    result = runner.run_with_target(
        f"open({'__AWF_SCICODE_TARGET_PICKLE_PATH__'!r})",
        {"answer": [1, 2, 3]},
        7.0,
    )
    source, destination = captured["mount"]

    assert result.returncode == 0
    assert captured["timeout"] == 7.0
    assert captured["target"] == {"answer": [1, 2, 3]}
    assert str(destination) in captured["script"]
    assert "__AWF_SCICODE_TARGET_PICKLE_PATH__" not in captured["script"]
    assert not source.exists()


def test_hdf5_fingerprint_is_cached_until_file_stat_changes(
    tmp_path,
    monkeypatch,
):
    hdf5_path = tmp_path / "targets.h5"
    hdf5_path.write_bytes(b"first")
    evaluator = SciCodeEvaluator(
        private_test_specs={},
        hdf5_path=hdf5_path,
    )
    monkeypatch.setitem(sys.modules, "h5py", object())
    calls = []

    def fake_hash(path):
        calls.append(path.read_bytes())
        return f"hash-{len(calls)}"

    monkeypatch.setattr(
        "benchmarks.scicode.evaluator._file_sha256",
        fake_hash,
    )
    first = evaluator.validate_assets()
    second = evaluator.validate_assets()
    assert first == second
    assert calls == [b"first"]

    hdf5_path.write_bytes(b"second-version")
    third = evaluator.validate_assets()
    assert third["sha256"] == "hash-2"
    assert calls == [b"first", b"second-version"]


def test_generated_function_cannot_resolve_private_target_directly():
    script = _build_test_script(
        code="def answer():\n    return target",
        dependencies="",
        test="assert answer() == target",
        target=42,
        marker="done",
    )
    with pytest.raises(NameError, match="target"):
        exec(script, {})


def test_evaluator_rejects_frame_introspection_before_execution(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    _, target = dataset.to_pairs()[0]
    runner = _AlwaysPassRunner()
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
        runner=runner,
    )
    code = """import inspect
def square(x):
    return inspect.currentframe().f_back.f_globals["target"]
"""
    result = evaluator.evaluate_detailed(code, target)

    assert result.compiled
    assert result.entry_point_present
    assert result.tests_executed == 0
    assert result.outcome_codes == ("unsafe_code",)
    assert runner.calls == 0


def test_unsafe_code_receives_zero_process_reward(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    query, target = dataset.to_pairs()[0]
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
        runner=_AlwaysPassRunner(),
    )
    reward = SciCodeReward(evaluator)
    trace = ExecutionTrace(
        trace_id="unsafe",
        query_text=query,
        workflow_name="fixture",
        workflow_version="1.0",
        success=True,
        steps=[
            TraceStep(
                step_id="s1",
                step_index=0,
                node_id="generate",
                node_type="llm",
                success=True,
            )
        ],
    )
    code = """import inspect
def square(x):
    return inspect.currentframe().f_back.f_globals["target"]
"""

    assert reward.hard_reward(query, target, code, trace) == 0.0
    assert reward.process_reward(query, target, code, trace) == 0.0
    assert trace.metadata["scicode_evaluation"]["policy_rejected"]


def test_evaluator_rejects_target_mount_discovery_before_execution(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    _, target = dataset.to_pairs()[0]
    runner = _AlwaysPassRunner()
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
        runner=runner,
    )
    code = """def square(x):
    files = __import__("os").listdir("/opt")
    return open(files[0], "rb").read()
"""
    result = evaluator.evaluate_detailed(code, target)

    assert result.outcome_codes == ("unsafe_code",)
    assert result.tests_executed == 0
    assert runner.calls == 0


def test_evaluator_rejects_proc_mount_table_probe_before_execution(tmp_path):
    path = tmp_path / "scicode.jsonl"
    _write(path, [_problem()])
    dataset = SciCodeDataset(path)
    dataset.load()
    _, target = dataset.to_pairs()[0]
    runner = _AlwaysPassRunner()
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
        runner=runner,
    )
    code = """def square(x):
    return np.loadtxt("/proc/self/mountinfo", dtype=str)
"""
    result = evaluator.evaluate_detailed(code, target)

    assert result.outcome_codes == ("unsafe_code",)
    assert result.tests_executed == 0
    assert runner.calls == 0


def test_target_function_parameter_is_not_rejected(tmp_path):
    problem = _problem()
    problem["sub_steps"][0]["function_header"] = (
        "def second_diff(target, values):\n    pass"
    )
    problem["sub_steps"][0]["test_cases"] = [
        "assert second_diff(1, [0, 1, 2]) == target"
    ]
    path = tmp_path / "scicode.jsonl"
    _write(path, [problem])
    dataset = SciCodeDataset(path)
    dataset.load()
    _, target = dataset.to_pairs()[0]
    evaluator = SciCodeEvaluator(
        private_test_specs=dataset.private_test_specs,
        hdf5_path=tmp_path / "missing.h5",
    )
    result = evaluator.evaluate_detailed(
        "def second_diff(target, values):\n    return values[target]",
        target,
    )

    assert result.outcome_codes == ("execution_disabled",)


def test_legacy_escape_in_official_header_emits_no_syntax_warning():
    header = r"""def square(x):
    '''legacy math notation: \o'''
    pass"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = _problem()
        problem["sub_steps"][0]["function_header"] = header
        format_scicode_prompt(problem, 0)
        from benchmarks.scicode.dataset import _extract_entry_point

        assert _extract_entry_point(header) == "square"
    assert not [
        item for item in caught if issubclass(item.category, SyntaxWarning)
    ]
