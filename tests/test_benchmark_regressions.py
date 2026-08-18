"""Regression tests for benchmark correctness and safe execution defaults."""

import json

import pytest

from awf.trace.schema import ExecutionTrace, ToolCallRecord, TraceStep
from benchmarks.agent_tasks.evaluator import AgentTaskEvaluator
from benchmarks.agent_tasks.reward import AgentReward
from benchmarks.code_generation.evaluator import (
    BubblewrapCodeRunner,
    CodeEvaluator,
    RunnerResult,
)
from benchmarks.code_generation.reward import CodeGenReward
from benchmarks.math_reasoning.dataset import GSM8KDataset, MATHDataset
from benchmarks.math_reasoning.evaluator import MathEvaluator
from benchmarks.math_reasoning.reward import MathReward


def _trace() -> ExecutionTrace:
    return ExecutionTrace(trace_id="regression")


def test_code_execution_is_disabled_by_default():
    evaluator = CodeEvaluator()
    result = evaluator.evaluate_detailed(
        "def f(): return 42",
        "def check(candidate): assert candidate() == 42",
        "f",
    )

    assert result.tests_executed == 0
    assert result.pass_rate == 0.0
    assert "Execution disabled" in result.summary()


def test_external_sandbox_runner_is_supported():
    class RecordingRunner:
        def __init__(self):
            self.script = ""

        def run(self, script, timeout_seconds):
            self.script = script
            marker = next(
                line.removeprefix("print(").split(",", 1)[0].strip("'\"")
                for line in script.splitlines()
                if line.startswith("print('__AWF_TEST_COMPLETE__")
            )
            return RunnerResult(returncode=0, stdout=marker)

    runner = RecordingRunner()
    evaluator = CodeEvaluator(sandbox_runner=runner)
    passed, _ = evaluator.evaluate(
        "def f(): return 42",
        "def check(candidate): assert candidate() == 42",
        "f",
    )

    assert passed is True
    assert "check(f)" in runner.script


def test_bubblewrap_command_has_closed_namespace_and_mount_policy():
    if not BubblewrapCodeRunner.is_available():
        pytest.skip("Linux bubblewrap and a bindable Python are unavailable")

    command = BubblewrapCodeRunner()._build_command("print(42)")

    assert "--unshare-all" in command
    assert "--unshare-net" in command
    assert "--clearenv" in command
    assert "--tmpfs" in command
    assert command[command.index("--tmpfs") + 1] == "/tmp"
    bindings = [
        tuple(command[index + 1:index + 3])
        for index, value in enumerate(command)
        if value == "--ro-bind"
    ]
    assert bindings
    assert all(source == target for source, target in bindings)
    assert {source for source, _ in bindings} <= {"/usr", "/lib", "/lib64"}
    assert "/home" not in command


def test_bubblewrap_runner_isolated_integration(monkeypatch):
    if not BubblewrapCodeRunner.is_available():
        pytest.skip("Linux bubblewrap and a bindable Python are unavailable")
    monkeypatch.setenv("AWF_SANDBOX_SECRET", "must-not-cross-boundary")
    runner = BubblewrapCodeRunner()
    result = runner.run(
        """
import os
import socket
from pathlib import Path

assert "AWF_SANDBOX_SECRET" not in os.environ
assert not Path("/etc/passwd").exists()
assert not Path("/home").exists()
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError("network namespace was not isolated")
print("isolated")
""",
        timeout_seconds=2,
    )
    if (
        result.returncode != 0
        and (
            "Operation not permitted" in result.stderr
            or "No permissions" in result.stderr
            or "Creating new namespace failed" in result.stderr
        )
    ):
        pytest.skip(f"Host kernel disallows bubblewrap: {result.stderr}")

    assert result.returncode == 0, result.stderr or result.error_message
    assert result.stdout.strip() == "isolated"

    passed, summary = CodeEvaluator(
        timeout_seconds=2,
        use_bubblewrap=True,
    ).evaluate(
        "def f(): return 42",
        "def check(candidate): assert candidate() == 42",
        "f",
    )
    assert passed is True, summary

    timeout = runner.run("while True: pass", timeout_seconds=0.1)
    assert timeout.timed_out is True


def test_code_evaluator_can_select_only_one_execution_boundary():
    with pytest.raises(ValueError, match="exactly one"):
        CodeEvaluator(
            sandbox_runner=object(),
            use_bubblewrap=True,
        )


def test_humaneval_harness_calls_check_with_entry_point():
    evaluator = CodeEvaluator(
        timeout_seconds=2,
        allow_local_execution=True,
    )
    tests = "def check(candidate):\n    assert candidate() == 42"

    wrong, _ = evaluator.evaluate("def f(): return 0", tests, "f")
    correct, _ = evaluator.evaluate("def f(): return 42", tests, "f")

    assert wrong is False
    assert correct is True


@pytest.mark.parametrize(
    "premature_exit",
    [
        "import sys\nsys.exit(0)",
        "import os\nos._exit(0)",
        "def f():\n    import os\n    os._exit(0)",
    ],
)
def test_zero_exit_before_harness_completion_never_passes(premature_exit):
    evaluator = CodeEvaluator(
        timeout_seconds=2,
        allow_local_execution=True,
    )
    code = (
        "def f():\n    return 42\n"
        + premature_exit
        if not premature_exit.startswith("def f")
        else premature_exit
    )
    result = evaluator.evaluate_detailed(
        code,
        "def check(candidate): assert candidate() == 42",
        "f",
    )

    assert result.passed_tests == 0
    assert result.tests_executed == 1
    assert "completion marker" in result.summary()


def test_mbpp_partial_pass_rate_and_reward():
    evaluator = CodeEvaluator(
        timeout_seconds=2,
        allow_local_execution=True,
    )
    reward = CodeGenReward(evaluator)
    ground_truth = {
        "test_list": [
            "assert add(1, 2) == 3",
            "assert add(-1, 1) == 0",
            "assert add(2, 2) == 5",
        ]
    }
    output = "def add(a, b):\n    return a + b"

    result = evaluator.evaluate_detailed(
        output,
        ground_truth["test_list"],
    )
    hard = reward.hard_reward("add", ground_truth, output, _trace())

    assert result.passed_tests == 2
    assert result.total_tests == 3
    assert result.pass_rate == pytest.approx(2 / 3)
    assert hard == pytest.approx(2 / 3)


@pytest.mark.parametrize("final_output", ["PASS", None])
def test_code_reward_never_falls_back_to_intermediate_code(final_output):
    evaluator = CodeEvaluator(
        timeout_seconds=2,
        allow_local_execution=True,
    )
    trace = _trace()
    trace.steps.append(
        TraceStep(
            step_id="end",
            step_index=0,
            node_id="end",
            node_type="end",
            state_after={
                "history": ["generate", "verify"],
                "outputs": {
                    "generate": "def f():\n    return 42",
                    "verify": "PASS",
                }
            },
        )
    )
    ground_truth = {
        "test": "def check(candidate): assert candidate() == 42",
        "entry_point": "f",
    }
    reward = CodeGenReward(evaluator)

    assert reward.hard_reward(
        "write f",
        ground_truth,
        final_output,
        trace,
    ) == 0.0
    assert reward.process_reward(
        "write f",
        ground_truth,
        final_output,
        trace,
    ) == 0.0
    assert trace.metadata["code_output_contract"]["valid"] is False


def test_code_reward_accepts_normal_finalized_code():
    evaluator = CodeEvaluator(
        timeout_seconds=2,
        allow_local_execution=True,
    )
    code = "def f():\n    return 42"
    trace = _trace()
    trace.steps.append(
        TraceStep(
            step_id="finalize",
            step_index=0,
            node_id="finalize",
            node_type="tool",
            action="execute",
            state_after={
                "history": ["generate", "verify", "finalize"],
                "outputs": {
                    "generate": code,
                    "verify": "PASS",
                    "finalize": code,
                },
            },
            metadata={"node_executed": True},
        )
    )
    ground_truth = {
        "test": "def check(candidate): assert candidate() == 42",
        "entry_point": "f",
    }
    reward = CodeGenReward(evaluator)

    assert reward.hard_reward("write f", ground_truth, code, trace) == 1.0
    assert reward.process_reward("write f", ground_truth, code, trace) == 1.0
    contract = trace.metadata["code_output_contract"]
    assert contract["valid"] is True
    assert contract["output_node_id"] == "finalize"
    assert contract["completion_mode"] == "finalize"


@pytest.mark.parametrize(
    ("code", "parseable", "entry_point_present"),
    [
        ("def other():\n    return 42", True, False),
        ("def f(:\n    return 42", False, False),
    ],
)
def test_code_reward_requires_parseable_final_entry_point(
    code,
    parseable,
    entry_point_present,
):
    trace = _trace()
    trace.steps.append(
        TraceStep(
            step_id="finalize",
            step_index=0,
            node_id="finalize",
            node_type="tool",
            action="execute",
            state_after={
                "history": ["finalize"],
                "outputs": {"finalize": code},
            },
            metadata={"node_executed": True},
        )
    )
    ground_truth = {
        "test": "def check(candidate): assert candidate() == 42",
        "entry_point": "f",
    }
    reward = CodeGenReward(
        CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    )

    assert reward.hard_reward("write f", ground_truth, code, trace) == 0.0
    assert reward.process_reward("write f", ground_truth, code, trace) == 0.0
    contract = trace.metadata["code_output_contract"]
    assert contract["parseable"] is parseable
    assert contract["entry_point_present"] is entry_point_present
    assert contract["valid"] is False


def test_code_extraction_tolerates_unclosed_fence_and_empty_gets_no_process_reward():
    evaluator = CodeEvaluator()
    assert evaluator.extract_code("```python\ndef f():\n    return 1") == (
        "def f():\n    return 1"
    )
    assert CodeGenReward(evaluator).process_reward(
        "q", {}, "", _trace()
    ) == 0.0


def test_math_extracts_nested_boxed_fraction_and_decimal():
    evaluator = MathEvaluator()

    assert evaluator.extract_answer(
        r"Work gives \boxed{\frac{1}{2}}."
    ) == r"\frac{1}{2}"
    assert evaluator.compare_answers(r"\frac{1}{2}", "0.5")
    assert evaluator.evaluate(r"\frac{1}{2}", r"\frac{1}{2}")
    assert evaluator.extract_answer("The final answer is 3.1400.") == "3.1400"
    assert evaluator.compare_answers("3.1400", "3.14")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            r"VERIFIED: \begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
            r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
        ),
        (r"VERIFIED: 100\text{ square units}", r"100\text{ square units}"),
        (r"VERIFIED: 5.4\text{ cents}", r"5.4\text{ cents}"),
    ],
)
def test_math_extracts_unboxed_structured_tex_and_units(response, expected):
    evaluator = MathEvaluator()
    assert evaluator.extract_answer(response) == expected


def test_math_numeric_answers_match_unit_bearing_ground_truth():
    evaluator = MathEvaluator()
    assert evaluator.evaluate("VERIFIED: 100", r"100\text{ square units}")
    assert evaluator.evaluate("VERIFIED: 5.4", r"5.4\text{ cents}")
    assert evaluator.evaluate("VERIFIED: 120", r"120^\circ")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"VERIFIED: \(\frac{37}{819}\)", r"\frac{37}{819}"),
        ("VERIFIED: A", "A"),
        (r"VERIFIED: \boxed{\text{B}}", "B"),
        (r"VERIFIED: \((-\infty,0]\cup[2,3)\)", r"(-\infty,0]\cup[2,3)"),
        (r"VERIFIED: \(\frac{3}{28}\)", r"\frac{3}{28}"),
        (r"**VERIFIED: \(\frac{37}{819}\)**", r"\frac{37}{819}"),
        (r"__VERIFIED: \(\frac{3}{28}\)__", r"\frac{3}{28}"),
        ("### **Final answer:** `\\(42\\)`", "42"),
        ("- *VERIFIED:* A", "A"),
    ],
)
def test_math_extracts_real_verifier_payloads(response, expected):
    evaluator = MathEvaluator()

    assert evaluator.extract_answer(response) == expected
    assert evaluator.evaluate(response, expected)


def test_math_error_prefers_explicit_correction_over_rejected_box():
    evaluator = MathEvaluator()
    response = (
        r"ERROR: the draft's \boxed{5} is an arithmetic mistake. "
        r"The correct answer is \(7\)."
    )

    assert evaluator.extract_answer(response) == "7"
    assert evaluator.evaluate(response, "7")
    assert not evaluator.evaluate(response, "5")


def test_math_markdown_error_prefers_bold_correct_answer():
    evaluator = MathEvaluator()
    response = (
        r"**ERROR:** the draft's \boxed{5} is wrong. "
        r"**Correct answer:** \(\frac{7}{2}\)"
    )

    assert evaluator.extract_answer(response) == r"\frac{7}{2}"
    assert evaluator.evaluate(response, "3.5")
    assert not evaluator.evaluate(response, "5")


def test_math_extracts_probability_below_final_answer_heading():
    evaluator = MathEvaluator()
    response = (
        "### Final Answer:\n"
        r"The correct probability is **$\frac{72}{425}$**."
    )

    assert evaluator.extract_answer(response) == r"\frac{72}{425}"
    assert evaluator.evaluate(response, r"\dfrac{72}{425}")


def test_math_does_not_use_arbitrary_last_number_from_explanation():
    evaluator = MathEvaluator()

    assert evaluator.extract_answer(
        "The calculation mentions 12, then checks intermediate value 99."
    ) == ""
    assert not evaluator.evaluate(
        "The calculation mentions 12, then checks intermediate value 99.",
        "99",
    )
    assert evaluator.extract_answer(
        "- Step 1: compute 12\n- Step 2: check intermediate value 99"
    ) == ""


def test_math_safe_polynomial_and_equation_equivalence():
    evaluator = MathEvaluator()

    assert evaluator.compare_answers(r"2(x+1)", r"2x+2")
    assert evaluator.compare_answers(r"x^2+2x+1", r"(x+1)^2")
    assert evaluator.compare_answers(r"x=2", r"2=x")
    assert evaluator.normalize_ground_truth(r"2x+2") == r"2x+2"
    assert not evaluator.compare_answers(r"x+1", r"x+2")


def test_math_finite_sets_intervals_and_fractions():
    evaluator = MathEvaluator()

    assert evaluator.compare_answers(
        r"\{1,2,\frac{3}{2}\}",
        r"\{1.5,2,1\}",
    )
    assert evaluator.compare_answers(
        r"(-\infty,0]\cup[2,3)",
        r"[2,3)\cup(-\infty,0]",
    )
    assert not evaluator.compare_answers(r"\{1,2\}", r"\{1,3\}")


def test_math_symbolic_parser_rejects_code_and_excessive_complexity(tmp_path):
    evaluator = MathEvaluator()
    marker = tmp_path / "must-not-exist"
    malicious = f"__import__('pathlib').Path('{marker}').touch()"

    assert not evaluator.compare_answers(malicious, "0")
    assert not marker.exists()
    assert not evaluator.compare_answers("x^999999", "x")


def test_math_optional_sympy_rational_equivalence():
    pytest.importorskip("sympy")
    evaluator = MathEvaluator()

    assert evaluator.compare_answers(
        r"\frac{x^2-1}{x-1}",
        r"x+1",
    )


def test_math_datasets_normalize_full_solutions(tmp_path):
    gsm_path = tmp_path / "gsm.jsonl"
    gsm_path.write_text(
        json.dumps(
            {
                "question": "q",
                "answer": "Reasoning with 1.2 first. #### 1,234.50",
            }
        )
        + "\n"
    )
    math_path = tmp_path / "math.jsonl"
    math_path.write_text(
        json.dumps(
            {
                "problem": "p",
                "solution": r"Steps. \boxed{\frac{1}{2}}",
                "type": "Number Theory",
                "level": "Level 5",
                "_aflow_source_split": "validate",
                "_aflow_source_index": 17,
            }
        )
        + "\n"
    )

    gsm = GSM8KDataset(gsm_path)
    gsm.load()
    math_dataset = MATHDataset(math_path)
    math_dataset.load()

    assert gsm.to_pairs() == [("q", "1,234.50")]
    assert math_dataset.to_pairs() == [
        (
            "p",
            {
                "answer": r"\frac{1}{2}",
                "domain": "Number Theory",
                "level": "Level 5",
                "source_split": "validate",
                "source_index": 17,
            },
        )
    ]
    math_ground_truth = math_dataset.to_pairs()[0][1]
    assert "solution" not in math_ground_truth
    assert MathEvaluator.normalize_ground_truth(
        math_ground_truth
    ) == r"\frac{1}{2}"
    assert MathReward().hard_reward(
        "q",
        "Reasoning. #### 42",
        r"\boxed{42}",
        _trace(),
    ) == 1.0
    assert MathReward().hard_reward(
        "p",
        math_ground_truth,
        r"VERIFIED: \(\frac{1}{2}\)",
        _trace(),
    ) == 1.0


def test_agent_custom_criterion_is_deny_by_default_and_can_be_registered():
    ground_truth = {
        "success_criteria": [
            {"type": "custom", "name": "approved"},
        ]
    }
    assert AgentTaskEvaluator().evaluate("ok", ground_truth) == (
        False,
        0.0,
    )

    evaluator = AgentTaskEvaluator(
        {
            "approved": lambda output, criterion, gt: output == "ok",
        }
    )
    assert evaluator.evaluate("ok", ground_truth) == (True, 1.0)


def test_agent_expected_outcome_fallback_and_empty_process_reward():
    evaluator = AgentTaskEvaluator()
    success, score = evaluator.evaluate(
        "The task is complete: report saved",
        {"expected_outcome": {"status": "complete", "artifact": "report"}},
    )

    assert success is True
    assert score == 1.0
    assert AgentReward(evaluator).process_reward(
        "q",
        {},
        None,
        _trace(),
    ) == 0.0


def test_agent_process_reward_values_valid_grounded_tool_use():
    trace = _trace()
    trace.steps.append(
        TraceStep(
            step_id="tool",
            step_index=0,
            node_id="search",
            node_type="tool",
            action="execute",
            tool_calls=[
                ToolCallRecord(
                    tool_call_id="call",
                    tool_name="search",
                    tool_args={"query": "answer"},
                    tool_result="observation 42",
                    success=True,
                )
            ],
        )
    )
    reward = AgentReward()

    grounded = reward.process_reward(
        "q", {}, "Based on observation 42, the answer is 42.", trace
    )
    ungrounded = reward.process_reward(
        "q", {}, "I guessed another answer.", trace
    )

    assert 0.0 < ungrounded < grounded <= 1.0


def test_math_cli_loader_auto_detects_math_schema(tmp_path):
    from experiments.scripts.run_optimization import _load_benchmark

    path = tmp_path / "math.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem": "Compute one half.",
                "solution": r"\boxed{\frac{1}{2}}",
            }
        )
        + "\n"
    )

    _, data, _ = _load_benchmark("math", str(path))

    assert data == [
        ("Compute one half.", {"answer": r"\frac{1}{2}"})
    ]


def test_bubblewrap_execution_mode_is_manifest_bound():
    from experiments.scripts.run_optimization import _run_metadata
    from experiments.scripts.run_test import _validate_run_metadata

    metadata = _run_metadata(
        "code_gen",
        False,
        use_bubblewrap_code_sandbox=True,
    )
    assert metadata["code_execution_mode"] == "bubblewrap"

    manifest = {"run_metadata": metadata}
    _validate_run_metadata(
        manifest,
        "code_gen",
        False,
        use_bubblewrap_code_sandbox=True,
    )
    with pytest.raises(ValueError, match="execution mode mismatch"):
        _validate_run_metadata(manifest, "code_gen", False)


def test_code_cli_loader_selects_bubblewrap(tmp_path):
    if not BubblewrapCodeRunner.is_available():
        pytest.skip("Linux bubblewrap and a bindable Python are unavailable")
    from experiments.scripts.run_optimization import _load_benchmark

    path = tmp_path / "humaneval.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "def f():",
                "canonical_solution": "\n    return 42",
                "test": "def check(candidate): assert candidate() == 42",
                "entry_point": "f",
            }
        )
        + "\n"
    )

    reward, data, _ = _load_benchmark(
        "code_gen",
        str(path),
        use_bubblewrap_code_sandbox=True,
    )

    assert isinstance(reward.evaluator.runner, BubblewrapCodeRunner)
    assert data[0][1]["task_id"] == "HumanEval/0"
