"""Correctness-grounding regressions for benchmark process rewards."""

import pytest

from awf.trace.schema import ExecutionTrace, TraceStep
from benchmarks.code_generation.evaluator import CodeEvaluator
from benchmarks.code_generation.reward import CodeGenReward
from benchmarks.math_reasoning.reward import MathReward


def _math_trace(solve_output: str, final_output: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="math-process",
        final_output=final_output,
        steps=[
            TraceStep(
                step_id="solve",
                step_index=0,
                node_id="solve",
                node_type="llm",
                action="execute",
                state_after={
                    "history": ["solve"],
                    "outputs": {"solve": solve_output},
                },
                success=True,
            ),
            TraceStep(
                step_id="finalize",
                step_index=1,
                node_id="finalize",
                node_type="tool",
                action="execute",
                state_after={
                    "history": ["solve", "finalize"],
                    "outputs": {
                        "solve": solve_output,
                        "finalize": final_output,
                    },
                },
                success=True,
            ),
        ],
    )


def _code_trace(code: str, verifier_output: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="code-process",
        final_output=code,
        steps=[
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
                        "verify": verifier_output,
                        "finalize": code,
                    },
                },
                success=True,
            )
        ],
    )


def test_math_polished_wrong_answer_cannot_earn_high_process_reward():
    reward = MathReward()
    solve = (
        r"Step 1: calculate \(40+3=43\). "
        r"Therefore the answer is \boxed{43}."
    )
    final = "VERIFIED: 43"
    trace = _math_trace(solve, final)

    assert reward.hard_reward("compute", "42", final, trace) == 0.0
    process = reward.process_reward("compute", "42", final, trace)

    assert process <= 0.30
    diagnostics = trace.metadata["math_evaluation"]
    assert diagnostics["final_answer_correct"] is False
    assert diagnostics["solve_answer_correct"] is False
    assert diagnostics["reasoning_structure_present"] is True
    assert diagnostics["visible_calculations_present"] is True


def test_math_correct_solve_but_wrong_final_is_capped_and_diagnosed():
    reward = MathReward()
    solve = r"Step 1: calculate \(40+2=42\). Therefore \boxed{42}."
    final = "VERIFIED: 43"
    trace = _math_trace(solve, final)

    process = reward.process_reward("compute", "42", final, trace)

    assert process == 0.35
    diagnostics = trace.metadata["math_evaluation"]
    assert diagnostics["solve_answer_correct"] is True
    assert diagnostics["final_answer_correct"] is False
    assert diagnostics["solve_final_consistent"] is False
    assert diagnostics["incorrect_final_cap_applied"] is True


def test_math_correct_reasoning_and_final_answer_reach_full_process_reward():
    reward = MathReward()
    solve = r"Step 1: calculate \(40+2=42\). Therefore \boxed{42}."
    final = "VERIFIED: 42"
    trace = _math_trace(solve, final)

    assert reward.hard_reward("compute", "42", final, trace) == 1.0
    assert reward.process_reward("compute", "42", final, trace) == 1.0


def test_humaneval_false_positive_verifier_is_low_and_diagnostic():
    evaluator = CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    reward = CodeGenReward(evaluator)
    code = "def f(value):\n    return value"
    ground_truth = {
        "entry_point": "f",
        "test": "def check(candidate): assert candidate(2) == 3",
    }
    trace = _code_trace(code, "PASS")

    assert reward.hard_reward("increment", ground_truth, code, trace) == 0.0
    process = reward.process_reward("increment", ground_truth, code, trace)

    assert process <= 0.35
    diagnostics = trace.metadata["reward_diagnostics"]["code_process"]
    assert diagnostics["verifier_verdict"] == "pass"
    assert diagnostics["verifier_alignment"] == "false_positive"
    assert diagnostics["test_pass_rate"] == 0.0
    assert diagnostics["test_outcome_counts"] == {
        "assertion_failure": 1
    }


def test_humaneval_correct_tests_and_verifier_reach_full_process_reward():
    evaluator = CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    reward = CodeGenReward(evaluator)
    code = "def f(value):\n    return value + 1"
    ground_truth = {
        "entry_point": "f",
        "test": "def check(candidate): assert candidate(2) == 3",
    }
    trace = _code_trace(code, "PASS")

    assert reward.hard_reward("increment", ground_truth, code, trace) == 1.0
    assert reward.process_reward("increment", ground_truth, code, trace) == 1.0
    diagnostics = trace.metadata["reward_diagnostics"]["code_process"]
    assert diagnostics["verifier_alignment"] == "correct_acceptance"


def test_humaneval_verifier_false_negative_is_distinguished():
    evaluator = CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    reward = CodeGenReward(evaluator)
    code = "def f(value):\n    return value + 1"
    ground_truth = {
        "entry_point": "f",
        "test": "def check(candidate): assert candidate(2) == 3",
    }
    trace = _code_trace(code, "FAIL: incorrectly suspected an edge case")

    assert reward.hard_reward("increment", ground_truth, code, trace) == 1.0
    process = reward.process_reward("increment", ground_truth, code, trace)

    assert process == 0.85
    diagnostics = trace.metadata["reward_diagnostics"]["code_process"]
    assert diagnostics["verifier_alignment"] == "false_negative"


def test_code_process_reward_never_executes_hidden_tests_itself():
    class FailIfCalledRunner:
        def __init__(self):
            self.calls = 0

        def run(self, script, timeout_seconds):
            self.calls += 1
            raise AssertionError("process_reward must not execute tests")

    runner = FailIfCalledRunner()
    reward = CodeGenReward(CodeEvaluator(sandbox_runner=runner))
    code = "def f(value):\n    return value + 1"
    ground_truth = {
        "entry_point": "f",
        "test": "def check(candidate): assert candidate(2) == 3",
    }
    trace = _code_trace(code, "PASS")

    process = reward.process_reward("increment", ground_truth, code, trace)

    assert runner.calls == 0
    assert process == pytest.approx(0.15)
    diagnostics = trace.metadata["reward_diagnostics"]["code_process"]
    assert diagnostics["evaluation_available"] is False
    assert diagnostics["verifier_alignment"] == "tests_unavailable"


def test_humaneval_parseable_but_uncompilable_code_never_reaches_tests():
    evaluator = CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    reward = CodeGenReward(evaluator)
    code = "def f():\n    break"
    ground_truth = {
        "entry_point": "f",
        "test": "def check(candidate): assert candidate() == 1",
    }
    trace = _code_trace(code, "PASS")

    assert reward.hard_reward("return one", ground_truth, code, trace) == 0.0
    assert reward.process_reward("return one", ground_truth, code, trace) == 0.0
    contract = trace.metadata["code_output_contract"]
    assert contract["parseable"] is True
    assert contract["compiled"] is False
    assert contract["compile_error_type"] == "SyntaxError"
    assert "code_evaluation" not in trace.metadata


def test_code_evaluator_diagnostics_never_echo_hidden_test_source():
    secret = "HIDDEN_ASSERTION_SENTINEL_MUST_NOT_LEAK"
    hidden_test = (
        "def check(candidate):\n"
        f"    assert candidate() == 42, {secret!r}"
    )
    evaluator = CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    result = evaluator.evaluate_detailed(
        "def f():\n    return 0",
        hidden_test,
        "f",
    )

    summary = result.summary()
    assert result.outcome_counts == {"assertion_failure": 1}
    assert "hidden assertion failed" in summary
    assert secret not in summary
    assert hidden_test not in summary
    assert "candidate() == 42" not in summary
