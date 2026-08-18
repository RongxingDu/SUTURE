"""Code-generation hard and process rewards."""

from __future__ import annotations

import ast
import re
from typing import Any, Sequence

from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from benchmarks.code_generation.evaluator import CodeEvaluator


class CodeGenReward(RewardEvaluator):
    """Reward evaluator for HumanEval- and MBPP-style tasks.

    Hard reward is the test pass rate.  A HumanEval ``test`` string is one
    suite; an MBPP ``test_list`` is evaluated item-by-item for partial credit.
    """

    def __init__(self, evaluator: CodeEvaluator | None = None):
        self.evaluator = evaluator or CodeEvaluator()

    def hard_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        tests, entry_point = _extract_tests(ground_truth)
        if not tests:
            return 0.0
        code, _ = _validate_final_code(
            self.evaluator,
            output,
            entry_point,
            trace,
        )
        if code is None:
            return 0.0

        fingerprint = _evaluation_fingerprint(code, tests, entry_point)
        cached = trace.metadata.get("code_evaluation")
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return float(cached.get("pass_rate", 0.0))

        result = self.evaluator.evaluate_detailed(code, tests, entry_point)
        trace.metadata["code_evaluation"] = {
            "fingerprint": fingerprint,
            "passed_tests": result.passed_tests,
            "total_tests": result.total_tests,
            "tests_executed": result.tests_executed,
            "execution_rate": result.execution_rate,
            "pass_rate": result.pass_rate,
            "all_passed": result.all_passed,
            "outcome_counts": result.outcome_counts,
            "summary": result.summary(),
        }
        return result.pass_rate

    def process_reward(
        self,
        query: str,
        ground_truth: Any,
        output: Any,
        trace: ExecutionTrace,
    ) -> float:
        """Score compile, semantic, verifier, and aggregate test evidence.

        The test suite is never executed here; this method consumes only the
        fingerprint-matched result cached by :meth:`hard_reward`. Test pass
        rate carries most of the score, while a zero-pass implementation is
        capped at ``0.35`` regardless of formatting or verifier confidence.
        """
        tests, entry_point = _extract_tests(ground_truth)
        code, tree = _validate_final_code(
            self.evaluator,
            output,
            entry_point,
            trace,
        )
        if code is None or tree is None:
            _record_code_process_diagnostics(
                trace,
                {
                    "output_contract_valid": False,
                    "process_reward": 0.0,
                    "reason": "invalid_final_code_contract",
                },
            )
            return 0.0

        contract = trace.metadata.get("code_output_contract")
        contract = contract if isinstance(contract, dict) else {}
        meaningful_steps = [
            step
            for step in trace.steps
            if step.node_type not in {"start", "end"}
            and step.action != "stop"
        ]
        execution_clean = bool(
            meaningful_steps
            and all(step.success for step in meaningful_steps)
        )

        evaluation = trace.metadata.get("code_evaluation")
        fingerprint = _evaluation_fingerprint(code, tests, entry_point)
        evaluation_available = bool(
            isinstance(evaluation, dict)
            and evaluation.get("fingerprint") == fingerprint
        )
        if evaluation_available:
            assert isinstance(evaluation, dict)
            total = _safe_nonnegative_int(evaluation.get("total_tests"))
            executed = _safe_nonnegative_int(
                evaluation.get("tests_executed")
            )
            pass_rate = _bounded_rate(evaluation.get("pass_rate"))
            execution_rate = (
                min(executed, total) / total if total > 0 else 0.0
            )
            all_passed = bool(
                total > 0
                and executed == total
                and pass_rate >= 1.0
            )
            outcome_counts = evaluation.get("outcome_counts", {})
            if not isinstance(outcome_counts, dict):
                outcome_counts = {}
        else:
            total = 0
            executed = 0
            pass_rate = 0.0
            execution_rate = 0.0
            all_passed = False
            outcome_counts = {}

        verifier_verdict = _classify_verifier_output(
            _trace_output(trace, "verify")
        )
        verifier_alignment = _verifier_alignment(
            verifier_verdict,
            evaluation_available=evaluation_available,
            all_passed=all_passed,
        )
        verifier_aligned = verifier_alignment in {
            "correct_acceptance",
            "correct_rejection",
        }

        components = {
            "compiled_contract": (
                0.05 if contract.get("compiled") is True else 0.0
            ),
            "meaningful_implementation": (
                0.05
                if contract.get("implementation_meaningful") is True
                else 0.0
            ),
            "clean_workflow_execution": (
                0.05 if execution_clean else 0.0
            ),
            "test_execution_coverage": 0.10 * execution_rate,
            "test_pass_rate": 0.60 * pass_rate,
            "verifier_test_alignment": (
                0.15 if verifier_aligned else 0.0
            ),
        }
        raw_score = sum(components.values())
        zero_pass_cap_applied = pass_rate <= 0.0 and raw_score > 0.35
        score = min(raw_score, 0.35) if pass_rate <= 0.0 else raw_score
        score = max(0.0, min(1.0, score))

        diagnostics = {
            "output_contract_valid": True,
            "compiled": contract.get("compiled") is True,
            "implementation_meaningful": (
                contract.get("implementation_meaningful") is True
            ),
            "implementation_is_placeholder": contract.get(
                "implementation_is_placeholder"
            ),
            "evaluation_available": evaluation_available,
            "tests_total": total,
            "tests_executed": executed,
            "test_execution_rate": execution_rate,
            "test_pass_rate": pass_rate,
            "test_outcome_counts": dict(sorted(outcome_counts.items())),
            "verifier_verdict": verifier_verdict,
            "verifier_alignment": verifier_alignment,
            "execution_clean": execution_clean,
            "component_scores": components,
            "raw_process_score": raw_score,
            "zero_pass_cap_applied": zero_pass_cap_applied,
            "process_reward": score,
        }
        _record_code_process_diagnostics(trace, diagnostics)
        return score


def _extract_tests(
    ground_truth: Any,
) -> tuple[str | Sequence[str], str | None]:
    if isinstance(ground_truth, str):
        return ground_truth, None
    if not isinstance(ground_truth, dict):
        return "", None

    if ground_truth.get("test_list"):
        tests: str | Sequence[str] = ground_truth["test_list"]
    elif ground_truth.get("tests"):
        tests = ground_truth["tests"]
    else:
        tests = ground_truth.get("test", "")
    return tests, ground_truth.get("entry_point") or None


def _validate_final_code(
    evaluator: CodeEvaluator,
    output: Any,
    entry_point: str | None,
    trace: ExecutionTrace,
) -> tuple[str | None, ast.Module | None]:
    """Validate only the workflow's actual final output.

    Intermediate generation nodes are intentionally never a fallback.  A
    traced workflow must complete through its code-finalizer or through an
    explicit early exit whose output itself satisfies the code contract.
    """
    output_node_id = _infer_output_node_id(trace)
    completion_mode = _completion_mode(trace, output_node_id)
    diagnostic: dict[str, Any] = {
        "source": "final_output",
        "output_node_id": output_node_id,
        "completion_mode": completion_mode,
        "entry_point": entry_point,
        "parseable": False,
        "compiled": False,
        "entry_point_present": False,
        "implementation_meaningful": False,
        "implementation_is_placeholder": None,
        "valid": False,
    }

    def reject(reason: str) -> tuple[None, None]:
        diagnostic["reason"] = reason
        trace.metadata["code_output_contract"] = diagnostic
        return None, None

    if output is None:
        return reject("missing final output")

    code = evaluator.extract_code(str(output))
    if not code.strip():
        return reject("empty final code")

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return reject("final output is not parseable Python")
    diagnostic["parseable"] = True
    try:
        compile(tree, "<generated>", "exec")
    except (SyntaxError, TypeError, ValueError) as exc:
        diagnostic["compile_error_type"] = type(exc).__name__
        return reject("final output is parseable but does not compile")
    diagnostic["compiled"] = True

    top_level_callables = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if entry_point:
        entry_point_present = entry_point in top_level_callables
    else:
        entry_point_present = bool(top_level_callables)
    diagnostic["entry_point_present"] = entry_point_present
    if not entry_point_present:
        if entry_point:
            return reject(
                f"final code does not define entry point {entry_point!r}"
            )
        return reject("final code does not define a top-level callable")

    diagnostic.update(_implementation_diagnostics(tree, entry_point))

    if completion_mode == "invalid":
        return reject(
            "workflow did not complete through finalize or a valid early exit"
        )

    diagnostic["valid"] = True
    diagnostic["reason"] = "valid final code"
    trace.metadata["code_output_contract"] = diagnostic
    return code, tree


def _implementation_diagnostics(
    tree: ast.Module,
    entry_point: str | None,
) -> dict[str, Any]:
    """Return bounded AST semantics without retaining generated source text."""
    definitions = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]
    targets = (
        [node for node in definitions if node.name == entry_point]
        if entry_point
        else definitions
    )
    placeholder = bool(targets) and all(
        _definition_is_placeholder(node) for node in targets
    )
    nodes = [
        child
        for target in targets
        for child in ast.walk(target)
    ]
    meaningful = bool(targets) and not placeholder
    return {
        "implementation_meaningful": meaningful,
        "implementation_is_placeholder": placeholder,
        "implementation_ast_nodes": min(len(nodes), 10_000),
        "implementation_has_return_or_yield": any(
            isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom))
            for child in nodes
        ),
        "implementation_has_control_flow": any(
            isinstance(
                child,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.Match,
                    ast.comprehension,
                ),
            )
            for child in nodes
        ),
    }


def _definition_is_placeholder(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> bool:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return True
    return all(_statement_is_placeholder(statement) for statement in body)


def _statement_is_placeholder(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    ):
        return True
    if isinstance(statement, ast.Raise):
        exception = statement.exc
        if isinstance(exception, ast.Name):
            return exception.id == "NotImplementedError"
        if (
            isinstance(exception, ast.Call)
            and isinstance(exception.func, ast.Name)
        ):
            return exception.func.id == "NotImplementedError"
    return False


def _trace_output(trace: ExecutionTrace, node_id: str) -> Any:
    for step in reversed(trace.steps):
        for state in (step.state_after, step.state_before):
            outputs = state.get("outputs") if isinstance(state, dict) else None
            if isinstance(outputs, dict) and outputs.get(node_id) is not None:
                return outputs[node_id]
    return None


def _classify_verifier_output(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    if re.fullmatch(r"PASS[.!]?", text, re.IGNORECASE):
        return "pass"
    if re.match(r"^(?:FAIL|ERROR|REJECT)\b", text, re.IGNORECASE):
        return "fail"
    return "unknown"


def _verifier_alignment(
    verdict: str,
    *,
    evaluation_available: bool,
    all_passed: bool,
) -> str:
    if not evaluation_available:
        return "tests_unavailable"
    if verdict == "pass":
        return "correct_acceptance" if all_passed else "false_positive"
    if verdict == "fail":
        return "false_negative" if all_passed else "correct_rejection"
    return "verdict_unavailable"


def _record_code_process_diagnostics(
    trace: ExecutionTrace,
    diagnostics: dict[str, Any],
) -> None:
    existing = trace.metadata.get("reward_diagnostics")
    reward_diagnostics = dict(existing) if isinstance(existing, dict) else {}
    reward_diagnostics["code_process"] = diagnostics
    trace.metadata["reward_diagnostics"] = reward_diagnostics


def _safe_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _bounded_rate(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, rate))


def _evaluation_fingerprint(
    code: str,
    tests: str | Sequence[str],
    entry_point: str | None,
) -> str:
    return repr((code, tests, entry_point))


def _infer_output_node_id(trace: ExecutionTrace) -> str | None:
    """Infer the node which produced the last non-None workflow output."""
    declared = trace.metadata.get("final_output_node_id")
    if isinstance(declared, str) and declared:
        return declared

    for step in reversed(trace.steps):
        for state in (step.state_after, step.state_before):
            if not isinstance(state, dict):
                continue
            outputs = state.get("outputs")
            history = state.get("history")
            if not isinstance(outputs, dict) or not isinstance(history, list):
                continue
            for node_id in reversed(history):
                if (
                    isinstance(node_id, str)
                    and outputs.get(node_id) is not None
                ):
                    return node_id
    return None


def _completion_mode(
    trace: ExecutionTrace,
    output_node_id: str | None,
) -> str:
    """Classify whether the workflow respected its final-output boundary."""
    if not trace.steps:
        return "direct"
    if _is_code_finalizer(trace, output_node_id):
        return "finalize"
    if trace.steps[-1].action.strip().lower() == "early_exit":
        return "early_exit"
    return "invalid"


def _is_code_finalizer(
    trace: ExecutionTrace,
    output_node_id: str | None,
) -> bool:
    if not output_node_id:
        return False
    declared = trace.metadata.get("code_final_output_node_id")
    for step in reversed(trace.steps):
        if (
            step.node_id == output_node_id
            and step.metadata.get("node_executed", True)
            and (
                (
                    output_node_id == "finalize"
                    and step.node_type == "tool"
                )
                or declared == output_node_id
                or any(
                    call.tool_name == "extract_final_code"
                    for call in step.tool_calls
                )
            )
        ):
            return True
    return False
