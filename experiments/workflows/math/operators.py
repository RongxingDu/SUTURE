"""Operator implementations for math reasoning workflow."""

from __future__ import annotations

import re

from awf.executor.context import ExecutionContext
from awf.executor.safety import counterfactual_safe
from benchmarks.math_reasoning.evaluator import MathEvaluator


@counterfactual_safe
def extract_boxed_answer(context: ExecutionContext) -> str:
    """Extract the boxed answer from the solve node output."""
    solve_output = context.get_output("solve")
    if solve_output is None:
        return ""

    output_str = str(solve_output)
    match = re.search(r'\\boxed\{([^}]+)\}', output_str)
    if match:
        return match.group(1).strip()
    return ""


@counterfactual_safe
def check_verification(context: ExecutionContext) -> str:
    """Check the verification result from the verify node."""
    verify_output = context.get_output("verify")
    if verify_output is None:
        return "ERROR: No verification output"

    output_str = str(verify_output)
    if "VERIFIED" in output_str.upper():
        return "PASS"
    return "FAIL"


@counterfactual_safe
def extract_final_answer(context: ExecutionContext) -> str:
    """Return the verified response, falling back to the original solution."""
    # Deterministic research blocks emit a fully adjudicated solution. Prefer
    # the latest such artifact over a verifier that still reads the incumbent
    # ``solve`` draft; this makes graph insertion semantically effective
    # without special-casing the runtime or mutating historical node outputs.
    for node_id in (
        "conditional_debate",
        "verify_repair",
        "format_repair",
        "dual_solve_judge",
        "self_refine",
    ):
        refined_output = context.get_output(node_id)
        if refined_output is not None and MathEvaluator.extract_answer(
            str(refined_output)
        ):
            return str(refined_output)
    verify_output = context.get_output("verify")
    if verify_output is not None:
        verify_text = str(verify_output)
        # A verifier following the positive contract can still emit a
        # payload the answer parser cannot recognize.  Preserve explicit
        # ERROR responses, but recover the solve result for an unparseable
        # VERIFIED/PASSED response so a formatting failure cannot turn a
        # correct boxed solve answer into a hard failure.
        positive_marker = re.search(
            r"^\s*(?:verified|passed?)\s*:",
            verify_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if positive_marker and not MathEvaluator.extract_answer(verify_text):
            solve_output = context.get_output("solve")
            if solve_output is not None:
                return str(solve_output)
        return verify_text
    solve_output = context.get_output("solve")
    return "" if solve_output is None else str(solve_output)
