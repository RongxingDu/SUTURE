"""Side-effect-free output normalization for SciCode."""

from __future__ import annotations

from awf.executor.context import ExecutionContext
from awf.executor.safety import counterfactual_safe
from benchmarks.scicode.evaluator import SciCodeEvaluator


@counterfactual_safe
def extract_scicode_code(context: ExecutionContext) -> str:
    """Extract code text without compiling or executing model output."""
    return SciCodeEvaluator.extract_code(context.get_output("generate"))
