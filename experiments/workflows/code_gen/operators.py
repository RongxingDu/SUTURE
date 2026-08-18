"""Operator implementations for code generation workflow.

These are additional operators beyond the default LLM-node execution.
"""

from __future__ import annotations

from awf.executor.context import ExecutionContext
from awf.executor.safety import counterfactual_safe


@counterfactual_safe
def extract_final_code(context: ExecutionContext) -> str:
    """Extract the final generated code from the generate node output."""
    generate_output = context.get_output("generate")
    if generate_output is None:
        return ""

    output_str = str(generate_output)

    # Extract from markdown code blocks
    if "```python" in output_str:
        start = output_str.index("```python") + len("```python")
        end = output_str.index("```", start) if "```" in output_str[start:] else len(output_str)
        return output_str[start:end].strip()
    elif "```" in output_str:
        start = output_str.index("```") + 3
        end = output_str.index("```", start) if "```" in output_str[start:] else len(output_str)
        return output_str[start:end].strip()

    return output_str.strip()
