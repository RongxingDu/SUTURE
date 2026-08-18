"""Operator implementations for agent task workflow."""

from __future__ import annotations

from awf.executor.context import ExecutionContext
from awf.executor.safety import counterfactual_safe


@counterfactual_safe
def execute_action(context: ExecutionContext) -> str:
    """Simulate executing an agent action.

    In a real implementation, this would call external APIs, run shell commands,
    or interact with the environment. For the framework, it returns the planned action.
    """
    plan_output = context.get_output("plan")
    if plan_output is None:
        return "No plan available"

    plan_str = str(plan_output)

    # In a real implementation, parse the plan and execute the next action
    # For now, return the first actionable step
    lines = plan_str.strip().split("\n")
    action_lines = [
        line.strip()
        for line in lines
        if line.strip() and any(
            keyword in line.lower()
            for keyword in ["step", "action", "call", "run", "execute", "search", "query", "get", "find"]
        )
    ]

    if action_lines:
        return f"Executed: {action_lines[0]}"
    return f"Action completed based on plan"
