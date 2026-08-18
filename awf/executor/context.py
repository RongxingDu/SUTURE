"""ExecutionContext — state management during workflow execution."""

from __future__ import annotations

from typing import Any, Optional

from awf.trace.sanitize import to_trace_value


class ExecutionContext:
    """Manages mutable state during a single workflow execution.

    Tracks the current position in the workflow graph, accumulated
    outputs, and execution history.
    """

    def __init__(self, query: str, workflow_name: str = ""):
        self.query = query
        self.workflow_name = workflow_name

        # Current position in the workflow
        self.current_node_id: Optional[str] = None
        self.previous_node_id: Optional[str] = None

        # Execution state
        self.outputs: dict[str, Any] = {}       # node_id -> output
        self.variables: dict[str, Any] = {}     # Shared variables
        self.history: list[str] = []            # Ordered node_ids executed
        self._output_history: list[Any] = []
        self.cost_summary: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
        }

        # Step counter
        self.step_count: int = 0

        # Termination
        self.finished: bool = False
        self.final_output: Optional[Any] = None
        self.success: bool = False
        self.error_message: Optional[str] = None

    def set_current_node(self, node_id: str) -> None:
        """Update the current node pointer."""
        if self.current_node_id:
            self.previous_node_id = self.current_node_id
        self.current_node_id = node_id

    def record_output(self, node_id: str, output: Any) -> None:
        """Record the output of a node execution.

        In addition to the canonical ``outputs`` mapping, expose an
        interpolation-friendly ``{node_id}_output`` variable.  Workflow YAML
        files use these aliases to pass intermediate results to later prompt
        templates.
        """
        self.outputs[node_id] = output
        self.variables[f"{node_id}_output"] = output
        self.history.append(node_id)
        self._output_history.append(output)

    def get_output(self, node_id: str) -> Optional[Any]:
        """Get the output of a previously executed node."""
        return self.outputs.get(node_id)

    def get_last_output(self) -> Optional[Any]:
        """Return the most recently recorded non-``None`` output."""
        for output in reversed(self._output_history):
            if output is not None:
                return output
        return None

    def set_variable(self, key: str, value: Any) -> None:
        """Set a shared variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a shared variable."""
        return self.variables.get(key, default)

    def mark_finished(self, final_output: Any = None,
                      success: bool = True,
                      error_message: Optional[str] = None) -> None:
        """Mark execution as complete."""
        self.finished = True
        self.final_output = final_output
        self.success = success
        self.error_message = error_message

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state for trace recording."""
        return {
            "current_node_id": self.current_node_id,
            "previous_node_id": self.previous_node_id,
            "step_count": self.step_count,
            "history": list(self.history),
            "outputs": to_trace_value(self.outputs),
            "variables": to_trace_value(self.variables),
            "cost_summary": to_trace_value(self.cost_summary),
            "finished": self.finished,
        }

    @classmethod
    def from_snapshot(cls, query: str, snapshot: dict[str, Any],
                      workflow_name: str = "") -> "ExecutionContext":
        """Reconstruct a context from a trace ``state_after`` snapshot."""
        ctx = cls(query=query, workflow_name=workflow_name)
        ctx.outputs = {k: v for k, v in snapshot.get("outputs", {}).items()}
        ctx.variables = {k: v for k, v in snapshot.get("variables", {}).items()}
        ctx.history = list(snapshot.get("history", []))
        # Preserve the observable last-output behavior when a context is
        # resumed.  Suffix replay used to restore only the mapping fields,
        # leaving ``_output_history`` empty and making scheduler gates see a
        # missing artifact unless they happened to fall back to ``outputs``.
        ctx._output_history = [
            ctx.outputs[node_id]
            for node_id in ctx.history
            if node_id in ctx.outputs
        ]
        ctx.current_node_id = snapshot.get("current_node_id")
        ctx.previous_node_id = snapshot.get("previous_node_id")
        ctx.step_count = snapshot.get("step_count", 0)
        ctx.finished = bool(snapshot.get("finished", False))
        cost = snapshot.get("cost_summary", {})
        if isinstance(cost, dict):
            ctx.cost_summary = dict(cost)
        return ctx

    def reset(self, query: str = "") -> None:
        """Reset the context for a new execution."""
        self.query = query or self.query
        self.current_node_id = None
        self.previous_node_id = None
        self.outputs.clear()
        self.variables.clear()
        self.history.clear()
        self._output_history.clear()
        self.cost_summary = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
        }
        self.step_count = 0
        self.finished = False
        self.final_output = None
        self.success = False
        self.error_message = None
