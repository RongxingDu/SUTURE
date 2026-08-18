"""TraceRecorder — accumulates per-step diagnostics during execution."""

from __future__ import annotations

import uuid
from datetime import datetime
from time import perf_counter
from typing import Any

from awf.trace.schema import (
    ExecutionTrace,
    LLMCallRecord,
    ToolCallRecord,
    TraceStep,
)
from awf.trace.sanitize import to_trace_value


class TraceRecorder:
    """Records execution traces during a workflow run."""

    def __init__(
        self,
        query_id: str = "",
        query_text: str = "",
        enabled: bool = True,
    ):
        self._trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            query_id=query_id,
            query_text=query_text,
        )
        self.enabled = enabled
        self._current_step_index = 0
        self._step_started_at: dict[str, float] = {}
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._llm_calls = 0
        self._tool_calls = 0
        self._cost_usd = 0.0
        self._cost_estimate_complete = True
        self._step_latency_seconds = 0.0

    @property
    def trace(self) -> ExecutionTrace:
        return self._trace

    def start(self, workflow_name: str = "", workflow_version: str = "") -> None:
        """Mark the start of execution."""
        self._trace.workflow_name = workflow_name
        self._trace.workflow_version = workflow_version
        self._trace.start_time = datetime.now()

    def end(self, success: bool = False,
            final_output: object = None,
            hard_reward: float | None = None,
            process_reward: float | None = None,
            error_message: str | None = None) -> ExecutionTrace:
        """Mark the end of execution and finalize the trace."""
        self._trace.end_time = datetime.now()
        self._trace.success = success
        self._trace.error_message = error_message
        self._trace.final_output = to_trace_value(final_output)
        self._trace.hard_reward = hard_reward
        self._trace.process_reward = process_reward
        # Aggregate costs
        self._trace.total_prompt_tokens = self._prompt_tokens
        self._trace.total_completion_tokens = self._completion_tokens
        self._trace.total_latency_seconds = self._step_latency_seconds
        self._trace.total_cost_usd = self._cost_usd
        self._trace.total_cost_estimate_complete = (
            self._cost_estimate_complete
        )
        self._trace.total_llm_calls = self._llm_calls
        self._trace.total_tool_calls = self._tool_calls
        return self._trace

    def record_step_start(self, node_id: str, node_type: str = "",
                          action: str = "", state_before: dict | None = None) -> str:
        """Begin recording a new step. Returns the step_id."""
        step_id = f"step_{self._current_step_index}"
        self._step_started_at[step_id] = perf_counter()
        self._current_step_index += 1
        if not self.enabled:
            return step_id
        step = TraceStep(
            step_id=step_id,
            step_index=self._current_step_index - 1,
            node_id=node_id,
            node_type=node_type,
            action=action,
            state_before=state_before or {},
        )
        self._trace.steps.append(step)
        return step_id

    def update_step(
        self,
        step_id: str,
        *,
        action: str | None = None,
        node_id: str | None = None,
        node_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
    ) -> None:
        """Update information only known after scheduler selection."""
        if not self.enabled:
            return
        for step in self._trace.steps:
            if step.step_id == step_id:
                if action is not None:
                    step.action = action
                if node_id is not None:
                    step.node_id = node_id
                if node_type is not None:
                    step.node_type = node_type
                if state_before is not None:
                    step.state_before = state_before
                if metadata:
                    step.metadata.update(metadata)
                return

    def record_step_end(self, step_id: str, state_after: dict | None = None,
                        success: bool = True, error_message: str | None = None,
                        step_reward: float | None = None) -> None:
        """Finish recording a step with its outcomes."""
        started_at = self._step_started_at.pop(step_id, None)
        duration = perf_counter() - started_at if started_at is not None else 0.0
        self._step_latency_seconds += duration
        if not self.enabled:
            return
        for step in self._trace.steps:
            if step.step_id == step_id:
                step.state_after = state_after or {}
                step.success = success
                step.error_message = error_message
                step.step_reward = step_reward
                # Compute duration from timestamp
                step.duration_seconds = duration
                return

    def record_llm_call(self, step_id: str, call: LLMCallRecord) -> None:
        """Record an LLM API call within a step."""
        self._prompt_tokens += call.prompt_tokens
        self._completion_tokens += call.completion_tokens
        self._llm_calls += 1
        self._cost_usd += call.cost_usd
        self._cost_estimate_complete = (
            self._cost_estimate_complete
            and call.cost_estimate_available
        )
        if not self.enabled:
            return
        for step in self._trace.steps:
            if step.step_id == step_id:
                step.llm_calls.append(call)
                step.input_tokens += call.prompt_tokens
                step.output_tokens += call.completion_tokens
                step.cost_usd += call.cost_usd
                return

    def record_tool_call(self, step_id: str, call: ToolCallRecord) -> None:
        """Record a tool invocation within a step."""
        self._tool_calls += 1
        self._cost_usd += call.cost_usd
        if not self.enabled:
            return
        for step in self._trace.steps:
            if step.step_id == step_id:
                step.tool_calls.append(call)
                step.cost_usd += call.cost_usd
                return

    def get_current_step(self) -> TraceStep | None:
        """Get the most recently started step."""
        if self._trace.steps:
            return self._trace.steps[-1]
        return None

    def seed_cost_summary(self, summary: dict[str, int | float]) -> None:
        """Seed aggregate costs when a trace resumes from a checkpoint.

        Suffix-replay traces contain only newly executed steps, but their
        utility must still be compared with a complete workflow execution.
        Initializing the recorder counters with the cached prefix preserves
        total tokens, calls, latency, and cost without fabricating prefix
        ``TraceStep`` records.
        """
        if not isinstance(summary, dict):
            return

        def _number(key: str, default: int | float = 0) -> int | float:
            value = summary.get(key, default)
            try:
                return float(value) if isinstance(default, float) else int(value)
            except (TypeError, ValueError):
                return default

        self._prompt_tokens = int(_number("prompt_tokens"))
        self._completion_tokens = int(_number("completion_tokens"))
        self._llm_calls = int(_number("llm_calls"))
        self._tool_calls = int(_number("tool_calls"))
        self._cost_usd = float(_number("cost_usd", 0.0))
        self._step_latency_seconds = float(
            _number("latency_seconds", 0.0)
        )
        self._cost_estimate_complete = bool(
            summary.get("cost_estimate_complete", True)
        )

    def get_cost_summary(self) -> dict[str, int | float]:
        """Return live trace-derived cost information for the scheduler."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "llm_calls": self._llm_calls,
            "tool_calls": self._tool_calls,
            "latency_seconds": self._step_latency_seconds,
            "cost_usd": self._cost_usd,
            "cost_estimate_complete": self._cost_estimate_complete,
        }
