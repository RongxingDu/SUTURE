"""Trace data models: TraceStep, ExecutionTrace, LLMCallRecord, ToolCallRecord."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call within a step."""

    call_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model: str = ""
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    response_text: str = ""
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Raw API response (optional, for debugging)
    raw_response: Optional[dict[str, Any]] = None
    # Latency in seconds
    latency_seconds: float = 0.0
    # Estimated provider price for this call.
    cost_usd: float = 0.0
    cost_estimate_available: bool = True
    # Distinguishes workflow-node calls from scheduler decisions.
    call_type: str = "workflow"
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Whether this call succeeded
    success: bool = True
    error_message: Optional[str] = None


class ToolCallRecord(BaseModel):
    """Record of a tool invocation within a step."""

    tool_call_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Any] = None
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class TraceStep(BaseModel):
    """A single step in an execution trace."""

    step_id: str
    step_index: int
    timestamp: datetime = Field(default_factory=datetime.now)
    node_id: str
    node_type: str = ""
    # Action taken by the scheduler for this step
    action: str = ""
    # State before and after the step
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)
    # LLM call records (if any)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    # Tool call records (if any)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # Step-level reward (if available)
    step_reward: Optional[float] = None
    # Duration of this step in seconds
    duration_seconds: float = 0.0
    # Step-level aggregates across scheduler/workflow LLM calls and tools.
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Whether this step completed successfully
    success: bool = True
    error_message: Optional[str] = None
    # Arbitrary metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Full execution trace for a single query.

    Contains all steps, metadata, and final outcomes.
    """

    trace_id: str
    query_id: str = ""
    query_text: str = ""
    workflow_name: str = ""
    workflow_version: str = ""
    # Ordered list of steps
    steps: list[TraceStep] = Field(default_factory=list)
    # Final outcome
    final_output: Optional[Any] = None
    success: bool = False
    error_message: Optional[str] = None
    # Aggregated rewards
    hard_reward: Optional[float] = None
    process_reward: Optional[float] = None
    # Aggregated costs
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_seconds: float = 0.0
    total_cost_usd: float = 0.0
    total_cost_estimate_complete: bool = True
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    # Execution metadata
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
