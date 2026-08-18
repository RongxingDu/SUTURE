"""BaseScheduler ABC + SchedulerAction enum."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from awf.executor.context import ExecutionContext
    from awf.workflow.ir import WorkflowTemplate


class SchedulerAction(str, Enum):
    """Actions the scheduler can select at each step."""

    # Canonical runtime actions from the SUTURE specification.
    CONTINUE = "continue"
    EARLY_EXIT = "early_exit"
    VERIFY = "verify"
    REPAIR = "repair"
    REROUTE = "reroute"
    FALLBACK = "fallback"

    # Legacy actions remain accepted so existing schedulers and serialized
    # traces continue to work.
    EXECUTE = "execute"      # Execute the current node
    SKIP = "skip"            # Skip the current node
    RETRY = "retry"          # Retry the previous node
    STOP = "stop"            # Stop execution early
    BRANCH = "branch"        # Take a conditional branch
    DEVIATE = "deviate"      # Deviate from the template (custom action)

    @classmethod
    def coerce(cls, action: "SchedulerAction | str") -> "SchedulerAction":
        """Normalize scheduler output while accepting legacy strings."""
        if isinstance(action, cls):
            return action
        try:
            return cls(str(action).strip().lower())
        except ValueError:
            return cls.CONTINUE


class BaseScheduler(ABC):
    """Abstract base class for workflow schedulers.

    A scheduler determines which action to take at each step of
    workflow execution, given the current context and workflow template.
    """

    @abstractmethod
    async def select_action(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> tuple[SchedulerAction, dict[str, Any]]:
        """Select the next action based on the current execution context.

        Args:
            workflow: The workflow template being executed.
            context: The current execution context.

        Returns:
            Tuple of (action, action_params).
            action_params is a dict of additional parameters for the action
            (e.g., target_node for BRANCH, custom_prompt for DEVIATE).
        """
        ...

    @abstractmethod
    async def initialize(self, workflow: WorkflowTemplate,
                         query: str) -> None:
        """Initialize the scheduler for a new query execution.

        Args:
            workflow: The workflow template.
            query: The input query text.
        """
        ...

    async def skip_to_node(self, workflow: "WorkflowTemplate",
                            node_id: str) -> None:
        """Advance internal state past all nodes before ``node_id`` in
        execution order.
        """
        ...

    def pop_last_llm_call(self) -> Any:
        """Return and clear diagnostics for the latest scheduler API call.

        Rule-based schedulers do not make an LLM call, so the default is
        ``None``.  LLM schedulers override this hook; the executor uses it to
        attach scheduler token, latency, and cost information to the trace.
        """
        return None

    def telemetry(self) -> dict[str, Any]:
        """Return optional scheduler diagnostics for the execution trace."""
        return {}
