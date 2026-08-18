"""Deterministic scheduler that follows the workflow graph exactly."""

from __future__ import annotations

from typing import Any

from awf.config.schema import SchedulerConfig
from awf.executor.context import ExecutionContext
from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.workflow.ir import WorkflowTemplate


class GraphScheduler(BaseScheduler):
    """Delegate all routing to the workflow graph and its CONDITION nodes.

    Unlike :class:`FixedScheduler`, this scheduler does not pre-compute a
    topological traversal or explicitly target nodes.  Returning ``CONTINUE``
    lets ``RuntimeExecutor`` execute the current node and advance along exactly
    one graph edge, including the branch selected by a CONDITION node.

    The policy is stateless and makes no LLM calls.  It is therefore suitable
    for evaluating conditional workflow execution without scheduler overhead.
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig(
            scheduler_type="graph",
            allow_deviation=False,
        )

    async def initialize(
        self,
        workflow: WorkflowTemplate,
        query: str,
    ) -> None:
        """Initialize a query execution; the graph policy has no state."""
        return None

    async def skip_to_node(self, workflow: WorkflowTemplate,
                            node_id: str) -> None:
        """No-op: GraphScheduler has no internal index to adjust."""
        return None

    async def select_action(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> tuple[SchedulerAction, dict[str, Any]]:
        """Execute the current node and follow its selected graph edge."""
        return SchedulerAction.CONTINUE, {}
