"""FixedScheduler — baseline scheduler that always follows the full template."""

from __future__ import annotations

from typing import Any

from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.workflow.ir import WorkflowTemplate
from awf.executor.context import ExecutionContext
from awf.workflow.nodes import NodeType


class FixedScheduler(BaseScheduler):
    """Baseline scheduler that always executes every node in order.

    This is a deterministic scheduler that follows the workflow template
    exactly without any deviation. Useful as a baseline comparison.
    """

    def __init__(self):
        self._node_order: list[str] = []
        self._index: int = 0

    async def initialize(self, workflow: WorkflowTemplate,
                         query: str) -> None:
        """Pre-compute a deterministic full-workflow traversal.

        A plain "take the first successor" walk drops sibling branches and can
        loop forever.  The fixed baseline instead visits every reachable
        executable node once in topological/BFS order, placing a single END
        node last.
        """
        entry_id = workflow.get_entry_node().node_id
        reachable: set[str] = set()
        queue = [entry_id]
        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue
            reachable.add(node_id)
            queue.extend(workflow.get_successors(node_id))

        ordered = [
            node_id
            for node_id in workflow.get_node_order()
            if node_id in reachable
        ]
        end_nodes = [
            node_id
            for node_id in ordered
            if workflow.nodes[node_id].node_type == NodeType.END
        ]
        self._node_order = [
            node_id
            for node_id in ordered
            if workflow.nodes[node_id].node_type != NodeType.END
        ]
        if end_nodes:
            self._node_order.append(end_nodes[0])
        self._index = 0

    async def skip_to_node(self, workflow: WorkflowTemplate,
                            node_id: str) -> None:
        """Advance internal index to ``node_id`` in ``_node_order``."""
        try:
            self._index = self._node_order.index(node_id)
        except ValueError:
            self._index = len(self._node_order)

    async def select_action(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> tuple[SchedulerAction, dict[str, Any]]:
        """Always select EXECUTE for the next node in order.

        Returns STOP when all nodes have been executed.
        """
        if self._index >= len(self._node_order):
            return SchedulerAction.STOP, {}

        target_node = self._node_order[self._index]
        self._index += 1
        # EXECUTE is retained as a backwards-compatible synonym of CONTINUE.
        # The explicit target is what makes branch/cycle traversal complete.
        return SchedulerAction.EXECUTE, {"next_unit_id": target_node}
