"""WorkflowTemplate = (Graph, ParameterHierarchy)."""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from awf.workflow.nodes import Node, NodeType
from awf.workflow.params import ParameterHierarchy
from awf.workflow.gates import SelectiveUpdateSpec


class WorkflowTemplate(BaseModel):
    """A workflow template combining a directed graph of nodes with a parameter hierarchy.

    The graph defines the control flow structure. The parameter hierarchy
    provides the tunable parameters at each level (stage/block/operator/prompt).
    """

    name: str = "unnamed"
    description: str = ""
    version: str = "1.0"

    # Nodes keyed by node_id
    nodes: dict[str, Node] = Field(default_factory=dict)
    # Edges as (source_node_id, target_node_id) pairs
    edges: list[tuple[str, str]] = Field(default_factory=list)
    # Entry point node id
    entry_node: str = ""
    # Parameter hierarchy
    parameters: ParameterHierarchy = Field(default_factory=ParameterHierarchy)
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional one-layer, query-gated candidate workflow. It is disabled by
    # default and therefore does not change historical workflow behavior.
    selective_update: Optional[SelectiveUpdateSpec] = Field(
        default=None,
        # Preserve byte-level/default artifact shape for workflows that do not
        # opt into S-CWU; old checkpoints therefore do not gain a null field.
        exclude_if=lambda value: value is None,
    )

    def to_networkx(self) -> nx.DiGraph:
        """Convert to a NetworkX directed graph for traversal."""
        g = nx.DiGraph()
        for node_id, node in self.nodes.items():
            g.add_node(node_id, node=node)
        for src, dst in self.edges:
            g.add_edge(src, dst)
        return g

    def get_entry_node(self) -> Node:
        """Get the entry node of the workflow."""
        if self.entry_node and self.entry_node in self.nodes:
            return self.nodes[self.entry_node]
        # Fall back to START node
        for node in self.nodes.values():
            if node.node_type == NodeType.START:
                return node
        raise ValueError("No entry node found in workflow")

    def get_successors(self, node_id: str) -> list[str]:
        """Get successor node ids for a given node."""
        return [dst for src, dst in self.edges if src == node_id]

    def get_node_order(self) -> list[str]:
        """Return topological ordering of node ids."""
        g = self.to_networkx()
        try:
            return list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, return BFS order from entry
            if self.entry_node:
                start = self.entry_node
            else:
                start = next(
                    (nid for nid, n in self.nodes.items() if n.node_type == NodeType.START),
                    list(self.nodes.keys())[0],
                )
            return list(nx.bfs_tree(g, start))

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("edges", mode="before")
    @classmethod
    def _ensure_tuples(cls, v: list) -> list[tuple[str, str]]:
        """Convert list edges to tuples for YAML/JSON compatibility."""
        result = []
        for item in v:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Each edge must contain exactly two node ids: {item!r}")
            result.append((str(item[0]), str(item[1])))
        return result

    @model_validator(mode="after")
    def _validate_graph_structure(self) -> "WorkflowTemplate":
        """Reject malformed, dangling, or unreachable workflow graphs."""
        if not self.nodes:
            raise ValueError("Workflow must contain at least one node")

        for key, node in self.nodes.items():
            if key != node.node_id:
                raise ValueError(
                    f"Node mapping key {key!r} does not match node_id "
                    f"{node.node_id!r}"
                )
            # Node.node_type is authoritative; keep the duplicated config field
            # consistent for serialization and optimizer edits.
            node.config.node_type = node.node_type

        if not self.entry_node:
            starts = [
                node.node_id
                for node in self.nodes.values()
                if node.node_type == NodeType.START
            ]
            if len(starts) != 1:
                raise ValueError("Workflow must define entry_node or one START node")
            self.entry_node = starts[0]
        if self.entry_node not in self.nodes:
            raise ValueError(f"Unknown entry_node: {self.entry_node!r}")

        if len(set(self.edges)) != len(self.edges):
            raise ValueError("Workflow contains duplicate edges")
        for src, dst in self.edges:
            if src not in self.nodes or dst not in self.nodes:
                raise ValueError(f"Dangling workflow edge: ({src!r}, {dst!r})")

        graph = self.to_networkx()
        if graph.in_degree(self.entry_node) != 0:
            raise ValueError(
                f"Workflow entry node {self.entry_node!r} must not have "
                "incoming edges"
            )
        for node_id, node in self.nodes.items():
            out_degree = graph.out_degree(node_id)
            if node.node_type == NodeType.END and out_degree:
                raise ValueError(
                    f"END node {node_id!r} cannot have outgoing edges"
                )
            if node.node_type == NodeType.CONDITION and out_degree != 2:
                raise ValueError(
                    f"CONDITION node {node_id!r} must have exactly two "
                    "ordered successors (true, false)"
                )
        reachable = {self.entry_node, *nx.descendants(graph, self.entry_node)}
        unreachable = set(self.nodes) - reachable
        if unreachable:
            raise ValueError(
                "Workflow contains nodes unreachable from entry: "
                + ", ".join(sorted(unreachable))
            )
        self._validate_parameter_hierarchy()
        if self.selective_update is not None:
            from awf.workflow.gates import validate_selective_update_attachment

            validate_selective_update_attachment(self)
        return self

    def _validate_parameter_hierarchy(self) -> None:
        """Ensure hierarchy references agree with executable node parameters."""
        seen_operators: set[str] = set()
        for stage_key, stage in self.parameters.stages.items():
            if stage_key != stage.stage_id:
                raise ValueError(
                    f"Stage mapping key {stage_key!r} does not match stage_id "
                    f"{stage.stage_id!r}"
                )
            for block_key, block in stage.blocks.items():
                if block_key != block.block_id:
                    raise ValueError(
                        f"Block mapping key {block_key!r} does not match block_id "
                        f"{block.block_id!r}"
                    )
                for operator_key, operator in block.operators.items():
                    if operator_key != operator.node_id:
                        raise ValueError(
                            f"Operator mapping key {operator_key!r} does not "
                            f"match node_id {operator.node_id!r}"
                        )
                    if operator.node_id not in self.nodes:
                        raise ValueError(
                            "Hierarchy references unknown workflow node: "
                            f"{operator.node_id!r}"
                        )
                    if operator.node_id in seen_operators:
                        raise ValueError(
                            "Workflow node appears in more than one hierarchy "
                            f"block: {operator.node_id!r}"
                        )
                    seen_operators.add(operator.node_id)
                    self._validate_operator_parameters(operator)

    def _validate_operator_parameters(self, operator: Any) -> None:
        node = self.nodes[operator.node_id]
        comparisons = {
            "model": (operator.model, node.config.model),
            "temperature": (
                operator.temperature,
                node.config.temperature,
            ),
            "max_tokens": (
                operator.max_tokens,
                node.config.max_tokens,
            ),
            "tool_name": (
                operator.tool_name,
                node.config.tool_name,
            ),
            "system_prompt": (
                operator.prompt.system_prompt,
                node.config.system_prompt,
            ),
            "user_template": (
                operator.prompt.user_template,
                node.config.prompt_template,
            ),
        }
        for parameter, (hierarchy_value, node_value) in comparisons.items():
            if hierarchy_value is not None and hierarchy_value != node_value:
                raise ValueError(
                    f"Hierarchy {parameter} for {operator.node_id!r} does not "
                    "match the executable node config"
                )
        if operator.tool_args and operator.tool_args != node.config.tool_args:
            raise ValueError(
                f"Hierarchy tool_args for {operator.node_id!r} do not match "
                "the executable node config"
            )
