"""Tests for workflow IR, nodes, parameters, and serialization."""

import os
import tempfile

import pytest

from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.params import ParameterHierarchy, OperatorParams, PromptParams
from awf.workflow.serializer import dump_workflow, load_workflow


class TestNodes:
    """Test Node and NodeConfig models."""

    def test_create_llm_node(self):
        config = NodeConfig(
            node_type=NodeType.LLM,
            prompt_template="Solve: {query}",
            system_prompt="You are helpful.",
            model="gpt-4o",
            temperature=0.0,
        )
        node = Node(node_id="llm_1", node_type=NodeType.LLM, config=config,
                    label="LLM Node")
        assert node.node_id == "llm_1"
        assert node.config.prompt_template == "Solve: {query}"
        assert node.config.temperature == 0.0

    def test_create_tool_node(self):
        config = NodeConfig(
            node_type=NodeType.TOOL,
            tool_name="calculator",
            tool_args={"expression": "2+2"},
        )
        node = Node(node_id="tool_1", node_type=NodeType.TOOL, config=config)
        assert node.config.tool_name == "calculator"

    def test_create_condition_node(self):
        config = NodeConfig(
            node_type=NodeType.CONDITION,
            condition_expr="outputs['verify'] == 'PASS'",
        )
        node = Node(node_id="cond_1", node_type=NodeType.CONDITION, config=config)
        assert node.config.condition_expr is not None


class TestParameterHierarchy:
    """Test parameter hierarchy model."""

    def test_set_and_get_operator(self):
        hierarchy = ParameterHierarchy()
        op = OperatorParams(
            node_id="llm_1",
            model="gpt-4o",
            temperature=0.3,
            prompt=PromptParams(
                system_prompt="You are helpful.",
                user_template="Solve: {query}",
            ),
        )
        hierarchy.set_operator("stage_1", "block_1", op)
        retrieved = hierarchy.get_operator("stage_1", "block_1", "llm_1")
        assert retrieved is not None
        assert retrieved.model == "gpt-4o"
        assert retrieved.temperature == 0.3

    def test_get_nonexistent_operator(self):
        hierarchy = ParameterHierarchy()
        result = hierarchy.get_operator("nonexistent", "block", "node")
        assert result is None


class TestWorkflowTemplate:
    """Test WorkflowTemplate model."""

    def _make_simple_workflow(self) -> WorkflowTemplate:
        """Helper to create a simple 3-node workflow."""
        return WorkflowTemplate(
            name="test_workflow",
            version="1.0",
            entry_node="start",
            nodes={
                "start": Node(
                    node_id="start", node_type=NodeType.START, label="Start"
                ),
                "llm_1": Node(
                    node_id="llm_1",
                    node_type=NodeType.LLM,
                    config=NodeConfig(
                        node_type=NodeType.LLM,
                        prompt_template="Q: {query}",
                        system_prompt="Answer briefly.",
                    ),
                    label="LLM Step",
                ),
                "end": Node(
                    node_id="end", node_type=NodeType.END, label="End"
                ),
            },
            edges=[("start", "llm_1"), ("llm_1", "end")],
        )

    def test_get_entry_node(self):
        wf = self._make_simple_workflow()
        entry = wf.get_entry_node()
        assert entry.node_id == "start"
        assert entry.node_type == NodeType.START

    def test_get_successors(self):
        wf = self._make_simple_workflow()
        successors = wf.get_successors("start")
        assert successors == ["llm_1"]
        successors = wf.get_successors("llm_1")
        assert successors == ["end"]
        successors = wf.get_successors("end")
        assert successors == []

    def test_get_node_order(self):
        wf = self._make_simple_workflow()
        order = wf.get_node_order()
        assert order == ["start", "llm_1", "end"]

    def test_to_networkx(self):
        wf = self._make_simple_workflow()
        g = wf.to_networkx()
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2

    def test_edge_list_to_tuple_conversion(self):
        """Test that list-style edges from YAML are converted to tuples."""
        wf = WorkflowTemplate(
            name="test",
            entry_node="a",
            nodes={
                "a": Node(node_id="a", node_type=NodeType.START),
                "b": Node(node_id="b", node_type=NodeType.END),
            },
            edges=[["a", "b"]],  # List form (as from YAML)
        )
        assert wf.edges[0] == ("a", "b")
        assert isinstance(wf.edges[0], tuple)


class TestWorkflowSerialization:
    """Test workflow YAML/JSON serialization round-trips."""

    def _make_workflow(self) -> WorkflowTemplate:
        return WorkflowTemplate(
            name="serial_test",
            version="1.0",
            entry_node="start",
            nodes={
                "start": Node(node_id="start", node_type=NodeType.START),
                "process": Node(
                    node_id="process",
                    node_type=NodeType.LLM,
                    config=NodeConfig(
                        node_type=NodeType.LLM,
                        prompt_template="Do: {query}",
                        temperature=0.2,
                    ),
                ),
                "end": Node(node_id="end", node_type=NodeType.END),
            },
            edges=[("start", "process"), ("process", "end")],
            parameters=ParameterHierarchy(
                global_params={"max_retries": 3},
            ),
        )

    def test_yaml_round_trip(self):
        wf = self._make_workflow()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            tmp_path = f.name

        try:
            dump_workflow(wf, tmp_path)
            loaded = load_workflow(tmp_path)
            assert loaded.name == wf.name
            assert loaded.version == wf.version
            assert loaded.entry_node == wf.entry_node
            assert len(loaded.nodes) == len(wf.nodes)
            assert len(loaded.edges) == len(wf.edges)
        finally:
            os.unlink(tmp_path)

    def test_json_round_trip(self):
        wf = self._make_workflow()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            dump_workflow(wf, tmp_path)
            loaded = load_workflow(tmp_path)
            assert loaded.name == wf.name
            assert loaded.entry_node == wf.entry_node
        finally:
            os.unlink(tmp_path)
