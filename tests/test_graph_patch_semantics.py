"""Regression tests for graph-patch semantic safety.

These tests describe the post-fix contract.  A graph patch is an atomic
transaction over both control-flow and data dependencies, and ordered
CONDITION successors are part of the workflow's meaning.
"""

import json

import pytest

from awf.config.schema import ExecutorConfig
from awf.executor.runtime import RuntimeExecutor
from awf.optimizer.candidate_generator import CandidateGenerator
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.params import OperatorParams, PromptParams


def _llm_node(node_id: str, prompt: str = "{query}") -> Node:
    return Node(
        node_id=node_id,
        node_type=NodeType.LLM,
        config=NodeConfig(
            node_type=NodeType.LLM,
            prompt_template=prompt,
        ),
    )


def _chain_workflow(*node_ids: str) -> WorkflowTemplate:
    nodes = {
        "start": Node(node_id="start", node_type=NodeType.START),
        **{node_id: _llm_node(node_id) for node_id in node_ids},
        "end": Node(node_id="end", node_type=NodeType.END),
    }
    path = ["start", *node_ids, "end"]
    return WorkflowTemplate(
        name="graph-patch-semantics",
        entry_node="start",
        nodes=nodes,
        edges=list(zip(path, path[1:])),
    )


def _declare_block(
    workflow: WorkflowTemplate,
    *node_ids: str,
) -> None:
    for node_id in node_ids:
        node = workflow.nodes[node_id]
        workflow.parameters.set_operator(
            "stage",
            "local",
            OperatorParams(
                node_id=node_id,
                prompt=PromptParams(
                    system_prompt=node.config.system_prompt,
                    user_template=node.config.prompt_template,
                ),
            ),
        )


def _parse_patch(
    workflow: WorkflowTemplate,
    anchor_id: str,
    patch: dict,
):
    generator = CandidateGenerator(
        object(),
        max_candidates=2,
        max_edit_distance=1.0,
    )
    response = json.dumps(
        {
            "candidates": [
                {
                    "scope": "block",
                    "node_id": anchor_id,
                    "description": "graph-patch semantic regression",
                    "changes": {"graph_patch": patch},
                }
            ]
        }
    )
    return generator._parse_candidates(
        response,
        workflow,
        allowed_anchor_ids={anchor_id},
        requested_scope="block",
        limit=1,
    )


def test_graph_path_scope_alias_maps_to_internal_block_scope() -> None:
    generator = CandidateGenerator(
        object(),
        max_candidates=1,
        allowed_scopes=["graph_path"],
    )
    assert generator.allowed_scopes == {"block"}


def test_graph_patch_rejects_condition_dependency_on_downstream_output():
    workflow = _chain_workflow("a", "b")
    patch = {
        "add_nodes": [
            {
                "node_id": "gate",
                "node_type": "condition",
                "config": {
                    "condition_expr": 'outputs["end"] is not None',
                },
            },
            {
                "node_id": "alternate",
                "node_type": "llm",
                "config": {"prompt_template": "{b_output}"},
            },
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [
            ["b", "gate"],
            ["gate", "end"],
            ["gate", "alternate"],
            ["alternate", "end"],
        ],
    }

    assert _parse_patch(workflow, "b", patch) == []


@pytest.mark.parametrize(
    "patch",
    [
        pytest.param(
            {
                "add_nodes": [
                    {
                        "node_id": "gate",
                        "node_type": "condition",
                        "config": {"condition_expr": "False"},
                    }
                ],
                "remove_edges": [["a", "b"]],
                "add_edges": [
                    ["a", "gate"],
                    ["gate", "b"],
                    ["gate", "c"],
                ],
            },
            id="branch-bypasses-required-producer",
        ),
        pytest.param(
            {
                "remove_nodes": ["b"],
                "add_edges": [["a", "c"]],
            },
            id="required-producer-is-removed",
        ),
    ],
)
def test_graph_patch_rejects_breaking_existing_data_dependency(patch):
    workflow = _chain_workflow("a", "b", "c")
    workflow.nodes["c"].config.prompt_template = "{b_output}"
    _declare_block(workflow, "a", "b", "c")

    assert _parse_patch(workflow, "a", patch) == []


def test_graph_patch_rejects_fanout_from_non_condition_node():
    workflow = _chain_workflow("a", "b")
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {"prompt_template": "{b_output}"},
            }
        ],
        "add_edges": [
            ["b", "repair"],
            ["repair", "end"],
        ],
    }

    assert _parse_patch(workflow, "b", patch) == []


def test_graph_patch_rejects_redundant_existing_edge_atomically():
    workflow = _chain_workflow("a", "b")
    before = workflow.model_dump(mode="python")
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {"prompt_template": "{b_output}"},
            }
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [
            ["a", "b"],
            ["b", "repair"],
            ["repair", "end"],
        ],
    }

    assert _parse_patch(workflow, "b", patch) == []
    assert workflow.model_dump(mode="python") == before
    assert "repair" not in workflow.nodes
    assert workflow.edges == [
        ("start", "a"),
        ("a", "b"),
        ("b", "end"),
    ]


def test_graph_patch_recognizes_condition_successor_order_only_change():
    workflow = WorkflowTemplate(
        name="condition-order",
        entry_node="start",
        nodes={
            "start": Node(node_id="start", node_type=NodeType.START),
            "gate": Node(
                node_id="gate",
                node_type=NodeType.CONDITION,
                config=NodeConfig(
                    node_type=NodeType.CONDITION,
                    condition_expr="True",
                ),
            ),
            "yes": _llm_node("yes"),
            "no": _llm_node("no"),
            "end": Node(node_id="end", node_type=NodeType.END),
        },
        edges=[
            ("start", "gate"),
            ("gate", "yes"),
            ("gate", "no"),
            ("yes", "end"),
            ("no", "end"),
        ],
    )
    patch = {
        "remove_edges": [
            ["gate", "yes"],
            ["gate", "no"],
        ],
        "add_edges": [
            ["gate", "no"],
            ["gate", "yes"],
        ],
    }

    candidates = _parse_patch(workflow, "gate", patch)

    assert len(candidates) == 1
    assert candidates[0].modified_workflow.get_successors("gate") == [
        "no",
        "yes",
    ]
    assert "gate" in candidates[0].changed_units


def test_graph_patch_rejects_cross_block_non_boundary_input():
    workflow = _chain_workflow("remote", "ingress", "a", "b")
    _declare_block(workflow, "a", "b")
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {"prompt_template": "{remote_output}"},
            }
        ],
        "remove_edges": [["a", "b"]],
        "add_edges": [
            ["a", "repair"],
            ["repair", "b"],
        ],
    }

    assert _parse_patch(workflow, "a", patch) == []


@pytest.mark.asyncio
async def test_accepted_graph_patch_executes_its_materialized_path():
    workflow = _chain_workflow("a", "b")
    patch = {
        "add_nodes": [
            {
                "node_id": "repair",
                "node_type": "llm",
                "config": {"prompt_template": "{b_output}"},
            }
        ],
        "remove_edges": [["b", "end"]],
        "add_edges": [["b", "repair"], ["repair", "end"]],
    }
    candidates = _parse_patch(workflow, "b", patch)
    assert len(candidates) == 1

    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(max_steps=8),
        operators={
            "a": lambda context: "draft",
            "b": lambda context: "diagnosed",
            "repair": lambda context: "repaired",
        },
    ).execute(
        candidates[0].modified_workflow,
        FixedScheduler(),
        "query",
    )

    assert output == "repaired"
    assert context.success is True
    assert context.history == ["start", "a", "b", "repair"]
    assert recorder.trace.success is True
