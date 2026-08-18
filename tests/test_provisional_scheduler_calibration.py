from __future__ import annotations

import json

from awf.workflow.serializer import dump_workflow
from experiments.scripts.run_provisional_scheduler_calibration import (
    reconstruct_workflow,
)


def test_reconstructs_selected_provisional_block_update(tmp_path) -> None:
    from awf.workflow.serializer import load_workflow

    initial = load_workflow("experiments/workflows/math/default_workflow.yaml")
    workflow_path = tmp_path / "initial.yaml"
    dump_workflow(initial, workflow_path)
    results = {
        "rounds": [
            {
                "deferred_updates": [
                    {
                        "accepted": True,
                        "update_index": "r1-c0",
                        "cluster_key": ["math", "Precalculus"],
                        "workflow_version_after": "1.1",
                        "summary": {
                            "candidate_evaluations": [
                                {
                                    "status": "selected",
                                    "final_selected": True,
                                    "scope": "block",
                                    "node_id": "verify",
                                    "description": "insert repair",
                                    "changes": {
                                        "graph_patch": {
                                            "add_nodes": [
                                                {
                                                    "node_id": "verify_repair",
                                                    "node_type": "llm",
                                                    "config": {
                                                        "system_prompt": "repair",
                                                        "prompt_template": (
                                                            "{query} {verify_output}"
                                                        ),
                                                    },
                                                }
                                            ],
                                            "remove_nodes": [],
                                            "remove_edges": [
                                                ["verify", "finalize"]
                                            ],
                                            "add_edges": [
                                                ["verify", "verify_repair"],
                                                ["verify_repair", "finalize"],
                                            ],
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ]
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    provisional = reconstruct_workflow(results_path, workflow_path)

    assert provisional.version == "1.1"
    assert "verify_repair" in provisional.nodes
    assert ("verify", "finalize") not in provisional.edges
    assert ("verify", "verify_repair") in provisional.edges
