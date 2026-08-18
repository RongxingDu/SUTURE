"""Anchor-local, scope-constrained workflow candidate construction."""

from __future__ import annotations

import ast
import copy
import difflib
import json
import re
import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import networkx as nx

from awf.llm.client import AsyncLLMClient
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import Node, NodeConfig, NodeType
from awf.workflow.params import OperatorParams, PromptParams

VALID_SCOPES = ("prompt", "operator", "block")
# ``multi`` is an opt-in research scope that composes prompt/operator/graph
# edits in one candidate. The historical three scopes remain the default so
# existing experiments and serialized traces retain their schema.
MULTI_SCOPE = "multi"
# ``block`` is the persisted/internal name for graph-path edits. Accept the
# research-facing spelling at the LLM boundary without creating a fourth
# optimization scope.
SCOPE_ALIASES = {
    "graph_path": "block",
    "graph-path": "block",
    "graph path": "block",
    "multi-level": MULTI_SCOPE,
    "multi_level": MULTI_SCOPE,
}


def _canonical_scope(value: Any) -> Any:
    if isinstance(value, str):
        return SCOPE_ALIASES.get(value, value)
    return value


_CANDIDATE_PROMPT = """You are a workflow optimization researcher.
Use the entire supplied failure cluster to propose several materially different
repairs for the requested scope. Structural changes, new reasoning nodes, and
coherent multi-level edits are encouraged when they address the shared failure.

Return one JSON object:
{
  "candidates": [
    {
      "scope": "<the requested scope>",
      "node_id": "<the supplied anchor id>",
      "description": "<what changed and why>",
      "changes": {}
    }
  ]
}

Allowed changes by scope:
- prompt: {"system_prompt": "...", "user_template": "..."}
- operator: {"temperature": 0.3, "model": "...", "max_tokens": 2048,
             "tool_name": "...", "tool_args": {...},
             "condition_expr": "..."}
- block / graph path (legacy single edit):
  {"add_node": {...}} OR {"remove_node": "..."} OR {"reorder": [...]}
- block / graph path (preferred atomic transaction):
  {"graph_patch": {
    "add_nodes": [
      {"node_id": "diagnose", "node_type": "llm", "config": {...}}
    ],
    "remove_nodes": [],
    "remove_edges": [["verify", "finalize"]],
    "add_edges": [
      ["verify", "diagnose"], ["diagnose", "finalize"]
    ]
  }}

For add_node, provide node_id, node_type, config, and either `after`,
`before`, or both. If `after` has multiple outgoing edges, specify `before`.
  For graph_patch, new nodes do not use `after` or `before`; wire every node
with explicit add_edges. The whole graph_patch is one transaction: if any
node, edge, locality, hierarchy, or graph invariant is invalid, none of it is
accepted. Do not mix graph_patch with legacy block keys. Block operations must
stay inside the allowed block/neighborhood. Keep a transaction minimal (at
most 8 new/removed nodes and 16 new/removed edges). A new TOOL node may only
reuse a tool_name already declared by the workflow. A new node may read only
local outputs or an external boundary output already consumed by that block.
  Ordinary nodes cannot fan out; use a CONDITION with exactly two ordered edges.
- multi: {
    "patches": [
      {"scope": "prompt|operator|block", "node_id": "...",
       "changes": {...}}
    ]
  }
  A multi candidate may combine one prompt edit, one operator edit, and one
  graph-path transaction across the supplied local cluster. Validate and
  apply the patches in order. A coherent multi-level repair may change several
  nodes and need not be the smallest textual edit.
"""


@dataclass
class WorkflowCandidate:
    """A validated candidate edit and its materialized workflow."""

    scope: str
    node_id: str
    description: str
    changes: dict
    modified_workflow: Optional[WorkflowTemplate] = None
    edit_distance: float = 0.0
    anchor_id: str = ""
    changed_units: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchor_id:
            self.anchor_id = self.node_id
        self.metadata.setdefault("anchor_id", self.anchor_id)
        self.metadata.setdefault("scope", self.scope)
        self.metadata.setdefault("changed_units", list(self.changed_units))


class CandidateGenerator:
    """Generate and validate candidates inside an anchor's local neighborhood."""

    def __init__(
        self,
        llm: AsyncLLMClient,
        max_candidates: int = 5,
        max_edit_distance: float | None = 1.0,
        allowed_execution_models: list[str] | None = None,
        execution_llm_defaults: dict[str, Any] | None = None,
        allowed_scopes: Iterable[str] | None = None,
        aggressive_generation: bool = False,
        allow_cross_block_graph_updates: bool = False,
        deterministic_block_candidates: Iterable[str] | None = None,
    ):
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if max_edit_distance is not None and max_edit_distance < 0:
            raise ValueError("max_edit_distance cannot be negative")
        self.llm = llm
        self.max_candidates = max_candidates
        self.max_edit_distance = max_edit_distance
        self.allowed_execution_models = frozenset(
            allowed_execution_models or ()
        )
        configured_scopes = tuple(
            _canonical_scope(scope)
            for scope in (
                VALID_SCOPES if allowed_scopes is None else allowed_scopes
            )
        )
        if not configured_scopes or any(
            scope not in {*VALID_SCOPES, MULTI_SCOPE}
            for scope in configured_scopes
        ):
            raise ValueError(
                "allowed_scopes must be a non-empty subset of VALID_SCOPES"
            )
        # The optimizer uses this allowlist to make prompt, operator, and graph
        # path edits explicit. Direct callers retain the historical all-scope
        # default (the internal scope name ``block`` is kept for compatibility).
        self.allowed_scopes = frozenset(configured_scopes)
        self.aggressive_generation = bool(aggressive_generation)
        self.allow_cross_block_graph_updates = bool(
            allow_cross_block_graph_updates
        )
        self.deterministic_block_candidates = tuple(
            deterministic_block_candidates or ()
        )
        self.last_generation_report: dict[str, Any] = {}
        raw_defaults = execution_llm_defaults or {}
        self.execution_llm_defaults = {
            key: copy.deepcopy(raw_defaults[key])
            for key in ("model", "temperature", "max_tokens")
            if key in raw_defaults
        }

    async def generate(
        self,
        workflow: WorkflowTemplate,
        anchors: list[dict],
        failure_context: str = "",
        experience_context: str = "",
    ) -> list[WorkflowCandidate]:
        """Compatibility API that validates a single multi-anchor response.

        The optimizer uses :meth:`generate_for_anchor` so it can evaluate
        prompt → operator → graph path incrementally. This method remains useful to
        callers that want one LLM request containing candidates for several
        anchors.
        """
        valid_anchor_ids = [
            self._anchor_id(anchor)
            for anchor in anchors
            if self._anchor_id(anchor) in workflow.nodes
        ]
        if not valid_anchor_ids:
            return []

        anchor_summary = self._format_anchors(anchors)
        workflow_summary = self._summarize_workflow(workflow)
        user_prompt = f"""Current workflow:
{workflow_summary}

Failure context:
{failure_context or 'Multiple queries failed on this workflow.'}

Prior candidate experience:
{experience_context or 'No prior candidate outcomes are available.'}

Allowed anchor ids:
{anchor_summary}

Propose distinct, failure-grounded edits. Respond with JSON only."""
        response, _ = await self.llm.generate_json(
            system_prompt=self._candidate_prompt(),
            user_prompt=user_prompt,
            temperature=0.2,
        )
        return self._parse_candidates(
            response,
            workflow,
            allowed_anchor_ids=set(valid_anchor_ids),
            requested_scope=None,
            limit=self.max_candidates,
        )

    async def generate_for_anchor(
        self,
        workflow: WorkflowTemplate,
        anchor: dict | str,
        scope: str,
        failure_context: str = "",
        experience_context: str = "",
        *,
        limit: int | None = None,
    ) -> list[WorkflowCandidate]:
        """Generate candidates for exactly one anchor and one edit scope."""
        anchor_id = anchor if isinstance(anchor, str) else self._anchor_id(anchor)
        scope = _canonical_scope(scope)
        if anchor_id not in workflow.nodes or scope not in self.allowed_scopes:
            return []
        requested = self.max_candidates if limit is None else limit
        request_limit = min(requested, self.max_candidates)
        if request_limit <= 0:
            return []

        allowed_units = sorted(self._allowed_units(workflow, anchor_id, scope))
        workflow_summary = self._summarize_workflow(workflow)
        anchor_summary = self._format_anchor_detail(anchor)
        user_prompt = f"""Current workflow:
{workflow_summary}

Failure context:
{failure_context or 'The current workflow failed on the counterfactual batch.'}

Prior candidate experience:
{experience_context or 'No prior candidate outcomes are available.'}

Anchor evidence:
{anchor_summary}

Anchor id: {anchor_id}
Requested scope: {scope}
Allowed existing units: {json.dumps(allowed_units)}
Maximum candidates: {request_limit}

Return only {scope}-scope candidates targeting {anchor_id}. Respond with JSON only."""
        response, _ = await self.llm.generate_json(
            system_prompt=self._candidate_prompt(),
            user_prompt=user_prompt,
            temperature=0.2,
        )
        llm_candidates = self._parse_candidates(
            response,
            workflow,
            allowed_anchor_ids={anchor_id},
            requested_scope=scope,
            limit=request_limit,
        )
        if scope != "block" or not self.deterministic_block_candidates:
            return llm_candidates

        # These are fixed research action templates, not hand-coded answers:
        # their prompts consume the query and incumbent artifacts at runtime.
        # They guarantee that block-level exploration is represented even when
        # the optimizer LLM emits an invalid graph transaction.
        macro_raw = self._deterministic_math_block_candidates(
            workflow,
            anchor_id,
        )
        macro_candidates = self._parse_candidates(
            json.dumps({"candidates": macro_raw}, ensure_ascii=False),
            workflow,
            allowed_anchor_ids={anchor_id},
            requested_scope="block",
            limit=max(request_limit, len(macro_raw)),
        )
        combined: list[WorkflowCandidate] = []
        seen: set[str] = set()
        # Keep an LLM-generated candidate in the finite budget, then add the
        # deterministic structural alternatives. The optimizer's evaluation
        # budget remains the final authority on how many are actually run.
        for candidate in [*macro_candidates, *llm_candidates]:
            fingerprint = str(candidate.metadata.get("patch_fingerprint", ""))
            if fingerprint and fingerprint in seen:
                continue
            if fingerprint:
                seen.add(fingerprint)
            combined.append(candidate)
        return combined

    def _deterministic_math_block_candidates(
        self,
        workflow: WorkflowTemplate,
        anchor_id: str,
    ) -> list[dict[str, Any]]:
        """Compile reusable self-refine and dual-solve graph macros."""
        if "solve" not in workflow.nodes or "verify" not in workflow.nodes:
            return []
        if ("solve", "verify") not in workflow.edges:
            return []
        rows: list[dict[str, Any]] = []
        if (
            "self_refine" in self.deterministic_block_candidates
            and "self_refine" not in workflow.nodes
        ):
            rows.append({
                "scope": "block",
                "node_id": anchor_id,
                "description": (
                    "Insert a deterministic self-refine block that diagnoses "
                    "the draft and emits a corrected boxed solution."
                ),
                "changes": {"graph_patch": {
                    "add_nodes": [{
                        "node_id": "self_refine",
                        "node_type": "llm",
                        "label": "Self-refine solution",
                        "config": {
                            "system_prompt": (
                                "You are a strict mathematical self-refiner. "
                                "Find the first substantive error and rewrite "
                                "the solution; never merely endorse the draft."
                            ),
                            "prompt_template": (
                                "Problem:\n{query}\n\nDraft solution:\n"
                                "{solve_output}\n\nIndependently check the result, "
                                "repair any error, and end with exactly one "
                                "final answer in \\boxed{{}}."
                            ),
                            "temperature": 0.0,
                        },
                    }],
                    "remove_nodes": [],
                    "remove_edges": [["solve", "verify"]],
                    "add_edges": [
                        ["solve", "self_refine"],
                        ["self_refine", "verify"],
                    ],
                }},
            })
        if (
            "dual_solve_judge" in self.deterministic_block_candidates
            and "dual_solve" not in workflow.nodes
            and "dual_solve_judge" not in workflow.nodes
        ):
            rows.append({
                "scope": "block",
                "node_id": anchor_id,
                "description": (
                    "Insert an independent second solver and a deterministic "
                    "judge that reconciles both solutions."
                ),
                "changes": {"graph_patch": {
                    "add_nodes": [
                        {
                            "node_id": "dual_solve",
                            "node_type": "llm",
                            "label": "Independent second solution",
                            "config": {
                                "system_prompt": (
                                    "Solve the mathematics independently. Do "
                                    "not assume the first draft is correct."
                                ),
                                "prompt_template": (
                                    "Problem:\n{query}\n\nProduce an independent "
                                    "derivation and end with \\boxed{{}}."
                                ),
                                "temperature": 0.0,
                            },
                        },
                        {
                            "node_id": "dual_solve_judge",
                            "node_type": "llm",
                            "label": "Judge two mathematical solutions",
                            "config": {
                                "system_prompt": (
                                    "You are a mathematical adjudicator. "
                                    "Recompute disputed steps and choose or "
                                    "repair the correct solution."
                                ),
                                "prompt_template": (
                                    "Problem:\n{query}\n\nSolution A:\n"
                                    "{solve_output}\n\nSolution B:\n"
                                    "{dual_solve_output}\n\nCompare them step by "
                                    "step and end with one answer in "
                                    "\\boxed{{}}."
                                ),
                                "temperature": 0.0,
                            },
                        },
                    ],
                    "remove_nodes": [],
                    "remove_edges": [["solve", "verify"]],
                    "add_edges": [
                        ["solve", "dual_solve"],
                        ["dual_solve", "dual_solve_judge"],
                        ["dual_solve_judge", "verify"],
                    ],
                }},
            })
        if (
            "verify_repair" in self.deterministic_block_candidates
            and "verify_repair" not in workflow.nodes
            and ("verify", "finalize") in workflow.edges
        ):
            rows.append({
                "scope": "block",
                "node_id": anchor_id,
                "description": (
                    "Insert a verifier-repair block that resolves a rejected "
                    "or inconsistent draft before finalization."
                ),
                "changes": {"graph_patch": {
                    "add_nodes": [{
                        "node_id": "verify_repair",
                        "node_type": "llm",
                        "label": "Repair after verification",
                        "config": {
                            "system_prompt": (
                                "You are a rigorous mathematical repairer. "
                                "Use the verifier as evidence, recompute the "
                                "disputed step, and never repeat a known error."
                            ),
                            "prompt_template": (
                                "Problem:\n{query}\n\nDraft:\n{solve_output}"
                                "\n\nVerifier report:\n{verify_output}\n\n"
                                "Repair the solution if needed and end with "
                                "exactly one answer in \\boxed{{}}."
                            ),
                            "temperature": 0.0,
                        },
                    }],
                    "remove_nodes": [],
                    "remove_edges": [["verify", "finalize"]],
                    "add_edges": [
                        ["verify", "verify_repair"],
                        ["verify_repair", "finalize"],
                    ],
                }},
            })
        if (
            "conditional_debate" in self.deterministic_block_candidates
            and "conditional_debate_gate" not in workflow.nodes
            and "conditional_debate" not in workflow.nodes
            and ("verify", "finalize") in workflow.edges
        ):
            rows.append({
                "scope": "block",
                "node_id": anchor_id,
                "description": (
                    "Insert a conditional debate branch only when verification "
                    "reports an error or lacks an explicit pass marker."
                ),
                "changes": {"graph_patch": {
                    "add_nodes": [
                        {
                            "node_id": "conditional_debate_gate",
                            "node_type": "condition",
                            "label": "Debate only uncertain verification",
                            "config": {
                                "condition_expr": (
                                    "('ERROR' in outputs.get('verify', '') "
                                    "or 'Error' in outputs.get('verify', '') "
                                    "or 'VERIFIED' not in "
                                    "outputs.get('verify', ''))"
                                ),
                            },
                        },
                        {
                            "node_id": "conditional_debate",
                            "node_type": "llm",
                            "label": "Resolve mathematical debate",
                            "config": {
                                "system_prompt": (
                                    "Act as a debate chair. Contrast the solver "
                                    "and verifier, recompute their disagreement, "
                                    "and synthesize only the defensible answer."
                                ),
                                "prompt_template": (
                                    "Problem:\n{query}\n\nSolver position:\n"
                                    "{solve_output}\n\nVerifier position:\n"
                                    "{verify_output}\n\nResolve the disagreement "
                                    "and end with one answer in \\boxed{{}}."
                                ),
                                "temperature": 0.0,
                            },
                        },
                    ],
                    "remove_nodes": [],
                    "remove_edges": [["verify", "finalize"]],
                    # CONDITION successor order is (true, false).
                    "add_edges": [
                        ["verify", "conditional_debate_gate"],
                        ["conditional_debate_gate", "conditional_debate"],
                        ["conditional_debate_gate", "finalize"],
                        ["conditional_debate", "finalize"],
                    ],
                }},
            })
        if (
            "format_repair" in self.deterministic_block_candidates
            and "format_repair" not in workflow.nodes
            and ("verify", "finalize") in workflow.edges
        ):
            rows.append({
                "scope": "block",
                "node_id": anchor_id,
                "description": (
                    "Insert a low-temperature format repair that preserves the "
                    "mathematical conclusion while making it extractable."
                ),
                "changes": {"graph_patch": {
                    "add_nodes": [{
                        "node_id": "format_repair",
                        "node_type": "llm",
                        "label": "Normalize final math answer",
                        "config": {
                            "system_prompt": (
                                "You are a mathematical answer formatter. "
                                "Preserve the supported conclusion and repair "
                                "only ambiguity or output format."
                            ),
                            "prompt_template": (
                                "Problem:\n{query}\n\nDraft solution:\n"
                                "{solve_output}\n\nVerification:\n"
                                "{verify_output}\n\nReturn a concise supported "
                                "final answer ending in \\boxed{{}}."
                            ),
                            "temperature": 0.0,
                            "max_tokens": 512,
                        },
                    }],
                    "remove_nodes": [],
                    "remove_edges": [["verify", "finalize"]],
                    "add_edges": [
                        ["verify", "format_repair"],
                        ["format_repair", "finalize"],
                    ],
                }},
            })
        return rows

    def _parse_candidates(
        self,
        response: str,
        workflow: WorkflowTemplate,
        *,
        allowed_anchor_ids: set[str],
        requested_scope: str | None,
        limit: int,
    ) -> list[WorkflowCandidate]:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            self.last_generation_report = {
                "raw_candidates": 0,
                "valid_candidates": 0,
                "rejections": {"invalid_json": 1},
            }
            return []
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            self.last_generation_report = {
                "raw_candidates": 0,
                "valid_candidates": 0,
                "rejections": {"missing_candidates_array": 1},
            }
            return []

        candidates: list[WorkflowCandidate] = []
        rejection_counts: dict[str, int] = {}
        def reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        fingerprints: set[str] = set()
        patch_fingerprints: set[str] = set()
        # Parse a few extra entries because invalid candidates do not consume
        # the caller's valid-candidate budget.
        for raw in data["candidates"][: max(limit * 3, limit)]:
            candidate = self._validate_raw_candidate(
                raw,
                workflow,
                allowed_anchor_ids,
                requested_scope,
            )
            if candidate is None:
                reject("invalid_scope_or_changes")
                continue
            modified = self._apply_to_workflow(candidate, workflow)
            if modified is None:
                reject("invalid_materialized_workflow")
                continue
            candidate.modified_workflow = modified
            candidate.changed_units = self._changed_units(
                workflow,
                modified,
                candidate,
            )
            if not candidate.changed_units:
                reject("no_effect")
                continue
            candidate.edit_distance = self._compute_edit_distance(
                candidate,
                workflow,
            )
            if (
                self.max_edit_distance is not None
                and candidate.edit_distance > self.max_edit_distance
            ):
                reject("edit_distance_exceeded")
                continue
            candidate.metadata.update(
                {
                    "anchor_id": candidate.anchor_id,
                    "scope": candidate.scope,
                    "changed_units": list(candidate.changed_units),
                }
            )

            patch_fingerprint = self.candidate_patch_fingerprint(candidate)
            if patch_fingerprint in patch_fingerprints:
                reject("duplicate_patch")
                continue
            patch_fingerprints.add(patch_fingerprint)
            candidate.metadata["patch_fingerprint"] = patch_fingerprint

            fingerprint = self.workflow_fingerprint(modified)
            if fingerprint in fingerprints:
                reject("duplicate_workflow")
                continue
            fingerprints.add(fingerprint)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break

        self.last_generation_report = {
            "raw_candidates": len(data["candidates"]),
            "valid_candidates": len(candidates),
            "rejections": rejection_counts,
        }
        return sorted(candidates, key=lambda item: item.edit_distance)

    def _validate_raw_candidate(
        self,
        raw: Any,
        workflow: WorkflowTemplate,
        allowed_anchor_ids: set[str],
        requested_scope: str | None,
    ) -> WorkflowCandidate | None:
        if not isinstance(raw, dict):
            return None
        scope = _canonical_scope(raw.get("scope"))
        requested_scope = _canonical_scope(requested_scope)
        node_id = raw.get("node_id", raw.get("anchor_id"))
        description = raw.get("description", raw.get("edit_summary", ""))
        changes = raw.get("changes")
        if (
            scope not in self.allowed_scopes
            or (requested_scope is not None and scope != requested_scope)
            or not isinstance(node_id, str)
            or node_id not in allowed_anchor_ids
            or node_id not in workflow.nodes
            or not isinstance(description, str)
            or not description.strip()
            or not isinstance(changes, dict)
            or not changes
        ):
            return None
        if not self._validate_changes(scope, changes, workflow, node_id):
            return None

        return WorkflowCandidate(
            scope=scope,
            node_id=node_id,
            anchor_id=node_id,
            description=description.strip()[:2000],
            changes=copy.deepcopy(changes),
        )

    def _validate_changes(
        self,
        scope: str,
        changes: dict,
        workflow: WorkflowTemplate,
        anchor_id: str,
    ) -> bool:
        if scope == MULTI_SCOPE:
            patches = changes.get("patches")
            if (
                not isinstance(patches, list)
                or not patches
                or len(patches) > 8
            ):
                return False
            allowed_units = self._allowed_units(
                workflow,
                anchor_id,
                "block",
            )
            seen_levels: set[tuple[str, str]] = set()
            for patch in patches:
                if not isinstance(patch, dict):
                    return False
                patch_scope = _canonical_scope(patch.get("scope"))
                patch_node = patch.get("node_id", patch.get("anchor_id"))
                patch_changes = patch.get("changes")
                if (
                    patch_scope not in VALID_SCOPES
                    or not isinstance(patch_node, str)
                    or patch_node not in allowed_units
                    or patch_node not in workflow.nodes
                    or not isinstance(patch_changes, dict)
                    or not patch_changes
                ):
                    return False
                level_key = (patch_scope, patch_node)
                if level_key in seen_levels:
                    return False
                seen_levels.add(level_key)
                if not self._validate_changes(
                    patch_scope,
                    patch_changes,
                    workflow,
                    patch_node,
                ):
                    return False
            return True

        if scope == "prompt":
            if (
                workflow.nodes[anchor_id].node_type != NodeType.LLM
                or not set(changes) <= {"system_prompt", "user_template"}
            ):
                return False
            if not all(isinstance(value, str) for value in changes.values()):
                return False
            if "user_template" in changes and not self._valid_prompt_template(
                changes["user_template"],
                workflow,
            ):
                return False
            return True

        if scope == "operator":
            node_type = workflow.nodes[anchor_id].node_type
            allowed_by_type = {
                NodeType.LLM: {"temperature", "model", "max_tokens"},
                NodeType.TOOL: {"tool_name", "tool_args"},
                NodeType.CONDITION: {"condition_expr"},
            }
            allowed = allowed_by_type.get(node_type, set())
            if not set(changes) <= allowed:
                return False
            if "temperature" in changes and (
                isinstance(changes["temperature"], bool)
                or not isinstance(changes["temperature"], (int, float))
                or not 0 <= changes["temperature"] <= 2
            ):
                return False
            if "max_tokens" in changes and (
                isinstance(changes["max_tokens"], bool)
                or not isinstance(changes["max_tokens"], int)
                or changes["max_tokens"] <= 0
            ):
                return False
            for key in ("model", "tool_name", "condition_expr"):
                if key in changes and not isinstance(changes[key], str):
                    return False
            if (
                "model" in changes
                and not self._execution_model_is_allowed(changes["model"])
            ):
                return False
            if (
                "condition_expr" in changes
                and self._condition_output_dependencies(
                    changes["condition_expr"]
                )
                is None
            ):
                return False
            if "tool_args" in changes and not isinstance(changes["tool_args"], dict):
                return False
            return True

        allowed_block_keys = {"add_node", "remove_node", "reorder", "graph_patch"}
        if not set(changes) <= allowed_block_keys:
            return False
        if "graph_patch" in changes:
            return (
                set(changes) == {"graph_patch"}
                and self._validate_graph_patch(
                    changes["graph_patch"],
                    workflow,
                    anchor_id,
                )
            )
        allowed_units = self._allowed_units(workflow, anchor_id, "block")
        if "remove_node" in changes:
            remove_id = changes["remove_node"]
            if (
                not isinstance(remove_id, str)
                or remove_id not in allowed_units
                or remove_id not in workflow.nodes
                or remove_id == workflow.entry_node
                or workflow.nodes[remove_id].node_type
                in {NodeType.START, NodeType.END}
            ):
                return False
        if "reorder" in changes:
            order = changes["reorder"]
            if (
                not isinstance(order, list)
                or len(order) < 2
                or any(not isinstance(item, str) for item in order)
                or len(order) != len(set(order))
                or anchor_id not in order
                or not set(order) <= allowed_units
                or any(item not in workflow.nodes for item in order)
                or any(
                    workflow.nodes[item].node_type
                    in {NodeType.START, NodeType.END}
                    for item in order
                )
            ):
                return False
        if "add_node" in changes:
            data = changes["add_node"]
            if not isinstance(data, dict):
                return False
            new_id = data.get("node_id")
            if (
                not isinstance(new_id, str)
                or not new_id
                or new_id in workflow.nodes
                or not (data.get("after") or data.get("before"))
            ):
                return False
            try:
                new_type = NodeType(data.get("node_type", "llm"))
            except (TypeError, ValueError):
                return False
            if new_type in {NodeType.START, NodeType.END}:
                return False
            config = data.get("config", {})
            if not isinstance(config, dict):
                return False
            try:
                NodeConfig(
                    **{
                        **copy.deepcopy(config),
                        "node_type": new_type,
                    }
                )
            except (TypeError, ValueError):
                return False
            if not self._valid_new_node_config(
                new_type,
                config,
                workflow,
            ):
                return False
            if (
                "prompt_template" in config
                and (
                    not isinstance(config["prompt_template"], str)
                    or not self._valid_prompt_template(
                        config["prompt_template"],
                        workflow,
                    )
                )
            ):
                return False
            if (
                "model" in config
                and not self._execution_model_is_allowed(config["model"])
            ):
                return False
            for boundary in ("after", "before"):
                if boundary in data:
                    value = data[boundary]
                    if (
                        not isinstance(value, str)
                        or value not in allowed_units
                        or value not in workflow.nodes
                    ):
                        return False
        return True

    def _validate_graph_patch(
        self,
        patch: Any,
        workflow: WorkflowTemplate,
        anchor_id: str,
    ) -> bool:
        """Validate an anchor-local multi-node graph edit as one transaction."""
        if not isinstance(patch, dict):
            return False
        allowed_keys = {
            "add_nodes",
            "remove_nodes",
            "add_edges",
            "remove_edges",
        }
        if not patch or not set(patch) <= allowed_keys:
            return False

        add_nodes = patch.get("add_nodes", [])
        remove_nodes = patch.get("remove_nodes", [])
        add_edges = patch.get("add_edges", [])
        remove_edges = patch.get("remove_edges", [])
        if (
            not isinstance(add_nodes, list)
            or not isinstance(remove_nodes, list)
            or not isinstance(add_edges, list)
            or not isinstance(remove_edges, list)
            or not any((add_nodes, remove_nodes, add_edges, remove_edges))
            or len(add_nodes) + len(remove_nodes) > 8
            or len(add_edges) + len(remove_edges) > 16
        ):
            return False

        allowed_units = self._allowed_units(workflow, anchor_id, "block")
        if (
            any(not isinstance(item, str) for item in remove_nodes)
            or len(remove_nodes) != len(set(remove_nodes))
        ):
            return False
        remove_set = set(remove_nodes)
        if anchor_id in remove_set and add_nodes:
            return False
        for node_id in remove_nodes:
            if (
                node_id not in allowed_units
                or node_id not in workflow.nodes
                or node_id == workflow.entry_node
                or workflow.nodes[node_id].node_type
                in {NodeType.START, NodeType.END}
            ):
                return False
            # Removing a block-boundary node would implicitly mutate edges in
            # a different block, even if those edges were omitted from patch.
            if any(
                (source == node_id and target not in allowed_units)
                or (target == node_id and source not in allowed_units)
                for source, target in workflow.edges
            ):
                return False
        if any(
            source in remove_set or target in remove_set
            for source, target in (
                tuple(edge)
                for edge in remove_edges
                if isinstance(edge, (list, tuple)) and len(edge) == 2
            )
        ):
            # Removing a node already removes all incident edges. Requiring the
            # patch to choose one representation makes every sub-operation
            # meaningful and prevents silent partial no-ops.
            return False

        new_ids: list[str] = []
        for data in add_nodes:
            if not isinstance(data, dict) or not set(data) <= {
                "node_id",
                "node_type",
                "config",
                "label",
                "position",
            }:
                return False
            new_id = data.get("node_id")
            if (
                not isinstance(new_id, str)
                or not new_id
                or new_id in workflow.nodes
                or new_id in new_ids
            ):
                return False
            try:
                new_type = NodeType(data.get("node_type", "llm"))
            except (TypeError, ValueError):
                return False
            if new_type in {NodeType.START, NodeType.END}:
                return False
            config = data.get("config", {})
            if not isinstance(config, dict):
                return False
            try:
                NodeConfig(**{**copy.deepcopy(config), "node_type": new_type})
            except (TypeError, ValueError):
                return False
            if not self._valid_new_node_config(
                new_type,
                config,
                workflow,
            ):
                return False
            if (
                "model" in config
                and not self._execution_model_is_allowed(config["model"])
            ):
                return False
            new_ids.append(new_id)

        allowed_prompt_nodes = (
            set(workflow.nodes) - remove_set
        ) | set(new_ids)
        for data in add_nodes:
            config = data.get("config", {})
            template = config.get("prompt_template")
            if (
                template is not None
                and (
                    not isinstance(template, str)
                    or not self._valid_prompt_template(
                        template,
                        workflow,
                        extra_node_ids=allowed_prompt_nodes,
                    )
                )
            ):
                return False

        def parse_edges(raw_edges: list[Any]) -> list[tuple[str, str]] | None:
            parsed: list[tuple[str, str]] = []
            for edge in raw_edges:
                if (
                    not isinstance(edge, (list, tuple))
                    or len(edge) != 2
                    or not all(isinstance(item, str) for item in edge)
                    or edge[0] == edge[1]
                ):
                    return None
                parsed.append((edge[0], edge[1]))
            return parsed

        parsed_add_edges = parse_edges(add_edges)
        parsed_remove_edges = parse_edges(remove_edges)
        if parsed_add_edges is None or parsed_remove_edges is None:
            return False
        if (
            len(parsed_add_edges) != len(set(parsed_add_edges))
            or len(parsed_remove_edges) != len(set(parsed_remove_edges))
            or any(edge not in workflow.edges for edge in parsed_remove_edges)
            or any(
                edge in workflow.edges and edge not in parsed_remove_edges
                for edge in parsed_add_edges
            )
        ):
            return False

        valid_after = (set(workflow.nodes) - remove_set) | set(new_ids)
        for source, target in parsed_add_edges:
            if source not in valid_after or target not in valid_after:
                return False
        patch_node_ids = {
            item
            for edge in (*parsed_add_edges, *parsed_remove_edges)
            for item in edge
        }
        if any(
            node_id not in allowed_units and node_id not in new_ids
            for node_id in patch_node_ids
        ):
            return False
        # New nodes must be wired by this transaction. Full reachability is
        # checked again after materialization.
        if any(
            not any(node_id in edge for edge in parsed_add_edges)
            for node_id in new_ids
        ):
            return False
        return True

    @classmethod
    def _valid_new_node_config(
        cls,
        node_type: NodeType,
        config: dict[str, Any],
        workflow: WorkflowTemplate,
    ) -> bool:
        """Enforce the runtime-relevant config shape for a newly added node."""
        common = {"metadata"}
        allowed_by_type = {
            NodeType.LLM: common
            | {
                "prompt_template",
                "system_prompt",
                "model",
                "temperature",
                "max_tokens",
            },
            NodeType.TOOL: common | {"tool_name", "tool_args"},
            NodeType.CONDITION: common | {"condition_expr"},
            NodeType.JOIN: common,
        }
        if not set(config) <= allowed_by_type.get(node_type, set()):
            return False
        if node_type == NodeType.TOOL:
            tool_name = config.get("tool_name")
            known_tools = {
                node.config.tool_name or node_id
                for node_id, node in workflow.nodes.items()
                if node.node_type == NodeType.TOOL
            }
            return (
                isinstance(tool_name, str)
                and bool(tool_name.strip())
                and tool_name in known_tools
            )
        if node_type == NodeType.CONDITION:
            expression = config.get("condition_expr")
            return (
                isinstance(expression, str)
                and bool(expression.strip())
                and cls._condition_output_dependencies(expression) is not None
            )
        return True

    @staticmethod
    def _valid_prompt_template(
        template: str,
        workflow: WorkflowTemplate,
        *,
        extra_node_ids: set[str] | None = None,
    ) -> bool:
        """Reject templates that cannot be rendered by the executor.

        Candidate prompts are ordinary ``str.format_map`` templates.  A model
        writing mathematical notation such as ``\\boxed{}`` accidentally
        creates an empty positional field and crashes every counterfactual
        execution.  Only the executor's named query/output variables are
        accepted; literal braces must use ``{{`` and ``}}``.
        """
        allowed_fields = {
            "query",
            *(
                f"{node_id}_output"
                for node_id in (extra_node_ids or set(workflow.nodes))
            ),
        }
        try:
            parsed = list(string.Formatter().parse(template))
        except ValueError:
            return False
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                not field_name
                or not field_name.isidentifier()
                or field_name not in allowed_fields
                or format_spec
                or conversion is not None
            ):
                return False
        return True

    def _execution_model_is_allowed(self, model: object) -> bool:
        """Validate an explicit candidate-selected execution model."""
        return (
            not self.allowed_execution_models
            or isinstance(model, str)
            and model in self.allowed_execution_models
        )

    def _candidate_prompt(self) -> str:
        """Include the model-selection policy in every candidate request."""
        if not self.allowed_execution_models:
            policy = (
                "Execution-model policy: no execution-model allowlist is "
                "configured, so explicit model edits are unrestricted."
            )
        else:
            allowed = json.dumps(
                sorted(self.allowed_execution_models),
                ensure_ascii=False,
            )
            policy = (
                "Execution-model policy: the only allowed execution model "
                f"identifiers are {allowed}. Any operator `changes.model` or "
                "block `add_node.config.model` / "
                "`graph_patch.add_nodes[].config.model` value MUST exactly "
                "match one "
                "of these identifiers. Never propose any other model. "
                "Omitting `model` keeps the configured runtime default."
            )
        return (
            f"{_CANDIDATE_PROMPT}\n{policy}\n"
            "The workflow summary reports effective execution settings after "
            "applying runtime defaults. Compare proposed values against those "
            "effective values; never describe 0.2 as lowering a current "
            "temperature of 0.0. Use the supplied anchor evidence and prior "
            "candidate outcomes, and do not repeat an exact failed patch."
        )

    def _apply_to_workflow(
        self,
        candidate: WorkflowCandidate,
        workflow: WorkflowTemplate,
    ) -> WorkflowTemplate | None:
        """Apply only validated changes and reject incoherent graph results."""
        modified = copy.deepcopy(workflow)
        node_id = candidate.node_id
        changes = candidate.changes

        if candidate.scope == MULTI_SCOPE:
            # Materialize each level against the workflow produced by the
            # previous level. This keeps prompt/operator hierarchy sync and
            # graph invariants identical to single-scope candidates.
            patches = changes.get("patches", [])
            for patch in patches:
                patch_scope = _canonical_scope(patch.get("scope"))
                patch_node = patch.get("node_id", patch.get("anchor_id"))
                patch_changes = copy.deepcopy(patch.get("changes", {}))
                if (
                    not isinstance(patch_scope, str)
                    or not isinstance(patch_node, str)
                    or patch_node not in modified.nodes
                    or not self._validate_changes(
                        patch_scope,
                        patch_changes,
                        modified,
                        patch_node,
                    )
                ):
                    return None
                sub_candidate = WorkflowCandidate(
                    scope=patch_scope,
                    node_id=patch_node,
                    anchor_id=patch_node,
                    description=candidate.description,
                    changes=patch_changes,
                )
                next_workflow = self._apply_to_workflow(
                    sub_candidate,
                    modified,
                )
                if next_workflow is None:
                    return None
                modified = next_workflow

        elif candidate.scope == "prompt":
            node = modified.nodes[node_id]
            if "system_prompt" in changes:
                node.config.system_prompt = changes["system_prompt"]
            if "user_template" in changes:
                node.config.prompt_template = changes["user_template"]
            self._sync_hierarchy(modified, node_id, changes, "prompt")

        elif candidate.scope == "operator":
            node = modified.nodes[node_id]
            for key, value in changes.items():
                setattr(node.config, key, copy.deepcopy(value))
            self._sync_hierarchy(modified, node_id, changes, "operator")
            if (
                node.node_type == NodeType.CONDITION
                and not self._validate_graph_patch_dependencies(
                    modified,
                    {},
                    workflow,
                    node_id,
                )
            ):
                return None

        else:
            try:
                if "graph_patch" in changes:
                    self._apply_graph_patch(
                        modified,
                        changes["graph_patch"],
                        node_id,
                    )
                    if not self._validate_graph_patch_dependencies(
                        modified,
                        changes["graph_patch"],
                        workflow,
                        node_id,
                    ):
                        raise ValueError(
                            "graph patch has unavailable input dependencies"
                        )
                if "remove_node" in changes:
                    self._remove_and_reconnect(modified, changes["remove_node"])
                if "add_node" in changes:
                    self._insert_node(modified, changes["add_node"], node_id)
                if "reorder" in changes:
                    self._reorder_nodes(modified, changes["reorder"])
            except (KeyError, TypeError, ValueError):
                return None

        if candidate.scope == "block" and "graph_patch" in changes:
            if len(modified.edges) != len(set(modified.edges)):
                return None
        else:
            self._deduplicate_edges(modified)
        if (
            modified.model_dump(mode="python")
            == workflow.model_dump(mode="python")
            or not self._validate_graph(modified, workflow)
        ):
            return None
        return modified

    @staticmethod
    def _apply_graph_patch(
        workflow: WorkflowTemplate,
        patch: dict,
        anchor_id: str,
    ) -> None:
        """Apply a pre-validated graph transaction to a private workflow copy."""
        remove_edges = [tuple(edge) for edge in patch.get("remove_edges", [])]
        for edge in remove_edges:
            if edge not in workflow.edges:
                raise ValueError(f"cannot remove missing edge: {edge!r}")
            workflow.edges.remove(edge)

        for node_id in patch.get("remove_nodes", []):
            if node_id not in workflow.nodes:
                raise ValueError(f"cannot remove missing node: {node_id!r}")
            workflow.edges = [
                (source, target)
                for source, target in workflow.edges
                if source != node_id and target != node_id
            ]
            workflow.nodes.pop(node_id)
            CandidateGenerator._remove_from_hierarchy(workflow, node_id)

        for data in patch.get("add_nodes", []):
            new_id = data["node_id"]
            node_type = NodeType(data.get("node_type", "llm"))
            config_data = copy.deepcopy(data.get("config", {}))
            config_data["node_type"] = node_type
            new_node = Node(
                node_id=new_id,
                node_type=node_type,
                config=NodeConfig(**config_data),
                label=data.get("label", ""),
                position=tuple(data.get("position", (0.0, 0.0))),
            )
            workflow.nodes[new_id] = new_node
            CandidateGenerator._add_to_anchor_block(
                workflow,
                anchor_id,
                new_node,
            )

        workflow.edges.extend(
            tuple(edge) for edge in patch.get("add_edges", [])
        )

    def _validate_graph_patch_dependencies(
        self,
        workflow: WorkflowTemplate,
        patch: dict,
        original: WorkflowTemplate,
        anchor_id: str,
    ) -> bool:
        """Validate all surviving data dependencies on every execution path."""
        graph = workflow.to_networkx()
        if not nx.is_directed_acyclic_graph(graph):
            return False
        try:
            dominators = nx.immediate_dominators(
                graph,
                workflow.entry_node,
            )
        except (KeyError, nx.NetworkXError):
            return False

        dependencies_by_node: dict[str, set[str]] = {}
        for node_id, node in workflow.nodes.items():
            referenced = self._node_output_dependencies(node)
            if referenced is None:
                return False
            dependencies_by_node[node_id] = referenced
            if any(
                dependency not in workflow.nodes
                or not self._dominates(
                    dominators,
                    dependency,
                    node_id,
                )
                for dependency in referenced
            ):
                return False

        allowed_units = self._allowed_units(
            original,
            anchor_id,
            "block",
        )
        original_boundary_inputs: set[str] = set()
        for node_id in allowed_units & set(original.nodes):
            dependencies = self._node_output_dependencies(
                original.nodes[node_id]
            )
            if dependencies is None:
                return False
            original_boundary_inputs.update(dependencies - allowed_units)
        new_ids = {
            data["node_id"] for data in patch.get("add_nodes", [])
        }
        allowed_new_dependencies = (
            allowed_units | original_boundary_inputs | new_ids
        )
        if any(
            not dependencies_by_node[node_id] <= allowed_new_dependencies
            for node_id in new_ids
        ):
            return False
        return True

    @classmethod
    def _node_output_dependencies(
        cls,
        node: Node,
    ) -> set[str] | None:
        dependencies = cls._output_dependencies(
            node.config.prompt_template
        )
        dependencies.update(
            cls._output_dependencies(node.config.tool_args)
        )
        if node.node_type == NodeType.CONDITION:
            expression = node.config.condition_expr
            if not isinstance(expression, str) or not expression.strip():
                return None
            condition_dependencies = cls._condition_output_dependencies(
                expression
            )
            if condition_dependencies is None:
                return None
            dependencies.update(condition_dependencies)
        return dependencies

    @staticmethod
    def _dominates(
        dominators: dict[str, str],
        source: str,
        target: str,
    ) -> bool:
        if source == target or target not in dominators:
            return False
        current = target
        while True:
            parent = dominators.get(current)
            if parent is None or parent == current:
                return False
            if parent == source:
                return True
            current = parent

    @classmethod
    def _condition_output_dependencies(
        cls,
        expression: str,
    ) -> set[str] | None:
        """Statically validate the restricted condition AST and its inputs."""
        if not isinstance(expression, str) or not expression.strip():
            return None
        if len(expression) > 2_000:
            return None
        try:
            parsed = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            return None

        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
            ast.Subscript,
            ast.Attribute,
            ast.Call,
            ast.BoolOp,
            ast.UnaryOp,
            ast.BinOp,
            ast.Compare,
            ast.IfExp,
            ast.Load,
            ast.And,
            ast.Or,
            ast.Not,
            ast.UAdd,
            ast.USub,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Is,
            ast.IsNot,
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in {
                "outputs",
                "vars",
                "context",
            }:
                return None
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    return None
                if (
                    node.attr != "get"
                    and not (
                        isinstance(node.value, ast.Name)
                        and node.value.id == "context"
                        and node.attr
                        in {
                            "query",
                            "current_node_id",
                            "previous_node_id",
                            "step_count",
                            "history",
                            "outputs",
                            "variables",
                            "cost_summary",
                            "finished",
                            "success",
                        }
                    )
                ):
                    return None
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Attribute)
                or node.func.attr != "get"
                or node.keywords
                or not 1 <= len(node.args) <= 2
            ):
                return None

        def literal_key(node: ast.AST) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            return None

        def mapping_kind(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return node.id if node.id in {"outputs", "vars", "context"} else None
            if isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "context"
                    and node.attr in {"outputs", "variables"}
                ):
                    return (
                        "outputs"
                        if node.attr == "outputs"
                        else "vars"
                    )
                return None
            if isinstance(node, ast.Subscript):
                owner = mapping_kind(node.value)
                key = literal_key(node.slice)
                if owner == "context" and key in {"outputs", "variables"}:
                    return "outputs" if key == "outputs" else "vars"
                return None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and mapping_kind(node.func.value) == "context"
            ):
                key = literal_key(node.args[0])
                if key in {"outputs", "variables"}:
                    return "outputs" if key == "outputs" else "vars"
            return None

        dependencies: set[str] = set()
        for node in ast.walk(parsed):
            owner_kind: str | None = None
            key: str | None = None
            if isinstance(node, ast.Subscript):
                owner_kind = mapping_kind(node.value)
                key = literal_key(node.slice)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
            ):
                owner_kind = mapping_kind(node.func.value)
                key = literal_key(node.args[0])
            if owner_kind not in {"outputs", "vars"}:
                continue
            if key is None:
                return None
            if owner_kind == "outputs":
                dependencies.add(key)
            elif key.endswith("_output"):
                dependencies.add(key[: -len("_output")])
        return dependencies

    @classmethod
    def _output_dependencies(cls, value: Any) -> set[str]:
        """Extract ``{node_id_output}`` dependencies from nested config."""
        if isinstance(value, dict):
            dependencies: set[str] = set()
            for nested in value.values():
                dependencies.update(cls._output_dependencies(nested))
            return dependencies
        if isinstance(value, (list, tuple)):
            dependencies = set()
            for nested in value:
                dependencies.update(cls._output_dependencies(nested))
            return dependencies
        if not isinstance(value, str):
            return set()
        fields = re.findall(
            r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})",
            value,
        )
        return {
            field[: -len("_output")]
            for field in fields
            if field.endswith("_output")
        }

    @staticmethod
    def _remove_and_reconnect(
        workflow: WorkflowTemplate,
        remove_id: str,
    ) -> None:
        if remove_id not in workflow.nodes or remove_id == workflow.entry_node:
            raise ValueError("cannot remove target node")
        predecessors = [src for src, dst in workflow.edges if dst == remove_id]
        successors = [dst for src, dst in workflow.edges if src == remove_id]
        workflow.edges = [
            (src, dst)
            for src, dst in workflow.edges
            if src != remove_id and dst != remove_id
        ]
        workflow.nodes.pop(remove_id)
        for source in predecessors:
            for target in successors:
                if source != target:
                    workflow.edges.append((source, target))
        CandidateGenerator._remove_from_hierarchy(workflow, remove_id)

    @staticmethod
    def _insert_node(
        workflow: WorkflowTemplate,
        data: dict,
        anchor_id: str,
    ) -> None:
        new_id = data["node_id"]
        node_type = NodeType(data.get("node_type", "llm"))
        config_data = copy.deepcopy(data.get("config", {}))
        config_data["node_type"] = node_type
        new_node = Node(
            node_id=new_id,
            node_type=node_type,
            config=NodeConfig(**config_data),
            label=data.get("label", ""),
        )
        after = data.get("after")
        before = data.get("before")
        edges = workflow.edges

        if after and before:
            if (after, before) not in edges:
                raise ValueError("add_node boundaries must identify an edge")
            edges.remove((after, before))
        elif after:
            outgoing = [dst for src, dst in edges if src == after]
            if len(outgoing) != 1:
                raise ValueError("ambiguous insertion after a branching node")
            before = outgoing[0]
            edges.remove((after, before))
        elif before:
            incoming = [src for src, dst in edges if dst == before]
            if len(incoming) != 1:
                raise ValueError("ambiguous insertion before a joining node")
            after = incoming[0]
            edges.remove((after, before))
        else:
            raise ValueError("new node needs an insertion boundary")

        if anchor_id not in {after, before}:
            raise ValueError("new node must be adjacent to the anchor")
        workflow.nodes[new_id] = new_node
        workflow.edges.extend([(after, new_id), (new_id, before)])
        CandidateGenerator._add_to_anchor_block(workflow, anchor_id, new_node)

    @staticmethod
    def _reorder_nodes(workflow: WorkflowTemplate, order: list[str]) -> None:
        node_set = set(order)
        external_predecessors = {
            src
            for src, dst in workflow.edges
            if dst in node_set and src not in node_set
        }
        external_successors = {
            dst
            for src, dst in workflow.edges
            if src in node_set and dst not in node_set
        }
        if not external_predecessors and workflow.entry_node not in node_set:
            raise ValueError("reorder would disconnect the block")

        workflow.edges = [
            (src, dst)
            for src, dst in workflow.edges
            if src not in node_set and dst not in node_set
        ]
        workflow.edges.extend((src, order[0]) for src in external_predecessors)
        workflow.edges.extend(
            (order[index], order[index + 1])
            for index in range(len(order) - 1)
        )
        workflow.edges.extend((order[-1], dst) for dst in external_successors)
        if workflow.entry_node in node_set:
            workflow.entry_node = order[0]

    @staticmethod
    def _validate_graph(
        candidate: WorkflowTemplate,
        original: WorkflowTemplate,
    ) -> bool:
        try:
            # Candidate edits mutate a deep copy in place, so re-run the
            # authoritative IR and hierarchy validators before execution.
            WorkflowTemplate.model_validate(
                candidate.model_dump(mode="python")
            )
        except (TypeError, ValueError):
            return False
        if (
            not candidate.nodes
            or candidate.entry_node not in candidate.nodes
            or candidate.entry_node != original.entry_node
        ):
            return False
        if any(
            source not in candidate.nodes
            or target not in candidate.nodes
            or source == target
            for source, target in candidate.edges
        ):
            return False

        graph = candidate.to_networkx()
        if graph.in_degree(candidate.entry_node) != 0:
            return False
        reachable = {candidate.entry_node} | nx.descendants(
            graph,
            candidate.entry_node,
        )
        if reachable != set(candidate.nodes):
            return False

        original_graph = original.to_networkx()
        for node_id, node in candidate.nodes.items():
            successors = candidate.get_successors(node_id)
            if (
                node.node_type != NodeType.CONDITION
                and len(successors) > 1
                and (
                    node_id not in original.nodes
                    or successors != original.get_successors(node_id)
                )
            ):
                # The edge-following executor uses only successor[0] for an
                # ordinary node, while FixedScheduler may visit all reachable
                # nodes. A patch must not introduce this ambiguous fan-out.
                return False
        if nx.is_directed_acyclic_graph(original_graph) and not (
            nx.is_directed_acyclic_graph(graph)
        ):
            return False

        original_end_count = sum(
            node.node_type == NodeType.END for node in original.nodes.values()
        )
        original_non_end_sinks = {
            node_id
            for node_id, node in original.nodes.items()
            if (
                original_graph.out_degree(node_id) == 0
                and node.node_type != NodeType.END
            )
        }
        candidate_non_end_sinks = {
            node_id
            for node_id, node in candidate.nodes.items()
            if graph.out_degree(node_id) == 0 and node.node_type != NodeType.END
        }
        if candidate_non_end_sinks - original_non_end_sinks:
            return False
        original_control_nodes = {
            node_id
            for node_id, node in original.nodes.items()
            if node.node_type in {NodeType.START, NodeType.END}
        }
        candidate_control_nodes = {
            node_id
            for node_id, node in candidate.nodes.items()
            if node.node_type in {NodeType.START, NodeType.END}
        }
        if original_control_nodes != candidate_control_nodes:
            return False
        if original_end_count and any(
            graph.out_degree(node_id) != 0
            for node_id, node in candidate.nodes.items()
            if node.node_type == NodeType.END
        ):
            return False
        if not CandidateGenerator._validate_output_contract(
            candidate,
            original,
        ):
            return False
        return True

    @staticmethod
    def _validate_output_contract(
        candidate: WorkflowTemplate,
        original: WorkflowTemplate,
    ) -> bool:
        """Preserve an optional declared terminal-output producer contract."""
        contract = original.metadata.get("output_contract")
        if contract is None:
            return True
        if (
            not isinstance(contract, dict)
            or set(contract) != {"terminal_producers"}
        ):
            return False
        producers = contract.get("terminal_producers")
        if not isinstance(producers, list) or not producers:
            return False

        producer_ids: set[str] = set()
        required_inputs_by_producer: dict[str, set[str]] = {}
        for specification in producers:
            if (
                not isinstance(specification, dict)
                or not set(specification)
                <= {
                    "node_id",
                    "node_type",
                    "tool_name",
                    "requires_outputs",
                }
            ):
                return False
            node_id = specification.get("node_id")
            if (
                not isinstance(node_id, str)
                or node_id in producer_ids
                or node_id not in candidate.nodes
            ):
                return False
            node = candidate.nodes[node_id]
            expected_type = specification.get("node_type")
            if (
                expected_type is not None
                and (
                    not isinstance(expected_type, str)
                    or node.node_type.value != expected_type
                )
            ):
                return False
            expected_tool = specification.get("tool_name")
            if (
                expected_tool is not None
                and (
                    not isinstance(expected_tool, str)
                    or node.config.tool_name != expected_tool
                )
            ):
                return False
            required_outputs = specification.get("requires_outputs", [])
            if (
                not isinstance(required_outputs, list)
                or any(
                    not isinstance(required, str) or not required
                    for required in required_outputs
                )
                or len(required_outputs) != len(set(required_outputs))
            ):
                return False
            required_inputs_by_producer[node_id] = set(required_outputs)
            producer_ids.add(node_id)

        end_ids = {
            node_id
            for node_id, node in candidate.nodes.items()
            if node.node_type == NodeType.END
        }
        if not end_ids:
            return False
        terminal_predecessors = {
            source
            for source, target in candidate.edges
            if target in end_ids
        }
        if not (
            terminal_predecessors
            and terminal_predecessors == producer_ids
        ):
            return False
        try:
            dominators = nx.immediate_dominators(
                candidate.to_networkx(),
                candidate.entry_node,
            )
        except (KeyError, nx.NetworkXError):
            return False
        return all(
            required in candidate.nodes
            and CandidateGenerator._dominates(
                dominators,
                required,
                producer_id,
            )
            for producer_id, required_inputs in (
                required_inputs_by_producer.items()
            )
            for required in required_inputs
        )

    def _compute_edit_distance(
        self,
        candidate: WorkflowCandidate,
        workflow: WorkflowTemplate,
    ) -> float:
        """Return a normalized, scope-aware template edit distance."""
        if candidate.scope == MULTI_SCOPE:
            distances: list[float] = []
            for patch in candidate.changes.get("patches", []):
                patch_scope = _canonical_scope(patch.get("scope"))
                patch_node = patch.get("node_id", patch.get("anchor_id"))
                patch_candidate = WorkflowCandidate(
                    scope=patch_scope,
                    node_id=patch_node,
                    description=candidate.description,
                    changes=patch.get("changes", {}),
                )
                distances.append(
                    self._compute_edit_distance(patch_candidate, workflow)
                )
            return round(sum(distances), 6)

        if candidate.scope == "prompt":
            node = workflow.nodes[candidate.node_id]
            ratios = []
            if "system_prompt" in candidate.changes:
                ratios.append(
                    self._text_change_ratio(
                        node.config.system_prompt or "",
                        candidate.changes["system_prompt"],
                    )
                )
            if "user_template" in candidate.changes:
                ratios.append(
                    self._text_change_ratio(
                        node.config.prompt_template or "",
                        candidate.changes["user_template"],
                    )
                )
            magnitude = sum(ratios) / len(ratios) if ratios else 0.0
            return round(0.1 + 0.15 * magnitude, 6)

        if candidate.scope == "operator":
            changed_fields = len(candidate.changes)
            return round(min(0.45, 0.25 + 0.05 * changed_fields), 6)

        operation_weight = 0.0
        if "add_node" in candidate.changes:
            operation_weight += 0.05
        if "remove_node" in candidate.changes:
            operation_weight += 0.05
        if "reorder" in candidate.changes:
            operation_weight += min(
                0.2,
                0.025 * len(candidate.changes["reorder"]),
            )
        if "graph_patch" in candidate.changes:
            patch = candidate.changes["graph_patch"]
            changed_nodes = len(patch.get("add_nodes", [])) + len(
                patch.get("remove_nodes", [])
            )
            changed_edges = len(patch.get("add_edges", [])) + len(
                patch.get("remove_edges", [])
            )
            operation_weight += min(
                0.45,
                0.04 * changed_nodes + 0.015 * changed_edges,
            )
        return round(min(1.0, 0.4 + operation_weight), 6)

    @staticmethod
    def _text_change_ratio(before: str, after: str) -> float:
        return 1.0 - difflib.SequenceMatcher(None, before, after).ratio()

    @staticmethod
    def _changed_units(
        original: WorkflowTemplate,
        modified: WorkflowTemplate,
        candidate: WorkflowCandidate,
    ) -> list[str]:
        changed = {
            node_id
            for node_id in set(original.nodes) | set(modified.nodes)
            if original.nodes.get(node_id) != modified.nodes.get(node_id)
        }
        original_edges = set(original.edges)
        modified_edges = set(modified.edges)
        for source, target in original_edges ^ modified_edges:
            changed.update((source, target))
        for node_id in set(original.nodes) & set(modified.nodes):
            if (
                original.nodes[node_id].node_type == NodeType.CONDITION
                and original.get_successors(node_id)
                != modified.get_successors(node_id)
            ):
                changed.add(node_id)
                changed.update(original.get_successors(node_id))
                changed.update(modified.get_successors(node_id))
        changed.discard("")
        # Prompt/operator edits must never claim graph-neighbor changes.
        if candidate.scope in {"prompt", "operator"}:
            return [candidate.node_id] if candidate.node_id in changed else []
        return sorted(changed)

    def _allowed_units(
        self,
        workflow: WorkflowTemplate,
        anchor_id: str,
        scope: str,
    ) -> set[str]:
        if scope in {"prompt", "operator"}:
            return {anchor_id}
        if self.allow_cross_block_graph_updates:
            return {
                node_id
                for node_id, node in workflow.nodes.items()
                if node.node_type not in {NodeType.START, NodeType.END}
            }
        for stage in workflow.parameters.stages.values():
            for block in stage.blocks.values():
                if anchor_id in block.operators:
                    return set(block.operators)
        # Graph-derived local block fallback for workflows whose optional
        # hierarchy has not yet been populated.
        return {
            anchor_id,
            *(
                source
                for source, target in workflow.edges
                if target == anchor_id
            ),
            *(
                target
                for source, target in workflow.edges
                if source == anchor_id
            ),
        }

    @staticmethod
    def _sync_hierarchy(
        workflow: WorkflowTemplate,
        node_id: str,
        changes: dict,
        scope: str,
    ) -> None:
        for stage in workflow.parameters.stages.values():
            for block in stage.blocks.values():
                operator = block.operators.get(node_id)
                if operator is None:
                    continue
                if scope == "prompt":
                    if "system_prompt" in changes:
                        operator.prompt.system_prompt = changes["system_prompt"]
                    if "user_template" in changes:
                        operator.prompt.user_template = changes["user_template"]
                else:
                    for key, value in changes.items():
                        if hasattr(operator, key):
                            setattr(operator, key, copy.deepcopy(value))

    @staticmethod
    def _remove_from_hierarchy(
        workflow: WorkflowTemplate,
        node_id: str,
    ) -> None:
        for stage in workflow.parameters.stages.values():
            for block in stage.blocks.values():
                block.operators.pop(node_id, None)

    @staticmethod
    def _add_to_anchor_block(
        workflow: WorkflowTemplate,
        anchor_id: str,
        node: Node,
    ) -> None:
        for stage in workflow.parameters.stages.values():
            for block in stage.blocks.values():
                if anchor_id in block.operators:
                    block.operators[node.node_id] = OperatorParams(
                        node_id=node.node_id,
                        model=node.config.model,
                        temperature=node.config.temperature,
                        max_tokens=node.config.max_tokens,
                        tool_name=node.config.tool_name,
                        tool_args=copy.deepcopy(node.config.tool_args),
                        prompt=PromptParams(
                            system_prompt=node.config.system_prompt,
                            user_template=node.config.prompt_template,
                        ),
                    )
                    return

    @staticmethod
    def _deduplicate_edges(workflow: WorkflowTemplate) -> None:
        workflow.edges = list(dict.fromkeys(workflow.edges))

    @staticmethod
    def workflow_fingerprint(workflow: WorkflowTemplate) -> str:
        return json.dumps(
            workflow.model_dump(mode="python"),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    @staticmethod
    def candidate_patch_fingerprint(candidate: WorkflowCandidate) -> str:
        """Return a stable identity for an anchor-local edit across rounds."""
        payload = {
            "anchor_id": candidate.anchor_id,
            "node_id": candidate.node_id,
            "scope": candidate.scope,
            "changes": candidate.changes,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    @staticmethod
    def _anchor_id(anchor: dict) -> str:
        value = anchor.get("unit_id", anchor.get("node_id", ""))
        return value if isinstance(value, str) else ""

    @staticmethod
    def _format_anchors(anchors: list[dict]) -> str:
        lines = []
        for anchor in anchors[:5]:
            lines.append(
                f"  - unit_id={CandidateGenerator._anchor_id(anchor)}, "
                f"rank={anchor.get('rank', '?')}, "
                f"score={anchor.get('score', anchor.get('confidence', 0))}: "
                f"{anchor.get('reason', anchor.get('reasoning', ''))}"
            )
        return "\n".join(lines) if lines else "No anchors identified."

    @staticmethod
    def _format_anchor_detail(anchor: dict | str) -> str:
        if isinstance(anchor, str):
            return f"unit_id={anchor}; no structured judge evidence supplied."
        summary = {
            "unit_id": CandidateGenerator._anchor_id(anchor),
            "rank": anchor.get("rank"),
            "score": anchor.get("score", anchor.get("confidence")),
            "evidence_count": anchor.get("evidence_count"),
            "reason": str(
                anchor.get("reason", anchor.get("reasoning", ""))
            )[:1000],
        }
        return json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _summarize_workflow(self, workflow: WorkflowTemplate) -> str:
        lines = [
            f"Name: {workflow.name} (v{workflow.version})",
            f"Entry: {workflow.entry_node}",
            f"Edges: {workflow.edges}",
            f"Nodes ({len(workflow.nodes)}):",
        ]
        for node_id, node in workflow.nodes.items():
            lines.append(f"  - {node_id}: {node.node_type.value}")
            if node.node_type == NodeType.LLM:
                model = (
                    node.config.model
                    if node.config.model is not None
                    else self.execution_llm_defaults.get("model")
                )
                temperature = (
                    node.config.temperature
                    if node.config.temperature is not None
                    else self.execution_llm_defaults.get("temperature")
                )
                max_tokens = (
                    node.config.max_tokens
                    if node.config.max_tokens is not None
                    else self.execution_llm_defaults.get("max_tokens")
                )
                lines.append(
                    "    effective_execution: "
                    f"model={model!r}, temperature={temperature!r}, "
                    f"max_tokens={max_tokens!r}"
                )
            elif node.node_type == NodeType.TOOL:
                lines.append(
                    "    tool: "
                    f"name={node.config.tool_name!r}, "
                    f"args={node.config.tool_args!r}"
                )
            elif node.node_type == NodeType.CONDITION:
                lines.append(
                    f"    condition: {node.config.condition_expr!r}"
                )
            if node.config.prompt_template:
                lines.append(
                    f"    prompt: {node.config.prompt_template[:200]}"
                )
            if node.config.system_prompt:
                lines.append(
                    f"    system: {node.config.system_prompt[:200]}"
                )
        return "\n".join(lines)
