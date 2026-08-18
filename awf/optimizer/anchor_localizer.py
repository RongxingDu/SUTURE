"""LLM judge for rank-weighted suspicious-unit localization."""

from __future__ import annotations

import json
from typing import Any

from awf.llm.client import AsyncLLMClient
from awf.trace.schema import ExecutionTrace
from awf.workflow.ir import WorkflowTemplate

_ANCHOR_PROMPT = """You are a workflow failure-localization judge.
Identify only the executed workflow units most likely related to the final
failure. Do not choose an edit scope and do not propose a fix.

Return exactly one JSON object with this shape:
{
  "anchors": [
    {
      "unit_id": "<existing executed node id>",
      "rank": 1,
      "reason": "<short evidence-based explanation>"
    }
  ]
}

Ranks must be unique positive integers (1 is most suspicious). Return at most
five anchors. Do not include prompt/operator/block scope decisions.
"""


class AnchorLocalizer:
    """Locate suspicious executed units and aggregate them with rank weights."""

    def __init__(self, llm: AsyncLLMClient, top_m: int = 5):
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        self.llm = llm
        self.top_m = top_m

    async def localize(
        self,
        failure_trace: ExecutionTrace,
        workflow: WorkflowTemplate,
    ) -> list[dict]:
        """Identify suspicious units in one failed execution trace."""
        trace_summary = self._summarize_trace(failure_trace, workflow)
        hierarchy = self._summarize_hierarchy(workflow)

        user_prompt = f"""Workflow: {workflow.name} (v{workflow.version})

Workflow hierarchy:
{hierarchy}

Execution and failure evidence:
{trace_summary}

Identify suspicious executed units. Respond with JSON only."""

        response, _ = await self.llm.generate_json(
            system_prompt=_ANCHOR_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return self._parse_anchors(response, failure_trace, workflow)

    async def localize_batch(
        self,
        failure_traces: list[ExecutionTrace],
        workflow: WorkflowTemplate,
        top_m: int | None = None,
    ) -> list[dict]:
        """Aggregate anchors using ``1, 1/2, 1/4, ...`` rank weights."""
        if not failure_traces:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        for trace in failure_traces:
            anchors = await self.localize(trace, workflow)
            # Count a unit once per trace, using its best reported rank.
            per_trace: dict[str, dict] = {}
            for anchor in anchors:
                node_id = anchor["node_id"]
                if (
                    node_id not in per_trace
                    or anchor["rank"] < per_trace[node_id]["rank"]
                ):
                    per_trace[node_id] = anchor

            for anchor in per_trace.values():
                node_id = anchor["node_id"]
                weight = 0.5 ** (anchor["rank"] - 1)
                record = aggregated.setdefault(
                    node_id,
                    {
                        "unit_id": node_id,
                        "node_id": node_id,
                        "score": 0.0,
                        "evidence_count": 0,
                        "reasons": [],
                    },
                )
                record["score"] += weight
                record["evidence_count"] += 1
                record["reasons"].append(anchor["reason"])

        ranked = sorted(
            aggregated.values(),
            key=lambda item: (
                -item["score"],
                -item["evidence_count"],
                item["node_id"],
            ),
        )
        for rank, anchor in enumerate(ranked, start=1):
            anchor["rank"] = rank
            anchor["reason"] = "; ".join(anchor.pop("reasons")[:3])
            # Compatibility for consumers that previously displayed confidence.
            anchor["confidence"] = anchor["score"] / len(failure_traces)

        limit = self.top_m if top_m is None else top_m
        if limit <= 0:
            return []
        return ranked[:limit]

    def _parse_anchors(
        self,
        response: str,
        trace: ExecutionTrace,
        workflow: WorkflowTemplate,
    ) -> list[dict]:
        """Parse and validate judge output without trusting LLM-provided IDs."""
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, dict) or not isinstance(data.get("anchors"), list):
            return []

        executed = {step.node_id for step in trace.steps}
        parsed: list[dict] = []
        used_ranks: set[int] = set()
        for position, raw in enumerate(data["anchors"][: self.top_m], start=1):
            if not isinstance(raw, dict):
                continue
            node_id = raw.get("unit_id", raw.get("node_id"))
            if (
                not isinstance(node_id, str)
                or node_id not in workflow.nodes
                or node_id not in executed
            ):
                continue

            rank_value = raw.get("rank", position)
            if isinstance(rank_value, bool):
                continue
            if isinstance(rank_value, int):
                rank = rank_value
            elif isinstance(rank_value, str) and rank_value.isdigit():
                rank = int(rank_value)
            else:
                continue
            if rank < 1 or rank > self.top_m or rank in used_ranks:
                continue

            reason = raw.get("reason", raw.get("reasoning", ""))
            if not isinstance(reason, str) or not reason.strip():
                continue

            anchor = {
                "unit_id": node_id,
                "node_id": node_id,
                "rank": rank,
                "reason": reason.strip()[:1000],
            }
            # Preserve old mock payloads for API compatibility only. New judge
            # prompts never request or use this value to choose an edit scope.
            if raw.get("scope") in {"prompt", "operator", "block"}:
                anchor["scope"] = raw["scope"]
            parsed.append(anchor)
            used_ranks.add(rank)

        return sorted(parsed, key=lambda item: item["rank"])

    def _summarize_trace(
        self,
        trace: ExecutionTrace,
        workflow: WorkflowTemplate,
    ) -> str:
        """Include final outcome, step state, and intermediate outputs."""
        del workflow  # Included in the hierarchy summary supplied separately.
        lines = [
            f"Query: {self._preview(trace.query_text, 400)}",
            f"Final output: {self._preview(trace.final_output, 600)}",
            f"Success: {trace.success}",
            f"Hard reward: {trace.hard_reward}",
            f"Process reward: {trace.process_reward}",
        ]
        errors = [trace.error_message] if trace.error_message else []
        errors.extend(
            step.error_message
            for step in trace.steps
            if step.error_message
        )
        metadata_error = trace.metadata.get("error_message")
        if metadata_error:
            errors.append(str(metadata_error))
        if errors:
            lines.append(f"Failure signal: {self._preview(errors, 500)}")

        for step in trace.steps:
            lines.extend(
                [
                    f"\nStep {step.step_index} ({step.node_id}, {step.node_type}):",
                    f"  action={step.action} success={step.success}",
                    f"  input_state={self._preview(step.state_before, 350)}",
                    f"  output_state={self._preview(step.state_after, 350)}",
                ]
            )
            if step.error_message:
                lines.append(f"  error={self._preview(step.error_message, 250)}")
            for llm_call in step.llm_calls[-2:]:
                lines.append(
                    f"  llm_response={self._preview(llm_call.response_text, 300)}"
                )
            for tool_call in step.tool_calls[-2:]:
                lines.append(
                    f"  tool_result={self._preview(tool_call.tool_result, 300)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _summarize_hierarchy(workflow: WorkflowTemplate) -> str:
        hierarchy = workflow.parameters.model_dump(mode="python")
        if hierarchy.get("stages"):
            return AnchorLocalizer._preview(hierarchy, 2000)

        # Empty hierarchy is common in early workflows. Give the judge an
        # explicit graph-derived fallback rather than hiding the missing data.
        graph_units = {
            node_id: {
                "type": node.node_type.value,
                "predecessors": [
                    src for src, dst in workflow.edges if dst == node_id
                ],
                "successors": [
                    dst for src, dst in workflow.edges if src == node_id
                ],
            }
            for node_id, node in workflow.nodes.items()
        }
        return "No parameter stages declared; graph fallback: " + (
            AnchorLocalizer._preview(graph_units, 2000)
        )

    @staticmethod
    def _preview(value: Any, limit: int) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."
