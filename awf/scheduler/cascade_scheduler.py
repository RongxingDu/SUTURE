"""A lightweight-gate/LLM cascade scheduler.

The implementation is intentionally small and research-oriented.  A
deterministic gate first scores the artifact produced by the last workflow
node.  Low-quality artifacts simply follow the graph, high-confidence
terminal artifacts early-exit, and only the intermediate band is sent to a
compact DeepSeek decision prompt.  This makes the outer layer measurable as
an independent token/latency intervention on top of the inner workflow
optimizer.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from awf.config.schema import SchedulerConfig
from awf.executor.context import ExecutionContext
from awf.llm.client import AsyncLLMClient
from awf.llm.cost_tracker import estimate_cost_usd, has_known_pricing
from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.trace.schema import LLMCallRecord
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import NodeType


_ERROR_PREFIX = re.compile(
    r"^\s*(?:error|exception|traceback|failed|failure|invalid)\b",
    re.IGNORECASE,
)
_FINAL_MARKER = re.compile(
    r"(?:final\s+answer|boxed\s*\{|\\boxed|answer\s*:|result\s*:|return\s+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_WORD = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class GateDecision:
    """The gate output attached to a scheduler step and execution trace."""

    route: str
    spec_score: float
    lite_score: float
    agreement_score: float
    history_reliability: float
    score: float
    risk: str
    reason: str
    artifact_node: str | None = None
    formula: str = "las"

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "spec_score": round(self.spec_score, 6),
            "lite_score": round(self.lite_score, 6),
            "agreement_score": round(self.agreement_score, 6),
            "history_reliability": round(self.history_reliability, 6),
            "score": round(self.score, 6),
            "risk": self.risk,
            "reason": self.reason,
            "artifact_node": self.artifact_node,
            "gate_formula": self.formula,
        }


class CascadeGate:
    """Deterministic, zero-generation-call gate.

    The four features mirror the paper's gate interface: specification
    adherence, a cheap local quality score, agreement with previous artifacts,
    and an exponentially updated per-node reliability.  They deliberately do
    not inspect labels/rewards, so the gate can also be used during held-out
    execution.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._reliability: dict[str, float] = {}
        self._invocations = 0
        self._route_counts: dict[str, int] = {
            "continue": 0,
            "invoke_scheduler": 0,
            "early_exit": 0,
        }

    def reset_query(self) -> None:
        """Reset per-query counters while retaining historical reliability."""
        self._invocations = 0
        self._route_counts = {
            "continue": 0,
            "invoke_scheduler": 0,
            "early_exit": 0,
        }

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value).strip()

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.lower() for token in _WORD.findall(value)}

    def _risk(self, query: str) -> str:
        # Query-only risk features keep the gate independent of ground-truth
        # labels and rewards.  Numeric-heavy and proof/code prompts are more
        # likely to benefit from an explicit verification route.
        numeric_count = len(_NUMBER.findall(query))
        words = len(_WORD.findall(query))
        lowered = query.lower()
        if (
            len(query) >= 500
            or words >= 110
            or numeric_count >= 5
            or any(term in lowered for term in ("prove", "proof", "derive"))
        ):
            return "high"
        if len(query) >= 220 or words >= 50 or numeric_count >= 3:
            return "medium"
        return "low"

    def _spec_score(self, artifact: str, context: ExecutionContext) -> float:
        if not artifact or context.error_message or _ERROR_PREFIX.search(artifact):
            return 0.0
        # A structural START/condition payload is not an answer artifact.
        if artifact.startswith('{"status": "started"'):
            return 0.0
        return 1.0

    def _lite_score(self, artifact: str, spec_score: float) -> float:
        if not spec_score:
            return 0.0
        score = 0.30
        if len(artifact) >= 20:
            score += 0.20
        elif len(artifact) >= 4:
            score += 0.10
        if _FINAL_MARKER.search(artifact):
            score += 0.30
        if _NUMBER.search(artifact) or len(self._tokens(artifact)) >= 8:
            score += 0.15
        if artifact.rstrip().endswith((".", "!", "?", "]", "}")):
            score += 0.05
        return min(score, 1.0)

    def _agreement_score(
        self,
        artifact: str,
        context: ExecutionContext,
        artifact_node: str | None,
    ) -> float:
        if not artifact:
            return 0.0
        previous: list[str] = []
        for node_id, value in context.outputs.items():
            if node_id == artifact_node:
                continue
            node_text = self._text(value)
            if not node_text or node_text.startswith('{"status": "started"'):
                continue
            previous.append(node_text)
        # LAS treats an artifact with no peer as locally consistent: there is
        # no disagreement evidence yet.  The legacy 0.5 prior remains useful
        # only for the explicit legacy formula below.
        if not previous:
            return 1.0 if self.config.gate_formula == "las" else 0.5
        current_tokens = self._tokens(artifact)
        if not current_tokens:
            return 0.0
        best = 0.0
        for candidate in previous:
            candidate_tokens = self._tokens(candidate)
            if not candidate_tokens:
                continue
            union = current_tokens | candidate_tokens
            jaccard = len(current_tokens & candidate_tokens) / len(union)
            best = max(best, jaccard)
            # Exact final-answer markers are a stronger local agreement cue
            # than prose overlap.
            current_marker = _FINAL_MARKER.search(artifact)
            candidate_marker = _FINAL_MARKER.search(candidate)
            if current_marker and candidate_marker:
                best = max(best, 0.85)
        return min(best, 1.0)

    def _terminal_candidate(
        self,
        workflow: WorkflowTemplate,
        current_node_id: str | None,
    ) -> bool:
        if not current_node_id or current_node_id not in workflow.nodes:
            return False
        node = workflow.nodes[current_node_id]
        if node.node_type == NodeType.END:
            return True
        if not workflow.get_successors(current_node_id):
            return True
        role = str(node.config.metadata.get("role", ""))
        haystack = f"{current_node_id} {node.label} {role}".lower()
        return any(term in haystack for term in ("final", "verify", "check", "format"))

    def evaluate(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> GateDecision:
        """Score the latest artifact and choose one of three routes."""
        self._invocations += 1
        current_node_id = context.current_node_id
        current_node = workflow.nodes.get(current_node_id or "")
        artifact_node = context.history[-1] if context.history else context.previous_node_id
        raw_artifact = context.get_last_output()
        # Checkpoint/suffix contexts restore ``outputs`` and ``history`` but
        # intentionally do not rebuild the private output-history list.
        if raw_artifact is None and artifact_node:
            raw_artifact = context.outputs.get(artifact_node)
        artifact = self._text(raw_artifact)
        risk = self._risk(context.query)

        allowed_nodes = set(self.config.gate_node_allowlist)
        if allowed_nodes and current_node_id not in allowed_nodes:
            decision = GateDecision(
                route="continue",
                spec_score=0.0,
                lite_score=0.0,
                agreement_score=0.0,
                history_reliability=0.0,
                score=0.0,
                risk=risk,
                reason="outside_intervention_node_allowlist",
                artifact_node=artifact_node,
                formula=self.config.gate_formula,
            )
            self._route_counts[decision.route] += 1
            return decision

        # Structural nodes must be traversed so the graph can establish its
        # next data dependency.  There is no meaningful artifact to schedule.
        if current_node is None or current_node.node_type in {
            NodeType.START,
            NodeType.CONDITION,
            NodeType.JOIN,
        } or not artifact:
            decision = GateDecision(
                route="continue",
                spec_score=0.0,
                lite_score=0.0,
                agreement_score=0.0,
                history_reliability=(
                    0.0 if self.config.gate_formula == "las" else 0.5
                ),
                score=0.0,
                risk=risk,
                reason="structural_or_missing_artifact",
                artifact_node=artifact_node,
                formula=self.config.gate_formula,
            )
            self._route_counts[decision.route] += 1
            return decision

        spec = self._spec_score(artifact, context)
        lite = self._lite_score(artifact, spec)
        agreement = self._agreement_score(artifact, context, artifact_node)
        # An unseen LAS node has no successful history yet, so initialize its
        # reliability at zero (maximal historical-risk term).  The legacy
        # formula retains the neutral 0.5 prior used by the original gate.
        reliability_prior = 0.0 if self.config.gate_formula == "las" else 0.5
        reliability = self._reliability.get(artifact_node or "", reliability_prior)
        weights = (
            self.config.gate_spec_weight,
            self.config.gate_lite_weight,
            self.config.gate_agreement_weight,
            self.config.gate_history_weight,
        )
        weight_sum = sum(weights)
        if weight_sum <= 0:
            weights = (0.0, 1.0, 0.0, 0.0)
            weight_sum = 1.0
        if self.config.gate_formula == "las":
            # The LAS paper's gate is a weighted confidence/risk score:
            # specification adherence and the compact judge are positive
            # evidence, while disagreement and low historical reliability are
            # risk terms.  Keep the same four observable features in the
            # telemetry so this remains an exact, label-free implementation.
            score = (
                weights[0] * spec
                + weights[1] * lite
                + weights[2] * (agreement - 1.0)
                + weights[3] * (1.0 - reliability)
            ) / weight_sum
        else:
            score = sum(value * weight for value, weight in zip(
                (spec, lite, agreement, reliability), weights
            )) / weight_sum
        if artifact_node:
            self._reliability[artifact_node] = (
                0.8 * reliability + 0.2 * lite
            )

        threshold = self.config.gate_early_exit_threshold
        if risk == "high":
            threshold = max(threshold, self.config.gate_high_risk_threshold)
        terminal = self._terminal_candidate(workflow, current_node_id)
        if (
            self.config.gate_direct_early_exit
            and self.config.gate_enabled
            and self.config.early_exit_enabled
            and spec >= 1.0
            and score >= threshold
            and terminal
        ):
            route = "early_exit"
            reason = "high_confidence_terminal_artifact"
        elif not self.config.gate_enabled or score >= self.config.gate_schedule_threshold:
            route = "invoke_scheduler"
            reason = (
                "las_promising_artifact_forwarded_to_scheduler"
                if self.config.gate_formula == "las"
                else "borderline_artifact_requires_outer_policy"
            )
        else:
            route = "continue"
            reason = "low_confidence_follow_template"
        decision = GateDecision(
            route=route,
            spec_score=spec,
            lite_score=lite,
            agreement_score=agreement,
            history_reliability=reliability,
            score=score,
            risk=risk,
            reason=reason,
            artifact_node=artifact_node,
            formula=self.config.gate_formula,
        )
        self._route_counts[route] += 1
        return decision

    def telemetry(self) -> dict[str, Any]:
        return {
            "gate_formula": self.config.gate_formula,
            "gate_schedule_threshold": self.config.gate_schedule_threshold,
            "gate_direct_early_exit": self.config.gate_direct_early_exit,
            "gate_invocations": self._invocations,
            "gate_route_counts": dict(self._route_counts),
            "historical_reliability": dict(self._reliability),
        }


class CascadeScheduler(BaseScheduler):
    """Gate + Language-model Scheduler (LAS) outer execution policy."""

    def __init__(
        self,
        config: SchedulerConfig,
        llm_client: Optional[AsyncLLMClient] = None,
    ):
        self.config = config
        self.llm = llm_client or AsyncLLMClient(config.llm)
        self.gate = CascadeGate(config)
        self._last_llm_call: LLMCallRecord | None = None
        self._last_gate_decision: GateDecision | None = None
        self._scheduler_invocations = 0
        self._scheduler_fallbacks = 0
        self._gate_latency_seconds = 0.0

    async def initialize(
        self,
        workflow: WorkflowTemplate,
        query: str,
    ) -> None:
        self.gate.reset_query()
        self._last_llm_call = None
        self._last_gate_decision = None
        self._scheduler_invocations = 0
        self._scheduler_fallbacks = 0
        self._gate_latency_seconds = 0.0

    async def skip_to_node(
        self,
        workflow: WorkflowTemplate,
        node_id: str,
    ) -> None:
        # The scheduler is graph-state based; checkpoint replay starts with the
        # supplied context node and does not need an index adjustment.
        return None

    @staticmethod
    def _compact(value: Any, limit: int) -> str:
        text = CascadeGate._text(value)
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)] + "..."

    def _catalog(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> list[dict[str, str]]:
        order = workflow.get_node_order()
        rows: list[dict[str, str]] = []
        for node_id in order:
            if node_id in context.history or node_id == context.current_node_id:
                continue
            node = workflow.nodes[node_id]
            rows.append({
                "id": node_id,
                "type": node.node_type.value,
                "label": self._compact(node.label, 80),
            })
        return rows[:24]

    def _prompt(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
        decision: GateDecision,
    ) -> tuple[str, str]:
        current = workflow.nodes.get(context.current_node_id or "")
        latest_node = context.history[-1] if context.history else None
        latest_output = context.get_last_output()
        if latest_output is None and latest_node:
            latest_output = context.outputs.get(latest_node)
        current_info = {
            "id": context.current_node_id,
            "type": current.node_type.value if current else "unknown",
            "label": self._compact(current.label, 80) if current else "",
            "successors": workflow.get_successors(context.current_node_id)
            if context.current_node_id
            else [],
        }
        system_prompt = self.config.system_prompt.strip() or (
            "You are a compact workflow scheduler. Choose only a machine "
            "readable action that preserves required data dependencies. "
            "Return JSON with action in {continue, early_exit, verify, "
            "reroute, repair, fallback} and optional target_node."
        )
        payload = {
            "query": self._compact(context.query, self.config.scheduler_query_max_chars),
            "risk": decision.risk,
            "current": current_info,
            "artifact": self._compact(
                latest_output,
                self.config.scheduler_artifact_max_chars,
            ),
            "gate": decision.as_dict(),
            "remaining_nodes": self._catalog(workflow, context),
            "allowed_actions": [
                "continue",
                "early_exit",
                "verify",
                "reroute",
                "repair",
                "fallback",
            ],
        }
        return system_prompt, json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _decode(response: Any) -> tuple[dict[str, Any], str]:
        if isinstance(response, Mapping):
            payload = dict(response)
            return payload, json.dumps(payload, ensure_ascii=False)
        text = str(response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return {}, text
        return (dict(payload) if isinstance(payload, Mapping) else {}), text

    def _map_decision(
        self,
        payload: dict[str, Any],
        workflow: WorkflowTemplate,
        context: ExecutionContext,
        gate: GateDecision,
    ) -> tuple[SchedulerAction, dict[str, Any]]:
        raw_action = payload.get("action", payload.get("route", "continue"))
        action = str(raw_action).strip().lower().replace("-", "_")
        target = (
            payload.get("target_node")
            or payload.get("next_node")
            or payload.get("next_unit_id")
            or payload.get("node_id")
        )
        if target is not None:
            target = str(target)
        params: dict[str, Any] = {
            "_awf_gate": gate.as_dict(),
            "scheduler_decision": action,
        }
        if action in {"stop", "early_exit", "exit"}:
            # A high-risk answer below the stricter threshold must still pass
            # through the graph (usually a verifier) despite an optimistic
            # scheduler response.
            if gate.risk == "high" and gate.score < self.config.gate_high_risk_threshold:
                params["risk_override"] = "high_risk_requires_verification"
                return SchedulerAction.CONTINUE, params
            return SchedulerAction.EARLY_EXIT, params
        if action in {"continue", "execute", "next"}:
            return SchedulerAction.CONTINUE, params
        if target is None or target not in workflow.nodes:
            self._scheduler_fallbacks += 1
            params["fallback_reason"] = "missing_or_unknown_target"
            return SchedulerAction.CONTINUE, params
        if target == context.current_node_id or workflow.nodes[target].node_type == NodeType.START:
            self._scheduler_fallbacks += 1
            params["fallback_reason"] = "non_progress_target"
            return SchedulerAction.CONTINUE, params
        if not self.config.allow_deviation:
            self._scheduler_fallbacks += 1
            params["fallback_reason"] = "deviation_disabled"
            return SchedulerAction.CONTINUE, params
        params["target_node"] = target
        if action in {"verify", "test", "verification"}:
            return SchedulerAction.VERIFY, params
        if action in {"repair", "refine", "refinement"}:
            return SchedulerAction.REPAIR, params
        if action in {"fallback"}:
            return SchedulerAction.FALLBACK, params
        if action in {"reroute", "branch", "route"}:
            return SchedulerAction.REROUTE, params
        self._scheduler_fallbacks += 1
        params["fallback_reason"] = "unknown_action"
        return SchedulerAction.CONTINUE, params

    async def select_action(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> tuple[SchedulerAction, dict[str, Any]]:
        self._last_llm_call = None
        gate_started = time.perf_counter()
        decision = self.gate.evaluate(workflow, context)
        self._gate_latency_seconds += time.perf_counter() - gate_started
        self._last_gate_decision = decision
        if decision.route == "early_exit":
            return SchedulerAction.EARLY_EXIT, {"_awf_gate": decision.as_dict()}
        if decision.route == "continue":
            return SchedulerAction.CONTINUE, {"_awf_gate": decision.as_dict()}

        self._scheduler_invocations += 1
        system_prompt, user_prompt = self._prompt(workflow, context, decision)
        model = str(getattr(getattr(self.llm, "config", None), "model", ""))
        call_started = time.perf_counter()
        usage: dict[str, Any] = {}
        response: Any = ""
        try:
            response, raw_usage = await self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=self.config.scheduler_max_tokens,
            )
            usage = dict(raw_usage or {})
            latency = float(usage.get("latency_seconds", 0.0) or 0.0)
            if latency <= 0:
                latency = time.perf_counter() - call_started
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
                or prompt_tokens + completion_tokens
            )
            explicit_cost = usage.get("cost_usd") is not None
            self._last_llm_call = LLMCallRecord(
                call_id="scheduler_pending",
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=str(response),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                cost_usd=float(
                    usage.get(
                        "cost_usd",
                        estimate_cost_usd(
                            model,
                            prompt_tokens,
                            completion_tokens,
                            prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
                            prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens"),
                        ),
                    )
                    or 0.0
                ),
                cost_estimate_available=bool(
                    usage.get("cost_estimate_available", explicit_cost or has_known_pricing(model))
                ),
                call_type="scheduler",
                metadata={
                    "request": {
                        "model": model,
                        "temperature": 0.0,
                        "max_tokens": self.config.scheduler_max_tokens,
                    },
                    "usage": usage,
                    "gate": decision.as_dict(),
                },
            )
            payload, _ = self._decode(response)
            return self._map_decision(payload, workflow, context, decision)
        except BaseException as exc:
            self._scheduler_fallbacks += 1
            self._last_llm_call = LLMCallRecord(
                call_id="scheduler_pending",
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=str(response or ""),
                latency_seconds=time.perf_counter() - call_started,
                cost_estimate_available=False,
                call_type="scheduler",
                metadata={"gate": decision.as_dict(), "usage": usage},
                success=False,
                error_message=str(exc),
            )
            return SchedulerAction.CONTINUE, {
                "_awf_gate": decision.as_dict(),
                "fallback_reason": "scheduler_exception",
            }

    def pop_last_llm_call(self) -> LLMCallRecord | None:
        call = self._last_llm_call
        self._last_llm_call = None
        return call

    def telemetry(self) -> dict[str, Any]:
        payload = self.gate.telemetry()
        payload.update({
            "scheduler_invocations": self._scheduler_invocations,
            "scheduler_fallbacks": self._scheduler_fallbacks,
            "gate_latency_seconds": self._gate_latency_seconds,
            "scheduler_model": str(getattr(getattr(self.llm, "config", None), "model", "")),
        })
        if self._last_gate_decision is not None:
            payload["last_gate_decision"] = self._last_gate_decision.as_dict()
        return payload
