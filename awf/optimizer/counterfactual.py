"""Counterfactual utility evaluation with an explicit full-rerun backend."""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Any, Optional

from awf.executor.context import ExecutionContext
from awf.llm.client import AsyncLLMClient
from awf.optimizer.suffix_replay import SuffixReplayEngine
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.utility.compute import UtilityComputer
from awf.workflow.gates import (
    GateSpec,
    evaluate_gate,
    extract_pre_execution_features,
)
from awf.workflow.ir import WorkflowTemplate

logger = logging.getLogger(__name__)


class CounterfactualEvaluator:
    """Compare candidates against one precomputed baseline trace batch.

    When an edit has a preceding checkpoint in the baseline trace, the runtime
    resumes from that state and re-executes only the changed suffix.  Candidates
    that edit the first node (or whose trace lacks a compatible checkpoint)
    explicitly fall back to a full rerun and report that mode in the result.
    """

    def __init__(
        self,
        reward_evaluator: RewardEvaluator,
        utility_computer: UtilityComputer,
        suffix_replay: Optional[SuffixReplayEngine] = None,
        reward_alpha: float = 0.8,
    ):
        if reward_alpha < 0:
            raise ValueError("reward_alpha cannot be negative")
        self.reward_evaluator = reward_evaluator
        self.utility_computer = utility_computer
        self.suffix_replay = suffix_replay
        # Failure-triggered inner updates may require suffix replay as a hard
        # correctness condition. Full-rerun fallback remains useful for
        # diagnostics, but is not promotion-eligible in that mode.
        self.require_suffix_replay = False
        self.reward_alpha = reward_alpha
        self._result_cache: dict[str, dict] = {}
        # Default evaluation mode; the per-evaluate result dict carries the
        # actual mode used during that specific call.
        self.evaluation_mode: str = "full_rerun"

    def clear_cache(self) -> None:
        """Drop per-round candidate results."""
        self._result_cache.clear()

    def prepare_baseline(
        self,
        traces: list[ExecutionTrace],
    ) -> dict[str, Any]:
        """Compute original rewards/utilities once for all candidates."""
        entries: list[dict[str, Any]] = []
        for trace in traces:
            query = trace.query_text
            ground_truth = trace.metadata.get("ground_truth")
            hard = trace.hard_reward
            process = trace.process_reward
            if hard is None:
                hard = self.reward_evaluator.hard_reward(
                    query,
                    ground_truth,
                    trace.final_output,
                    trace,
                )
            if process is None:
                process = self.reward_evaluator.process_reward(
                    query,
                    ground_truth,
                    trace.final_output,
                    trace,
                )
            reward = self._combine(float(hard), float(process))
            entries.append(
                {
                    "trace_id": trace.trace_id,
                    "query": query,
                    "hard": float(hard),
                    "process": float(process),
                    "reward": reward,
                    # Utility is intentionally based on hard reward only.
                    # ``reward`` remains the composite value for reporting
                    # process-reward effects, but it must not leak into the
                    # scalar workflow objective.
                    "utility": self.utility_computer.compute(
                        float(hard),
                        trace,
                    ),
                    "diagnostics": self._trace_diagnostics(trace),
                    "gate_features": extract_pre_execution_features(query),
                }
            )
        return {
            "entries": entries,
            "batch_signature": self._batch_signature(traces),
            "reward_alpha": self.reward_alpha,
        }

    async def evaluate(
        self,
        candidate_workflow: WorkflowTemplate,
        failure_traces: list[ExecutionTrace],
        executor: object,
        scheduler: object,
        llm_client: Optional[AsyncLLMClient] = None,
        *,
        baseline: dict[str, Any] | None = None,
        edit_node_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict:
        """Evaluate a candidate workflow against the failure-heavy batch.

        When ``self.suffix_replay`` is available and ``edit_node_id`` is
        supplied, only the suffix from that node onward is re-executed;
        prefix results are reconstructed from the baseline trace.
        """
        traces = failure_traces
        baseline_data = baseline or self.prepare_baseline(traces)
        self._validate_baseline(baseline_data, traces)

        cache_key = self._cache_key(
            candidate_workflow,
            baseline_data["batch_signature"],
            scheduler,
            llm_client,
            edit_node_id=edit_node_id,
        )
        cached = self._result_cache.get(cache_key) if use_cache else None
        if cached is not None:
            result = copy.deepcopy(cached)
            result["cache_hit"] = True
            return result

        per_query: list[dict[str, Any]] = []
        suffix_replay_queries: int = 0
        suffix_replay_trace_ids: list[str] = []
        full_rerun_trace_ids: list[str] = []
        for index, trace in enumerate(traces):
            query = trace.query_text
            ground_truth = trace.metadata.get("ground_truth")

            # --- try suffix replay ---
            checkpoint: ExecutionContext | None = None
            checkpoint_reason: str | None = None
            checkpoint_step_index: int | None = None
            if (
                self.suffix_replay is not None
                and edit_node_id is not None
                and hasattr(executor, "execute_from_checkpoint")
            ):
                checkpoint, checkpoint_reason, checkpoint_step_index = (
                    self._build_suffix_checkpoint(
                        trace,
                        query,
                        candidate_workflow,
                        edit_node_id,
                    )
                )
            elif edit_node_id is None:
                checkpoint_reason = "missing_edit_node"
            elif self.suffix_replay is None:
                checkpoint_reason = "suffix_replay_disabled"
            else:
                checkpoint_reason = "executor_checkpoint_resume_unavailable"

            if checkpoint is not None:
                output, _, new_trace_recorder = await executor.execute_from_checkpoint(
                    checkpoint,
                    edit_node_id,
                    candidate_workflow,
                    scheduler,
                    llm_client=llm_client,
                    counterfactual=True,
                )
                suffix_replay_queries += 1
                suffix_replay_trace_ids.append(trace.trace_id)
            elif llm_client is None:
                full_rerun_trace_ids.append(trace.trace_id)
                output, _, new_trace_recorder = await executor.execute(
                    candidate_workflow,
                    scheduler,
                    query,
                    counterfactual=True,
                )
            else:
                full_rerun_trace_ids.append(trace.trace_id)
                output, _, new_trace_recorder = await executor.execute(
                    candidate_workflow,
                    scheduler,
                    query,
                    llm_client=llm_client,
                    counterfactual=True,
                )
            new_trace = new_trace_recorder.trace
            if checkpoint is not None:
                # ``execute_from_checkpoint`` seeds aggregate token/cost
                # counters from the checkpoint.  Preserve the exact prefix
                # LLM latency too, because UtilityComputer otherwise sees
                # only suffix calls and overestimates the candidate utility
                # when lambda_latency is non-zero.
                prefix_nodes = set(checkpoint.history)
                prefix_latency = sum(
                    max(float(call.latency_seconds), 0.0)
                    for step in trace.steps
                    if step.node_id in prefix_nodes
                    for call in step.llm_calls
                )
                new_trace.metadata[
                    "suffix_replay_prefix_llm_latency_seconds"
                ] = prefix_latency
            new_trace.metadata["ground_truth"] = ground_truth

            # Evaluate each reward component exactly once. Calling
            # combined_reward here would invoke both functions a second time.
            new_hard = float(
                self.reward_evaluator.hard_reward(
                    query,
                    ground_truth,
                    output,
                    new_trace,
                )
            )
            new_process = float(
                self.reward_evaluator.process_reward(
                    query,
                    ground_truth,
                    output,
                    new_trace,
                )
            )
            new_trace.hard_reward = new_hard
            new_trace.process_reward = new_process
            new_reward = self._combine(new_hard, new_process)

            original = baseline_data["entries"][index]
            original_u = original["utility"]
            # Keep process/composite reward in the row for diagnostics, while
            # comparing candidates on the reward-only utility.
            candidate_u = self.utility_computer.compute(
                new_hard,
                new_trace,
            )
            original_diagnostics = original.get(
                "diagnostics",
                self._trace_diagnostics(trace),
            )
            candidate_diagnostics = self._trace_diagnostics(new_trace)
            per_query.append(
                {
                    "query": query[:100],
                    "trace_id": trace.trace_id,
                    "candidate_trace_id": new_trace.trace_id,
                    "original_hard": original["hard"],
                    "candidate_hard": new_hard,
                    "original_process": original["process"],
                    "candidate_process": new_process,
                    "original_reward": original["reward"],
                    "candidate_reward": new_reward,
                    "original_u": original_u,
                    "candidate_u": candidate_u,
                    "delta_u": candidate_u - original_u,
                    "gate_features": copy.deepcopy(
                        original["gate_features"]
                    ),
                    **{
                        f"original_{key}": value
                        for key, value in original_diagnostics.items()
                    },
                    **{
                        f"candidate_{key}": value
                        for key, value in candidate_diagnostics.items()
                    },
                    "candidate_suffix_replay_used": checkpoint is not None,
                    "candidate_suffix_replay_reason": checkpoint_reason,
                    "candidate_suffix_checkpoint_step_index": (
                        checkpoint_step_index
                    ),
                    "candidate_evaluation_mode": (
                        "suffix_replay"
                        if checkpoint is not None
                        else "full_rerun"
                    ),
                }
            )

        count = len(per_query)
        suffix_replay_used = bool(
            suffix_replay_queries > 0 and suffix_replay_queries == count
        )
        evaluation_mode = (
            "suffix_replay" if suffix_replay_used else "full_rerun"
        )
        token_deltas: list[float] = []
        for item in per_query:
            try:
                token_deltas.append(
                    float(item["candidate_total_tokens"])
                    - float(item["original_total_tokens"])
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        result = {
            "delta_u": (
                sum(item["delta_u"] for item in per_query) / count
                if count
                else 0.0
            ),
            "original_u": (
                sum(item["original_u"] for item in per_query) / count
                if count
                else 0.0
            ),
            "candidate_u": (
                sum(item["candidate_u"] for item in per_query) / count
                if count
                else 0.0
            ),
            "mean_token_delta": (
                sum(token_deltas) / len(token_deltas)
                if token_deltas
                else 0.0
            ),
            "per_query_results": per_query,
            "num_queries": count,
            "evaluation_mode": evaluation_mode,
            "suffix_replay_requested": self.suffix_replay is not None,
            "suffix_replay_used": suffix_replay_used,
            "suffix_replay_trace_ids": suffix_replay_trace_ids,
            "full_rerun_trace_ids": full_rerun_trace_ids,
            "suffix_replay_query_count": suffix_replay_queries,
            "require_suffix_replay": bool(self.require_suffix_replay),
            "cache_hit": False,
        }
        if self.suffix_replay is not None and not suffix_replay_used:
            result["fallback_reason"] = (
                "suffix replay could not be used for all queries "
                f"(used for {suffix_replay_queries}/{count})"
            )
        if use_cache:
            self._result_cache[cache_key] = copy.deepcopy(result)
        return result

    @staticmethod
    def _build_suffix_checkpoint(
        trace: ExecutionTrace,
        query: str,
        candidate_workflow: WorkflowTemplate,
        edit_node_id: str,
    ) -> tuple[ExecutionContext | None, str | None, int | None]:
        """Build a checkpoint only from a completed, actual trace prefix.

        The previous implementation inferred the prefix from a topological
        predecessor and accepted any matching ``state_after`` snapshot. That
        can reuse the state of a failed predecessor (or a different branch / a
        later loop visit), which is not strict suffix replay. The edited node
        must occur in the observed execution, every earlier step must have
        completed successfully, and the snapshot must not already be terminal.
        """
        if trace.workflow_name and trace.workflow_name != candidate_workflow.name:
            return None, "workflow_name_mismatch", None
        if (
            trace.workflow_version
            and trace.workflow_version != candidate_workflow.version
        ):
            return None, "workflow_version_mismatch", None

        edit_positions = [
            index
            for index, step in enumerate(trace.steps)
            if step.node_id == edit_node_id
        ]
        if not edit_positions:
            return None, "edit_node_not_observed", None
        edit_index = edit_positions[0]
        if edit_index <= 0:
            return None, "edit_node_at_trace_start", edit_index

        prefix_steps = trace.steps[:edit_index]
        failed_prefix = next(
            (
                step
                for step in prefix_steps
                if not step.success or step.error_message
            ),
            None,
        )
        if failed_prefix is not None:
            return (
                None,
                f"prefix_step_failed:{failed_prefix.node_id}",
                edit_index,
            )

        prefix_step = prefix_steps[-1]
        snapshot = prefix_step.state_after
        if not isinstance(snapshot, dict):
            return None, "prefix_snapshot_missing", edit_index
        if bool(snapshot.get("finished", False)):
            return None, "prefix_snapshot_terminal", edit_index

        history = snapshot.get("history")
        if not isinstance(history, list) or not history:
            return None, "prefix_history_missing", edit_index
        if history[-1] != prefix_step.node_id:
            return None, "prefix_history_mismatch", edit_index
        if any(node_id not in candidate_workflow.nodes for node_id in history):
            return None, "prefix_contains_removed_node", edit_index

        checkpoint = ExecutionContext.from_snapshot(
            query,
            snapshot,
            candidate_workflow.name,
        )
        return checkpoint, None, edit_index

    @staticmethod
    def compose_with_gate(
        result: dict[str, Any],
        gate: GateSpec,
    ) -> dict[str, Any]:
        """Offline-compose parent/candidate rows under ``gate``.

        Candidate rollouts have already been observed on the exact same trace
        ids. A false gate replaces every candidate-side outcome and diagnostic
        with its parent counterpart; no executor or LLM call is made.
        """
        rows = result.get("per_query_results")
        if not isinstance(rows, list) or not rows:
            raise ValueError("counterfactual result has no per-query rows")

        composed_rows: list[dict[str, Any]] = []
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                raise ValueError("counterfactual row must be a mapping")
            features = raw_row.get("gate_features")
            if not isinstance(features, dict):
                raise ValueError("counterfactual row is missing gate features")
            applied = evaluate_gate(gate, features)
            row = copy.deepcopy(raw_row)
            if not applied:
                for key in list(row):
                    if not key.startswith("candidate_"):
                        continue
                    original_key = "original_" + key[len("candidate_") :]
                    if original_key in row:
                        row[key] = copy.deepcopy(row[original_key])
                row["candidate_trace_id"] = row.get("trace_id")
                row["delta_u"] = 0.0
            row["gate_applied"] = applied
            row["gate_fingerprint"] = gate.fingerprint
            composed_rows.append(row)

        def mean(key: str) -> float:
            values: list[float] = []
            for row in composed_rows:
                value = row.get(key)
                if isinstance(value, bool):
                    raise ValueError(f"{key} must be numeric")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{key} must be numeric") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{key} must be finite")
                values.append(number)
            return sum(values) / len(values)

        composed = copy.deepcopy(result)
        composed.update(
            {
                "delta_u": mean("delta_u"),
                "original_u": mean("original_u"),
                "candidate_u": mean("candidate_u"),
                "per_query_results": composed_rows,
                "num_queries": len(composed_rows),
                "gate": gate.model_dump(mode="json"),
                "gate_fingerprint": gate.fingerprint,
                "gate_coverage": (
                    sum(bool(row["gate_applied"]) for row in composed_rows)
                    / len(composed_rows)
                ),
                "evaluation_mode": (
                    str(result.get("evaluation_mode", "full_rerun"))
                    + "+offline_gate"
                ),
            }
        )
        return composed

    @staticmethod
    def _trace_diagnostics(trace: ExecutionTrace) -> dict[str, Any]:
        """Return observation-only execution diagnostics for one query."""
        errors: list[dict[str, Any]] = []
        if trace.error_message:
            errors.append(
                {
                    "source": "trace",
                    "message": trace.error_message,
                }
            )
        metadata_error = trace.metadata.get("error_message")
        if metadata_error:
            errors.append(
                {
                    "source": "metadata",
                    "message": str(metadata_error),
                }
            )
        for step in trace.steps:
            if step.error_message:
                errors.append(
                    {
                        "source": "step",
                        "step_index": step.step_index,
                        "node_id": step.node_id,
                        "message": step.error_message,
                    }
                )

        action_counts: dict[str, int] = {}
        node_counts: dict[str, int] = {}
        for step in trace.steps:
            if step.action:
                action_counts[step.action] = (
                    action_counts.get(step.action, 0) + 1
                )
            if (
                step.metadata.get("node_executed", True)
                and step.metadata.get(
                    "scheduler_action_validated",
                    True,
                )
            ):
                node_counts[step.node_id] = (
                    node_counts.get(step.node_id, 0) + 1
                )
        return {
            "prompt_tokens": trace.total_prompt_tokens,
            "completion_tokens": trace.total_completion_tokens,
            # Use the same aggregate/call/step fallback as the gain scorer so
            # legacy traces without aggregate counters still receive a token
            # delta in candidate selection.
            "total_tokens": UtilityComputer.compute_execution_cost(trace),
            "llm_calls": trace.total_llm_calls,
            "latency_seconds": trace.total_latency_seconds,
            "llm_latency_seconds": sum(
                call.latency_seconds
                for step in trace.steps
                for call in step.llm_calls
            ),
            "cost_usd": trace.total_cost_usd,
            "cost_estimate_complete": (
                trace.total_cost_estimate_complete
            ),
            "utility_breakdown": trace.metadata.get("utility_breakdown", {}),
            "success": trace.success,
            "path": [step.node_id for step in trace.steps],
            "executed_path": [
                step.node_id
                for step in trace.steps
                if (
                    step.metadata.get("node_executed", True)
                    and step.metadata.get(
                        "scheduler_action_validated",
                        True,
                    )
                )
            ],
            "actions": [step.action for step in trace.steps if step.action],
            "action_counts": action_counts,
            "node_counts": node_counts,
            "error": errors[0]["message"] if errors else None,
            "errors": errors,
        }

    def _combine(self, hard: float, process: float) -> float:
        """Use the project specification: hard + alpha * process."""
        return hard + self.reward_alpha * process

    def _validate_baseline(
        self,
        baseline: dict[str, Any],
        traces: list[ExecutionTrace],
    ) -> None:
        if (
            baseline.get("batch_signature") != self._batch_signature(traces)
            or not math.isclose(
                baseline.get("reward_alpha", float("nan")),
                self.reward_alpha,
            )
            or not isinstance(baseline.get("entries"), list)
            or len(baseline["entries"]) != len(traces)
        ):
            raise ValueError(
                "baseline does not match the counterfactual trace batch"
            )

    @staticmethod
    def _batch_signature(traces: list[ExecutionTrace]) -> str:
        payload = [
            {
                "trace_id": trace.trace_id,
                "workflow_name": trace.workflow_name,
                "workflow_version": trace.workflow_version,
                "query": trace.query_text,
                "hard": trace.hard_reward,
                "process": trace.process_reward,
                "ground_truth": trace.metadata.get("ground_truth"),
            }
            for trace in traces
        ]
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    @staticmethod
    def _cache_key(
        workflow: WorkflowTemplate,
        batch_signature: str,
        scheduler: object,
        llm_client: object,
        *,
        edit_node_id: Optional[str] = None,
    ) -> str:
        workflow_data = json.dumps(
            workflow.model_dump(mode="python"),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return "|".join(
            (
                workflow_data,
                batch_signature,
                str(id(scheduler)),
                str(id(llm_client)),
                str(edit_node_id),
            )
        )
