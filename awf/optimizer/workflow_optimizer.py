"""LLM workflow optimizer with local, proximal, counterfactual search."""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Any, Optional

from awf.config.schema import OptimizerConfig
from awf.llm.client import AsyncLLMClient
from awf.optimizer.acceptance import AcceptanceCriterion
from awf.optimizer.anchor_localizer import AnchorLocalizer
from awf.optimizer.candidate_archive import CandidateArchive
from awf.optimizer.candidate_generator import (
    MULTI_SCOPE,
    VALID_SCOPES,
    CandidateGenerator,
    WorkflowCandidate,
)
from awf.optimizer.counterfactual import CounterfactualEvaluator
from awf.optimizer.failure_buffer import FailureBuffer
from awf.optimizer.scorer import CandidateScorer
from awf.optimizer.selective_gate import SelectiveGateSearcher
from awf.optimizer.suffix_replay import SuffixReplayEngine
from awf.reward.base import RewardEvaluator
from awf.trace.schema import ExecutionTrace
from awf.utility.compute import UtilityComputer
from awf.workflow.gates import (
    attach_selective_update,
    joint_update_fingerprint,
)
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import NodeType

logger = logging.getLogger(__name__)


class LLMWorkflowOptimizer:
    """Run one round of anchor-local, minimal-effective workflow edits."""

    def __init__(
        self,
        config: OptimizerConfig,
        reward_evaluator: RewardEvaluator,
        utility_computer: Optional[UtilityComputer] = None,
        reward_alpha: float = 0.8,
        execution_llm_defaults: Optional[dict[str, Any]] = None,
    ):
        self.config = config
        self.reward_evaluator = reward_evaluator
        # The current research line updates workflow content and graph-path
        # structure, while scheduler/path execution optimization remains
        # disabled. Suffix replay is an evaluation primitive here: it proves
        # that a proposed local repair works from the unchanged prefix.
        self.workflow_content_only = bool(
            getattr(config, "workflow_content_only", True)
        )
        self._candidate_scopes = (
            (*VALID_SCOPES, MULTI_SCOPE)
            if bool(getattr(config, "multi_level_updates", False))
            else VALID_SCOPES
        )
        configured_replay_gate = getattr(
            config,
            "require_failure_suffix_replay",
            None,
        )
        self._require_failure_suffix_replay = (
            self.workflow_content_only
            if configured_replay_gate is None
            else bool(configured_replay_gate)
        )
        self.utility_computer = utility_computer or UtilityComputer(
            lambda_cost=config.lambda_cost,
            lambda_latency=config.lambda_latency,
            lambda_api_cost=config.lambda_api_cost,
            rho_omega=config.rho_omega,
        )
        self.llm = AsyncLLMClient(config.llm)
        self.failure_buffer = FailureBuffer(
            capacity=config.failure_buffer_capacity,
            success_threshold=config.hard_success_threshold,
        )
        self.anchor_localizer = AnchorLocalizer(self.llm)
        self.candidate_generator = CandidateGenerator(
            self.llm,
            max_candidates=config.candidates_per_round,
            max_edit_distance=config.max_edit_distance,
            allowed_execution_models=config.allowed_execution_models,
            execution_llm_defaults=execution_llm_defaults,
            allowed_scopes=self._candidate_scopes,
            aggressive_generation=config.aggressive_candidate_generation,
            allow_cross_block_graph_updates=(
                config.allow_cross_block_graph_updates
            ),
            deterministic_block_candidates=(
                config.deterministic_block_candidates
            ),
        )
        suffix_replay = None
        if (
            config.use_suffix_replay
            or getattr(config, "online_failure_updates", False)
            or getattr(config, "failure_update_mode", None)
            in {"online", "deferred_sequential"}
            or self._require_failure_suffix_replay
        ):
            suffix_replay = SuffixReplayEngine(
                max_cache_size=config.max_suffix_replay_cache,
            )
        self.counterfactual = CounterfactualEvaluator(
            reward_evaluator=reward_evaluator,
            utility_computer=self.utility_computer,
            suffix_replay=suffix_replay,
            reward_alpha=reward_alpha,
        )
        self.counterfactual.require_suffix_replay = (
            self._require_failure_suffix_replay
        )
        # Utility is reward-only.  Token usage is folded into candidate gain
        # alongside edit distance, so a counterfactual can trade reward
        # improvement against a measurable token delta without changing the
        # reward metric itself.
        self.scorer = CandidateScorer(
            mu_edit=config.mu_edit,
            lambda_tokens=config.lambda_cost,
        )
        self.acceptance = AcceptanceCriterion(
            epsilon_stat=config.epsilon_stat,
        )
        self.selective_gate_searcher = (
            SelectiveGateSearcher(
                features=config.gate_features,
                min_leaf_support=config.gate_min_leaf_support,
                hard_success_threshold=config.hard_success_threshold,
                hard_regression_tolerance=(
                    config.gate_hard_regression_tolerance
                ),
                min_effect=config.epsilon_stat,
                simplicity_penalty=config.gate_simplicity_penalty,
            )
            if (
                not self.workflow_content_only
                and config.selective_update_enabled
            )
            else None
        )
        self.candidate_archive = CandidateArchive(
            capacity=config.candidate_archive_size,
        )
        self.round_history: list[dict] = []
        # Stable patch identities survive workflow version bumps within this
        # optimizer run. Values retain their candidate records by reference so
        # final scope/selection statuses remain available to later rounds.
        self._candidate_outcomes: dict[str, dict[str, Any]] = {}
        ignored_outer_flags = [
            name
            for name, enabled in (
                ("selective_update_enabled", config.selective_update_enabled),
                (
                    "efficiency_optimization_enabled",
                    config.efficiency_optimization_enabled,
                ),
            )
            if enabled
        ]
        if self.workflow_content_only and ignored_outer_flags:
            logger.info(
                "workflow_content_only=True; ignoring outer execution "
                "optimizations: %s",
                ", ".join(ignored_outer_flags),
            )

    async def optimize_round(
        self,
        workflow: WorkflowTemplate,
        executor: object,
        scheduler: object,
        llm_client: Optional[AsyncLLMClient] = None,
    ) -> tuple[WorkflowTemplate, dict]:
        """Run one optimization round with the runtime LLM client supplied.

        Candidate scopes are generated and evaluated incrementally. In
        ordinary gain-driven mode, a sufficient local edit stops escalation.
        Failure repair instead searches globally by intervention level:
        prompt first, operator second, then block/graph and optional multi-level
        edits. A stable repair at a lower level stops generation of every more
        invasive level. Conservative runs may use edit distance as a
        tie-break; clustered aggressive runs remove that preference and let
        reward/token gain select repairs within the first successful level.
        """
        round_num = len(self.round_history) + 1
        logger.info("Starting optimization round %s", round_num)
        # Keep lightweight test doubles/legacy callers that construct the
        # optimizer without running ``__init__`` compatible with the explicit
        # legacy outer-layer mode.
        if not hasattr(self, "workflow_content_only"):
            self.workflow_content_only = bool(
                getattr(self.config, "workflow_content_only", False)
            )
        if not hasattr(self, "_candidate_scopes"):
            self._candidate_scopes = (
                (*VALID_SCOPES, MULTI_SCOPE)
                if bool(getattr(self.config, "multi_level_updates", False))
                else VALID_SCOPES
            )
        if not hasattr(self, "_require_failure_suffix_replay"):
            configured_replay_gate = getattr(
                self.config,
                "require_failure_suffix_replay",
                None,
            )
            self._require_failure_suffix_replay = (
                self.workflow_content_only
                if configured_replay_gate is None
                else bool(configured_replay_gate)
            )
        if hasattr(self, "counterfactual") and not hasattr(
            self.counterfactual,
            "require_suffix_replay",
        ):
            self.counterfactual.require_suffix_replay = (
                self._require_failure_suffix_replay
            )
        # A checkpoint produced by an earlier outer-layer ablation may still
        # carry a selective policy.  In the current research mode the policy
        # is discarded and its base workflow content remains the incumbent, so
        # inner prompt/operator learning can continue.
        ignored_outer_policy = False
        if self.workflow_content_only and workflow.selective_update is not None:
            payload = workflow.model_dump(mode="json")
            payload["selective_update"] = None
            workflow = WorkflowTemplate.model_validate(payload)
            ignored_outer_policy = True
        selective_enabled = (
            not self.workflow_content_only
            and bool(getattr(self.config, "selective_update_enabled", False))
        )
        if workflow.selective_update is not None:
            # The first research implementation deliberately supports one
            # conditional intervention. Optimizing a policy-of-policies would
            # invalidate anchor localization and counterfactual attribution.
            return self._finish_round(
                workflow,
                {
                    "round": round_num,
                    "workflow_name": workflow.name,
                    "workflow_version": workflow.version,
                    "accepted": False,
                    "gain": 0.0,
                    "candidate_description": "",
                    "candidate_evaluations": [],
                    "scope_attempts": [],
                    "selective_update_enabled": selective_enabled,
                    "selective_update_rejection_reason": (
                        "nested_policy_not_supported"
                    ),
                },
            )

        observed_successes = [
            trace
            for trace in self.failure_buffer.successes
            if (
                trace.workflow_name == workflow.name
                and trace.workflow_version == workflow.version
            )
        ]
        trace_batch = self.failure_buffer.consume_optimization_round(
            workflow.name,
            workflow.version,
            success_fraction=getattr(
                self.config,
                "success_guard_fraction",
                0.2,
            ),
            min_success_guards=getattr(
                self.config,
                "min_success_guards",
                0,
            ),
            efficiency_enabled=(
                not self.workflow_content_only
                and bool(
                    getattr(
                        self.config,
                        "efficiency_optimization_enabled",
                        False,
                    )
                )
            ),
            efficiency_fraction=getattr(
                self.config,
                "efficiency_anchor_fraction",
                0.2,
            ),
            efficiency_min_relative_cost=getattr(
                self.config,
                "efficiency_min_relative_cost",
                1.25,
            ),
            max_efficiency_anchors=getattr(
                self.config,
                "efficiency_max_anchors",
                5,
            ),
            efficiency_cost=self._trace_efficiency_cost,
        )
        failure_traces = trace_batch.failures
        efficiency_traces = (
            []
            if self.workflow_content_only
            else trace_batch.efficiency_anchors
        )
        success_traces = trace_batch.success_guards
        cf_traces = trace_batch.counterfactual_batch
        gate_fit_traces = (
            self._select_gate_fit_traces(
                trace_batch.representative_traces,
                max_traces=self.config.gate_fit_max_traces,
            )
            if selective_enabled
            else []
        )
        evaluation_traces = (
            self._unique_traces(cf_traces + gate_fit_traces)
            if selective_enabled
            else cf_traces
        )
        trigger_traces = failure_traces + efficiency_traces
        failure_repair_mode = bool(
            failure_traces and self._require_failure_suffix_replay
        )
        round_summary = {
            "round": round_num,
            "workflow_name": workflow.name,
            "workflow_version": workflow.version,
            "utility_definition": "hard_reward",
            "gain_definition": (
                "delta_hard_reward - lambda_cost*delta_tokens "
                "- mu_edit*edit_distance; failure repair requires "
                "suffix-replay success, with no hard edit-distance limit"
            ),
            "num_failures": len(failure_traces),
            # Keep observed successes distinct from the much smaller sampled
            # regression-guard subset placed in B_cf.
            "num_successes": len(observed_successes),
            "num_efficiency_anchors": len(efficiency_traces),
            "num_success_guards": len(success_traces),
            "trigger_types": [
                trigger
                for trigger, present in (
                    ("hard_failure", bool(failure_traces)),
                    ("high_cost_success", bool(efficiency_traces)),
                )
                if present
            ],
            "counterfactual_batch_size": len(cf_traces),
            "candidate_evaluation_batch_size": len(evaluation_traces),
            "gate_fit_batch_size": len(gate_fit_traces),
            "selective_update_enabled": selective_enabled,
            "accepted": False,
            "gain": 0.0,
            "candidate_description": "",
            "generated_candidates": 0,
            "evaluated_candidates": 0,
            "duplicate_candidates": 0,
            "primary_candidate_budget": self.config.candidates_per_round,
            "exploration_budget": getattr(
                self.config,
                "exploration_budget",
                0,
            ),
            "exploration_candidates_evaluated": 0,
            "exploration_acceptance_floor": float(
                getattr(self.config, "exploration_acceptance_floor", 0.0)
            ),
            "candidate_evaluations": [],
            "scope_attempts": [],
            "evaluation_mode": self.counterfactual.evaluation_mode,
            "suffix_replay_requested": self.counterfactual.suffix_replay
            is not None,
            "suffix_replay_used": False,
            "workflow_content_only": self.workflow_content_only,
            "outer_execution_optimization_enabled": (
                not self.workflow_content_only
            ),
            "candidate_scopes": list(self._candidate_scopes),
            "failure_repair_suffix_replay_required": failure_repair_mode,
            "failure_repair_min_coverage": float(
                getattr(self.config, "failure_cluster_min_coverage", 1.0)
            ),
            "failure_repair_gate_passed": None,
            "outer_policy_ignored": ignored_outer_policy,
        }

        if not trigger_traces:
            logger.info(
                "No current-version hard failures or high-cost successes; "
                "skipping round"
            )
            return self._finish_round(workflow, round_summary)

        self._ensure_optimization_traces(evaluation_traces)
        failure_anchors = await self.anchor_localizer.localize_batch(
            failure_traces,
            workflow,
            top_m=min(5, self.config.candidates_per_round),
        )
        efficiency_anchors = (
            self._localize_efficiency_anchors(
                efficiency_traces,
                workflow,
                top_m=min(5, self.config.candidates_per_round),
            )
            if not self.workflow_content_only
            else []
        )
        anchors = (
            self._merge_anchors(
                failure_anchors,
                efficiency_anchors,
                top_m=min(5, self.config.candidates_per_round),
            )
            if efficiency_anchors
            else failure_anchors
        )
        round_summary["anchors"] = [
            {
                "node_id": anchor["node_id"],
                "rank": anchor["rank"],
                "score": anchor["score"],
            }
            for anchor in anchors
        ]
        if not anchors:
            logger.info("No valid anchors identified; skipping round")
            return self._finish_round(workflow, round_summary)

        failure_context = self._build_failure_context(
            failure_traces,
            success_traces,
            efficiency_traces=efficiency_traces,
        )
        experience_context = self._build_experience_context()
        self.counterfactual.clear_cache()
        baseline = self.counterfactual.prepare_baseline(evaluation_traces)
        remaining_budget = self.config.candidates_per_round
        exploration_remaining = getattr(
            self.config,
            "exploration_budget",
            0,
        )
        efficiency_mode_enabled = (
            not self.workflow_content_only
            and bool(
                getattr(
                    self.config,
                    "efficiency_optimization_enabled",
                    False,
                )
            )
        )
        protected_success_trace_ids = {
            trace.trace_id
            for trace in efficiency_traces + success_traces
        }
        if selective_enabled:
            protected_success_trace_ids.update(
                trace.trace_id
                for trace in gate_fit_traces
                if (
                    trace.hard_reward is not None
                    and trace.hard_reward
                    >= self.config.hard_success_threshold
                )
            )
        # Candidate workflow selection is a single utility decision.  A
        # selective policy is fitted only after that decision, so its protected
        # success constraints must not filter the candidate update itself.
        # In the aggressive clustered protocol, sampled successful traces are
        # still regression guards even though the outer efficiency optimizer is
        # disabled.  Keeping this local guard (rather than a full validation
        # pass) lets multi-failure updates explore while protecting the small
        # set of already-correct paths supplied to the counterfactual batch.
        enforce_hard_success_guard = bool(
            protected_success_trace_ids
            and (
                efficiency_mode_enabled
                or bool(getattr(self.config, "failure_cluster_enabled", False))
            )
        )
        archive_expected_trace_ids = {
            trace.trace_id for trace in evaluation_traces
        }
        archive_comparison_key = json.dumps(
            {
                "workflow_name": workflow.name,
                "workflow_version": workflow.version,
                "trace_ids": sorted(archive_expected_trace_ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        sufficient: list[tuple[WorkflowCandidate, float]] = []
        candidate_records: dict[int, dict] = {}
        candidate_results: dict[int, dict] = {}

        candidate_scopes = self._candidate_scopes
        if failure_repair_mode:
            # Search globally by intervention level. Every localized anchor
            # gets a prompt attempt before any operator attempt, and every
            # operator attempt precedes block/graph search. Once one level
            # contains a valid repair, finish that level and never generate a
            # more invasive level for this failure cluster.
            search_pairs = [
                (anchor, scope_index, scope)
                for scope_index, scope in enumerate(candidate_scopes)
                for anchor in anchors
            ]
        else:
            search_pairs = [
                (anchor, scope_index, scope)
                for anchor in anchors
                for scope_index, scope in enumerate(candidate_scopes)
            ]
        active_repair_scope: str | None = None
        repair_found_in_active_scope = False
        previous_anchor_id: str | None = None
        exploring_broader_scope = False
        for anchor, scope_index, scope in search_pairs:
            if (
                failure_repair_mode
                and active_repair_scope is not None
                and scope != active_repair_scope
                and repair_found_in_active_scope
            ):
                round_summary["scope_first_stop"] = {
                    "stopped_before_scope": scope,
                    "selected_scope": active_repair_scope,
                    "reason": "lower_level_failure_repair_found",
                }
                break
            if failure_repair_mode and scope != active_repair_scope:
                active_repair_scope = scope
                repair_found_in_active_scope = False
            anchor_id = str(anchor["node_id"])
            if anchor_id != previous_anchor_id:
                exploring_broader_scope = False
                previous_anchor_id = anchor_id
            if remaining_budget <= 0 and not exploring_broader_scope:
                break
            active_budget = (
                exploration_remaining
                if exploring_broader_scope
                else remaining_budget
            )
            if active_budget <= 0:
                break
            broader_scope_count = len(candidate_scopes) - scope_index - 1
            if (
                failure_repair_mode
                and not repair_found_in_active_scope
                and active_budget <= broader_scope_count
            ):
                # Preserve at least one evaluation for every more invasive
                # level when all candidates at the current level fail.
                continue
            # Reserve one evaluation for each broader inner scope so
            # several prompt suggestions cannot exhaust the round before
            # operator scope is attempted after a failed prompt scope.
            scope_limit = max(
                1,
                active_budget - broader_scope_count,
            )
            if bool(
                getattr(self.config, "aggressive_candidate_generation", False)
            ):
                # Prevent numerous same-level prompt variants from
                # starving structural actions.  Block gets room for both
                # deterministic macros plus an LLM graph proposal.
                if scope == "block":
                    scope_limit = min(active_budget, max(3, scope_limit))
                else:
                    scope_limit = min(scope_limit, 2)
            attempt = {
                "anchor_id": anchor["node_id"],
                "scope": scope,
                "generated": 0,
                "evaluated": 0,
                "duplicates": 0,
                "sufficient": False,
            }
            if getattr(self.config, "exploration_budget", 0) > 0:
                attempt["evaluation_budget"] = (
                    "exploration"
                    if exploring_broader_scope
                    else "primary"
                )
            round_summary["scope_attempts"].append(attempt)
            generation_kwargs: dict[str, Any] = {"limit": scope_limit}
            if experience_context:
                generation_kwargs["experience_context"] = (
                    experience_context
                )
            candidates = await self.candidate_generator.generate_for_anchor(
                workflow,
                anchor,
                scope,
                failure_context,
                **generation_kwargs,
            )
            attempt["generation_report"] = copy.deepcopy(
                getattr(
                    self.candidate_generator,
                    "last_generation_report",
                    {},
                )
            )
            attempt["generated"] = len(candidates)
            round_summary["generated_candidates"] += len(candidates)

            scoped_scores: list[tuple[WorkflowCandidate, float]] = []
            for candidate in candidates:
                active_budget = (
                    exploration_remaining
                    if exploring_broader_scope
                    else remaining_budget
                )
                if active_budget <= 0:
                    break
                patch_fingerprint = str(
                    candidate.metadata.get("patch_fingerprint")
                    or CandidateGenerator.candidate_patch_fingerprint(
                        candidate
                    )
                )
                candidate.metadata["patch_fingerprint"] = (
                    patch_fingerprint
                )
                candidate_record = {
                    "candidate_index": len(
                        round_summary["candidate_evaluations"]
                    ),
                    "anchor_id": candidate.anchor_id,
                    "scope": candidate.scope,
                    "node_id": candidate.node_id,
                    "description": candidate.description,
                    "changes": copy.deepcopy(candidate.changes),
                    "patch_fingerprint": patch_fingerprint,
                    "changed_units": list(candidate.changed_units),
                    "edit_distance": candidate.edit_distance,
                    "delta_u": None,
                    "gain": None,
                    "status": "evaluating",
                    "per_query_results": [],
                    "final_selected": False,
                }
                round_summary["candidate_evaluations"].append(
                    candidate_record
                )
                outcome_store = getattr(
                    self,
                    "_candidate_outcomes",
                    None,
                )
                if outcome_store is None:
                    outcome_store = {}
                    self._candidate_outcomes = outcome_store
                previous = outcome_store.get(patch_fingerprint)
                if previous is not None:
                    previous_record = previous.get("record", {})
                    candidate_record.update(
                        {
                            "status": "duplicate_history",
                            "duplicate_of_round": previous.get("round"),
                            "duplicate_of_status": previous_record.get(
                                "status"
                            ),
                        }
                    )
                    attempt["duplicates"] += 1
                    round_summary["duplicate_candidates"] += 1
                    continue

                if exploring_broader_scope:
                    exploration_remaining -= 1
                    round_summary[
                        "exploration_candidates_evaluated"
                    ] += 1
                else:
                    remaining_budget -= 1
                attempt["evaluated"] += 1
                round_summary["evaluated_candidates"] += 1
                candidate_records[id(candidate)] = candidate_record
                outcome_store[patch_fingerprint] = {
                    "round": round_num,
                    "record": candidate_record,
                }
                replay_edit_node_id = self._counterfactual_edit_node_id(
                    candidate,
                    workflow,
                )
                try:
                    result = await self.counterfactual.evaluate(
                        candidate.modified_workflow,
                        evaluation_traces,
                        executor,
                        scheduler,
                        llm_client=llm_client,
                        baseline=baseline,
                        edit_node_id=replay_edit_node_id,
                    )
                    if failure_repair_mode:
                        result = await self._confirm_candidate_failure_repairs(
                            result,
                            candidate.modified_workflow,
                            failure_traces,
                            executor,
                            scheduler,
                            llm_client=llm_client,
                            edit_node_id=replay_edit_node_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "Counterfactual evaluation failed for %s/%s: %s",
                        candidate.node_id,
                        candidate.scope,
                        exc,
                    )
                    candidate_record.update(
                        {
                            "status": "evaluation_error",
                            "error": str(exc),
                        }
                    )
                    continue
                candidate_results[id(candidate)] = result
                # Select the workflow patch using reward delta plus the
                # external token/edit gain terms in both modes. Selective
                # applicability is decided only after this candidate has
                # won.
                selection_delta_u = self._selection_delta_u(
                    result,
                    {trace.trace_id for trace in trigger_traces},
                )
                selection_token_delta = self._selection_token_delta(
                    result,
                    {trace.trace_id for trace in trigger_traces},
                )
                gain = self.scorer.score(
                    candidate,
                    selection_delta_u,
                    token_delta=selection_token_delta,
                )
                token_weight = self._token_gain_weight()
                candidate_record.update(
                    {
                        "delta_u": selection_delta_u,
                        "token_delta": selection_token_delta,
                        "token_penalty": (
                            token_weight * selection_token_delta
                        ),
                        "edit_penalty": (
                            self.scorer.mu_edit
                            * candidate.edit_distance
                        ),
                        "gain": gain,
                        "status": "evaluated",
                        "per_query_results": copy.deepcopy(
                            result.get("per_query_results", [])
                        ),
                        "evaluation_mode": result.get(
                            "evaluation_mode",
                            self.counterfactual.evaluation_mode,
                        ),
                        "cache_hit": bool(
                            result.get("cache_hit", False)
                        ),
                    }
                )
                if result.get("fallback_reason"):
                    candidate_record["fallback_reason"] = result[
                        "fallback_reason"
                    ]
                if result.get("failure_repair_confirmation"):
                    candidate_record["failure_repair_confirmation"] = (
                        copy.deepcopy(
                            result["failure_repair_confirmation"]
                        )
                    )
                round_summary["suffix_replay_used"] = (
                    round_summary["suffix_replay_used"]
                    or result.get("suffix_replay_used", False)
                )
                if result.get("fallback_reason"):
                    round_summary["suffix_replay_fallback_reason"] = (
                        result["fallback_reason"]
                    )
                candidate.metadata.update(
                    {
                        "delta_u": selection_delta_u,
                        "token_delta": selection_token_delta,
                        "gain": gain,
                        "evaluation_mode": result["evaluation_mode"],
                        "cache_hit": result["cache_hit"],
                    }
                )
                hard_guard = self._hard_success_guard(
                    result,
                    protected_success_trace_ids,
                    required=enforce_hard_success_guard,
                )
                if enforce_hard_success_guard:
                    candidate_record["hard_success_guard"] = hard_guard
                if not hard_guard["passed"]:
                    candidate_record["status"] = (
                        "hard_success_regression"
                    )
                    continue
                failure_repair_gate = self._failure_suffix_replay_gate(
                    result,
                    {trace.trace_id for trace in failure_traces},
                    min_coverage=float(
                        getattr(
                            self.config,
                            "failure_cluster_min_coverage",
                            1.0,
                        )
                    ),
                )
                if failure_repair_mode:
                    candidate_record["failure_repair_gate"] = (
                        copy.deepcopy(failure_repair_gate)
                    )
                    if not failure_repair_gate["passed"]:
                        candidate_record["status"] = (
                            "failure_repair_gate_failed"
                        )
                        continue
                    candidate_record["status"] = (
                        "failure_repair_eligible"
                    )
                archive = getattr(self, "candidate_archive", None)
                if archive is not None and not selective_enabled:
                    archive.add(
                        candidate_record,
                        round_num=round_num,
                        split="optimization",
                        comparison_key=archive_comparison_key,
                        expected_trace_ids=archive_expected_trace_ids,
                    )
                scoped_scores.append((candidate, gain))

            best_in_scope = self._select_best_candidate(
                scoped_scores,
                selective=False,
                failure_repair=failure_repair_mode,
            )
            for scoped_candidate, _ in scoped_scores:
                record = candidate_records[id(scoped_candidate)]
                record["status"] = (
                    "sufficient"
                    if (
                        best_in_scope is not None
                        and scoped_candidate is best_in_scope[0]
                    )
                    else (
                        "not_scope_selected"
                        if best_in_scope is not None
                        else "insufficient_gain"
                    )
                )
            if best_in_scope is not None:
                attempt["sufficient"] = True
                sufficient.append(best_in_scope)
                if failure_repair_mode:
                    repair_found_in_active_scope = True
                    continue
                # The default remains minimal-effective greedy search. An
                # explicit exploration budget continues into broader scopes
                # and may surface a better Pareto-safe alternative.
                if (
                    not exploring_broader_scope
                    and exploration_remaining > 0
                    and scope_index < len(candidate_scopes) - 1
                ):
                    exploring_broader_scope = True
                    continue
                if not exploring_broader_scope:
                    break

        best = self._select_best_candidate(
            sufficient,
            selective=False,
            failure_repair=failure_repair_mode,
        )
        if best is None:
            if failure_repair_mode:
                round_summary["failure_repair_gate_passed"] = False
            logger.info(
                "No candidate met the %s acceptance rule",
                "suffix-replay failure-repair" if failure_repair_mode else "gain",
            )
            return self._finish_round(workflow, round_summary)

        best_candidate, best_gain = best
        if best_candidate.modified_workflow is None:
            candidate_records[id(best_candidate)]["status"] = (
                "invalid_modified_workflow"
            )
            return self._finish_round(workflow, round_summary)

        if selective_enabled:
            try:
                searcher = self.selective_gate_searcher
                if searcher is None:
                    raise ValueError("selective gate searcher is unavailable")
                joint = searcher.search(
                    candidate_results[id(best_candidate)],
                    fit_trace_ids={
                        trace.trace_id for trace in gate_fit_traces
                    },
                    protected_trace_ids=protected_success_trace_ids,
                    edit_distance=best_candidate.edit_distance,
                    failure_mode=bool(failure_traces),
                )
                selected_gate = joint.gate
                joint_fingerprint = joint_update_fingerprint(
                    str(best_candidate.metadata.get("patch_fingerprint", "")),
                    selected_gate,
                )
                best_candidate.metadata.update(
                    {
                        "gate": selected_gate.model_dump(mode="json"),
                        "gate_metrics": copy.deepcopy(joint.metrics),
                        "joint_rank_key": tuple(joint.rank_key),
                        "joint_positive": joint.positive,
                        "joint_fingerprint": joint_fingerprint,
                    }
                )
                candidate_records[id(best_candidate)].update(
                    {
                        "gate": selected_gate.model_dump(mode="json"),
                        "gate_metrics": copy.deepcopy(joint.metrics),
                        "gate_coverage": joint.metrics["coverage"],
                        "joint_rank_key": list(joint.rank_key),
                        "joint_fingerprint": joint_fingerprint,
                    }
                )
                if not joint.positive or selected_gate.kind == "never":
                    candidate_records[id(best_candidate)].update(
                        {
                            "status": "selective_no_positive_gate",
                            "final_selected": False,
                        }
                    )
                    return self._finish_round(workflow, round_summary)
                # The archive records the realized policy, not the ungated
                # patch.  This keeps archive diagnostics separate from the
                # scalar workflow-patch decision above.
                # Keep the patch-level gain selected above.  The gate search
                # reports a separate policy score for telemetry; replacing
                # ``gain`` here would silently discard the token/edit terms
                # used to select the workflow patch.
                candidate_records[id(best_candidate)].update(
                    {
                        "delta_u": joint.composed_result.get(
                            "delta_u", best_gain
                        ),
                        "policy_report_score": joint.report_score,
                        "per_query_results": copy.deepcopy(
                            joint.composed_result.get(
                                "per_query_results", []
                            )
                        ),
                    }
                )
                archive = getattr(self, "candidate_archive", None)
                if archive is not None:
                    archive.add(
                        candidate_records[id(best_candidate)],
                        round_num=round_num,
                        split="optimization",
                        comparison_key=archive_comparison_key,
                        expected_trace_ids=archive_expected_trace_ids,
                    )
                if selected_gate.kind == "never":
                    raise ValueError("never gate cannot be promoted")
                if selected_gate.kind == "always":
                    updated_workflow = copy.deepcopy(
                        best_candidate.modified_workflow
                    )
                else:
                    updated_workflow = attach_selective_update(
                        workflow,
                        best_candidate.modified_workflow,
                        selected_gate,
                        patch_fingerprint=str(
                            best_candidate.metadata["patch_fingerprint"]
                        ),
                        changed_units=list(
                            best_candidate.changed_units
                        ),
                    )
            except Exception as exc:
                candidate_records[id(best_candidate)].update(
                    {
                        "status": "policy_materialization_error",
                        "error": str(exc),
                        "final_selected": False,
                    }
                )
                return self._finish_round(workflow, round_summary)
        else:
            updated_workflow = copy.deepcopy(
                best_candidate.modified_workflow
            )
        for sufficient_candidate, _ in sufficient:
            record = candidate_records[id(sufficient_candidate)]
            if sufficient_candidate is best_candidate:
                record["status"] = "selected"
                record["final_selected"] = True
            else:
                record["status"] = "not_final_selected"
        updated_workflow.version = self._bump_version(workflow.version)
        selected_token_delta = self._safe_float(
            best_candidate.metadata.get("token_delta", 0.0)
        )
        if selected_token_delta is None:
            selected_token_delta = 0.0
        round_summary.update(
            {
                "accepted": True,
                "failure_repair_gate_passed": (
                    True if failure_repair_mode else None
                ),
                "gain": best_gain,
                "delta_u": best_candidate.metadata.get("delta_u", 0.0),
                "token_delta": selected_token_delta,
                "token_penalty": (
                    self._token_gain_weight() * selected_token_delta
                ),
                "edit_penalty": (
                    self.scorer.mu_edit * best_candidate.edit_distance
                ),
                "candidate_description": best_candidate.description,
                "candidate_scope": best_candidate.scope,
                "candidate_node_id": best_candidate.node_id,
                "candidate_patch_fingerprint": best_candidate.metadata.get(
                    "patch_fingerprint",
                ),
                "candidate_joint_fingerprint": best_candidate.metadata.get(
                    "joint_fingerprint",
                ),
                "selected_gate": copy.deepcopy(
                    best_candidate.metadata.get("gate")
                ),
                "selected_gate_metrics": copy.deepcopy(
                    best_candidate.metadata.get("gate_metrics")
                ),
                "anchor_id": best_candidate.anchor_id,
                "changed_units": list(best_candidate.changed_units),
                "edit_distance": best_candidate.edit_distance,
            }
        )
        logger.info(
            "Round %s accepted %s/%s with gain %.4f",
            round_num,
            best_candidate.node_id,
            best_candidate.scope,
            best_gain,
        )
        return self._finish_round(updated_workflow, round_summary)

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Add an execution trace to the pending optimization epoch."""
        self.failure_buffer.add(trace)

    def add_traces(self, traces: list[ExecutionTrace]) -> None:
        """Add execution traces to the pending optimization epoch."""
        self.failure_buffer.extend(traces)

    async def _confirm_candidate_failure_repairs(
        self,
        result: dict[str, Any],
        candidate_workflow: WorkflowTemplate,
        failure_traces: list[ExecutionTrace],
        executor: object,
        scheduler: object,
        *,
        llm_client: AsyncLLMClient | None,
        edit_node_id: str | None,
    ) -> dict[str, Any]:
        """Repeat only target-failure candidate runs and stabilize labels.

        Successful guards are not repeated. Each target failure already has
        one candidate observation in ``result``; this method adds the
        configured number of uncached executions and converts its hard label
        to a stable success iff the required number of runs succeeds.
        """
        if not bool(
            getattr(
                self.config,
                "failure_repair_confirmation_enabled",
                False,
            )
        ):
            return result
        repeats = int(
            getattr(
                self.config,
                "failure_repair_confirmation_repeats",
                2,
            )
        )
        min_successes = int(
            getattr(
                self.config,
                "failure_repair_confirmation_min_successes",
                2,
            )
        )
        if not failure_traces or repeats <= 0:
            return result

        baseline = self.counterfactual.prepare_baseline(failure_traces)
        repeated_results: list[dict[str, Any]] = []
        for _ in range(repeats):
            repeated_results.append(
                await self.counterfactual.evaluate(
                    candidate_workflow,
                    failure_traces,
                    executor,
                    scheduler,
                    llm_client=llm_client,
                    baseline=baseline,
                    edit_node_id=edit_node_id,
                    use_cache=False,
                )
            )

        rows = result.get("per_query_results", [])
        by_id = {
            str(row.get("trace_id")): row
            for row in rows
            if isinstance(row, dict) and row.get("trace_id") is not None
        }
        repeated_by_id: dict[str, list[dict[str, Any]]] = {}
        for repeated in repeated_results:
            for row in repeated.get("per_query_results", []):
                if not isinstance(row, dict) or row.get("trace_id") is None:
                    continue
                repeated_by_id.setdefault(str(row["trace_id"]), []).append(row)

        try:
            threshold = float(self.config.hard_success_threshold)
        except (TypeError, ValueError, OverflowError, AttributeError):
            threshold = 1.0
        confirmation_rows: list[dict[str, Any]] = []
        for trace in failure_traces:
            trace_id = str(trace.trace_id)
            first = by_id.get(trace_id)
            if first is None:
                continue
            observations = [first, *repeated_by_id.get(trace_id, [])]
            successes = 0
            hard_rewards: list[float] = []
            modes: list[str] = []
            for observation in observations:
                try:
                    hard = float(observation.get("candidate_hard", 0.0))
                except (TypeError, ValueError, OverflowError):
                    hard = 0.0
                hard_rewards.append(hard)
                modes.append(
                    str(observation.get("candidate_evaluation_mode", "unknown"))
                )
                successes += int(
                    bool(observation.get("candidate_success", False))
                    and hard >= threshold
                )
            stable_success = successes >= min_successes
            first["candidate_hard_single_run"] = first.get("candidate_hard")
            first["candidate_success_single_run"] = first.get("candidate_success")
            first["candidate_hard"] = 1.0 if stable_success else 0.0
            first["candidate_success"] = stable_success
            first["candidate_u"] = first["candidate_hard"]
            try:
                original_u = float(first.get("original_u", 0.0))
            except (TypeError, ValueError, OverflowError):
                original_u = 0.0
            first["delta_u"] = first["candidate_u"] - original_u
            first["repair_confirmation"] = {
                "total_runs": len(observations),
                "successes": successes,
                "min_successes": min_successes,
                "passed": stable_success,
                "hard_rewards": hard_rewards,
                "evaluation_modes": modes,
            }
            confirmation_rows.append(
                {"trace_id": trace_id, **first["repair_confirmation"]}
            )

        if rows:
            result["candidate_u"] = sum(
                float(row.get("candidate_u", 0.0))
                for row in rows
                if isinstance(row, dict)
            ) / len(rows)
            result["delta_u"] = sum(
                float(row.get("delta_u", 0.0))
                for row in rows
                if isinstance(row, dict)
            ) / len(rows)
        result["failure_repair_confirmation"] = {
            "enabled": True,
            "additional_runs_per_failure": repeats,
            "min_successes": min_successes,
            "rows": confirmation_rows,
        }
        return result

    def _failure_suffix_replay_gate(
        self,
        result: dict[str, Any],
        failure_trace_ids: set[str],
        *,
        min_coverage: float = 1.0,
    ) -> dict[str, Any]:
        """Check cluster-local repair evidence independently of scalar gain.

        The conservative compatibility mode retains coverage + suffix replay.
        Aggressive clustered experiments accept a candidate when at least one
        member becomes correct; inserted nodes may require a full run because
        no valid incumbent prefix contains their outputs.  Sampled-success
        non-regression is enforced separately before this predicate.
        """
        required_ids = sorted(str(item) for item in failure_trace_ids)
        try:
            coverage_target = min(max(float(min_coverage), 0.0), 1.0)
        except (TypeError, ValueError, OverflowError):
            coverage_target = 1.0
        rows = result.get("per_query_results")
        if not isinstance(rows, list):
            return {
                "required": True,
                "passed": False,
                "reason": "missing_per_query_results",
                "required_failure_traces": required_ids,
                "repaired_failure_traces": [],
                "missing_failure_traces": required_ids,
                "non_repaired_failure_traces": required_ids,
            }

        by_id = {
            str(row.get("trace_id")): row
            for row in rows
            if isinstance(row, dict) and row.get("trace_id") is not None
        }
        repaired: list[str] = []
        missing: list[str] = []
        non_repaired: list[str] = []
        try:
            threshold = float(
                getattr(self.config, "hard_success_threshold", 1.0)
            )
        except (TypeError, ValueError, OverflowError):
            threshold = 1.0
        for trace_id in required_ids:
            row = by_id.get(trace_id)
            if row is None:
                missing.append(trace_id)
                non_repaired.append(trace_id)
                continue
            replay_used = bool(
                row.get("candidate_suffix_replay_used", False)
            )
            execution_success = bool(row.get("candidate_success", False))
            try:
                hard_success = float(row.get("candidate_hard", 0.0)) >= threshold
            except (TypeError, ValueError, OverflowError):
                hard_success = False
            allow_full_rerun = bool(
                getattr(self.config, "allow_failure_full_rerun", False)
            )
            if (replay_used or allow_full_rerun) and execution_success and hard_success:
                repaired.append(trace_id)
            else:
                non_repaired.append(trace_id)

        coverage = (
            len(repaired) / len(required_ids)
            if required_ids
            else 0.0
        )
        acceptance = str(
            getattr(self.config, "failure_repair_acceptance", "coverage")
        )
        if acceptance == "any_repair":
            passed = bool(required_ids) and bool(repaired)
        else:
            passed = (
                bool(required_ids)
                and not missing
                and bool(repaired)
                and coverage >= coverage_target
            )
        return {
            "required": True,
            "passed": passed,
            "reason": None if passed else "failure_cluster_not_repaired",
            "required_failure_traces": required_ids,
            "repaired_failure_traces": repaired,
            "missing_failure_traces": missing,
            "non_repaired_failure_traces": non_repaired,
            "repaired_coverage": coverage,
            "required_coverage": coverage_target,
            "acceptance": acceptance,
            "full_rerun_admissible": bool(
                getattr(self.config, "allow_failure_full_rerun", False)
            ),
        }

    @staticmethod
    def _counterfactual_edit_node_id(
        candidate: WorkflowCandidate,
        incumbent: WorkflowTemplate,
    ) -> str | None:
        """Choose the first changed executable node for suffix replay.

        Prompt/operator edits have a stable anchor. A graph-path transaction
        can insert or remove that anchor, however, so blindly passing the
        original id would force an unnecessary full rerun and reject an
        otherwise replayable repair. For graph edits, walk the candidate's
        topological order and use the first surviving changed non-boundary
        node; if no such node exists, fall back to the original anchor.
        """
        if (
            candidate.scope not in {"block", MULTI_SCOPE}
            or candidate.modified_workflow is None
        ):
            return candidate.node_id
        modified = candidate.modified_workflow
        changed = {
            str(node_id)
            for node_id in candidate.changed_units
            if node_id is not None
        }
        boundary_types = {NodeType.START, NodeType.END}
        # A graph transaction that inserts a node *before* the first surviving
        # changed node cannot be resumed from an old trace: the new prefix node
        # has never executed.  Returning ``None`` deliberately forces a full
        # rerun, which the inner failure-repair gate will not promote as suffix
        # evidence.  Insertions after the replay boundary remain replayable.
        added_nodes = set(modified.nodes) - set(incumbent.nodes)
        if added_nodes:
            try:
                modified_order = modified.get_node_order()
            except (AttributeError, ValueError):
                modified_order = list(modified.nodes)
            existing_changed = [
                node_id
                for node_id in modified_order
                if node_id in changed
                and node_id in incumbent.nodes
                and node_id in modified.nodes
                and modified.nodes[node_id].node_type not in boundary_types
            ]
            if not existing_changed:
                return None
            first_existing = min(
                modified_order.index(node_id) for node_id in existing_changed
            )
            if any(
                modified_order.index(node_id) < first_existing
                for node_id in added_nodes
                if node_id in modified_order
            ):
                return None

        try:
            # Use the incumbent order for the first changed executable unit.
            # A multi-level patch may add/remove graph nodes, while the
            # original trace still provides the only valid prefix boundary.
            order = incumbent.get_node_order()
        except (AttributeError, ValueError):
            order = list(incumbent.nodes)
        for node_id in order:
            if node_id in changed and node_id in modified.nodes:
                if modified.nodes[node_id].node_type not in boundary_types:
                    return node_id
        if candidate.node_id in modified.nodes:
            return candidate.node_id
        for node_id in order:
            if (
                node_id in modified.nodes
                and modified.nodes[node_id].node_type
                not in boundary_types
            ):
                return node_id
        return candidate.node_id

    def _select_best_candidate(
        self,
        scored_candidates: list[tuple[WorkflowCandidate, float]],
        *,
        selective: bool,
        failure_repair: bool = False,
    ) -> tuple[WorkflowCandidate, float] | None:
        """Select a workflow patch with the shared scalar acceptance rule.

        ``selective`` remains an API-compatible argument for callers and
        telemetry, but applicability is intentionally not part of patch
        selection.  A gate is fitted only for the winning patch below.
        """
        del selective
        if failure_repair:
            # Correctness has already been established by the per-candidate
            # suffix-replay gate. The conservative/default protocol retains
            # its historical lightest-edit tie-break. Clustered aggressive
            # runs explicitly remove that preference so a multi-level repair
            # can win on reward/token gain.
            if not scored_candidates:
                return None
            if not bool(
                getattr(
                    getattr(self, "config", None),
                    "failure_cluster_enabled",
                    False,
                )
            ):
                return min(
                    scored_candidates,
                    key=lambda item: (
                        float(item[0].edit_distance),
                        -float(item[1]),
                        item[0].scope,
                        item[0].node_id,
                    ),
                )
            return max(
                scored_candidates,
                key=lambda item: (
                    float(item[1]),
                    -float(item[0].edit_distance),
                    item[0].scope,
                    item[0].node_id,
                ),
            )
        ranked = sorted(
            scored_candidates,
            key=lambda item: item[1],
            reverse=True,
        )
        if bool(getattr(self.config, "failure_cluster_enabled", False)):
            # The aggressive research setting may deliberately retain a
            # slightly negative local gain when it buys a repaired path or a
            # useful exploratory graph transaction.  The hard suffix/guard
            # predicates are still applied before this scalar floor.
            floor = float(
                getattr(self.config, "exploration_acceptance_floor", 0.0)
            )
            for candidate, gain in ranked:
                if float(gain) >= floor:
                    return candidate, gain
            return None
        return self.acceptance.select_best(ranked)

    @staticmethod
    def _selection_delta_u(
        result: dict[str, Any],
        trace_ids: set[str],
    ) -> float:
        """Average candidate utility over the workflow-selection batch."""
        rows = result.get("per_query_results")
        if not isinstance(rows, list):
            return float(result.get("delta_u", 0.0))
        values = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if trace_ids and str(row.get("trace_id")) not in trace_ids:
                continue
            try:
                values.append(float(row["delta_u"]))
            except (KeyError, TypeError, ValueError):
                continue
        if values:
            return sum(values) / len(values)
        return float(result.get("delta_u", 0.0))

    @staticmethod
    def _selection_token_delta(
        result: dict[str, Any],
        trace_ids: set[str],
    ) -> float:
        """Average candidate-minus-incumbent token delta for gain scoring.

        Counterfactual rows carry the paired token observations.  Restricting
        this aggregation to the same trigger ids used for ``delta_u`` keeps
        reward and token terms on an identical selection batch.  Missing
        legacy rows fall back to the aggregate ``mean_token_delta`` when it is
        available; otherwise no token penalty is applied.
        """
        rows = result.get("per_query_results")
        if not isinstance(rows, list):
            try:
                return float(result.get("mean_token_delta", 0.0))
            except (TypeError, ValueError, OverflowError):
                return 0.0
        values: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if trace_ids and str(row.get("trace_id")) not in trace_ids:
                continue
            try:
                candidate_tokens = float(row["candidate_total_tokens"])
                original_tokens = float(row["original_total_tokens"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(candidate_tokens) or not math.isfinite(
                original_tokens
            ):
                continue
            values.append(candidate_tokens - original_tokens)
        if values:
            return sum(values) / len(values)
        try:
            return float(result.get("mean_token_delta", 0.0))
        except (TypeError, ValueError, OverflowError):
            return 0.0

    def _token_gain_weight(self) -> float:
        """Resolve the token coefficient for telemetry/legacy scorers."""
        raw = getattr(self.scorer, "lambda_tokens", None)
        if raw is None:
            raw = getattr(self.config, "lambda_cost", 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return value if math.isfinite(value) and value >= 0.0 else 0.0

    @staticmethod
    def _select_gate_fit_traces(
        traces: list[ExecutionTrace],
        *,
        max_traces: int,
    ) -> list[ExecutionTrace]:
        """Deterministically bound the representative optimization epoch."""
        if max_traces <= 0:
            return []
        unique = LLMWorkflowOptimizer._unique_traces(traces)
        ranked = sorted(
            unique,
            key=lambda trace: (
                trace.trace_id,
            ),
        )
        return ranked[:max_traces]

    @staticmethod
    def _unique_traces(
        traces: list[ExecutionTrace],
    ) -> list[ExecutionTrace]:
        """Preserve first occurrence while de-duplicating exact trace ids."""
        result: list[ExecutionTrace] = []
        seen: set[str] = set()
        for trace in traces:
            if trace.trace_id in seen:
                continue
            seen.add(trace.trace_id)
            result.append(trace)
        return result

    def _trace_efficiency_cost(self, trace: ExecutionTrace) -> float:
        """Return the external efficiency-anchor cost for one trace.

        This signal is used only to identify unusually expensive successful
        traces for the optional efficiency ablation.  It is not utility and
        is not part of the candidate gain, whose only resource term is the
        paired token delta.
        """
        utility = self.utility_computer
        return float(
            utility.lambda_cost * utility.compute_execution_cost(trace)
            + utility.lambda_latency * utility.compute_llm_latency(trace)
            + utility.lambda_api_cost * utility.compute_known_api_cost(trace)
            + utility.rho_omega
            * utility.compute_runtime_complexity(trace)
        )

    def _localize_efficiency_anchors(
        self,
        traces: list[ExecutionTrace],
        workflow: WorkflowTemplate,
        *,
        top_m: int,
    ) -> list[dict[str, Any]]:
        """Localize high-cost successes deterministically by node cost share."""
        if not traces or top_m <= 0:
            return []

        utility = self.utility_computer
        aggregate: dict[str, dict[str, Any]] = {}
        for trace in traces:
            per_node: dict[str, float] = {}
            editable_steps = [
                step
                for step in trace.steps
                if (
                    step.node_id in workflow.nodes
                    and workflow.nodes[step.node_id].node_type.value
                    not in {"start", "end"}
                    and step.metadata.get("node_executed", True)
                )
            ]
            visit_counts: dict[str, int] = {}
            for step in editable_steps:
                prior_visits = visit_counts.get(step.node_id, 0)
                visit_counts[step.node_id] = (
                    prior_visits + 1
                )
                all_calls = step.llm_calls
                calls = [
                    call
                    for call in all_calls
                    if call.call_type != "scheduler"
                ]
                tokens = sum(
                    call.prompt_tokens + call.completion_tokens
                    for call in calls
                )
                if not all_calls:
                    tokens = step.input_tokens + step.output_tokens
                latency = sum(
                    max(float(call.latency_seconds), 0.0)
                    for call in calls
                )
                api_cost = sum(
                    max(float(call.cost_usd), 0.0) for call in calls
                )
                action = step.action.lower()
                complexity = (
                    utility.beta_repair
                    * float(action in {"repair", "retry"})
                    + utility.beta_reroute
                    * float(action in {"reroute", "branch"})
                    + utility.beta_fallback
                    * float(action in {"fallback", "deviate"})
                    + utility.beta_loop * float(prior_visits > 0)
                )
                contribution = (
                    utility.lambda_cost * float(tokens)
                    + utility.lambda_latency * latency
                    + utility.lambda_api_cost * api_cost
                    + utility.rho_omega * complexity
                )
                per_node[step.node_id] = (
                    per_node.get(step.node_id, 0.0) + contribution
                )

            # Aggregate-only legacy traces may lack per-call data. Allocate the
            # observed total penalty evenly rather than inventing a hot node.
            total = sum(per_node.values())
            has_call_level_data = any(
                step.llm_calls for step in editable_steps
            )
            if (
                total <= 0.0
                and editable_steps
                and not has_call_level_data
            ):
                total_penalty = self._trace_efficiency_cost(trace)
                unique_nodes = sorted({step.node_id for step in editable_steps})
                if total_penalty > 0.0:
                    equal_share = total_penalty / len(unique_nodes)
                    per_node = {
                        node_id: equal_share for node_id in unique_nodes
                    }
                    total = total_penalty
            if total <= 0.0:
                continue

            for node_id, contribution in per_node.items():
                record = aggregate.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "unit_id": node_id,
                        "score": 0.0,
                        "evidence_count": 0,
                        "trigger_types": {"high_cost_success"},
                        "reasons": [],
                    },
                )
                share = contribution / total
                record["score"] += share
                record["evidence_count"] += 1
                record["reasons"].append(
                    f"configured runtime-cost share {share:.1%}"
                )

        ranked = sorted(
            aggregate.values(),
            key=lambda item: (
                -item["score"],
                -item["evidence_count"],
                item["node_id"],
            ),
        )[:top_m]
        for rank, anchor in enumerate(ranked, start=1):
            anchor["rank"] = rank
            anchor["reason"] = "; ".join(anchor.pop("reasons")[:3])
            anchor["trigger_types"] = sorted(anchor["trigger_types"])
        return ranked

    @staticmethod
    def _merge_anchors(
        failure_anchors: list[dict[str, Any]],
        efficiency_anchors: list[dict[str, Any]],
        *,
        top_m: int,
    ) -> list[dict[str, Any]]:
        """Merge correctness and efficiency evidence by workflow node."""
        merged: dict[str, dict[str, Any]] = {}
        for trigger, anchors in (
            ("hard_failure", failure_anchors),
            ("high_cost_success", efficiency_anchors),
        ):
            for anchor in anchors:
                node_id = str(anchor["node_id"])
                target = merged.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "unit_id": node_id,
                        "score": 0.0,
                        "evidence_count": 0,
                        "reasons": [],
                        "trigger_types": set(),
                    },
                )
                score = LLMWorkflowOptimizer._safe_float(
                    anchor.get("score")
                )
                target["score"] += score if score is not None else 0.0
                target["evidence_count"] += int(
                    anchor.get("evidence_count", 1)
                )
                reason = str(anchor.get("reason", "")).strip()
                if reason:
                    target["reasons"].append(reason)
                target["trigger_types"].add(trigger)

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                -item["score"],
                -item["evidence_count"],
                item["node_id"],
            ),
        )[:top_m]
        for rank, anchor in enumerate(ranked, start=1):
            anchor["rank"] = rank
            anchor["reason"] = "; ".join(anchor.pop("reasons")[:4])
            anchor["trigger_types"] = sorted(anchor["trigger_types"])
            anchor["confidence"] = anchor["score"]
        return ranked

    @staticmethod
    def _hard_success_guard(
        result: dict[str, Any],
        protected_trace_ids: set[str],
        *,
        required: bool,
    ) -> dict[str, Any]:
        """Fail closed if an efficiency update regresses any sampled success."""
        if not required:
            return {
                "required": False,
                "passed": True,
                "protected_traces": 0,
                "regressions": 0,
                "missing_measurements": 0,
            }

        rows = result.get("per_query_results")
        by_id = {
            str(row.get("trace_id")): row
            for row in rows
            if isinstance(row, dict) and row.get("trace_id") is not None
        } if isinstance(rows, list) else {}
        regressions = 0
        missing = 0
        for trace_id in protected_trace_ids:
            row = by_id.get(str(trace_id))
            if row is None:
                missing += 1
                continue
            original = LLMWorkflowOptimizer._safe_float(
                row.get("original_hard")
            )
            candidate = LLMWorkflowOptimizer._safe_float(
                row.get("candidate_hard")
            )
            if original is None or candidate is None:
                missing += 1
            elif candidate < original - 1e-12:
                regressions += 1
        return {
            "required": True,
            "passed": regressions == 0 and missing == 0,
            "protected_traces": len(protected_trace_ids),
            "regressions": regressions,
            "missing_measurements": missing,
        }

    def _finish_round(
        self,
        workflow: WorkflowTemplate,
        summary: dict,
    ) -> tuple[WorkflowTemplate, dict]:
        archive = getattr(self, "candidate_archive", None)
        if archive is not None and archive.enabled:
            summary["candidate_archive"] = archive.snapshot()
        self.round_history.append(summary)
        return workflow, summary

    def _build_failure_context(
        self,
        failure_traces: list[ExecutionTrace],
        success_traces: list[ExecutionTrace] | None = None,
        *,
        efficiency_traces: list[ExecutionTrace] | None = None,
    ) -> str:
        """Summarize correctness and efficiency triggers without held-out data."""
        self._ensure_optimization_traces(
            failure_traces
            + list(efficiency_traces or [])
            + list(success_traces or [])
        )
        lines = [f"Current-version failures: {len(failure_traces)}"]
        # Cluster size is already bounded by the protocol. Expose every member
        # and all semantically relevant math-node artifacts to the optimizer
        # LLM so a shared update is grounded in the actual failure family.
        for index, trace in enumerate(failure_traces, start=1):
            diagnostics = self._structured_failure_diagnostics(trace)
            lines.extend(
                [
                    f"\nFailure {index}:",
                    f"  Query: {self._preview(trace.query_text, 200)}",
                    f"  Final output / prediction: "
                    f"{self._preview(trace.final_output, 500)}",
                    f"  Expected (safe summary): "
                    f"{self._preview(diagnostics['expected'], 300)}",
                    f"  Hard/process reward: "
                    f"{trace.hard_reward}/{trace.process_reward}",
                    f"  Steps: {trace.total_steps}",
                    f"  Failure classification: "
                    f"{diagnostics['failure_kind']}",
                    f"  Evaluator/task diagnostics: "
                    f"{self._preview(diagnostics['details'], 1200)}",
                ]
            )
            errors = diagnostics["errors"]
            if errors:
                lines.append(f"  Errors: {self._preview(errors, 500)}")
            relevant_steps = [
                step
                for step in trace.steps
                if step.node_id in {
                    "analyze", "solve", "self_refine", "dual_solve",
                    "dual_solve_judge", "verify", "finalize",
                }
                or not step.success
            ]
            for step in relevant_steps:
                lines.append(
                    f"  Step {step.step_index} ({step.node_id}): "
                    f"success={step.success}, "
                    f"state={self._preview(step.state_after, 700)}"
                )

        efficiency_references = efficiency_traces or []
        if efficiency_references:
            lines.append(
                "\nHigh-cost hard-success efficiency anchors: "
                f"{len(efficiency_references)}"
            )
            lines.append(
                "Preserve hard reward exactly while reducing configured "
                "runtime cost (tokens, latency, API cost, or complexity)."
            )
            for trace in efficiency_references[:5]:
                breakdown = trace.metadata.get("utility_breakdown", {})
                lines.append(
                    f"  Query={self._preview(trace.query_text, 120)} "
                    f"hard={trace.hard_reward} "
                    f"tokens={trace.total_tokens} "
                    f"llm_latency_seconds="
                    f"{self.utility_computer.compute_llm_latency(trace):.6g} "
                    f"configured_cost_penalty="
                    f"{self._trace_efficiency_cost(trace):.6g} "
                    f"breakdown={self._preview(breakdown, 350)}"
                )

        references = success_traces or []
        if references:
            lines.append(
                f"\nSuccessful regression guards: {len(references)}"
            )
            for trace in references[:3]:
                lines.append(
                    f"  Query={self._preview(trace.query_text, 120)} "
                    f"output={self._preview(trace.final_output, 180)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _ensure_optimization_traces(
        traces: list[ExecutionTrace],
    ) -> None:
        """Fail closed before optimizer LLMs can see held-out examples."""
        invalid = sorted(
            {
                (
                    "<missing>"
                    if trace.metadata.get("split") is None
                    else str(trace.metadata.get("split"))
                )
                for trace in traces
                if trace.metadata.get("split") != "optimization"
            }
        )
        if invalid:
            raise ValueError(
                "workflow optimizer context accepts only optimization-split "
                f"traces; received: {', '.join(invalid)}"
            )

    def _structured_failure_diagnostics(
        self,
        trace: ExecutionTrace,
    ) -> dict[str, Any]:
        """Build bounded benchmark diagnostics with sensitive labels redacted."""
        details: dict[str, Any] = {}
        ground_truth = trace.metadata.get("ground_truth")
        sensitive_artifacts = self._sensitive_artifact_texts(ground_truth)

        def sanitized(value: object) -> object:
            return self._sanitize_diagnostic_value(
                self._redact_sensitive_value(value, sensitive_artifacts)
            )

        code_evaluation = trace.metadata.get("code_evaluation")
        if isinstance(code_evaluation, dict):
            details["code_evaluation"] = {
                key: sanitized(code_evaluation.get(key))
                for key in (
                    "passed_tests",
                    "total_tests",
                    "tests_executed",
                    "pass_rate",
                    "summary",
                )
                if key in code_evaluation
            }
        code_output_contract = trace.metadata.get("code_output_contract")
        if isinstance(code_output_contract, dict):
            details["code_output_contract"] = sanitized(
                code_output_contract
            )

        for key in (
            "math_evaluation",
            "reward_diagnostics",
            "evaluator_diagnostics",
        ):
            value = trace.metadata.get(key)
            if value is not None:
                details[key] = sanitized(value)

        errors = [str(trace.error_message)] if trace.error_message else []
        errors.extend(
            str(step.error_message)
            for step in trace.steps
            if step.error_message
        )
        metadata_error = trace.metadata.get("error_message")
        if metadata_error:
            errors.append(str(metadata_error))
        errors = [
            self._bounded_text(
                self._redact_sensitive_value(error, sensitive_artifacts),
                500,
            )
            for error in errors
        ]

        return {
            "failure_kind": self._classify_failure(
                trace,
                code_evaluation,
                code_output_contract,
                errors,
            ),
            "expected": self._safe_expected_summary(
                ground_truth,
            ),
            "details": details,
            "errors": errors,
        }

    @staticmethod
    def _classify_failure(
        trace: ExecutionTrace,
        code_evaluation: object,
        code_output_contract: object,
        errors: list[str],
    ) -> str:
        metadata = trace.metadata
        if any(
            metadata.get(key)
            for key in ("evaluator_error", "reward_evaluation_error")
        ):
            return "evaluator_failure"

        scheduler_failed = any(
            call.call_type == "scheduler" and not call.success
            for step in trace.steps
            for call in step.llm_calls
        )
        if scheduler_failed or any(
            "scheduler" in error.lower() for error in errors
        ):
            return "scheduler_failure"

        infrastructure_markers = (
            "rate limit",
            "authentication",
            "api connection",
            "service unavailable",
            "provider",
        )
        if any(
            marker in error.lower()
            for error in errors
            for marker in infrastructure_markers
        ):
            return "infrastructure_failure"

        if isinstance(code_evaluation, dict):
            total = LLMWorkflowOptimizer._safe_int(
                code_evaluation.get("total_tests")
            )
            executed = LLMWorkflowOptimizer._safe_int(
                code_evaluation.get("tests_executed")
            )
            passed = LLMWorkflowOptimizer._safe_int(
                code_evaluation.get("passed_tests")
            )
            if total > 0 and executed == 0:
                return "evaluator_failure"
            if total > 0 and passed < total:
                return "task_failure"

        if (
            isinstance(code_output_contract, dict)
            and not code_output_contract.get("valid", False)
        ):
            return "output_contract_failure"

        if errors or not trace.success:
            return "workflow_runtime_failure"
        if trace.hard_reward is not None and trace.hard_reward < 1.0:
            return "task_failure"
        return "unknown_failure"

    def _safe_expected_summary(self, ground_truth: object) -> object:
        """Expose task labels while replacing code/test artifacts by metadata."""
        if not isinstance(ground_truth, dict):
            return self._bounded_text(ground_truth, 300)

        safe: dict[str, Any] = {}
        scalar_keys = (
            "task_id",
            "entry_point",
            "answer",
            "expected",
            "target",
            "label",
            "type",
            "domain",
            "level",
            "source_split",
            "source_index",
        )
        for key in scalar_keys:
            if key in ground_truth:
                safe[key] = self._sanitize_diagnostic_value(
                    ground_truth[key]
                )

        sensitive_artifacts = {
            "test": "withheld_test_suite",
            "tests": "withheld_tests",
            "test_list": "withheld_test_list",
            "canonical_solution": "reference_solution",
            "solution": "reference_solution",
            "code": "reference_solution",
        }
        for key, label in sensitive_artifacts.items():
            if key in ground_truth and ground_truth[key]:
                safe[label] = self._artifact_summary(ground_truth[key])

        other_keys = sorted(
            str(key)
            for key in ground_truth
            if key not in scalar_keys and key not in sensitive_artifacts
        )
        if other_keys:
            safe["other_fields"] = other_keys[:20]
        return safe

    @classmethod
    def _sanitize_diagnostic_value(
        cls,
        value: object,
        *,
        depth: int = 0,
    ) -> object:
        if depth >= 3:
            return cls._bounded_text(value, 200)
        if isinstance(value, dict):
            sensitive = {
                "test",
                "tests",
                "test_list",
                "canonical_solution",
                "solution",
                "code",
            }
            return {
                str(key): (
                    cls._artifact_summary(item)
                    if str(key).lower() in sensitive
                    else cls._sanitize_diagnostic_value(
                        item,
                        depth=depth + 1,
                    )
                )
                for key, item in list(value.items())[:30]
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._sanitize_diagnostic_value(item, depth=depth + 1)
                for item in list(value)[:20]
            ]
        return cls._bounded_text(value, 800)

    @staticmethod
    def _artifact_summary(value: object) -> dict[str, Any]:
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            serialized = repr(value)
        return {
            "present": True,
            "kind": type(value).__name__,
            "characters": len(serialized),
            "content_prefix": serialized[:12],
        }

    @staticmethod
    def _sensitive_artifact_texts(ground_truth: object) -> tuple[str, ...]:
        """Collect exact source/reference strings that prompts must not echo."""
        if not isinstance(ground_truth, dict):
            return ()
        texts: set[str] = set()

        def collect(value: object) -> None:
            if isinstance(value, str):
                if value:
                    texts.add(value)
                    stripped = value.strip()
                    if stripped:
                        texts.add(stripped)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        for key in (
            "test",
            "tests",
            "test_list",
            "canonical_solution",
            "solution",
            "code",
        ):
            if key not in ground_truth:
                continue
            artifact = ground_truth[key]
            collect(artifact)
            try:
                texts.add(
                    json.dumps(
                        artifact,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
            except (TypeError, ValueError):
                pass
        return tuple(sorted((text for text in texts if text), key=len, reverse=True))

    @classmethod
    def _redact_sensitive_value(
        cls,
        value: object,
        sensitive_artifacts: tuple[str, ...],
    ) -> object:
        """Recursively replace exact withheld source artifacts by fingerprints."""
        if isinstance(value, dict):
            return {
                key: cls._redact_sensitive_value(item, sensitive_artifacts)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._redact_sensitive_value(item, sensitive_artifacts)
                for item in value
            ]
        if not isinstance(value, str):
            return value
        redacted = value
        for artifact in sensitive_artifacts:
            if artifact not in redacted:
                continue
            marker = (
                "<redacted_ground_truth_artifact>"
            )
            redacted = redacted.replace(artifact, marker)
        return redacted

    def _build_experience_context(self) -> str:
        """Return a compact, aggregate-only summary of the previous round."""
        history = getattr(self, "round_history", [])
        if not history:
            return ""
        previous = history[-1]
        candidates = previous.get("candidate_evaluations", [])
        if not isinstance(candidates, list) or not candidates:
            return ""

        lines = [
            f"Previous round {previous.get('round', '?')} candidate outcomes.",
            "Do not repeat an exact patch fingerprint. Treat noisy outcomes as "
            "evidence, not absolute rules.",
        ]
        confirmation = previous.get("optimization_confirmation")
        confirmation_passed: bool | None = None
        if isinstance(confirmation, dict) and confirmation.get("performed"):
            confirmation_passed = bool(confirmation.get("passed"))
        for record in candidates[:8]:
            if not isinstance(record, dict):
                continue
            status = str(record.get("status", "unknown"))
            if status == "duplicate_history":
                continue
            if status == "selected":
                if confirmation_passed is True:
                    label = "positive_full_opt_confirmation"
                elif confirmation_passed is False:
                    label = "negative_full_opt_confirmation"
                else:
                    label = "positive_counterfactual"
            elif status in {"sufficient", "not_final_selected"}:
                label = "promising_not_selected"
            else:
                label = "negative_or_failed"
            effects = self._candidate_effect_summary(
                record.get("per_query_results")
            )
            lines.append(
                "- "
                f"{label}; status={status}; "
                f"patch={str(record.get('patch_fingerprint', ''))[:16]}; "
                f"anchor={record.get('anchor_id')}; "
                f"scope={record.get('scope')}; "
                f"gain={record.get('gain')}; "
                f"effects={self._preview(effects, 300)}; "
                f"description={self._bounded_text(record.get('description'), 220)}"
            )

        if isinstance(confirmation, dict) and confirmation.get("performed"):
            compact_confirmation = {
                key: confirmation.get(key)
                for key in (
                    "decision",
                    "reason",
                    "completed_repeats",
                    "hard_reward_effect",
                    "mean_utility_effect",
                    "hard_non_regression_passed",
                    "mean_utility_effect_passed",
                )
                if key in confirmation
            }
            lines.append(
                "Full optimization-split confirmation (aggregate only): "
                + self._preview(compact_confirmation, 500)
            )

        archive = getattr(self, "candidate_archive", None)
        if archive is not None and archive.enabled:
            archive_context = archive.experience_context(limit=8)
            if archive_context:
                lines.append(archive_context)

        # Deliberately ignore ``is_best`` and ``validation_metrics``. Validation
        # may select/checkpoint a workflow but must not become adaptive training
        # feedback for the next optimizer prompt.
        return self._bounded_text("\n".join(lines), 5000)

    @staticmethod
    def _candidate_effect_summary(value: object) -> dict[str, Any]:
        if not isinstance(value, list):
            return {}
        rows = [item for item in value if isinstance(item, dict)]
        if not rows:
            return {}
        hard_pairs = [
            pair
            for row in rows
            if (
                pair := LLMWorkflowOptimizer._numeric_pair(
                    row,
                    "candidate_hard",
                    "original_hard",
                )
            )
            is not None
        ]
        hard_fixes = sum(candidate > original for candidate, original in hard_pairs)
        hard_regressions = sum(
            candidate < original for candidate, original in hard_pairs
        )
        token_deltas = LLMWorkflowOptimizer._numeric_deltas(
            rows,
            "candidate_total_tokens",
            "original_total_tokens",
        )
        latency_deltas = LLMWorkflowOptimizer._numeric_deltas(
            rows,
            "candidate_latency_seconds",
            "original_latency_seconds",
        )
        return {
            "queries": len(rows),
            "hard_fixes": hard_fixes,
            "hard_regressions": hard_regressions,
            "mean_token_delta": (
                sum(token_deltas) / len(token_deltas)
                if token_deltas
                else None
            ),
            "mean_latency_delta_seconds": (
                sum(latency_deltas) / len(latency_deltas)
                if latency_deltas
                else None
            ),
        }

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _safe_int(value: object) -> int:
        number = LLMWorkflowOptimizer._safe_float(value)
        return int(number) if number is not None else 0

    @staticmethod
    def _numeric_pair(
        row: dict[str, Any],
        left_key: str,
        right_key: str,
    ) -> tuple[float, float] | None:
        left = LLMWorkflowOptimizer._safe_float(row.get(left_key))
        right = LLMWorkflowOptimizer._safe_float(row.get(right_key))
        if left is None or right is None:
            return None
        return left, right

    @staticmethod
    def _numeric_deltas(
        rows: list[dict[str, Any]],
        candidate_key: str,
        original_key: str,
    ) -> list[float]:
        pairs = [
            pair
            for row in rows
            if (
                pair := LLMWorkflowOptimizer._numeric_pair(
                    row,
                    candidate_key,
                    original_key,
                )
            )
            is not None
        ]
        return [candidate - original for candidate, original in pairs]

    @staticmethod
    def _bounded_text(value: object, limit: int) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def _preview(value: object, limit: int) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    @staticmethod
    def _bump_version(version: str) -> str:
        """Increment the final numeric version component."""
        try:
            parts = version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except (ValueError, IndexError):
            return version + ".1"
