"""ExperimentRunner — algorithm-level optimization loop.

Manages the full experiment lifecycle:
1. Split data into opt/val/test
2. Optimization loop: execute → collect traces → optimize workflow
   (or stream failures into immediate suffix-replay updates)
3. Periodic validation on held-out val set
4. Persist a manifest for a separate, one-shot held-out test process

Failure updates can be scheduled in three modes: one batch update after an
epoch, deferred updates after the complete epoch (optionally clustered into
multi-failure transactions), or the legacy streaming online ablation.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from awf.config.schema import ExperimentConfig, SchedulerConfig
from awf.executor.runtime import RuntimeExecutor
from awf.llm.client import AsyncLLMClient
from awf.optimizer.workflow_optimizer import LLMWorkflowOptimizer
from awf.protocol.checkpoint import CheckpointManager
from awf.protocol.manifest import (
    atomic_write_json_0600,
    bind_best_checkpoint,
    create_manifest,
    public_scientific_config,
)
from awf.protocol.split_manager import SplitManager
from awf.reward.base import RewardEvaluator
from awf.scheduler.base import BaseScheduler
from awf.scheduler.cascade_scheduler import CascadeScheduler
from awf.scheduler.calibration import FrozenWorkflowGateCalibrator
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.scheduler.graph_scheduler import GraphScheduler
from awf.trace.schema import ExecutionTrace
from awf.utility.compute import UtilityComputer
from awf.workflow.ir import WorkflowTemplate

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs the full optimization experiment.

    Coordinates data splits, optimization rounds, validation,
    and final test evaluation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        workflow: WorkflowTemplate,
        reward_evaluator: RewardEvaluator,
        operators: Optional[dict[str, callable]] = None,
        run_metadata: Optional[dict[str, Any]] = None,
    ):
        self.config = config
        self.workflow = workflow
        self.reward_evaluator = reward_evaluator
        self.operators = operators or {}
        self.run_metadata = dict(run_metadata or {})
        if (
            config.optimizer.workflow_content_only
            and self.workflow.selective_update is not None
        ):
            # The current experiment line measures inner workflow-content
            # learning only.  Ignore a policy checkpoint from an older
            # outer-layer ablation and execute its base workflow directly.
            payload = self.workflow.model_dump(mode="json")
            payload["selective_update"] = None
            self.workflow = WorkflowTemplate.model_validate(payload)
            logger.info(
                "workflow_content_only=True; discarded an inherited "
                "selective execution policy"
            )
        if config.scheduler.llm.seed is None:
            config.scheduler.llm.seed = config.seed
        workflow_llm_config = config.workflow_llm or config.scheduler.llm
        if workflow_llm_config.seed is None:
            workflow_llm_config.seed = config.seed
        if config.optimizer.llm.seed is None:
            config.optimizer.llm.seed = config.seed + 1

        # Ensure output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create components
        self.split_manager = SplitManager(
            seed=config.seed,
            opt_ratio=config.opt_split_ratio,
            val_ratio=config.val_split_ratio,
            test_ratio=config.test_split_ratio,
        )

        self.llm_client = AsyncLLMClient(workflow_llm_config)

        self.executor = RuntimeExecutor(
            config=config.executor,
            operators=self.operators,
        )

        scheduler_config = config.scheduler
        if (
            config.optimizer.workflow_content_only
            and scheduler_config.scheduler_type != "cascade"
            and scheduler_config.allow_deviation
        ):
            # Scheduler deviation is an outer execution/path intervention. It
            # is disabled for inner-only runs while retaining the fixed/graph
            # scheduler as the execution substrate.
            scheduler_config.allow_deviation = False
            logger.info(
                "workflow_content_only=True; scheduler deviation disabled"
            )
        # In the bilevel protocol the workflow action space is learned first
        # under a frozen template-following scheduler.  The configured cascade
        # is instantiated only in its dedicated calibration round.
        if config.scheduler_calibration_round is not None:
            self.scheduler = GraphScheduler(
                SchedulerConfig(
                    scheduler_type="graph",
                    llm=scheduler_config.llm.model_copy(deep=True),
                    allow_deviation=False,
                )
            )
        else:
            self.scheduler = self._create_scheduler(scheduler_config)

        self.utility_computer = UtilityComputer(
            lambda_cost=config.optimizer.lambda_cost,
            lambda_latency=config.optimizer.lambda_latency,
            lambda_api_cost=config.optimizer.lambda_api_cost,
            rho_omega=config.optimizer.rho_omega,
        )

        self.optimizer = LLMWorkflowOptimizer(
            config=config.optimizer,
            reward_evaluator=reward_evaluator,
            utility_computer=self.utility_computer,
            reward_alpha=config.reward.alpha_process,
            execution_llm_defaults={
                "model": workflow_llm_config.model,
                "temperature": workflow_llm_config.temperature,
                "max_tokens": workflow_llm_config.max_tokens,
            },
        )

        # One promotion threshold governs validation gating, checkpointing,
        # and early-stopping progress.  The legacy early-stopping field remains
        # a conservative lower bound for existing non-default configurations.
        self._validation_min_delta = max(
            config.validation_min_delta,
            config.early_stopping_min_delta,
        )
        # The metric that drives checkpoint "best" selection and promotion.
        # With hard_success_priority the workflow update is accuracy-first.
        self._selection_metric_key = (
            "hard_reward" if config.hard_success_priority else "runtime_utility"
        )
        self.checkpoint = CheckpointManager(
            output_dir=self.output_dir / "checkpoints",
            maximize=True,
            min_delta=self._validation_min_delta,
        )

        # Data splits
        self.opt_data: list[tuple[str, Any]] = []
        self.val_data: list[tuple[str, Any]] = []
        self.test_data: list[tuple[str, Any]] = []
        self.split_indices: dict[str, list[int]] = {}
        self.manifest: Optional[dict[str, Any]] = None
        self._initial_workflow = workflow.model_copy(deep=True)
        self._persist_traces = True
        self._current_round: int | str = 0
        # Trace files append across rounds within this runner, but the first
        # write for each split truncates artifacts left by an older run.
        self._initialized_trace_splits: set[str] = set()

    @staticmethod
    def _create_scheduler(config: SchedulerConfig) -> BaseScheduler:
        """Factory method for creating the scheduler."""
        if config.scheduler_type == "fixed":
            return FixedScheduler()
        if config.scheduler_type == "graph":
            return GraphScheduler(config)
        if config.scheduler_type == "cascade":
            return CascadeScheduler(config)
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")

    @staticmethod
    def _outer_scheduler_enabled(config: ExperimentConfig) -> bool:
        """Whether an outer execution policy is active for this run.

        ``workflow_content_only`` disables learned workflow-policy updates,
        but a cascade scheduler is an explicit execution-layer intervention
        and therefore remains enabled for the dual-layer ablation.
        """
        scheduler = getattr(config, "scheduler", None)
        optimizer = getattr(config, "optimizer", None)
        return bool(
            getattr(scheduler, "scheduler_type", None) == "cascade"
            or not getattr(optimizer, "workflow_content_only", True)
        )

    def load_data(
        self,
        data: list[tuple[str, Any]],
        key_fn: Optional[callable] = None,
        dataset_source_path: str | Path | None = None,
    ) -> None:
        """Load and split the dataset.

        Args:
            data: List of (query, ground_truth) pairs.
            key_fn: Optional function to extract stable keys for splitting.
        """
        if not data:
            raise ValueError("Experiment dataset is empty")

        # Split indexed rows so the manifest records exact membership in the
        # original ordered dataset, including when ``key_fn`` first reorders
        # rows or when duplicate values are present.
        indexed_data = list(enumerate(data))
        effective_key_fn = key_fn or self._default_split_key
        indexed_key_fn = lambda item: effective_key_fn(item[1])
        if self.config.split_source_field:
            opt_rows, val_rows, test_rows = self.split_manager.split_source(
                indexed_data,
                field=self.config.split_source_field,
                dev_value=self.config.split_dev_value,
                test_value=self.config.split_test_value,
                key_fn=indexed_key_fn,
                reuse_dev=self.config.split_validate_reuse,
            )
        else:
            opt_rows, val_rows, test_rows = self.split_manager.split(
                indexed_data,
                key_fn=indexed_key_fn,
            )
        self.split_indices = {
            "optimization": [index for index, _ in opt_rows],
            "validation": [index for index, _ in val_rows],
            "test": [index for index, _ in test_rows],
        }
        self.opt_data = [item for _, item in opt_rows]
        self.val_data = [item for _, item in val_rows]
        self.test_data = [item for _, item in test_rows]
        required_splits = {
            "optimization": (self.config.opt_split_ratio, self.opt_data),
            "validation": (self.config.val_split_ratio, self.val_data),
            "test": (self.config.test_split_ratio, self.test_data),
        }
        empty = [
            name
            for name, (ratio, items) in required_splits.items()
            if ratio > 0 and not items
        ]
        if empty:
            raise ValueError(
                "Dataset is too small for non-empty "
                + "/".join(empty)
                + " splits"
            )
        logger.info(
            f"Data split: opt={len(self.opt_data)}, "
            f"val={len(self.val_data)}, test={len(self.test_data)}"
        )
        self.manifest = create_manifest(
            config=self.config,
            data=data,
            split_indices=self.split_indices,
            initial_workflow=self._initial_workflow,
            run_metadata=self.run_metadata,
            dataset_source_path=dataset_source_path,
        )

    @staticmethod
    def _default_split_key(item: tuple[str, Any]) -> str:
        """Group duplicate task ids/queries to prevent cross-split leakage."""
        query, ground_truth = item
        if isinstance(ground_truth, dict):
            task_id = ground_truth.get("task_id")
            if task_id is not None and str(task_id).strip():
                return f"task_id:{task_id}"
        return f"query:{query}"

    async def run(self) -> dict[str, Any]:
        """Run the full optimization experiment.

        Returns:
            Experiment results dict.
        """
        results = {
            "config": self._public_config(),
            "workflow_content_only": bool(
                getattr(self.config.optimizer, "workflow_content_only", True)
            ),
            "outer_execution_optimization_enabled": self._outer_scheduler_enabled(
                self.config
            ),
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "best_val_score": None,
            "test_score": None,
            "test_metrics": None,
            "official_test_metrics": None,
            "test_evaluated_at": None,
            "official_test_evaluated": False,
            "backend_usage": None,
            "stopped_early": False,
            "early_stop_round": None,
            "early_stop_reason": None,
        }
        if self.manifest is None:
            raise RuntimeError("load_data() must be called before run()")

        aggressive_local = (
            getattr(self.config, "promotion_mode", "validation")
            == "counterfactual_local_aggressive"
        )
        results["promotion_mode"] = (
            "counterfactual_local_aggressive"
            if aggressive_local
            else "validation"
        )

        # The aggressive protocol intentionally does not execute the full
        # validation split.  Round-0 is still persisted as a local baseline so
        # the one-shot held-out test protocol can bind a checkpoint.
        if aggressive_local:
            baseline_metrics = None
            results["baseline_backend_usage_delta"] = {
                "skipped": True,
                "reason": "counterfactual_local_aggressive",
            }
            results["baseline_validation_metrics"] = None
            self.checkpoint.promote_local(
                self.workflow,
                round_num=0,
                score=None,
                metadata={
                    "baseline": True,
                    "promotion_mode": "counterfactual_local_aggressive",
                },
            )
            incumbent_validation_metrics = None
        else:
            # Round 0 is the immutable baseline. This lets validation select
            # the original workflow if every optimization update regresses.
            baseline_usage_before = self._backend_usage_snapshot()
            baseline_metrics = await self._evaluate_dataset(
                self.workflow, self.val_data, split_name="validation_baseline"
            )
            results["baseline_backend_usage_delta"] = (
                self._backend_usage_delta(
                    baseline_usage_before,
                    self._backend_usage_snapshot(),
                )
            )
            baseline_score = baseline_metrics[self._selection_metric_key]
            self.checkpoint.update(
                self.workflow,
                baseline_score,
                round_num=0,
                metadata={"validation_metrics": baseline_metrics, "baseline": True},
            )
            results["baseline_validation_metrics"] = baseline_metrics
            incumbent_validation_metrics = baseline_metrics
        rounds_without_improvement = 0

        # Optimization loop
        for round_num in range(1, self.config.optimizer.max_rounds + 1):
            logger.info(f"=== Optimization Round {round_num} ===")
            self._current_round = round_num
            round_usage_before = self._backend_usage_snapshot()
            round_start_workflow = self.workflow.model_copy(deep=True)

            if round_num == self.config.scheduler_calibration_round:
                round_summary = await self._run_scheduler_calibration_round(
                    round_num
                )
                round_summary["backend_usage_delta"] = self._backend_usage_delta(
                    round_usage_before,
                    self._backend_usage_snapshot(),
                )
                results["rounds"].append(round_summary)
                results["scheduler_calibration"] = round_summary.get(
                    "scheduler_calibration"
                )
                continue
            workflow_rounds = self.config.workflow_optimization_rounds
            if workflow_rounds is not None and round_num > workflow_rounds:
                # No joint updates: after the frozen-workflow calibration
                # round, later optimizer rounds have no research role.
                break

            failure_update_mode = self._failure_update_mode()
            online_mode = failure_update_mode == "online"
            deferred_mode = failure_update_mode == "deferred_sequential"
            round_traces, opt_metrics = await self._run_optimization_epoch()

            if online_mode:
                updated_workflow = self.workflow
                online_updates = list(
                    getattr(self, "_online_round_updates", [])
                )
                counterfactual_accepted = any(
                    item.get("accepted", False) for item in online_updates
                )
                round_summary = self._summarize_online_round(
                    round_num,
                    round_start_workflow,
                    updated_workflow,
                    online_updates,
                )
            elif deferred_mode:
                updated_workflow, round_summary = (
                    await self._run_deferred_failure_updates(
                        round_num,
                        round_start_workflow,
                        round_traces,
                    )
                )
                deferred_updates = list(
                    getattr(self, "_deferred_round_updates", [])
                )
                counterfactual_accepted = any(
                    item.get("accepted", False)
                    for item in deferred_updates
                )
            else:
                # Counterfactual baselines must belong to the current workflow
                # version, so traces from older rounds are never mixed in.
                self.optimizer.failure_buffer.clear()
                self.optimizer.add_traces(round_traces)

                # Run optimization round
                updated_workflow, round_summary = (
                    await self.optimizer.optimize_round(
                        self.workflow,
                        self.executor,
                        self.scheduler,
                        llm_client=self.llm_client,
                    )
                )
                counterfactual_accepted = bool(
                    round_summary.get("accepted", False)
                )
            round_summary["counterfactual_accepted"] = (
                counterfactual_accepted
            )
            workflow_content_only = bool(
                getattr(
                    getattr(self.config, "optimizer", None),
                    "workflow_content_only",
                    True,
                )
            )
            round_summary["workflow_content_only"] = workflow_content_only
            scheduler_frozen = bool(
                self.config.scheduler_calibration_round is not None
                and round_num <= int(self.config.workflow_optimization_rounds or 0)
            )
            round_summary["scheduler_frozen"] = scheduler_frozen
            round_summary["active_scheduler_type"] = (
                "cascade"
                if isinstance(self.scheduler, CascadeScheduler)
                else "graph"
                if isinstance(self.scheduler, GraphScheduler)
                else "fixed"
            )
            round_summary["outer_execution_optimization_enabled"] = (
                False
                if scheduler_frozen
                else self._outer_scheduler_enabled(self.config)
            )
            round_summary["optimization_metrics"] = opt_metrics
            round_summary["incumbent_workflow_version_before"] = (
                round_start_workflow.version
            )

            if aggressive_local:
                # Counterfactual suffix repair (plus the sampled local guard)
                # is the complete promotion predicate in this mode.  No
                # optimization confirmation or full validation rerun is
                # performed. Accepted updates become the *provisional*
                # incumbent, while checkpoint selection uses the complete
                # optimization epoch that was just observed for the
                # round-start workflow. This prevents a score measured on
                # vN from being attached to an unevaluated vN+1 patch.
                if counterfactual_accepted and updated_workflow is not self.workflow:
                    # Deferred/online modes may already bind their candidate;
                    # assigning the copy is harmless and makes batch mode
                    # follow the same immediate-promotion semantics.
                    self.workflow = updated_workflow
                local_promoted = bool(
                    counterfactual_accepted
                    and self.workflow.version != round_start_workflow.version
                )
                confirmation = {
                    "enabled": False,
                    "performed": False,
                    "configured_repeats": 0,
                    "completed_repeats": 0,
                    "baseline_metrics": None,
                    "repeat_metrics": [],
                    "mean_metrics": None,
                    "hard_reward_effect": None,
                    "mean_utility_effect": None,
                    "hard_non_regression_passed": None,
                    "mean_utility_effect_passed": None,
                    "passed": local_promoted,
                    "decision": (
                        "counterfactual_local_accept"
                        if local_promoted
                        else "counterfactual_local_reject"
                    ),
                    "reason": "full_validation_gate_removed",
                }
                round_summary["optimization_confirmation"] = confirmation
                round_summary["confirmation_metrics"] = None
                round_summary["confirmation_decision"] = confirmation[
                    "decision"
                ]
                local_metrics = round_summary.get("optimization_metrics") or {}
                local_score = local_metrics.get(self._selection_metric_key)
                try:
                    local_score = float(local_score)
                except (TypeError, ValueError, OverflowError):
                    local_score = None
                observed_is_best = False
                if local_score is not None:
                    observed_is_best = self.checkpoint.update(
                        round_start_workflow,
                        local_score,
                        round_num,
                        metadata={
                            "promotion_mode": "counterfactual_local_aggressive",
                            "selection_basis": (
                                "complete_optimization_epoch_before_updates"
                            ),
                            "evaluated_workflow_version": (
                                round_start_workflow.version
                            ),
                            "provisional_workflow_version_after_updates": (
                                self.workflow.version
                            ),
                            "optimization_metrics": local_metrics,
                        },
                    )
                validation_gate = {
                    "evaluated": False,
                    "passed": local_promoted,
                    "reason": (
                        None
                        if local_promoted
                        else "counterfactual_rejected"
                    ),
                    "promotion_mode": "counterfactual_local_aggressive",
                    "full_validation_gate_removed": True,
                }
                round_summary.update(
                    {
                        "val_score": None,
                        "validation_metrics": None,
                        "candidate_validation_metrics": None,
                        "validation_candidate_evaluated": False,
                        "validation_reused": True,
                        "validation_reuse_reason": "audit_only_aggressive_mode",
                        "validation_gate": validation_gate,
                        "validation_gate_passed": local_promoted,
                        "accepted": local_promoted,
                        "promotion_rejection_reason": validation_gate["reason"],
                        "is_best": observed_is_best,
                        "local_observed_score": local_score,
                        "checkpointed_workflow_version": (
                            round_start_workflow.version
                            if observed_is_best
                            else None
                        ),
                        "incumbent_workflow_version_after": self.workflow.version,
                        "incumbent_validation_metrics": None,
                        "backend_usage_delta": self._backend_usage_delta(
                            round_usage_before,
                            self._backend_usage_snapshot(),
                        ),
                    }
                )
                results["rounds"].append(round_summary)
                logger.info(
                    "Round %s aggressive local promotion: accepted=%s version=%s",
                    round_num,
                    counterfactual_accepted,
                    self.workflow.version,
                )
                # Validation patience has no meaning when validation is an
                # audit-only split. Continue all configured local rounds.
                continue

            if online_mode:
                # Each online candidate has already been evaluated on its
                # triggering failure by the counterfactual evaluator.  A full
                # optimization-split confirmation would defeat the purpose of
                # streaming updates and suffix replay, so record an explicit
                # local decision and defer the costly global check to val.
                confirmation = {
                    "enabled": bool(self.config.confirm_on_opt),
                    "performed": False,
                    "configured_repeats": self.config.confirm_repeats,
                    "completed_repeats": 0,
                    "baseline_metrics": {
                        "hard_reward": opt_metrics["hard_reward"],
                        "runtime_utility": opt_metrics["runtime_utility"],
                    },
                    "repeat_metrics": [],
                    "mean_metrics": None,
                    "hard_reward_effect": None,
                    "mean_utility_effect": None,
                    "hard_non_regression_passed": None,
                    "mean_utility_effect_passed": None,
                    "passed": counterfactual_accepted,
                    "decision": (
                        "online_local_accept"
                        if counterfactual_accepted
                        else "online_no_update"
                    ),
                    "reason": "online_failure_triggered_update",
                    "online_updates": len(online_updates),
                }
            else:
                confirmation = await self._confirm_candidate_on_opt(
                    updated_workflow,
                    opt_metrics,
                    counterfactual_accepted=counterfactual_accepted,
                )
            round_summary["optimization_confirmation"] = confirmation
            round_summary["confirmation_metrics"] = confirmation.get(
                "mean_metrics"
            )
            round_summary["confirmation_decision"] = confirmation[
                "decision"
            ]

            candidate_eligible = (
                counterfactual_accepted
                and bool(confirmation["passed"])
            )
            candidate_validation_metrics: dict[str, Any] | None = None
            if candidate_eligible:
                candidate_validation_metrics = await self._evaluate_dataset(
                    updated_workflow,
                    self.val_data,
                    split_name="validation",
                )
                validation_gate = self._validation_gate_decision(
                    candidate_validation_metrics,
                    incumbent_validation_metrics,
                )
                validation_reused = False
                validation_reuse_reason = None
                val_metrics = candidate_validation_metrics
            else:
                validation_reused = True
                validation_reuse_reason = (
                    "counterfactual_rejected"
                    if not counterfactual_accepted
                    else "optimization_confirmation_failed"
                )
                validation_gate = self._unevaluated_validation_gate(
                    incumbent_validation_metrics,
                    reason=validation_reuse_reason,
                )
                # The incumbent was already measured on this exact validation
                # split. Do not spend calls re-running an unchanged workflow.
                val_metrics = incumbent_validation_metrics

            val_score = val_metrics[self._selection_metric_key]
            round_summary.update(
                {
                    "val_score": val_score,
                    "validation_metrics": val_metrics,
                    "candidate_validation_metrics": (
                        candidate_validation_metrics
                    ),
                    "validation_candidate_evaluated": (
                        candidate_validation_metrics is not None
                    ),
                    "validation_reused": validation_reused,
                    "validation_reuse_reason": validation_reuse_reason,
                    "validation_gate": validation_gate,
                    "validation_gate_passed": bool(
                        validation_gate["passed"]
                    ),
                    # ``accepted`` is the final promotion decision. The
                    # optimizer-local decision remains separately observable
                    # as ``counterfactual_accepted``.
                    "accepted": bool(validation_gate["passed"]),
                    "promotion_rejection_reason": (
                        None
                        if validation_gate["passed"]
                        else validation_gate["reason"]
                    ),
                }
            )
            round_summary["backend_usage_delta"] = (
                self._backend_usage_delta(
                    round_usage_before,
                    self._backend_usage_snapshot(),
                )
            )

            # A hard-reward regression never reaches score-only checkpoint
            # selection. For an eligible candidate the checkpoint uses the
            # same utility min_delta as the validation gate.
            is_best = False
            if validation_gate["passed"]:
                is_best = self.checkpoint.update(
                    updated_workflow,
                    candidate_validation_metrics[self._selection_metric_key],
                    round_num,
                    metadata=round_summary,
                )
                if not is_best:
                    # This should be unreachable while checkpoint and
                    # incumbent remain synchronized, but fail closed if an
                    # externally restored manager violates that invariant.
                    validation_gate["passed"] = False
                    validation_gate["reason"] = (
                        "checkpoint_min_delta_not_met"
                    )
                    round_summary["validation_gate_passed"] = False
                    round_summary["accepted"] = False
                    round_summary["promotion_rejection_reason"] = (
                        validation_gate["reason"]
                    )

            if is_best:
                self.workflow = updated_workflow
                incumbent_validation_metrics = candidate_validation_metrics
            elif online_mode or deferred_mode:
                # Streaming and deferred updates are provisional until the
                # same validation gate used by the batch algorithm promotes
                # them. Restore the round-start workflow after a rejected
                # candidate so later rounds never inherit an unvalidated patch.
                self.workflow = round_start_workflow

            round_summary["is_best"] = is_best
            round_summary["incumbent_workflow_version_after"] = (
                self.workflow.version
            )
            round_summary["incumbent_validation_metrics"] = (
                incumbent_validation_metrics
            )

            results["rounds"].append(round_summary)

            logger.info(
                "Round %s: candidate_val=%s, incumbent_val=%.4f, "
                "counterfactual_accepted=%s, accepted=%s, is_best=%s",
                round_num,
                (
                    f"{candidate_validation_metrics['runtime_utility']:.4f}"
                    if candidate_validation_metrics is not None
                    else "not_evaluated"
                ),
                incumbent_validation_metrics["runtime_utility"],
                counterfactual_accepted,
                round_summary["accepted"],
                is_best,
            )

            # Checkpoint promotion is the sole definition of meaningful
            # validation progress, so patience cannot disagree with min_delta.
            if is_best:
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if (
                is_best
                and self.workflow.selective_update is not None
                and round_num < self.config.optimizer.max_rounds
            ):
                # The research implementation has an explicit one-layer policy
                # contract. Do not spend another optimization epoch only to
                # rediscover that a policy-of-policies is unsupported.
                results["stopped_early"] = True
                results["early_stop_round"] = round_num
                results["early_stop_reason"] = (
                    "single_layer_selective_policy_promoted"
                )
                break

            patience = self.config.early_stopping_patience
            if (
                patience is not None
                and rounds_without_improvement >= patience
                and round_num < self.config.optimizer.max_rounds
            ):
                results["stopped_early"] = True
                results["early_stop_round"] = round_num
                results["early_stop_reason"] = "validation_patience"
                logger.info("Early stopping at round %s", round_num)
                break

        # Select and bind the local winner (aggressive mode) or validation
        # winner (legacy mode), but never execute the held-out test split here.
        # Official test evaluation is a separate one-shot command.
        best_workflow = self.checkpoint.load_best()
        if best_workflow is None:
            raise RuntimeError(
                "No local/validation workflow checkpoint was produced"
            )
        results["best_val_score"] = (
            None if aggressive_local else self.checkpoint.best_score
        )
        results["best_local_observed_score"] = (
            self.checkpoint.best_score if aggressive_local else None
        )
        results["final_workflow_version"] = best_workflow.version
        # The object held by the runner must agree with the manifest and the
        # checkpoint consumed by independent evaluation commands.
        self.workflow = best_workflow

        results["end_time"] = datetime.now().isoformat()
        results["checkpoint_summary"] = self.checkpoint.get_summary()
        results["backend_usage"] = self._backend_usage_snapshot()
        checkpoint_path = self.output_dir / "checkpoints" / "best_workflow.yaml"
        self.manifest = bind_best_checkpoint(
            self.manifest,
            checkpoint_path=checkpoint_path,
            workflow=best_workflow,
            round_num=self.checkpoint.best_round,
        )
        results["manifest"] = self.manifest

        # Save results
        self._save_results(results)

        return results

    async def _run_scheduler_calibration_round(
        self,
        round_num: int,
    ) -> dict[str, Any]:
        """Freeze workflow, fit the LAS gate, then audit it on full val.

        Each full validation pass executes every item once. Only items that
        are wrong in that pass receive additional generations for a stable
        majority label; already-correct items are never repeated. This keeps
        stochastic answer noise out of both gate fitting and the deployment
        guard without tripling the cost of the complete validation split.
        """
        frozen_version = self.workflow.version
        passive_config = self.config.scheduler.model_copy(deep=True)
        passive_config.gate_direct_early_exit = False
        passive_config.gate_schedule_threshold = 1.0
        passive_config.allow_deviation = False
        self.scheduler = CascadeScheduler(passive_config)
        validation_traces, baseline_metrics = await self._execute_dataset(
            self.workflow,
            self.val_data,
            split_name="scheduler_calibration_baseline",
        )
        stable_validation_traces, baseline_confirmation = (
            await self._confirm_scheduler_error_traces(
                self.workflow,
                validation_traces,
                split_prefix="scheduler_calibration_baseline_error_confirmation",
            )
        )
        stable_baseline_metrics = self._scheduler_trace_metrics(
            stable_validation_traces
        )
        calibrator = FrozenWorkflowGateCalibrator(
            self.config.scheduler,
            self.workflow,
            self.reward_evaluator,
        )
        calibration = calibrator.fit(stable_validation_traces)
        calibrated_config = calibrator.apply(calibration)
        self.scheduler = CascadeScheduler(calibrated_config)
        audit_traces, audit_metrics = await self._execute_dataset(
            self.workflow,
            self.val_data,
            split_name="scheduler_calibration_audit",
        )
        stable_audit_traces, audit_confirmation = (
            await self._confirm_scheduler_error_traces(
                self.workflow,
                audit_traces,
                split_prefix="scheduler_calibration_audit_error_confirmation",
            )
        )
        stable_audit_metrics = self._scheduler_trace_metrics(
            stable_audit_traces
        )
        audit_regression = (
            stable_baseline_metrics["hard_reward"]
            - stable_audit_metrics["hard_reward"]
        )
        hard_safe = (
            audit_regression
            <= self.config.scheduler.calibration_max_hard_regression + 1e-12
        )
        actual_token_delta = (
            stable_audit_metrics["total_tokens"]
            - stable_baseline_metrics["total_tokens"]
        )
        actual_latency_delta = (
            stable_audit_metrics["latency_seconds"]
            - stable_baseline_metrics["latency_seconds"]
        )
        efficiency_safe = (
            actual_token_delta <= 0
            and actual_latency_delta <= 1e-12
            and (actual_token_delta < 0 or actual_latency_delta < -1e-12)
        )
        calibration_deployed = hard_safe and efficiency_safe
        if not calibration_deployed:
            # Validation remains the final outer-layer success guard. A gate
            # whose real execution regresses is recorded but not deployed.
            calibrated_config = passive_config
            self.scheduler = CascadeScheduler(calibrated_config)
        calibration_payload = {
            **calibration.as_dict(),
            "workflow_frozen": True,
            "frozen_workflow_version": frozen_version,
            "baseline_validation_metrics": baseline_metrics,
            "stabilized_baseline_metrics": stable_baseline_metrics,
            "baseline_error_confirmation": baseline_confirmation,
            "audit_validation_metrics": audit_metrics,
            "stabilized_audit_metrics": stable_audit_metrics,
            "audit_error_confirmation": audit_confirmation,
            "actual_hard_reward_delta": (
                stable_audit_metrics["hard_reward"]
                - stable_baseline_metrics["hard_reward"]
            ),
            "deployed": calibration_deployed,
            "deployment_reason": (
                "validation_non_regression_and_efficiency_improvement"
                if calibration_deployed
                else (
                    "actual_validation_hard_regression"
                    if not hard_safe
                    else "actual_validation_efficiency_not_improved"
                )
            ),
            "hard_non_regression_passed": hard_safe,
            "efficiency_improvement_passed": efficiency_safe,
            "actual_token_delta": actual_token_delta,
            "actual_latency_delta_seconds": actual_latency_delta,
            "calibrated_scheduler_config": calibrated_config.model_dump(
                mode="json"
            ),
        }
        self.workflow.metadata["scheduler_calibration"] = (
            calibrated_config.model_dump(mode="json")
        )
        self.checkpoint.promote_local(
            self.workflow,
            round_num=round_num,
            score=float(stable_audit_metrics["hard_reward"]),
            metadata={"scheduler_calibration": calibration_payload},
        )
        return {
            "round": round_num,
            "phase": "scheduler_calibration",
            "accepted": True,
            "counterfactual_accepted": False,
            "workflow_frozen": True,
            "workflow_version": frozen_version,
            "workflow_version_after": self.workflow.version,
            "scheduler_calibration": calibration_payload,
            "validation_metrics": stable_audit_metrics,
            "optimization_metrics": None,
            "candidate_evaluations": [],
            "scope_attempts": [],
            "generated_candidates": 0,
            "evaluated_candidates": 0,
        }

    async def _confirm_scheduler_error_traces(
        self,
        workflow: WorkflowTemplate,
        traces: list[ExecutionTrace],
        *,
        split_prefix: str,
    ) -> tuple[list[ExecutionTrace], dict[str, Any]]:
        """Majority-stabilize only the hard failures of one scheduler pass."""
        optimizer_config = getattr(self.config, "optimizer", None)
        threshold = float(
            getattr(optimizer_config, "hard_success_threshold", 1.0)
        )
        repeats = int(
            getattr(optimizer_config, "failure_confirmation_repeats", 1)
        )
        min_failures = int(
            getattr(
                optimizer_config,
                "failure_confirmation_min_failures",
                2,
            )
        )
        failure_positions = [
            index
            for index, trace in enumerate(traces)
            if not self._is_hard_success(trace, threshold)
        ]
        report: dict[str, Any] = {
            "initial_examples": len(traces),
            "initial_failures": len(failure_positions),
            "repeated_successes": 0,
            "additional_runs_per_initial_failure": repeats,
            "minimum_failures": min_failures,
            "additional_executions": 0,
            "additional_input_tokens": 0,
            "additional_output_tokens": 0,
            "additional_total_tokens": 0,
            "additional_latency_seconds": 0.0,
            "cases": [],
        }
        if not failure_positions or repeats <= 0:
            return list(traces), report

        repeat_groups: dict[int, list[ExecutionTrace]] = {
            index: [] for index in failure_positions
        }
        failure_data = [self.val_data[index] for index in failure_positions]
        for repeat_index in range(repeats):
            repeated, metrics = await self._execute_dataset(
                workflow,
                failure_data,
                split_name=f"{split_prefix}_{repeat_index + 1}",
            )
            for position, repeated_trace in zip(failure_positions, repeated):
                # Preserve the position in the complete validation split; the
                # subset executor otherwise numbers from zero.
                repeated_trace.metadata["split_example_index"] = position
                repeat_groups[position].append(repeated_trace)
            report["additional_executions"] += len(repeated)
            report["additional_input_tokens"] += int(metrics["input_tokens"])
            report["additional_output_tokens"] += int(metrics["output_tokens"])
            report["additional_total_tokens"] += int(metrics["total_tokens"])
            report["additional_latency_seconds"] += float(
                metrics["latency_seconds"]
            )

        stabilized = list(traces)
        for position in failure_positions:
            observations = [traces[position], *repeat_groups[position]]
            failed = [
                trace
                for trace in observations
                if not self._is_hard_success(trace, threshold)
            ]
            successful = [
                trace
                for trace in observations
                if self._is_hard_success(trace, threshold)
            ]
            stable_failure = len(failed) >= min_failures
            representative = (
                failed[-1]
                if stable_failure or not successful
                else successful[-1]
            )
            representative.metadata["scheduler_failure_confirmation"] = {
                "total_runs": len(observations),
                "failures": len(failed),
                "successes": len(successful),
                "minimum_failures": min_failures,
                "stable_failure": stable_failure,
                "hard_rewards": [item.hard_reward for item in observations],
            }
            stabilized[position] = representative
            if not stable_failure:
                report["repeated_successes"] += 1
            report["cases"].append(
                {
                    "split_example_index": position,
                    **representative.metadata[
                        "scheduler_failure_confirmation"
                    ],
                }
            )
        return stabilized, report

    @staticmethod
    def _scheduler_trace_metrics(
        traces: list[ExecutionTrace],
    ) -> dict[str, Any]:
        """Report one representative execution per validation item."""
        if not traces:
            raise ValueError("scheduler trace metrics require non-empty traces")
        hard = [float(trace.hard_reward or 0.0) for trace in traces]
        return {
            "num_examples": len(traces),
            "hard_reward": sum(hard) / len(hard),
            "hard_success_rate": sum(value >= 1.0 for value in hard) / len(hard),
            "input_tokens": sum(trace.total_prompt_tokens for trace in traces),
            "output_tokens": sum(trace.total_completion_tokens for trace in traces),
            "total_tokens": sum(trace.total_tokens for trace in traces),
            "llm_call_count": sum(trace.total_llm_calls for trace in traces),
            "latency_seconds": sum(trace.total_latency_seconds for trace in traces),
        }

    async def _confirm_candidate_on_opt(
        self,
        candidate_workflow: WorkflowTemplate,
        baseline_metrics: dict[str, Any],
        *,
        counterfactual_accepted: bool,
    ) -> dict[str, Any]:
        """Optionally confirm a local edit on the complete optimization split.

        The current incumbent's full-split metrics were already collected by
        ``_run_optimization_epoch``. Only the candidate is re-run, keeping the
        gate both paired and free of validation/test feedback.
        """
        enabled = bool(self.config.confirm_on_opt)
        confirmation: dict[str, Any] = {
            "enabled": enabled,
            "performed": False,
            "configured_repeats": self.config.confirm_repeats,
            "completed_repeats": 0,
            "baseline_metrics": {
                "hard_reward": baseline_metrics["hard_reward"],
                "runtime_utility": baseline_metrics["runtime_utility"],
            },
            "repeat_metrics": [],
            "mean_metrics": None,
            "hard_reward_effect": None,
            "mean_utility_effect": None,
            "hard_non_regression_passed": None,
            "mean_utility_effect_passed": None,
            "passed": False,
            "decision": "not_run",
            "reason": None,
        }
        if not counterfactual_accepted:
            confirmation["reason"] = "counterfactual_rejected"
            return confirmation
        if not enabled:
            confirmation.update(
                {
                    "passed": True,
                    "decision": "not_required",
                    "reason": "confirm_on_opt_disabled",
                }
            )
            return confirmation

        confirmation["performed"] = True
        try:
            for repeat_index in range(self.config.confirm_repeats):
                metrics = await self._evaluate_dataset(
                    candidate_workflow,
                    self.opt_data,
                    split_name=(
                        f"optimization_confirmation_{repeat_index + 1}"
                    ),
                )
                confirmation["repeat_metrics"].append(metrics)
        except Exception as exc:
            confirmation.update(
                {
                    "completed_repeats": len(
                        confirmation["repeat_metrics"]
                    ),
                    "decision": "rejected",
                    "reason": "confirmation_evaluation_error",
                    "error": str(exc),
                }
            )
            return confirmation

        mean_metrics = self._mean_numeric_metrics(
            confirmation["repeat_metrics"]
        )
        hard_effect = (
            mean_metrics["hard_reward"] - baseline_metrics["hard_reward"]
        )
        utility_effect = (
            mean_metrics["runtime_utility"]
            - baseline_metrics["runtime_utility"]
        )
        priority = bool(self.config.hard_success_priority)
        if priority:
            # Accuracy-first: the edit must improve hard reward, and utility
            # only needs to non-regress beyond the configured tolerance.
            hard_passed = hard_effect > 0.0
            utility_passed = (
                utility_effect
                >= -self.config.validation_hard_regression_tolerance
            )
            fail_hard_label = "optimization_hard_not_improved"
            fail_utility_label = "optimization_utility_regression_exceeded"
        else:
            hard_passed = hard_effect >= 0.0
            utility_passed = utility_effect > 0.0
            fail_hard_label = "optimization_hard_regression"
            fail_utility_label = (
                "optimization_mean_utility_not_improved"
            )
        passed = hard_passed and utility_passed
        confirmation.update(
            {
                "completed_repeats": len(
                    confirmation["repeat_metrics"]
                ),
                "mean_metrics": mean_metrics,
                "promotion_mode": (
                    "hard_priority" if priority else "utility_delta"
                ),
                "hard_reward_effect": hard_effect,
                "mean_utility_effect": utility_effect,
                "hard_non_regression_passed": hard_passed,
                "mean_utility_effect_passed": utility_passed,
                "passed": passed,
                "decision": "accepted" if passed else "rejected",
                "reason": (
                    None
                    if passed
                    else self._gate_failure_reason(
                        hard_passed,
                        utility_passed,
                        hard_label=fail_hard_label,
                        utility_label=fail_utility_label,
                    )
                ),
            }
        )
        return confirmation

    @staticmethod
    def _mean_numeric_metrics(
        metrics: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Average scalar numeric metrics across equal-sized repeat runs."""
        if not metrics:
            raise ValueError("Cannot aggregate an empty metrics sequence")
        means: dict[str, float] = {}
        all_keys = list(dict.fromkeys(k for m in metrics for k in m))
        for key in all_keys:
            values = [item.get(key) for item in metrics]
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in values
            ):
                means[key] = sum(values) / len(values)
        return means

    def _validation_gate_decision(
        self,
        candidate_metrics: dict[str, Any],
        incumbent_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the hard non-regression and strict utility promotion gates.

        By default runtime utility drives promotion (candidate must beat the
        incumbent by ``validation_min_delta``) while hard reward only needs to
        non-regress. With ``hard_success_priority`` the roles invert: hard
        reward (accuracy) must improve by ``validation_min_delta`` and runtime
        utility only needs to non-regress beyond the tolerance.
        """
        tolerance = self.config.validation_hard_regression_tolerance
        incumbent_hard = incumbent_metrics["hard_reward"]
        candidate_hard = candidate_metrics["hard_reward"]
        incumbent_utility = incumbent_metrics["runtime_utility"]
        candidate_utility = candidate_metrics["runtime_utility"]
        priority = bool(self.config.hard_success_priority)
        if priority:
            hard_passed = (
                candidate_hard
                > incumbent_hard + self._validation_min_delta
            )
            utility_passed = (
                candidate_utility >= incumbent_utility - tolerance
            )
            fail_hard_label = "hard_priority_improvement_not_met"
            fail_utility_label = "utility_regression_exceeded"
            hard_floor = incumbent_hard + self._validation_min_delta
            required_utility = incumbent_utility - tolerance
        else:
            hard_passed = candidate_hard >= incumbent_hard - tolerance
            utility_passed = (
                candidate_utility
                > incumbent_utility + self._validation_min_delta
            )
            fail_hard_label = "validation_hard_regression"
            fail_utility_label = "validation_utility_min_delta_not_met"
            hard_floor = incumbent_hard - tolerance
            required_utility = (
                incumbent_utility + self._validation_min_delta
            )
        passed = hard_passed and utility_passed
        return {
            "evaluated": True,
            "passed": passed,
            "reason": (
                None
                if passed
                else self._gate_failure_reason(
                    hard_passed,
                    utility_passed,
                    hard_label=fail_hard_label,
                    utility_label=fail_utility_label,
                )
            ),
            "promotion_mode": (
                "hard_priority" if priority else "utility_delta"
            ),
            "selection_metric": self._selection_metric_key,
            "hard_regression_tolerance": tolerance,
            "validation_min_delta": self._validation_min_delta,
            "incumbent_hard_reward": incumbent_hard,
            "candidate_hard_reward": candidate_hard,
            "hard_reward_effect": candidate_hard - incumbent_hard,
            "hard_reward_floor": hard_floor,
            "hard_non_regression_passed": hard_passed,
            "incumbent_runtime_utility": incumbent_utility,
            "candidate_runtime_utility": candidate_utility,
            "runtime_utility_effect": (
                candidate_utility - incumbent_utility
            ),
            "required_runtime_utility": required_utility,
            "runtime_utility_passed": utility_passed,
        }

    def _unevaluated_validation_gate(
        self,
        incumbent_metrics: dict[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Describe a safely skipped validation without fabricating metrics."""
        priority = bool(self.config.hard_success_priority)
        tolerance = self.config.validation_hard_regression_tolerance
        if priority:
            hard_floor = (
                incumbent_metrics["hard_reward"]
                + self._validation_min_delta
            )
            required_utility = (
                incumbent_metrics["runtime_utility"] - tolerance
            )
        else:
            hard_floor = (
                incumbent_metrics["hard_reward"] - tolerance
            )
            required_utility = (
                incumbent_metrics["runtime_utility"]
                + self._validation_min_delta
            )
        return {
            "evaluated": False,
            "passed": False,
            "reason": reason,
            "promotion_mode": (
                "hard_priority" if priority else "utility_delta"
            ),
            "selection_metric": self._selection_metric_key,
            "hard_regression_tolerance": tolerance,
            "validation_min_delta": self._validation_min_delta,
            "incumbent_hard_reward": incumbent_metrics["hard_reward"],
            "candidate_hard_reward": None,
            "hard_reward_effect": None,
            "hard_reward_floor": hard_floor,
            "hard_non_regression_passed": None,
            "incumbent_runtime_utility": incumbent_metrics[
                "runtime_utility"
            ],
            "candidate_runtime_utility": None,
            "runtime_utility_effect": None,
            "required_runtime_utility": required_utility,
            "runtime_utility_passed": None,
        }

    @staticmethod
    def _gate_failure_reason(
        hard_passed: bool,
        utility_passed: bool,
        *,
        hard_label: str,
        utility_label: str,
    ) -> str:
        if not hard_passed and not utility_passed:
            return f"{hard_label}+{utility_label}"
        if not hard_passed:
            return hard_label
        return utility_label

    async def _run_optimization_epoch(
        self,
    ) -> tuple[list[ExecutionTrace], dict[str, Any]]:
        """Execute the optimization split and collect traces.

        Deferred sequential mode deliberately takes the ordinary full-epoch
        path. Failures are replayed and updated only after this method returns.
        The legacy online mode keeps its streaming callback behavior.
        """
        online = self._failure_update_mode() == "online"
        if online:
            self._online_round_updates = []
            self.optimizer.failure_buffer.clear()
            traces, metrics = await self._execute_dataset(
                self.workflow,
                self.opt_data,
                split_name="optimization",
                workflow_provider=lambda: self.workflow,
                sample_callback=self._handle_online_sample,
            )
            updates = list(getattr(self, "_online_round_updates", []))
            metrics.update(
                {
                    "failure_update_mode": "online",
                    "online_failure_updates": True,
                    "online_update_count": len(updates),
                    "online_accepted_count": sum(
                        item.get("accepted", False) for item in updates
                    ),
                    "online_suffix_replay_used_count": sum(
                        item.get("suffix_replay_used", False)
                        for item in updates
                    ),
                }
            )
            return traces, metrics
        if self._failure_update_mode() == "deferred_sequential":
            self.optimizer.failure_buffer.clear()
            traces, metrics = await self._execute_dataset(
                self.workflow,
                self.opt_data,
                split_name="optimization",
                sample_callback=self._buffer_deferred_failure,
            )
            metrics["failure_update_mode"] = "deferred_sequential"
            metrics["deferred_buffered_failure_count"] = len(
                self.optimizer.failure_buffer.failures
            )
            return traces, metrics
        traces, metrics = await self._execute_dataset(
            self.workflow,
            self.opt_data,
            split_name="optimization",
        )
        metrics["failure_update_mode"] = self._failure_update_mode()
        return traces, metrics

    async def _buffer_deferred_failure(
        self,
        trace: ExecutionTrace,
        hard_reward: float,
        process_reward: float,
        composite_reward: float,
        utility: float,
    ) -> None:
        """Record failures during a deferred optimization epoch only."""
        del process_reward, composite_reward, utility
        threshold = float(
            getattr(
                getattr(self.config, "optimizer", None),
                "hard_success_threshold",
                1.0,
            )
        )
        if not self._is_hard_success(trace, threshold):
            self.optimizer.failure_buffer.add(trace)

    def _failure_update_mode(self) -> str:
        """Resolve the explicit failure-update schedule.

        The old ``online_failure_updates`` flag remains a compatibility alias
        when no explicit mode is configured.
        """
        configured = getattr(
            getattr(self.config, "optimizer", None),
            "failure_update_mode",
            None,
        )
        if configured in {"batch", "deferred_sequential", "online"}:
            return str(configured)
        return (
            "online"
            if bool(
                getattr(
                    getattr(self.config, "optimizer", None),
                    "online_failure_updates",
                    False,
                )
            )
            else "batch"
        )

    @staticmethod
    def _is_hard_success(
        trace: ExecutionTrace,
        threshold: float,
    ) -> bool:
        return bool(
            trace.success
            and trace.hard_reward is not None
            and trace.hard_reward >= threshold
        )

    @classmethod
    def _partition_replayed_guards(
        cls,
        guard_replays: list[ExecutionTrace],
        threshold: float,
    ) -> tuple[list[ExecutionTrace], list[ExecutionTrace]]:
        """Split replayed guards without changing their assigned role.

        A guard that is already incorrect under the incumbent is an unstable
        observation, not a newly discovered target failure. Returning it in
        the second list lets callers audit stochastic instability while only
        the first list enters the non-regression counterfactual batch.
        """
        stable: list[ExecutionTrace] = []
        unstable: list[ExecutionTrace] = []
        for replay in guard_replays:
            target = stable if cls._is_hard_success(replay, threshold) else unstable
            target.append(replay)
        return stable, unstable

    async def _replay_deferred_failure(
        self,
        source_trace: ExecutionTrace,
        update_index: int | str,
    ) -> ExecutionTrace:
        """Re-execute one buffered failure under the current workflow.

        Epoch traces all belong to the pre-update workflow. Replaying the
        query before each sequential update gives counterfactual evaluation a
        version-matched baseline and keeps suffix-replay checkpoints valid.
        """
        query = source_trace.query_text
        ground_truth = source_trace.metadata.get("ground_truth")
        output, _, recorder = await self.executor.execute(
            self.workflow,
            self.scheduler,
            query,
            llm_client=self.llm_client,
        )
        trace = recorder.trace
        # A deferred replay is still an optimization-split observation: it is
        # generated exclusively from an optimization failure and is fed back
        # into CWU as local counterfactual evidence.  Keep the original split
        # contract so the optimizer's held-out-data guard does not reject the
        # replay, while retaining an explicit phase marker for trace audits.
        source_split = source_trace.metadata.get("split")
        trace.metadata.update(
            {
                "ground_truth": ground_truth,
                # Preserve the source split rather than relabeling arbitrary
                # inputs; deferred updates are only optimizer-safe when the
                # source itself is from the optimization split.
                "split": source_split,
                "trace_phase": "deferred_failure_replay",
                "replay_source_split": source_split,
                "experiment_round": self._current_round,
                "split_example_index": source_trace.metadata.get(
                    "split_example_index",
                ),
                "deferred_source_trace_id": source_trace.trace_id,
                "deferred_update_index": update_index,
            }
        )
        hard = self.reward_evaluator.hard_reward(
            query,
            ground_truth,
            output,
            trace,
        )
        process = self.reward_evaluator.process_reward(
            query,
            ground_truth,
            output,
            trace,
        )
        trace.hard_reward = hard
        trace.process_reward = process
        logger.info(
            "[deferred replay][failure %s] hard_reward=%.4f "
            "process_reward=%.4f workflow=%s version=%s",
            update_index,
            hard,
            process,
            trace.workflow_name,
            trace.workflow_version,
        )
        return trace

    async def _run_deferred_failure_updates(
        self,
        round_num: int,
        round_start_workflow: WorkflowTemplate,
        round_traces: list[ExecutionTrace],
    ) -> tuple[WorkflowTemplate, dict[str, Any]]:
        """Apply buffered optimization failures sequentially after the epoch.

        The complete epoch is measured with one incumbent workflow first.
        Failures are retained as an ordered queue, then each query is replayed
        under the latest accepted workflow before its own CWU update.
        """
        self._deferred_round_updates = []
        threshold = float(
            getattr(
                getattr(self.config, "optimizer", None),
                "hard_success_threshold",
                1.0,
            )
        )
        failures = [
            trace
            for trace in round_traces
            if not self._is_hard_success(trace, threshold)
        ]

        optimizer_config = getattr(self.config, "optimizer", None)
        if bool(getattr(optimizer_config, "failure_cluster_enabled", False)):
            return await self._run_clustered_failure_updates(
                round_num,
                round_start_workflow,
                round_traces,
            )
        efficiency_enabled = (
            not bool(getattr(optimizer_config, "workflow_content_only", True))
            and bool(
                getattr(
                    optimizer_config,
                    "efficiency_optimization_enabled",
                    False,
                )
            )
        )

        # Keep the normal FailureBuffer observable, while retaining the full
        # ordered failure queue even when its diagnostic capacity is smaller.
        self.optimizer.failure_buffer.clear()
        self.optimizer.failure_buffer.extend(round_traces)
        trace_batch = None
        consume_round = getattr(
            self.optimizer.failure_buffer,
            "consume_optimization_round",
            None,
        )
        if callable(consume_round):
            trace_batch = consume_round(
                self.workflow.name,
                self.workflow.version,
                success_fraction=getattr(
                    optimizer_config,
                    "success_guard_fraction",
                    0.2,
                ),
                min_success_guards=getattr(
                    optimizer_config,
                    "min_success_guards",
                    0,
                ),
                efficiency_enabled=efficiency_enabled,
                efficiency_fraction=getattr(
                    optimizer_config,
                    "efficiency_anchor_fraction",
                    0.2,
                ),
                efficiency_min_relative_cost=getattr(
                    optimizer_config,
                    "efficiency_min_relative_cost",
                    1.25,
                ),
                max_efficiency_anchors=getattr(
                    optimizer_config,
                    "efficiency_max_anchors",
                    5,
                ),
                efficiency_cost=(
                    self.optimizer._trace_efficiency_cost
                    if efficiency_enabled
                    else None
                ),
            )
            buffered_failure_count = len(trace_batch.failures)
            guard_sources = list(trace_batch.success_guards)
            efficiency_sources = list(trace_batch.efficiency_anchors)
        else:
            buffered_failure_count = len(
                getattr(self.optimizer.failure_buffer, "failures", [])
            )
            guard_sources = []
            efficiency_sources = []
            self.optimizer.failure_buffer.clear()

        triggers: list[tuple[ExecutionTrace, str]] = [
            (trace, "hard_failure") for trace in failures
        ]
        trigger_ids = {trace.trace_id for trace, _ in triggers}
        if efficiency_enabled:
            triggers.extend(
                (trace, "high_cost_success")
                for trace in efficiency_sources
                if trace.trace_id not in trigger_ids
            )

        replay_traces: list[ExecutionTrace] = []
        guard_cache_version: str | None = None
        guard_replays: list[ExecutionTrace] = []

        confirmation_repeats = int(
            getattr(optimizer_config, "failure_confirmation_repeats", 1)
        )
        confirmation_min_failures = int(
            getattr(
                optimizer_config,
                "failure_confirmation_min_failures",
                2,
            )
        )
        for update_index, (source_trace, trigger_type) in enumerate(triggers):
            # Only an observed hard failure is sampled repeatedly. The epoch
            # observation is run one; these are the additional generations.
            # Successful efficiency anchors and success guards are never
            # repeated by this randomness filter.
            repetitions = (
                confirmation_repeats
                if trigger_type == "hard_failure"
                else 1
            )
            confirmations: list[ExecutionTrace] = []
            for repeat_index in range(repetitions):
                replay = await self._replay_deferred_failure(
                    source_trace,
                    f"{update_index}-confirm-{repeat_index}",
                )
                confirmations.append(replay)
                replay_traces.append(replay)
            failed_confirmations = [
                replay
                for replay in confirmations
                if not self._is_hard_success(replay, threshold)
            ]
            observed_failures = (
                int(trigger_type == "hard_failure")
                + len(failed_confirmations)
            )
            total_runs = (
                int(trigger_type == "hard_failure") + len(confirmations)
            )
            if (
                trigger_type == "hard_failure"
                and (
                    observed_failures < confirmation_min_failures
                    or not failed_confirmations
                )
            ):
                representative = confirmations[-1]
                self._deferred_round_updates.append(
                    {
                        "update_index": f"probe-{update_index}",
                        "sample_index": source_trace.metadata.get(
                            "split_example_index"
                        ),
                        "source_trace_id": source_trace.trace_id,
                        "replay_trace_id": representative.trace_id,
                        "replay_trace_ids": [
                            item.trace_id for item in confirmations
                        ],
                        "trigger_type": trigger_type,
                        "update_status": "stochastic_failure_filtered",
                        "accepted": False,
                        "workflow_version_before": self.workflow.version,
                        "workflow_version_after": self.workflow.version,
                        "source_hard_reward": source_trace.hard_reward,
                        "replay_hard_reward": representative.hard_reward,
                        "confirmation_hard_rewards": [
                            item.hard_reward for item in confirmations
                        ],
                        "failure_confirmation_total_runs": total_runs,
                        "failure_confirmation_failures": observed_failures,
                        "failure_confirmation_min_failures": (
                            confirmation_min_failures
                        ),
                        "input_tokens": sum(
                            item.total_prompt_tokens for item in confirmations
                        ),
                        "output_tokens": sum(
                            item.total_completion_tokens for item in confirmations
                        ),
                        "latency_seconds": sum(
                            item.total_latency_seconds for item in confirmations
                        ),
                        "summary": {
                            "accepted": False,
                            "reason": "stochastic_failure_filtered",
                        },
                    }
                )
                continue
            replay_trace = (
                failed_confirmations[-1]
                if trigger_type == "hard_failure"
                else confirmations[-1]
            )
            if trigger_type == "hard_failure":
                replay_trace.metadata["failure_confirmation"] = {
                    "total_runs": total_runs,
                    "failures": observed_failures,
                    "min_failures": confirmation_min_failures,
                    "hard_rewards": [
                        source_trace.hard_reward,
                        *[item.hard_reward for item in confirmations],
                    ],
                }
            # A fresh trace is required after every accepted update because
            # FailureBuffer deliberately rejects stale workflow versions.
            self.optimizer.failure_buffer.clear()
            self.optimizer.failure_buffer.add(replay_trace)
            if guard_sources and guard_cache_version != self.workflow.version:
                guard_replays = []
                for guard_index, guard_source in enumerate(guard_sources):
                    guard_replay = await self._replay_deferred_failure(
                        guard_source,
                        f"{update_index}-guard-{guard_index}",
                    )
                    guard_replays.append(guard_replay)
                    replay_traces.append(guard_replay)
                guard_cache_version = self.workflow.version

            # Preserve the role assigned before replay. A sampled success is
            # a regression guard, never a target failure. If its incumbent
            # replay is already wrong, it is an unstable observation and
            # cannot express non-regression; omit it from candidate scoring
            # while retaining the replay in the audit trace file.
            stable_guard_replays, unstable_guard_replays = (
                self._partition_replayed_guards(guard_replays, threshold)
            )
            for guard_replay in stable_guard_replays:
                self.optimizer.failure_buffer.add(guard_replay)
            previous_workflow = self.workflow
            try:
                candidate_workflow, summary = (
                    await self.optimizer.optimize_round(
                        previous_workflow,
                        self.executor,
                        self.scheduler,
                        llm_client=self.llm_client,
                    )
                )
            except Exception as exc:
                logger.exception(
                    "[deferred failure %s] workflow update failed",
                    update_index,
                )
                candidate_workflow = previous_workflow
                summary = {
                    "accepted": False,
                    "error": str(exc),
                    "round": len(self.optimizer.round_history) + 1,
                }
            accepted = bool(summary.get("accepted", False))
            update_record = {
                "update_index": update_index,
                "sample_index": source_trace.metadata.get(
                    "split_example_index",
                ),
                "source_trace_id": source_trace.trace_id,
                "replay_trace_id": replay_trace.trace_id,
                "replay_trace_ids": [
                    item.trace_id for item in confirmations
                ],
                "trigger_type": trigger_type,
                "workflow_version_before": previous_workflow.version,
                "workflow_version_after": candidate_workflow.version,
                "accepted": accepted,
                "source_hard_reward": source_trace.hard_reward,
                "replay_hard_reward": replay_trace.hard_reward,
                "source_process_reward": source_trace.process_reward,
                "replay_process_reward": replay_trace.process_reward,
                "input_tokens": sum(
                    item.total_prompt_tokens for item in confirmations
                ),
                "output_tokens": sum(
                    item.total_completion_tokens for item in confirmations
                ),
                "latency_seconds": sum(
                    item.total_latency_seconds for item in confirmations
                ),
                "failure_confirmation_total_runs": total_runs,
                "failure_confirmation_failures": observed_failures,
                "failure_confirmation_min_failures": (
                    confirmation_min_failures
                    if trigger_type == "hard_failure"
                    else None
                ),
                "guard_replay_count": len(guard_replays),
                "stable_guard_replay_count": len(stable_guard_replays),
                "unstable_guard_replay_count": len(unstable_guard_replays),
                "unstable_guard_replay_ids": [
                    item.trace_id for item in unstable_guard_replays
                ],
                "guard_replay_input_tokens": sum(
                    item.total_prompt_tokens for item in guard_replays
                ),
                "guard_replay_output_tokens": sum(
                    item.total_completion_tokens for item in guard_replays
                ),
                "guard_replay_latency_seconds": sum(
                    item.total_latency_seconds for item in guard_replays
                ),
                "suffix_replay_requested": bool(
                    self.optimizer.counterfactual.suffix_replay
                ),
                "suffix_replay_used": bool(
                    summary.get("suffix_replay_used", False)
                ),
                "summary": summary,
            }
            self._deferred_round_updates.append(update_record)
            if accepted:
                self.workflow = candidate_workflow
                logger.info(
                    "[deferred failure %s] accepted workflow update %s -> %s",
                    update_index,
                    previous_workflow.version,
                    candidate_workflow.version,
                )
            else:
                logger.info(
                    "[deferred failure %s] workflow update rejected",
                    update_index,
                )

        self._save_traces("deferred_failure_replay", replay_traces)

        summary = self._summarize_deferred_round(
            round_num,
            round_start_workflow,
            self.workflow,
            self._deferred_round_updates,
            failure_count=len(failures),
            trigger_count=len(triggers),
            efficiency_trigger_count=len(efficiency_sources),
            buffered_failure_count=buffered_failure_count,
        )
        return self.workflow, summary

    @staticmethod
    def _failure_cluster_key(trace: ExecutionTrace) -> tuple[str, ...]:
        """Return a deterministic math-aware failure subclass.

        These features are produced by the benchmark evaluator on the
        optimization trace. They do not use held-out examples and map directly
        to a likely repair level instead of collapsing every wrong answer at
        the terminal ``end`` node.
        """
        error = str(getattr(trace, "error_message", "") or "").lower()
        if "max step" in error or "step limit" in error:
            kind = "max_steps"
        elif any(term in error for term in ("timeout", "timed out")):
            kind = "timeout"
        elif any(term in error for term in ("connection", "rate limit", "429")):
            kind = "transport"
        elif error:
            kind = "runtime_error"
        else:
            diagnostics = trace.metadata.get("math_evaluation", {})
            ground_truth = trace.metadata.get("ground_truth", {})
            if isinstance(diagnostics, dict) and diagnostics:
                domain = (
                    str(ground_truth.get("domain", "unknown"))
                    if isinstance(ground_truth, dict)
                    else "unknown"
                )
                output_present = bool(diagnostics.get("output_present", False))
                final_extractable = bool(
                    diagnostics.get("final_answer_extractable", False)
                )
                solve_extractable = bool(
                    diagnostics.get("solve_answer_extractable", False)
                )
                final_correct = bool(
                    diagnostics.get("final_answer_correct", False)
                )
                solve_correct = bool(
                    diagnostics.get("solve_answer_correct", False)
                )
                consistent = bool(
                    diagnostics.get("solve_final_consistent", False)
                )
                if not output_present:
                    subtype, repair_level = "missing_output", "finalize"
                elif solve_correct and not final_correct:
                    subtype, repair_level = "finalizer_corruption", "finalize"
                elif not final_extractable and solve_extractable:
                    subtype, repair_level = "final_format_failure", "finalize"
                elif not solve_extractable and not final_extractable:
                    subtype, repair_level = "reasoning_format_failure", "solve"
                elif not consistent:
                    subtype, repair_level = "verification_disagreement", "verify"
                else:
                    subtype, repair_level = "consistent_reasoning_error", "solve"
                return "math", domain, subtype, repair_level
            kind = "answer_failure"
        failed_node = "unknown"
        for step in getattr(trace, "steps", []) or []:
            if not bool(getattr(step, "success", True)):
                failed_node = str(getattr(step, "node_id", "unknown"))
                break
        if failed_node == "unknown" and getattr(trace, "steps", None):
            failed_node = str(getattr(trace.steps[-1], "node_id", "unknown"))
        return kind, failed_node

    async def _run_clustered_failure_updates(
        self,
        round_num: int,
        round_start_workflow: WorkflowTemplate,
        round_traces: list[ExecutionTrace],
    ) -> tuple[WorkflowTemplate, dict[str, Any]]:
        """Apply one CWU transaction per cluster of persistent failures.

        Clustering is deliberately an inner-loop research intervention.  Each
        cluster is replayed against the *current* workflow immediately before
        its update, so a prior accepted patch never leaves stale trace versions
        in the counterfactual baseline.  A candidate may contain a ``multi``
        patch spanning prompt, operator and graph-path units.  Full validation
        is not consulted; sampled successes remain local hard guards and a
        short rollback window prevents a run of rejected cluster repairs from
        permanently carrying a bad aggressive edit.
        """
        self._deferred_round_updates = []
        optimizer_config = getattr(self.config, "optimizer", None)
        threshold = float(getattr(optimizer_config, "hard_success_threshold", 1.0))
        efficiency_enabled = (
            not bool(getattr(optimizer_config, "workflow_content_only", True))
            and bool(getattr(optimizer_config, "efficiency_optimization_enabled", False))
        )
        failures = [
            trace for trace in round_traces
            if not self._is_hard_success(trace, threshold)
        ]

        # Build the same deterministic success/efficiency guard sample used by
        # the ordinary deferred path.  The initial consume only supplies guard
        # sources; every actual cluster is then replayed at its current version.
        self.optimizer.failure_buffer.clear()
        self.optimizer.failure_buffer.extend(round_traces)
        trace_batch = self.optimizer.failure_buffer.consume_optimization_round(
            self.workflow.name,
            self.workflow.version,
            success_fraction=getattr(optimizer_config, "success_guard_fraction", 0.2),
            min_success_guards=getattr(optimizer_config, "min_success_guards", 0),
            efficiency_enabled=efficiency_enabled,
            efficiency_fraction=getattr(optimizer_config, "efficiency_anchor_fraction", 0.2),
            efficiency_min_relative_cost=getattr(optimizer_config, "efficiency_min_relative_cost", 1.25),
            max_efficiency_anchors=getattr(optimizer_config, "efficiency_max_anchors", 5),
            efficiency_cost=(
                self.optimizer._trace_efficiency_cost if efficiency_enabled else None
            ),
        )
        guard_sources = list(trace_batch.success_guards)
        efficiency_sources = list(trace_batch.efficiency_anchors)
        triggers: list[tuple[ExecutionTrace, str]] = [
            (trace, "hard_failure") for trace in failures
        ]
        trigger_ids = {trace.trace_id for trace, _ in triggers}
        if efficiency_enabled:
            triggers.extend(
                (trace, "high_cost_success")
                for trace in efficiency_sources
                if trace.trace_id not in trigger_ids
            )

        replay_traces: list[ExecutionTrace] = []
        # Only observed hard failures receive repeated generation. The epoch
        # observation is run one; configured replays provide the remaining
        # evidence for a stable 2/3-style failure label. Successful epoch rows
        # are never repeated by this mechanism.
        persistent: list[tuple[ExecutionTrace, str, tuple[str, ...]]] = []
        probe_replays: dict[str, ExecutionTrace] = {}
        incumbent_replay_successes = 0
        confirmation_repeats = int(
            getattr(optimizer_config, "failure_confirmation_repeats", 1)
        )
        confirmation_min_failures = int(
            getattr(
                optimizer_config,
                "failure_confirmation_min_failures",
                2,
            )
        )
        for trigger_index, (source_trace, trigger_type) in enumerate(triggers):
            repetitions = (
                confirmation_repeats
                if trigger_type == "hard_failure"
                else 1
            )
            confirmations: list[ExecutionTrace] = []
            for repeat_index in range(repetitions):
                replay = await self._replay_deferred_failure(
                    source_trace,
                    f"cluster-probe-{trigger_index}-{repeat_index}",
                )
                replay_traces.append(replay)
                confirmations.append(replay)
            failed_confirmations = [
                replay
                for replay in confirmations
                if not self._is_hard_success(replay, threshold)
            ]
            initial_failure = int(trigger_type == "hard_failure")
            observed_failures = initial_failure + len(failed_confirmations)
            total_runs = initial_failure + len(confirmations)
            is_persistent = bool(
                trigger_type == "hard_failure"
                and observed_failures >= confirmation_min_failures
                and failed_confirmations
            )
            representative = (
                failed_confirmations[-1]
                if failed_confirmations
                else confirmations[-1]
            )
            probe_replays[source_trace.trace_id] = representative
            if not is_persistent:
                incumbent_replay_successes += 1
                self._deferred_round_updates.append(
                    {
                        "update_index": f"probe-{trigger_index}",
                        "sample_index": source_trace.metadata.get("split_example_index"),
                        "source_trace_id": source_trace.trace_id,
                        "replay_trace_id": representative.trace_id,
                        "replay_trace_ids": [
                            item.trace_id for item in confirmations
                        ],
                        "trigger_type": trigger_type,
                        "update_status": "stochastic_failure_filtered",
                        "accepted": False,
                        "workflow_version_before": self.workflow.version,
                        "workflow_version_after": self.workflow.version,
                        "source_hard_reward": source_trace.hard_reward,
                        "replay_hard_reward": representative.hard_reward,
                        "confirmation_hard_rewards": [
                            item.hard_reward for item in confirmations
                        ],
                        "failure_confirmation_total_runs": total_runs,
                        "failure_confirmation_failures": observed_failures,
                        "failure_confirmation_min_failures": (
                            confirmation_min_failures
                        ),
                        "source_process_reward": source_trace.process_reward,
                        "replay_process_reward": representative.process_reward,
                        "input_tokens": sum(
                            item.total_prompt_tokens for item in confirmations
                        ),
                        "output_tokens": sum(
                            item.total_completion_tokens for item in confirmations
                        ),
                        "latency_seconds": sum(
                            item.total_latency_seconds for item in confirmations
                        ),
                        "cluster_probe": True,
                        "summary": {
                            "accepted": False,
                            "reason": "stochastic_failure_filtered",
                        },
                    }
                )
            else:
                representative.metadata["failure_confirmation"] = {
                    "total_runs": total_runs,
                    "failures": observed_failures,
                    "min_failures": confirmation_min_failures,
                    "hard_rewards": [
                        source_trace.hard_reward,
                        *[item.hard_reward for item in confirmations],
                    ],
                }
                persistent.append(
                    (
                        source_trace,
                        trigger_type,
                        self._failure_cluster_key(representative),
                    )
                )

        clusters_by_key: dict[
            tuple[str, ...], list[tuple[ExecutionTrace, str]]
        ] = {}
        for source_trace, trigger_type, key in persistent:
            clusters_by_key.setdefault(key, []).append((source_trace, trigger_type))
        max_cluster = int(getattr(optimizer_config, "failure_cluster_max_size", 8))
        clusters: list[
            tuple[tuple[str, ...], list[tuple[ExecutionTrace, str]]]
        ] = []
        for key, members in clusters_by_key.items():
            for start in range(0, len(members), max_cluster):
                clusters.append((key, members[start:start + max_cluster]))

        guard_cache_version: str | None = None
        guard_replays: list[ExecutionTrace] = []
        rollback_count = 0

        for cluster_index, (cluster_key, members) in enumerate(clusters):
            cluster_id = f"r{round_num}-c{cluster_index}"
            previous_workflow = self.workflow
            # Re-execute every source at the current incumbent version.  This
            # is the key difference from the old sequential implementation:
            # the optimizer sees a multi-failure batch rather than one query.
            current_failures: list[ExecutionTrace] = []
            source_replays: list[tuple[ExecutionTrace, ExecutionTrace, str]] = []
            for member_index, (source_trace, trigger_type) in enumerate(members):
                # The probe is already version-matched for the first incumbent
                # (and remains valid after a rejected cluster).  Re-run only
                # after an accepted workflow version changes, avoiding a
                # duplicate full replay cost in the common first cluster.
                replay = probe_replays.get(source_trace.trace_id)
                if (
                    replay is None
                    or replay.workflow_version != self.workflow.version
                ):
                    replay = await self._replay_deferred_failure(
                        source_trace,
                        f"{cluster_id}-{member_index}",
                    )
                    replay_traces.append(replay)
                source_replays.append((source_trace, replay, trigger_type))
                if not self._is_hard_success(replay, threshold):
                    current_failures.append(replay)

            if guard_sources and guard_cache_version != self.workflow.version:
                guard_replays = []
                for guard_index, guard_source in enumerate(guard_sources):
                    guard_replay = await self._replay_deferred_failure(
                        guard_source,
                        f"{cluster_id}-guard-{guard_index}",
                    )
                    guard_replays.append(guard_replay)
                    replay_traces.append(guard_replay)
                guard_cache_version = self.workflow.version

            # Freeze pre-replay roles: only persistent cluster members are
            # target failures. A nominal guard that fails under the incumbent
            # is too unstable to prove non-regression and must not be allowed
            # to satisfy the any-repair gate.
            stable_guard_replays, unstable_guard_replays = (
                self._partition_replayed_guards(guard_replays, threshold)
            )

            if not current_failures:
                # All members were repaired by a preceding accepted cluster;
                # retain an explicit audit row without asking the LLM again.
                self._deferred_round_updates.append(
                    {
                        "update_index": cluster_id,
                        "cluster_id": cluster_id,
                        "cluster_key": list(cluster_key),
                        "cluster_size": len(members),
                        "source_trace_ids": [item[0].trace_id for item in source_replays],
                        "replay_trace_ids": [item[1].trace_id for item in source_replays],
                        "trigger_type": "cluster",
                        "update_status": "cluster_replay_success",
                        "accepted": False,
                        "workflow_version_before": self.workflow.version,
                        "workflow_version_after": self.workflow.version,
                        "replay_hard_rewards": [item[1].hard_reward for item in source_replays],
                        "input_tokens": sum(item[1].total_prompt_tokens for item in source_replays),
                        "output_tokens": sum(item[1].total_completion_tokens for item in source_replays),
                        "latency_seconds": sum(item[1].total_latency_seconds for item in source_replays),
                        "guard_replay_count": len(guard_replays),
                        "stable_guard_replay_count": len(stable_guard_replays),
                        "unstable_guard_replay_count": len(unstable_guard_replays),
                        "summary": {"accepted": False, "reason": "cluster_replay_success"},
                    }
                )
                continue

            self.optimizer.failure_buffer.clear()
            self.optimizer.failure_buffer.extend(current_failures)
            for guard_replay in stable_guard_replays:
                self.optimizer.failure_buffer.add(guard_replay)
            try:
                candidate_workflow, summary = await self.optimizer.optimize_round(
                    previous_workflow,
                    self.executor,
                    self.scheduler,
                    llm_client=self.llm_client,
                )
            except Exception as exc:
                logger.exception("[failure cluster %s] workflow update failed", cluster_id)
                candidate_workflow = previous_workflow
                summary = {
                    "accepted": False,
                    "error": str(exc),
                    "round": len(self.optimizer.round_history) + 1,
                }
            accepted = bool(summary.get("accepted", False))
            update_record = {
                "update_index": cluster_id,
                "cluster_id": cluster_id,
                "cluster_key": list(cluster_key),
                "cluster_size": len(members),
                "source_trace_ids": [item[0].trace_id for item in source_replays],
                "replay_trace_ids": [item[1].trace_id for item in source_replays],
                "sample_indices": [item[0].metadata.get("split_example_index") for item in source_replays],
                "trigger_type": "cluster",
                "workflow_version_before": previous_workflow.version,
                "workflow_version_after": candidate_workflow.version,
                "accepted": accepted,
                "source_hard_rewards": [item[0].hard_reward for item in source_replays],
                "replay_hard_rewards": [item[1].hard_reward for item in source_replays],
                "input_tokens": sum(item[1].total_prompt_tokens for item in source_replays),
                "output_tokens": sum(item[1].total_completion_tokens for item in source_replays),
                "latency_seconds": sum(item[1].total_latency_seconds for item in source_replays),
                "guard_replay_count": len(guard_replays),
                "stable_guard_replay_count": len(stable_guard_replays),
                "unstable_guard_replay_count": len(unstable_guard_replays),
                "unstable_guard_replay_ids": [
                    item.trace_id for item in unstable_guard_replays
                ],
                "guard_replay_input_tokens": sum(item.total_prompt_tokens for item in guard_replays),
                "guard_replay_output_tokens": sum(item.total_completion_tokens for item in guard_replays),
                "guard_replay_latency_seconds": sum(item.total_latency_seconds for item in guard_replays),
                "suffix_replay_requested": bool(self.optimizer.counterfactual.suffix_replay),
                "suffix_replay_used": bool(summary.get("suffix_replay_used", False)),
                "candidate_scopes": list(getattr(self.optimizer, "_candidate_scopes", ())),
                "summary": summary,
            }
            self._deferred_round_updates.append(update_record)
            if accepted:
                self.workflow = candidate_workflow
                logger.info(
                    "[failure cluster %s] accepted multi-failure update %s -> %s",
                    cluster_id,
                    previous_workflow.version,
                    candidate_workflow.version,
                )
            else:
                logger.info("[failure cluster %s] workflow update rejected", cluster_id)

        self._save_traces("deferred_failure_replay", replay_traces)
        summary = self._summarize_deferred_round(
            round_num,
            round_start_workflow,
            self.workflow,
            self._deferred_round_updates,
            failure_count=len(failures),
            trigger_count=len(triggers),
            efficiency_trigger_count=len(efficiency_sources),
            buffered_failure_count=len(trace_batch.failures),
            failure_cluster_count=len(clusters),
            rollback_count=rollback_count,
            clustered=True,
            incumbent_replay_success_count=incumbent_replay_successes,
        )
        return self.workflow, summary

    @staticmethod
    def _summarize_deferred_round(
        round_num: int,
        start_workflow: WorkflowTemplate,
        final_workflow: WorkflowTemplate,
        updates: list[dict[str, Any]],
        *,
        failure_count: int,
        trigger_count: int,
        efficiency_trigger_count: int,
        buffered_failure_count: int,
        failure_cluster_count: int = 0,
        rollback_count: int = 0,
        clustered: bool = False,
        incumbent_replay_success_count: int = 0,
    ) -> dict[str, Any]:
        accepted = [item for item in updates if item.get("accepted")]
        return {
            "round": round_num,
            "workflow_name": start_workflow.name,
            "workflow_version": start_workflow.version,
            "workflow_version_after_deferred_updates": final_workflow.version,
            "failure_update_mode": (
                "deferred_clustered" if clustered else "deferred_sequential"
            ),
            "failure_cluster_enabled": clustered,
            "failure_cluster_count": failure_cluster_count,
            "rollback_count": rollback_count,
            "incumbent_replay_success_count": incumbent_replay_success_count,
            "deferred_failure_count": failure_count,
            "deferred_trigger_count": trigger_count,
            "deferred_efficiency_trigger_count": efficiency_trigger_count,
            "buffered_failure_count": buffered_failure_count,
            "deferred_update_count": len(updates),
            "deferred_accepted_count": len(accepted),
            "deferred_rejected_count": len(updates) - len(accepted),
            "deferred_replay_input_tokens": sum(
                int(item.get("input_tokens", 0))
                + int(item.get("guard_replay_input_tokens", 0))
                for item in updates
            ),
            "deferred_replay_output_tokens": sum(
                int(item.get("output_tokens", 0))
                + int(item.get("guard_replay_output_tokens", 0))
                for item in updates
            ),
            "deferred_replay_latency_seconds": sum(
                float(item.get("latency_seconds", 0.0))
                + float(item.get("guard_replay_latency_seconds", 0.0))
                for item in updates
            ),
            "deferred_guard_replay_count": sum(
                int(item.get("guard_replay_count", 0)) for item in updates
            ),
            "deferred_updates": updates,
            "accepted": bool(accepted),
            "gain": sum(
                float(item.get("summary", {}).get("gain", 0.0))
                for item in accepted
            ),
            "candidate_description": (
                accepted[-1].get("summary", {}).get(
                    "candidate_description", ""
                )
                if accepted
                else ""
            ),
            "candidate_evaluations": [],
            "scope_attempts": [],
            "generated_candidates": sum(
                int(item.get("summary", {}).get("generated_candidates", 0))
                for item in updates
            ),
            "evaluated_candidates": sum(
                int(item.get("summary", {}).get("evaluated_candidates", 0))
                for item in updates
            ),
            "suffix_replay_requested": any(
                item.get("suffix_replay_requested") for item in updates
            ),
            "suffix_replay_used": any(
                item.get("suffix_replay_used") for item in updates
            ),
        }

    async def _handle_online_sample(
        self,
        trace: ExecutionTrace,
        hard_reward: float,
        process_reward: float,
        composite_reward: float,
        utility: float,
    ) -> None:
        """Log one optimization sample and update immediately on failure.

        The failure buffer is deliberately reset after every attempted local
        update.  This makes each counterfactual baseline belong to exactly the
        workflow version that produced the triggering trace and lets
        ``CounterfactualEvaluator`` use that trace's prefix checkpoint for
        suffix replay.
        """
        threshold = float(
            getattr(
                getattr(self.config, "optimizer", None),
                "hard_success_threshold",
                1.0,
            )
        )
        success = bool(trace.success and hard_reward >= threshold)
        logger.info(
            "[optimization][sample %s] success=%s hard_reward=%.4f "
            "process_reward=%.4f utility=%.4f workflow=%s version=%s "
            "input_tokens=%s output_tokens=%s latency=%.3fs",
            trace.metadata.get("split_example_index", "?"),
            success,
            hard_reward,
            process_reward,
            utility,
            trace.workflow_name,
            trace.workflow_version,
            trace.total_prompt_tokens,
            trace.total_completion_tokens,
            trace.total_latency_seconds,
        )
        if success:
            # Successes are intentionally not optimization triggers.  They can
            # still be used as guards when a caller configures a non-zero
            # ``min_success_guards`` and a failure arrives before the buffer is
            # consumed.
            self.optimizer.failure_buffer.add(trace)
            return

        logger.info(
            "[optimization][sample %s] failure detected; triggering "
            "counterfactual workflow update (suffix_replay=%s)",
            trace.metadata.get("split_example_index", "?"),
            bool(self.optimizer.counterfactual.suffix_replay),
        )
        self.optimizer.failure_buffer.add(trace)
        previous_workflow = self.workflow
        try:
            candidate_workflow, summary = await self.optimizer.optimize_round(
                previous_workflow,
                self.executor,
                self.scheduler,
                llm_client=self.llm_client,
            )
        except Exception as exc:
            logger.exception(
                "[optimization][sample %s] online update failed",
                trace.metadata.get("split_example_index", "?"),
            )
            summary = {
                "accepted": False,
                "error": str(exc),
                "round": len(self.optimizer.round_history) + 1,
            }
            candidate_workflow = previous_workflow
        accepted = bool(summary.get("accepted", False))
        update_record = {
            "sample_index": trace.metadata.get("split_example_index"),
            "trace_id": trace.trace_id,
            "workflow_version_before": previous_workflow.version,
            "workflow_version_after": candidate_workflow.version,
            "accepted": accepted,
            "hard_reward": hard_reward,
            "process_reward": process_reward,
            "input_tokens": trace.total_prompt_tokens,
            "output_tokens": trace.total_completion_tokens,
            "latency_seconds": trace.total_latency_seconds,
            "suffix_replay_requested": bool(
                self.optimizer.counterfactual.suffix_replay
            ),
            "suffix_replay_used": bool(
                summary.get("suffix_replay_used", False)
            ),
            "summary": summary,
        }
        self._online_round_updates.append(update_record)
        if accepted:
            self.workflow = candidate_workflow
            logger.info(
                "[optimization][sample %s] online workflow update accepted: "
                "%s -> %s, gain=%.4f, suffix_replay_used=%s",
                trace.metadata.get("split_example_index", "?"),
                previous_workflow.version,
                candidate_workflow.version,
                float(summary.get("gain", 0.0)),
                bool(summary.get("suffix_replay_used", False)),
            )
        else:
            logger.info(
                "[optimization][sample %s] online workflow update rejected",
                trace.metadata.get("split_example_index", "?"),
            )

    @staticmethod
    def _summarize_online_round(
        round_num: int,
        start_workflow: WorkflowTemplate,
        final_workflow: WorkflowTemplate,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a batch-compatible summary for streamed local updates."""
        accepted = [item for item in updates if item.get("accepted")]
        return {
            "round": round_num,
            "workflow_name": start_workflow.name,
            "workflow_version": start_workflow.version,
            "workflow_version_after_online_updates": final_workflow.version,
            "online_failure_updates": True,
            "online_update_count": len(updates),
            "online_accepted_count": len(accepted),
            "online_rejected_count": len(updates) - len(accepted),
            "online_updates": updates,
            "accepted": bool(accepted),
            "gain": sum(
                float(item.get("summary", {}).get("gain", 0.0))
                for item in accepted
            ),
            "candidate_description": (
                accepted[-1].get("summary", {}).get(
                    "candidate_description", ""
                )
                if accepted
                else ""
            ),
            "candidate_evaluations": [],
            "scope_attempts": [],
            "generated_candidates": sum(
                int(item.get("summary", {}).get("generated_candidates", 0))
                for item in updates
            ),
            "evaluated_candidates": sum(
                int(item.get("summary", {}).get("evaluated_candidates", 0))
                for item in updates
            ),
            "suffix_replay_requested": any(
                item.get("suffix_replay_requested") for item in updates
            ),
            "suffix_replay_used": any(
                item.get("suffix_replay_used") for item in updates
            ),
        }

    async def _validate(self, workflow: WorkflowTemplate) -> float:
        """Compatibility wrapper returning validation runtime utility."""
        metrics = await self._evaluate_dataset(
            workflow, self.val_data, split_name="validation"
        )
        return metrics["runtime_utility"]

    async def _evaluate_test(self, workflow: WorkflowTemplate) -> float:
        """Compatibility wrapper returning held-out composite reward."""
        metrics = await self._evaluate_dataset(
            workflow, self.test_data, split_name="test"
        )
        return metrics["composite_reward"]

    async def evaluate_test(self, workflow: WorkflowTemplate) -> dict[str, Any]:
        """Evaluate a selected workflow once on the held-out test split."""
        return await self._evaluate_dataset(
            workflow, self.test_data, split_name="test"
        )

    @classmethod
    def for_evaluation(
        cls,
        config: ExperimentConfig,
        workflow: WorkflowTemplate,
        reward_evaluator: RewardEvaluator,
        operators: Optional[dict[str, callable]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> "ExperimentRunner":
        """Create an evaluation-only runner without constructing an optimizer.

        This deliberately bypasses ``__init__`` so official held-out testing
        cannot instantiate the optimizer or its separate API client.
        """
        runner = cls.__new__(cls)
        runner.config = config.model_copy(deep=True)
        if (
            runner.config.optimizer.workflow_content_only
            and runner.config.scheduler.scheduler_type != "cascade"
        ):
            runner.config.scheduler.allow_deviation = False
            if workflow.selective_update is not None:
                payload = workflow.model_dump(mode="json")
                payload["selective_update"] = None
                workflow = WorkflowTemplate.model_validate(payload)
                logger.info(
                    "workflow_content_only=True; discarded an inherited "
                    "selective execution policy for evaluation"
                )
        if runner.config.scheduler.llm.seed is None:
            runner.config.scheduler.llm.seed = runner.config.seed
        workflow_llm_config = (
            runner.config.workflow_llm or runner.config.scheduler.llm
        )
        if workflow_llm_config.seed is None:
            workflow_llm_config.seed = runner.config.seed
        if runner.config.optimizer.llm.seed is None:
            runner.config.optimizer.llm.seed = runner.config.seed + 1
        calibration_payload = workflow.metadata.get("scheduler_calibration")
        if isinstance(calibration_payload, dict):
            runner.config.scheduler = SchedulerConfig.model_validate(
                calibration_payload
            )
        runner.workflow = workflow
        runner.reward_evaluator = reward_evaluator
        runner.operators = operators or {}
        runner.run_metadata = {}
        runner.output_dir = Path(
            output_dir
            if output_dir is not None
            else Path(runner.config.output_dir) / runner.config.name
        )
        runner.llm_client = AsyncLLMClient(workflow_llm_config)
        runner.executor = RuntimeExecutor(
            config=runner.config.executor,
            operators=runner.operators,
        )
        runner.scheduler = runner._create_scheduler(runner.config.scheduler)
        runner.utility_computer = UtilityComputer(
            lambda_cost=runner.config.optimizer.lambda_cost,
            lambda_latency=runner.config.optimizer.lambda_latency,
            lambda_api_cost=runner.config.optimizer.lambda_api_cost,
            rho_omega=runner.config.optimizer.rho_omega,
        )
        runner.opt_data = []
        runner.val_data = []
        runner.test_data = []
        runner.split_indices = {}
        runner.manifest = None
        runner._initial_workflow = workflow.model_copy(deep=True)
        # Official evaluation persists its own private trace artifact under
        # ``official_test/`` when trace persistence is enabled in the config.
        runner._persist_traces = True
        runner._current_round = "official_test"
        runner._initialized_trace_splits = set()
        return runner

    def load_exact_test_data(
        self,
        data: list[tuple[str, Any]],
        indices: list[int],
    ) -> None:
        """Load only the manifest-selected rows for held-out evaluation."""
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(data)
            for index in indices
        ):
            raise ValueError("Manifest contains invalid held-out test indices")
        if len(indices) != len(set(indices)):
            raise ValueError("Manifest contains duplicate held-out test indices")
        if not indices:
            raise ValueError("Manifest held-out test split is empty")
        self.test_data = [data[index] for index in indices]

    async def _evaluate_dataset(
        self,
        workflow: WorkflowTemplate,
        data: list[tuple[str, Any]],
        split_name: str,
    ) -> dict[str, Any]:
        _, metrics = await self._execute_dataset(workflow, data, split_name)
        return metrics

    async def _execute_dataset(
        self,
        workflow: WorkflowTemplate,
        data: list[tuple[str, Any]],
        split_name: str,
        *,
        workflow_provider: Optional[Callable[[], WorkflowTemplate]] = None,
        sample_callback: Optional[
            Callable[[ExecutionTrace, float, float, float, float], Awaitable[None]]
        ] = None,
    ) -> tuple[list[ExecutionTrace], dict[str, Any]]:
        """Execute a split and compute task, cost, and structural metrics."""
        if not data:
            raise ValueError(f"{split_name} split is empty")

        traces: list[ExecutionTrace] = []
        composite_rewards: list[float] = []
        utilities: list[float] = []
        hard_rewards: list[float] = []
        process_rewards: list[float] = []

        for example_index, (query, ground_truth) in enumerate(data):
            active_workflow = (
                workflow_provider() if workflow_provider is not None else workflow
            )
            output, _, recorder = await self.executor.execute(
                active_workflow,
                self.scheduler,
                query,
                llm_client=self.llm_client,
            )
            trace = recorder.trace
            trace.metadata.update(
                {
                    "ground_truth": ground_truth,
                    "split": split_name,
                    "experiment_round": self._current_round,
                    "split_example_index": example_index,
                }
            )

            hard = self.reward_evaluator.hard_reward(
                query, ground_truth, output, trace
            )
            process = self.reward_evaluator.process_reward(
                query, ground_truth, output, trace
            )
            composite = hard + self.config.reward.alpha_process * process
            trace.hard_reward = hard
            trace.process_reward = process

            hard_rewards.append(hard)
            process_rewards.append(process)
            composite_rewards.append(composite)
            # Runtime utility is the hard-reward metric only.  Composite
            # (hard + process) reward remains a reported process diagnostic;
            # token and edit penalties are applied later by candidate gain
            # scoring, not folded into this per-execution utility.
            utilities.append(self.utility_computer.compute(hard, trace))
            traces.append(trace)

            success = bool(
                trace.success
                and hard
                >= float(
                    getattr(
                        getattr(self.config, "optimizer", None),
                        "hard_success_threshold",
                        1.0,
                    )
                )
            )
            logger.info(
                "[%s][sample %s/%s] success=%s hard_reward=%.4f "
                "process_reward=%.4f input_tokens=%s output_tokens=%s "
                "latency=%.3fs workflow=%s version=%s",
                split_name,
                example_index + 1,
                len(data),
                success,
                hard,
                process,
                trace.total_prompt_tokens,
                trace.total_completion_tokens,
                trace.total_latency_seconds,
                trace.workflow_name,
                trace.workflow_version,
            )
            if sample_callback is not None:
                await sample_callback(
                    trace,
                    float(hard),
                    float(process),
                    float(composite),
                    float(utilities[-1]),
                )

        complexity_counts = {
            "repair": 0,
            "reroute": 0,
            "fallback": 0,
            "loop": 0,
        }
        for trace in traces:
            components = self.utility_computer.runtime_complexity_components(
                trace
            )
            for name, count in components.items():
                complexity_counts[name] += count
        total_input = sum(t.total_prompt_tokens for t in traces)
        total_output = sum(t.total_completion_tokens for t in traces)
        llm_calls = sum(trace.total_llm_calls for trace in traces)
        tool_calls = sum(trace.total_tool_calls for trace in traces)
        cost_estimate_complete = all(
            trace.total_cost_estimate_complete for trace in traces
        )
        known_api_cost = sum(t.total_cost_usd for t in traces)
        llm_breakdown = self._summarize_trace_llm_calls(traces)
        selective_decisions = [
            decision
            for trace in traces
            if isinstance(
                decision := trace.metadata.get("selective_update"),
                dict,
            )
        ]
        action_counts: Counter[str] = Counter()
        decision_node_counts: Counter[str] = Counter()
        executed_node_counts: Counter[str] = Counter()
        for trace in traces:
            for step in trace.steps:
                if step.action:
                    action_counts[step.action] += 1
                decision_node_id = step.metadata.get(
                    "decision_node_id",
                    step.node_id,
                )
                if decision_node_id:
                    decision_node_counts[str(decision_node_id)] += 1
                if (
                    step.metadata.get("node_executed", True)
                    and step.metadata.get(
                        "scheduler_action_validated",
                        True,
                    )
                ):
                    executed_node_counts[step.node_id] += 1

        gate_invocations = 0
        gate_latency_seconds = 0.0
        gate_route_counts: Counter[str] = Counter()
        gate_formulas: Counter[str] = Counter()
        scheduler_fallbacks = 0
        for trace in traces:
            telemetry = trace.metadata.get("scheduler_telemetry", {})
            if not isinstance(telemetry, dict):
                continue
            try:
                gate_invocations += int(telemetry.get("gate_invocations", 0))
            except (TypeError, ValueError):
                pass
            try:
                gate_latency_seconds += float(
                    telemetry.get("gate_latency_seconds", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                pass
            route_counts = telemetry.get("gate_route_counts", {})
            if isinstance(route_counts, dict):
                for route, count in route_counts.items():
                    try:
                        gate_route_counts[str(route)] += int(count)
                    except (TypeError, ValueError):
                        continue
            formula = telemetry.get("gate_formula")
            if formula:
                gate_formulas[str(formula)] += 1
            try:
                scheduler_fallbacks += int(
                    telemetry.get("scheduler_fallbacks", 0)
                )
            except (TypeError, ValueError):
                pass

        optimizer_config = getattr(self.config, "optimizer", None)
        workflow_content_only = bool(
            getattr(optimizer_config, "workflow_content_only", True)
        )
        metrics: dict[str, Any] = {
            "split": split_name,
            "num_examples": len(traces),
            "hard_success_rate": sum(r >= 1.0 for r in hard_rewards) / len(traces),
            "hard_reward": sum(hard_rewards) / len(traces),
            "process_reward": sum(process_rewards) / len(traces),
            "composite_reward": sum(composite_rewards) / len(traces),
            "runtime_utility": sum(utilities) / len(traces),
            "runtime_utility_definition": "hard_reward",
            "workflow_content_only": workflow_content_only,
            "outer_execution_optimization_enabled": self._outer_scheduler_enabled(
                self.config
            ),
            "scheduler_deviation_enabled": bool(
                getattr(
                    getattr(self.scheduler, "config", None),
                    "allow_deviation",
                    False,
                )
            ),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "llm_call_count": llm_calls,
            "tool_call_count": tool_calls,
            "latency_seconds": sum(t.total_latency_seconds for t in traces),
            "estimated_api_cost_usd": (
                known_api_cost if cost_estimate_complete else None
            ),
            "known_partial_api_cost_usd": known_api_cost,
            "api_cost_estimate_complete": cost_estimate_complete,
            "workflow_llm": llm_breakdown["workflow"],
            "scheduler_llm": llm_breakdown["scheduler"],
            "workflow_llm_call_count": (
                llm_breakdown["workflow"]["call_count"]
            ),
            "workflow_llm_total_tokens": (
                llm_breakdown["workflow"]["total_tokens"]
            ),
            "workflow_llm_latency_seconds": (
                llm_breakdown["workflow"]["latency_seconds"]
            ),
            "workflow_llm_estimated_api_cost_usd": (
                llm_breakdown["workflow"]["estimated_api_cost_usd"]
            ),
            "scheduler_llm_call_count": (
                llm_breakdown["scheduler"]["call_count"]
            ),
            "scheduler_llm_total_tokens": (
                llm_breakdown["scheduler"]["total_tokens"]
            ),
            "scheduler_llm_latency_seconds": (
                llm_breakdown["scheduler"]["latency_seconds"]
            ),
            "scheduler_llm_estimated_api_cost_usd": (
                llm_breakdown["scheduler"]["estimated_api_cost_usd"]
            ),
            "scheduler_gate_invocations": gate_invocations,
            "scheduler_gate_latency_seconds": gate_latency_seconds,
            "scheduler_gate_route_counts": dict(sorted(gate_route_counts.items())),
            "scheduler_gate_formulas": dict(sorted(gate_formulas.items())),
            "scheduler_gate_early_exit_count": gate_route_counts.get(
                "early_exit", 0
            ),
            "scheduler_gate_dispatch_count": gate_route_counts.get(
                "invoke_scheduler", 0
            ),
            "scheduler_gate_continue_count": gate_route_counts.get(
                "continue", 0
            ),
            "scheduler_fallback_count": scheduler_fallbacks,
            "selective_gate_configured_count": len(selective_decisions),
            "selective_gate_applied_count": sum(
                decision.get("applied") is True
                for decision in selective_decisions
            ),
            "selective_gate_coverage": (
                sum(
                    decision.get("applied") is True
                    for decision in selective_decisions
                )
                / len(selective_decisions)
                if selective_decisions
                else None
            ),
            "selective_gate_fail_closed_count": sum(
                decision.get("fail_closed") is True
                for decision in selective_decisions
            ),
            "action_count": sum(action_counts.values()),
            "action_counts": dict(sorted(action_counts.items())),
            "decision_node_count": sum(decision_node_counts.values()),
            "decision_node_counts": dict(
                sorted(decision_node_counts.items())
            ),
            "node_execution_count": sum(executed_node_counts.values()),
            "executed_node_counts": dict(
                sorted(executed_node_counts.items())
            ),
            **{
                f"{name}_count": count
                for name, count in complexity_counts.items()
            },
        }
        self._save_traces(split_name, traces)
        return traces, metrics

    @staticmethod
    def _summarize_trace_llm_calls(
        traces: list[ExecutionTrace],
    ) -> dict[str, dict[str, Any]]:
        """Split trace-recorded LLM work into workflow and scheduler calls."""
        summaries: dict[str, dict[str, Any]] = {
            role: {
                "call_count": 0,
                "successful_call_count": 0,
                "failed_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "reasoning_tokens": 0,
                "latency_seconds": 0.0,
                "known_partial_api_cost_usd": 0.0,
                "api_cost_estimate_complete": True,
            }
            for role in ("workflow", "scheduler")
        }
        for trace in traces:
            for step in trace.steps:
                for call in step.llm_calls:
                    role = (
                        "scheduler"
                        if call.call_type == "scheduler"
                        else "workflow"
                    )
                    summary = summaries[role]
                    summary["call_count"] += 1
                    if call.success:
                        summary["successful_call_count"] += 1
                    else:
                        summary["failed_call_count"] += 1
                    summary["input_tokens"] += call.prompt_tokens
                    summary["output_tokens"] += call.completion_tokens
                    summary["total_tokens"] += (
                        call.prompt_tokens + call.completion_tokens
                    )
                    summary["latency_seconds"] += call.latency_seconds
                    summary["known_partial_api_cost_usd"] += call.cost_usd
                    summary["api_cost_estimate_complete"] = bool(
                        summary["api_cost_estimate_complete"]
                        and call.success
                        and call.cost_estimate_available
                    )
                    usage = call.metadata.get("usage", {})
                    if not isinstance(usage, dict):
                        continue
                    for source, target in (
                        (
                            "prompt_cache_hit_tokens",
                            "prompt_cache_hit_tokens",
                        ),
                        (
                            "prompt_cache_miss_tokens",
                            "prompt_cache_miss_tokens",
                        ),
                        ("reasoning_tokens", "reasoning_tokens"),
                    ):
                        try:
                            value = max(int(usage.get(source, 0) or 0), 0)
                        except (TypeError, ValueError):
                            value = 0
                        summary[target] += value

        for summary in summaries.values():
            summary["estimated_api_cost_usd"] = (
                summary["known_partial_api_cost_usd"]
                if summary["api_cost_estimate_complete"]
                else None
            )
        return summaries

    def _backend_usage_snapshot(self) -> dict[str, Any]:
        """Snapshot every logical LLM backend used by an experiment."""
        scheduler_client = getattr(
            getattr(self, "scheduler", None),
            "llm",
            None,
        )
        optimizer_client = getattr(
            getattr(self, "optimizer", None),
            "llm",
            None,
        )
        clients = {
            "workflow_client": getattr(self, "llm_client", None),
            "scheduler_client": scheduler_client,
            "optimizer_client": optimizer_client,
        }
        snapshot = {
            role: self._client_usage_snapshot(client)
            for role, client in clients.items()
        }
        snapshot["total"] = self._aggregate_usage(snapshot)
        return snapshot

    @staticmethod
    def _client_usage_snapshot(client: object | None) -> dict[str, Any]:
        fields: dict[str, int | float] = {
            "num_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "reasoning_tokens": 0,
            "total_cost_usd": 0.0,
            "total_latency_seconds": 0.0,
            "unknown_pricing_calls": 0,
        }
        getter = getattr(client, "get_usage_summary", None)
        if not callable(getter):
            return {
                "available": False,
                "model": None,
                **fields,
                "cost_estimate_complete": True,
            }
        raw = getter()
        if not isinstance(raw, dict):
            return {
                "available": False,
                "model": None,
                **fields,
                "cost_estimate_complete": True,
            }
        for key, default in fields.items():
            value = raw.get(key, default)
            try:
                fields[key] = (
                    float(value)
                    if isinstance(default, float)
                    else int(value)
                )
            except (TypeError, ValueError):
                fields[key] = default
        return {
            "available": True,
            "model": raw.get(
                "model",
                getattr(getattr(client, "config", None), "model", None),
            ),
            **fields,
            "cost_estimate_complete": bool(
                raw.get(
                    "cost_estimate_complete",
                    fields["unknown_pricing_calls"] == 0,
                )
            ),
        }

    @classmethod
    def _aggregate_usage(
        cls,
        usage_by_role: dict[str, Any],
    ) -> dict[str, Any]:
        numeric_fields = (
            "num_calls",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
            "total_cost_usd",
            "total_latency_seconds",
            "unknown_pricing_calls",
        )
        components = [
            value
            for role, value in usage_by_role.items()
            if role != "total" and isinstance(value, dict)
        ]
        aggregate = {
            key: sum(component.get(key, 0) for component in components)
            for key in numeric_fields
        }
        aggregate["available"] = any(
            component.get("available", False) for component in components
        )
        aggregate["models"] = sorted(
            {
                str(component["model"])
                for component in components
                if component.get("available") and component.get("model")
            }
        )
        aggregate["cost_estimate_complete"] = all(
            component.get("cost_estimate_complete", True)
            for component in components
        )
        return aggregate

    @classmethod
    def _backend_usage_delta(
        cls,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Subtract two monotonic tracker snapshots role by role."""
        numeric_fields = (
            "num_calls",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
            "total_cost_usd",
            "total_latency_seconds",
            "unknown_pricing_calls",
        )
        delta: dict[str, Any] = {}
        for role in (
            "workflow_client",
            "scheduler_client",
            "optimizer_client",
        ):
            old = before.get(role, {})
            new = after.get(role, {})
            role_delta: dict[str, Any] = {
                "available": bool(new.get("available", False)),
                "model": new.get("model"),
            }
            for field in numeric_fields:
                value = new.get(field, 0) - old.get(field, 0)
                # Tracker values are monotonic; clamp tiny floating-point
                # subtraction noise without masking a genuine reset.
                role_delta[field] = max(value, 0)
            role_delta["cost_estimate_complete"] = (
                role_delta["unknown_pricing_calls"] == 0
            )
            delta[role] = role_delta
        delta["total"] = cls._aggregate_usage(delta)
        return delta

    def _save_traces(
        self,
        split_name: str,
        traces: list[ExecutionTrace],
    ) -> None:
        if (
            not self.config.executor.trace_enabled
            or not getattr(self, "_persist_traces", True)
        ):
            return
        trace_dir = self.output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{split_name}.jsonl"
        first_write = split_name not in self._initialized_trace_splits
        write_mode = os.O_TRUNC if first_write else os.O_APPEND
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | write_mode,
            0o600,
        )
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a") as f:
            for trace in traces:
                payload = trace.model_dump(
                    mode="json",
                    warnings="none",
                    fallback=str,
                )
                f.write(
                    json.dumps(payload, ensure_ascii=False, default=str) + "\n"
                )
        self._initialized_trace_splits.add(split_name)

    def _public_config(self) -> dict[str, Any]:
        """Return a recursively redacted configuration for result artifacts."""
        return public_scientific_config(self.config)

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save experiment results to disk."""
        atomic_write_json_0600(self.output_dir / "results.json", results)
