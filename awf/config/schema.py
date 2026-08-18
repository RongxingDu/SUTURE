"""Pydantic configuration models for the SUTURE framework."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictConfigModel(BaseModel):
    """Configuration base that rejects misspelled or obsolete fields."""

    model_config = ConfigDict(extra="forbid")


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    ALIYUN = "aliyun"
    CUSTOM = "custom"


class LLMConfig(_StrictConfigModel):
    """Configuration for an LLM endpoint used by scheduler or optimizer."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = Field(default="gpt-4o", min_length=1)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    seed: Optional[int] = None
    # Additional chat-completion request options (for example ``top_p``).
    # Core request fields cannot be overridden here.
    extra_kwargs: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_extra_kwargs(self) -> "LLMConfig":
        if self.provider == LLMProvider.ANTHROPIC:
            raise ValueError(
                "provider 'anthropic' is reserved but not implemented; "
                "configure an OpenAI-compatible API endpoint instead"
            )
        if (
            self.provider in {LLMProvider.DEEPSEEK, LLMProvider.ALIYUN, LLMProvider.CUSTOM}
            and not self.api_base
        ):
            raise ValueError(
                f"provider '{self.provider.value}' requires an explicit "
                "api_base"
            )
        reserved = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "seed",
            "stop",
            "response_format",
        }
        conflicts = reserved.intersection(self.extra_kwargs)
        if conflicts:
            raise ValueError(
                "extra_kwargs cannot override core request fields: "
                + ", ".join(sorted(conflicts))
            )
        return self


class SchedulerConfig(_StrictConfigModel):
    """Configuration for the LLM-based scheduler."""

    scheduler_type: Literal[
        "fixed",
        "graph",
        "cascade",
    ] = "fixed"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_actions_per_query: int = Field(default=50, gt=0)
    system_prompt: str = ""
    # Whether the scheduler can deviate from the workflow template
    allow_deviation: bool = True
    # Two-stage outer scheduler controls.  The gate is deliberately made of
    # deterministic, query-local features so it adds no generative call or
    # token cost.  The defaults follow the ACL cascade ablation weights for
    # the HumanEval-style setting (spec/lite/agreement/history).
    gate_enabled: bool = True
    gate_spec_weight: float = Field(default=0.07, ge=0.0)
    gate_lite_weight: float = Field(default=0.82, ge=0.0)
    gate_agreement_weight: float = Field(default=0.07, ge=0.0)
    gate_history_weight: float = Field(default=0.04, ge=0.0)
    gate_schedule_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    gate_early_exit_threshold: float = Field(default=0.824, ge=0.0, le=1.0)
    gate_high_risk_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    # LAS-faithful mode forwards promising artifacts to the LLM scheduler;
    # the historical zero-call direct early-exit remains available as an
    # explicit ablation for backwards-compatible tests/configurations.
    gate_formula: Literal["las", "legacy"] = "las"
    gate_direct_early_exit: bool = True
    early_exit_enabled: bool = True
    # When non-empty, the outer policy is evaluated only immediately before
    # these intervention nodes. This turns workflow-update regressions into a
    # selective application decision instead of repeatedly scheduling every
    # ordinary reasoning node.
    gate_node_allowlist: list[str] = Field(default_factory=list)
    scheduler_max_tokens: int = Field(default=384, gt=0)
    scheduler_query_max_chars: int = Field(default=320, gt=0)
    scheduler_artifact_max_chars: int = Field(default=1200, gt=0)
    # Research calibration is performed only after the workflow action space
    # has been frozen.  The grid is intentionally small and deterministic so
    # the validation split selects a gate rather than training another opaque
    # generative policy.
    calibration_enabled: bool = False
    calibration_thresholds: list[float] = Field(
        default_factory=lambda: [0.45, 0.55, 0.65, 0.75],
    )
    calibration_weight_candidates: list[list[float]] = Field(
        default_factory=lambda: [
            [0.10, 0.70, 0.10, 0.10],
            [0.15, 0.60, 0.15, 0.10],
            [0.20, 0.55, 0.15, 0.10],
        ],
    )
    calibration_max_hard_regression: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_calibration_grid(self) -> "SchedulerConfig":
        if len(self.gate_node_allowlist) != len(set(self.gate_node_allowlist)):
            raise ValueError("gate_node_allowlist must not contain duplicates")
        if any(not item.strip() for item in self.gate_node_allowlist):
            raise ValueError("gate_node_allowlist entries must be non-empty")
        if any(value < 0.0 or value > 1.0 for value in self.calibration_thresholds):
            raise ValueError("scheduler calibration thresholds must be in [0, 1]")
        if not self.calibration_thresholds:
            raise ValueError("scheduler calibration thresholds cannot be empty")
        for weights in self.calibration_weight_candidates:
            if len(weights) != 4 or any(value < 0.0 for value in weights):
                raise ValueError(
                    "each scheduler calibration weight candidate must contain "
                    "four non-negative values"
                )
            if sum(weights) <= 0.0:
                raise ValueError("scheduler calibration weights must have positive sum")
        return self


class OptimizerConfig(_StrictConfigModel):
    """Configuration for the workflow optimizer."""

    # Research default: optimize the workflow's inner content only.  The
    # candidate scopes are prompt, operator, and graph path (the latter uses
    # the historical internal ``block`` scope name).  Selective gates and
    # efficiency-triggered outer updates remain dormant, while suffix replay
    # is retained as the correctness check for failure repairs.
    workflow_content_only: bool = True
    max_rounds: int = Field(default=10, gt=0)
    failure_buffer_capacity: int = Field(default=100, gt=0)
    # Counterfactual batches are failure-heavy. ``success_guard_fraction`` is
    # an upper target for s / (f + s); small failure batches may therefore
    # select zero guards. Set ``min_success_guards`` explicitly when at least
    # one regression guard is required even if that exceeds the fraction.
    success_guard_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    min_success_guards: int = Field(default=0, ge=0)
    # Only fully correct traces are success/efficiency anchors by default.
    # Benchmarks with a different hard-reward contract may opt in explicitly.
    hard_success_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    # Optional dual trigger: in addition to hard failures, optimize the most
    # expensive hard-success traces. It is disabled by default and ignored in
    # workflow-content-only mode so the active trigger remains failure-only.
    efficiency_optimization_enabled: bool = False
    efficiency_anchor_fraction: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
    )
    efficiency_min_relative_cost: float = Field(default=1.25, ge=1.0)
    efficiency_max_anchors: int = Field(default=5, gt=0)
    candidates_per_round: int = Field(default=5, gt=0)
    # A zero-sized archive and zero exploration budget preserve the historical
    # greedy, minimal-effective search. When enabled, the archive retains a
    # hard-safe Pareto frontier plus top scalar-gain candidates. Exploration
    # evaluations continue into broader scopes after a sufficient local edit.
    candidate_archive_size: int = Field(default=0, ge=0)
    exploration_budget: int = Field(default=0, ge=0)
    # Selective Counterfactual Workflow Update (S-CWU). Disabled by default
    # and ignored in workflow-content-only mode.
    # Gate inputs are a fixed query-only allowlist computed before execution;
    # labels, rewards, traces, and candidate outputs are never gate features.
    selective_update_enabled: bool = False
    gate_features: list[
        Literal[
            "query_chars",
            "query_words",
            "query_lines",
            "numeric_literals",
        ]
    ] = Field(
        default_factory=lambda: [
            "query_chars",
            "query_words",
            "query_lines",
            "numeric_literals",
        ]
    )
    gate_min_leaf_support: int = Field(default=2, gt=0)
    gate_fit_max_traces: int = Field(default=32, gt=0)
    gate_hard_regression_tolerance: float = Field(
        default=0.0,
        ge=0.0,
    )
    gate_simplicity_penalty: float = Field(default=1e-3, ge=0.0)
    # Execution models that candidate edits may select explicitly. An empty
    # list preserves the unrestricted behavior for backwards compatibility.
    allowed_execution_models: list[str] = Field(default_factory=list)
    # Edit budget parameters. ``None`` removes the hard edit-distance filter;
    # the soft ``mu_edit`` term remains available for a secondary preference.
    max_edit_distance: Optional[float] = Field(default=0.5, ge=0.0)
    # Candidate-gain coefficients. Utility itself is hard reward only;
    # lambda_cost penalizes candidate-minus-incumbent token deltas in gain.
    lambda_cost: float = Field(default=1e-4, ge=0.0)
    # These coefficients belong to the separate high-cost-success efficiency
    # signal and are retained for that ablation; they do not alter utility or
    # candidate gain.
    lambda_latency: float = Field(default=0.0, ge=0.0)
    lambda_api_cost: float = Field(default=0.0, ge=0.0)
    rho_omega: float = Field(default=1e-3, ge=0.0)
    # Edit distance is a tie-break preference after a failure repair has
    # cleared the suffix-replay correctness gate, so keep its scalar penalty
    # deliberately small.
    mu_edit: float = Field(default=0.02, ge=0.0)
    epsilon_stat: float = Field(default=0.01, ge=0.0)
    # Optimization LLM (can differ from scheduler LLM)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Suffix replay settings
    use_suffix_replay: bool = False
    max_suffix_replay_cache: int = Field(default=1000, gt=0)
    # When unset, inner-only failure repairs require suffix-replay evidence by
    # default, while legacy outer-layer runs retain their historical fallback.
    require_failure_suffix_replay: Optional[bool] = None
    # Failure-update scheduling. ``deferred_sequential`` executes the whole
    # optimization epoch first, then replays buffered failures and applies
    # one counterfactual update at a time. ``online`` is retained for the
    # streaming ablation and ``batch`` preserves the historical behavior.
    failure_update_mode: Literal[
        "batch",
        "deferred_sequential",
        "online",
    ] | None = None
    # Backwards-compatible alias for the old streaming implementation.
    online_failure_updates: bool = False
    # Research aggressive-update controls. The conservative defaults preserve
    # the historical single-failure behavior; the aggressive experiment
    # enables clustering, multi-level patches, local promotion, and rollback.
    failure_cluster_enabled: bool = False
    failure_cluster_max_size: int = Field(default=8, gt=0)
    failure_cluster_min_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    # A hard failure is the only event that triggers repeated generation.
    # ``failure_confirmation_repeats`` counts additional executions after
    # the first failed epoch observation. With repeats=2 and
    # min_failures=2, a case must fail at least two of three total runs before
    # it can drive an update. Successful cases incur no repeat cost.
    failure_confirmation_repeats: int = Field(default=1, ge=1)
    failure_confirmation_min_failures: int = Field(default=2, ge=1)
    failure_repair_confirmation_enabled: bool = False
    failure_repair_confirmation_repeats: int = Field(default=2, ge=1)
    failure_repair_confirmation_min_successes: int = Field(default=2, ge=1)
    # ``any_repair`` is the aggressive cluster-local research predicate: the
    # patch is admissible when at least one member is repaired and sampled
    # successes do not regress.  It deliberately accepts full reruns caused by
    # inserted prefix nodes; suffix replay remains an efficiency mechanism,
    # not an acceptance constraint.
    failure_repair_acceptance: Literal["coverage", "any_repair"] = "coverage"
    allow_failure_full_rerun: bool = False
    aggressive_candidate_generation: bool = False
    allow_cross_block_graph_updates: bool = False
    deterministic_block_candidates: list[
        Literal[
            "self_refine",
            "dual_solve_judge",
            "verify_repair",
            "conditional_debate",
            "format_repair",
        ]
    ] = Field(default_factory=list)
    multi_level_updates: bool = False
    exploration_acceptance_floor: float = 0.0
    rollback_window: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _validate_efficiency_search(self) -> "OptimizerConfig":
        if (
            not self.workflow_content_only
            and self.efficiency_optimization_enabled
            and self.lambda_cost == 0.0
            and self.lambda_latency == 0.0
            and self.lambda_api_cost == 0.0
            and self.rho_omega == 0.0
        ):
            raise ValueError(
                "efficiency optimization requires at least one non-zero "
                "runtime-cost coefficient"
            )
        if (
            not self.workflow_content_only
            and self.efficiency_optimization_enabled
            and self.min_success_guards < 1
        ):
            raise ValueError(
                "efficiency optimization requires min_success_guards >= 1 "
                "to protect ordinary successful paths"
            )
        if self.exploration_budget > 0 and self.candidate_archive_size == 0:
            raise ValueError(
                "exploration_budget requires candidate_archive_size > 0"
            )
        if (
            self.failure_confirmation_min_failures
            > self.failure_confirmation_repeats + 1
        ):
            raise ValueError(
                "failure_confirmation_min_failures cannot exceed the first "
                "failed observation plus failure_confirmation_repeats"
            )
        if (
            self.failure_repair_confirmation_min_successes
            > self.failure_repair_confirmation_repeats + 1
        ):
            raise ValueError(
                "failure_repair_confirmation_min_successes cannot exceed "
                "the first candidate run plus confirmation repeats"
            )
        if len(self.gate_features) != len(set(self.gate_features)):
            raise ValueError("gate_features must not contain duplicates")
        if not self.workflow_content_only and self.selective_update_enabled:
            if not self.gate_features:
                raise ValueError(
                    "selective update requires at least one gate feature"
                )
            if self.gate_fit_max_traces < 2 * self.gate_min_leaf_support:
                raise ValueError(
                    "gate_fit_max_traces must support both stump leaves"
                )
        return self


class RewardConfig(_StrictConfigModel):
    """Composite reward configuration."""

    alpha_process: float = Field(default=0.8, ge=0.0)


class ExecutorConfig(_StrictConfigModel):
    """Configuration for the runtime executor."""

    max_steps: int = Field(default=100, gt=0)
    timeout_per_step: float = Field(default=300.0, gt=0.0)
    # Whether ExperimentRunner persists detailed traces to disk. Runtime
    # always keeps an in-memory trace because rewards and utility depend on it.
    trace_enabled: bool = True


class ExperimentConfig(_StrictConfigModel):
    """Top-level configuration for an experiment."""

    # Random seed for reproducibility
    seed: int = 42
    # Data split ratios
    opt_split_ratio: float = Field(default=0.6, gt=0.0, lt=1.0)
    val_split_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_split_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    # Optional source-split-driven partitioning. When set, rows are assigned
    # to splits by a ground-truth metadata field instead of random ratios:
    # rows whose field equals ``split_test_value`` form the held-out test
    # split; all other rows form the development pool (further divided into
    # optimization/validation by the configured ratios).
    split_source_field: Optional[str] = None
    split_dev_value: str = "validate"
    split_test_value: str = "test"
    # When True and split_source_field is set, the full development pool (the
    # source ``split_dev_value`` rows) is used for BOTH optimization and
    # validation, matching the common single-split practice in AFlow-style
    # papers. opt and val then refer to the same rows; only test is held out.
    split_validate_reuse: bool = False
    # Component configs
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    # Optional backend for workflow-node execution.  Historically workflow
    # nodes reused ``scheduler.llm``; keeping this nullable preserves that
    # behavior while allowing the outer cascade to use a separate DeepSeek
    # scheduler over a Qwen/base workflow.
    workflow_llm: Optional[LLMConfig] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    # Output directory
    output_dir: str = Field(default="experiments/results", min_length=1)
    # Experiment name
    name: str = Field(default="default", min_length=1)
    # Stability gates between the optimizer's local counterfactual decision
    # and promotion of a workflow to the validation-selected incumbent.
    validation_min_delta: float = Field(default=0.0, ge=0.0)
    validation_hard_regression_tolerance: float = Field(
        default=0.0,
        ge=0.0,
    )
    # When True, hard success (accuracy) is the primary promotion criterion
    # during workflow updates: a candidate must improve hard_reward by
    # ``validation_min_delta``, while runtime utility only needs to non-regress
    # beyond ``validation_hard_regression_tolerance``. When False (default),
    # runtime utility drives promotion and hard_reward only needs to
    # non-regress. The same primary metric drives checkpoint "best" selection.
    hard_success_priority: bool = False
    # ``validation`` is the original offline promotion protocol. The
    # counterfactual-local protocol promotes suffix-repaired updates
    # immediately and treats validation as an audit-only split.
    promotion_mode: Literal[
        "validation",
        "counterfactual_local_aggressive",
    ] = "validation"
    confirm_on_opt: bool = False
    confirm_repeats: int = Field(default=1, ge=0)
    # Validation-based early stopping. Set patience to null to disable.
    # ``early_stopping_min_delta`` is retained for existing configurations;
    # ExperimentRunner folds it into the single checkpoint/gate threshold.
    early_stopping_patience: Optional[int] = Field(default=3, ge=1)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    # Bilevel research schedule.  With scheduler_calibration_round=3, rounds
    # one and two optimize workflow content using a frozen graph scheduler;
    # round three freezes that workflow and calibrates the outer LAS gate on
    # the complete validation split.
    workflow_optimization_rounds: Optional[int] = Field(default=None, ge=1)
    scheduler_calibration_round: Optional[int] = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _validate_split_ratios(self) -> "ExperimentConfig":
        ratios = (
            self.opt_split_ratio,
            self.val_split_ratio,
            self.test_split_ratio,
        )
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")
        if self.confirm_on_opt and self.confirm_repeats < 1:
            raise ValueError(
                "confirm_repeats must be at least 1 when confirm_on_opt=true"
            )
        if (
            not self.optimizer.workflow_content_only
            and self.optimizer.selective_update_enabled
            and self.scheduler.scheduler_type not in {"graph", "cascade"}
        ):
            raise ValueError(
                "selective workflow updates require scheduler_type='graph' "
                "or 'cascade' "
                "so unselected branches are not executed"
            )
        if self.scheduler_calibration_round is not None:
            if not self.scheduler.calibration_enabled:
                raise ValueError(
                    "scheduler_calibration_round requires "
                    "scheduler.calibration_enabled=true"
                )
            if self.scheduler.scheduler_type != "cascade":
                raise ValueError(
                    "scheduler calibration requires scheduler_type='cascade'"
                )
            workflow_rounds = self.workflow_optimization_rounds
            if workflow_rounds is None:
                workflow_rounds = self.scheduler_calibration_round - 1
                self.workflow_optimization_rounds = workflow_rounds
            if workflow_rounds >= self.scheduler_calibration_round:
                raise ValueError(
                    "workflow optimization rounds must end before scheduler "
                    "calibration"
                )
            if self.optimizer.max_rounds < self.scheduler_calibration_round:
                raise ValueError(
                    "optimizer.max_rounds must include the scheduler "
                    "calibration round"
                )
        return self
