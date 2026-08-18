"""Tests for config loading and schema validation."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from awf.config.loader import load_config
from awf.config.schema import (
    ExperimentConfig,
    LLMConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from pydantic import ValidationError


class TestConfigSchema:
    """Test Pydantic config model validation."""

    def test_default_config(self):
        config = ExperimentConfig()
        assert config.seed == 42
        assert config.scheduler.scheduler_type == "fixed"
        assert config.optimizer.max_rounds == 10

    def test_llm_config_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_retries == 3

    def test_scheduler_config(self):
        config = SchedulerConfig()
        assert config.scheduler_type == "fixed"
        assert config.max_actions_per_query == 50
        assert config.gate_node_allowlist == []

        with pytest.raises(ValidationError, match="gate_node_allowlist"):
            SchedulerConfig(gate_node_allowlist=["verify", "verify"])

    def test_optimizer_execution_model_allowlist_defaults_to_unrestricted(self):
        assert OptimizerConfig().allowed_execution_models == []
        assert OptimizerConfig().mu_edit == 0.02
        assert OptimizerConfig().require_failure_suffix_replay is None
        assert OptimizerConfig(
            allowed_execution_models=["deepseek-v4-flash"]
        ).allowed_execution_models == ["deepseek-v4-flash"]

    def test_optimizer_runtime_cost_coefficients_are_explicit_and_nonnegative(self):
        defaults = OptimizerConfig()
        assert defaults.lambda_latency == 0.0
        assert defaults.lambda_api_cost == 0.0
        configured = OptimizerConfig(
            lambda_latency=0.25,
            lambda_api_cost=4.0,
        )
        assert configured.lambda_latency == 0.25
        assert configured.lambda_api_cost == 4.0

        with pytest.raises(ValidationError):
            OptimizerConfig(lambda_latency=-0.1)
        with pytest.raises(ValidationError):
            OptimizerConfig(lambda_api_cost=-0.1)

    def test_optimizer_success_guard_controls_are_explicit(self):
        defaults = OptimizerConfig()
        assert defaults.success_guard_fraction == 0.2
        assert defaults.min_success_guards == 0
        configured = OptimizerConfig(
            success_guard_fraction=0.3,
            min_success_guards=2,
        )
        assert configured.success_guard_fraction == 0.3
        assert configured.min_success_guards == 2

        with pytest.raises(ValidationError):
            OptimizerConfig(success_guard_fraction=1.0)
        with pytest.raises(ValidationError):
            OptimizerConfig(min_success_guards=-1)

    def test_failure_confirmation_controls_require_feasible_majorities(self):
        defaults = OptimizerConfig()
        assert defaults.failure_confirmation_repeats == 1
        assert defaults.failure_confirmation_min_failures == 2
        assert defaults.failure_repair_confirmation_enabled is False
        assert defaults.failure_repair_confirmation_repeats == 2
        assert defaults.failure_repair_confirmation_min_successes == 2

        configured = OptimizerConfig(
            failure_confirmation_repeats=2,
            failure_confirmation_min_failures=2,
            failure_repair_confirmation_enabled=True,
            failure_repair_confirmation_repeats=4,
            failure_repair_confirmation_min_successes=3,
        )
        assert configured.failure_confirmation_repeats == 2
        assert configured.failure_repair_confirmation_min_successes == 3

        with pytest.raises(ValidationError, match="min_failures"):
            OptimizerConfig(
                failure_confirmation_repeats=1,
                failure_confirmation_min_failures=3,
            )
        with pytest.raises(ValidationError, match="min_successes"):
            OptimizerConfig(
                failure_repair_confirmation_repeats=1,
                failure_repair_confirmation_min_successes=3,
            )

    def test_nested_config_construction(self):
        config = ExperimentConfig(
            name="test",
            seed=123,
            optimizer={"max_rounds": 5},
        )
        assert config.name == "test"
        assert config.seed == 123
        assert config.optimizer.max_rounds == 5

    def test_unknown_config_fields_fail_closed(self):
        with pytest.raises(ValidationError, match="schedulre"):
            ExperimentConfig(schedulre={"scheduler_type": "fixed"})
        with pytest.raises(ValidationError, match="max_retry"):
            LLMConfig(max_retry=99)

    def test_extra_request_kwargs_cannot_override_core_fields(self):
        with pytest.raises(ValidationError, match="core request fields"):
            LLMConfig(extra_kwargs={"model": "hidden-override"})


class TestConfigLoader:
    """Test YAML config loading."""

    def test_load_basic_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"name": "test_exp", "seed": 99}, f)
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config.name == "test_exp"
            assert config.seed == 99
        finally:
            os.unlink(tmp_path)

    def test_load_full_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "name": "full_test",
                    "seed": 42,
                    "scheduler": {
                        "scheduler_type": "fixed",
                        "max_actions_per_query": 20,
                    },
                    "optimizer": {
                        "max_rounds": 3,
                            "lambda_cost": 0.001,
                    },
                },
                f,
            )
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config.name == "full_test"
            assert config.scheduler.scheduler_type == "fixed"
            assert config.scheduler.max_actions_per_query == 20
            assert config.optimizer.max_rounds == 3
        finally:
            os.unlink(tmp_path)

    def test_env_var_interpolation(self):
        os.environ["TEST_MODEL"] = "gpt-4o-mini"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "name": "env_test",
                    "scheduler": {
                        "llm": {"model": "${TEST_MODEL}"},
                    },
                },
                f,
            )
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config.scheduler.llm.model == "gpt-4o-mini"
        finally:
            os.unlink(tmp_path)

    def test_missing_env_var_is_an_explicit_error(self):
        os.environ.pop("DEFINITELY_MISSING_AWF_VAR", None)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "scheduler": {
                        "llm": {
                            "api_key": "${DEFINITELY_MISSING_AWF_VAR}",
                        }
                    }
                },
                f,
            )
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="DEFINITELY_MISSING_AWF_VAR"):
                load_config(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_env_var_explicit_default(self):
        os.environ.pop("MISSING_WITH_DEFAULT", None)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "scheduler": {
                        "llm": {"api_key": "${MISSING_WITH_DEFAULT:-test}"},
                    }
                },
                f,
            )
            tmp_path = f.name

        try:
            assert load_config(tmp_path).scheduler.llm.api_key == "test"
        finally:
            os.unlink(tmp_path)
