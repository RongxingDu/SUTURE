"""YAML config loading with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from awf.config.schema import ExperimentConfig

_ENV_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate ${ENV_VAR} patterns in strings."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default
            raise ValueError(
                f"Undefined environment variable in configuration: {var_name}"
            )

        interpolated = _ENV_VAR_PATTERN.sub(_replace, value)
        if "${" in interpolated:
            raise ValueError(
                f"Malformed environment interpolation: {value!r}"
            )
        return interpolated
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without dropping nested sibling values."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(
    path: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load YAML and resolve ``extends`` before schema construction."""
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(p) for p in (*stack, resolved))
        raise ValueError(f"Circular config inheritance: {chain}")

    with open(resolved, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")

    raw = dict(raw)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    extends_path = Path(extends)
    if not extends_path.is_absolute():
        extends_path = resolved.parent / extends_path
    base = _load_raw_config(extends_path, (*stack, resolved))
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Supports recursive ``extends`` merging and ${ENV_VAR} interpolation.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated ExperimentConfig instance.
    """
    raw = _interpolate_env_vars(_load_raw_config(Path(path)))
    return ExperimentConfig(**raw)
