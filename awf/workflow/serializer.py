"""YAML/JSON dump and load for WorkflowTemplate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from awf.workflow.ir import WorkflowTemplate


def dump_workflow(workflow: WorkflowTemplate, path: Union[str, Path]) -> None:
    """Serialize a WorkflowTemplate to a YAML or JSON file.

    Format is determined by file extension (.yaml/.yml vs .json).
    """
    path = Path(path)
    data = workflow.model_dump(mode="json")

    if path.suffix in (".yaml", ".yml"):
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    elif path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_workflow(path: Union[str, Path]) -> WorkflowTemplate:
    """Deserialize a WorkflowTemplate from a YAML or JSON file."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return WorkflowTemplate(**data)
