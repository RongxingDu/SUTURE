"""Agent task data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class AgentTaskDataset:
    """Loader for agent task datasets.

    Generic loader for task-oriented agent benchmarks.
    Expects a JSONL file with fields: task_id, instruction, expected_outcome, metadata.
    """

    def __init__(self, data_path: Optional[str | Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self._tasks: list[dict[str, Any]] = []

    def load(self, path: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Load agent tasks from a JSONL file."""
        filepath = Path(path) if path else self.data_path
        if filepath is None:
            raise ValueError("No data path provided")

        tasks = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        self._tasks = tasks
        return tasks

    def to_pairs(self) -> list[tuple[str, Any]]:
        """Convert tasks to (query, ground_truth) pairs.

        Query = instruction, Ground truth = dict with expected outcome and metadata.
        """
        return [
            (t.get("instruction", ""), {
                "expected_outcome": t.get("expected_outcome", {}),
                "success_criteria": t.get("success_criteria", []),
                "task_id": t.get("task_id", ""),
                "metadata": t.get("metadata", {}),
            })
            for t in self._tasks
        ]


def load_agent_tasks(path: str | Path) -> list[tuple[str, Any]]:
    """Convenience function to load agent tasks as (query, gt) pairs."""
    dataset = AgentTaskDataset(path)
    dataset.load()
    return dataset.to_pairs()
