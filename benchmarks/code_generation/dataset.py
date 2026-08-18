"""HumanEval, MBPP data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class HumanEvalDataset:
    """Loader for the HumanEval code generation dataset.

    Expects a JSONL file with fields: task_id, prompt, canonical_solution, test.
    """

    def __init__(self, data_path: Optional[str | Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self._problems: list[dict[str, Any]] = []

    def load(self, path: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Load HumanEval problems from a JSONL file.

        Args:
            path: Path to the JSONL file. Uses data_path if not provided.

        Returns:
            List of problem dicts.
        """
        filepath = Path(path) if path else self.data_path
        if filepath is None:
            raise ValueError("No data path provided")

        problems = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        self._problems = problems
        return problems

    def to_pairs(self) -> list[tuple[str, Any]]:
        """Convert problems to (query, ground_truth) pairs.

        Query = prompt, Ground truth = dict with canonical_solution and test.
        """
        return [
            (p["prompt"], {
                "canonical_solution": p.get("canonical_solution", ""),
                "test": p.get("test", ""),
                "entry_point": p.get("entry_point", ""),
                "task_id": p.get("task_id", ""),
            })
            for p in self._problems
        ]


class MBPPDataset:
    """Loader for the MBPP (Mostly Basic Python Problems) dataset.

    Expects a JSONL file with fields: task_id, text, code, test_list.
    """

    def __init__(self, data_path: Optional[str | Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self._problems: list[dict[str, Any]] = []

    def load(self, path: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Load MBPP problems from a JSONL file."""
        filepath = Path(path) if path else self.data_path
        if filepath is None:
            raise ValueError("No data path provided")

        problems = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        self._problems = problems
        return problems

    def to_pairs(self) -> list[tuple[str, Any]]:
        """Convert problems to (query, ground_truth) pairs."""
        return [
            (p["text"], {
                "code": p.get("code", ""),
                "test_list": p.get("test_list", []),
                "task_id": p.get("task_id", ""),
            })
            for p in self._problems
        ]


def load_humaneval(path: str | Path) -> list[tuple[str, Any]]:
    """Convenience function to load HumanEval as (query, gt) pairs."""
    dataset = HumanEvalDataset(path)
    dataset.load()
    return dataset.to_pairs()


def load_mbpp(path: str | Path) -> list[tuple[str, Any]]:
    """Convenience function to load MBPP as (query, gt) pairs."""
    dataset = MBPPDataset(path)
    dataset.load()
    return dataset.to_pairs()
