"""GSM8K, MATH data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from benchmarks.math_reasoning.evaluator import MathEvaluator


class GSM8KDataset:
    """Loader for the GSM8K grade-school math dataset.

    Expects a JSONL file with fields: question, answer.
    """

    def __init__(self, data_path: Optional[str | Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self._problems: list[dict[str, Any]] = []

    def load(self, path: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Load GSM8K problems from a JSONL file."""
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
            (
                p["question"],
                MathEvaluator.normalize_ground_truth(p.get("answer", "")),
            )
            for p in self._problems
        ]


class MATHDataset:
    """Loader for the MATH competition dataset.

    Expects a JSONL file with fields: problem, solution, type, level.
    """

    def __init__(self, data_path: Optional[str | Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self._problems: list[dict[str, Any]] = []

    def load(self, path: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Load MATH problems from a JSONL file."""
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
        """Convert problems to pairs with a compact, analysis-ready target."""
        pairs: list[tuple[str, Any]] = []
        for problem in self._problems:
            ground_truth: dict[str, Any] = {
                "answer": MathEvaluator.normalize_ground_truth(
                    problem.get("solution", ""),
                ),
            }
            metadata_fields = {
                "domain": problem.get("domain", problem.get("type")),
                "level": problem.get("level"),
                "source_split": problem.get(
                    "source_split",
                    problem.get("_aflow_source_split"),
                ),
                "source_index": problem.get(
                    "source_index",
                    problem.get("_aflow_source_index"),
                ),
            }
            ground_truth.update(
                {
                    key: value
                    for key, value in metadata_fields.items()
                    if value is not None
                }
            )
            pairs.append((problem["problem"], ground_truth))
        return pairs


def load_gsm8k(path: str | Path) -> list[tuple[str, Any]]:
    """Convenience function to load GSM8K as (query, gt) pairs."""
    dataset = GSM8KDataset(path)
    dataset.load()
    return dataset.to_pairs()


def load_math(path: str | Path) -> list[tuple[str, Any]]:
    """Convenience function to load MATH as (query, gt) pairs."""
    dataset = MATHDataset(path)
    dataset.load()
    return dataset.to_pairs()
