"""BaseBenchmark ABC — generic benchmark interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from awf.reward.base import RewardEvaluator


class BaseBenchmark(ABC):
    """Abstract base for benchmark implementations.

    Each benchmark provides:
    1. A dataset loader — returns list of (query, ground_truth) pairs
    2. A reward evaluator — implements awf.reward.base.RewardEvaluator
    3. An evaluator — runs test suites, extracts answers, checks correctness
    """

    @abstractmethod
    def load_dataset(self, split: str = "train") -> list[tuple[str, Any]]:
        """Load dataset items as (query, ground_truth) pairs.

        Args:
            split: Dataset split name (e.g., "train", "test", "validation").

        Returns:
            List of (query, ground_truth) tuples.
        """
        ...

    @abstractmethod
    def get_reward_evaluator(self) -> RewardEvaluator:
        """Get the task-specific reward evaluator.

        Returns:
            A RewardEvaluator instance for this benchmark.
        """
        ...

    @abstractmethod
    def evaluate(self, output: Any, ground_truth: Any) -> bool:
        """Check if an output is correct given the ground truth.

        Args:
            output: The model/operator output.
            ground_truth: The expected answer.

        Returns:
            True if the output matches the ground truth.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        ...
