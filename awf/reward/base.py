"""RewardEvaluator ABC — clean interface for task-specific rewards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from awf.trace.schema import ExecutionTrace


class RewardEvaluator(ABC):
    """Abstract base class for task-specific reward evaluation.

    Subclasses implement:
    - hard_reward: The objective task success metric (e.g., test pass rate).
    - process_reward: Intermediate signal for process quality.
    """

    @abstractmethod
    def hard_reward(self, query: str, ground_truth: Any,
                    output: Any, trace: ExecutionTrace) -> float:
        """Compute the hard (objective) reward for a task execution.

        Args:
            query: The input query text.
            ground_truth: The expected answer/output.
            output: The actual output produced by the workflow.
            trace: The full execution trace.

        Returns:
            A float reward value (typically 0.0 to 1.0).
        """
        ...

    @abstractmethod
    def process_reward(self, query: str, ground_truth: Any,
                       output: Any, trace: ExecutionTrace) -> float:
        """Compute a process-based reward from the execution trace.

        Args:
            query: The input query text.
            ground_truth: The expected answer/output.
            output: The actual output produced by the workflow.
            trace: The full execution trace.

        Returns:
            A float reward value (typically 0.0 to 1.0).
        """
        ...

    def combined_reward(self, query: str, ground_truth: Any,
                        output: Any, trace: ExecutionTrace,
                        alpha: float = 0.8) -> float:
        """Compute combined reward: hard + alpha * process.

        Args:
            query: The input query text.
            ground_truth: The expected answer/output.
            output: The actual output produced by the workflow.
            trace: The full execution trace.
            alpha: Contribution of the dense process reward (default 0.8).

        Returns:
            Weighted combined reward.
        """
        r_hard = self.hard_reward(query, ground_truth, output, trace)
        r_process = self.process_reward(query, ground_truth, output, trace)
        return r_hard + alpha * r_process
