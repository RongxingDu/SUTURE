"""AcceptanceCriterion — external candidate gain threshold."""

from __future__ import annotations


class AcceptanceCriterion:
    """Check whether an externally-computed candidate gain is sufficient.

    The gain supplied here already contains token and edit-distance penalties;
    :class:`awf.utility.compute.UtilityComputer` remains reward-only.
    """

    def __init__(self, epsilon_stat: float = 0.01):
        """Initialize the minimum gain threshold."""
        self.epsilon_stat = epsilon_stat

    def accept(self, gain: float) -> bool:
        """Return whether ``gain`` strictly exceeds the threshold."""
        return gain > self.epsilon_stat

    def select_best(
        self,
        scored_candidates: list[tuple[object, float]],
    ) -> tuple[object, float] | None:
        """Return the highest-scoring candidate above the threshold."""
        for candidate, gain in scored_candidates:
            if self.accept(gain):
                return candidate, gain
        return None
