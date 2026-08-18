"""Candidate scoring with reward, token, and edit-distance terms."""

from __future__ import annotations

from awf.optimizer.candidate_generator import WorkflowCandidate


class CandidateScorer:
    """Score candidates using the external gain objective.

    Utility is deliberately reward-only.  Candidate comparison adds the
    resource and edit terms around the reward delta:

    ``G = delta_u - lambda_tokens * delta_tokens - mu_edit * D_edit``

    ``delta_tokens`` is candidate minus incumbent total-token usage on the
    same counterfactual batch.  A negative delta therefore becomes a positive
    gain contribution (token saving), while an increase is penalized.
    """

    def __init__(
        self,
        mu_edit: float = 0.02,
        lambda_tokens: float = 1e-4,
        *,
        lambda_cost: float | None = None,
    ):
        if lambda_cost is not None:
            lambda_tokens = float(lambda_cost)
        if mu_edit < 0.0:
            raise ValueError("mu_edit cannot be negative")
        if lambda_tokens < 0.0:
            raise ValueError("lambda_tokens cannot be negative")
        self.mu_edit = mu_edit
        self.lambda_tokens = lambda_tokens
        # Configurations historically call this coefficient ``lambda_cost``.
        # Keep the alias while using the more explicit scorer-level name.
        self.lambda_cost = lambda_tokens

    def score(
        self,
        candidate: WorkflowCandidate,
        delta_u: float,
        token_delta: float = 0.0,
    ) -> float:
        """Return ``delta_u`` minus token and edit penalties.

        ``token_delta`` defaults to zero for compatibility with callers that
        score synthetic candidates without execution traces.
        """
        return (
            float(delta_u)
            - self.lambda_tokens * float(token_delta)
            - self.mu_edit * candidate.edit_distance
        )

    def score_batch(
        self,
        candidates: list[WorkflowCandidate],
        delta_us: list[float],
        token_deltas: list[float] | None = None,
    ) -> list[tuple[WorkflowCandidate, float]]:
        """Score multiple candidates and sort them by descending gain."""
        deltas = token_deltas or [0.0] * len(candidates)
        scored = []
        for candidate, delta_u, token_delta in zip(
            candidates,
            delta_us,
            deltas,
        ):
            gain = self.score(candidate, delta_u, token_delta)
            scored.append((candidate, gain))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
