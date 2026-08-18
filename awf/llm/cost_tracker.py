"""TokenUsageTracker — per-call and aggregated cost tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    # End-to-end latency of the successful logical request, including retries.
    latency_seconds: float = 0.0
    # Cost in USD (computed from pricing)
    cost_usd: float = 0.0
    cost_estimate_available: bool = True


# Approximate pricing per 1M tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    "qwen3-235b-a22b": (0.50, 2.00),
    "qwen-max": (2.00, 8.00),
    "qwen-plus": (0.80, 3.20),
    "qwen3.5-flash-2026-02-23": (0.10, 0.30),
}

# Cache-hit input prices per 1M tokens. Models absent from this table use their
# ordinary input price for all prompt tokens.
_MODEL_CACHE_HIT_PRICING: dict[str, float] = {
    "deepseek-v4-flash": 0.0028,
    "deepseek-v4-pro": 0.003625,
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: Optional[int] = None,
    prompt_cache_miss_tokens: Optional[int] = None,
) -> float:
    """Estimate an API call's price from the local pricing table."""
    pricing_model = _resolve_pricing_model(model)
    if pricing_model is None:
        return 0.0
    input_price, output_price = _MODEL_PRICING[pricing_model]
    cache_hit_price = _MODEL_CACHE_HIT_PRICING.get(
        pricing_model,
        input_price,
    )
    cache_hit, cache_miss = _normalize_cache_tokens(
        prompt_tokens,
        prompt_cache_hit_tokens,
        prompt_cache_miss_tokens,
    )
    return (
        cache_hit / 1_000_000 * cache_hit_price
        + cache_miss / 1_000_000 * input_price
        + completion_tokens / 1_000_000 * output_price
    )


def has_known_pricing(model: str) -> bool:
    """Return whether local estimated pricing covers this model name."""
    return _resolve_pricing_model(model) is not None


def _resolve_pricing_model(model: str) -> Optional[str]:
    normalized = (model or "").strip().lower()
    if normalized.startswith("ft:"):
        normalized = normalized[3:]
    for known in sorted(_MODEL_PRICING, key=len, reverse=True):
        if normalized == known or normalized.startswith(f"{known}-"):
            return known
    return None


def _normalize_cache_tokens(
    prompt_tokens: int,
    cache_hit_tokens: Optional[int],
    cache_miss_tokens: Optional[int],
) -> tuple[int, int]:
    """Return a complete, bounded cache hit/miss prompt breakdown."""
    prompt = max(int(prompt_tokens), 0)
    if cache_hit_tokens is None and cache_miss_tokens is None:
        return 0, prompt

    if cache_hit_tokens is None:
        cache_miss = min(max(int(cache_miss_tokens or 0), 0), prompt)
        return prompt - cache_miss, cache_miss
    if cache_miss_tokens is None:
        cache_hit = min(max(int(cache_hit_tokens), 0), prompt)
        return cache_hit, prompt - cache_hit

    # Prefer the more conservative cache-miss category if a malformed provider
    # response reports more categorized tokens than the total prompt.
    cache_miss = min(max(int(cache_miss_tokens), 0), prompt)
    cache_hit = min(
        max(int(cache_hit_tokens), 0),
        prompt - cache_miss,
    )
    cache_miss += prompt - cache_hit - cache_miss
    return cache_hit, cache_miss


class TokenUsageTracker:
    """Tracks token usage and cost across multiple LLM calls.

    Supports both per-call recording and aggregated statistics.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._calls: list[TokenUsage] = []
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    @property
    def num_calls(self) -> int:
        return len(self._calls)

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
        prompt_cache_hit_tokens: Optional[int] = None,
        prompt_cache_miss_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        latency_seconds: float = 0.0,
        cost_estimate_available: Optional[bool] = None,
    ) -> TokenUsage:
        """Record a single LLM call's token usage.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.
            model: Override model for pricing (uses tracker's default if None).
            prompt_cache_hit_tokens: Prompt tokens served from provider cache.
            prompt_cache_miss_tokens: Prompt tokens not served from cache.
            reasoning_tokens: Completion tokens used for hidden reasoning.
            latency_seconds: Successful logical-request latency, including
                any retry delay.
            cost_estimate_available: Optional explicit availability override.
                This is false for successful responses that omit token usage.

        Returns:
            The TokenUsage record.
        """
        model_name = model or self.model
        cache_hit, cache_miss = _normalize_cache_tokens(
            prompt_tokens,
            prompt_cache_hit_tokens,
            prompt_cache_miss_tokens,
        )
        cost = estimate_cost_usd(
            model_name,
            prompt_tokens,
            completion_tokens,
            prompt_cache_hit_tokens=cache_hit,
            prompt_cache_miss_tokens=cache_miss,
        )

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cache_hit_tokens=cache_hit,
            prompt_cache_miss_tokens=cache_miss,
            reasoning_tokens=max(int(reasoning_tokens or 0), 0),
            latency_seconds=max(float(latency_seconds), 0.0),
            cost_usd=cost,
            cost_estimate_available=(
                has_known_pricing(model_name)
                if cost_estimate_available is None
                else bool(cost_estimate_available)
            ),
        )
        self._calls.append(usage)
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        return usage

    def reset(self) -> None:
        """Reset all accumulated usage."""
        self._calls.clear()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def snapshot(self) -> dict:
        """Return a summary of accumulated usage."""
        return {
            "model": self.model,
            "num_calls": self.num_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": sum(
                call.prompt_cache_hit_tokens for call in self._calls
            ),
            "prompt_cache_miss_tokens": sum(
                call.prompt_cache_miss_tokens for call in self._calls
            ),
            "reasoning_tokens": sum(
                call.reasoning_tokens for call in self._calls
            ),
            "total_latency_seconds": sum(
                call.latency_seconds for call in self._calls
            ),
            "total_cost_usd": self.total_cost_usd,
            "cost_estimate_complete": all(
                call.cost_estimate_available for call in self._calls
            ),
            "unknown_pricing_calls": sum(
                not call.cost_estimate_available for call in self._calls
            ),
        }
