"""AsyncLLMClient — unified async API for scheduler + optimizer LLM calls."""

from __future__ import annotations

import time
from typing import Any, Optional

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from awf.config.schema import LLMConfig, LLMProvider
from awf.llm.cost_tracker import TokenUsageTracker


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMConnectionError(LLMError):
    """Connection/timeout errors."""


class LLMRateLimitError(LLMError):
    """Rate limit errors."""


class AsyncLLMClient:
    """Unified async client for LLM API calls.

    Wraps AsyncOpenAI with retry logic, cost tracking, and a consistent
    interface used by both the scheduler and optimizer.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.tracker = TokenUsageTracker(model=config.model)

        if config.provider == LLMProvider.ANTHROPIC:
            raise NotImplementedError(
                "The Anthropic provider is not implemented by AsyncLLMClient; "
                "use an OpenAI-compatible endpoint or add an Anthropic client"
            )
        if (
            config.provider in {LLMProvider.DEEPSEEK, LLMProvider.ALIYUN, LLMProvider.CUSTOM}
            and not config.api_base
        ):
            raise ValueError(
                f"provider '{config.provider.value}' requires an explicit api_base"
            )

        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "max_retries": 0,  # We handle retries via tenacity
            "timeout": config.timeout_seconds,
        }
        if config.api_base:
            kwargs["base_url"] = config.api_base
        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._client = AsyncOpenAI(**kwargs)

    async def generate(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        messages: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a completion from the LLM.

        Args:
            system_prompt: System message (ignored if messages provided).
            user_prompt: User message (ignored if messages provided).
            messages: Full message list (overrides system_prompt/user_prompt).
            temperature: Sampling temperature (uses config default if None).
            max_tokens: Max tokens (uses config default if None).
            stop: Stop sequences.
            json_mode: Whether to request JSON output format.
            model: Per-call model override (uses config default if None).
            seed: Reproducibility seed (uses config default if None).

        Returns:
            Tuple of (response_text, usage_info_dict).
        """
        if messages is None:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
        else:
            msgs = list(messages)

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        model_name = model or self.config.model
        request_seed = seed if seed is not None else self.config.seed

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": msgs,
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if (
            request_seed is not None
            and self.config.provider != LLMProvider.DEEPSEEK
        ):
            kwargs["seed"] = request_seed
        if stop:
            kwargs["stop"] = stop
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.config.extra_kwargs)
        # DeepSeek V4 does not document/support OpenAI's seed parameter.  This
        # final removal also guards against accidentally injecting it through
        # ``extra_kwargs``.
        if self.config.provider == LLMProvider.DEEPSEEK:
            kwargs.pop("seed", None)

        call_started_at = time.perf_counter()
        response = await self._call_with_retry(**kwargs)
        latency_seconds = time.perf_counter() - call_started_at

        content = response.choices[0].message.content or ""
        usage: dict[str, Any] = {
            "latency_seconds": latency_seconds,
        }
        if response.usage:
            prompt_tokens = int(
                self._get_response_field(
                    response.usage,
                    "prompt_tokens",
                    0,
                )
            )
            completion_tokens = int(
                self._get_response_field(
                    response.usage,
                    "completion_tokens",
                    0,
                )
            )
            total_tokens = int(
                self._get_response_field(
                    response.usage,
                    "total_tokens",
                    prompt_tokens + completion_tokens,
                )
            )
            cache_hit_tokens = self._get_response_field(
                response.usage,
                "prompt_cache_hit_tokens",
            )
            cache_miss_tokens = self._get_response_field(
                response.usage,
                "prompt_cache_miss_tokens",
            )
            completion_details = self._get_response_field(
                response.usage,
                "completion_tokens_details",
            )
            reasoning_tokens = self._get_response_field(
                completion_details,
                "reasoning_tokens",
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency_seconds,
            }
            if cache_hit_tokens is not None:
                usage["prompt_cache_hit_tokens"] = int(cache_hit_tokens)
            if cache_miss_tokens is not None:
                usage["prompt_cache_miss_tokens"] = int(cache_miss_tokens)
            if reasoning_tokens is not None:
                usage["reasoning_tokens"] = int(reasoning_tokens)
            token_usage = self.tracker.record(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model_name,
                prompt_cache_hit_tokens=(
                    int(cache_hit_tokens)
                    if cache_hit_tokens is not None
                    else None
                ),
                prompt_cache_miss_tokens=(
                    int(cache_miss_tokens)
                    if cache_miss_tokens is not None
                    else None
                ),
                reasoning_tokens=(
                    int(reasoning_tokens)
                    if reasoning_tokens is not None
                    else None
                ),
                latency_seconds=latency_seconds,
            )
            usage["cost_usd"] = token_usage.cost_usd
            usage["cost_estimate_available"] = (
                token_usage.cost_estimate_available
            )
        else:
            # A successful response is still a backend call even when a
            # provider omits usage. Count its latency while explicitly
            # marking the zero-token cost as unavailable.
            token_usage = self.tracker.record(
                prompt_tokens=0,
                completion_tokens=0,
                model=model_name,
                latency_seconds=latency_seconds,
                cost_estimate_available=False,
            )
            usage["cost_usd"] = token_usage.cost_usd
            usage["cost_estimate_available"] = False

        return content, usage

    @staticmethod
    def _get_response_field(
        value: Any,
        field_name: str,
        default: Any = None,
    ) -> Any:
        """Read normal and provider-specific fields from SDK response models."""
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(field_name, default)
        result = getattr(value, field_name, None)
        if result is not None:
            return result
        model_extra = getattr(value, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(field_name, default)
        return default

    async def _call_with_retry(self, **kwargs: Any) -> Any:
        """Execute an API call with the configured retry budget."""
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(
                (LLMConnectionError, LLMRateLimitError)
            ),
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._call_once(**kwargs)
        raise LLMError("LLM retry loop ended without a result")

    async def _call_once(self, **kwargs: Any) -> Any:
        """Execute one provider request and normalize retryable errors."""
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str:
                raise LLMRateLimitError(str(e)) from e
            if "timeout" in error_str or "connection" in error_str:
                raise LLMConnectionError(str(e)) from e
            raise LLMError(str(e)) from e

    async def generate_json(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        messages: Optional[list[dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Convenience method for JSON-mode generation."""
        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            model=model,
            seed=seed,
        )

    def get_usage_summary(self) -> dict:
        """Get accumulated usage summary."""
        return self.tracker.snapshot()

    def reset_usage(self) -> None:
        """Reset accumulated usage tracking."""
        self.tracker.reset()
