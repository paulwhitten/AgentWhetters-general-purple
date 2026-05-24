"""Token usage tracking for all LLM API calls.

Extracts usage data from OpenAI response objects (both Responses API
and Chat Completions API) and accumulates per-model totals. Each call
is logged at INFO level; a summary can be retrieved at any time.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger("agentwhetters.usage")


@dataclass
class _ModelUsage:
    """Accumulated token usage for a single model."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """Thread-safe token usage tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, _ModelUsage] = {}

    def record(self, response: object, *, label: str = "") -> None:
        """Extract and record usage from an OpenAI response object.

        Works with both Responses API (response.usage.input_tokens) and
        Chat Completions API (response.usage.prompt_tokens).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        model = getattr(response, "model", "unknown")

        # Responses API fields
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        # Chat Completions API fields (different names)
        if input_tokens == 0:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        if output_tokens == 0:
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Detail breakdowns
        cached_tokens = 0
        reasoning_tokens = 0

        input_details = getattr(usage, "input_tokens_details", None)
        if input_details:
            cached_tokens = getattr(input_details, "cached_tokens", 0) or 0

        output_details = getattr(usage, "output_tokens_details", None)
        if output_details:
            reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

        # Chat Completions detail format
        if cached_tokens == 0:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        if reasoning_tokens == 0:
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        with self._lock:
            m = self._models.setdefault(model, _ModelUsage())
            m.calls += 1
            m.input_tokens += input_tokens
            m.output_tokens += output_tokens
            m.cached_tokens += cached_tokens
            m.reasoning_tokens += reasoning_tokens

        tag = f" [{label}]" if label else ""
        logger.info(
            "LLM%s model=%s in=%d out=%d cached=%d reasoning=%d",
            tag, model, input_tokens, output_tokens,
            cached_tokens, reasoning_tokens,
        )

    def summary(self) -> dict[str, dict[str, int]]:
        """Return accumulated usage per model."""
        with self._lock:
            return {
                model: {
                    "calls": u.calls,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cached_tokens": u.cached_tokens,
                    "reasoning_tokens": u.reasoning_tokens,
                    "total_tokens": u.total_tokens,
                }
                for model, u in self._models.items()
            }

    def log_summary(self) -> None:
        """Log accumulated usage summary."""
        for model, stats in self.summary().items():
            logger.info(
                "USAGE TOTAL model=%s calls=%d in=%d out=%d "
                "cached=%d reasoning=%d total=%d",
                model, stats["calls"], stats["input_tokens"],
                stats["output_tokens"], stats["cached_tokens"],
                stats["reasoning_tokens"], stats["total_tokens"],
            )

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._models.clear()


# Global singleton
tracker = UsageTracker()
