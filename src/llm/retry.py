from __future__ import annotations

import time
from abc import abstractmethod

from src.llm.base import BaseLLM
from src.llm.errors import LLMQuotaExceededError, LLMUnavailableError, classify_llm_error


class RetryableLLM(BaseLLM):
    """Provider base class that retries transient failures with backoff.

    Subclasses must implement :meth:`_generate_once` for the actual provider
    call. :meth:`generate` wraps it and:

    - retries HTTP 408/429/5xx errors (throttling and temporary outages),
    - honours ``Retry-After``/``retryDelay`` hints returned by the provider,
    - gives up immediately for auth (401), permission (403) and request
      (400/404) errors by raising a typed :class:`src.llm.errors.LLMError`.

    Attributes:
        provider_name: Human readable name injected into error messages.
        max_attempts: Maximum number of attempts (including the first one).
        base_delay_seconds: Base exponential backoff delay.
        max_delay_seconds: Upper bound for a single sleep.
    """

    provider_name = "unknown"

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 60.0,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.base_delay_seconds = float(base_delay_seconds)
        self.max_delay_seconds = float(max_delay_seconds)

    @abstractmethod
    def _generate_once(self, prompt: str) -> str:
        """Perform a single provider call. Implemented by subclasses."""

    def generate(self, prompt: str) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._generate_once(prompt)
            except Exception as exc:
                error = classify_llm_error(self.provider_name, exc)

                can_retry = isinstance(error, (LLMQuotaExceededError, LLMUnavailableError))
                can_retry = can_retry and attempt < self.max_attempts

                if not can_retry:
                    raise error from exc

                hint = error.retry_after_seconds
                if hint is None:
                    hint = self.base_delay_seconds * (2 ** (attempt - 1))

                delay = min(max(float(hint), 0.0), self.max_delay_seconds)
                time.sleep(delay)

        raise RuntimeError("retry loop exhausted without a result")  # pragma: no cover