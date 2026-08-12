from __future__ import annotations


class LLMError(Exception):

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.original_exception = original_exception

    def __str__(self) -> str:
        if self.status_code is None:
            return f"{self.provider}: {self.message}"

        return f"{self.provider} ({self.status_code}): {self.message}"


class LLMQuotaExceededError(LLMError):
    pass


class LLMForbiddenError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


def _extract_status_code(exc: Exception) -> int | None:
    for attribute_name in ("status_code", "status", "code"):
        value = getattr(exc, attribute_name, None)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    return None


def classify_llm_error(provider: str, exc: Exception) -> LLMError:
    message = str(exc)
    lower_message = message.lower()
    status_code = _extract_status_code(exc)

    if status_code == 429 or "resource_exhausted" in lower_message or "quota" in lower_message or "rate limit" in lower_message:
        return LLMQuotaExceededError(
            provider=provider,
            message="quota or rate limit reached; retry later",
            status_code=status_code,
            original_exception=exc,
        )

    if status_code == 403 or "forbidden" in lower_message or "permission" in lower_message or "unauthorized" in lower_message:
        return LLMForbiddenError(
            provider=provider,
            message="access forbidden or API key rejected",
            status_code=status_code,
            original_exception=exc,
        )

    if status_code == 503 or "unavailable" in lower_message or "high demand" in lower_message:
        return LLMUnavailableError(
            provider=provider,
            message="service temporarily unavailable",
            status_code=status_code,
            original_exception=exc,
        )

    return LLMError(
        provider=provider,
        message="LLM request failed",
        status_code=status_code,
        original_exception=exc,
    )