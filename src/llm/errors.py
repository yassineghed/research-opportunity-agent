from __future__ import annotations

import re
from typing import Any

#: HTTP status codes for which it is safe to retry the request.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(Exception):
    """Base error raised by the generic LLM layer.

    Attributes:
        provider: Human readable provider name, e.g. "Gemini" or "Grok".
        message: Short, actionable summary of the failure.
        status_code: HTTP status code when one was reported by the provider.
        original_exception: The underlying provider-specific exception.
        retry_after_seconds: Seconds the provider asked us to wait, if any.
        hint: Optional actionable advice to fix the problem.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        original_exception: Exception | None = None,
        retry_after_seconds: float | None = None,
        hint: str | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.status_code = status_code
        self.original_exception = original_exception
        self.retry_after_seconds = retry_after_seconds
        self.hint = hint

    def __str__(self) -> str:
        details = ""
        if self.original_exception is not None:
            details = f" | raw: {self.original_exception}"
        hint = f" | hint: {self.hint}" if self.hint else ""
        if self.status_code is None:
            return f"{self.provider}: {self.message}{details}{hint}"
        return f"{self.provider} ({self.status_code}): {self.message}{details}{hint}"


class LLMAuthenticationError(LLMError):
    """The API key is missing, invalid, expired or rejected (HTTP 401)."""


class LLMForbiddenError(LLMError):
    """The key is valid but lacks permission or the account has no credits (HTTP 403)."""


class LLMQuotaExceededError(LLMError):
    """Rate limit or quota exceeded; the request can be retried later (HTTP 429)."""


class LLMUnavailableError(LLMError):
    """The provider is temporarily unavailable and the request can be retried (HTTP 5xx)."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMBadRequestError(LLMError):
    """The request was rejected by the provider, e.g. unknown model or bad payload (HTTP 4xx)."""


def _extract_status_code(exc: Exception) -> int | None:
    for attribute_name in ("status_code", "code"):
        value = getattr(exc, attribute_name, None)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str) and status.isdigit():
            return int(status)

    return None


def _extract_status_text(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if isinstance(status, str):
        return status

    message = str(exc)
    match = re.search(r"\b([A-Z_]{3,})\b", message)
    return match.group(1) if match else ""


_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$")


def _parse_duration(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    match = _DURATION_PATTERN.match(str(value))
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2) or "s"
    factor = {"ms": 1 / 1000.0, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return amount * factor


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is not None:
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "retry-after":
                parsed = _parse_duration(value)
                if parsed is not None:
                    return parsed

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        error_payload = details.get("error")
        if isinstance(error_payload, dict):
            for entry in error_payload.get("details") or []:
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("@type", ""))
                if "RetryInfo" not in entry_type and "retryDelay" not in entry:
                    continue
                parsed = _parse_duration(entry.get("retryDelay"))
                if parsed is not None:
                    return parsed

    return None


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True

    class_name = type(exc).__name__
    if "Timeout" in class_name:
        return True

    lowered_message = str(exc).lower()
    timeout_phrases = (
        "timed out",
        "time out",
        "read timeout",
        "connect timeout",
        "write timeout",
        "operation timed",
        "deadline exceeded",
        "deadline_exceeded",
    )
    return any(phrase in lowered_message for phrase in timeout_phrases)


def _forbidden_hint(provider: str) -> str:
    if provider.lower() in {"grok", "xai"}:
        return (
            "the key may be valid but the team needs credits or licenses "
            "(top up on console.x.ai); or the key lacks the required permissions"
        )
    return "check the key's permissions and billing status on the provider console"


def _unavailable_hint(provider: str) -> str:
    if provider.lower() in {"grok", "xai"}:
        return "transient xAI outage; retry with backoff"
    return "transient Google outage; retry with backoff"


def _provider_message(exc: Exception, limit: int = 300) -> str:
    """Extract the provider's own human-readable explanation, if available."""
    details = getattr(exc, "details", None)
    error_payload = details.get("error") if isinstance(details, dict) else None
    friendly = error_payload.get("message") if isinstance(error_payload, dict) else None

    message = friendly or str(exc)
    cleaned = " ".join(str(message).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "..."


def classify_llm_error(
    provider: str,
    exc: Exception,
) -> LLMError:
    """Map a raw provider exception to a typed :class:`LLMError`.

    The classification is deliberately provider-agnostic: it relies on the
    status code, a small set of well-known keywords and the error payload
    structure shared by common providers.
    """
    message = str(exc)
    lower_message = message.lower()
    status_code = _extract_status_code(exc)
    status_text = _extract_status_text(exc).lower()
    retry_after_seconds = _extract_retry_after_seconds(exc)

    if _is_timeout(exc):
        return LLMTimeoutError(
            provider=provider,
            message="request timed out",
            status_code=status_code,
            original_exception=exc,
            retry_after_seconds=retry_after_seconds,
            hint="increase the provider timeout or retry later",
        )

    if status_code == 401 or "authentication" in lower_message:
        return LLMAuthenticationError(
            provider=provider,
            message="API key is missing, invalid or rejected",
            status_code=status_code,
            original_exception=exc,
            hint="regenerate the API key on the provider console and update .env",
        )

    if (
        status_code == 403
        or "forbidden" in lower_message
        or "permission" in lower_message
        or "credits" in lower_message
        or "licenses" in lower_message
    ):
        return LLMForbiddenError(
            provider=provider,
            message="access forbidden (key lacks permission or account has no credits)",
            status_code=status_code,
            original_exception=exc,
            retry_after_seconds=retry_after_seconds,
            hint=_forbidden_hint(provider),
        )

    if status_code == 429 or "resource_exhausted" in lower_message or "quota" in lower_message or "rate limit" in lower_message:
        return LLMQuotaExceededError(
            provider=provider,
            message="quota or rate limit reached; retry later",
            status_code=status_code,
            original_exception=exc,
            retry_after_seconds=retry_after_seconds,
            hint="wait for the rate limit to reset or increase the account quota",
        )

    if "api_key_invalid" in lower_message or ("api key" in lower_message and "invalid" in lower_message):
        return LLMAuthenticationError(
            provider=provider,
            message="API key is invalid or rejected",
            status_code=status_code,
            original_exception=exc,
            hint="regenerate the API key on the provider console and update .env",
        )

    model_unavailable_phrases = (
        "no longer available",
        "is not available",
        "not supported for",
        "is not found for",
        "does not exist",
        "doesn't exist",
    )
    if any(phrase in lower_message for phrase in model_unavailable_phrases):
        return LLMBadRequestError(
            provider=provider,
            message="model is not available for this API key (deprecated or not entitled)",
            status_code=status_code,
            original_exception=exc,
            hint=f"the provider reported: {_provider_message(exc)}",
        )

    if status_code == 404 or ("model" in lower_message and "not found" in lower_message):
        return LLMBadRequestError(
            provider=provider,
            message="model or endpoint not found",
            status_code=status_code,
            original_exception=exc,
            hint="check that the configured model name is supported by this provider",
        )

    if status_code == 400 or "invalid-argument" in lower_message or "invalid_argument" in status_text or "badrequest" in lower_message:
        return LLMBadRequestError(
            provider=provider,
            message="invalid request (e.g. unknown model or malformed payload)",
            status_code=status_code,
            original_exception=exc,
            hint="check the model name, prompt format and request parameters",
        )

    if status_code in {500, 502, 503, 504} or "unavailable" in lower_message or "high demand" in lower_message:
        return LLMUnavailableError(
            provider=provider,
            message="service temporarily unavailable; retry later",
            status_code=status_code,
            original_exception=exc,
            retry_after_seconds=retry_after_seconds,
            hint=_unavailable_hint(provider),
        )

    return LLMError(
        provider=provider,
        message="LLM request failed",
        status_code=status_code,
        original_exception=exc,
        retry_after_seconds=retry_after_seconds,
    )