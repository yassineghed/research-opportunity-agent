import os

from google import genai
from google.genai import types

from src.llm.config import env_float, env_int, get_api_key
from src.llm.retry import RetryableLLM


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name for the provider SDK.

    The ``models/`` prefix is accepted by the Gemini REST API but is redundant
    (and sometimes misleading) when passed to the ``google-genai`` SDK, so it
    is stripped here.
    """
    normalized = model_name.strip()
    if normalized.lower().startswith("models/"):
        normalized = normalized[len("models/"):]
    return normalized.strip()


class GeminiLLM(RetryableLLM):

    provider_name = "Gemini"

    def __init__(self, model_name: str = None):
        super().__init__(
            max_attempts=env_int("GEMINI_MAX_ATTEMPTS", 3),
            base_delay_seconds=env_float("GEMINI_RETRY_BASE_DELAY_SECONDS", 2.0),
            max_delay_seconds=env_float("GEMINI_RETRY_MAX_DELAY_SECONDS", 60.0),
        )

        api_key = get_api_key("GEMINI_API_KEY")
        timeout_seconds = env_float("GEMINI_TIMEOUT_SECONDS", 60.0)
        resolved_model_name = model_name or os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL")

        if not api_key:
            raise ValueError(
                "Gemini API key is not configured. "
                "Set GEMINI_API_KEY in the project .env file."
            )

        if not resolved_model_name:
            raise ValueError(
                "No Gemini model configured. "
                "Set GEMINI_MODEL (or the legacy LLM_MODEL) in the project .env file."
            )

        self.model_name = _normalize_model_name(resolved_model_name)

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def _generate_once(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text or ""