import os

from openai import OpenAI

from src.llm.config import env_float, env_int, get_api_key
from src.llm.retry import RetryableLLM


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name for the xAI API.

    Guards against accidentally inheriting a Gemini-style ``models/...`` name,
    which xAI rejects with ``Model not found``.
    """
    normalized = model_name.strip()
    if normalized.lower().startswith("models/"):
        normalized = normalized[len("models/"):]
    return normalized.strip()


class GrokLLM(RetryableLLM):

    provider_name = "Grok"

    #: Default xAI-compatible endpoint (Chat Completions and Responses APIs).
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    #: Default model when GROK_MODEL is not configured.
    DEFAULT_MODEL = "grok-4.5-latest"

    def __init__(self, model_name: str = None):
        super().__init__(
            max_attempts=env_int("GROK_MAX_ATTEMPTS", 3),
            base_delay_seconds=env_float("GROK_RETRY_BASE_DELAY_SECONDS", 2.0),
            max_delay_seconds=env_float("GROK_RETRY_MAX_DELAY_SECONDS", 60.0),
        )

        api_key = get_api_key("GROK_API_KEY", "XAI_API_KEY", "GORK_API_KEY")
        timeout_seconds = env_float("GROK_TIMEOUT_SECONDS", 60.0)
        self.base_url = os.getenv("GROK_API_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_model_name = model_name or os.getenv("GROK_MODEL", self.DEFAULT_MODEL)

        if not api_key:
            raise ValueError(
                "Grok API key is not configured. "
                "Set GROK_API_KEY in the project .env file."
            )

        self.model_name = _normalize_model_name(resolved_model_name)

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def _generate_once(self, prompt: str) -> str:
        try:
            return self._generate_with_responses_api(prompt)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in (404, 405):
                return self._generate_with_chat_completions(prompt)
            raise

    def _generate_with_responses_api(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text
        raise ValueError("Grok responses API returned no text output.")

    def _generate_with_chat_completions(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        message = response.choices[0].message.content
        return message or ""