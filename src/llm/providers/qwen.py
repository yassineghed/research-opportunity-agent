import os

from openai import OpenAI

from src.llm.config import env_float, env_int, get_api_key
from src.llm.retry import RetryableLLM


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name for the DashScope API.

    Guards against accidentally inheriting a Gemini-style ``models/...`` name,
    which DashScope rejects with ``Model not found``.
    """
    normalized = model_name.strip()
    if normalized.lower().startswith("models/"):
        normalized = normalized[len("models/"):]
    return normalized.strip()


class QwenLLM(RetryableLLM):

    provider_name = "Qwen"

    #: DashScope international OpenAI-compatible endpoint (no workspace ID needed).
    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    #: Default model when QWEN_MODEL is not configured.
    DEFAULT_MODEL = "qwen-plus"

    def __init__(self, model_name: str = None):
        super().__init__(
            max_attempts=env_int("QWEN_MAX_ATTEMPTS", 3),
            base_delay_seconds=env_float("QWEN_RETRY_BASE_DELAY_SECONDS", 2.0),
            max_delay_seconds=env_float("QWEN_RETRY_MAX_DELAY_SECONDS", 60.0),
        )

        api_key = get_api_key("QWEN_API_KEY", "DASHSCOPE_API_KEY")
        timeout_seconds = env_float("QWEN_TIMEOUT_SECONDS", 60.0)
        self.base_url = os.getenv("QWEN_API_BASE_URL", self.DEFAULT_BASE_URL)
        resolved_model_name = model_name or os.getenv("QWEN_MODEL", self.DEFAULT_MODEL)

        if not api_key:
            raise ValueError(
                "Qwen API key is not configured. "
                "Set QWEN_API_KEY in the project .env file."
            )

        self.model_name = _normalize_model_name(resolved_model_name)

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def _generate_once(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        message = response.choices[0].message.content
        return message or ""
