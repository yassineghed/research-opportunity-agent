import os

from openai import OpenAI

from src.llm.base import BaseLLM
from src.llm.errors import classify_llm_error


class GrokLLM(BaseLLM):

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("GORK_API_KEY")
        timeout_seconds = float(os.getenv("GROK_TIMEOUT_SECONDS", "30"))

        if not api_key:
            raise ValueError(
                "GROK_API_KEY is not set."
            )

        if not model_name:
            model_name = os.getenv("GROK_MODEL", "grok-4.5-latest")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=timeout_seconds,
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            return self._generate_with_chat_completions(prompt)

        except Exception as exc:
            if getattr(exc, "status_code", None) == 405:
                try:
                    return self._generate_with_responses_api(prompt)
                except Exception as retry_exc:
                    raise classify_llm_error("Grok", retry_exc) from retry_exc

            raise classify_llm_error("Grok", exc) from exc

    def _generate_with_chat_completions(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        message = response.choices[0].message.content
        return message or ""

    def _generate_with_responses_api(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        text = getattr(response, "output_text", None)
        if text:
            return text

        raise ValueError("Grok responses API returned no text output.")
