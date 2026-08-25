import os

from google import genai
from google.genai import types

from src.llm.base import BaseLLM
from src.llm.errors import classify_llm_error


class GeminiLLM(BaseLLM):

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        timeout_seconds = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        if not model_name:
            raise ValueError(
                "LLM_MODEL is not set."
            )

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            return response.text or ""

        except Exception as exc:
            raise classify_llm_error("Gemini", exc) from exc
