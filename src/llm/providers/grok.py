import os

from openai import OpenAI

from src.llm.base import BaseLLM
from src.llm.errors import classify_llm_error


class GrokLLM(BaseLLM):

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("GORK_API_KEY")

        if not api_key:
            raise ValueError(
                "GROK_API_KEY is not set."
            )

        if not model_name:
            model_name = os.getenv("GROK_MODEL", "grok-4.5-latest")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
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

        except Exception as exc:
            raise classify_llm_error("Grok", exc) from exc