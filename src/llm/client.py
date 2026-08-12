import os
from typing import Optional

from src.llm.providers.gemini import GeminiLLM
from src.llm.providers.grok import GrokLLM


class LLMClient:

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        llm=None,
    ):

        if llm is not None:
            self.llm = llm
            return

        provider_name = provider or os.getenv(
            "LLM_PROVIDER",
            "gemini"
        )
        provider_name = provider_name.strip().lower()

        resolved_model_name = model_name or os.getenv("LLM_MODEL")

        if provider_name == "gemini":
            self.llm = GeminiLLM(
                model_name=resolved_model_name
            )
        elif provider_name in {"grok", "xai"}:
            self.llm = GrokLLM(
                model_name=resolved_model_name
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}"
            )

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)