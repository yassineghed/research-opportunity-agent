import os
from typing import Optional

from src.llm.providers.gemini import GeminiLLM
from src.llm.providers.grok import GrokLLM
from src.llm.providers.qwen import QwenLLM


class LLMClient:

    """Provider-agnostic entry point for the generic LLM layer.

    The client only resolves *which* provider to build; model names and API
    keys are resolved by the provider itself from the environment (``.env``).
    The original constructor signature is preserved for compatibility:

        LLMClient()                          -> provider from LLM_PROVIDER
        LLMClient(provider="gemini")         -> explicit provider
        LLMClient(provider="grok", llm=...)  -> reuse an existing LLM
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        llm=None,
    ):

        if llm is not None:
            self.llm = llm
            self.provider_name = getattr(llm, "provider_name", "custom")
            self.model_name = getattr(llm, "model_name", None)
            return

        provider_name = provider or os.getenv("LLM_PROVIDER", "gemini")
        provider_name = provider_name.strip().lower()

        if provider_name == "gemini":
            self.llm = GeminiLLM(model_name=model_name)
        elif provider_name in {"grok", "xai"}:
            self.llm = GrokLLM(model_name=model_name)
        elif provider_name in {"qwen", "dashscope"}:
            self.llm = QwenLLM(model_name=model_name)
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name!r}. "
                f"Expected one of: 'gemini', 'grok', 'qwen'."
            )

        self.provider_name = provider_name
        self.model_name = self.llm.model_name

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)