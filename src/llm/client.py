import os

from src.llm.provider.gemini import GeminiLLM


class LLMClient:

    def __init__(self):

        provider = os.getenv(
            "LLM_PROVIDER",
            "gemini"
        )

        model_name = os.getenv(
            "LLM_MODEL"
        )

        if provider == "gemini":

            self.llm = GeminiLLM(
                model_name=model_name
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)