import os
from google import genai
from src.llm.base import BaseLLM



class GeminiLLM(BaseLLM):

    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text