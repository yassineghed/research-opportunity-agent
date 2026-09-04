from __future__ import annotations

import json
import re

from src.llm.client import LLMClient
from src.models.researcher import Researcher


class CVProfileExtractor:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient(provider="gemini")

    def extract(self, cv_text: str) -> Researcher:
        if not cv_text.strip():
            raise ValueError("The uploaded document did not contain readable text.")

        response = self.client.generate(self._prompt(cv_text))
        profile = self._parse_response(response)
        return Researcher(
            id=0,
            fullname=str(profile.get("fullname", "")).strip(),
            institution=str(profile.get("institution", "")).strip(),
            research_domains=self._list_field(profile.get("research_domains")),
            research_interests=self._list_field(profile.get("research_interests")),
            skills=self._list_field(profile.get("skills")),
            keywords=self._list_field(profile.get("keywords")),
            publications=self._list_field(profile.get("publications")),
        )

    @staticmethod
    def _list_field(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]
        return []

    @staticmethod
    def _parse_response(response: str) -> dict:
        cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip(), flags=re.IGNORECASE)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("CV extraction returned an invalid profile object.")
        return parsed

    @staticmethod
    def _prompt(cv_text: str) -> str:
        return f"""Extract a researcher profile from the CV below.
Return JSON only with exactly these keys:
fullname, institution, research_domains, research_interests, skills, keywords, publications.
Use arrays of concise strings for every field except fullname and institution.
Do not invent information. Use an empty array or empty string when a field is absent.

CV text:
{cv_text}
"""