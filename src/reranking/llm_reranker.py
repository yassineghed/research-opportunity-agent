"""LLM-based reranker — scores and explains researcher–opportunity matches.

Uses a structured prompt with a few-shot example and returns typed
:class:`RerankResult` objects so callers never have to deal with raw dicts.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined matching areas — keeps LLM output consistent
# ---------------------------------------------------------------------------
MATCHING_AREAS: list[str] = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Cybersecurity",
    "Data Science",
    "Health / Medical AI",
    "Food Science / Agritech",
    "Climate / Environment",
    "Energy / Sustainability",
    "Transport / Mobility",
    "Digital Transformation",
    "Education / Training",
    "Ethics / Trustworthy AI",
    "HPC / Quantum Computing",
    "Cybersecurity / Privacy",
    "Blockchain / Distributed Systems",
    "IoT / Edge Computing",
    "Other",
]

_MAX_DESC_CHARS = 300


# ---------------------------------------------------------------------------
# Typed return value
# ---------------------------------------------------------------------------
@dataclass
class RerankItem:
    opportunity_id: str | int
    score: float
    reason: str
    matching_areas: list[str] = field(default_factory=list)


@dataclass
class RerankResult:
    recommendations: list[RerankItem]
    raw_json: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, max_chars: int = _MAX_DESC_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_few_shot_block() -> str:
    return """
--- EXAMPLE ---

Researcher:
Name: Maria Costa
Institution: University of Lisbon
Research domains: Computer Vision, Medical Imaging
Research interests: Explainable AI, surgical navigation
Skills: Python, PyTorch, DICOM
Keywords: XAI, medical image segmentation

Candidate opportunities:
Opportunity ID: FT-100
Title: AI for Medical Imaging Challenge
Type: Innovation Action
Organization: European Commission
Description: Funding for projects applying AI to improve medical image analysis in radiology and pathology.
Keywords: AI, medical imaging, radiology
Topics: Health, Digital Health
Eligibility: EU-27 + associated countries
Deadline: 2026-03-15

Opportunity ID: FT-200
Title: Cybersecurity for Smart Grids
Type: Research & Innovation Action
Organization: ENISA
Description: Projects enhancing cybersecurity of energy infrastructure using AI-based threat detection.
Keywords: cybersecurity, energy, smart grid
Topics: Energy, Security
Eligibility: EU-27
Deadline: 2026-04-01

Expected output:
{
    "recommendations": [
        {
            "opportunity_id": "FT-100",
            "score": 90,
            "reason": "Directly aligns with medical imaging and explainable AI research.",
            "matching_areas": ["Computer Vision", "Health / Medical AI", "Ethics / Trustworthy AI"]
        },
        {
            "opportunity_id": "FT-200",
            "score": 25,
            "reason": "Partial overlap on AI methods but different application domain (energy vs health).",
            "matching_areas": ["Artificial Intelligence", "Cybersecurity / Privacy"]
        }
    ]
}
--- END EXAMPLE ---
"""


# ---------------------------------------------------------------------------
# Main reranker class
# ---------------------------------------------------------------------------
class LLMReranker:
    """Reranks candidate opportunities for a researcher using an LLM."""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def rerank(
        self,
        researcher: Any,
        opportunities: list[Any],
        timeout_hint: float = 30.0,
    ) -> dict[str, Any]:
        """Score and explain each opportunity for the researcher.

        Returns a dict with key ``"recommendations"`` (list of dicts) for
        backward compatibility with ``main.py`` and ``PipelineEvaluator``.
        """
        prompt = self._build_prompt(researcher, opportunities)

        t0 = time.perf_counter()
        try:
            response = self.llm_client.generate(prompt)
        except Exception:
            raise
        elapsed = time.perf_counter() - t0

        result = self._parse_response(response)
        result["elapsed_seconds"] = elapsed
        logger.info(
            "LLM rerank completed in %.2fs — %d recommendations",
            elapsed,
            len(result.get("recommendations", [])),
        )
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, researcher: Any, opportunities: list[Any]) -> str:
        opps_text = self._format_opportunities(opportunities)
        areas_enum = "\n".join(f"  - {a}" for a in MATCHING_AREAS)
        few_shot = _build_few_shot_block()

        return f"""You are a scientific funding opportunity recommendation system.

Your task: evaluate how relevant each candidate opportunity is for the given researcher.

--- MATCHING AREAS (use ONLY these values) ---
{areas_enum}

{few_shot}

--- REAL TASK ---

Researcher:
Name: {researcher.fullname}
Institution: {researcher.institution}
Research domains: {", ".join(researcher.research_domains)}
Research interests: {", ".join(researcher.research_interests)}
Skills: {", ".join(researcher.skills)}
Keywords: {", ".join(researcher.keywords)}
Publications: {", ".join(researcher.publications)}

Candidate opportunities:
{opps_text}

Instructions:
1. For each opportunity, give a relevance score from 0 to 100.
2. Explain in ONE sentence why it matches (or does not).
3. Pick 1-3 matching areas from the allowed list above.
4. Return ONLY valid JSON — no commentary before or after.
5. Sort by score descending.

Return this exact JSON structure:
{{
    "recommendations": [
        {{
            "opportunity_id": "<id>",
            "score": <int 0-100>,
            "reason": "<one sentence>",
            "matching_areas": ["<area1>", "<area2>"]
        }}
    ]
}}
"""

    @staticmethod
    def _format_opportunities(opportunities: list[Any]) -> str:
        parts: list[str] = []
        for opp in opportunities:
            parts.append(
                f"Opportunity ID: {opp.id}\n"
                f"Title: {opp.title}\n"
                f"Type: {opp.type}\n"
                f"Organization: {opp.organization}\n"
                f"Description: {_truncate(opp.description)}\n"
                f"Keywords: {', '.join(opp.keywords)}\n"
                f"Topics: {', '.join(opp.topics)}\n"
                f"Eligibility: {opp.eligibility}\n"
                f"Deadline: {opp.deadline}"
            )
        return "\n---\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> dict[str, Any]:
        payload = self._extract_json(response)

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, attempting repair")
            repaired = self._repair_json(payload)
            parsed = json.loads(repaired)

        validated = self._validate_schema(parsed)
        return validated

    @staticmethod
    def _extract_json(response: str) -> str:
        if not isinstance(response, str):
            raise ValueError("LLM response must be a string")

        trimmed = response.strip()

        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", trimmed, re.IGNORECASE | re.DOTALL)
        if fence:
            return fence.group(1).strip()

        first_brace = trimmed.find("{")
        last_brace = trimmed.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return trimmed[first_brace : last_brace + 1]

        return trimmed

    @staticmethod
    def _repair_json(text: str) -> str:
        text = text.strip()
        if not text.startswith("{"):
            text = "{" + text
        if not text.endswith("}"):
            text = text + "}"
        return text

    @staticmethod
    def _validate_schema(parsed: dict[str, Any]) -> dict[str, Any]:
        recs = parsed.get("recommendations")
        if not isinstance(recs, list):
            raise ValueError("Missing or non-list 'recommendations' key")

        valid_areas = set(MATCHING_AREAS)
        cleaned: list[dict[str, Any]] = []

        for rec in recs:
            if not isinstance(rec, dict):
                continue
            opp_id = rec.get("opportunity_id")
            score = rec.get("score")
            reason = rec.get("reason", "")
            areas = rec.get("matching_areas", [])

            if opp_id is None or score is None:
                continue

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            if isinstance(areas, list):
                areas = [a for a in areas if a in valid_areas]
            else:
                areas = []

            cleaned.append(
                {
                    "opportunity_id": str(opp_id),
                    "score": score,
                    "reason": str(reason),
                    "matching_areas": areas,
                }
            )

        cleaned.sort(key=lambda r: r["score"], reverse=True)
        return {"recommendations": cleaned}
