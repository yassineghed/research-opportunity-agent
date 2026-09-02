"""Output models for the recommendation agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationItem:
    """A single recommended opportunity with explanation."""

    opportunity_id: str | int
    title: str
    organization: str
    score: float
    reason: str
    matching_areas: list[str] = field(default_factory=list)
    source: str = "reranked"  # "retrieval" | "reranked"
    deadline: str = ""
    url: str = ""
    opportunity_type: str = ""


@dataclass
class RecommendationResult:
    """Full result of a recommendation query for one researcher."""

    researcher_id: int | str
    researcher_name: str
    institution: str
    recommendations: list[RecommendationItem]
    retrieval_top: list[RecommendationItem]
    provider_results: dict[str, list[RecommendationItem]]
    provider_timing: dict[str, float]
    provider_errors: dict[str, str | None]
    elapsed_total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "researcher_id": self.researcher_id,
            "researcher_name": self.researcher_name,
            "institution": self.institution,
            "recommendations": [
                {
                    "opportunity_id": r.opportunity_id,
                    "title": r.title,
                    "organization": r.organization,
                    "score": r.score,
                    "reason": r.reason,
                    "matching_areas": r.matching_areas,
                    "source": r.source,
                }
                for r in self.recommendations
            ],
            "retrieval_top": [
                {
                    "opportunity_id": r.opportunity_id,
                    "title": r.title,
                    "score": r.score,
                }
                for r in self.retrieval_top
            ],
            "provider_timing": self.provider_timing,
            "provider_errors": self.provider_errors,
            "elapsed_total": self.elapsed_total,
        }
