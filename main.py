from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.llm.client import LLMClient
from src.loaders import DataLoader
from src.matching.ranker import OpportunityRanker
from src.reranking.llm_reranker import LLMReranker

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K_RETRIEVAL = 5
TOP_K_FINAL = 3


def _build_opportunity_index(opportunities, builder, embedder):
    opportunity_texts = [builder.build(opportunity) for opportunity in opportunities]
    opportunity_vectors = embedder.encode_batch(opportunity_texts)
    return opportunity_vectors


def _format_recommendations(recommendations: list[dict[str, Any]], opportunities_by_id: dict[int, Any]) -> str:
    lines: list[str] = []

    for index, recommendation in enumerate(recommendations[:TOP_K_FINAL], start=1):
        opportunity = opportunities_by_id.get(recommendation["opportunity_id"])
        title = opportunity.title if opportunity else "Unknown opportunity"
        organization = opportunity.organization if opportunity else "Unknown organization"
        score = recommendation.get("score", 0)
        reason = recommendation.get("reason", "")
        matching_areas = recommendation.get("matching_areas", [])

        lines.append(f"{index}. {title} - {organization}")
        lines.append(f"   Score: {score}")
        if matching_areas:
            lines.append(f"   Matching areas: {', '.join(matching_areas)}")
        if reason:
            lines.append(f"   Reason: {reason}")

    return "\n".join(lines)


def main() -> None:
    researchers = DataLoader.load_researchers(PROJECT_ROOT / "data" / "mock" / "researchers.json")
    opportunities = DataLoader.load_opportunities(PROJECT_ROOT / "data" / "mock" / "opportunities.json")

    if not researchers:
        raise ValueError("No researchers found in mock data.")
    if not opportunities:
        raise ValueError("No opportunities found in mock data.")

    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder(EMBEDDING_MODEL)
    ranker = OpportunityRanker()
    llm_client = LLMClient()
    reranker = LLMReranker(llm_client)

    opportunity_vectors = _build_opportunity_index(opportunities, opportunity_builder, embedder)
    opportunities_by_id = {opportunity.id: opportunity for opportunity in opportunities}

    print(f"Loaded {len(researchers)} researchers and {len(opportunities)} opportunities")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Retrieval top-k: {TOP_K_RETRIEVAL} | Final top-k: {TOP_K_FINAL}")
    print()

    for researcher in researchers:
        researcher_text = profile_builder.build(researcher)
        researcher_vector = embedder.encode(researcher_text)

        retrieval_results = ranker.rank(
            researcher_vector,
            opportunity_vectors,
            opportunities,
        )
        top_candidates = retrieval_results[:TOP_K_RETRIEVAL]
        candidate_opportunities = [opportunities_by_id[item["opportunity_id"]] for item in top_candidates]

        reranked = reranker.rerank(researcher, candidate_opportunities)
        recommendations = reranked.get("recommendations", [])
        recommendations = sorted(
            recommendations,
            key=lambda item: item.get("score", 0),
            reverse=True,
        )[:TOP_K_FINAL]

        print("=" * 80)
        print(f"Researcher: {researcher.fullname} - {researcher.institution}")
        print("\nTop retrieval candidates:")
        for item in top_candidates:
            print(f"- {item['title']} | similarity={item['score']:.3f}")

        print("\nTop 3 after LLM reranking:")
        if recommendations:
            print(_format_recommendations(recommendations, opportunities_by_id))
        else:
            print("No valid recommendations returned by the LLM.")
        print()


if __name__ == "__main__":
    main()
