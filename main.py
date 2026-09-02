from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.llm.client import LLMClient
from src.llm.errors import LLMError
from src.loaders import DataLoader
from src.matching.ranker import OpportunityRanker
from src.reranking.llm_reranker import LLMReranker


def _format_recommendations(recommendations: list[dict[str, Any]], opportunities_by_id: dict) -> str:
    lines: list[str] = []
    for index, rec in enumerate(recommendations, start=1):
        opp = opportunities_by_id.get(rec["opportunity_id"])
        title = opp.title if opp else "Unknown"
        org = opp.organization if opp else "Unknown"
        score = rec.get("score", 0)
        reason = rec.get("reason", "")
        areas = rec.get("matching_areas", [])

        lines.append(f"{index}. {title} - {org}")
        lines.append(f"   Score: {score}")
        if areas:
            lines.append(f"   Matching areas: {', '.join(areas)}")
        if reason:
            lines.append(f"   Reason: {reason}")
    return "\n".join(lines)


def _format_retrieval(results: list[dict[str, Any]], opportunities_by_id: dict) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        opp = opportunities_by_id.get(item["opportunity_id"])
        title = opp.title if opp else item.get("title", "Unknown")
        org = opp.organization if opp else item.get("organization", "Unknown")
        lines.append(f"{index}. {title} - {org}")
        lines.append(f"   Similarity: {item['score']:.3f}")
    return "\n".join(lines)


def main() -> None:
    config = PipelineConfig()

    researchers = DataLoader.load_researchers(config.researchers_file)
    opportunities = DataLoader.load_opportunities(config.opportunities_file)

    if not researchers:
        raise ValueError("No researchers found.")
    if not opportunities:
        raise ValueError("No opportunities found.")

    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder(config.embedding_model)
    ranker = OpportunityRanker()

    llm_rerankers = [
        (name, LLMReranker(LLMClient(provider=name)))
        for name in config.llm_providers
    ]

    print(f"Loaded {len(researchers)} researchers, {len(opportunities)} opportunities")
    print(f"Embedding model: {config.embedding_model}")
    print("Building opportunity index...", flush=True)

    index_records = build_opportunity_index(
        opportunities, opportunity_builder, embedder, show_progress=True,
    )
    opportunities_by_id = {opp.id: opp for opp in opportunities}

    print(f"Index: {len(index_records)} records, dim={embedder.embedding_dim}")
    print(f"Retrieval top_k={config.top_k_retrieval}, final top_k={config.top_k_final}")
    print(f"LLM providers: {[name for name, _ in llm_rerankers]}")
    print(f"LLM fallback: {config.llm_allow_fallback}")
    print(flush=True)

    # Per-provider stats
    provider_stats: dict[str, dict[str, Any]] = {
        name: {"calls": 0, "errors": 0, "total_seconds": 0.0}
        for name, _ in llm_rerankers
    }

    for idx, researcher in enumerate(researchers, start=1):
        print(f"[{idx}/{len(researchers)}] {researcher.fullname} - {researcher.institution}", flush=True)

        researcher_vector = embedder.encode(profile_builder.build(researcher))
        retrieval_results = ranker.rank(researcher_vector, index_records, top_k=config.top_k_retrieval)
        top_candidates = retrieval_results[:config.top_k_retrieval]
        candidate_opps = [opportunities_by_id[item["opportunity_id"]] for item in top_candidates]

        provider_results: list[tuple[str, list[dict[str, Any]], Exception | None, float]] = []

        for provider_name, reranker in llm_rerankers:
            error: Exception | None = None
            recommendations: list[dict[str, Any]] = []
            elapsed = 0.0

            try:
                t0 = time.perf_counter()
                reranked = reranker.rerank(researcher, candidate_opps)
                elapsed = time.perf_counter() - t0
                recommendations = sorted(
                    reranked.get("recommendations", []),
                    key=lambda r: r.get("score", 0),
                    reverse=True,
                )[:config.top_k_final]
            except Exception as exc:
                if not isinstance(exc, LLMError):
                    raise
                error = exc
                print(f"  {provider_name} failed: {exc}", flush=True)
                if config.llm_allow_fallback:
                    recommendations = [
                        {
                            "opportunity_id": item["opportunity_id"],
                            "score": round(item["score"] * 100, 0),
                            "reason": f"{provider_name} unavailable; cosine fallback.",
                            "matching_areas": [],
                        }
                        for item in top_candidates[:config.top_k_final]
                    ]

            provider_stats[provider_name]["calls"] += 1
            provider_stats[provider_name]["total_seconds"] += elapsed
            if error is not None:
                provider_stats[provider_name]["errors"] += 1

            provider_results.append((provider_name, recommendations, error, elapsed))

        retrieval_top = top_candidates[:config.top_k_final]
        retrieval_ids = [item["opportunity_id"] for item in retrieval_top]

        print("=" * 80)
        print(f"Researcher: {researcher.fullname} - {researcher.institution}")
        print("\nTop 3 cosine similarity:")
        print(_format_retrieval(retrieval_top, opportunities_by_id))

        for provider_name, recommendations, error, elapsed in provider_results:
            rerank_ids = [r["opportunity_id"] for r in recommendations]
            overlap = len(set(retrieval_ids) & set(rerank_ids))

            print(f"\nTop 3 after {provider_name} reranking ({elapsed:.1f}s):")
            if recommendations:
                print(_format_recommendations(recommendations, opportunities_by_id))
            else:
                print("No valid recommendations returned.")

            if error is not None:
                print(f"  {provider_name} error: {error}")
                if config.llm_allow_fallback:
                    print("  Cosine fallback displayed.")

            print(f"\nOverlap cosine vs {provider_name}: {overlap}/3")
        print()

    # Summary
    print("=" * 80)
    print("PROVIDER SUMMARY")
    print("=" * 80)
    for name, stats in provider_stats.items():
        avg = stats["total_seconds"] / stats["calls"] if stats["calls"] else 0
        print(
            f"  {name}: {stats['calls']} calls, "
            f"{stats['errors']} errors, "
            f"avg {avg:.1f}s/call"
        )


if __name__ == "__main__":
    main()
