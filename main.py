from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent import RecommendationAgent
from src.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _print_result(result) -> None:
    print("=" * 80)
    print(f"Researcher: {result.researcher_name} - {result.institution}")
    print(f"Total time: {result.elapsed_total:.1f}s")

    print("\nTop {n} cosine retrieval:".format(n=len(result.retrieval_top)))
    for i, item in enumerate(result.retrieval_top, start=1):
        print(f"  {i}. {item.title} - {item.organization} (score: {item.score:.3f})")

    for provider, items in result.provider_results.items():
        print(f"\nTop {len(items)} after {provider} reranking ({result.provider_timing[provider]:.1f}s):")
        for i, item in enumerate(items, start=1):
            print(f"  {i}. {item.title} - {item.organization}")
            print(f"     Score: {item.score}")
            if item.matching_areas:
                print(f"     Matching areas: {', '.join(item.matching_areas)}")
            if item.reason:
                print(f"     Reason: {item.reason}")
        if result.provider_errors.get(provider):
            print(f"  ({provider} error: {result.provider_errors[provider]})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Opportunity Recommendation Agent")
    parser.add_argument("--load", type=str, metavar="DIR", help="Load a previously-saved FAISS index from DIR")
    parser.add_argument("--persist", type=str, metavar="DIR", help="Save the FAISS index to DIR after building")
    args = parser.parse_args()

    config = PipelineConfig()

    print(f"Embedding model: {config.embedding_model}")
    print(f"LLM providers: {config.llm_providers}")
    print(f"Retrieval top_k={config.top_k_retrieval}, final top_k={config.top_k_final}")

    agent = RecommendationAgent(config)
    agent.load_data()

    if args.load:
        print(f"Loading FAISS index from {args.load}...")
        agent.load_index(args.load)
    else:
        print("Building FAISS index...")
        agent.build_index(show_progress=True, persist_dir=args.persist)

    for researcher in agent.researchers:
        print(f"\n[Researcher {researcher.id}] {researcher.fullname} - {researcher.institution}", flush=True)
        result = agent.query(researcher.id)
        _print_result(result)


if __name__ == "__main__":
    main()
