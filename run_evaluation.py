from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, PipelineConfig
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.llm.client import LLMClient
from src.loaders import DataLoader
from src.matching.ranker import OpportunityRanker
from src.reranking.llm_reranker import LLMReranker
from src.evaluation.benchmark import EvalDataset
from src.evaluation.evaluator import PipelineEvaluator


def main() -> None:
    config = PipelineConfig()

    print("Loading datasets...", flush=True)
    researchers = DataLoader.load_researchers(config.researchers_file)
    opportunities = DataLoader.load_opportunities(config.opportunities_file)

    eval_path = PROJECT_ROOT / "data" / "eval" / "benchmark_queries.json"
    try:
        eval_dataset = EvalDataset.load_from_json(eval_path)
    except FileNotFoundError:
        print(f"Evaluation dataset not found: {eval_path}")
        return

    researchers_by_id = {r.id: r for r in researchers}
    opportunities_by_id = {opp.id: opp for opp in opportunities}

    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder(config.embedding_model)
    ranker = OpportunityRanker()

    llm_rerankers = [
        (name, LLMReranker(LLMClient(provider=name)))
        for name in config.llm_providers
    ]

    print("Building index...", flush=True)
    index_records = build_opportunity_index(
        opportunities, opportunity_builder, embedder, show_progress=True,
    )

    evaluator = PipelineEvaluator(
        embedder=embedder,
        ranker=ranker,
        profile_builder=profile_builder,
        llm_rerankers=llm_rerankers,
        top_k_retrieval=config.top_k_retrieval,
        top_k_final=config.top_k_final,
    )

    print("\nStarting evaluation...", flush=True)
    results = evaluator.evaluate(
        eval_dataset, researchers_by_id, opportunities_by_id, index_records,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    print("=" * 50)


if __name__ == "__main__":
    main()
