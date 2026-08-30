import sys
from pathlib import Path
import os
import json

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

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

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K_RETRIEVAL = 5
TOP_K_FINAL = 3


def main() -> None:
    print("Loading datasets...", flush=True)
    researchers = DataLoader.load_researchers(PROJECT_ROOT / "data" / "mock" / "researchers.json")
    opportunities = DataLoader.load_opportunities(PROJECT_ROOT / "data" / "processed" / "opportunities.json")
    
    try:
        eval_dataset = EvalDataset.load_from_json(PROJECT_ROOT / "data" / "eval" / "benchmark_queries.json")
    except FileNotFoundError:
        print("Evaluation dataset not found. Please create data/eval/benchmark_queries.json.")
        return

    researchers_by_id = {r.id: r for r in researchers}
    opportunities_by_id = {opp.id: opp for opp in opportunities}

    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder(EMBEDDING_MODEL)
    ranker = OpportunityRanker()
    gemini_client = LLMClient(provider="gemini")
    qwen_client = LLMClient(provider="qwen")

    llm_rerankers = [
        ("Gemini", LLMReranker(gemini_client)),
        ("Qwen", LLMReranker(qwen_client)),
    ]

    print(f"Building index...", flush=True)
    index_records = build_opportunity_index(
        opportunities,
        opportunity_builder,
        embedder,
        show_progress=True,
    )

    evaluator = PipelineEvaluator(
        embedder=embedder,
        ranker=ranker,
        profile_builder=profile_builder,
        llm_rerankers=llm_rerankers,
        top_k_retrieval=TOP_K_RETRIEVAL,
        top_k_final=TOP_K_FINAL,
    )

    print("\nStarting evaluation...", flush=True)
    results = evaluator.evaluate(
        eval_dataset,
        researchers_by_id,
        opportunities_by_id,
        index_records,
    )

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    print("="*50)


if __name__ == "__main__":
    main()
