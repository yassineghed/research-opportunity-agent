"""Test script for Task 4: Add LLM reranking.

Loads one researcher (or runs for a couple), retrieves candidate opportunities using
semantic search, and invokes the provider-independent LLMReranker to perform the
second-stage ranking. Outputs the final structured JSON results.
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.loaders.json_loader import DataLoader
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.vector_store.faiss_index import VectorIndex
from src.llm.client import LLMClient
from src.reranking.llm_reranker import LLMReranker


def test_llm_reranking():
    parser = argparse.ArgumentParser(description="Test LLM Reranking pipeline.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "qwen"],
        default="gemini",
        help="LLM provider to test reranking with."
    )
    parser.add_argument(
        "--researcher-idx",
        type=int,
        default=1,
        help="1-based index of the researcher in mock data to test with."
    )
    parser.add_argument(
        "--top-k-retrieval",
        type=int,
        default=5,
        help="Number of candidates to retrieve semantically before reranking."
    )
    args = parser.parse_args()

    print(f"=== Running LLM Reranking Test (Provider: {args.provider}) ===")
    
    # 1. Load data
    researchers = DataLoader.load_researchers(str(PROJECT_ROOT / "data" / "mock" / "researchers.json"))
    opportunities = DataLoader.load_opportunities(str(PROJECT_ROOT / "data" / "processed" / "opportunities.json"))
    
    if args.researcher_idx < 1 or args.researcher_idx > len(researchers):
        print(f"ERROR: researcher-idx must be between 1 and {len(researchers)}")
        sys.exit(1)
        
    researcher = researchers[args.researcher_idx - 1]
    print(f"Target Researcher: {researcher.fullname} - {researcher.institution}")
    
    # 2. Embed and retrieve candidate set
    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    
    index_records = build_opportunity_index(opportunities, opportunity_builder, embedder, show_progress=False)
    vector_store = VectorIndex()
    vector_store.build(index_records)
    
    profile_text = profile_builder.build(researcher)
    query_vec = embedder.encode(profile_text)
    
    print(f"Retrieving top {args.top_k_retrieval} candidate opportunities...")
    retrieved_results = vector_store.search(query_vec, top_k=args.top_k_retrieval)
    
    # Map retrieved results to opportunity objects
    opps_by_id = {opp.id: opp for opp in opportunities}
    candidate_opps = [opps_by_id[res["opportunity_id"]] for res in retrieved_results]
    
    print("Candidates retrieved:")
    for idx, cand in enumerate(candidate_opps, start=1):
        print(f"  {idx}. [{cand.id}] {cand.title}")
        
    # 3. LLM client and reranker setup
    print(f"\nInitializing LLMClient for provider '{args.provider}'...")
    llm_client = LLMClient(provider=args.provider)
    reranker = LLMReranker(llm_client)
    
    print("Running LLM rerank...")
    try:
        rerank_output = reranker.rerank(researcher, candidate_opps)
        
        print("\n--- Reranker Output (JSON) ---")
        print(json.dumps(rerank_output, ensure_ascii=False, indent=2))
        
        # Validate output schema
        assert isinstance(rerank_output, dict), "Rerank output is not a JSON object"
        assert "recommendations" in rerank_output, "Missing 'recommendations' key"
        recs = rerank_output["recommendations"]
        assert isinstance(recs, list), "'recommendations' must be a list"
        
        for r in recs:
            assert "opportunity_id" in r, "Missing 'opportunity_id'"
            assert "score" in r, "Missing 'score'"
            assert "reason" in r, "Missing 'reason'"
            assert "matching_areas" in r, "Missing 'matching_areas'"
            
        print("\nStructured JSON validation: PASSED")
        print("Reranking test run finished successfully.")
        
    except Exception as exc:
        print(f"\nRerank failed with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_llm_reranking()
