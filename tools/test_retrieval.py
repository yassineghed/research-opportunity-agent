"""Test script for Task 3: Building semantic retrieval.

Loads researchers and opportunities, builds the index, retrieves the Top-K
opportunities for each researcher using the BGE embeddings, and prints them
for manual/logical relevance evaluation. Does not use LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.json_loader import DataLoader
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index
from src.vector_store.faiss_index import VectorIndex


def test_semantic_retrieval():
    print("=== Running Semantic Retrieval Tests ===")
    
    # 1. Load data
    researchers = DataLoader.load_researchers(str(PROJECT_ROOT / "data" / "mock" / "researchers.json"))
    opportunities = DataLoader.load_opportunities(str(PROJECT_ROOT / "data" / "processed" / "opportunities.json"))
    
    print(f"Loaded {len(researchers)} researchers and {len(opportunities)} opportunities.")
    
    # 2. Build index
    profile_builder = StructuredProfileBuilder()
    opportunity_builder = StructuredOpportunityBuilder()
    model_name = "BAAI/bge-small-en-v1.5"
    embedder = Embedder(model_name)
    
    print("Building opportunity embedding index...")
    index_records = build_opportunity_index(opportunities, opportunity_builder, embedder, show_progress=False)
    
    print("Initializing VectorIndex and loading index records...")
    vector_store = VectorIndex()
    vector_store.build(index_records)
    print(f"Vector Index built with size: {vector_store.size}, dim: {vector_store.dim}")
    
    # 3. Retrieve Top-K for each researcher
    top_k = 5
    print(f"\nRetrieving Top-{top_k} opportunities for each researcher:")
    print("=" * 80)
    
    for r in researchers:
        print(f"\nRESEARCHER: {r.fullname}")
        print(f"Institution: {r.institution}")
        print(f"Domains:     {', '.join(r.research_domains)}")
        print(f"Interests:   {', '.join(r.research_interests)}")
        print(f"Keywords:    {', '.join(r.keywords)}")
        print("-" * 40)
        
        # Build profile text and embed
        profile_text = profile_builder.build(r)
        query_vec = embedder.encode(profile_text)
        
        # Retrieve
        results = vector_store.search(query_vec, top_k=top_k)
        
        for idx, result in enumerate(results, start=1):
            print(f"  {idx}. [Score: {result['score']:.4f}] {result['title']}")
            print(f"     Org: {result['organization']} | Type: {result['type']}")
            # Truncate description preview
            desc = result.get('description', '')
            desc_preview = desc[:120].strip().replace('\n', ' ') + "..." if len(desc) > 120 else desc
            print(f"     Preview: {desc_preview}")
            print()
            
    print("=" * 80)
    print("Retrieval test run finished.")


if __name__ == "__main__":
    test_semantic_retrieval()
