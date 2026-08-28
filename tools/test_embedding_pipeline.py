"""Test script for Task 2: Building the real opportunity embedding pipeline.

Verifies:
- Loading real processed opportunities from data/processed/opportunities.json
- Extracting searchable text correctly
- Encoding using BGE-small (dimensions, non-zero values, successful output)
- Integrity checks: missing/invalid fields, duplicate IDs, dimension matching.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders.json_loader import DataLoader
from src.builders.opportunity.structured_opportunity_builder import StructuredOpportunityBuilder
from src.embeddings.embedder import Embedder
from src.embeddings.opportunity_index import build_opportunity_index, validate_index


def test_embedding_pipeline():
    print("=== Running Embedding Pipeline Tests ===")
    
    # 1. Load data
    processed_path = PROJECT_ROOT / "data" / "processed" / "opportunities.json"
    print(f"Loading opportunities from {processed_path}...")
    opportunities = DataLoader.load_opportunities(str(processed_path))
    num_opps = len(opportunities)
    print(f"Loaded {num_opps} opportunities.")
    
    if num_opps == 0:
        print("ERROR: No opportunities found.")
        sys.exit(1)
        
    # 2. Build index & generate embeddings
    builder = StructuredOpportunityBuilder()
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Initializing Embedder with model: {model_name}...")
    embedder = Embedder(model_name)
    expected_dim = embedder.embedding_dim
    print(f"Model dimensions: {expected_dim}")
    
    print("Building opportunity index (encoding texts)...")
    records = build_opportunity_index(opportunities, builder, embedder, show_progress=False)
    
    # 3. Validate
    print("Running validation checks...")
    report = validate_index(records, expected_dim=expected_dim)
    
    print("\n--- Validation Report ---")
    print(f"Total processed:  {report['total']}")
    print(f"Embedding dim:    {report['embedding_dim']}")
    print(f"Dim match expected: {report['dim_ok']}")
    print(f"Missing IDs:      {len(report['missing_id'])}")
    print(f"Empty texts:      {len(report['empty_text'])}")
    print(f"Zero vectors:     {len(report['zero_vector'])}")
    print(f"Duplicate IDs:    {report['duplicate_ids']}")
    print(f"Overall status:   {'PASSED' if report['ok'] else 'FAILED'}")
    
    # Assertions
    assert report['total'] == num_opps, f"Total records mismatch: {report['total']} vs {num_opps}"
    assert report['dim_ok'], f"Dimension mismatch. Expected {expected_dim}, got {report['embedding_dim']}"
    assert not report['missing_id'], f"Found records with missing IDs: {report['missing_id']}"
    assert not report['empty_text'], f"Found records with empty searchable texts: {report['empty_text']}"
    assert not report['zero_vector'], f"Found records with zero embedding vectors: {report['zero_vector']}"
    assert not report['duplicate_ids'], f"Found duplicate opportunity IDs: {report['duplicate_ids']}"
    assert report['ok'], "Index validation failed overall."
    
    print("\nAll checks passed successfully!")


if __name__ == "__main__":
    try:
        test_embedding_pipeline()
    except AssertionError as e:
        print(f"\nAssertionError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
