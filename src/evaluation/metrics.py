"""Evaluation metrics for information retrieval and reranking."""
import math
from scipy.stats import spearmanr


def recall_at_k(retrieved_ids: list[str | int], relevant_ids: set[str | int], k: int) -> float:
    """Calculate Recall@K: proportion of relevant items found in top K."""
    if not relevant_ids:
        return 1.0

    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for item_id in retrieved_k if item_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[str | int], relevant_ids: set[str | int], k: int) -> float:
    """Calculate Mean Reciprocal Rank (MRR) at K."""
    retrieved_k = retrieved_ids[:k]
    for i, item_id in enumerate(retrieved_k):
        if item_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved_ids: list[str | int], relevant_ids: set[str | int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K."""
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for i, item_id in enumerate(retrieved_k):
        if item_id in relevant_ids:
            # Assuming binary relevance (1 or 0) for now
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: list[str | int], relevant_ids: set[str | int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K."""
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    
    # Calculate IDCG (Ideal DCG)
    ideal_retrieved = list(relevant_ids)[:k]
    idcg = dcg_at_k(ideal_retrieved, relevant_ids, k)
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def rank_correlation(baseline_ids: list[str | int], reranked_ids: list[str | int]) -> float:
    """Calculate Spearman rank correlation between two lists.
    Returns 1.0 for identical rankings, -1.0 for reversed, 0.0 for uncorrelated.
    """
    # Create rank mappings
    baseline_ranks = {item_id: rank for rank, item_id in enumerate(baseline_ids)}
    
    # Only compare items present in both lists (typically reranker only reranks top K)
    common_items = [item_id for item_id in reranked_ids if item_id in baseline_ranks]
    
    if len(common_items) < 2:
        return 0.0
        
    reranked_positions = list(range(len(common_items)))
    baseline_positions = [baseline_ranks[item_id] for item_id in common_items]
    
    correlation, _ = spearmanr(reranked_positions, baseline_positions)
    # Handle NaN when all values are identical
    if math.isnan(correlation):
        return 1.0
    return float(correlation)
