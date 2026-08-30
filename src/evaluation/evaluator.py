"""Pipeline evaluator for executing benchmarks and producing metrics."""
import time
from typing import Any

from src.evaluation.benchmark import EvalDataset
from src.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k, rank_correlation


class PipelineEvaluator:
    def __init__(
        self,
        embedder,
        ranker,
        profile_builder,
        llm_rerankers: list[tuple[str, Any]],
        top_k_retrieval: int = 5,
        top_k_final: int = 3,
    ):
        self.embedder = embedder
        self.ranker = ranker
        self.profile_builder = profile_builder
        self.llm_rerankers = llm_rerankers
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final

    def evaluate(
        self,
        dataset: EvalDataset,
        researchers_by_id: dict,
        opportunities_by_id: dict,
        index_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the evaluation benchmark on the pipeline."""
        
        results = {
            "retrieval": {"recall": [], "mrr": [], "ndcg": []},
            "rerankers": {name: {"recall": [], "mrr": [], "ndcg": [], "correlation": [], "errors": 0} for name, _ in self.llm_rerankers}
        }

        print(f"Starting evaluation on {len(dataset.queries)} queries...", flush=True)

        for i, query in enumerate(dataset.queries, start=1):
            researcher = researchers_by_id.get(query.researcher_id)
            if not researcher:
                print(f"Skipping query {query.query_id}: Researcher {query.researcher_id} not found.")
                continue

            relevant_ids = set(query.relevant_opportunity_ids)
            if not relevant_ids:
                print(f"Skipping query {query.query_id}: No relevant opportunity IDs specified.")
                continue
                
            print(f"Evaluating [{i}/{len(dataset.queries)}]: {researcher.fullname}")

            # 1. Retrieval Stage
            researcher_text = self.profile_builder.build(researcher)
            researcher_vector = self.embedder.encode(researcher_text)

            retrieval_results = self.ranker.rank(
                researcher_vector,
                index_records,
                top_k=self.top_k_retrieval,
            )
            retrieved_ids = [item["opportunity_id"] for item in retrieval_results]

            # Calculate Retrieval Metrics (at retrieval top_k)
            results["retrieval"]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, self.top_k_retrieval))
            results["retrieval"]["mrr"].append(mrr_at_k(retrieved_ids, relevant_ids, self.top_k_retrieval))
            results["retrieval"]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant_ids, self.top_k_retrieval))

            candidate_opportunities = [opportunities_by_id[op_id] for op_id in retrieved_ids if op_id in opportunities_by_id]

            # 2. Reranking Stage
            for provider_name, reranker in self.llm_rerankers:
                try:
                    reranked = reranker.rerank(researcher, candidate_opportunities)
                    recommendations = reranked.get("recommendations", [])
                    recommendations = sorted(
                        recommendations,
                        key=lambda item: item.get("score", 0),
                        reverse=True,
                    )[:self.top_k_final]
                    
                    reranked_ids = [item["opportunity_id"] for item in recommendations]
                    
                    # If LLM returns fewer items, pad with remainder from retrieved for metric fairness
                    if len(reranked_ids) < self.top_k_final:
                        for op_id in retrieved_ids:
                            if op_id not in reranked_ids:
                                reranked_ids.append(op_id)
                            if len(reranked_ids) >= self.top_k_final:
                                break

                    results["rerankers"][provider_name]["recall"].append(recall_at_k(reranked_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["mrr"].append(mrr_at_k(reranked_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["ndcg"].append(ndcg_at_k(reranked_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["correlation"].append(rank_correlation(retrieved_ids, reranked_ids))
                
                except Exception as e:
                    print(f"Error in {provider_name} reranker: {e}")
                    results["rerankers"][provider_name]["errors"] += 1
                    
                    # Fallback metric (assume baseline retrieval ordering is preserved as fallback)
                    fallback_ids = retrieved_ids[:self.top_k_final]
                    results["rerankers"][provider_name]["recall"].append(recall_at_k(fallback_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["mrr"].append(mrr_at_k(fallback_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["ndcg"].append(ndcg_at_k(fallback_ids, relevant_ids, self.top_k_final))
                    results["rerankers"][provider_name]["correlation"].append(1.0)  # Perfect correlation with itself on fallback
                    
            # Adding a small sleep to avoid LLM rate limits during evaluation
            time.sleep(2)

        return self._aggregate_results(results)

    def _aggregate_results(self, raw_results: dict) -> dict:
        """Calculate averages for all metrics."""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        agg = {
            "retrieval": {
                f"recall@{self.top_k_retrieval}": avg(raw_results["retrieval"]["recall"]),
                f"mrr@{self.top_k_retrieval}": avg(raw_results["retrieval"]["mrr"]),
                f"ndcg@{self.top_k_retrieval}": avg(raw_results["retrieval"]["ndcg"]),
            },
            "rerankers": {}
        }

        for provider, metrics in raw_results["rerankers"].items():
            agg["rerankers"][provider] = {
                f"recall@{self.top_k_final}": avg(metrics["recall"]),
                f"mrr@{self.top_k_final}": avg(metrics["mrr"]),
                f"ndcg@{self.top_k_final}": avg(metrics["ndcg"]),
                "rank_correlation": avg(metrics["correlation"]),
                "errors": metrics["errors"],
            }
            
        return agg
