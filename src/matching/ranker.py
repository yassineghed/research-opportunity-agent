"""Opportunity ranker — retrieval interface.

Delegates to a ``FAISSVectorIndex`` when one is supplied, otherwise falls back
to the original numpy-based cosine similarity.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class OpportunityRanker:
    """Rank opportunities against a researcher embedding.

    Two modes:

    1. **FAISS mode** (preferred) — pass a ``FAISSVectorIndex`` via
       ``vector_index``; the ``index_records`` argument is ignored.
    2. **Numpy fallback** — pass ``index_records`` (legacy behaviour).
    """

    def rank(
        self,
        researcher_vector: np.ndarray,
        index_records: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        vector_index: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return opportunities ranked by cosine similarity.

        Parameters
        ----------
        researcher_vector:
            1-D float32 array produced by the embedder.
        index_records:
            Structured records from ``build_opportunity_index()``.  Used only
            when *vector_index* is ``None``.
        top_k:
            When set, only the top-K results are returned.
        vector_index:
            A ``FAISSVectorIndex`` instance.  When provided it is used for
            the search and *index_records* is ignored.
        """
        if vector_index is not None:
            return vector_index.search(researcher_vector, top_k=top_k or 10)

        if not index_records:
            return []

        return self._numpy_search(researcher_vector, index_records, top_k)

    @staticmethod
    def _numpy_search(
        researcher_vector: np.ndarray,
        index_records: list[dict[str, Any]],
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        matrix = np.stack([rec["vector"] for rec in index_records])

        query_norm = researcher_vector / (np.linalg.norm(researcher_vector) + 1e-10)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_normalized = matrix / matrix_norms

        scores: np.ndarray = matrix_normalized @ query_norm

        results: list[dict[str, Any]] = []
        for record, score in zip(index_records, scores):
            meta = record.get("metadata", {})
            results.append(
                {
                    "opportunity_id": record["opportunity_id"],
                    "score": float(score),
                    "title": meta.get("title", ""),
                    "organization": meta.get("organization", ""),
                    "type": meta.get("type", ""),
                    "deadline": meta.get("deadline", ""),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
