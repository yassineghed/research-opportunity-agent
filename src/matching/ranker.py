"""Opportunity ranker — batched cosine similarity retrieval.

Uses a single numpy matrix multiply instead of N individual cosine calls,
which is significantly faster for large opportunity sets.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class OpportunityRanker:
    """Rank opportunities against a researcher vector using cosine similarity.

    Accepts the structured index records produced by
    :func:`~src.embeddings.opportunity_index.build_opportunity_index` so the
    ranker is fully decoupled from how opportunities were loaded or built.
    """

    def rank(
        self,
        researcher_vector: np.ndarray,
        index_records: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return opportunities ranked by cosine similarity.

        Parameters
        ----------
        researcher_vector:
            1-D float32 array produced by the embedder.
        index_records:
            List of structured records from ``build_opportunity_index()``.
            Each record must have ``opportunity_id``, ``vector``, and
            ``metadata`` keys.
        top_k:
            When set, only the top-K results are returned.

        Returns
        -------
        list[dict]
            Records sorted by ``score`` (descending). Each result dict
            contains ``opportunity_id``, ``score``, ``title``,
            ``organization``, ``type``, ``deadline``.
        """
        if not index_records:
            return []

        # Stack all opportunity vectors into a matrix (N × D).
        matrix = np.stack([rec["vector"] for rec in index_records])  # (N, D)

        # Normalise both sides to get cosine similarity via dot product.
        query_norm = researcher_vector / (np.linalg.norm(researcher_vector) + 1e-10)
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_normalized = matrix / matrix_norms

        scores: np.ndarray = matrix_normalized @ query_norm  # (N,)

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