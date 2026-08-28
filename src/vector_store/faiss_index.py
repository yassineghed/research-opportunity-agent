"""Numpy-based vector store — a lightweight FAISS-ready index.

Provides the same ``build`` / ``search`` interface as a FAISS flat index but
implemented entirely with numpy so there is no additional dependency.  When the
dataset grows beyond ~100 k vectors, swap the backend by replacing
``_cosine_scores`` with a ``faiss.IndexFlatIP`` lookup — the public API stays
identical.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class VectorIndex:
    """In-memory cosine-similarity index backed by a numpy matrix.

    Usage::

        index = VectorIndex()
        index.build(index_records)          # list from build_opportunity_index()
        results = index.search(query_vec, top_k=10)

    Each result dict has ``opportunity_id``, ``score``, and all ``metadata``
    keys (``title``, ``organization``, ``type``, ``deadline``, …).
    """

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None          # (N, D) normalised
        self._ids: list = []
        self._metadata: list[dict[str, Any]] = []
        self._dim: int | None = None

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, index_records: list[dict[str, Any]]) -> None:
        """Load records and pre-normalise the matrix for fast cosine search.

        Parameters
        ----------
        index_records:
            Output of :func:`~src.embeddings.opportunity_index.build_opportunity_index`.
        """
        if not index_records:
            raise ValueError("Cannot build an index from an empty record list.")

        vectors = np.stack([rec["vector"] for rec in index_records]).astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        self._matrix = vectors / norms              # pre-normalised (N, D)
        self._dim = self._matrix.shape[1]
        self._ids = [rec["opportunity_id"] for rec in index_records]
        self._metadata = [rec.get("metadata", {}) for rec in index_records]

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the top-K most similar opportunities.

        Parameters
        ----------
        query_vector:
            1-D float32 array (researcher embedding).
        top_k:
            Number of results to return.

        Returns
        -------
        list[dict]
            Sorted by ``score`` descending. Each dict has
            ``opportunity_id``, ``score``, and all metadata keys.
        """
        if self._matrix is None:
            raise RuntimeError("Index is empty — call build() first.")

        query = query_vector.astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        scores: np.ndarray = self._matrix @ query_norm   # (N,)

        k = min(top_k, len(self._ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            meta = self._metadata[idx]
            results.append(
                {
                    "opportunity_id": self._ids[idx],
                    "score": float(scores[idx]),
                    **meta,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of opportunities in the index."""
        return len(self._ids)

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality (None if not yet built)."""
        return self._dim
