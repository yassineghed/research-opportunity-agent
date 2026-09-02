"""FAISS-backed vector store for scalable similarity search.

Replaces the numpy-only ``VectorIndex`` when ``faiss-cpu`` (or ``faiss-gpu``) is
available.  Provides the same ``search`` interface so it can be used
interchangeably by ``OpportunityRanker`` and ``RecommendationAgent``.

Persisted files:

- ``opportunities.faiss`` — the FAISS binary index
- ``metadata.json`` — ``ids`` and ``metadata`` lists in parallel order

Usage::

    index = FAISSVectorIndex(dimension=384)
    index.build(index_records)
    results = index.search(query_vec, top_k=10)
    index.save("vector_store/index")

    restored = FAISSVectorIndex.load("vector_store/index")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorIndex:
    """Cosine-similarity index backed by ``faiss.IndexFlatIP``.

    Vectors are L2-normalised before insertion so that inner product equals
    cosine similarity.
    """

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str | int] = []
        self._metadata: list[dict[str, Any]] = []
        self._built = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, index_records: list[dict[str, Any]]) -> None:
        """Populate the index from ``build_opportunity_index()`` output."""
        if not index_records:
            raise ValueError("Cannot build an index from an empty record list.")

        vectors = np.stack([rec["vector"] for rec in index_records]).astype(np.float32)
        self._dimension = vectors.shape[1]

        faiss.normalize_L2(vectors)

        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)

        self._ids = [rec["opportunity_id"] for rec in index_records]
        self._metadata = [rec.get("metadata", {}) for rec in index_records]
        self._built = True

        logger.info(
            "FAISS index built: %d vectors, dim=%d",
            self._index.ntotal,
            self._dimension,
        )

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
        if not self._built or self._index is None:
            raise RuntimeError("Index is empty — call build() first.")

        query = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "opportunity_id": self._ids[idx],
                    "score": float(score),
                    **meta,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write the FAISS index and metadata to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss_path = directory / "opportunities.faiss"
        meta_path = directory / "metadata.json"

        faiss.write_index(self._index, str(faiss_path))

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"ids": self._ids, "metadata": self._metadata},
                fh,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("Index saved to %s (%d vectors)", directory, self._index.ntotal)

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorIndex":
        """Restore a previously-saved index from *directory*."""
        directory = Path(directory)
        faiss_path = directory / "opportunities.faiss"
        meta_path = directory / "metadata.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        index = faiss.read_index(str(faiss_path))

        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        instance = cls(dimension=index.d)
        instance._index = index
        instance._ids = data["ids"]
        instance._metadata = data["metadata"]
        instance._built = True

        logger.info("Index loaded from %s (%d vectors, dim=%d)", directory, index.ntotal, index.d)
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def dimension(self) -> int | None:
        return self._dimension
