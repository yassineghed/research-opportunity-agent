"""Opportunity embedding index.

Converts a list of :class:`~src.models.opportunity.Opportunity` objects into
structured index records that couple each opportunity's vector with its ID and
searchable text so downstream retrieval code never has to re-join by position.

Each record is a plain dict::

    {
        "opportunity_id": "DIGITAL-2026-AI-...",
        "searchable_text":  "<full structured text used for embedding>",
        "vector":           np.ndarray(shape=(D,), dtype=float32),
        "metadata": {
            "title":        str,
            "type":         str,
            "organization": str,
            "deadline":     str,
            "status":       str,
            "url":          str,
            "source":       str,
        },
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.embeddings.embedder import Embedder
from src.models.opportunity import Opportunity


def build_opportunity_index(
    opportunities: list[Opportunity],
    builder,
    embedder: Embedder,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Embed every opportunity and return a list of structured index records.

    Parameters
    ----------
    opportunities:
        Opportunity objects loaded from ``data/processed/opportunities.json``.
    builder:
        Any ``OpportunityBuilder`` subclass (e.g. ``StructuredOpportunityBuilder``).
        Its ``build()`` method produces the ``searchable_text`` for each record.
    embedder:
        An :class:`~src.embeddings.embedder.Embedder` instance.
    show_progress:
        Whether to display a ``tqdm`` progress bar during batch encoding.

    Returns
    -------
    list[dict]
        One record per opportunity, in the same order as *opportunities*.
        Each record contains ``opportunity_id``, ``searchable_text``,
        ``vector`` (float32 ndarray), and ``metadata`` dict.
    """
    if not opportunities:
        return []

    # Build searchable texts for all opportunities in one pass.
    searchable_texts: list[str] = [builder.build(opp) for opp in opportunities]

    # Batch-encode — returns a 2-D matrix of shape (N, D).
    vectors: np.ndarray = embedder.encode_batch(
        searchable_texts,
        show_progress=show_progress,
    )

    records: list[dict[str, Any]] = []
    for opp, text, vec in zip(opportunities, searchable_texts, vectors):
        records.append(
            {
                "opportunity_id": opp.id,
                "searchable_text": text,
                "vector": vec,
                "metadata": {
                    "title": opp.title,
                    "type": opp.type,
                    "organization": opp.organization,
                    "deadline": opp.deadline,
                    "status": opp.status,
                    "url": opp.url,
                    "source": opp.source,
                },
            }
        )

    return records


def validate_index(
    records: list[dict[str, Any]],
    expected_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the index and return a summary report.

    Parameters
    ----------
    records:
        Output of :func:`build_opportunity_index`.
    expected_dim:
        When provided, every vector's shape is checked against this value.

    Returns
    -------
    dict with keys:
        ``total``, ``embedding_dim``, ``missing_id``, ``empty_text``,
        ``zero_vector``, ``duplicate_ids``, ``ok``.
    """
    total = len(records)
    embedding_dim: int | None = None
    missing_id: list[int] = []
    empty_text: list[int] = []
    zero_vector: list[int] = []
    seen_ids: dict = {}
    duplicate_ids: list = []

    for i, record in enumerate(records):
        oid = record.get("opportunity_id")
        text = record.get("searchable_text", "")
        vec: np.ndarray = record.get("vector")

        # ID checks
        if not oid:
            missing_id.append(i)
        else:
            if oid in seen_ids:
                duplicate_ids.append(oid)
            seen_ids[oid] = i

        # Text check
        if not text or not text.strip():
            empty_text.append(i)

        # Vector checks
        if vec is not None:
            if embedding_dim is None:
                embedding_dim = vec.shape[0]
            if np.allclose(vec, 0):
                zero_vector.append(i)

    dim_ok = True
    if expected_dim is not None and embedding_dim is not None:
        dim_ok = embedding_dim == expected_dim

    ok = (
        total > 0
        and not missing_id
        and not empty_text
        and not zero_vector
        and not duplicate_ids
        and dim_ok
    )

    return {
        "total": total,
        "embedding_dim": embedding_dim,
        "missing_id": missing_id,
        "empty_text": empty_text,
        "zero_vector": zero_vector,
        "duplicate_ids": duplicate_ids,
        "dim_ok": dim_ok,
        "ok": ok,
    }
