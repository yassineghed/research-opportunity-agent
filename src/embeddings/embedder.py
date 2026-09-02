from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around SentenceTransformer for text embedding.

    Supports single and batch encoding. All outputs are ``float32`` numpy
    arrays so they are immediately compatible with numpy cosine similarity
    and FAISS-style indices.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""
        return self.model.get_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a float32 vector."""
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode a list of texts into a 2-D float32 matrix (N × D)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)