from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityMatcher:

    def compute_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:

        score = cosine_similarity(
            [vector1],
            [vector2]
        )[0][0]

        return float(score)