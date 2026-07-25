from src.matching.similarity import SimilarityMatcher


class OpportunityRanker:

    def __init__(self):
        self.matcher = SimilarityMatcher()


    def rank(
        self,
        researcher_vector,
        opportunity_vectors,
        opportunities
    ):

        results = []

        for opportunity, vector in zip(
            opportunities,
            opportunity_vectors
        ):

            score = self.matcher.compute_similarity(
                researcher_vector,
                vector
            )

            results.append(
                {
                    "opportunity": opportunity,
                    "score": score
                }
            )


        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results