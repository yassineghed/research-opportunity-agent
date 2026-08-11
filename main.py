from src.matching.ranker import OpportunityRanker
from src.matching.similarity import SimilarityMatcher
from src.embeddings.embedder import Embedder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.builders.profil.natural_profil_builder import NaturalLanguageProfileBuilder

from src.models.opportunity import Opportunity

from src.builders.opportunity.structured_opportunity_builder import (
    StructuredOpportunityBuilder
)

from src.builders.opportunity.natural_opportunity_builder import (
    NaturalLanguageOpportunityBuilder
)

from dotenv import load_dotenv

load_dotenv()

# LLM imports
from src.llm.client import LLMClient
from src.reranking.llm_reranker import LLMReranker

from src.loaders import DataLoader

researchers = DataLoader.load_researchers(
    "data/mock/researchers.json"
)

opportunities = DataLoader.load_opportunities(
    "data/mock/opportunities.json"
)

llm_client = LLMClient()

reranker = LLMReranker(
    llm_client
)


researcher = researchers[0]

candidate_opportunities = opportunities[:5]


results = reranker.rerank(
    researcher,
    candidate_opportunities
)


print(results)

"""
structured_builder = StructuredProfileBuilder()
researcher_texts = []

for researcher in researchers:
    text = structured_builder.build(researcher)
    researcher_texts.append(text)

mini = Embedder(
    "all-MiniLM-L6-v2"
)

bge = Embedder(
    "BAAI/bge-small-en-v1.5"
)
researcher_vector = mini.encode_batch(researcher_texts)


opportunity_builder = StructuredOpportunityBuilder()


opportunity_texts = []

for opportunity in opportunities:
    text = opportunity_builder.build(opportunity)
    opportunity_texts.append(text)


opp_vector = mini.encode_batch(opportunity_texts)



ranker = OpportunityRanker()

"""
"""

recommendations = ranker.rank(
    researcher_vector[0],
    opp_vector,
    opportunities
)
print(researcher_vector.shape)
print(opp_vector.shape)

for rec in recommendations[:5]:
    print(rec)"""

"""

#### testing all the researchers 
for i, researcher in enumerate(researchers):

    recommendations = ranker.rank(
        researcher_vector[i],
        opp_vector,
        opportunities
    )

    print("\n")
    print(researcher.fullname)


    for rec in recommendations[:3]:
        print(
            rec["title"],
            "→",
            round(rec["score"],3)

        )
"""
"""
print("===== STRUCTURED =====")

print(
    structured_builder.build(researcher)
)


print("\n===== NATURAL LANGUAGE =====")

print(
    natural_builder.build(researcher)
)

builder = StructuredOpportunityBuilder()
natural_builder = NaturalLanguageOpportunityBuilder()

print(natural_builder.build(opportunity))
embedder = Embedder()

vector = embedder.encode(natural_builder.build(opportunity))
print(vector.shape)
print(type(vector))
print(vector[:5]) """