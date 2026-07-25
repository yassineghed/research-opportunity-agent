from src.embeddings.embedder import Embedder
from src.builders.profil.structured_profil_builder import StructuredProfileBuilder
from src.builders.profil.natural_profil_builder import NaturalLanguageProfileBuilder


"""researcher = Researcher(
    fullname="Yassine Ghedira",
    institution="ISIMM",
    research_domains=[
        "Artificial Intelligence",
        "Computer Vision"
    ],
    research_interests=[
        "Marine Conservation",
        "Underwater Species Monitoring"
    ],
    skills=[
        "Python",
        "Deep Learning",
        "YOLO",
        "Machine Learning"
    ],
    keywords=[
        "Object Detection",
        "Biodiversity",
        "Remote Sensing"
    ],
    publications=[
        "Deep Learning approaches for fish species detection"
    ]
)


structured_builder = StructuredProfileBuilder()

natural_builder = NaturalLanguageProfileBuilder()


print("===== STRUCTURED =====")

print(
    structured_builder.build(researcher)
)


print("\n===== NATURAL LANGUAGE =====")

print(
    natural_builder.build(researcher)
)"""
from src.models.opportunity import Opportunity

from src.builders.opportunity.structured_opportunity_builder import (
    StructuredOpportunityBuilder
)

from src.builders.opportunity.natural_opportunity_builder import (
    NaturalLanguageOpportunityBuilder
)


opportunity = Opportunity(
    title="AI for Ocean Monitoring Research Grant",
    type="Funding Opportunity",
    organization="European Marine Research Foundation",
    description="Grant supporting AI and computer vision for marine biodiversity monitoring.",
    keywords=[
        "Artificial Intelligence",
        "Computer Vision",
        "Marine Biology"
    ],
    topics=[
        "Ocean Monitoring",
        "Species Detection"
    ],
    eligibility="Researchers and universities",
    deadline="2026-12-15"
)


builder = StructuredOpportunityBuilder()
natural_builder = NaturalLanguageOpportunityBuilder()



print(natural_builder.build(opportunity))
"""embedder = Embedder()

vector = embedder.encode(natural_builder.build(opportunity))
print(vector.shape)
print(type(vector))
print(vector[:5])"""