from models.researcher import Researcher
from builders.structured_builder import StructuredProfileBuilder
from builders.natural_builder import NaturalLanguageProfileBuilder


researcher = Researcher(
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
)