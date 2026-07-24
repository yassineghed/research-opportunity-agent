from .base_builder import ProfileBuilder


class NaturalLanguageProfileBuilder(ProfileBuilder):

    def build(self, researcher):

        text = f"""
{researcher.fullname} is a researcher from {researcher.institution}.

Their research focuses on {', '.join(researcher.research_domains)}.

Their main research interests include {', '.join(researcher.research_interests)}.

They have technical expertise in {', '.join(researcher.skills)}.

Important research keywords are: {', '.join(researcher.keywords)}.

Their previous publications include: {', '.join(researcher.publications)}.
"""

        return text.strip()