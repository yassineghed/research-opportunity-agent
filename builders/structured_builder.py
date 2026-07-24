from .base_builder import ProfileBuilder


class StructuredProfileBuilder(ProfileBuilder):

    def build(self, researcher):

        text = f"""
Research Domain:
{', '.join(researcher.research_domains)}

Research Interests:
{', '.join(researcher.research_interests)}

Skills:
{', '.join(researcher.skills)}

Keywords:
{', '.join(researcher.keywords)}

Publications:
{', '.join(researcher.publications)}
"""

        return text.strip()