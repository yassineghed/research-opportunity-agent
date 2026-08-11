import json


class LLMReranker:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def rerank(self, researcher, opportunities):

        prompt = self._build_prompt(
            researcher,
            opportunities
        )

        response = self.llm_client.generate(prompt)

        return self._parse_response(response)

    def _build_prompt(self, researcher, opportunities):

        opportunities_text = ""

        for opportunity in opportunities:

            opportunities_text += f"""
Opportunity ID: {opportunity.id}
Title: {opportunity.title}
Type: {opportunity.type}
Organization: {opportunity.organization}
Description: {opportunity.description}
Keywords: {", ".join(opportunity.keywords)}
Topics: {", ".join(opportunity.topics)}
Eligibility: {opportunity.eligibility}
Deadline: {opportunity.deadline}
-------------------------
"""

        prompt = f"""
You are a scientific opportunity recommendation system.

Evaluate how relevant each opportunity is for the researcher.

Researcher:
Name: {researcher.fullname}
Institution: {researcher.institution}

Research domains:
{", ".join(researcher.research_domains)}

Research interests:
{", ".join(researcher.research_interests)}

Skills:
{", ".join(researcher.skills)}

Keywords:
{", ".join(researcher.keywords)}

Publications:
{", ".join(researcher.publications)}

Candidate opportunities:
{opportunities_text}

For each opportunity:

1. Give a relevance score from 0 to 100.
2. Explain briefly why it matches.
3. Identify the main matching areas.

Return ONLY valid JSON using this structure:

{{
    "recommendations": [
        {{
            "opportunity_id": 1,
            "score": 85,
            "reason": "Short explanation",
            "matching_areas": [
                "Artificial Intelligence",
                "Computer Vision"
            ]
        }}
    ]
}}

Sort the recommendations from highest score to lowest score.
"""

        return prompt

    def _parse_response(self, response):

        try:
            return json.loads(response)

        except json.JSONDecodeError:

            raise ValueError(
                "The LLM did not return valid JSON."
            )