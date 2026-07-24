from .base_opportunity_builder import OpportunityBuilder


class StructuredOpportunityBuilder(OpportunityBuilder):

    def build(self, opportunity):

        text = f"""
Opportunity Title:
{opportunity.title}

Opportunity Type:
{opportunity.type}

Organization:
{opportunity.organization}

Description:
{opportunity.description}

Keywords:
{', '.join(opportunity.keywords)}

Research Topics:
{', '.join(opportunity.topics)}

Eligibility:
{opportunity.eligibility}

Deadline:
{opportunity.deadline}
"""

        return text.strip()