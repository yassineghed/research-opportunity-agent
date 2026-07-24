from .base_opportunity_builder import OpportunityBuilder


class NaturalLanguageOpportunityBuilder(OpportunityBuilder):

    def build(self, opportunity):

        text = f"""
{opportunity.title} is a {opportunity.type} 
provided by {opportunity.organization}.

This opportunity focuses on: {opportunity.description}

The main research areas include: {', '.join(opportunity.topics)}.

Important keywords related to this opportunity are: {', '.join(opportunity.keywords)}.

The opportunity is available for: {opportunity.eligibility}

The application deadline is: {opportunity.deadline}.
"""

        return text.strip()