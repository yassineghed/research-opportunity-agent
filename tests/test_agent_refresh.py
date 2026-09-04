import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.recommendation_agent import RecommendationAgent
from src.models.opportunity import Opportunity


class AgentRefreshTests(unittest.TestCase):
    def test_refresh_adds_new_opportunities_and_rebuilds_index(self):
        existing = Opportunity(
            id="OLD", title="Old", type="grant", organization="Org",
            description="", keywords=[], topics=[], eligibility="", deadline="",
        )
        fetched = Opportunity(
            id="NEW", title="New", type="grant", organization="Org",
            description="", keywords=[], topics=[], eligibility="", deadline="",
        )
        agent = RecommendationAgent.__new__(RecommendationAgent)
        agent.opportunities = [existing]
        agent.opportunities_by_id = {existing.id: existing}
        agent.config = SimpleNamespace(index_persist_path=Path("vector_store/test-index"))

        with patch("src.agent.recommendation_agent.FundingTendersCollector") as collector_type:
            collector_type.return_value.search_for_profile.return_value = [fetched]
            with patch.object(agent, "build_index") as build_index:
                summary = agent.refresh_for_profile(SimpleNamespace(
                    research_domains=["biology"], research_interests=[], keywords=[], skills=[]
                ))

        self.assertEqual(summary["added"], 1)
        self.assertTrue(summary["index_refreshed"])
        self.assertEqual([item.id for item in agent.opportunities], ["OLD", "NEW"])
        build_index.assert_called_once()


if __name__ == "__main__":
    unittest.main()