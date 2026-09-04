import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.collectors.funding_tenders_collector import FundingTendersCollector


class ProfileSearchTests(unittest.TestCase):
    def test_searches_profile_terms_and_deduplicates_results(self):
        researcher = SimpleNamespace(
            research_domains=["Marine biology"],
            research_interests=["Ocean ecosystems"],
            keywords=[],
            skills=[],
        )
        result = {
            "metadata": {
                "identifier": "OPP-1",
                "title": "Ocean research",
                "status": FundingTendersCollector.STATUS_OPEN,
                "organisation": "European Commission",
                "descriptionByte": "Marine research",
                "deadlineDate": "2026-12-01",
            }
        }
        collector = FundingTendersCollector()

        with patch.object(collector, "search", return_value={"results": [result]}):
            opportunities = collector.search_for_profile(researcher, max_workers=2)

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].id, "OPP-1")
        self.assertEqual(opportunities[0].title, "Ocean research")

    def test_empty_profile_does_not_call_api(self):
        researcher = SimpleNamespace(
            research_domains=[], research_interests=[], keywords=[], skills=[]
        )
        collector = FundingTendersCollector()

        with patch.object(collector, "search") as search:
            self.assertEqual(collector.search_for_profile(researcher), [])

        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()