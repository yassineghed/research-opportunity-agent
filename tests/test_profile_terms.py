import unittest
from types import SimpleNamespace

from src.search import extract_profile_terms


class ProfileTermsTests(unittest.TestCase):
    def test_prioritizes_domains_interests_and_keywords(self):
        researcher = SimpleNamespace(
            research_domains=["Marine biology", "Climate science"],
            research_interests=["Ocean ecosystems"],
            keywords=["coral reefs"],
            skills=["Python"],
        )

        self.assertEqual(
            extract_profile_terms(researcher),
            ["Marine biology", "Climate science", "Ocean ecosystems", "coral reefs", "Python"],
        )

    def test_removes_duplicates_and_generic_terms(self):
        researcher = SimpleNamespace(
            research_domains=["Research", "AI", "ai", "  Data   Science  "],
            research_interests=[],
            keywords=["AI"],
            skills=[],
        )

        self.assertEqual(extract_profile_terms(researcher), ["AI", "Data Science"])

    def test_limits_number_of_terms(self):
        researcher = SimpleNamespace(
            research_domains=["one", "two", "three"],
            research_interests=["four"],
            keywords=["five"],
            skills=[],
        )

        self.assertEqual(extract_profile_terms(researcher, max_terms=3), ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()