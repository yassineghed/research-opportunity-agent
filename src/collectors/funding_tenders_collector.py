import requests
import json
from pathlib import Path


class FundingTendersCollector:

    BASE_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

    def __init__(self):
        self.params = {
            "apiKey": "SEDIA",
            "language": "en"
        }

    def search(self, text):
        """Search the Funding & Tenders Portal."""

        params = {
            **self.params,
            "text": text
        }

        print(f"\nSearching Funding & Tenders for: {text}")

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30
        )

        print("Status:", response.status_code)

        response.raise_for_status()

        data = response.json()

        print("Total search results:", data.get("totalResults"))
        print("Results returned:", len(data.get("results", [])))

        return data

    def is_opportunity(self, result):
        """Check whether a SEDIA result is an actual funding opportunity."""

        metadata = result.get("metadata", {})

        database = result.get("database")

        result_type = metadata.get("type", [])

        identifier = metadata.get("identifier", [])

        title = metadata.get("title", [])

        return (
            database == "SEDIA"
            and "1" in result_type
            and len(identifier) > 0
            and len(title) > 0
        )

    def parse_opportunity(self, result):
        """Convert a raw SEDIA result into our Opportunity structure."""

        metadata = result.get("metadata", {})

        def first(field, default=None):
            values = metadata.get(field, [])
            return values[0] if values else default

        return {
            "id": first("identifier"),
            "title": first("title"),
            "type": first("typesOfAction"),
            "organization": "European Commission",
            "description": first("descriptionByte"),
            "keywords": metadata.get("keywords", []),
            "topics": metadata.get("tags", []),
            "eligibility": first("topicConditions"),
            "deadline": first("deadlineDate"),
            "url": first("url")
        }

    def collect(self, text):
        """Search and extract actual opportunities."""

        data = self.search(text)

        results = data.get("results", [])

        opportunities = []

        for result in results:

            if self.is_opportunity(result):

                opportunity = self.parse_opportunity(result)

                opportunities.append(opportunity)

        print(f"Actual opportunities found: {len(opportunities)}")

        return opportunities

    def save(self, opportunities, output_path):
        """Save opportunities to JSON."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                opportunities,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Saved {len(opportunities)} opportunities to:")
        print(output_path)


if __name__ == "__main__":

    collector = FundingTendersCollector()

    opportunities = collector.collect(
        text="biodiversity"
    )

    collector.save(
        opportunities,
        "data/processed/opportunities.json"
    )

    print("\nFIRST OPPORTUNITY:")

    if opportunities:
        print(
            json.dumps(
                opportunities[0],
                indent=2,
                ensure_ascii=False
            )
        )