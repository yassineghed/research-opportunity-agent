import requests

from src.collectors.base_collector import BaseCollector
from src.models.opportunity import Opportunity


class CordisCollector(BaseCollector):

    def __init__(self):
        self.api_url = "YOUR_CORDIS_API_URL"


    def collect(self):

        response = requests.get(
            self.api_url,
            timeout=10
        )

        if response.status_code != 200:
            raise Exception(
                f"CORDIS API error: {response.status_code}"
            )

        data = response.json()

        opportunities = []

        for index, item in enumerate(data):

            opportunity = self.parse_opportunity(
                item,
                index
            )

            opportunities.append(opportunity)

        return opportunities


    def parse_opportunity(self, item, index):

        return Opportunity(
            id=index,
            title=item.get("title", ""),
            type="Grant",
            organization=item.get(
                "organization",
                ""
            ),
            description=item.get(
                "description",
                ""
            ),
            keywords=[],
            topics=[],
            eligibility="Researchers",
            deadline=item.get(
                "deadline",
                ""
            ),
            url=item.get(
                "url",
                ""
            )
        )