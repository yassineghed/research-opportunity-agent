from src.collectors.base_collector import BaseCollector
from src.models.opportunity import Opportunity
import requests


class CordisCollector:

    BASE_URL = "YOUR_CORDIS_ENDPOINT"


    def collect(self):

        response = requests.get(
            self.BASE_URL
        )

        if response.status_code != 200:
            raise Exception(
                "Failed to fetch CORDIS data"
            )

        data = response.json()

        opportunities = []


        for index, item in enumerate(data["projects"]):

            opportunity = self.parse_opportunity(
                item,
                index
            )

            opportunities.append(
                opportunity
            )


        return opportunities



    def parse_opportunity(self, item, index):

        return Opportunity(
            id=index,
            title=item.get("title", ""),
            type="Grant",
            organization=item.get(
                "programme",
                ""
            ),
            description=item.get(
                "objective",
                ""
            ),
            keywords=[],
            topics=[],
            eligibility="Researchers",
            deadline=item.get(
                "endDate",
                ""
            ),
            url=item.get(
                "url",
                ""
            )
        )