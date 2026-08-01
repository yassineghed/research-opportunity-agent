import json

from src.models.opportunity import Opportunity
from src.collectors.base_collector import BaseCollector


class CordisCollector(BaseCollector):

    def __init__(self, file_path):
        self.file_path = file_path


    def collect(self):

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)


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