import requests
import time

from src.models.opportunity import Opportunity
from src.collectors.base_collector import BaseCollector


class CordisCollector(BaseCollector):

    BASE_URL = (
        "https://cordis.europa.eu/api"
    )


    def __init__(self, api_key):
        self.api_key = api_key


    def collect(self, query):
        task_id = self.create_extraction(query)
        file_url = self.wait_for_extraction(task_id)
        data = self.download_result(file_url)
        return self.parse_opportunities(data)

    def create_extraction(self, query):
        url = (
            f"{self.BASE_URL}"
            "/dataextractions/getExtraction"
        )
        params = {
            "query": query,
            "key": self.api_key,
            "outputFormat": "json",
            "archived": "false"
        }
        response = requests.get(
            url,
            params=params
        )
        response.raise_for_status()
        data = response.json()
        task_id = data["payload"]["taskID"]
        return task_id

    def wait_for_extraction(self, task_id):
        url = (
            f"{self.BASE_URL}"
            "/dataextractions/getExtractionStatus"
        )
        while True:
            params = {
                "key": self.api_key,
                "taskId": task_id
            }
            response = requests.get(
                url,
                params=params
            )
            data = response.json()
            status = data["payload"]["progress"]
            print(
                "Progress:",
                status
            )
            if status == "100%":
                return data["payload"]["destinationFileUri"]
            time.sleep(5)