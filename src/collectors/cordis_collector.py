import time
import requests

from src.collectors.base_collector import BaseCollector


class CordisCollector(BaseCollector):

    BASE_URL = "https://cordis.europa.eu/api"

    def __init__(self, api_key):
        self.api_key = api_key

    def create_extraction(self, query):
        """
        Create a CORDIS data extraction.

        Returns:
            task_id: ID of the extraction task.
        """

        url = f"{self.BASE_URL}/dataextractions/getExtraction"

        params = {
            "query": query,
            "key": self.api_key,
            "outputFormat": "json",
            "archived": "false"
        }

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        print("Create extraction response:")
        print(data)

        task_id = data["payload"]["taskID"]

        return task_id

    def get_status(self, task_id):
        """
        Get the current status of an extraction.

        Returns:
            payload containing progress and other information.
        """

        url = f"{self.BASE_URL}/dataextractions/getExtractionStatus"

        params = {
            "key": self.api_key,
            "taskId": task_id
        }

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return data["payload"]

    def download_file(self, file_url):
        """
        Download the generated CORDIS JSON file.

        Returns:
            Parsed JSON data.
        """

        response = requests.get(
            file_url,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    def collect(self, query):
        """
        Complete CORDIS collection process.

        1. Create extraction
        2. Wait for completion
        3. Download result

        Returns:
            Raw CORDIS JSON data.
        """

        print("Creating CORDIS extraction...")

        task_id = self.create_extraction(query)

        print(f"Task ID: {task_id}")

        while True:

            status_data = self.get_status(task_id)

            progress = status_data.get("progress")

            print(f"Progress: {progress}")

            if progress == "100%":

                file_url = status_data.get(
                    "destinationFileUri"
                )

                if not file_url:
                    raise ValueError(
                        "Extraction completed but "
                        "no destination file was provided."
                    )

                print("Extraction completed.")
                print("Downloading result...")

                return self.download_file(file_url)

            time.sleep(5)