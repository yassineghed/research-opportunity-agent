import io
import json
import time
import zipfile

import requests

from src.collectors.base_collector import BaseCollector


class CordisCollector(BaseCollector):

    BASE_URL = "https://cordis.europa.eu/api"

    def __init__(self, api_key):
        self.api_key = api_key

    @staticmethod
    def normalize_query(query):
        """Normalize a user query before sending it to CORDIS."""
        if query is None:
            raise ValueError("CORDIS query cannot be None.")

        normalized = str(query).strip()

        if not normalized:
            raise ValueError("CORDIS query cannot be empty.")

        normalized = normalized.replace("“", '"').replace("”", '"')

        if normalized.count('"') % 2 == 1:
            # Copy/pasted user inputs sometimes end with an extra quote.
            # Remove the unmatched trailing quote if present; otherwise,
            # drop the unmatched leading quote.
            if normalized.endswith('"'):
                normalized = normalized[:-1]
            elif normalized.startswith('"'):
                normalized = normalized[1:]

        return normalized.strip()

    @staticmethod
    def _get_task_id_from_payload(payload):
        if isinstance(payload, dict):
            for key in ("taskID", "taskId", "id", "extractionId"):
                if key in payload and payload[key] not in (None, ""):
                    return payload[key]

        if isinstance(payload, list):
            for item in payload:
                task_id = CordisCollector._get_task_id_from_payload(item)
                if task_id:
                    return task_id

        return None

    def list_extractions(self):
        """Return the list of active extractions for this profile."""
        endpoints = [
            "getExtractionList",
            "getExtractions",
            "getMyExtractions",
            "listExtractions",
            "getExistingExtractions",
        ]

        for endpoint in endpoints:
            url = f"{self.BASE_URL}/dataextractions/{endpoint}"
            try:
                response = requests.get(
                    url,
                    params={"key": self.api_key, "archived": "false"},
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                payload = data.get("payload") if isinstance(data, dict) else data

                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    for key in ("extractions", "data", "items", "results"):
                        if isinstance(payload.get(key), list):
                            return payload[key]
                if isinstance(data, list):
                    return data
            except (ValueError, TypeError, requests.RequestException):
                pass

        return []

    def delete_extraction(self, task_id):
        """Delete a CORDIS extraction by task ID."""
        if task_id in (None, ""):
            return False

        params = {"key": self.api_key, "taskId": task_id}

        for endpoint in ("deleteExtraction", "removeExtraction"):
            url = f"{self.BASE_URL}/dataextractions/{endpoint}"
            for method in ("get", "post"):
                try:
                    response = requests.request(
                        method,
                        url,
                        params=params if method == "get" else None,
                        data=params if method == "post" else None,
                        headers={"Accept": "application/json"},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        if isinstance(payload, dict):
                            status = payload.get("status")
                            if status is False:
                                return False
                        return True
                except (ValueError, TypeError, requests.RequestException):
                    continue

        return False

    def _clear_oldest_extraction(self):
        extractions = self.list_extractions()
        if not extractions:
            return False

        oldest_task_id = None
        oldest_created_at = None

        for item in extractions:
            if not isinstance(item, dict):
                continue

            task_id = self._get_task_id_from_payload(item)
            if task_id is None:
                continue

            created_at = item.get("createdAt") or item.get("created_at") or item.get("created")
            if created_at is None:
                created_at = "9999-12-31T23:59:59Z"

            if oldest_created_at is None or created_at < oldest_created_at:
                oldest_created_at = created_at
                oldest_task_id = task_id

        if oldest_task_id is None:
            return False

        return self.delete_extraction(oldest_task_id)

    def create_extraction(self, query):
        """
        Create a CORDIS data extraction.

        Returns:
            task_id: ID of the extraction task.
        """

        normalized_query = self.normalize_query(query)
        url = f"{self.BASE_URL}/dataextractions/getExtraction"

        params = {
            "query": normalized_query,
            "key": self.api_key,
            "outputFormat": "json",
            "archived": False
        }

        response = requests.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict) and data.get("status") is False:
            error_message = str(data.get("payload", {}).get("error") or data.get("message") or "")
            if "maximum number of possible extractions" in error_message.lower():
                print("CORDIS extraction limit reached. Deleting the oldest extraction before retrying.")
                if self._clear_oldest_extraction():
                    return self.create_extraction(normalized_query)
                raise RuntimeError(
                    "CORDIS extraction limit reached and no eligible extraction could be deleted. "
                    "Delete an existing extraction manually or wait for it to expire."
                )

        print("Create extraction response:")
        print(data)

        task_id = self._get_task_id_from_payload(data.get("payload", data))
        if task_id is None:
            raise ValueError(f"Unexpected CORDIS extraction response: {data}")

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
            headers={"Accept": "application/json"},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        return data["payload"]

    def download_file(self, file_url):
        """
        Download the generated CORDIS extraction payload.

        CORDIS may return a raw JSON document or an archive containing the
        generated JSON file(s). This method handles both cases.

        Returns:
            Parsed JSON data.
        """

        response = requests.get(
            file_url,
            headers={"Accept": "application/json, application/zip"},
            timeout=60
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        content_type_lower = content_type.lower()
        payload = response.content

        if not payload:
            raise ValueError(
                f"CORDIS download returned an empty response for {file_url}."
            )

        if "text/html" in content_type_lower or payload.lstrip().startswith(b"<"):
            raise ValueError(
                "CORDIS download returned HTML instead of JSON. "
                f"Check that the extraction URL is valid and accessible. "
                f"Response preview: {payload[:500]!r}"
            )

        if "zip" in content_type_lower or payload.startswith(b"PK"):
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    for member_name in archive.namelist():
                        if member_name.lower().endswith((".json", ".geojson", ".txt")):
                            with archive.open(member_name) as member_file:
                                member_data = member_file.read()
                            try:
                                return json.loads(member_data.decode("utf-8"))
                            except UnicodeDecodeError:
                                try:
                                    return json.loads(member_data.decode("utf-8-sig"))
                                except Exception:
                                    continue
                            except json.JSONDecodeError:
                                continue

                    raise ValueError(
                        "CORDIS ZIP download did not contain a readable JSON file. "
                        f"Archive members: {archive.namelist()}"
                    )
            except zipfile.BadZipFile as exc:
                raise ValueError(
                    "CORDIS download returned a ZIP archive that could not be read. "
                    f"Content-Type: {content_type}."
                ) from exc

        text = response.text.strip()
        if not text:
            raise ValueError(
                "CORDIS download returned a binary payload that was not a ZIP or JSON file. "
                f"Content-Type: {content_type}."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ValueError(
                "CORDIS download did not return valid JSON. "
                f"Content-Type: {content_type}. "
                f"Body preview: {text[:500]}"
            ) from exc

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

            if progress == "Finished":

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