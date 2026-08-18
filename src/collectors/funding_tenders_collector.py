import requests
import json
import re
from datetime import datetime
from html import unescape
from typing import List, Dict, Any, Optional


class FundingTendersCollector:
    """
    Collector for the European Commission Funding & Tenders Portal
    using the SEDIA Search API.

    API:
    https://api.tech.ec.europa.eu/search-api/prod/rest/search
    """

    BASE_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
    API_KEY = "SEDIA"

    # Funding & Tenders opportunity types
    TYPE_TOPIC = "1"

    # Opportunity statuses
    STATUS_FORTHCOMING = "31094501"
    STATUS_OPEN = "31094502"

    ALLOWED_STATUSES = {
        STATUS_FORTHCOMING,
        STATUS_OPEN
    }

    def __init__(
        self,
        api_key: str = API_KEY,
        language: str = "en"
    ):
        self.api_key = api_key
        self.language = language

    # ============================================================
    # HTTP REQUEST
    # ============================================================

    def search(
        self,
        text: str,
        page_number: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Search the Funding & Tenders Portal.

        IMPORTANT:
        The API requires POST for the query body.
        GET requests return HTTP 405.
        """

        params = {
            "apiKey": self.api_key,
            "text": text,
            "language": self.language
        }

        # Elasticsearch-style query understood by SEDIA
        query = {
            "bool": {
                "must": [
                    {
                        "terms": {
                            "type": [
                                self.TYPE_TOPIC
                            ]
                        }
                    },
                    {
                        "terms": {
                            "status": [
                                self.STATUS_FORTHCOMING,
                                self.STATUS_OPEN
                            ]
                        }
                    }
                ]
            }
        }

        body = {
            "query": json.dumps(query),
            "pageNumber": page_number,
            "pageSize": page_size
        }

        print("\n" + "=" * 70)
        print("FUNDING & TENDERS API REQUEST")
        print("=" * 70)

        print(f"Search text : {text}")
        print(f"Page        : {page_number}")
        print(f"Page size   : {page_size}")
        print(f"URL         : {self.BASE_URL}")

        query = {
            "bool": {
                "must": [
                    {
                        "terms": {
                            "type": ["1"]
                        }
                    },
                    {
                        "terms": {
                            "status": [
                                "31094501",
                                "31094502"
                            ]
                        }
                    }
                ]
            }
        }

        languages = ["en"]

        sort = {
            "field": "sortStatus",
            "order": "ASC"
        }

        response = requests.post(
            self.BASE_URL,
            params=params,
            files={
                "query": (
                    "blob",
                    json.dumps(query),
                    "application/json"
                ),
                "languages": (
                    "blob",
                    json.dumps(languages),
                    "application/json"
                ),
                "sort": (
                    "blob",
                    json.dumps(sort),
                    "application/json"
                )
            },
            timeout=30
        )

        print(f"Status      : {response.status_code}")

        response.raise_for_status()

        return response.json()

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    @staticmethod
    def clean_html(text: Optional[str]) -> str:
        """
        Remove HTML tags and decode HTML entities.
        """

        if not text:
            return ""

        text = unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ============================================================
    # METADATA HELPER
    # ============================================================

    @staticmethod
    def get_metadata(
        metadata: Dict[str, Any],
        field: str,
        default: Any = None
    ) -> Any:
        """
        SEDIA metadata fields are usually stored as lists.

        Example:
            "title": ["My opportunity"]

        This helper returns:
            "My opportunity"
        """

        value = metadata.get(field, default)

        if isinstance(value, list):

            if not value:
                return default

            return value[0]

        return value

    # ============================================================
    # DATE PARSER
    # ============================================================

    @staticmethod
    def parse_date(date_string: Optional[str]) -> Optional[str]:
        """
        Convert SEDIA date format to YYYY-MM-DD.

        Example:
            2027-12-01T00:00:00.000+0000

        becomes:
            2027-12-01
        """

        if not date_string:
            return None

        try:
            return datetime.fromisoformat(
                date_string.replace("Z", "+00:00")
            ).date().isoformat()

        except ValueError:

            # Fallback for SEDIA's +0000 format
            match = re.match(
                r"(\d{4}-\d{2}-\d{2})",
                str(date_string)
            )

            if match:
                return match.group(1)

        return None

    # ============================================================
    # EXTRACT KEYWORDS
    # ============================================================

    @staticmethod
    def extract_keywords(metadata: Dict[str, Any]) -> List[str]:
        """
        Extract useful keywords from SEDIA metadata.
        """

        keywords = []

        # Explicit keywords
        raw_keywords = metadata.get("keywords", [])

        if isinstance(raw_keywords, list):
            for keyword in raw_keywords:
                if keyword and keyword not in keywords:
                    keywords.append(str(keyword))

        # Cross-cutting priorities
        priorities = metadata.get(
            "crossCuttingPriorities",
            []
        )

        if isinstance(priorities, list):
            for priority in priorities:

                if priority and priority not in keywords:
                    keywords.append(str(priority))

        return keywords

    # ============================================================
    # EXTRACT TOPICS
    # ============================================================

    @staticmethod
    def extract_topics(metadata: Dict[str, Any]) -> List[str]:
        """
        Extract topic/programme information.
        """

        topics = []

        fields = [
            "callTitle",
            "destinationDescription",
            "frameworkProgramme",
            "programmePeriod"
        ]

        for field in fields:

            value = metadata.get(field, [])

            if isinstance(value, list):

                for item in value:

                    if item and str(item) not in topics:
                        topics.append(str(item))

            elif value:

                if str(value) not in topics:
                    topics.append(str(value))

        return topics

    # ============================================================
    # EXTRACT ELIGIBILITY
    # ============================================================

    @staticmethod
    def extract_eligibility(
        metadata: Dict[str, Any]
    ) -> str:
        """
        Extract eligibility information from topic conditions.
        """

        conditions = metadata.get(
            "topicConditions",
            []
        )

        if isinstance(conditions, list):

            if conditions:
                return FundingTendersCollector.clean_html(
                    conditions[0]
                )

        elif conditions:

            return FundingTendersCollector.clean_html(
                str(conditions)
            )

        return ""

    # ============================================================
    # MAP API RESULT
    # ============================================================

    def parse_result(
        self,
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert one SEDIA result into the project's Opportunity format.
        """

        metadata = result.get(
            "metadata",
            {}
        )

        # --------------------------------------------------------
        # ID
        # --------------------------------------------------------

        opportunity_id = (
            self.get_metadata(
                metadata,
                "identifier"
            )
            or result.get("reference")
        )

        if not opportunity_id:
            return None

        # --------------------------------------------------------
        # TITLE
        # --------------------------------------------------------

        title = (
            self.get_metadata(
                metadata,
                "title"
            )
            or result.get("summary")
            or ""
        )

        # --------------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------------

        description = (
            self.get_metadata(
                metadata,
                "descriptionByte"
            )
            or result.get("content")
            or result.get("summary")
            or ""
        )

        description = self.clean_html(
            description
        )

        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        status = self.get_metadata(
            metadata,
            "status"
        )

        # Safety check.
        # The API has been observed returning records that don't
        # respect the requested status filter.
        if status not in self.ALLOWED_STATUSES:
            return None

        # --------------------------------------------------------
        # DEADLINE
        # --------------------------------------------------------

        deadline_raw = self.get_metadata(
            metadata,
            "deadlineDate"
        )

        deadline = self.parse_date(
            deadline_raw
        )

        # --------------------------------------------------------
        # URL
        # --------------------------------------------------------

        url = (
            self.get_metadata(
                metadata,
                "url"
            )
            or result.get("url")
        )

        # Prefer the human-readable Funding & Tenders page.
        if url and "/data/topicDetails/" in url:

            url = (
                "https://ec.europa.eu/info/funding-tenders/"
                "opportunities/portal/screen/opportunities/"
                f"topic-details/{opportunity_id}"
            )

        # --------------------------------------------------------
        # CALL
        # --------------------------------------------------------

        call = self.get_metadata(
            metadata,
            "callTitle",
            ""
        )

        call_identifier = self.get_metadata(
            metadata,
            "callIdentifier",
            ""
        )

        # --------------------------------------------------------
        # ORGANIZATION
        # --------------------------------------------------------

        # SEDIA does not always expose a simple "organization"
        # field for a topic. The programme/call is therefore safer
        # than inventing a grant authority.
        organization = (
            "European Commission"
        )

        # --------------------------------------------------------
        # BUILD OPPORTUNITY
        # --------------------------------------------------------

        opportunity = {
            "id": opportunity_id,

            "title": title,

            "type": "funding_opportunity",

            "organization": organization,

            "description": description,

            "keywords": self.extract_keywords(
                metadata
            ),

            "topics": self.extract_topics(
                metadata
            ),

            "eligibility": self.extract_eligibility(
                metadata
            ),

            "deadline": deadline,

            "url": url,

            # Additional useful fields
            "status": status,

            "call": call,

            "call_identifier": call_identifier,

            "programme_period": self.get_metadata(
                metadata,
                "programmePeriod"
            ),

            "action_type": self.get_metadata(
                metadata,
                "typesOfAction"
            ),

            "opening_date": self.parse_date(
                self.get_metadata(
                    metadata,
                    "startDate"
                )
            )
        }

        return opportunity

    # ============================================================
    # COLLECT
    # ============================================================

    def collect(
        self,
        text: str,
        max_pages: int = 5,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect funding opportunities.

        Parameters
        ----------
        text:
            Search query, e.g. "biodiversity"

        max_pages:
            Maximum number of API pages to retrieve.

        page_size:
            Number of results requested per API page.
        """

        print("\n")
        print("=" * 70)
        print(f"SEARCHING FUNDING & TENDERS FOR: {text}")
        print("=" * 70)

        opportunities = {}

        for page in range(1, max_pages + 1):

            print(f"\nFetching page {page}...")

            try:

                data = self.search(
                    text=text,
                    page_number=page,
                    page_size=page_size
                )

            except requests.RequestException as e:

                print(
                    f"API request failed on page {page}: {e}"
                )

                break

            results = data.get(
                "results",
                []
            )

            total = data.get(
                "totalResults",
                0
            )

            print(
                f"API total results: {total}"
            )

            print(
                f"Results returned: {len(results)}"
            )

            if not results:
                print("No more results.")
                break

            # ----------------------------------------------------
            # PROCESS RESULTS
            # ----------------------------------------------------

            page_new = 0

            for result in results:

                opportunity = self.parse_result(
                    result
                )

                if not opportunity:
                    continue

                opportunity_id = opportunity["id"]

                # ------------------------------------------------
                # DEDUPLICATION
                # ------------------------------------------------

                if opportunity_id not in opportunities:

                    opportunities[
                        opportunity_id
                    ] = opportunity

                    page_new += 1

            print(
                f"New opportunities from page: {page_new}"
            )

            print(
                f"Total unique opportunities: "
                f"{len(opportunities)}"
            )

            # ----------------------------------------------------
            # STOP IF LAST PAGE
            # ----------------------------------------------------

            if len(results) == 0:
                print("No more results.")
                break

        return list(
            opportunities.values()
        )


# ================================================================
# MAIN TEST
# ================================================================

if __name__ == "__main__":

    collector = FundingTendersCollector()

    opportunities = collector.collect(
        text="biodiversity",
        max_pages=5,
        page_size=50
    )

    print("\n")
    print("=" * 70)
    print(
        f"FINAL UNIQUE OPPORTUNITIES: "
        f"{len(opportunities)}"
    )
    print("=" * 70)

    for i, opportunity in enumerate(
        opportunities,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Opportunity {i}"
        )

        print(
            f"ID: {opportunity['id']}"
        )

        print(
            f"Title: {opportunity['title']}"
        )

        print(
            f"Type: {opportunity['type']}"
        )

        print(
            f"Status: {opportunity['status']}"
        )

        print(
            f"Deadline: {opportunity['deadline']}"
        )

        print(
            f"Call: {opportunity['call']}"
        )

        print(
            f"URL: {opportunity['url']}"
        )