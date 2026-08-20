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

    Goal:
    - Search Funding & Tenders opportunities
    - Retrieve all matching pages
    - Keep only open/forthcoming topics
    - Clean and normalize the data
    - Deduplicate opportunities
    """

    BASE_URL = (
        "https://api.tech.ec.europa.eu/"
        "search-api/prod/rest/search"
    )

    API_KEY = "SEDIA"
    LANGUAGE = "en"

    # Funding & Tenders topic
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
        language: str = LANGUAGE
    ):
        self.api_key = api_key
        self.language = language

    # ============================================================
    # SEARCH API
    # ============================================================

    def search(
        self,
        text: str,
        page_number: int
    ) -> Dict[str, Any]:
        """
        Retrieve one page of Funding & Tenders results.

        The API currently returns a maximum of 10 results
        per request, so pagination is handled internally.
        """

        params = {
            "apiKey": self.api_key,
            "text": text,
            "language": self.language
        }

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

        languages = [
            self.language
        ]

        sort = {
            "field": "sortStatus",
            "order": "ASC"
        }

        files = {
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
        }

        response = requests.post(
            self.BASE_URL,
            params=params,
            files=files,
            data={
                "pageNumber": page_number,
                "pageSize": 10
            },
            timeout=30
        )

        print(
            f"Page {page_number} | "
            f"HTTP {response.status_code}"
        )

        response.raise_for_status()

        return response.json()

    # ============================================================
    # CLEAN HTML
    # ============================================================

    @staticmethod
    def clean_html(
        text: Optional[str]
    ) -> str:

        if not text:
            return ""

        text = unescape(str(text))

        text = re.sub(
            r"<[^>]+>",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ============================================================
    # METADATA
    # ============================================================

    @staticmethod
    def get_metadata(
        metadata: Dict[str, Any],
        field: str,
        default: Any = None
    ) -> Any:

        value = metadata.get(
            field,
            default
        )

        if isinstance(value, list):

            if not value:
                return default

            return value[0]

        return value

    # ============================================================
    # DATE
    # ============================================================

    @staticmethod
    def parse_date(
        date_string: Optional[str]
    ) -> Optional[str]:

        if not date_string:
            return None

        date_string = str(date_string)

        try:

            return datetime.fromisoformat(
                date_string.replace(
                    "Z",
                    "+00:00"
                )
            ).date().isoformat()

        except ValueError:

            match = re.match(
                r"(\d{4}-\d{2}-\d{2})",
                date_string
            )

            if match:
                return match.group(1)

        return None

    # ============================================================
    # KEYWORDS
    # ============================================================

    @staticmethod
    def extract_keywords(
        metadata: Dict[str, Any]
    ) -> List[str]:

        keywords = []

        fields = [
            "keywords",
            "crossCuttingPriorities"
        ]

        for field in fields:

            values = metadata.get(
                field,
                []
            )

            if not isinstance(values, list):
                values = [values]

            for value in values:

                if value:

                    value = str(value).strip()

                    if (
                        value
                        and value not in keywords
                    ):
                        keywords.append(value)

        return keywords

    # ============================================================
    # TOPICS
    # ============================================================

    @staticmethod
    def extract_topics(
        metadata: Dict[str, Any]
    ) -> List[str]:

        topics = []

        fields = [
            "callTitle",
            "destinationDescription",
            "frameworkProgramme",
            "programmePeriod"
        ]

        for field in fields:

            values = metadata.get(
                field,
                []
            )

            if not isinstance(values, list):
                values = [values]

            for value in values:

                if value:

                    value = str(value).strip()

                    if (
                        value
                        and value not in topics
                    ):
                        topics.append(value)

        return topics

    # ============================================================
    # ELIGIBILITY
    # ============================================================

    @staticmethod
    def extract_eligibility(
        metadata: Dict[str, Any]
    ) -> str:

        conditions = metadata.get(
            "topicConditions",
            []
        )

        if isinstance(
            conditions,
            list
        ):

            cleaned = []

            for condition in conditions:

                condition = (
                    FundingTendersCollector
                    .clean_html(condition)
                )

                if condition:
                    cleaned.append(condition)

            return " ".join(cleaned)

        return (
            FundingTendersCollector
            .clean_html(conditions)
        )

    # ============================================================
    # PARSE RESULT
    # ============================================================

    def parse_result(
        self,
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

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
        # STATUS
        # --------------------------------------------------------

        status = self.get_metadata(
            metadata,
            "status"
        )

        if status not in self.ALLOWED_STATUSES:
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

        title = self.clean_html(
            title
        )

        if not title:
            return None

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
        # DEADLINE
        # --------------------------------------------------------

        deadline = self.parse_date(
            self.get_metadata(
                metadata,
                "deadlineDate"
            )
        )

        # --------------------------------------------------------
        # OPENING DATE
        # --------------------------------------------------------

        opening_date = self.parse_date(
            self.get_metadata(
                metadata,
                "startDate"
            )
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
        # URL
        # --------------------------------------------------------

        url = (
            self.get_metadata(
                metadata,
                "url"
            )
            or result.get("url")
        )

        if url and "/data/topicDetails/" in url:

            url = (
                "https://ec.europa.eu/info/"
                "funding-tenders/opportunities/"
                "portal/screen/opportunities/"
                f"topic-details/{opportunity_id}"
            )

        # --------------------------------------------------------
        # ORGANIZATION
        # --------------------------------------------------------

        organization = (
            self.get_metadata(
                metadata,
                "organisation"
            )
            or self.get_metadata(
                metadata,
                "organization"
            )
            or "European Commission"
        )

        # --------------------------------------------------------
        # BUILD OPPORTUNITY
        # --------------------------------------------------------

        return {

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

            "status": status,

            "call": call,

            "call_identifier": call_identifier,

            "programme_period": (
                self.get_metadata(
                    metadata,
                    "programmePeriod"
                )
            ),

            "action_type": (
                self.get_metadata(
                    metadata,
                    "typesOfAction"
                )
            ),

            "opening_date": opening_date
        }

    # ============================================================
    # COLLECT ALL
    # ============================================================

    def collect(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Search and retrieve ALL matching opportunities.

        Pagination is handled internally.

        Example:
            collector.collect("AI")
        """

        print("\n")
        print("=" * 70)
        print(
            f"SEARCHING FUNDING & TENDERS FOR: {text}"
        )
        print("=" * 70)

        opportunities = {}

        page = 1

        while True:

            print(
                f"\nFetching page {page}..."
            )

            try:

                data = self.search(
                    text=text,
                    page_number=page
                )

            except requests.RequestException as e:

                print(
                    f"Request failed: {e}"
                )

                break

            results = data.get(
                "results",
                []
            )

            total_results = data.get(
                "totalResults",
                0
            )

            print(
                f"API total results: "
                f"{total_results}"
            )

            print(
                f"Results returned: "
                f"{len(results)}"
            )

            # ----------------------------------------------------
            # NO MORE RESULTS
            # ----------------------------------------------------

            if not results:

                print(
                    "No more results."
                )

                break

            # ----------------------------------------------------
            # PROCESS PAGE
            # ----------------------------------------------------

            page_new = 0

            for result in results:

                opportunity = self.parse_result(
                    result
                )

                if not opportunity:
                    continue

                opportunity_id = (
                    opportunity["id"]
                )

                if opportunity_id not in opportunities:

                    opportunities[
                        opportunity_id
                    ] = opportunity

                    page_new += 1

            print(
                f"New opportunities: "
                f"{page_new}"
            )

            print(
                f"Total unique opportunities: "
                f"{len(opportunities)}"
            )

            # ----------------------------------------------------
            # LAST PAGE
            # ----------------------------------------------------

            if (
                len(results) < 10
                or
                len(opportunities) >= total_results
            ):

                print(
                    "\nAll API results have been processed."
                )

                break

            page += 1

        return list(
            opportunities.values()
        )


# ================================================================
# TEST
# ================================================================

from pprint import pprint


if __name__ == "__main__":

    collector = FundingTendersCollector()

    opportunities = collector.collect(
        text="AI"
    )

    print("\n")
    print("=" * 70)
    print(
        f"FINAL UNIQUE OPPORTUNITIES: "
        f"{len(opportunities)}"
    )
    print("=" * 70)

    if not opportunities:
        print("\nNo opportunities found.")
        exit()

    opportunity = opportunities[0]

    print("\n")
    print("=" * 70)
    print("FIRST OPPORTUNITY")
    print("=" * 70)

    for field, value in opportunity.items():

        print(f"\nFIELD: {field}")
        print(f"TYPE : {type(value).__name__}")
        print(f"VALUE: ", end="")

        pprint(
            value,
            sort_dicts=False
        )

    print("\n")
    print("=" * 70)
    print("ALL AVAILABLE FIELDS")
    print("=" * 70)

    print(
        list(opportunity.keys())
    )