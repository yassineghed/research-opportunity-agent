import re
from html import unescape
from typing import Any

from src.models.opportunity import Opportunity


class OpportunityProcessor:
    """
    Normalize raw opportunity records from external collectors so they fit
    the existing Opportunity model without inventing missing information.
    """

    _HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    _WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
    _BLANK_LINE_PATTERN = re.compile(r"\n{3,}")

    @classmethod
    def process(
        cls,
        raw_opportunity: Opportunity | dict[str, Any],
        source: str | None = None,
    ) -> Opportunity:
        if isinstance(raw_opportunity, Opportunity):
            return cls._normalize_existing_opportunity(raw_opportunity)

        if not isinstance(raw_opportunity, dict):
            raise TypeError(
                "raw_opportunity must be an Opportunity or a dictionary."
            )

        normalized_source = cls._normalize_scalar(
            raw_opportunity.get("source") or source
        )
        normalized_url = cls._normalize_scalar(raw_opportunity.get("url"))
        normalized_status = cls._normalize_scalar(raw_opportunity.get("status"))

        return Opportunity(
            id=cls._resolve_id(raw_opportunity),
            title=cls._normalize_scalar(raw_opportunity.get("title")),
            type=cls._normalize_scalar(raw_opportunity.get("type")),
            organization=cls._normalize_scalar(raw_opportunity.get("organization")),
            description=cls._normalize_text(raw_opportunity.get("description")),
            keywords=cls._normalize_list(raw_opportunity.get("keywords")),
            topics=cls._normalize_list(raw_opportunity.get("topics")),
            eligibility=cls._normalize_text(raw_opportunity.get("eligibility")),
            deadline=cls._normalize_scalar(raw_opportunity.get("deadline")),
            url=normalized_url,
            source=normalized_source,
            status=normalized_status,
        )

    @classmethod
    def process_many(
        cls,
        raw_opportunities: list[Opportunity | dict[str, Any]],
        source: str | None = None,
    ) -> list[Opportunity]:
        return [
            cls.process(raw_opportunity, source=source)
            for raw_opportunity in raw_opportunities
        ]

    @classmethod
    def _normalize_existing_opportunity(cls, opportunity: Opportunity) -> Opportunity:
        return Opportunity(
            id=opportunity.id,
            title=cls._normalize_scalar(opportunity.title),
            type=cls._normalize_scalar(opportunity.type),
            organization=cls._normalize_scalar(opportunity.organization),
            description=cls._normalize_text(opportunity.description),
            keywords=cls._normalize_list(opportunity.keywords),
            topics=cls._normalize_list(opportunity.topics),
            eligibility=cls._normalize_text(opportunity.eligibility),
            deadline=cls._normalize_scalar(opportunity.deadline),
            url=cls._normalize_scalar(opportunity.url),
            source=cls._normalize_scalar(opportunity.source),
            status=cls._normalize_scalar(opportunity.status),
        )

    @classmethod
    def _resolve_id(cls, raw_opportunity: dict[str, Any]) -> int | str:
        for field_name in ("id", "identifier", "opportunity_id", "reference"):
            value = raw_opportunity.get(field_name)
            if value not in (None, ""):
                if isinstance(value, str):
                    stripped_value = value.strip()
                    return stripped_value if stripped_value else ""
                return value

        raise ValueError("Opportunity is missing an identifier.")

    @classmethod
    def _normalize_scalar(cls, value: Any) -> str:
        if value is None:
            return ""

        text = cls._normalize_line_breaks(unescape(str(value)))
        text = cls._HTML_TAG_PATTERN.sub(" ", text)
        text = cls._WHITESPACE_PATTERN.sub(" ", text)

        return text.strip()

    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        text = cls._normalize_line_breaks(unescape(str(value))) if value is not None else ""
        text = cls._HTML_TAG_PATTERN.sub(" ", text)

        lines: list[str] = []
        seen_lines: set[str] = set()

        for line in text.split("\n"):
            normalized_line = cls._WHITESPACE_PATTERN.sub(" ", line).strip()
            if not normalized_line:
                continue

            dedupe_key = normalized_line.casefold()
            if dedupe_key in seen_lines:
                continue

            seen_lines.add(dedupe_key)
            lines.append(normalized_line)

        normalized_text = "\n".join(lines)
        normalized_text = cls._BLANK_LINE_PATTERN.sub("\n\n", normalized_text)

        return normalized_text.strip()

    @classmethod
    def _normalize_list(cls, values: Any) -> list[str]:
        if values is None:
            return []

        if not isinstance(values, list):
            values = [values]

        normalized_values: list[str] = []
        seen_values: set[str] = set()

        for value in values:
            normalized_value = cls._normalize_scalar(value)
            if not normalized_value:
                continue

            dedupe_key = normalized_value.casefold()
            if dedupe_key in seen_values:
                continue

            seen_values.add(dedupe_key)
            normalized_values.append(normalized_value)

        return normalized_values

    @staticmethod
    def _normalize_line_breaks(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")
