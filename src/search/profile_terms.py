from __future__ import annotations

import re
from typing import Any


_WHITESPACE = re.compile(r"\s+")
_GENERIC_TERMS = {
    "research",
    "science",
    "technology",
    "innovation",
    "academic",
    "university",
}


def extract_profile_terms(researcher: Any, max_terms: int = 8) -> list[str]:
    """Return prioritized, deduplicated search phrases from a profile."""
    if max_terms < 1:
        return []

    fields = (
        "research_domains",
        "research_interests",
        "keywords",
        "skills",
    )
    terms: list[str] = []
    seen: set[str] = set()

    for field_name in fields:
        values = getattr(researcher, field_name, []) or []
        if isinstance(values, str):
            values = [values]

        for value in values:
            term = _normalize_term(value)
            key = term.casefold()
            if not term or key in seen or key in _GENERIC_TERMS:
                continue
            seen.add(key)
            terms.append(term)
            if len(terms) >= max_terms:
                return terms

    return terms


def _normalize_term(value: Any) -> str:
    if value is None:
        return ""
    term = _WHITESPACE.sub(" ", str(value)).strip(" ,;\t\r\n")
    if len(term) < 2:
        return ""
    return term