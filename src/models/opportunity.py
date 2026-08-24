from dataclasses import dataclass
from typing import Optional


@dataclass
class Opportunity:
    id: int | str
    title: str
    type: str
    organization: str
    description: str
    keywords: list[str]
    topics: list[str]
    eligibility: str
    deadline: str
    url: Optional[str] = ""
    source: Optional[str] = ""
    status: Optional[str] = ""
