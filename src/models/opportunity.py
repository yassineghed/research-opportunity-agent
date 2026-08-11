from dataclasses import dataclass
from typing import Optional


@dataclass
class Opportunity:
    id: int
    title: str
    type: str
    organization: str
    description: str
    keywords: list[str]
    topics: list[str]
    eligibility: str
    deadline: str
    url: Optional[str] = ""