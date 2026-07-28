from dataclasses import dataclass

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