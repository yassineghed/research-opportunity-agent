from dataclasses import dataclass

@dataclass 
class Researcher:
    fullname: str
    institution: str
    research_domains: list[str]
    research_interests: list[str]
    skills: list[str]
    keywords: list[str]
    publications: list[str]