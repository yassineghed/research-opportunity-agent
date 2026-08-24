import json
from dataclasses import asdict
from pathlib import Path

from src.models.researcher import Researcher
from src.models.opportunity import Opportunity


class DataLoader:
    """
    Loads mock data from JSON files and converts them
    into Python objects.
    """

    @staticmethod
    def load_researchers(file_path: str):
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        researchers = [
            Researcher(**researcher)
            for researcher in data
        ]

        return researchers


    @staticmethod
    def load_opportunities(file_path: str):
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        opportunities = [
            Opportunity(**opportunity)
            for opportunity in data
        ]

        return opportunities

    @staticmethod
    def save_opportunities(
        file_path: str | Path,
        opportunities: list[Opportunity],
    ) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [
            asdict(opportunity)
            for opportunity in opportunities
        ]

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
