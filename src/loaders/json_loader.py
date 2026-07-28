import json
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

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        researchers = [
            Researcher(**researcher)
            for researcher in data
        ]

        return researchers


    @staticmethod
    def load_opportunities(file_path: str):

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        opportunities = [
            Opportunity(**opportunity)
            for opportunity in data
        ]

        return opportunities