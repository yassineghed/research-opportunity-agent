"""Benchmark data models and loader for evaluation."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuery:
    query_id: str
    researcher_id: int
    relevant_opportunity_ids: list[str | int]
    irrelevant_opportunity_ids: list[str | int]
    notes: str = ""


class EvalDataset:
    def __init__(self, queries: list[EvalQuery]):
        self.queries = queries

    @classmethod
    def load_from_json(cls, filepath: Path | str) -> "EvalDataset":
        """Load evaluation benchmark dataset from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append(
                EvalQuery(
                    query_id=item.get("query_id", ""),
                    researcher_id=item["researcher_id"],
                    relevant_opportunity_ids=item.get("relevant_opportunity_ids", []),
                    irrelevant_opportunity_ids=item.get("irrelevant_opportunity_ids", []),
                    notes=item.get("notes", ""),
                )
            )
        return cls(queries)

    def save_to_json(self, filepath: Path | str) -> None:
        """Save dataset to JSON."""
        data = [
            {
                "query_id": q.query_id,
                "researcher_id": q.researcher_id,
                "relevant_opportunity_ids": q.relevant_opportunity_ids,
                "irrelevant_opportunity_ids": q.irrelevant_opportunity_ids,
                "notes": q.notes,
            }
            for q in self.queries
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
