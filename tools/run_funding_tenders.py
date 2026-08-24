from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.collectors.funding_tenders_collector import FundingTendersCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Funding & Tenders opportunities, normalize them, "
            "merge them into the processed store, and print a summary."
        )
    )
    parser.add_argument(
        "query",
        help="Search query to send to the Funding & Tenders API.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. Defaults to "
            "data/processed/opportunities.json"
        ),
    )
    args = parser.parse_args()

    collector = FundingTendersCollector()
    result = collector.collect_processed_and_save(
        text=args.query,
        output_path=args.output,
    )

    summary = result["summary"]
    output_path = result["output_path"]

    print(f"Saved merged opportunities to {output_path}")
    print(
        "Summary: "
        f"new={summary['new']} | "
        f"updated={summary['updated']} | "
        f"unchanged={summary['unchanged']} | "
        f"total_before={summary['total_before']} | "
        f"total_after={summary['total_after']}"
    )


if __name__ == "__main__":
    main()
