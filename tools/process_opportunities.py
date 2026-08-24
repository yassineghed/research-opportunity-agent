from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.loaders import DataLoader
from src.processors import OpportunityProcessor
from src.collectors.cordis_collector import CordisCollector

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "opportunities.json"


def _load_raw_payload(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for field_name in ("results", "opportunities", "data", "items", "payload"):
            value = payload.get(field_name)
            if isinstance(value, list):
                return value

    raise ValueError(
        "Input JSON must be a list of opportunities or contain one under "
        "'results', 'opportunities', 'data', 'items', or 'payload'."
    )


def _parse_source_records(source: str, payload):
    if source == "cordis":
        collector = CordisCollector(api_key="")
        return collector.parse_payload(payload)

    return _extract_records(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw opportunity JSON into the Opportunity model."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the raw JSON file.",
    )
    parser.add_argument(
        "--source",
        default="",
        help="Optional source label to apply when records do not include one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip records that cannot be normalized instead of failing the whole run.",
    )
    args = parser.parse_args()

    raw_payload = _load_raw_payload(args.input_path)
    raw_records = _parse_source_records(
        (args.source or "").strip().lower(),
        raw_payload,
    )

    opportunities = []
    skipped_records = 0

    for raw_record in raw_records:
        try:
            opportunity = OpportunityProcessor.process(
                raw_record,
                source=args.source or None,
            )
        except (TypeError, ValueError):
            if not args.skip_invalid:
                raise
            skipped_records += 1
            continue

        opportunities.append(opportunity)

    DataLoader.save_opportunities(args.output, opportunities)

    message = (
        f"Saved {len(opportunities)} processed opportunities to {args.output}"
    )

    if skipped_records:
        message += f" | skipped {skipped_records} invalid record(s)"

    print(message)


if __name__ == "__main__":
    main()
