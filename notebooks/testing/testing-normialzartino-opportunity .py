import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
from src.collectors.funding_tenders_collector import FundingTendersCollector

collector = FundingTendersCollector()
items = collector.collect_processed_and_save("AI")

print(len(items))
print(items[0] if items else "No results")