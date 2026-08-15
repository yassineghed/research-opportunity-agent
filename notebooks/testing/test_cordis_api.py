import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json

from dotenv import load_dotenv

from src.collectors.cordis_collector import CordisCollector


# 1. Load environment variables

load_dotenv()

api_key = os.getenv("CORDIS_API_KEY")
# 2. Check API key
if not api_key:
    raise ValueError("CORDIS_API_KEY was not found in .env")

print("API key loaded successfully.")

# 3. Create collector
collector = CordisCollector(api_key)

# 4. Test query
query = '"computer vision" AND biodiversity"'
query = collector.normalize_query(query)
print("\nStarting CORDIS extraction...")
print(f"Query: {query}")

# 5. Collect data
data = collector.collect(query)

# 6. Inspect result
print("\nExtraction finished.")
print("Type of returned data:")
print(type(data))

# 7. Inspect JSON structure
if isinstance(data, dict):
    print("\nTop-level keys:")
    for key in data.keys():
        print("-", key)

elif isinstance(data, list):
    print("\nNumber of returned items:")
    print(len(data))

# 8. Print first part of result
print("\nSample of returned data:")
print(
    json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )[:5000]
)

# 9. Save raw result
os.makedirs(
    "data/raw",
    exist_ok=True
)

output_path = (
    "data/raw/"
    "cordis_ai_raw.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        data,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    f"\nRaw data saved to: {output_path}"
)