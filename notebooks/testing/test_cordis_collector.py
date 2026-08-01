import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))


from src.collectors.cordis_collector import CordisCollector

collector = CordisCollector(
    "data/mock/cordis_response.json"
)


opportunities = collector.collect()


print(
    f"Number of opportunities: {len(opportunities)}"
)


for opportunity in opportunities:

    print("-------------------------")

    print("ID:", opportunity.id)
    print("Title:", opportunity.title)
    print("Type:", opportunity.type)
    print("Organization:", opportunity.organization)
    print("Description:", opportunity.description)
    print("Deadline:", opportunity.deadline)
    print("URL:", opportunity.url)