"""import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

url = "https://cordis.europa.eu/api/dataextractions/getExtraction"

params = {
    "query": "artificial intelligence",
    "key": "620a8386-b3eb-49a6-a950-9b05a8e4fc38",
    "outputFormat": "json",
    "archived": "false"
}

response = requests.get(url, params=params)

print(response.status_code)
print(response.text[:1000])"""

import requests

url = "https://cordis.europa.eu/api/dataextractions/getExtraction"

params = {
    "query": "artificial intelligence",
    "key": "620a8386-b3eb-49a6-a950-9b05a8e4fc38",
    "outputFormat": "json",
    "archived": "false"
}

response = requests.get(url, params=params)

print(response.url)
print(response.status_code)
print(response.text)