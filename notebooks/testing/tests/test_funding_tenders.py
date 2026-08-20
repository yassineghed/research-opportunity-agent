import requests
import json


BASE_URL = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"
API_KEY = "SEDIA"

SEARCH_TERMS = [
    "biodiversity",
    "computer vision",
    "YOLO",
    "marine conservation",
    "underwater monitoring"
]

PAGE_SIZE = 10

# Funding & Tenders statuses
# 31094501 = Forthcoming
# 31094502 = Open
STATUSES = ["31094501", "31094502"]


# ============================================================
# HELPER
# ============================================================

def get_first(metadata, field):
    value = metadata.get(field)

    if isinstance(value, list):
        return value[0] if value else None

    return value


# ============================================================
# SEARCH
# ============================================================

def search(term):

    query = {
        "bool": {
            "must": [
                {
                    "terms": {
                        "type": ["1"]
                    }
                },
                {
                    "terms": {
                        "status": STATUSES
                    }
                }
            ]
        }
    }

    params = {
        "apiKey": API_KEY,
        "text": term,
        "language": "en"
    }

    body = {
        "query": json.dumps(query),
        "pageNumber": 1,
        "pageSize": PAGE_SIZE
    }

    print("\n" + "=" * 80)
    print(f"SEARCH TERM: {term}")
    print("=" * 80)

    print("\nURL:")
    print(BASE_URL)

    print("\nPARAMS:")
    print(json.dumps(params, indent=2))

    print("\nBODY:")
    print(json.dumps(body, indent=2))

    response = requests.post(
        BASE_URL,
        params=params,
        data=body,
        timeout=30
    )

    print("\nSTATUS:", response.status_code)

    if response.status_code != 200:

        print("\nAPI ERROR:")
        print(response.text)

        return None

    return response.json()


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(term, data):

    if not data:
        return

    results = data.get("results", [])

    print("\n" + "-" * 80)
    print("RESULT SUMMARY")
    print("-" * 80)

    print("Search term       :", term)
    print("API total results :", data.get("totalResults"))
    print("Results returned   :", len(results))

    if not results:
        print("\nNo opportunities found.")
        return

    print("\n" + "-" * 80)
    print("OPPORTUNITIES")
    print("-" * 80)

    for i, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        identifier = get_first(metadata, "identifier")
        title = get_first(metadata, "title")
        description = get_first(metadata, "descriptionByte")
        content = result.get("content")
        summary = result.get("summary")

        keywords = metadata.get("keywords", [])
        call = get_first(metadata, "callTitle")
        status = get_first(metadata, "status")
        deadline = get_first(metadata, "deadlineDate")
        url = get_first(metadata, "url")

        print(f"\n{'=' * 60}")
        print(f"RESULT {i}")
        print(f"{'=' * 60}")

        print("ID:")
        print(identifier)

        print("\nTITLE:")
        print(title)

        print("\nCALL:")
        print(call)

        print("\nSTATUS:")
        print(status)

        print("\nDEADLINE:")
        print(deadline)

        print("\nKEYWORDS:")
        print(keywords)

        print("\nSUMMARY:")
        print(summary)

        print("\nCONTENT:")
        if content:
            print(content[:1000])
        else:
            print("None")

        print("\nDESCRIPTION:")
        if description:
            print(description[:1000])
        else:
            print("None")

        print("\nHIGHLIGHTED FRAGMENTS:")
        print(result.get("highlightedFragments", []))

        print("\nURL:")
        print(url)


# ============================================================
# CHECK WHETHER SEARCH TERM APPEARS IN RESULT
# ============================================================

def analyze_matching(term, data):

    if not data:
        return

    results = data.get("results", [])

    term_lower = term.lower()

    print("\n" + "-" * 80)
    print("SEARCH TERM ANALYSIS")
    print("-" * 80)

    for i, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        title = str(get_first(metadata, "title") or "")
        summary = str(result.get("summary") or "")
        content = str(result.get("content") or "")
        keywords = " ".join(
            str(x) for x in metadata.get("keywords", [])
        )

        combined_text = (
            title + " " +
            summary + " " +
            content + " " +
            keywords
        ).lower()

        found = term_lower in combined_text

        print(
            f"Result {i}: "
            f"{'TERM FOUND IN RETURNED DATA' if found else 'TERM NOT FOUND IN EXTRACTED FIELDS'}"
        )


# ============================================================
# RUN ONE SEARCH
# ============================================================

def run_test(term):

    data = search(term)

    if data is None:
        return

    display_results(term, data)

    analyze_matching(term, data)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("#" * 80)
    print("# FUNDING & TENDERS SEARCH TERM TEST")
    print("#" * 80)

    print("\nTesting terms:")

    for term in SEARCH_TERMS:
        print(" -", term)

    for term in SEARCH_TERMS:

        run_test(term)

    print("\n")
    print("#" * 80)
    print("# ALL TESTS COMPLETED")
    print("#" * 80)