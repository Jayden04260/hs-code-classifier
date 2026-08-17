"""
fetch_data.py

Downloads the UK government's Trade Tariff commodity-code reference
data and saves a local snapshot at data/search_references.json. Run
this once at setup, and again whenever you want to refresh against the
live tariff:

    python fetch_data.py

The rest of this project reads only the local snapshot, never the live
API - same local-first reasoning as this author's other projects
(local-rag, Notes-bot): reproducible without a network dependency at
classification time, and the tariff only actually changes a handful of
times a year, so a live call on every run would be pure overhead.

Source: www.trade-tariff.service.gov.uk/api/v2/search_references - the
free, public, unauthenticated JSON API that powers the UK's own
"Trade Tariff: look up commodity codes" tool
(https://www.gov.uk/trade-tariff). Not the same as the separate OAuth-
gated api.trade-tariff.service.gov.uk developer API - that one needs
registered credentials; this one doesn't. Returns ~9,000 (product name,
commodity code) reference pairs used by HMRC's own search tool -
exactly the "common name -> code" mapping this project needs.
"""

import json
import urllib.request
from pathlib import Path

SOURCE_URL = "https://www.trade-tariff.service.gov.uk/api/v2/search_references"
OUTPUT_PATH = Path(__file__).parent / "data" / "search_references.json"


def fetch_references(url: str = SOURCE_URL, timeout: int = 60) -> list[dict]:
    """
    Fetch the full reference list and return it as a list of
    {"title": str, "code": str, "level": str} dicts - "level" is the
    granularity the reference points at (Chapter/Heading/Subheading/
    Commodity), since not every product name in this data is classified
    all the way to a full 10-digit commodity code.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "hs-code-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    references = []
    for item in payload["data"]:
        attrs = item["attributes"]
        references.append(
            {
                "title": attrs["title"],
                "code": attrs["goods_nomenclature_item_id"],
                "level": attrs["referenced_class"],
            }
        )
    return references


def main() -> None:
    references = fetch_references()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(references)} references to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
