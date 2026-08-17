"""
fetch_nomenclature.py

Downloads the UK Trade Tariff's official chapter and heading
descriptions (the actual legal nomenclature text, not the informal
product-name synonyms in fetch_data.py's search_references) and saves
a local snapshot at data/nomenclature.json.

    python fetch_nomenclature.py

Why this exists alongside search_references.json: the reference titles
("abrasive stones, natural") are short, informal product names. The
official nomenclature text ("Millstones, grindstones, grinding wheels
and the like...") is longer, more precise, and sometimes closer to how
a real product description is actually phrased - indexing both gives
the retrieval step more surface area to match against. See README "How
accurate is it?" for whether this measurably helped.

One API call per chapter (98 total) - each chapter response includes
all of its headings' descriptions, so this doesn't need a separate call
per heading.
"""

import json
import urllib.request
from pathlib import Path

CHAPTERS_URL = "https://www.trade-tariff.service.gov.uk/api/v2/chapters"
CHAPTER_DETAIL_URL = "https://www.trade-tariff.service.gov.uk/api/v2/chapters/{}.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "nomenclature.json"


def _get_json(url: str, timeout: int = 60) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "hs-code-classifier/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_nomenclature() -> list[dict]:
    """
    Returns a list of {"title": str, "code": str, "level": str} dicts -
    same shape as fetch_data.fetch_references - covering every chapter
    and heading's official description.
    """
    chapters = _get_json(CHAPTERS_URL)["data"]
    entries = []

    for chapter in chapters:
        chapter_id = chapter["attributes"]["goods_nomenclature_item_id"][:2]
        detail = _get_json(CHAPTER_DETAIL_URL.format(chapter_id))

        chapter_attrs = detail["data"]["attributes"]
        entries.append(
            {
                "title": chapter_attrs["formatted_description"],
                "code": chapter_attrs["goods_nomenclature_item_id"],
                "level": "Chapter",
            }
        )

        for item in detail.get("included", []):
            if item["type"] != "heading":
                continue
            attrs = item["attributes"]
            entries.append(
                {
                    "title": attrs["formatted_description"],
                    "code": attrs["goods_nomenclature_item_id"],
                    "level": "Heading",
                }
            )

    return entries


def main() -> None:
    entries = fetch_nomenclature()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(entries)} chapter/heading descriptions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
