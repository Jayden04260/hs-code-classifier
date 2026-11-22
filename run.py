"""
run.py

One-command commodity/HS code lookup: builds a retrieval index over the
UK Trade Tariff reference data (official titles + nomenclature text),
then answers questions about a product description in an interactive
loop.

Usage:
    python run.py
"""

import argparse

from classify import CONFIDENCE_GAP_THRESHOLD, build_index, classify, confidence_gap
from duty import fetch_standard_duty_rate
from references import DEFAULT_NOMENCLATURE_PATH, load_references


def _format_code(code: str) -> str:
    """10-digit code -> "8485 00 00 00" grouped for readability, matching
    how the UK Trade Tariff site itself displays commodity codes."""
    return f"{code[:4]} {code[4:6]} {code[6:8]} {code[8:10]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up a UK commodity/HS code for a product description.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to show (default: %(default)s)")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-score candidates with a cross-encoder (off by default: it measurably hurt accuracy on "
        "real product listings with the fine-tuned embedder - see README)",
    )
    parser.add_argument(
        "--no-duty", action="store_true", help="Skip the live duty-rate lookup for the top candidate"
    )
    args = parser.parse_args()

    print("Loading reference data and building index...")
    search_refs = load_references()
    nomenclature_refs = load_references(DEFAULT_NOMENCLATURE_PATH)
    index = build_index(search_refs + nomenclature_refs)
    print(f"Indexed {len(search_refs)} reference titles + {len(nomenclature_refs)} nomenclature entries.")
    print("Describe a product, or type 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        candidates = classify(query, index, top_k=args.top_k, use_rerank=args.rerank)
        if not candidates:
            print("No matches found.\n")
            continue

        gap = confidence_gap(candidates)
        if gap is not None and gap < CONFIDENCE_GAP_THRESHOLD:
            print(
                "  [low confidence - top candidates scored close together; "
                "this description may be genuinely ambiguous]"
            )

        print()
        for i, c in enumerate(candidates, start=1):
            print(f"  [{i}] {_format_code(c.code)}  (score: {c.score:.3f})")
            print(f"      matched: {c.title!r} ({c.level})")
            if i == 1 and not args.no_duty:
                duty_rate = fetch_standard_duty_rate(c.code)
                if duty_rate:
                    print(f"      standard (non-preferential) duty rate: {duty_rate}")
        print()


if __name__ == "__main__":
    main()
