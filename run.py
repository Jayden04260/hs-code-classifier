"""
run.py

One-command commodity/HS code lookup: builds a retrieval index over the
UK Trade Tariff reference data, then answers questions about a product
description in an interactive loop.

Usage:
    python run.py
"""

import argparse

from classify import build_index, classify
from references import load_references


def _format_code(code: str) -> str:
    """10-digit code -> "8485 00 00 00" grouped for readability, matching
    how the UK Trade Tariff site itself displays commodity codes."""
    return f"{code[:4]} {code[4:6]} {code[6:8]} {code[8:10]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up a UK commodity/HS code for a product description.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to show (default: %(default)s)")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip cross-encoder re-ranking (faster, less accurate)"
    )
    args = parser.parse_args()

    print("Loading reference data and building index...")
    references = load_references()
    index = build_index(references)
    print(f"Indexed {len(references)} reference titles ({len({r.code for r in references})} distinct codes).")
    print("Describe a product, or type 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        candidates = classify(query, index, top_k=args.top_k, use_rerank=not args.no_rerank)
        if not candidates:
            print("No matches found.\n")
            continue

        print()
        for i, c in enumerate(candidates, start=1):
            print(f"  [{i}] {_format_code(c.code)}  (score: {c.score:.2f})")
            print(f"      matched: {c.title!r} ({c.level})")
        print()


if __name__ == "__main__":
    main()
