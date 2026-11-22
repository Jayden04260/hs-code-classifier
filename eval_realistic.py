"""
eval_realistic.py

eval_classification.py measures accuracy on official reference titles
held out from the training data - a legitimate test, but one where both
the query and the index are drawn from the same short, formal-register
text. Real product descriptions aren't phrased like that ("Men's 100%
cotton crew-neck t-shirts, machine knitted, various colors, retail
packed" vs. the reference data's terse "T-shirts"). No public dataset
pairing realistic free-text descriptions with verified HS codes was
found while building this project (see README "Limitations") - so this
is a small, hand-written set instead.

Ground truth honesty: every expected code here is taken directly from
data/search_references.json - the same authoritative UK government
data the tool is built on, not a code guessed or verified by hand from
general knowledge. Each entry's SOURCE field names the exact official
reference title that code came from. What's hand-written is only the
query phrasing - a realistic paraphrase of that title, written without
looking at the code afterward, so the "how would someone actually
describe this" step stays independent of already knowing the answer.

Run it directly:

    python eval_realistic.py
"""

from classify import SEARCH_POOL, _dedupe_by_code, build_index
from embeddings import embed_query
from references import DEFAULT_NOMENCLATURE_PATH, load_references

MAX_K = 5

# (realistic query, expected code, source reference title the code came from)
EVAL_SET = [
    ("Men's 100% cotton crew-neck t-shirts, machine knitted, various colors, retail packed", "6109100010", "T-shirts"),
    ("Wireless over-ear bluetooth headphones with noise cancellation", "8518300090", "headphones, wireless"),
    ("Himalayan pink rock salt, food grade, sold in 500g bags for retail", "2501009100", "pink salt"),
    (
        "Digital black box flight data recorder unit for commercial aircraft",
        "8543700400",
        "flight data recorders, digital",
    ),
    ("Battery powered ride-on electric scooter for children aged 5-8", "9503001000", "electric scooter, childrens"),
    ("Powder-coated steel bed headboard, king size, flat pack", "9403202000", "headboards, metal"),
    ("Fresh star fruit, imported, sold loose by the kilogram", "0810902090", "carambolas, fresh"),
    ("Cold-pressed hemp seed oil, unrefined, for culinary use, bottled", "1515909100", "hemp oil"),
    ("Water-based acrylic emulsion paint for interior walls, 5 litre tins", "3209900000", "water-based paint"),
    ("Set of four rubber floor mats for cars, moulded to fit", "4016910000", "car mats, rubber"),
    ("Blank magnetic stripe swipe cards for hotel key card systems", "8523210000", "swipe cards"),
    ("Handheld signal flare gun for maritime distress signalling", "9303900000", "flare guns"),
    ("Heavy duty plastic protective gauntlets for industrial handling", "3926909790", "gauntlets, plastic"),
    ("Phenolic resin in solid block form for industrial moulding", "3909400000", "phenolic resins"),
    ("Inflatable camping air mattress, single size, PVC", "6306400000", "pneumatic mattresses"),
    ("Motorised snowplough vehicle for clearing roads", "8705908000", "snowploughs, motorised"),
    ("Rechargeable electric shaver for men, rotary head", "8510100000", "electric shavers"),
    ("Multi-layer engineered wood flooring panels, tropical hardwood surface", "4412332010", "multi layered floor"),
    ("Stainless steel examination table for veterinary clinics", "9402900000", "furniture, veterinary surgeries"),
    ("Industrial metal planing machine for workshop use", "8461200000", "planing machines for metal working"),
]


def _rank_of_correct_code(candidates, expected_code: str) -> int | None:
    for rank, (ref, _score) in enumerate(candidates, start=1):
        if ref.code == expected_code:
            return rank
    return None


def main() -> None:
    search_refs = load_references()
    nomenclature_refs = load_references(DEFAULT_NOMENCLATURE_PATH)
    index = build_index(search_refs + nomenclature_refs)

    print(f"Evaluating {len(EVAL_SET)} hand-written realistic product descriptions\n")

    ranks = []
    for query, expected_code, source in EVAL_SET:
        query_vector = embed_query(query)
        candidates = index.search(query_vector, top_k=SEARCH_POOL)
        deduped = _dedupe_by_code(candidates)[:MAX_K]

        rank = _rank_of_correct_code(deduped, expected_code)
        ranks.append(rank)

        top1 = deduped[0][0].title if deduped else "(no match)"
        status = f"rank {rank}" if rank else "not in top 5"
        print(f"[{status:>12}] {query}")
        print(f"              expected: {expected_code} ({source!r})  top1 matched: {top1!r}")

    n = len(ranks)
    top1 = sum(1 for r in ranks if r == 1) / n
    top5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr = sum((1 / r) if r else 0.0 for r in ranks) / n
    print(f"\ntop1={top1:.2f}  top5={top5:.2f}  MRR={mrr:.2f}  (n={n})")


if __name__ == "__main__":
    main()
