"""
eval_hscodecomp.py

The realistic test this project was missing: HSCodeComp
(https://huggingface.co/datasets/ATH-MaaS/HSCodeComp, Apache-2.0) -
632 real e-commerce product listings (noisy titles, platform categories,
attributes) each classified by customs experts, from the paper
"HSCodeComp: A Realistic and Expert-Level Benchmark for Deep Search
Agents in Hierarchical Rule Application" (arXiv:2510.19631).

Its codes are 10-digit *US* codes. Only the first 6 digits are the
internationally harmonized HS code (the rest are national extensions),
so a UK code can't be compared past that point: this scores the top
candidate's 2-digit chapter, 4-digit heading and 6-digit subheading.
Heading is the headline number - it's what a classifier narrows to first.

The query is what a seller would have to hand: the listing title, the
platform category path and the stated material.

Also reports the confidence-gap quartiles (see
classify.CONFIDENCE_GAP_THRESHOLD) on this data. HSCodeComp is never
used for training (see train_embedder.py).

    python eval_hscodecomp.py            # default config (no reranking)
    python eval_hscodecomp.py --rerank   # with cross-encoder re-ranking
"""

import argparse
import json

from classify import CONFIDENCE_GAP_THRESHOLD, SEARCH_POOL, _dedupe_by_code, build_index
from embeddings import embed_queries
from references import DEFAULT_NOMENCLATURE_PATH, load_references
from rerank import rerank as rerank_candidates

DATASET = "ATH-MaaS/HSCodeComp"
DATA_FILE = "data/test_data.jsonl"
_UNINFORMATIVE = {"", "others", "other", "none"}


def listing_to_query(row: dict) -> str:
    """Turn one HSCodeComp listing into a free-text description."""
    attrs = json.loads(row["product_attributes"]) if row.get("product_attributes") else {}
    material = str(attrs.get("Material") or attrs.get("material") or "").strip()
    query = f"{row['product_name']}. Category: {row['cate_lv2_desc']} > {row['cate_lv3_desc']}."
    if material.lower() not in _UNINFORMATIVE:
        query += f" Material: {material}."
    return query


def load_hscodecomp() -> list[tuple[str, str]]:
    """(query, 10-digit expected code) pairs. Downloads ~2MB on first run."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [(listing_to_query(r), str(r["hs_code"]).zfill(10)) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder re-ranking")
    args = parser.parse_args()

    examples = load_hscodecomp()
    index = build_index(load_references() + load_references(DEFAULT_NOMENCLATURE_PATH))
    vectors = embed_queries([q for q, _ in examples])

    chapter = heading = subheading = 0
    gaps = []
    for (query, code), vector in zip(examples, vectors):
        candidates = index.search(vector, top_k=SEARCH_POOL)
        if args.rerank:
            candidates = rerank_candidates(query, candidates, top_k=SEARCH_POOL)
        deduped = _dedupe_by_code(candidates)[:5]
        top = deduped[0][0].code if deduped else ""
        chapter += top[:2] == code[:2]
        heading += top[:4] == code[:4]
        subheading += top[:6] == code[:6]
        if len(deduped) >= 2:
            gaps.append((deduped[0][1] - deduped[1][1], top[:4] == code[:4]))

    n = len(examples)
    print(f"HSCodeComp: {n} expert-classified e-commerce listings ({'re-ranked' if args.rerank else 'cosine'})")
    print(f"  top-1 chapter (2 digits):    {chapter / n:.3f}")
    print(f"  top-1 heading (4 digits):    {heading / n:.3f}")
    print(f"  top-1 subheading (6 digits): {subheading / n:.3f}")

    gaps.sort()
    q = len(gaps) // 4
    print("  confidence-gap quartiles (heading accuracy per bucket, lowest gap first):")
    for i in range(4):
        bucket = gaps[i * q : (i + 1) * q if i < 3 else len(gaps)]
        acc = sum(ok for _, ok in bucket) / len(bucket)
        print(f"    Q{i + 1} (gap {bucket[0][0]:.3f} to {bucket[-1][0]:.3f}, n={len(bucket)}): {acc:.2f}")
    if not args.rerank:
        low = [ok for g, ok in gaps if g < CONFIDENCE_GAP_THRESHOLD]
        high = [ok for g, ok in gaps if g >= CONFIDENCE_GAP_THRESHOLD]
        if low and high:
            print(
                f"  gate at {CONFIDENCE_GAP_THRESHOLD}: flagged {len(low) / len(gaps):.0%} of listings "
                f"(heading acc {sum(low) / len(low):.2f}) vs unflagged {sum(high) / len(high):.2f}"
            )


if __name__ == "__main__":
    main()
