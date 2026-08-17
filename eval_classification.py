"""
eval_classification.py

Measures classification accuracy against a held-out split of the real
UK Trade Tariff reference data (see references.split_train_eval) - not
a hand-labelled toy set, an honest train/index vs. test split of the
same authoritative data the tool is built on.

Reports top-1/top-3/top-5 exact-code accuracy, heading-level accuracy
(first 4 digits - the level most of this data is actually classified
to, and often what a freight forwarder cares about first), chapter-
level accuracy (first 2 digits), and mean reciprocal rank (MRR) - with
and without cross-encoder re-ranking, so the "does re-ranking help"
question has a real number behind it, in the same spirit as this
author's local-rag project.

Capped at MAX_EVAL_QUERIES (default 500) of the ~1,600 eligible held-out
queries: re-ranking runs one cross-encoder pass per query, and running
all ~1,600 would take a long time for a script meant to be run and
re-run during development. The subsample is a deterministic shuffle
(same DEFAULT_SEED as the train/eval split itself), not cherry-picked.

Also reports a confidence-bucket breakdown: candidates are grouped by
the score gap between the top and second result (classify.confidence_gap)
into quartiles, with top-1 accuracy reported per bucket. This is what
justifies (or would have disproven) adding a confidence gate - see
classify.CONFIDENCE_GAP_THRESHOLD, picked from this same measurement
rather than guessed, the same way local-rag's NO_ANSWER_THRESHOLD was
picked from eval_retrieval_gate.py's threshold sweep.

Not a pass/fail test - a measurement script. Run it directly:

    python eval_classification.py
"""

import random

from classify import SEARCH_POOL, _dedupe_by_code, build_index
from embeddings import embed_texts
from references import (
    DEFAULT_NOMENCLATURE_PATH,
    DEFAULT_SEED,
    Reference,
    load_references,
    split_train_eval,
)
from rerank import rerank as rerank_candidates

MAX_EVAL_QUERIES = 500
MAX_K = 5


def _rank_of_correct_code(candidates: list[tuple[Reference, float]], expected_code: str) -> int | None:
    for rank, (ref, _score) in enumerate(candidates, start=1):
        if ref.code == expected_code:
            return rank
    return None


def _report(label: str, ranks: list[int | None], eval_refs: list[Reference], top1_refs: list[Reference | None]) -> None:
    n = len(ranks)
    exact_at = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5)}
    mrr = sum((1 / r) if r is not None else 0.0 for r in ranks) / n

    heading_hits = sum(
        1 for expected, top1 in zip(eval_refs, top1_refs) if top1 is not None and top1.heading == expected.heading
    )
    chapter_hits = sum(
        1 for expected, top1 in zip(eval_refs, top1_refs) if top1 is not None and top1.chapter == expected.chapter
    )

    print(f"  {label}:")
    print(f"    exact code  - top1={exact_at[1]:.2f}  top3={exact_at[3]:.2f}  top5={exact_at[5]:.2f}  MRR={mrr:.2f}")
    print(f"    top1 heading match (first 4 digits): {heading_hits / n:.2f}")
    print(f"    top1 chapter match (first 2 digits):  {chapter_hits / n:.2f}")


def _run(
    index, eval_refs: list[Reference], use_rerank: bool
) -> tuple[list[int | None], list[Reference | None], list[float | None]]:
    queries = [ref.title for ref in eval_refs]
    query_vectors = embed_texts(queries)

    ranks: list[int | None] = []
    top1_refs: list[Reference | None] = []
    gaps: list[float | None] = []
    for expected, query, query_vector in zip(eval_refs, queries, query_vectors):
        candidates = index.search(query_vector, top_k=SEARCH_POOL)
        if use_rerank:
            candidates = rerank_candidates(query, candidates, top_k=SEARCH_POOL)
        deduped = _dedupe_by_code(candidates)[:MAX_K]

        ranks.append(_rank_of_correct_code(deduped, expected.code))
        top1_refs.append(deduped[0][0] if deduped else None)
        gaps.append(deduped[0][1] - deduped[1][1] if len(deduped) >= 2 else None)

    return ranks, top1_refs, gaps


def _report_confidence_buckets(ranks: list[int | None], gaps: list[float | None]) -> None:
    pairs = [(gap, rank) for gap, rank in zip(gaps, ranks) if gap is not None]
    pairs.sort(key=lambda p: p[0])

    n = len(pairs)
    quartile_size = n // 4
    print("  confidence-gap quartiles (top1 accuracy per bucket, lowest gap first):")
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else n
        bucket = pairs[start:end]
        if not bucket:
            continue
        gap_lo, gap_hi = bucket[0][0], bucket[-1][0]
        acc = sum(1 for _gap, rank in bucket if rank == 1) / len(bucket)
        print(f"    Q{q + 1} (gap {gap_lo:.2f} to {gap_hi:.2f}, n={len(bucket)}): top1 accuracy = {acc:.2f}")


def main() -> None:
    references = load_references()
    index_refs, eval_refs = split_train_eval(references)
    nomenclature_refs = load_references(DEFAULT_NOMENCLATURE_PATH)

    if len(eval_refs) > MAX_EVAL_QUERIES:
        random.Random(DEFAULT_SEED).shuffle(eval_refs)
        eval_refs = eval_refs[:MAX_EVAL_QUERIES]

    print(f"Eval: {len(eval_refs)} held-out queries.\n")

    print(f"=== Index: {len(index_refs)} reference titles only ===\n")
    index = build_index(index_refs)
    for label, use_rerank in [("cosine only", False), ("+ re-ranking", True)]:
        ranks, top1_refs, gaps = _run(index, eval_refs, use_rerank)
        _report(label, ranks, eval_refs, top1_refs)
        if use_rerank:
            # Reranked mode is what run.py defaults to, so that's the
            # mode the confidence gate needs to be calibrated against.
            _report_confidence_buckets(ranks, gaps)
        print()

    combined_refs = index_refs + nomenclature_refs
    print(
        f"=== Index: {len(index_refs)} reference titles + "
        f"{len(nomenclature_refs)} official nomenclature entries ===\n"
    )
    combined_index = build_index(combined_refs)
    for label, use_rerank in [("cosine only", False), ("+ re-ranking", True)]:
        ranks, top1_refs, gaps = _run(combined_index, eval_refs, use_rerank)
        _report(label, ranks, eval_refs, top1_refs)
        if use_rerank:
            _report_confidence_buckets(ranks, gaps)
        print()


if __name__ == "__main__":
    main()
