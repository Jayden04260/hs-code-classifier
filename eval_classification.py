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

Not a pass/fail test - a measurement script. Run it directly:

    python eval_classification.py
"""

import random

from classify import SEARCH_POOL, _dedupe_by_code, build_index
from embeddings import embed_texts
from references import DEFAULT_SEED, Reference, load_references, split_train_eval
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


def _run(index, eval_refs: list[Reference], use_rerank: bool) -> tuple[list[int | None], list[Reference | None]]:
    queries = [ref.title for ref in eval_refs]
    query_vectors = embed_texts(queries)

    ranks: list[int | None] = []
    top1_refs: list[Reference | None] = []
    for expected, query, query_vector in zip(eval_refs, queries, query_vectors):
        candidates = index.search(query_vector, top_k=SEARCH_POOL)
        if use_rerank:
            candidates = rerank_candidates(query, candidates, top_k=SEARCH_POOL)
        deduped = _dedupe_by_code(candidates)[:MAX_K]

        ranks.append(_rank_of_correct_code(deduped, expected.code))
        top1_refs.append(deduped[0][0] if deduped else None)

    return ranks, top1_refs


def main() -> None:
    references = load_references()
    index_refs, eval_refs = split_train_eval(references)

    if len(eval_refs) > MAX_EVAL_QUERIES:
        random.Random(DEFAULT_SEED).shuffle(eval_refs)
        eval_refs = eval_refs[:MAX_EVAL_QUERIES]

    print(f"Index: {len(index_refs)} references. Eval: {len(eval_refs)} held-out queries.\n")

    index = build_index(index_refs)

    for label, use_rerank in [("cosine only", False), ("+ re-ranking", True)]:
        ranks, top1_refs = _run(index, eval_refs, use_rerank)
        _report(label, ranks, eval_refs, top1_refs)
        print()


if __name__ == "__main__":
    main()
