"""
classify.py

Orchestrates the pipeline: build a retrieval index over reference
titles, then for a free-text product description, retrieve and
optionally re-rank candidate commodity codes.

Deduplication matters here in a way it doesn't for a plain document RAG
pipeline: 1,624 of the 3,071 distinct codes in this data have more than
one reference title ("abalone" and "abalone, prepared or preserved" are
different titles that could both legitimately match a query, but they
point at different codes so that's fine; more often, near-duplicate
synonyms of the *same* code shouldn't each occupy a separate slot in
the top-k). Candidates are deduplicated by code, keeping each code's
best-scoring synonym match, so the final top_k is k distinct candidate
codes, not k possibly-repeated ones.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embeddings import DEFAULT_MODEL_NAME, embed_query, embed_texts
from index import ReferenceIndex
from references import Reference
from rerank import rerank as rerank_candidates

# How many candidates cosine search retrieves (and, if reranking, how
# many survive reranking) before deduplication narrows down to top_k
# distinct codes. Wider than top_k on purpose - several of these raw
# hits often collapse into the same code once deduplicated, so top_k
# alone wouldn't leave enough room for top_k *distinct* codes to survive.
SEARCH_POOL = 20

# Below this score gap between the top and second candidate, treat the
# classification as low-confidence. Picked from eval_classification.py's
# confidence-gap quartile breakdown, run against the configuration this
# ships with (fine-tuned embedder, cosine scores, no re-ranking): held-out
# queries in the bottom two quartiles (gap < ~0.04) averaged 33% top-1
# accuracy, the top two 68%. Checked on HSCodeComp's real listings too
# (eval_hscodecomp.py), where heading accuracy climbs from 27% in the
# lowest-gap quartile to 68% in the highest. A measured split, not a guessed
# number. (The previous 1.6 was calibrated on cross-encoder logits, a
# different scale - it would flag every cosine result.) Same reasoning as
# local-rag's NO_ANSWER_THRESHOLD: don't present a low-confidence guess
# with the same visual weight as a confident one.
CONFIDENCE_GAP_THRESHOLD = 0.04


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str  # the reference title that produced this match
    level: str  # Chapter/Heading/Subheading/Commodity - see references.Reference
    score: float


# Embedding ~10k reference titles with the default (large) model takes a
# few seconds on a GPU but a minute or two on a CPU, so the vectors are
# cached on disk, keyed by model name + the exact titles - any change to
# either (e.g. after fetch_data.py refreshes the snapshot) misses the
# cache and re-embeds.
CACHE_DIR = Path(os.environ.get("HS_CACHE_DIR", Path.home() / ".cache" / "hs-code-classifier"))


def _cache_path(titles: list[str], model_name: str) -> Path:
    digest = hashlib.sha1("\n".join([model_name, *titles]).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"index-{digest}.npy"


def build_index(
    references: list[Reference], model_name: str = DEFAULT_MODEL_NAME, use_cache: bool = True
) -> ReferenceIndex:
    titles = [r.title for r in references]
    path = _cache_path(titles, model_name)
    if use_cache and path.exists():
        vectors = np.load(path)
        if vectors.shape[0] == len(references):
            return ReferenceIndex(vectors, references)
    vectors = embed_texts(titles, model_name)
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        except OSError:
            pass  # caching is an optimisation; a read-only home dir shouldn't break classification
    return ReferenceIndex(vectors, references)


def _dedupe_by_code(candidates: list[tuple[Reference, float]]) -> list[tuple[Reference, float]]:
    best_per_code: dict[str, tuple[Reference, float]] = {}
    for ref, score in candidates:
        current = best_per_code.get(ref.code)
        if current is None or score > current[1]:
            best_per_code[ref.code] = (ref, score)
    return sorted(best_per_code.values(), key=lambda pair: pair[1], reverse=True)


def classify(
    query: str,
    index: ReferenceIndex,
    top_k: int = 5,
    use_rerank: bool = False,
) -> list[Candidate]:
    """
    Return up to top_k distinct candidate commodity codes for `query`,
    highest confidence first.
    """
    query_vector = embed_query(query)
    raw = index.search(query_vector, top_k=SEARCH_POOL)

    if use_rerank:
        raw = rerank_candidates(query, raw, top_k=SEARCH_POOL)

    deduped = _dedupe_by_code(raw)[:top_k]
    return [Candidate(code=ref.code, title=ref.title, level=ref.level, score=score) for ref, score in deduped]


def confidence_gap(candidates: list[Candidate]) -> float | None:
    """
    Score gap between the top and second candidate - see
    eval_classification.py's confidence-bucket report for why this is a
    reasonable proxy for "how sure should you be about the top result":
    a large gap means the top candidate stood out clearly; a small one
    means the model was choosing between near-equally-plausible codes.
    None if there's only one (or zero) candidates - nothing to compare
    the top one against.
    """
    if len(candidates) < 2:
        return None
    return candidates[0].score - candidates[1].score
