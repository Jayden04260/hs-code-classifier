"""
embeddings.py

Thin wrapper around sentence-transformers for turning text into
L2-normalized embedding vectors.

The default model is BAAI/bge-large-en-v1.5 fine-tuned on this project's
own tariff data (see train_embedder.py and README "How accurate is it?"):
on 632 real e-commerce listings classified by customs experts it picks
the right 4-digit heading 42.7% of the time vs 29.9% for the previous
all-MiniLM-L6-v2 + cross-encoder setup. Override with the
HS_EMBED_MODEL environment variable (any sentence-transformers model
name or local path) - e.g. HS_EMBED_MODEL=all-MiniLM-L6-v2 for the old,
~15x smaller model on a slow machine.

Queries get bge's retrieval instruction prefix; reference titles are
embedded as-is. L2-normalizing here (rather than at the call site) means
every embedding this module produces is comparable via plain dot product.
"""

import os

import numpy as np

DEFAULT_MODEL_NAME = os.environ.get("HS_EMBED_MODEL", "Jayden04260/hs-code-embedder-bge-large")

# bge/mxbai-style retrieval instruction, prepended to queries only. The
# fine-tuned default was trained with exactly this prefix on its queries.
# Models that don't use one (e.g. all-MiniLM-L6-v2) get no prefix.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
_NO_PROMPT_MODELS = ("all-MiniLM", "all-mpnet", "paraphrase-")

_model = None
_model_name = None


def query_prompt(model_name: str = DEFAULT_MODEL_NAME) -> str:
    return "" if any(tag in model_name for tag in _NO_PROMPT_MODELS) else BGE_QUERY_PROMPT


def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME, prompt: str = "") -> np.ndarray:
    """
    Embed a list of strings into an (n, dim) array of L2-normalized
    float32 vectors. Empty input returns an (0, dim)-shaped array rather
    than erroring, so callers don't need to special-case an empty index.
    `prompt` is prepended to every text (used for queries).
    """
    model = _get_model(model_name)
    if not texts:
        return np.zeros((0, model.get_embedding_dimension()), dtype=np.float32)

    vectors = model.encode(
        [prompt + t for t in texts], batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.astype(np.float32)


def embed_queries(queries: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Embed product-description queries (with the model's query prefix, if it uses one)."""
    return embed_texts(queries, model_name, prompt=query_prompt(model_name))


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Embed a single query string, returning a 1D (dim,) vector."""
    return embed_queries([query], model_name)[0]
