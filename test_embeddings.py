from embeddings import BGE_QUERY_PROMPT, query_prompt


def test_bge_style_models_get_the_query_prompt():
    assert query_prompt("Jayden04260/hs-code-embedder-bge-large") == BGE_QUERY_PROMPT
    assert query_prompt("BAAI/bge-base-en-v1.5") == BGE_QUERY_PROMPT


def test_models_without_an_instruction_get_none():
    assert query_prompt("all-MiniLM-L6-v2") == ""
    assert query_prompt("sentence-transformers/all-mpnet-base-v2") == ""
