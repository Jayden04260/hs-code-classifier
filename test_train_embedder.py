from references import load_references, split_train_eval
from train_embedder import build_pairs


def test_training_pairs_never_contain_held_out_titles():
    _index_refs, held_out = split_train_eval(load_references())
    anchors, positives = build_pairs(query_prompt="")
    seen = set(anchors) | set(positives)
    # a held-out title may coincide with an identical title kept for another code,
    # so only titles that appear nowhere in the index side must be absent
    index_titles = {r.title for r in _index_refs}
    leaked = [r.title for r in held_out if r.title in seen and r.title not in index_titles]
    assert leaked == []


def test_pairs_are_aligned_and_prompted():
    anchors, positives = build_pairs(query_prompt="Q: ")
    assert len(anchors) == len(positives) > 10_000
    assert all(a.startswith("Q: ") for a in anchors)
    assert not any(p.startswith("Q: ") for p in positives)
