"""
train_embedder.py

Fine-tunes the retrieval embedder on the tariff data itself - how the
default model (Jayden04260/hs-code-embedder-bge-large) was made.

Contrastive training (MultipleNegativesRankingLoss): each pair says "these
two texts mean the same commodity", and every other pair in the batch
serves as a negative. Pairs, all built from the *retrieval-index side* of
references.split_train_eval - so the 500 held-out titles that
eval_classification.py scores stay unseen, and HSCodeComp
(eval_hscodecomp.py) is never used for training at all:

  - a reference title -> another reference title for the same code
  - a reference title -> its heading's official nomenclature text
    ("headphones, wireless" -> "Microphones and stands therefor; loudspeakers...")
  - a nomenclature heading -> its chapter text (teaches the hierarchy)

What was measured along the way (README "How accurate is it?" has the
tables): one epoch is the sweet spot - two epochs overfit to the short,
formal title register and lost 8 points of heading accuracy on real
listings. Use plain MultipleNegativesRankingLoss: the cached (GradCache)
variant collapsed every run on this setup (all embeddings identical),
and mxbai-embed-large-v1 collapsed even with the plain loss, while
bge-base and bge-large trained stably.

Takes ~15 minutes on a 16GB laptop GPU (RTX 5080) at batch size 32.

    pip install -r requirements-train.txt
    python train_embedder.py                        # bge-large, as shipped
    python train_embedder.py --base BAAI/bge-base-en-v1.5 --batch-size 64   # ~3x smaller, ~1 point worse
    HS_EMBED_MODEL=models/hs-embedder python eval_hscodecomp.py
"""

import argparse
import random

from embeddings import BGE_QUERY_PROMPT
from references import DEFAULT_NOMENCLATURE_PATH, load_references, split_train_eval


def build_pairs(query_prompt: str = BGE_QUERY_PROMPT, seed: int = 0) -> tuple[list[str], list[str]]:
    """(anchors, positives) from the retrieval-index references and the nomenclature."""
    index_refs, _held_out = split_train_eval(load_references())
    nomenclature = load_references(DEFAULT_NOMENCLATURE_PATH)
    nomenclature_by_code = {n.code: n.title for n in nomenclature}
    rng = random.Random(seed)

    titles_by_code: dict[str, list[str]] = {}
    for ref in index_refs:
        titles_by_code.setdefault(ref.code, []).append(ref.title)

    anchors, positives = [], []
    for code, titles in titles_by_code.items():
        heading = nomenclature_by_code.get(code[:4] + "000000")
        for title in titles:
            synonyms = [t for t in titles if t != title]
            if synonyms:
                anchors.append(query_prompt + title)
                positives.append(rng.choice(synonyms))
            if heading and heading != title:
                anchors.append(query_prompt + title)
                positives.append(heading)
    for entry in nomenclature:
        if entry.level == "Heading":
            chapter = nomenclature_by_code.get(entry.code[:2] + "00000000")
            if chapter:
                anchors.append(query_prompt + entry.title)
                positives.append(chapter)
    return anchors, positives


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the HS code retrieval embedder.")
    parser.add_argument("--base", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--out", default="models/hs-embedder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32, help="32 fits bge-large in 16GB")
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    torch.backends.cuda.matmul.allow_tf32 = True
    anchors, positives = build_pairs()
    print(f"{len(anchors)} training pairs")
    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives}).shuffle(seed=0)

    model = SentenceTransformer(args.base, device="cuda" if torch.cuda.is_available() else "cpu")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.out + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # a duplicate text in a batch would be a false negative
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=0,
    )
    SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
    ).train()
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
