# HS Code Classifier

![Tests](https://github.com/Jayden04260/hs-code-classifier/actions/workflows/tests.yml/badge.svg)

Given a free-text product description, suggests candidate UK commodity
(HS/tariff) codes - the codes freight forwarders and customs brokers
use to classify goods for import/export. Runs fully locally, no API
key, no cloud model call.

## Why

Getting the commodity code wrong on a customs declaration causes
delays, fines, and mis-charged duty. Classification is a genuinely
specialised skill - not something this tool claims to fully automate.
What it's actually useful for: narrowing thousands of possible codes
down to a handful of strong candidates, fast, so a human classifier's
job becomes "verify one of these five" instead of "search from
scratch." See "How accurate is it?" below for what that narrowing
actually achieves, measured honestly.

## Data source

[www.trade-tariff.service.gov.uk/api/v2/search_references](https://www.trade-tariff.service.gov.uk/api/v2/search_references)
- the free, public, unauthenticated JSON API that powers the UK
government's own ["Trade Tariff: look up commodity
codes"](https://www.gov.uk/trade-tariff) tool. Not the same as the
separate OAuth-gated `api.trade-tariff.service.gov.uk` developer API -
that one needs registered credentials; this one doesn't. Returns 9,108
(product name, commodity code) reference pairs - exactly the "common
name -> code" mapping this project needs, maintained by HMRC.

A local snapshot is bundled at `data/search_references.json` rather
than calling the live API at classification time - same local-first
reasoning as this author's other projects. Refresh it with:

```bash
python fetch_data.py
```

## How it works

```
data/search_references.json   (9,108 product-name -> code pairs)
        |
        v
  embeddings.py                 (sentence-transformers, local model)
        |
        v
  index.py                      (hand-rolled cosine-similarity search)
        |
        v
  rerank.py                     (cross-encoder re-scores candidates)
        |
        v
  classify.py                   (dedupe by code -> top-k candidates)
```

No LLM/GGUF model involved - classification is pure retrieval, so
there's no multi-gigabyte model download needed to run this. (An LLM
step that writes a plain-English justification citing the matched
reference is a natural extension - see Roadmap - but it's kept
separate from the classification decision itself, so accuracy numbers
stay deterministic and don't depend on a generation step.)

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

```
> wireless bluetooth headphones

  [1] 8517 00 00 00  (score: 3.12)
      matched: 'bluetooth wireless speaker adapters' (Heading)
  [2] 8518 30 00 90  (score: 2.59)
      matched: 'headphones, wireless' (Commodity)
  [3] 8517 62 00 00  (score: 2.59)
      matched: 'bluetooth headset' (Commodity)
```

Every candidate shows the reference title it actually matched against
and its score, so you can judge for yourself whether it's a good match
- not a black-box answer. Pass `--no-rerank` to skip cross-encoder
re-ranking (faster, and per the numbers below, not meaningfully less
accurate on this data).

## How accurate is it?

```bash
python eval_classification.py
```

Measured against a genuine held-out split of the same reference data
(500 titles never seen by the retrieval index - see `references.py`
for exactly how the split avoids leaking the answer), not a hand-picked
set:

| | exact code top-1 | top-3 | top-5 | MRR | top-1 heading match | top-1 chapter match |
|---|---|---|---|---|---|---|
| cosine only | 0.34 | 0.53 | 0.61 | 0.44 | 0.45 | 0.69 |
| + re-ranking | 0.34 | 0.53 | 0.62 | 0.45 | 0.45 | 0.69 |

Read honestly: getting the *exact* 10-digit code right on the first
guess happens about a third of the time - genuinely hard, which tracks
with classification being a real profession and not a lookup problem.
Getting the right **chapter** (broad product category, first 2 digits)
happens 69% of the time, and the right code is in the top 5 candidates
61-62% of the time - that's the tool doing its actual job, narrowing
the search space fast, not resolving it end to end.

**Re-ranking barely moves the numbers here** - unlike this author's
local-rag project, where re-ranking measurably fixed a real recall gap.
The likely reason: local-rag's re-ranking corpus had genuine cross-
document vocabulary overlap (a decoy document sharing keywords with the
right one); here, both the query and every candidate are short official
reference titles drawn from the *same* vocabulary register, so there's
less ambiguity for a cross-encoder to resolve. Reported rather than
hidden - the point of measuring is finding out what's actually true,
including when a technique that helped elsewhere doesn't help here.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

Tests mock the network fetch and heavy models where the underlying
logic doesn't need them (deduplication, train/eval splitting, index
search); `test_rerank.py` runs the real small cross-encoder model
directly, same testing philosophy as this author's other local-first
projects.

## Limitations

- The 9,108 reference titles are short, clean product names ("abrasive
  stones, natural"), not messy real invoice line items. No public
  dataset pairing realistic free-text product descriptions with HS
  codes was found while building this - if one exists, swapping in a
  more realistic eval set would be a natural next step.
- UK-specific (10-digit UK Trade Tariff codes). The first 6 digits are
  the internationally standardized HS code, but the last 4 are a UK
  extension - codes won't directly match another country's national
  tariff schedule past that point.
- Not legal advice and shouldn't be the sole basis for a customs
  declaration - see "Why" above.

## Roadmap

- LLM-generated plain-English justification for a candidate, citing
  the matched reference and (via the Trade Tariff API's chapter/section
  endpoints) the relevant legal classification notes - closer to how a
  human customs classifier actually justifies a decision.
- Index the official heading/chapter nomenclature text itself, not just
  the 9,108 common-name references, for broader semantic coverage.
