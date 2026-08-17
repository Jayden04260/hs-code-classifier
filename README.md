# HS Code Classifier

![Tests](https://github.com/Jayden04260/hs-code-classifier/actions/workflows/tests.yml/badge.svg)

Given a free-text product description, suggests candidate UK commodity
(HS/tariff) codes - the codes freight forwarders and customs brokers
use to classify goods for import/export - and looks up the standard
duty rate for the top match. Runs fully locally, no API key, no cloud
model call for the classification itself.

## Why

Getting the commodity code wrong on a customs declaration causes
delays, fines, and mis-charged duty. Classification is a genuinely
specialised skill - not something this tool claims to fully automate.
What it's actually useful for: narrowing thousands of possible codes
down to a handful of strong candidates, fast, so a human classifier's
job becomes "verify one of these five" instead of "search from
scratch." It also flags when it isn't sure (see "Confidence gate"
below) instead of presenting a shaky guess with the same weight as a
confident one - the same industry-standard behaviour commercial tools
like [Zonos Classify](https://zonos.com/docs/supply-chain/classify/features/confidence-score)
use, not a novel idea, but a deliberately-included one.

## Data sources

Two, both from the same free, public, unauthenticated part of the UK
government's Trade Tariff API - not the separate OAuth-gated
`api.trade-tariff.service.gov.uk` developer API, which needs registered
credentials.

- **`search_references`** (`fetch_data.py`) - 9,108 informal product-name
  -> commodity-code pairs (e.g. "abrasive stones, natural" -> a specific
  code). The same data that powers the UK government's own ["Trade
  Tariff: look up commodity codes"](https://www.gov.uk/trade-tariff)
  search tool.
- **Chapter/heading nomenclature** (`fetch_nomenclature.py`) - the
  official, longer-form legal classification text for all 98 chapters
  and ~1,264 headings (e.g. "Machines for the reception, conversion and
  transmission or regeneration of voice, images or other data...").
  Broadens what the retrieval step can match against beyond the short
  informal titles - see "How accurate is it?" for the measured effect.

Both are bundled as local snapshots in `data/` rather than called live
at classification time - same local-first reasoning as this author's
other projects. Refresh them with:

```bash
python fetch_data.py
python fetch_nomenclature.py
```

## How it works

```
data/search_references.json + data/nomenclature.json
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
  classify.py                   (dedupe by code -> top-k candidates,
        |                        + confidence-gap check)
        v
  duty.py                       (live lookup: standard duty rate for
                                  the top candidate)
```

No LLM/GGUF model involved - classification is pure retrieval, so
there's no multi-gigabyte model download needed to run this.

## Usage

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

```
> wireless bluetooth headphones

  [low confidence - top candidates scored close together; this description may be genuinely ambiguous]

  [1] 8517 00 00 00  (score: 3.12)
      matched: 'bluetooth wireless speaker adapters' (Heading)
  [2] 8518 30 00 90  (score: 2.59)
      matched: 'headphones, wireless' (Commodity)
  [3] 8517 62 00 00  (score: 2.59)
      matched: 'bluetooth headset' (Commodity)
```

(Real output, not cleaned up for the README - this query happens to be
a case the confidence gate correctly flags: the top match is a heading
one level too broad, and the actual best answer is rank 2. That's the
gate doing its job, not a cherry-picked failure.)

Every candidate shows the reference title it actually matched against
and its score, so you can judge for yourself whether it's a good match
- not a black-box answer. Pass `--no-rerank` to skip cross-encoder
re-ranking, `--no-duty` to skip the live duty-rate lookup.

Or run the web version (no LLM download needed here either, so this is
genuinely lightweight to try):

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

Or double-click `run.bat` (creates a `.venv` and installs both
`requirements.txt` and `requirements-app.txt` on first run if one
doesn't exist yet) to launch the web version without touching a
terminal. `hs-code-classifier.ico` is a matching icon if you want to
set one on a desktop shortcut to it (right-click the shortcut ->
Properties -> Change Icon).

## Confidence gate

When the score gap between the top and second candidate is small, the
tool flags the result as low-confidence instead of presenting it with
the same visual weight as a clear match - the description may
genuinely be ambiguous between two plausible codes. `CONFIDENCE_GAP_THRESHOLD`
in `classify.py` (1.6) wasn't guessed - it's picked from
`eval_classification.py`'s confidence-gap quartile breakdown: queries
in the bottom two quartiles (small gap) averaged **31%** top-1 accuracy;
the top two quartiles (large gap) averaged **53%** - a real, measured
split. Same reasoning as this author's local-rag project's
`NO_ANSWER_THRESHOLD`, applied to a different problem.

## Duty rate lookup

`duty.py` fetches the standard ("Third country duty", i.e. the rate
that applies unless a specific trade agreement or preference
overrides it) rate for the top candidate, live, from the Trade Tariff
API. Verified against real, independently-checkable cases before being
kept in: T-shirts (6109100010) returned 12.00%, matching the
widely-known real-world apparel rate; a random sample of 5 other
commodity codes all returned plausible, correctly-shaped percentages.
Deliberately narrow scope - only the standard rate, not origin-specific
preferential rates or quotas, which depend on more than the code alone
- see `duty.py`'s docstring.

## How accurate is it?

```bash
python eval_classification.py    # held-out split of the real reference data
python eval_realistic.py         # small hand-written realistic-description set
```

**Held-out reference-title split** (500 titles never seen by the
retrieval index - see `references.py` for exactly how the split avoids
leaking the answer):

| corpus | mode | exact top-1 | top-3 | top-5 | MRR | heading match | chapter match |
|---|---|---|---|---|---|---|---|
| titles only | cosine | 0.34 | 0.53 | 0.61 | 0.44 | 0.45 | 0.69 |
| titles only | + re-ranking | 0.34 | 0.53 | 0.62 | 0.45 | 0.45 | 0.69 |
| + nomenclature | cosine | 0.38 | 0.58 | 0.66 | 0.48 | 0.49 | 0.71 |
| **+ nomenclature** | **+ re-ranking** | **0.42** | **0.60** | **0.68** | **0.52** | **0.56** | **0.75** |

The bottom row is what this ships with. Two things worth calling out
honestly:

- **Adding the official nomenclature text measurably helped** - exact
  top-1 accuracy went from 0.34 to 0.42, chapter accuracy from 0.69 to
  0.75. This was tested, not assumed, before being kept.
- **Re-ranking's benefit depends on the corpus.** Against titles only,
  it barely moved the numbers (0.34 -> 0.34 top-1) - both query and
  candidate are short titles from the same vocabulary register, leaving
  little for a cross-encoder to resolve. Against the combined corpus,
  where candidates now range from three-word titles to full legal
  descriptions, it measurably helps (0.38 -> 0.42 top-1). Reported both
  ways rather than only the flattering one.

**Hand-written realistic descriptions** (20 examples, e.g. "Men's 100%
cotton crew-neck t-shirts, machine knitted, various colors, retail
packed" rather than the reference data's terse "T-shirts" - see
`eval_realistic.py` for the full set and how ground truth stays honest:
every expected code is taken directly from the real reference data,
only the query phrasing is hand-written):

**top1 = 0.70, top5 = 0.90, MRR = 0.80**

Higher than the reference-title eval, which was initially surprising -
the likely reason is that longer, more specific descriptions give the
embedding model more distinguishing text to work with than a three-word
official title does. Not every case worked: "fresh star fruit" matched
"kiwi fruit, fresh" instead (a plausible tropical-fruit mix-up) and "a
veterinary examination table" matched "garnish pick, of stainless
steel" (a bad miss - "stainless steel" keyword overlap apparently
outweighed the actual product category for that one). Both failures are
printed by `eval_realistic.py`, not hidden.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

Tests mock the network fetch and heavy models where the underlying
logic doesn't need them (deduplication, train/eval splitting, index
search, duty-rate extraction); `test_rerank.py` runs the real small
cross-encoder model directly, same testing philosophy as this author's
other local-first projects.

## Limitations

- The realistic-description eval is small (20 hand-written examples) -
  useful as a sanity check on real-world phrasing, not a statistically
  powered claim the way the 500-query held-out split is. No larger
  public dataset pairing realistic descriptions with verified HS codes
  was found while building this.
- UK-specific (10-digit UK Trade Tariff codes). The first 6 digits are
  the internationally standardized HS code, but the last 4 are a UK
  extension - codes won't directly match another country's national
  tariff schedule past that point.
- Duty rate lookup covers only the standard (ERGA OMNES) rate - not
  preferential trade-agreement rates, quotas, or anti-dumping duties.
- Not legal advice and shouldn't be the sole basis for a customs
  declaration - see "Why" above.

## Roadmap

- LLM-generated plain-English justification for a candidate, citing
  the matched reference and the relevant chapter/section legal notes -
  closer to how a human customs classifier actually justifies a
  decision. Kept out of the core classification path deliberately, so
  accuracy numbers stay deterministic and don't depend on a generation
  step.
