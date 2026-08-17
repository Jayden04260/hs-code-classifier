# Contributing

## Setup

```bash
git clone https://github.com/Jayden04260/hs-code-classifier.git
cd hs-code-classifier
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements.txt
```

## Running checks

```bash
pytest -v
ruff check .
```

Both run in CI on every push and pull request.

## Notes for contributors

- `data/search_references.json` and `data/nomenclature.json` are
  committed snapshots, not fetched live - refresh them with
  `python fetch_data.py` / `python fetch_nomenclature.py` if you need
  current tariff data, but don't assume CI or a fresh clone has network
  access to the live API.
- Keep dependencies in `requirements.txt` minimal; add dev-only tools to
  `requirements-dev.txt` and the Streamlit UI's dependency to
  `requirements-app.txt` (`app.py` isn't imported by anything else, so
  it doesn't need to be in the base install).
- `eval_classification.py` and `eval_realistic.py` aren't pass/fail
  tests - they're measurement scripts. If you change
  `index.py`/`rerank.py`/`classify.py`, re-run both and update the
  numbers in README.md's "How accurate is it?" section, and re-check
  `classify.CONFIDENCE_GAP_THRESHOLD` against the new confidence-gap
  quartile breakdown rather than leaving it calibrated to stale numbers.
