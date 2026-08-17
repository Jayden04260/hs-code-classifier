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

- `data/search_references.json` is a committed snapshot, not fetched
  live - refresh it with `python fetch_data.py` if you need current
  tariff data, but don't assume CI or a fresh clone has network access
  to the live API.
- Keep dependencies in `requirements.txt` minimal; add dev-only tools to
  `requirements-dev.txt`.
- `eval_classification.py` isn't a pass/fail test - it's a measurement
  script. If you change `index.py`/`rerank.py`/`classify.py`, re-run it
  and update the numbers in README.md's "How accurate is it?" table
  rather than letting them go stale.
