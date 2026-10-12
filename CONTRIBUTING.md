# Contributing

1. `pip install -r requirements-dev.txt`
2. To add an issue type, phrasing or PII type, edit `cnlp/data/generator.py` and run `make data`; keep two held-out phrasings per issue.
3. `make lint test eval` must pass. Never lower a gate threshold to make a change pass; explain metric changes in the PR.
4. PII leaks and missed fraud / regulatory escalations block a merge. Never commit real customer data.
