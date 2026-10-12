# Architecture

| Stage | Module | Notes |
|---|---|---|
| Redaction | `cnlp/pii.py` | Spans with type and position; replacement tokens `[CARD]`, `[AADHAAR]`, … |
| Normalisation | `cnlp/model.py::normalise` | Redact → placeholders for amounts, dates, references, masked cards |
| Classification | `cnlp/model.py` | Product (multinomial LR), issues (OvR LR + per-label thresholds), novelty |
| Triage | `cnlp/triage.py` | Urgency reasons, priority score, entities, queue, SLA, needs-review flag |
| Reply draft | `cnlp/pipeline.py` | Template, or LLM accepted only if it passes the promise / digit / PII guard |
| Monitoring | `cnlp/monitoring.py` | Weekly spikes, emerging topics (NMF), drift (PSI, OOV, novelty) |
| Learning loop | `cnlp/active.py`, `cnlp/cli.py train` | Review selection, retraining, champion/challenger registry in `models/registry.json` |

Queues and SLAs: Fraud Response (7 days), Executive Escalations (7), Payments Disputes (10), others 30; novel or
unclassifiable complaints go to Triage Review.
