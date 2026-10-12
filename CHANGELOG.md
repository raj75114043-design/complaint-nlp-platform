# Changelog

## 1.0.0 — 2026-10-12
- Synthetic, reproducible complaint stream (6,000 complaints; English, Hinglish, terse; typos; PII; decoys; spike; new issue)
- PII redaction with Luhn / Verhoeff validation and fuzzy context; payment references preserved
- Multi-label issue and product classification with per-label thresholds and novelty detection
- Triage: urgency, regulatory risk, entities, queues, SLA; guarded reply drafts (template or LLM)
- Monitoring: spike detection, emerging topics, drift; active learning with champion/challenger registry
- FastAPI, Streamlit console, Docker, CI quality gate, GitLab mirror
