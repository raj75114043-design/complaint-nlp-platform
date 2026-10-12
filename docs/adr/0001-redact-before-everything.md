# ADR-0001: Redact before any model, log or LLM sees the text

**Status:** Accepted · 2026-10-12

## Context
Complaints contain card, Aadhaar, PAN, account and phone numbers. They also contain payment references (UTR, UPI
reference numbers) that investigators need. Generic "redact every long number" rules destroy the references; generic
NER misses typo'd context ("acount numbr").

## Decision
Redaction is the first step of the pipeline. Numbers are classified by checksum (Luhn for cards, Verhoeff for Aadhaar)
and by fuzzy context (SequenceMatcher ≥ 0.8 against account and reference keywords). Reference context wins over the
generic number rule; an explicit Aadhaar label wins over reference context. The classifier, logs, API responses and the
LLM only receive redacted text; the API returns the PII *types* found, never the values.

## Consequences
+ 100% recall on every PII type and 0 decoys redacted on the evaluation set; rules are auditable.
+ A leak caught in review becomes a test case, not a retraining job.
− Free-text names and addresses without salutations need a learned NER model.
