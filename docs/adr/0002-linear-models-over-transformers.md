# ADR-0002: TF-IDF + linear models instead of a fine-tuned transformer

**Status:** Accepted · 2026-10-12

## Context
Complaints are short, noisy (typos, Hinglish transliteration) and the taxonomy changes. The model must retrain in
seconds after reviewers label new examples, run on CPU, and be explainable to operations and audit.

## Decision
Word 1–2-gram and character 2–5-gram TF-IDF features with logistic regression (one-vs-rest for issues, per-label
thresholds tuned on validation). Character n-grams absorb spelling variation. A keyword baseline is evaluated alongside
and must be beaten.

## Consequences
+ Training takes ~10 s, so active learning and champion/challenger cycles are cheap; the hosted demo trains on start.
+ Coefficients are inspectable per label.
− Paraphrases with no lexical overlap are the weak spot (unseen-phrasing F1 0.71 before feedback). The registry accepts a transformer challenger under the same gate.
