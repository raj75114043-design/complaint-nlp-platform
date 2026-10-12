# ADR-0003: Time-based evaluation, held-out phrasings and novelty detection

**Status:** Accepted · 2026-10-12

## Context
Random train/test splits of template-like data overstate accuracy. In production the model always meets the future:
new phrasings, incidents and issues that did not exist at training time.

## Decision
Train only on complaints received before the cut-off (2026-09-01). Hold out two phrasings per issue from training and
report F1 separately for seen and unseen phrasings. Score novelty against the training set and route novel or
unclassifiable complaints to review; cluster them to surface emerging topics. Gate drift detection on both a no-change
period (must not alarm) and real/synthetic shifts (must alarm).

## Consequences
+ The headline numbers reflect deployment conditions, and the gap (0.71 unseen) motivates active learning, which closes it to 0.99 with 100 labels.
− The review rate (23% of known complaints) is a deliberate trade-off for catching 98% of new issues.
