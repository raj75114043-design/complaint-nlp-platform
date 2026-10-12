# Evaluation method

- **Split:** training = complaints received before 2026-09-01 (minus a validation slice); test = held-out split.
- **Seen vs unseen phrasing:** each issue has 8 phrasings; 2 never appear in training.
- **PII:** gold spans from the generator; recall per type, decoy preservation, false-redaction rate.
- **New issues:** complaints with no taxonomy label must be flagged (novel or other); known complaints flagged = review load.
- **Spikes:** the planted W37 refund-delay spike must be found with no other issue alarm (unclassified surges are a separate alert kind).
- **Emerging topics:** the e-mandate topic must be among the emerging NMF topics.
- **Drift:** Jul→Aug must not alarm; Jul→Sep and a synthetic noisy-transcript channel must alarm.
- **Active learning:** 100 reviewer labels chosen by the model vs 100 random labels from post-cut-off complaints; compared on unseen-phrasing F1.

Results are written to `reports/eval.json` and uploaded as a CI artifact.
