"""Human-in-the-loop active learning.

Production complaints received after the training cut-off are scored; the most informative ones (novel, other, or
lowest-margin) are sent to reviewers. Their labels are added to the training set and a challenger model is trained.
The challenger is promoted only if it beats the champion on the held-out test set without regressing on seen phrasing.
"""
from __future__ import annotations

import numpy as np

from cnlp.model import CUTOFF, ComplaintModel, train


def select_for_review(model: ComplaintModel, pool: list[dict], budget: int) -> list[dict]:
    preds = model.predict([r["text"] for r in pool])
    scores = []
    for r, p in zip(pool, preds, strict=True):
        probs = sorted(p["issue_probs"].values(), reverse=True)
        margin = probs[0] - probs[1]
        scores.append((p["novelty"] + (1 - margin) + (1.0 if p["other"] else 0.0), r))
    scores.sort(key=lambda x: -x[0])
    return [r for _, r in scores[:budget]]


def retrain_with_feedback(rows: list[dict], reviewed: list[dict], version: str) -> ComplaintModel:
    """Reviewed complaints (with human labels) join the training data; out-of-taxonomy ones are left for a taxonomy change."""
    extra = [{**r, "split": "train", "received": "2026-08-31"} for r in reviewed if not r["label_other"]]
    ids = {r["complaint_id"] for r in extra}
    return train([r for r in rows if r["complaint_id"] not in ids] + extra, version)


def production_pool(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["received"] >= CUTOFF and r["split"] in ("train", "val")]


def summary(before: dict, after: dict) -> dict:
    return {k: {"before": before[k], "after": after[k], "delta": round(after[k] - before[k], 3)} for k in before if isinstance(before[k], float)}


__all__ = ["select_for_review", "retrain_with_feedback", "production_pool", "summary", "np"]
