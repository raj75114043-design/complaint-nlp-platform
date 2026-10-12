"""Evaluation and quality gate.

    python -m cnlp.eval.run --gate

PII · classification (time-based split, seen vs unseen phrasing, by language style, vs keyword baseline) · urgency ·
new-issue detection · spike detection · emerging topics · drift (no false alarm; real and synthetic shift) ·
active learning (labels chosen by the model vs random labels, same budget).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from cnlp.active import production_pool, retrain_with_feedback, select_for_review
from cnlp.data.generator import ISSUES, _typos
from cnlp.model import baseline_predict, train
from cnlp.monitoring import detect_spikes, drift, emerging_topics, weekly_counts
from cnlp.pii import redact
from cnlp.pipeline import load_rows
from cnlp.triage import assess

THRESHOLDS = {"pii_min_recall": 0.99, "pii_decoy_preserved": 0.99, "pii_false_redaction_rate": 0.01, "issue_f1_micro": 0.85, "issue_f1_seen": 0.95,
              "beats_baseline": 1.0, "urgency_recall": 0.9, "urgency_precision": 0.8, "novel_recall": 0.85, "known_flag_rate_max": 0.25,
              "spike_found": 1.0, "spike_false_alarms_max": 0, "emerging_topic_found": 1.0, "drift_no_false_alarm": 1.0, "drift_detected_shift": 1.0,
              "active_learning_gain": 0.10, "active_beats_random": 1.0}
MAX_KEYS = {"pii_false_redaction_rate", "known_flag_rate_max", "spike_false_alarms_max"}
MLB = MultiLabelBinarizer(classes=ISSUES).fit([[]])


def _f1(rows, preds, avg="micro"):
    return float(f1_score(MLB.transform([r["issues"] for r in rows]), MLB.transform([p["issues"] for p in preds]), average=avg, zero_division=0))


def pii_metrics(rows) -> dict:
    miss, tot, decoy_ok, decoys, false_red, reds = Counter(), Counter(), 0, 0, 0, 0
    for r in rows:
        red, spans = redact(r["text"])
        for p in r["pii"]:
            tot[p["type"]] += 1
            miss[p["type"]] += p["value"] in red
        for d in r["decoys"]:
            decoys += 1
            decoy_ok += d in red
        gv = [p["value"] for p in r["pii"]]
        reds += len(spans)
        false_red += sum(1 for s in spans if not any(s.value in g or g in s.value for g in gv))
    rec = {k: 1 - miss[k] / tot[k] for k in tot}
    return {"pii_recall_by_type": {k: round(v, 4) for k, v in rec.items()}, "pii_min_recall": min(rec.values()),
            "pii_decoy_preserved": decoy_ok / max(1, decoys), "pii_false_redaction_rate": false_red / max(1, reds), "pii_items": sum(tot.values())}


def evaluate(seed: int = 0) -> dict:
    rows = load_rows()
    model = train(rows, "eval")
    test = [r for r in rows if r["split"] == "test" and not r["label_other"]]
    preds = model.predict([r["text"] for r in test])
    base = baseline_predict([r["text"] for r in test])
    m = {**pii_metrics([r for r in rows if r["split"] == "test"])}
    m["issue_f1_micro"], m["issue_f1_macro"] = _f1(test, preds), _f1(test, preds, "macro")
    m["baseline_f1_micro"] = _f1(test, base)
    m["beats_baseline"] = float(m["issue_f1_micro"] > m["baseline_f1_micro"])
    m["product_accuracy"] = accuracy_score([r["product"] for r in test], [p["product"] for p in preds])
    for flag, name in ((False, "seen"), (True, "unseen")):
        sub = [(r, p) for r, p in zip(test, preds, strict=True) if r["unseen_phrasing"] == flag]
        m[f"issue_f1_{name}"] = _f1([r for r, _ in sub], [p for _, p in sub])
    m["issue_f1_by_style"] = {st: round(_f1([r for r in test if r["style"] == st], [p for r, p in zip(test, preds, strict=True) if r["style"] == st]), 3)
                              for st in ("en", "hi", "terse")}
    tri = [assess(r["text"], p, r["received"]) for r, p in zip(test, preds, strict=True)]
    y, yh = [r["urgency"] == "high" for r in test], [t["urgency"] == "high" for t in tri]
    m["urgency_recall"], m["urgency_precision"] = recall_score(y, yh), precision_score(y, yh)
    novel = [r for r in rows if r["label_other"]]
    pn = model.predict([r["text"] for r in novel])
    m["novel_recall"] = float(np.mean([p["novel"] or p["other"] for p in pn]))
    m["known_flag_rate_max"] = float(np.mean([p["novel"] or p["other"] for p in preds]))

    all_preds = model.predict([r["text"] for r in rows])
    spikes = detect_spikes(weekly_counts(rows, all_preds))
    m["spikes"] = spikes
    m["spike_found"] = float(any(s["week"] == "2026-W37" and s["issue"] == "refund_delay" for s in spikes))
    m["spike_false_alarms_max"] = sum(1 for s in spikes if s["kind"] == "issue_spike" and not (s["week"] == "2026-W37" and s["issue"] == "refund_delay"))
    m["unclassified_surges"] = [s["week"] for s in spikes if s["kind"] == "unclassified_surge"]
    topics = emerging_topics(rows, all_preds)
    m["emerging_topics"] = [{k: t[k] for k in ("terms", "size", "recent_share", "earlier_share", "emerging")} for t in topics]
    m["emerging_topic_found"] = float(any(t["emerging"] and {"mandate", "nach", "autopay", "auto debit"} & set(" ".join(t["terms"]).split() + t["terms"])
                                          for t in topics))
    vocab = set(model.featurizer.word.vocabulary_)
    by = lambda lo, hi: [(r, p) for r, p in zip(rows, all_preds, strict=True) if lo <= r["received"] < hi]  # noqa: E731
    jul, aug, sep = by("2026-07-01", "2026-08-01"), by("2026-08-01", "2026-09-01"), by("2026-09-01", "2026-10-01")
    unz = lambda a: ([r for r, _ in a], [p for _, p in a])  # noqa: E731
    d_aug, d_sep = drift(*unz(jul), *unz(aug), vocab), drift(*unz(jul), *unz(sep), vocab)
    rng = random.Random(seed)   # synthetic channel shift: noisy call-centre transcripts
    shifted = [{**r, "text": _typos(rng, r["text"], 0.35).replace(".", " uh") } for r, _ in aug]
    d_shift = drift(*unz(jul), shifted, model.predict([r["text"] for r in shifted]), vocab)
    m["drift"] = {"jul_vs_aug": d_aug, "jul_vs_sep": d_sep, "jul_vs_noisy_transcripts": d_shift}
    m["drift_no_false_alarm"] = float(not d_aug["drift_detected"])
    m["drift_detected_shift"] = float(d_sep["drift_detected"] and d_shift["drift_detected"])

    pool, budget = production_pool(rows), 100
    unseen_test = [r for r in test if r["unseen_phrasing"]]
    al = retrain_with_feedback(rows, select_for_review(model, pool, budget), "al")
    rnd = retrain_with_feedback(rows, random.Random(seed).sample(pool, budget), "rand")
    f_base, f_al, f_rnd = (_f1(unseen_test, x.predict([r["text"] for r in unseen_test])) for x in (model, al, rnd))
    m["active_learning"] = {"budget": budget, "unseen_f1_before": f_base, "unseen_f1_active": f_al, "unseen_f1_random": f_rnd}
    m["active_learning_gain"] = f_al - f_base
    m["active_beats_random"] = float(f_al > f_rnd)
    m["model_metadata"] = model.metadata
    m["test_rows"] = len(test)
    return m


def gate(m: dict) -> list[str]:
    out = []
    for k, thr in THRESHOLDS.items():
        bad = m[k] > thr if k in MAX_KEYS else m[k] < thr
        if bad:
            out.append(f"{k}={m[k]:.3f} (gate {'<=' if k in MAX_KEYS else '>='} {thr})")
    return out


def main() -> None:
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default="reports/eval.json")
    a = ap.parse_args()
    m = evaluate()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=2, default=str))
    summary = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k not in ("model_metadata", "emerging_topics")}
    print(json.dumps(summary, indent=1, default=str)[:4000])
    br = gate(m)
    if a.gate:
        print("QUALITY GATE " + ("FAILED: " + "; ".join(br) if br else "PASSED"))
        sys.exit(1 if br else 0)


if __name__ == "__main__":
    main()
