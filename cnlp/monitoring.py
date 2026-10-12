"""Production monitoring: issue spikes, emerging (out-of-taxonomy) topics and data drift.

* Spike detection — weekly counts per predicted issue; a week is anomalous when it exceeds the trailing median by more
  than 3 robust standard deviations (1.4826 × MAD) and by at least 10 complaints.
* Emerging topics — complaints flagged novel/other are clustered (TF-IDF + NMF); a topic is reported when its weekly
  share keeps rising, with its top terms and example complaints.
* Drift — Population Stability Index of the predicted-issue mix and of text-length buckets, plus the out-of-vocabulary
  rate against the training vocabulary. PSI > 0.2 is treated as significant drift.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from cnlp.model import normalise

# English + common Hinglish function words, and pipeline placeholder tokens
TOPIC_STOP = list(TfidfVectorizer(stop_words="english").get_stop_words()) + """hai hain ka ki ke ko se ye yeh mera mere meri raha rahi rahe
ho gaya gaye gayi kar karo karna nahi bhi abhi har aur par lekin ab kab kis kaise main hoon tha thi amt date ref maskedcard card ending
number please pls""".split()


def week_of(d: str) -> str:
    y, w, _ = date.fromisoformat(d).isocalendar()
    return f"{y}-W{w:02d}"


def weekly_counts(rows: list[dict], preds: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r, p in zip(rows, preds, strict=True):
        for iss in p["issues"] or ["other"]:
            out[week_of(r["received"])][iss] += 1
    return dict(sorted(out.items()))


def detect_spikes(weekly: dict[str, Counter], min_history: int = 6) -> list[dict]:
    weeks = list(weekly)
    labels = {lab for c in weekly.values() for lab in c}
    alerts = []
    for i in range(min_history, len(weeks)):
        for lab in labels:
            hist = np.array([weekly[w][lab] for w in weeks[i - min_history:i]], dtype=float)
            med = float(np.median(hist))
            mad = 1.4826 * float(np.median(np.abs(hist - med))) or 1.0
            cur = weekly[weeks[i]][lab]
            z = (cur - med) / mad
            if z > 3 and cur - med >= 10 and cur >= 1.5 * max(med, 1):   # robust z, absolute and relative increase
                alerts.append({"week": weeks[i], "issue": lab, "count": cur, "baseline": med, "robust_z": round(z, 1),
                               "kind": "unclassified_surge" if lab == "other" else "issue_spike"})
    return alerts


def emerging_topics(rows: list[dict], preds: list[dict], n_topics: int = 6, top_terms: int = 6) -> list[dict]:
    # candidates for a *new issue*: nothing in the taxonomy fits (no label clears its threshold), or novel text with a weak best label
    flagged = [(r, p) for r, p in zip(rows, preds, strict=True)
               if p["other"] or (p.get("novel") and max(p["issue_probs"].values()) < 0.5)]
    if len(flagged) < 20:
        return []
    texts = [normalise(r["text"]) for r, _ in flagged]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words=TOPIC_STOP, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    x = vec.fit_transform(texts)
    nmf = NMF(n_components=min(n_topics, x.shape[0] - 1), random_state=0, init="nndsvda", max_iter=500)
    w = nmf.fit_transform(x)
    terms = np.array(vec.get_feature_names_out())
    weeks = sorted({week_of(r["received"]) for r in rows})
    total_by_week = Counter(week_of(r["received"]) for r in rows)
    topics = []
    for k in range(nmf.n_components):
        members = [i for i in range(len(flagged)) if w[i].argmax() == k and w[i, k] > 0]
        by_week = Counter(week_of(flagged[i][0]["received"]) for i in members)
        share = [by_week[wk] / total_by_week[wk] for wk in weeks]
        recent, before = np.mean(share[-3:]), np.mean(share[:-3]) if len(share) > 3 else 0.0
        topics.append({"topic": k, "terms": terms[nmf.components_[k].argsort()[::-1][:top_terms]].tolist(), "size": len(members),
                       "recent_share": round(float(recent), 4), "earlier_share": round(float(before), 4),
                       "emerging": bool(recent > 3 * max(before, 0.002) and len(members) >= 15),
                       "examples": [flagged[i][0]["text"][:160] for i in members[:3]], "member_ids": [flagged[i][0]["complaint_id"] for i in members]})
    return sorted(topics, key=lambda t: (-t["emerging"], -t["recent_share"]))


def psi(expected: Counter, actual: Counter, eps: float = 1e-4) -> float:
    keys = set(expected) | set(actual)
    e_tot, a_tot = sum(expected.values()) or 1, sum(actual.values()) or 1
    total = 0.0
    for k in keys:
        e, a = max(expected[k] / e_tot, eps), max(actual[k] / a_tot, eps)
        total += (a - e) * math.log(a / e)
    return total


def _len_bucket(text: str) -> str:
    n = len(text.split())
    return "<10" if n < 10 else "10-25" if n < 25 else "25-45" if n < 45 else "45+"


def drift(ref_rows, ref_preds, cur_rows, cur_preds, vocab: set[str]) -> dict:
    ref_mix = Counter(i for p in ref_preds for i in (p["issues"] or ["other"]))
    cur_mix = Counter(i for p in cur_preds for i in (p["issues"] or ["other"]))
    ref_len, cur_len = Counter(_len_bucket(r["text"]) for r in ref_rows), Counter(_len_bucket(r["text"]) for r in cur_rows)

    def oov(rows):
        toks = [t for r in rows for t in normalise(r["text"]).split() if t.isalpha()]
        return sum(t not in vocab for t in toks) / max(1, len(toks))

    out = {"psi_issue_mix": round(psi(ref_mix, cur_mix), 3), "psi_length": round(psi(ref_len, cur_len), 3),
           "oov_rate_ref": round(oov(ref_rows), 3), "oov_rate_cur": round(oov(cur_rows), 3),
           "novel_rate_ref": round(float(np.mean([p.get("novel", False) for p in ref_preds])), 3),
           "novel_rate_cur": round(float(np.mean([p.get("novel", False) for p in cur_preds])), 3)}
    out["signals"] = [k for k, hit in [("issue_mix", out["psi_issue_mix"] > 0.2), ("length", out["psi_length"] > 0.2),
                                       ("vocabulary", out["oov_rate_cur"] > 1.5 * max(out["oov_rate_ref"], 0.01)),
                                       ("novelty", out["novel_rate_cur"] > 1.75 * max(out["novel_rate_ref"], 0.01))] if hit]
    out["drift_detected"] = bool(out["signals"])
    return out
