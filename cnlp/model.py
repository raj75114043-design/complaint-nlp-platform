"""Product and issue classification with a versioned model registry.

* Text is PII-redacted and normalised (amounts, dates, references → placeholder tokens) before featurisation.
* Features: TF-IDF word 1–2-grams + character 2–5-grams (robust to typos and Hinglish spelling variation).
* Product: multinomial logistic regression. Issues: one-vs-rest logistic regression (multi-label) with a probability
  threshold per label tuned on the validation set; if no label clears its threshold the complaint is "other"
  (candidate new issue, routed to review).
* Training data is time-bounded (received before CUTOFF) so evaluation on later complaints is honest.
* A keyword baseline is trained alongside for comparison; a model is promoted only if it beats the champion.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from cnlp.data.generator import ISSUES, PRODUCTS
from cnlp.pii import redact

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CUTOFF = "2026-09-01"


def normalise(text: str) -> str:
    t, _ = redact(text)
    t = re.sub(r"(?:rs\.?|inr|₹)\s*[\d,]+|\b[\d,]+\s*rupees\b", " _amt_ ", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", " _date_ ", t, flags=re.I)
    t = re.sub(r"(?:x{4}[ -]?){3}\d{4}", " _maskedcard_ ", t, flags=re.I)
    t = re.sub(r"\b\d{6,}\b", " _ref_ ", t)
    t = re.sub(r"\[(\w+)\]", lambda m: f" _{m.group(1).lower()}_ ", t)
    return re.sub(r"\s+", " ", t.lower()).strip()


class Featurizer:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


@dataclass
class ComplaintModel:
    version: str
    featurizer: Featurizer
    product_clf: LogisticRegression
    issue_clf: OneVsRestClassifier
    mlb: MultiLabelBinarizer
    thresholds: dict[str, float]
    metadata: dict = field(default_factory=dict)
    train_word: object = None          # word TF-IDF of training texts, for novelty scoring
    novelty_threshold: float = 0.5

    def novelty(self, norm_texts: list[str]) -> np.ndarray:
        """1 − max cosine similarity to any training complaint, on word features (unseen vocabulary ⇒ high novelty)."""
        w = self.featurizer.word.transform(norm_texts)
        sims = (w @ self.train_word.T).max(axis=1).toarray().ravel() if self.train_word is not None else np.zeros(w.shape[0])
        return 1 - sims

    def predict(self, texts: list[str]) -> list[dict]:
        norm = [normalise(t) for t in texts]
        x = self.featurizer.transform(norm)
        pp = self.product_clf.predict_proba(x)
        ip = self.issue_clf.predict_proba(x)
        nov = self.novelty(norm)
        out = []
        for i in range(len(texts)):
            probs = {lab: float(ip[i, j]) for j, lab in enumerate(self.mlb.classes_)}
            issues = [lab for lab, p in probs.items() if p >= self.thresholds[lab]]
            best = max(probs.values())
            pj = int(np.argmax(pp[i]))
            out.append({"product": self.product_clf.classes_[pj], "product_confidence": round(float(pp[i, pj]), 3),
                        "issues": sorted(issues, key=lambda lab: -probs[lab]), "issue_probs": {k: round(v, 3) for k, v in probs.items()},
                        "other": not issues, "novelty": round(float(nov[i]), 3), "novel": bool(nov[i] >= self.novelty_threshold),
                        "confidence": round(min(float(pp[i, pj]), best), 3)})
        return out


def _split(rows):
    train = [r for r in rows if r["split"] == "train" and r["received"] < CUTOFF and not r["label_other"]]
    val = [r for r in rows if r["split"] == "val" and r["received"] < CUTOFF and not r["label_other"]]
    return train, val


def train(rows: list[dict], version: str | None = None, C: float = 8.0) -> ComplaintModel:
    tr, va = _split(rows)
    feat = Featurizer()
    xtr = feat.fit_transform([normalise(r["text"]) for r in tr])
    xva = feat.transform([normalise(r["text"]) for r in va])
    prod = LogisticRegression(C=C, max_iter=2000).fit(xtr, [r["product"] for r in tr])
    mlb = MultiLabelBinarizer(classes=ISSUES)
    ytr, yva = mlb.fit_transform([r["issues"] for r in tr]), mlb.transform([r["issues"] for r in va])
    iss = OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, class_weight="balanced")).fit(xtr, ytr)
    pva = iss.predict_proba(xva)
    thresholds = {}
    for j, lab in enumerate(mlb.classes_):   # per-label threshold maximising F1 on validation
        grid = np.linspace(0.2, 0.9, 29)
        f1s = [f1_score(yva[:, j], pva[:, j] >= t, zero_division=0) for t in grid]
        thresholds[lab] = float(grid[int(np.argmax(f1s))])
    data_hash = hashlib.sha256("".join(r["complaint_id"] + r["text"] for r in tr).encode()).hexdigest()[:12]
    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    meta = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "train_rows": len(tr), "val_rows": len(va),
            "data_cutoff": CUTOFF, "train_data_hash": data_hash, "algorithm": "tfidf(word1-2+char2-5) + logistic regression (OvR, balanced)",
            "C": C, "thresholds": thresholds, "labels": {"products": PRODUCTS, "issues": ISSUES}}
    train_word = feat.word.transform([normalise(r["text"]) for r in tr])
    # novelty threshold: 99th percentile of validation novelty, so ~1% of familiar complaints are flagged
    model = ComplaintModel(version, feat, prod, iss, mlb, thresholds, meta, train_word)
    model.novelty_threshold = float(np.quantile(model.novelty([normalise(r["text"]) for r in va]), 0.99))
    meta["novelty_threshold"] = round(model.novelty_threshold, 3)
    return model


# ---------------------------------------------------------------------------------------------- keyword baseline
KEYWORDS = {
    "unauthorized_txn": ["never did", "not done by me", "fraud", "without my knowledge", "did not authorise", "nahi kiya"],
    "refund_delay": ["refund", "reversal"], "charges_fees": ["fee", "charge", "penalty"], "kyc_update": ["kyc"],
    "emi_issue": ["emi", "instalment"], "login_access": ["login", "otp", "sign in"], "card_blocked": ["blocked", "block", "hotlisted"],
    "failed_txn_debited": ["failed", "fail", "not dispense", "debited but"],
}
PRODUCT_KEYWORDS = {"credit_card": ["credit card", "cc", "platinum"], "debit_card": ["debit card", "atm"], "upi": ["upi"],
                    "savings_account": ["savings", "sb account", "minimum balance"], "personal_loan": ["loan", "emi"],
                    "net_banking": ["net banking", "netbanking", "internet banking", "login", "app"]}


def baseline_predict(texts: list[str]) -> list[dict]:
    out = []
    for t in texts:
        low = t.lower()
        issues = [k for k, kws in KEYWORDS.items() if any(w in low for w in kws)]
        prod = next((p for p, kws in PRODUCT_KEYWORDS.items() if any(re.search(rf"\b{re.escape(w)}\b", low) for w in kws)), "savings_account")
        out.append({"product": prod, "issues": issues, "other": not issues})
    return out


# ---------------------------------------------------------------------------------------------- registry
def save(model: ComplaintModel, metrics: dict, promote: bool) -> Path:
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"{model.version}.joblib"
    joblib.dump(model, path)
    reg_path = MODELS / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"champion": None, "models": []}
    reg["models"].append({**model.metadata, "path": path.name, "metrics": metrics})
    if promote:
        reg["champion"] = model.version
    reg_path.write_text(json.dumps(reg, indent=2))
    return path


def load_champion() -> ComplaintModel | None:
    reg_path = MODELS / "registry.json"
    if not reg_path.exists():
        return None
    reg = json.loads(reg_path.read_text())
    if not reg.get("champion"):
        return None
    return joblib.load(MODELS / f"{reg['champion']}.joblib")
