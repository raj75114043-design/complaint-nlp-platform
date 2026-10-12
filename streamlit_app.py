"""Complaint intelligence console (Streamlit)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import streamlit as st  # noqa: E402

try:
    for key in ("GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

import pandas as pd  # noqa: E402

from cnlp.monitoring import detect_spikes, drift, emerging_topics, week_of, weekly_counts  # noqa: E402
from cnlp.pii import redact  # noqa: E402
from cnlp.pipeline import get_llm, get_model, load_rows, process  # noqa: E402
from cnlp.triage import assess  # noqa: E402

logging.disable(logging.INFO)
st.set_page_config(page_title="Complaint NLP Platform", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading model (first start trains it, ~10 s)…")
def model():
    return get_model()


@st.cache_data(show_spinner="Scoring the complaint stream…")
def stream():
    rows = load_rows()
    preds = model().predict([r["text"] for r in rows])
    tri = [assess(r["text"], p, r["received"]) for r, p in zip(rows, preds, strict=True)]
    return rows, preds, tri


m = model()
llm = get_llm()
EXAMPLES = {
    "Hinglish fraud + PII + Ombudsman": "mere credit card se Rs 18,000 ka transaction hua hai jo maine nahi kiya. UTR 412345678901. My card number is "
                                        "4111 1111 1111 1111. Call me on 9876543210. If this is not resolved I will complain to the Banking Ombudsman.",
    "Refund delay (typos)": "I cancelld my order with StreamBox on 02/09/2026 but the refnd of Rs 4,999 is stil not credited. Transaction ID 556677889900.",
    "Two issues in one": "My personal loan EMI of Rs 12500 was debited twice this month. A late payment fee of Rs 499 was also added.",
    "New issue (not in taxonomy)": "My e-mandate auto-debit for StreamBox failed even though the mandate is active. NACH bounce fee charged.",
    "Vulnerable customer": "I am a senior citizen and this is my pension account. KYC update is pending and I cannot do any transactions. aadhaar 2345 6789 0124",
}

with st.sidebar:
    st.markdown("### 🏦 Complaint NLP Platform")
    st.caption("Kestrel Bank (fictional) · PII redaction · multi-label triage · urgency · trends · drift · active learning")
    st.markdown(f"**Model:** `{m.version}`  \n**Training cut-off:** {m.metadata['data_cutoff']}  \n**Reply writer:** "
                f"`{llm.provider if llm.available else 'templates (no LLM)'}`")
    st.caption("All complaints, names and numbers are synthetic.")

tabs = st.tabs(["📨 Triage", "📊 Operations & trends", "🧑‍⚖️ Review & active learning", "📉 Drift", "🔐 PII", "🧪 Evaluation", "🏗️ Architecture"])

with tabs[0]:
    ex = st.selectbox("Example", list(EXAMPLES))
    text = st.text_area("Complaint text", EXAMPLES[ex], height=130)
    c1, c2 = st.columns(2)
    received = c1.date_input("Received").isoformat()
    channel = c2.selectbox("Channel", ["web_form", "email", "app_chat", "call_transcript", "social"])
    if st.button("Process", type="primary"):
        r = process(text, received, channel, model=m, llm=llm)
        t, p = r["triage"], r["prediction"]
        k = st.columns(5)
        k[0].metric("Queue", t["queue"])
        k[1].metric("Urgency", t["urgency"].upper())
        k[2].metric("Priority score", t["priority_score"])
        k[3].metric("Due", t["due"])
        k[4].metric("Needs review", "yes" if t["needs_review"] else "no")
        st.markdown("**Redacted text** (what agents and models see)")
        st.code(r["redacted_text"], language=None)
        st.markdown(f"**Issues:** {', '.join(p['issues']) or '— none in taxonomy —'} · **Product:** {p['product']} ({p['product_confidence']}) · "
                    f"**Novelty:** {p['novelty']} {'⚠️ new language/issue' if p['novel'] else ''}  \n**Reasons:** {', '.join(t['reasons']) or '—'}")
        st.json(t["entities"])
        st.bar_chart(pd.Series(p["issue_probs"]).sort_values(), horizontal=True)
        st.markdown("**Reply draft** (" + r["reply_mode"] + ")")
        st.info(r["reply_draft"])

with tabs[1]:
    rows, preds, tri = stream()
    df = pd.DataFrame([{"week": week_of(r["received"]), "queue": t["queue"], "urgency": t["urgency"], "issue": (p["issues"] or ["other"])[0]}
                       for r, p, t in zip(rows, preds, tri, strict=True)])
    a, b, c = st.columns(3)
    a.metric("Complaints (Jul–Sep)", len(df))
    b.metric("High urgency", f"{(df.urgency == 'high').mean():.0%}")
    c.metric("Sent to review", f"{sum(t['needs_review'] for t in tri) / len(tri):.0%}")
    st.markdown("**Weekly volume by predicted issue**")
    st.line_chart(df.pivot_table(index="week", columns="issue", aggfunc="size", fill_value=0))
    st.markdown("**Alerts**")
    st.dataframe(pd.DataFrame(detect_spikes(weekly_counts(rows, preds))), hide_index=True, width="stretch")
    st.markdown("**Emerging topics among complaints the taxonomy cannot place**")
    for tpc in emerging_topics(rows, preds)[:4]:
        (st.error if tpc["emerging"] else st.write)(f"{'🆕 EMERGING · ' if tpc['emerging'] else ''}{', '.join(tpc['terms'])} — {tpc['size']} complaints, "
                                                   f"share {tpc['earlier_share']:.1%} → {tpc['recent_share']:.1%}")
    st.markdown("**Queue load**")
    st.bar_chart(df.groupby(["queue", "urgency"]).size().unstack(fill_value=0))

with tabs[2]:
    st.markdown("Complaints received **after** the training cut-off are scored; the most informative go to reviewers. Their labels train a "
                "challenger, compared on the held-out test set with a challenger trained on the same number of *random* labels.")
    budget = st.slider("Review budget (complaints labelled by people)", 50, 300, 100, step=50)
    if st.button("Run review cycle", type="primary"):
        import random

        from cnlp.active import production_pool, retrain_with_feedback, select_for_review
        from cnlp.eval.run import _f1
        rows = load_rows()
        pool = production_pool(rows)
        with st.spinner("Selecting, retraining two challengers…"):
            sel = select_for_review(m, pool, budget)
            al, rnd = retrain_with_feedback(rows, sel, "active"), retrain_with_feedback(rows, random.Random(0).sample(pool, budget), "random")
        test = [r for r in rows if r["split"] == "test" and not r["label_other"] and r["unseen_phrasing"]]
        f = {n: _f1(test, x.predict([r["text"] for r in test])) for n, x in (("champion", m), ("active", al), ("random", rnd))}
        x1, x2, x3 = st.columns(3)
        x1.metric("Champion — unseen phrasing F1", f"{f['champion']:.2f}")
        x2.metric("Active-learning challenger", f"{f['active']:.2f}", f"{f['active'] - f['champion']:+.2f}")
        x3.metric("Random-label challenger", f"{f['random']:.2f}", f"{f['random'] - f['champion']:+.2f}")
        st.dataframe(pd.DataFrame([{"id": r["complaint_id"], "text": r["text"][:140], "human label": ", ".join(r["issues"]) or "OUT OF TAXONOMY"} for r in sel[:20]]),
                     hide_index=True, width="stretch")

with tabs[3]:
    rows, preds, _ = stream()
    vocab = set(m.featurizer.word.vocabulary_)
    sel = lambda lo, hi: [(r, p) for r, p in zip(rows, preds, strict=True) if lo <= r["received"] < hi]  # noqa: E731
    unz = lambda a: ([r for r, _ in a], [p for _, p in a])  # noqa: E731
    ref = sel("2026-07-01", "2026-08-01")
    res = {"Aug vs Jul": drift(*unz(ref), *unz(sel("2026-08-01", "2026-09-01")), vocab), "Sep vs Jul": drift(*unz(ref), *unz(sel("2026-09-01", "2026-10-01")), vocab)}
    st.dataframe(pd.DataFrame(res).T, width="stretch")
    st.caption("PSI > 0.2 on the issue mix or text length, a 1.5× rise in out-of-vocabulary rate, or a 1.75× rise in novelty rate raises a drift alert.")

with tabs[4]:
    sample = st.text_area("Text to redact", "Regards, Priya Sharma. My account numbr is 92172490651467, PAN ABCPK1234Z, aadhaar 2345 6789 0124. "
                                            "UPI ref no 073699836613. Card ending 3001 (XXXX XXXX XXXX 3001).", height=110)
    red, spans = redact(sample)
    st.code(red, language=None)
    st.dataframe(pd.DataFrame([s.__dict__ for s in spans]), hide_index=True, width="stretch")
    st.caption("Kept on purpose: UPI references (needed by investigators), masked cards. Validated: Luhn (cards), Verhoeff (Aadhaar).")

with tabs[5]:
    if st.button("Run full evaluation (~30 s)", type="primary"):
        from cnlp.eval.run import evaluate, gate
        ev = evaluate()
        cols = st.columns(4)
        for col, k in zip(cols * 3, ["issue_f1_micro", "issue_f1_seen", "issue_f1_unseen", "baseline_f1_micro", "pii_min_recall", "pii_decoy_preserved",
                                      "urgency_recall", "novel_recall", "active_learning_gain", "product_accuracy"], strict=False):
            col.metric(k.replace("_", " "), f"{ev[k]:.2f}")
        br = gate(ev)
        (st.error if br else st.success)("Quality gate " + ("FAILED: " + "; ".join(br) if br else "PASSED"))
        st.json({k: ev[k] for k in ("issue_f1_by_style", "pii_recall_by_type", "active_learning", "spikes")})

with tabs[6]:
    st.graphviz_chart("""
    digraph P { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f4", fontname=Helvetica];
      I [label="Complaint\\n(email · chat · call · social)"]; R [label="PII redaction\\nLuhn · Verhoeff · context"]; N [label="Normalise\\namounts · dates · refs"];
      C [label="Classifier\\nTF-IDF word+char · LR\\nmulti-label + thresholds"]; V [label="Novelty score"]; T [label="Triage\\nurgency · regulatory · SLA · queue"];
      D [label="Reply draft\\n(template / guarded LLM)"]; Q [label="Review queue"]; A [label="Active learning\\nchallenger vs champion"];
      M [label="Monitoring\\nspikes · emerging topics · drift"];
      I -> R -> N -> C -> T -> D; N -> V -> Q; C -> Q; Q -> A -> C; C -> M; V -> M; }""")
