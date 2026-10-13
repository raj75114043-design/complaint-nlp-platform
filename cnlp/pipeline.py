"""Complaint processing: redact → classify → triage → (optional) reply draft. One call per complaint, batch-friendly."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path

from cnlp.config import get_settings
from cnlp.llm import LLMClient, LLMUsage, build_llm
from cnlp.model import ComplaintModel, load_champion, train
from cnlp.pii import redact
from cnlp.triage import assess

log = logging.getLogger("cnlp.pipeline")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "complaints.jsonl"
OWN_NAMES = {"Kestrel Bank", "Kestrel Bank Customer", "Kestrel Bank Customer Care"}
PROMISE = re.compile(r"\b(will|shall) (?:be )?(?:refund|credit|reverse|waive)\w*|\bguarantee|\bwithin \d+ (?:hours|days)\b", re.I)


def load_rows() -> list[dict]:
    return [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]


@lru_cache(maxsize=2)
def _model(sig: str) -> ComplaintModel:
    m = load_champion()
    if m is None:   # first start (e.g. hosted demo): train from the bundled data, ~10 s
        m = train(load_rows(), "bootstrap")
    return m


def get_model() -> ComplaintModel:
    reg = ROOT / "models" / "registry.json"
    return _model(reg.read_text() if reg.exists() else "")


def get_llm() -> LLMClient:
    return build_llm(get_settings())


TEMPLATES = {
    "Fraud Response": "We have registered your report of an unauthorised transaction as a priority case. Our fraud team will review it and contact you.",
    "Payments Disputes": "We have raised a dispute for the payment you mentioned and will update you once the investigation with the payment network is complete.",
    "Executive Escalations": "Your complaint has been escalated to our executive grievance team, who will contact you directly.",
    "Triage Review": "Thank you for writing to us. A specialist is reviewing your complaint and will get back to you.",
}


def draft_reply(redacted: str, triage: dict, llm: LLMClient, usage: LLMUsage) -> tuple[str, str]:
    template = (TEMPLATES.get(triage["queue"]) or "We have registered your complaint with our {q} team and will update you.".format(q=triage["queue"])) + \
        f" Reference: your complaint is due for resolution by {triage['due']}."
    if not llm.available:
        return template, "template"
    text = llm.chat("Write a short, empathetic acknowledgement (max 70 words) from Kestrel Bank to a customer. Do not promise refunds, "
                    "amounts or timelines other than the due date given. Do not include any personal data. Sign off exactly as "
                    "'Kestrel Bank Customer Care'.",
                    json.dumps({"complaint": redacted, "queue": triage["queue"], "due": triage["due"]}), usage=usage, max_tokens=250)
    if not isinstance(text, str) or not text.strip():
        return template, "template_fallback (LLM unavailable)"
    leaked = [s for s in redact(text)[1] if s.value not in OWN_NAMES]   # the bank's own name in a sign-off is not personal data
    if PROMISE.search(text):
        return template, "template_fallback (LLM draft made a promise)"
    if re.search(r"\d{6,}|@", text) or leaked:
        return template, "template_fallback (LLM draft contained personal data)"
    return text.strip(), "llm"


def process(text: str, received: str, channel: str = "web_form", model: ComplaintModel | None = None, llm: LLMClient | None = None) -> dict:
    t0 = time.perf_counter()
    model = model or get_model()
    redacted, spans = redact(text)
    pred = model.predict([text])[0]
    tri = assess(text, pred, received)
    usage = LLMUsage()
    reply, reply_mode = draft_reply(redacted, tri, llm or LLMClient(), usage)
    out = {"id": "CASE-" + uuid.uuid4().hex[:8].upper(), "received": received, "channel": channel, "redacted_text": redacted,
           "pii_found": sorted({s.type for s in spans}), "prediction": pred, "triage": tri, "reply_draft": reply, "reply_mode": reply_mode,
           "model_version": model.version, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    log.info("complaint_processed", extra={"queue": tri["queue"], "urgency": tri["urgency"], "model": model.version})
    return out


__all__ = ["process", "get_model", "get_llm", "load_rows", "os"]
