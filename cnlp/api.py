"""FastAPI service.

  POST /v1/complaints          process one complaint (redact, classify, triage, reply draft)
  POST /v1/complaints/batch    process up to 500 complaints
  POST /v1/redact              PII redaction only
  GET  /v1/model               champion model card (metadata, thresholds)
  GET  /health, /ready, /metrics
"""
from __future__ import annotations

import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from cnlp import logging_setup
from cnlp.config import get_settings
from cnlp.pii import redact
from cnlp.pipeline import get_llm, get_model, process

logging_setup.configure()
app = FastAPI(title="Complaint NLP Platform", version="1.0.0")
REQ = Counter("cnlp_http_requests_total", "HTTP requests", ["path", "status"])
ROUTED = Counter("cnlp_complaints_total", "Complaints processed", ["queue", "urgency"])
PII = Counter("cnlp_pii_redactions_total", "PII redactions", ["type"])
LAT = Histogram("cnlp_process_seconds", "Processing latency", buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2))


@app.middleware("http")
async def ctx(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    tok = logging_setup.request_id.set(rid)
    try:
        resp = await call_next(request)
    finally:
        logging_setup.request_id.reset(tok)
    resp.headers["X-Request-ID"] = rid
    REQ.labels(request.url.path, resp.status_code).inc()
    return resp


def auth(x_api_key: str | None = Header(default=None)):
    if x_api_key not in get_settings().api_keys:
        raise HTTPException(401, "Invalid or missing API key")


class Complaint(BaseModel):
    text: str = Field(min_length=5, max_length=5000)
    received: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-10-01"])
    channel: str = Field(default="web_form", max_length=30)


class Batch(BaseModel):
    complaints: list[Complaint] = Field(min_length=1, max_length=500)


class Text(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    m = get_model()
    return {"status": "ready", "model_version": m.version, "llm": get_llm().provider}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _one(c: Complaint) -> dict:
    t0 = time.perf_counter()
    r = process(c.text, c.received, c.channel, llm=get_llm())
    LAT.observe(time.perf_counter() - t0)
    ROUTED.labels(r["triage"]["queue"], r["triage"]["urgency"]).inc()
    for t in r["pii_found"]:
        PII.labels(t).inc()
    return r


@app.post("/v1/complaints", dependencies=[Depends(auth)])
def complaint(c: Complaint):
    return _one(c)


@app.post("/v1/complaints/batch", dependencies=[Depends(auth)])
def batch(b: Batch):
    return [_one(c) for c in b.complaints]


@app.post("/v1/redact", dependencies=[Depends(auth)])
def redact_only(t: Text):
    red, spans = redact(t.text)
    return {"redacted_text": red, "types": [s.type for s in spans]}


@app.get("/v1/model", dependencies=[Depends(auth)])
def model_card():
    return get_model().metadata
