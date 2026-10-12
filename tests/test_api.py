from fastapi.testclient import TestClient

from cnlp.api import app

client = TestClient(app)
H = {"X-API-Key": "test-key"}


def test_health_and_auth():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/redact", json={"text": "hi"}).status_code == 401


def test_process_complaint():
    body = {"text": "Rs 18,000 transaction I did not make on my credit card 4111 1111 1111 1111. I will complain to the Banking Ombudsman.",
            "received": "2026-10-01"}
    r = client.post("/v1/complaints", headers=H, json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["triage"]["queue"] == "Fraud Response" and "CARD" in j["pii_found"] and "4111" not in j["redacted_text"]
    assert j["reply_mode"] == "template" and r.headers["X-Request-ID"]


def test_validation_and_metrics():
    assert client.post("/v1/complaints", headers=H, json={"text": "hello there", "received": "01-10-2026"}).status_code == 422
    assert "cnlp_complaints_total" in client.get("/metrics").text
