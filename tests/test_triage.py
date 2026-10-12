from cnlp.triage import assess, entities

BASE = {"issues": ["refund_delay"], "other": False, "novel": False, "confidence": 0.9}


def test_fraud_goes_to_fraud_queue_with_short_sla():
    t = assess("Unknown txn of Rs 18,000 on my card", {**BASE, "issues": ["unauthorized_txn"]}, "2026-10-01")
    assert t["queue"] == "Fraud Response" and t["urgency"] == "high" and t["due"] == "2026-10-08"


def test_ombudsman_threat_is_regulatory_escalation():
    t = assess("Refund pending, I will go to the Banking Ombudsman", BASE, "2026-10-01")
    assert t["queue"] == "Executive Escalations" and t["regulatory_risk"] and "escalation:ombudsman" in t["reasons"]


def test_vulnerable_customer_raises_urgency():
    t = assess("I am a senior citizen, my pension refund is pending", BASE, "2026-10-01")
    assert t["urgency"] == "high" and "vulnerable_customer" in t["reasons"]


def test_novel_complaint_goes_to_review():
    t = assess("something new", {**BASE, "issues": [], "other": True, "confidence": 0.2}, "2026-10-01")
    assert t["queue"] == "Triage Review" and t["needs_review"]


def test_entities():
    e = entities("Rs 4,999 refund from StreamBox, UTR 412345678901 on 02/09/2026")
    assert e["max_amount_inr"] == 4999 and e["payment_refs"] == ["412345678901"] and e["merchants"] == ["StreamBox"]
