"""Urgency, regulatory flags, entities, routing and SLA for a classified complaint.

The SLA rules are illustrative, modelled on common Indian banking practice (resolve within 30 days before the customer
may approach the Banking Ombudsman; prioritise unauthorised electronic transactions, where timely reporting limits
customer liability). They are configuration, not legal advice.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

ESCALATION = {
    "ombudsman": r"ombuds\w*", "rbi": r"\brbi\b|reserve bank", "legal": r"legal notice|consumer (?:court|forum)|lawyer|advocate",
    "social": r"twitter|\bx\.com\b|facebook|instagram|social media|post (?:this|it) on|go viral",
}
VULNERABLE = r"senior citizen|pension|hospital|treatment|medical|\b(?:6|7|8|9)\d saal\b|\b(?:6|7|8|9)\d (?:years|yrs) old|disab\w+|widow"
MERCHANTS = ["Zomrato", "FlipKart Mart", "QuickCab", "MegaMart", "TrainEase", "StreamBox", "PayFuel", "UrbanBazaar"]
QUEUES = {
    "unauthorized_txn": "Fraud Response", "refund_delay": "Payments Disputes", "failed_txn_debited": "Payments Disputes",
    "charges_fees": "Retail Servicing", "kyc_update": "KYC Operations", "emi_issue": "Loan Servicing",
    "login_access": "Digital Channels", "card_blocked": "Card Operations",
}
SLA_DAYS = {"Fraud Response": 7, "Executive Escalations": 7, "Payments Disputes": 10, "default": 30}


def entities(text: str) -> dict:
    amounts = []
    for m in re.finditer(r"(?:rs\.?|inr|₹)\s*([\d,]+)|([\d,]+)\s*rupees", text, re.I):
        v = (m.group(1) or m.group(2)).replace(",", "")
        if v.isdigit():
            amounts.append(int(v))
    refs = re.findall(r"(?:utr|ref(?:erence)?\s*no\.?|transaction\s*id|txn\s*id)\s*[:#-]?\s*(\d{10,16})", text, re.I)
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*", text)
    merchants = [m for m in MERCHANTS if m.lower() in text.lower()]
    return {"amounts_inr": amounts, "max_amount_inr": max(amounts) if amounts else None, "payment_refs": refs, "dates": dates, "merchants": merchants}


def assess(text: str, prediction: dict, received: str) -> dict:
    low = text.lower()
    escalation = [k for k, rx in ESCALATION.items() if re.search(rx, low)]
    vulnerable = bool(re.search(VULNERABLE, low))
    fraud = "unauthorized_txn" in prediction["issues"]
    reasons = [f"escalation:{e}" for e in escalation] + (["vulnerable_customer"] if vulnerable else []) + (["suspected_fraud"] if fraud else [])
    urgency = "high" if reasons else "normal"
    if prediction.get("novel") or prediction["other"]:
        queue = "Triage Review"
    elif escalation and {"ombudsman", "rbi", "legal"} & set(escalation):
        queue = "Executive Escalations"
    else:
        queue = QUEUES[prediction["issues"][0]] if prediction["issues"] else "Triage Review"
    if fraud:
        queue = "Fraud Response"
    sla = SLA_DAYS.get(queue, SLA_DAYS["default"])
    due = date.fromisoformat(received) + timedelta(days=sla)
    ent = entities(text)
    score = (40 if fraud else 0) + (30 if {"ombudsman", "rbi", "legal"} & set(escalation) else 10 if escalation else 0) + (20 if vulnerable else 0) + \
        min(10, (ent["max_amount_inr"] or 0) // 5000)
    return {"urgency": urgency, "priority_score": score, "reasons": reasons, "regulatory_risk": bool({"ombudsman", "rbi", "legal"} & set(escalation)),
            "queue": queue, "sla_days": sla, "due": due.isoformat(), "entities": ent,
            "needs_review": bool(prediction.get("novel") or prediction["other"] or prediction["confidence"] < 0.5)}
