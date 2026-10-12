"""PII detection and redaction with validation, so identifiers needed for the investigation survive.

| Type     | Detection                                                    | Validation / context                          |
|----------|--------------------------------------------------------------|-----------------------------------------------|
| EMAIL    | RFC-ish pattern                                              | —                                             |
| CARD     | 16 digits, optional spaces/dashes                            | Luhn checksum; masked cards are left alone    |
| AADHAAR  | 12 digits (4-4-4 or contiguous), first digit 2–9             | Verhoeff checksum, unless a UTR/ref context   |
| PAN      | AAAAA9999A with a valid holder-type letter                   | —                                             |
| PHONE    | Indian mobile, optional +91, 6–9 start                       | not part of a longer number                   |
| ACCOUNT  | 9–18 digits                                                  | only after "account number / a/c / acct"      |
| NAME     | "my name is …", "this is … writing", "regards, …"            | capitalised tokens                            |

UPI UTR references (12 digits, like Aadhaar) and masked cards are deliberately kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from cnlp.data.generator import luhn_ok, verhoeff_ok

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
CARD = re.compile(r"(?<![\dX])(?:\d{4}[ -]?){3}\d{4}(?!\d)")
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ ]?\d{4}[ ]?\d{4}(?!\d)")
PAN = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b")
PHONE = re.compile(r"(?<![\d+])(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")
ACCOUNT_CTX = re.compile(r"(?:account\s*(?:number|no\.?|num)?|a/c\s*(?:no\.?)?|acct\.?\s*(?:no\.?)?)\s*(?:is|:|-)?\s*(\d{9,18})(?!\d)", re.I)
NAME = re.compile(r"(?:[Mm]y name is|[Tt]his is|[Rr]egards,|[Nn]ame:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
LONG_NUMBER = re.compile(r"(?<!\d)\d{11,18}(?!\d)")
ACCOUNT_WORDS = ("account", "acct", "a/c", "accnt")
REF_WORDS = ("utr", "ref", "reference", "transaction", "txn", "rrn")


def _ref_context(before: str) -> bool:
    """Typo-tolerant check for a payment-reference label right before a number ("Transactin ID 7992…")."""
    words = re.findall(r"[a-z]+", before.lower())[-3:]
    return any(w == r or (len(w) > 4 and SequenceMatcher(None, w, r).ratio() >= 0.8) for w in words for r in REF_WORDS)


@dataclass
class Span:
    start: int
    end: int
    type: str
    value: str


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def detect(text: str) -> list[Span]:
    spans: list[Span] = []

    def add(m, kind, group=0):
        s, e = m.span(group)
        if not any(s < x.end and e > x.start for x in spans):
            spans.append(Span(s, e, kind, text[s:e]))

    for m in EMAIL.finditer(text):
        add(m, "EMAIL")
    for m in CARD.finditer(text):
        if luhn_ok(_digits(m.group(0))):
            add(m, "CARD")
    for m in ACCOUNT_CTX.finditer(text):
        add(m, "ACCOUNT", 1)
    for m in AADHAAR.finditer(text):
        before = text[max(0, m.start() - 25):m.start()]
        labelled = bool(re.search(r"aadha+r\w*\s*(?:no\.?|number|numbre)?\s*(?:is|:)?\s*$", before, re.I))
        if not labelled and _ref_context(before):
            continue                                   # a payment reference, needed by the investigation team
        if labelled or verhoeff_ok(_digits(m.group(0))):
            add(m, "AADHAAR")
    for m in LONG_NUMBER.finditer(text):
        # typo-tolerant account context ("accont number", "acocunt no"), else conservative redaction of unknown long numbers
        before = text[max(0, m.start() - 30):m.start()]
        if _ref_context(before):
            continue
        words = re.findall(r"[a-z/]+", before.lower())
        kind = "ACCOUNT" if any(SequenceMatcher(None, w, a).ratio() >= 0.7 for w in words[-4:] for a in ACCOUNT_WORDS) else "NUMBER"
        add(m, kind)
    for m in PAN.finditer(text):
        add(m, "PAN")
    for m in PHONE.finditer(text):
        add(m, "PHONE")
    for m in NAME.finditer(text):
        name = m.group(1)
        if name.split()[0].lower() not in {"the", "my", "a", "not", "very", "regarding", "about"}:
            add(m, "NAME", 1)
    return sorted(spans, key=lambda s: s.start)


def redact(text: str) -> tuple[str, list[Span]]:
    spans = detect(text)
    out, last = [], 0
    for s in spans:
        out += [text[last:s.start], f"[{s.type}]"]
        last = s.end
    out.append(text[last:])
    return "".join(out), spans
