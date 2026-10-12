"""Synthetic customer complaints for "Kestrel Bank" (fictional, India).

Realism built in:
* English, Hinglish (code-mixed) and terse styles; typos; multiple issues in one complaint
* PII: names, mobile numbers, e-mails, Luhn-valid card numbers, account numbers, PAN, Verhoeff-valid Aadhaar numbers —
  and look-alikes that must NOT be redacted (12-digit UPI UTR references, masked cards, amounts, dates)
* Urgency cues: Banking Ombudsman / RBI escalation, legal notice, social-media threats, vulnerable customers
* A dated stream (Jul–Sep 2026) with an injected spike (UPI refund delays after a switch upgrade) and a new,
  out-of-taxonomy issue that starts mid-September (e-mandate auto-debit failures)
* Template families: each issue has phrasings reserved for the test set, so evaluation measures generalisation
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ["credit_card", "debit_card", "upi", "savings_account", "personal_loan", "net_banking"]
ISSUES = ["unauthorized_txn", "refund_delay", "charges_fees", "kyc_update", "emi_issue", "login_access", "card_blocked", "failed_txn_debited"]
COMPAT = {
    "credit_card": ["unauthorized_txn", "refund_delay", "charges_fees", "card_blocked"],
    "debit_card": ["unauthorized_txn", "card_blocked", "failed_txn_debited", "charges_fees"],
    "upi": ["failed_txn_debited", "refund_delay", "unauthorized_txn"],
    "savings_account": ["charges_fees", "kyc_update", "unauthorized_txn"],
    "personal_loan": ["emi_issue", "charges_fees"],
    "net_banking": ["login_access", "failed_txn_debited", "kyc_update"],
}
PRODUCT_WORDS = {"credit_card": ["credit card", "CC", "Kestrel Platinum card"], "debit_card": ["debit card", "ATM card"],
                 "upi": ["UPI", "UPI payment", "Kestrel Pay UPI"], "savings_account": ["savings account", "SB account", "account"],
                 "personal_loan": ["personal loan", "loan"], "net_banking": ["net banking", "internet banking", "netbanking"]}
MERCHANTS = ["Zomrato", "FlipKart Mart", "QuickCab", "MegaMart", "TrainEase", "StreamBox", "PayFuel", "UrbanBazaar"]

# (text, style); the last two per issue are held out for the test set ("unseen" phrasing)
FRAGMENTS = {
    "unauthorized_txn": [
        ("I see a transaction of {amt} on {dt} at {merchant} on my {pw} which I never did.", "en"),
        ("Someone has used my {pw} without my knowledge, {amt} debited on {dt}.", "en"),
        ("I did not authorise the {amt} payment to {merchant}. Please block and reverse it.", "en"),
        ("mere {pw} se {amt} ka transaction hua hai jo maine nahi kiya, please check karo", "hi"),
        ("fraud txn {amt} {merchant} not done by me", "terse"),
        ("Got an SMS for {amt} spent at {merchant} but I was asleep at home, this is fraud.", "en"),
        ("My {pw} was charged {amt} by an unknown merchant and I suspect it has been cloned.", "en"),
        ("bina OTP ke {amt} kat gaye mere account se, ye fraud hai", "hi"),
    ],
    "refund_delay": [
        ("I cancelled my order with {merchant} on {dt} but the refund of {amt} is still not credited.", "en"),
        ("Refund of {amt} from {merchant} pending for more than 10 days.", "en"),
        ("{merchant} says they processed the refund of {amt} but it has not reached my {pw}.", "en"),
        ("refund abhi tak nahi aaya, {amt} {merchant} ka, 2 hafte ho gaye", "hi"),
        ("refund {amt} not received {merchant}", "terse"),
        ("The reversal for {amt} promised on {dt} has not shown up in my statement.", "en"),
        ("Still waiting for my money back from {merchant}, {amt}, nobody is helping.", "en"),
        ("paisa wapas kab aayega? {amt} ka refund pending hai", "hi"),
    ],
    "charges_fees": [
        ("I have been charged an annual fee of {amt} on my {pw} which was supposed to be lifetime free.", "en"),
        ("Why was a penalty of {amt} levied for minimum balance on {dt}?", "en"),
        ("Unexplained charges of {amt} appear in my {pw} statement.", "en"),
        ("bina bataye {amt} charges kaat liye gaye hain", "hi"),
        ("wrong fee {amt} pls reverse", "terse"),
        ("A late payment fee of {amt} was added even though I paid before the due date.", "en"),
        ("GST and processing charges of {amt} were deducted that nobody told me about.", "en"),
        ("har mahine {amt} ka charge lag raha hai, kis cheez ka hai?", "hi"),
    ],
    "kyc_update": [
        ("I submitted my KYC documents on {dt} but my {pw} is still frozen.", "en"),
        ("KYC update is pending and I cannot do any transactions.", "en"),
        ("Re-KYC request rejected twice without any reason.", "en"),
        ("KYC ke liye documents de diye phir bhi account freeze hai", "hi"),
        ("kyc pending acct frozen", "terse"),
        ("The branch says my address proof is not valid even though it is my passport.", "en"),
        ("Video KYC keeps failing at the last step and the account remains restricted.", "en"),
        ("teen baar KYC kar chuka hoon, abhi bhi update nahi hua", "hi"),
    ],
    "emi_issue": [
        ("My {pw} EMI of {amt} was debited twice this month.", "en"),
        ("EMI bounce charges were applied even though there was sufficient balance on {dt}.", "en"),
        ("I requested foreclosure of my {pw} but the EMI of {amt} is still being deducted.", "en"),
        ("loan ki EMI do baar kat gayi, {amt} wapas chahiye", "hi"),
        ("emi double debit {amt}", "terse"),
        ("The interest rate on my {pw} was increased and my instalment jumped to {amt} without notice.", "en"),
        ("My moratorium request was approved but the instalment of {amt} was still collected.", "en"),
        ("EMI ka amount {amt} galat calculate hua hai", "hi"),
    ],
    "login_access": [
        ("I am unable to login to {pw}, it says user locked.", "en"),
        ("OTP is not coming for {pw} login since {dt}.", "en"),
        ("The app crashes every time I try to sign in.", "en"),
        ("{pw} mein login nahi ho raha, OTP hi nahi aata", "hi"),
        ("cant login app error", "terse"),
        ("My password reset link expires before I can use it, and I am stuck outside my account.", "en"),
        ("After the latest update the app shows a blank screen at the login page.", "en"),
        ("password reset karne ke baad bhi access nahi mil raha", "hi"),
    ],
    "card_blocked": [
        ("My {pw} was blocked on {dt} without any intimation.", "en"),
        ("Card got blocked after wrong PIN and the unblock request is pending.", "en"),
        ("I am travelling and my {pw} has been blocked, I need it urgently.", "en"),
        ("mera {pw} block ho gaya hai bina kisi wajah ke", "hi"),
        ("card blocked pls unblock", "terse"),
        ("Merchants keep declining my {pw} and customer care says it is hotlisted.", "en"),
        ("The bank deactivated my card for international use and now it will not work anywhere.", "en"),
        ("card band kar diya, ab kaise payment karun?", "hi"),
    ],
    "failed_txn_debited": [
        ("Payment of {amt} to {merchant} failed on {dt} but money was debited from my account.", "en"),
        ("ATM did not dispense cash but {amt} was debited.", "en"),
        ("{pw} transfer of {amt} shows failed but amount deducted.", "en"),
        ("payment fail ho gaya par {amt} kat gaye", "hi"),
        ("txn failed amt debited {amt}", "terse"),
        ("The transaction timed out at {merchant} yet my balance went down by {amt}.", "en"),
        ("My bill payment of {amt} was declined but the money left my account anyway.", "en"),
        ("paise kat gaye lekin {merchant} ko mile hi nahi", "hi"),
    ],
}
NOVEL = [  # appears from 2026-09-14; not in the taxonomy (true label: "other")
    "My e-mandate auto-debit for {merchant} failed even though the mandate is active.",
    "Auto debit mandate registration keeps failing on {pw}.",
    "NACH mandate for my SIP was rejected and now I am charged a bounce fee.",
    "autopay mandate fail ho raha hai har baar",
]
ESCALATION = [("If this is not resolved I will complain to the Banking Ombudsman.", "ombudsman"),
              ("I will escalate this to RBI.", "rbi"), ("I am sending a legal notice to the bank.", "legal"),
              ("I will post this on Twitter and tag the bank.", "social"),
              ("warna main RBI mein complaint karunga", "rbi")]
VULNERABLE = ["I am a senior citizen and this is my pension account.", "My father is in hospital and I need this money for treatment.",
              "main 72 saal ka hoon, please jaldi help karo"]
FIRST = ["Ramesh", "Priya", "Anil", "Sunita", "Farhan", "Kavita", "Joseph", "Meenakshi", "Gurpreet", "Ananya", "Vikas", "Rekha"]
LAST = ["Kumar", "Sharma", "Iyer", "Khan", "Das", "Patel", "Reddy", "Singh", "Nair", "Ghosh"]

# ---------------------------------------------------------------------------------------- checksums for realistic PII
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
      [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
      [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
      [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_digit(num: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def luhn_ok(num: str) -> bool:
    s = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch) * (2 if i % 2 else 1)
        s += d - 9 if d > 9 else d
    return s % 10 == 0


def luhn_digit(num: str) -> str:
    for d in "0123456789":
        if luhn_ok(num + d):
            return d
    raise ValueError


def _typos(rng, text: str, rate: float) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 4 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:] if rng.random() < 0.5 else w[:i] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def _amount(rng) -> str:
    v = rng.choice([499, 1200, 2350, 4999, 5000, 7800, 12500, 18000, 25000, 49999])
    return rng.choice([f"Rs {v}", f"Rs. {v:,}", f"INR {v}", f"₹{v:,}", f"{v} rupees"])


def _pii(rng, kind: str) -> str:
    if kind == "phone":
        n = str(rng.choice([6, 7, 8, 9])) + "".join(rng.choice("0123456789") for _ in range(9))
        return rng.choice([n, f"+91 {n}", f"+91-{n[:5]}-{n[5:]}"])
    if kind == "email":
        return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'yahoo.in', 'outlook.com'])}"
    if kind == "card":
        base = rng.choice(["4", "5"]) + "".join(rng.choice("0123456789") for _ in range(14))
        n = base + luhn_digit(base)
        return rng.choice([n, " ".join(n[i:i + 4] for i in range(0, 16, 4)), "-".join(n[i:i + 4] for i in range(0, 16, 4))])
    if kind == "account":
        return "".join(rng.choice("0123456789") for _ in range(rng.choice([11, 12, 14])))
    if kind == "pan":
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        return "".join(rng.choice(letters) for _ in range(3)) + "P" + rng.choice(letters) + "".join(rng.choice("0123456789") for _ in range(4)) + rng.choice(letters)
    if kind == "aadhaar":
        base = str(rng.randint(2, 9)) + "".join(rng.choice("0123456789") for _ in range(10))
        n = base + verhoeff_digit(base)
        return rng.choice([n, f"{n[:4]} {n[4:8]} {n[8:]}"])
    raise ValueError(kind)


PII_FRAMES = {"name": ["My name is {v}.", "This is {v} writing.", "Regards, {v}"],
              "phone": ["Call me on {v}.", "My mobile is {v}.", "mera number {v} hai"],
              "email": ["Reply to {v}.", "Email: {v}"],
              "card": ["My card number is {v}.", "card no {v}"],
              "account": ["My account number is {v}.", "a/c no {v}"],
              "pan": ["My PAN is {v}."],
              "aadhaar": ["My Aadhaar number is {v}.", "aadhaar {v}"]}


def generate(n: int = 6000, seed: int = 21) -> list[dict]:
    rng = random.Random(seed)
    start, rows = date(2026, 7, 1), []
    for i in range(n):
        day = start + timedelta(days=rng.randrange(92))
        split = rng.choices(["train", "val", "test"], [0.65, 0.13, 0.22])[0]
        frag_pool = 8 if split == "test" or day >= date(2026, 9, 1) else 6   # production language evolves after the training cut-off
        novel = day >= date(2026, 9, 14) and rng.random() < 0.06 + 0.02 * (day - date(2026, 9, 14)).days / 7
        spike = date(2026, 9, 7) <= day <= date(2026, 9, 13) and rng.random() < 0.35
        if novel:
            product, issues = rng.choice(["savings_account", "upi", "net_banking"]), []
            parts, unseen = [rng.choice(NOVEL)], True
        else:
            product = "upi" if spike else rng.choice(PRODUCTS)
            primary = "refund_delay" if spike else rng.choice(COMPAT[product])
            issues = [primary]
            if rng.random() < 0.22:
                others = [x for x in COMPAT[product] if x != primary]
                if others:
                    issues.append(rng.choice(others))
            parts, unseen = [], False
            for iss in issues:
                k = rng.randrange(frag_pool)
                parts.append(FRAGMENTS[iss][k][0])
                unseen |= k >= 6
        styles = set()
        ctx = {"amt": _amount(rng), "dt": (day - timedelta(days=rng.randint(1, 20))).strftime(rng.choice(["%d/%m/%Y", "%d %b", "%d-%m-%Y"])),
               "merchant": rng.choice(MERCHANTS), "pw": rng.choice(PRODUCT_WORDS[product])}
        text_parts = [p.format(**ctx) for p in parts]
        for p in parts:
            styles.add(next((s for iss in ISSUES for f, s in FRAGMENTS[iss] if f == p), "en"))
        decoys = []
        if rng.random() < 0.3:   # 12-digit UPI UTR: looks like Aadhaar, must be kept for the investigation
            utr = "".join(rng.choice("0123456789") for _ in range(12))
            text_parts.insert(1, rng.choice([f"UTR {utr}.", f"UPI ref no {utr}.", f"Transaction ID {utr}."]))
            decoys.append(utr)
        if rng.random() < 0.15:
            masked = "XXXX XXXX XXXX " + "".join(rng.choice("0123456789") for _ in range(4))
            text_parts.insert(1, f"Card ending {masked[-4:]} ({masked}).")
            decoys.append(masked)
        esc, vuln = None, False
        if rng.random() < 0.14:
            line, esc = rng.choice(ESCALATION)
            text_parts.append(line)
        if rng.random() < 0.05:
            text_parts.append(rng.choice(VULNERABLE))
            vuln = True
        text = " ".join(text_parts)
        if rng.random() < 0.25:
            text = _typos(rng, text, 0.12)        # typos hit the narrative and context words, not the identifiers below
        pii = []
        for kind in rng.sample(list(PII_FRAMES), k=rng.choice([0, 1, 1, 2, 3])):
            v = f"{rng.choice(FIRST)} {rng.choice(LAST)}" if kind == "name" else _pii(rng, kind)
            frame = rng.choice(PII_FRAMES[kind])
            text += " " + frame.format(v=v)
            pii.append({"type": kind, "value": v})
        urgency = "high" if (esc or vuln or "unauthorized_txn" in issues) else "normal"
        rows.append({"complaint_id": f"CMP-{i + 1:05d}", "received": day.isoformat(),
                     "channel": rng.choice(["email", "app_chat", "call_transcript", "web_form", "social"]),
                     "text": text, "product": product, "issues": issues, "label_other": not issues, "urgency": urgency,
                     "escalation": esc, "vulnerable": vuln, "pii": pii, "decoys": decoys, "split": split, "unseen_phrasing": unseen,
                     "style": "hi" if "hi" in styles else ("terse" if styles == {"terse"} else "en"),
                     "scenario": "novel" if novel else ("spike" if spike else "normal")})
    return rows


def write(path: Path = ROOT / "data" / "complaints.jsonl") -> None:
    rows = generate()
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} complaints to {path}")


if __name__ == "__main__":
    write()
