from cnlp.data.generator import luhn_digit, luhn_ok, verhoeff_digit, verhoeff_ok
from cnlp.pii import redact


def test_checksums():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert luhn_ok("411111111111111" + luhn_digit("411111111111111"))
    assert verhoeff_ok("23456789012" + verhoeff_digit("23456789012"))


def test_redacts_core_types():
    red, spans = redact("Priya Sharma here. Card 4111 1111 1111 1111, PAN ABCPK1234Z, mail priya.s@example.com, phone 9876543210.")
    types = {s.type for s in spans}
    assert {"CARD", "PAN", "EMAIL", "PHONE"} <= types
    assert "4111" not in red and "ABCPK1234Z" not in red and "9876543210" not in red


def test_typo_in_account_context_still_redacted():
    red, spans = redact("my acount numbr is 92172490651467 please check")
    assert "92172490651467" not in red and spans[0].type == "ACCOUNT"


def test_payment_references_are_kept_for_investigators():
    text = "UPI ref no 073699836613 and UTR 412345678901 failed. Card ending XXXX XXXX XXXX 3001."
    red, _ = redact(text)
    assert "073699836613" in red and "412345678901" in red and "3001" in red


def test_aadhaar_needs_valid_checksum_or_label():
    good = "23456789012" + verhoeff_digit("23456789012")
    assert good not in redact(f"aadhaar {good[:4]} {good[4:8]} {good[8:]}")[0].replace(" ", "")


def test_generated_test_set_full_recall(rows):
    for r in [r for r in rows if r["split"] == "test"][:400]:
        red, _ = redact(r["text"])
        assert all(p["value"] not in red for p in r["pii"]), r["complaint_id"]
        assert all(d in red for d in r["decoys"]), r["complaint_id"]
