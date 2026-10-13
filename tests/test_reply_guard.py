from cnlp.llm import LLMUsage
from cnlp.pipeline import draft_reply

TRI = {"queue": "Payments Disputes", "due": "2026-10-11"}


class FakeLLM:
    available = True
    provider = "fake"

    def __init__(self, text):
        self.text = text

    def chat(self, *a, **k):
        return self.text


def test_llm_reply_with_promise_is_rejected():
    text, mode = draft_reply("x", TRI, FakeLLM("We will refund Rs 500 within 2 days."), LLMUsage())
    assert mode.startswith("template_fallback") and "2026-10-11" in text


def test_llm_reply_with_pii_is_rejected():
    assert draft_reply("x", TRI, FakeLLM("Dear customer, we called 9876543210."), LLMUsage())[1].startswith("template_fallback")


def test_safe_llm_reply_is_used():
    assert draft_reply("x", TRI, FakeLLM("We are sorry for the trouble. Our team is reviewing your dispute."), LLMUsage())[1] == "llm"


def test_bank_sign_off_is_not_personal_data():
    text, mode = draft_reply("x", TRI, FakeLLM("We are sorry for the trouble and are reviewing your dispute. Regards, Kestrel Bank"), LLMUsage())
    assert mode == "llm"


def test_llm_failure_is_labelled():
    assert draft_reply("x", TRI, FakeLLM(None), LLMUsage())[1] == "template_fallback (LLM unavailable)"


def test_reference_from_redacted_complaint_may_be_quoted():
    red = "Rs 18,000 debited, UTR 412345678901, card [CARD]"
    assert draft_reply(red, TRI, FakeLLM("We are reviewing transaction 412345678901 for you."), LLMUsage())[1] == "llm"
    assert draft_reply(red, TRI, FakeLLM("We are reviewing account 998877665544 for you."), LLMUsage())[1].startswith("template_fallback")
