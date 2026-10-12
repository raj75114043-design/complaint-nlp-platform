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
    assert mode == "template_fallback" and "2026-10-11" in text


def test_llm_reply_with_pii_is_rejected():
    assert draft_reply("x", TRI, FakeLLM("Dear customer, we called 9876543210."), LLMUsage())[1] == "template_fallback"


def test_safe_llm_reply_is_used():
    assert draft_reply("x", TRI, FakeLLM("We are sorry for the trouble. Our team is reviewing your dispute."), LLMUsage())[1] == "llm"
