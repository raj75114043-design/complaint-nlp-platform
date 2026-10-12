from cnlp.data.generator import ISSUES
from cnlp.model import CUTOFF, normalise


def test_normalise_replaces_volatile_tokens():
    t = normalise("Rs 4,999 debited on 02/09/2026, ref 556677889900, card XXXX XXXX XXXX 1234")
    assert "_amt_" in t and "_date_" in t and "_ref_" in t and "_maskedcard_" in t


def test_training_respects_time_cutoff(model):
    assert model.metadata["data_cutoff"] == CUTOFF
    assert set(model.thresholds) == set(ISSUES)


def test_prediction_contract(model):
    p = model.predict(["My personal loan EMI was debited twice this month and a late fee was added."])[0]
    assert {"product", "issues", "issue_probs", "other", "novelty", "novel", "confidence"} <= set(p)
    assert "emi_issue" in p["issues"]


def test_multi_label(model):
    p = model.predict(["Money debited but UPI payment failed, and the refund for my cancelled order has not come for 3 weeks."])[0]
    assert {"failed_txn_debited", "refund_delay"} & set(p["issues"])


def test_out_of_taxonomy_text_is_flagged(model):
    p = model.predict(["The e-mandate auto debit for my subscription failed though mandate is active, NACH bounce fee charged."])[0]
    assert p["novel"] or p["other"]
