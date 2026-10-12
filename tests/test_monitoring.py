from collections import Counter

from cnlp.monitoring import detect_spikes, psi, week_of


def test_week_of():
    assert week_of("2026-09-07") == "2026-W37"


def test_spike_detection_robust_and_relative():
    weekly = {f"2026-W{w:02d}": Counter({"refund_delay": 40 + (w % 3), "kyc_update": 5}) for w in range(27, 40)}
    weekly["2026-W37"]["refund_delay"] = 95
    weekly["2026-W38"]["kyc_update"] = 9   # small wobble: not an alert
    alerts = detect_spikes(weekly)
    assert [(a["week"], a["issue"]) for a in alerts] == [("2026-W37", "refund_delay")]


def test_psi():
    a = Counter({"x": 50, "y": 50})
    assert psi(a, a) < 1e-9 and psi(a, Counter({"x": 90, "y": 10})) > 0.2
