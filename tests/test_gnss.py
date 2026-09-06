from src.ins.gnss import GnssQualityGate


def test_quality_gate_accepts_good_fresh_fix():
    decision = GnssQualityGate().evaluate(satellites=12, accuracy_m=4.0, age_s=1.0)
    assert decision.accepted
    assert decision.reason == "accepted"


def test_quality_gate_reports_each_primary_rejection():
    gate = GnssQualityGate(minimum_satellites=8, maximum_accuracy_m=10, maximum_fix_age_s=12)
    assert gate.evaluate(satellites=20, accuracy_m=3, age_s=0, externally_available=False).reason == "unavailable"
    assert gate.evaluate(satellites=20, accuracy_m=3, age_s=13).reason == "stale"
    assert gate.evaluate(satellites=7, accuracy_m=3, age_s=0).reason == "satellites"
    assert gate.evaluate(satellites=20, accuracy_m=11, age_s=0).reason == "accuracy"
