from forex_alert.scoring import score_signal

def test_correlation_threshold_score():
    assert score_signal(0.65) >= 65
