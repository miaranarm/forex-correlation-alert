from forex_alert.scoring import score_signal, relationship_alignment

def test_correlation_threshold_score():
    assert score_signal(0.65, 1, 1, 1) == 100

def test_positive_relationship():
    assert relationship_alignment(0.80, 1, 1) == 1

def test_negative_relationship():
    assert relationship_alignment(-0.80, 1, -1) == 1
