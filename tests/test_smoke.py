from forex_alert.scoring import score_signal, relationship_alignment, timeframe_confluence

def test_full_confluence():
    assert score_signal(0.65,1,1,1)==100
    assert timeframe_confluence((1,1,1))=="FULL"

def test_positive_relationship():
    assert relationship_alignment(0.80,1,1)==1

def test_negative_relationship():
    assert relationship_alignment(-0.80,1,-1)==1

def test_weak_confluence():
    assert timeframe_confluence((0,1,1))=="WEAK"
