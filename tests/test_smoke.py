from forex_alert.scoring import score_signal, relationship_alignment, timeframe_confluence
from forex_alert.correlation import correlation_stability

def test_full_confluence():
    assert score_signal(0.65,1,1,1)==100
    assert timeframe_confluence((1,1,1))=="FULL"

def test_positive_relationship():
    assert relationship_alignment(0.80,1,1)==1

def test_negative_relationship():
    assert relationship_alignment(-0.80,1,-1)==1

def test_weak_confluence():
    assert timeframe_confluence((0,1,1))=="WEAK"

def test_stability_empty():
    import pandas as pd
    x=pd.DataFrame({"a":[1,2],"b":[1,2]})
    assert correlation_stability(x,"a","b",window=10,segments=4)==0.0


def test_pair_correlation_rejects_insufficient_observations():
    import pandas as pd
    from forex_alert.correlation import pair_correlation
    x = pd.DataFrame({"a": range(10), "b": range(10)})
    assert pd.isna(pair_correlation(x, "a", "b", window=96, min_observations=60))


def test_pair_correlation_rejects_constant_series():
    import pandas as pd
    from forex_alert.correlation import pair_correlation
    x = pd.DataFrame({"a": [1.0] * 100, "b": list(range(100))})
    assert pd.isna(pair_correlation(x, "a", "b", window=96, min_observations=60))
