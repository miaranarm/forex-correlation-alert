def trend_direction(series, fast=20, slow=50):
    if len(series)<slow:
        return 0.0
    fast_v=series.ewm(span=fast,adjust=False).mean().iloc[-1]
    slow_v=series.ewm(span=slow,adjust=False).mean().iloc[-1]
    return 1.0 if fast_v>slow_v else -1.0

def correlation_quality(correlation, threshold=0.65):
    return max(0.0,min(abs(correlation)/threshold,1.0))

def relationship_alignment(correlation, direction_a, direction_b):
    if direction_a==0 or direction_b==0:
        return 0.0
    expected=1.0 if correlation>=0 else -1.0
    return 1.0 if direction_a*direction_b==expected else 0.0

def timeframe_confluence(alignments):
    h4,h1,m15=alignments
    if h4==1 and h1==1 and m15==1:
        return "FULL"
    if h4==1 and h1==1:
        return "H4_H1"
    if h4==1:
        return "H4_ONLY"
    return "WEAK"

def score_signal(correlation, h4_alignment, h1_alignment, m15_alignment, weights=None, threshold=0.65):
    weights=weights or {"correlation":40,"h4":25,"h1":20,"m15":15}
    corr=correlation_quality(correlation,threshold)*weights["correlation"]
    score=corr+h4_alignment*weights["h4"]+h1_alignment*weights["h1"]+m15_alignment*weights["m15"]
    return round(score,2)
