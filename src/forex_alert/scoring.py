def trend_direction(series, fast=20, slow=50):
    if len(series) < slow:
        return 0.0
    ema_fast = series.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = series.ewm(span=slow, adjust=False).mean().iloc[-1]
    return 1.0 if ema_fast > ema_slow else -1.0

def score_signal(correlation: float, h4_alignment: float, h1_alignment: float, m15_alignment: float, weights=None) -> float:
    weights = weights or {"correlation": 40, "h4": 25, "h1": 20, "m15": 15}
    corr = min(abs(correlation) / 0.65, 1.0) * weights["correlation"]
    conf = (
        max(0.0, min(h4_alignment, 1.0)) * weights["h4"]
        + max(0.0, min(h1_alignment, 1.0)) * weights["h1"]
        + max(0.0, min(m15_alignment, 1.0)) * weights["m15"]
    )
    return round(corr + conf, 2)

def relationship_alignment(correlation: float, direction_a: float, direction_b: float) -> float:
    if direction_a == 0 or direction_b == 0:
        return 0.0
    expected = 1.0 if correlation >= 0 else -1.0
    observed = direction_a * direction_b
    return 1.0 if observed == expected else 0.0
