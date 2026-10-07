def trend_direction(series, fast=20, slow=50):
    if len(series) < slow:
        return 0.0
    fast_v = series.ewm(span=fast, adjust=False).mean().iloc[-1]
    slow_v = series.ewm(span=slow, adjust=False).mean().iloc[-1]
    return 1.0 if fast_v > slow_v else -1.0


def correlation_quality(correlation, threshold=0.65, discriminating=False):
    magnitude = abs(correlation)
    if discriminating:
        if magnitude <= threshold:
            return 0.0
        if threshold >= 1.0:
            return 1.0 if magnitude >= 1.0 else 0.0
        return min((magnitude - threshold) / (1.0 - threshold), 1.0)
    return max(0.0, min(magnitude / threshold, 1.0))


def relationship_alignment(correlation, direction_a, direction_b):
    if direction_a == 0 or direction_b == 0:
        return 0.0
    expected = 1.0 if correlation >= 0 else -1.0
    return 1.0 if direction_a * direction_b == expected else 0.0


def timeframe_confluence(alignments):
    h4, h1, m15 = alignments
    if h4 == 1 and h1 == 1 and m15 == 1:
        return "FULL"
    if h4 == 1 and h1 == 1:
        return "H4_H1"
    if h4 == 1:
        return "H4_ONLY"
    return "WEAK"


CONFLUENCE_RANK = {
    "WEAK": 0,
    "H4_ONLY": 1,
    "H4_H1": 2,
    "FULL": 3,
}


def confluence_meets_minimum(actual, minimum):
    return CONFLUENCE_RANK.get(actual, -1) >= CONFLUENCE_RANK.get(minimum, 1)


def score_signal(correlation, h4_alignment, h1_alignment, m15_alignment, weights=None, threshold=0.65, discriminating=False):
    weights = weights or {"correlation": 40, "h4": 25, "h1": 20, "m15": 15}
    corr = correlation_quality(correlation, threshold, discriminating) * weights["correlation"]
    score = corr + h4_alignment * weights["h4"] + h1_alignment * weights["h1"] + m15_alignment * weights["m15"]
    return round(score, 2)
