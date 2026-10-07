def score_signal(correlation: float, h4: float = 0.0, h1: float = 0.0, m15: float = 0.0, weights=None) -> float:
    weights = weights or {"correlation": 40, "h4": 25, "h1": 20, "m15": 15}
    corr_component = min(abs(correlation) / 0.65, 1.0) * weights["correlation"]
    directional = (
        max(0.0, min(abs(h4), 1.0)) * weights["h4"]
        + max(0.0, min(abs(h1), 1.0)) * weights["h1"]
        + max(0.0, min(abs(m15), 1.0)) * weights["m15"]
    )
    return round(corr_component + directional, 2)
