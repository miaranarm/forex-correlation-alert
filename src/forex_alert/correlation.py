import pandas as pd


def returns(series):
    return series.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()


def pair_correlation(prices, a, b, window=96, min_observations=60):
    x = prices[[a, b]].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna().tail(window)
    if len(x) < min_observations or x[a].nunique() < 2 or x[b].nunique() < 2:
        return float("nan")
    value = x[a].corr(x[b], method="pearson")
    return float(value) if pd.notna(value) else float("nan")


def correlation_matrix(prices, window=96, min_observations=60):
    x = returns(prices).tail(window)
    if len(x) < min_observations:
        return pd.DataFrame(index=prices.columns, columns=prices.columns, dtype=float)
    return x.corr(method="pearson")


def rolling_correlation(prices, a, b, window=96, short_window=24, min_observations=60):
    x = prices[[a, b]].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(x) < max(window, min_observations) or short_window < 3:
        return None
    recent = x.tail(short_window)
    full = x.tail(window)
    if recent[a].nunique() < 2 or recent[b].nunique() < 2 or full[a].nunique() < 2 or full[b].nunique() < 2:
        return None
    short = recent[a].corr(recent[b])
    long = full[a].corr(full[b])
    if pd.isna(short) or pd.isna(long):
        return None
    return {"short": float(short), "long": float(long), "spread": float(abs(short) - abs(long))}


def correlation_stability(prices, a, b, window=96, segments=4, min_observations=60):
    x = prices[[a, b]].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna().tail(window)
    if segments < 2 or len(x) < max(segments * 10, min_observations):
        return 0.0
    parts = [x.iloc[i * len(x) // segments:(i + 1) * len(x) // segments] for i in range(segments)]
    vals = []
    for part in parts:
        if len(part) > 2 and part[a].nunique() >= 2 and part[b].nunique() >= 2:
            value = part[a].corr(part[b])
            if pd.notna(value):
                vals.append(float(value))
    if len(vals) != segments:
        return 0.0
    reference_sign = vals[-1] >= 0
    same_sign = sum(1 for value in vals if (value >= 0) == reference_sign)
    magnitude = sum(abs(value) for value in vals) / len(vals)
    return round(0.5 * (same_sign / len(vals)) + 0.5 * min(magnitude / 0.65, 1), 4)


def strong_pairs(matrix, threshold=0.65):
    rows = []
    for i, a in enumerate(matrix.columns):
        for b in matrix.columns[i + 1:]:
            r = matrix.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"pair_a": a, "pair_b": b, "correlation": float(r)})
    return sorted(rows, key=lambda x: abs(x["correlation"]), reverse=True)
