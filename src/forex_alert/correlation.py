import pandas as pd

def correlation_matrix(prices: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns.tail(window).corr(method="pearson")

def strong_pairs(matrix: pd.DataFrame, threshold: float = 0.65):
    rows = []
    for i, a in enumerate(matrix.columns):
        for b in matrix.columns[i + 1:]:
            r = float(matrix.loc[a, b])
            if abs(r) >= threshold:
                rows.append({"pair_a": a, "pair_b": b, "correlation": r})
    return sorted(rows, key=lambda x: abs(x["correlation"]), reverse=True)

def pair_correlation(prices: pd.DataFrame, a: str, b: str, window: int) -> float:
    x = prices[[a,b]].pct_change().dropna().tail(window)
    return float(x[a].corr(x[b]))
