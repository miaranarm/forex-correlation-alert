from .config import load_config
from .data import fetch_daily
from .correlation import correlation_matrix, strong_pairs
from .scoring import score_signal
from .alerts import emit

def run():
    cfg = load_config()
    prices = []
    for pair in cfg["pairs"]:
        try:
            prices.append(fetch_daily(pair, cfg["data"]["request_timeout_seconds"]))
        except Exception as exc:
            print(f"WARNING: {pair}: {exc}")
    if len(prices) < 2:
        raise RuntimeError("Not enough market data to calculate correlations.")
    merged = __import__("pandas").concat(prices, axis=1).dropna()
    matrix = correlation_matrix(merged, cfg["correlation"]["window"])
    candidates = strong_pairs(matrix, cfg["correlation"]["threshold"])
    alerts = []
    for c in candidates:
        c["score"] = score_signal(c["correlation"])
        if c["score"] >= cfg["scoring"]["minimum_alert_score"]:
            alerts.append(c)
    emit(alerts)

if __name__ == "__main__":
    run()
