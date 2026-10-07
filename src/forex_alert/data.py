from __future__ import annotations

import time
import requests
import pandas as pd

YAHOO_SYMBOLS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "USDCHF": "CHF=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "CHFJPY": "CHFJPY=X",
}


def fetch_15m(pair: str, period_days: int = 30, timeout: int = 20) -> pd.DataFrame:
    symbol = YAHOO_SYMBOLS[pair]
    period2 = int(time.time())
    # Yahoo limits 15m intraday history to roughly 60 days; keep a safety margin.
    effective_days = min(period_days, 59)
    period1 = period2 - effective_days * 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "15m",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    payload = r.json()["chart"]["result"]
    if not payload:
        raise ValueError(f"No Yahoo data for {pair}")
    result = payload[0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    close = result["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({pair: close}, index=idx).dropna()
    return df[~df.index.duplicated(keep="last")]


def build_timeframes(m15: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Right-labelled bars prevent a backtest snapshot from using prices
    # that occur after the stated signal timestamp.
    return {
        "M15": m15.resample("15min", label="right", closed="right").last().dropna(),
        "H1": m15.resample("1h", label="right", closed="right").last().dropna(),
        "H4": m15.resample("4h", label="right", closed="right").last().dropna(),
    }
