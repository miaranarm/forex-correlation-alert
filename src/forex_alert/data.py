from __future__ import annotations
import io
import requests
import pandas as pd

STOOQ_SYMBOLS = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "USDCHF": "usdchf",
    "AUDUSD": "audusd",
    "NZDUSD": "nzdusd",
    "USDCAD": "usdcad",
    "EURGBP": "eurgbp",
    "EURJPY": "eurjpy",
    "GBPJPY": "gbpjpy",
    "AUDJPY": "audjpy",
    "CHFJPY": "chfjpy",
}

def fetch_daily(pair: str, timeout: int = 20) -> pd.DataFrame:
    symbol = STOOQ_SYMBOLS[pair]
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1=20200101&d2=20991231&i=d"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Close" not in df:
        raise ValueError(f"No usable data for {pair}")
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df.set_index("Date")[["Close"]].rename(columns={"Close": pair}).dropna()
