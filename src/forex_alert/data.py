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


def fetch_15m(pair: str, period_days: int = 30, timeout: int = 20, retries: int = 3) -> pd.DataFrame:
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
    last_error = None
    for attempt in range(max(1, retries)):
        try:
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
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(2 ** attempt)
            else:
                raise last_error
    result = payload[0]
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    close = result["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({pair: close}, index=idx).dropna()
    return df[~df.index.duplicated(keep="last")]


def build_timeframes(m15: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    # Right-labelled bars make the timestamp represent the bar close.
    # Drop the still-forming bucket so live alerts never use an incomplete
    # H4/H1/M15 candle. Historical backtests remain unchanged because their
    # snapshots are taken only from completed timestamps.
    frames = {
        "M15": m15.resample("15min", label="right", closed="right").last().dropna(),
        "H1": m15.resample("1h", label="right", closed="right").last().dropna(),
        "H4": m15.resample("4h", label="right", closed="right").last().dropna(),
    }
    now = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    for tf, freq in (("M15", "15min"), ("H1", "1h"), ("H4", "4h")):
        cutoff = now.floor(freq)
        if now != cutoff:
            cutoff -= pd.Timedelta(freq)
        frames[tf] = frames[tf].loc[frames[tf].index <= cutoff]
    return frames
