from forex_alert.scoring import score_signal, relationship_alignment, timeframe_confluence
from forex_alert.correlation import correlation_stability

def test_full_confluence():
    assert score_signal(0.65,1,1,1)==100
    assert timeframe_confluence((1,1,1))=="FULL"

def test_positive_relationship():
    assert relationship_alignment(0.80,1,1)==1

def test_negative_relationship():
    assert relationship_alignment(-0.80,1,-1)==1

def test_weak_confluence():
    assert timeframe_confluence((0,1,1))=="WEAK"

def test_stability_empty():
    import pandas as pd
    x=pd.DataFrame({"a":[1,2],"b":[1,2]})
    assert correlation_stability(x,"a","b",window=10,segments=4)==0.0


def test_stability_penalizes_sign_instability():
    import pandas as pd
    from forex_alert.correlation import correlation_stability

    returns_a = [1, 2, 3, 4] * 24
    returns_b = returns_a[:48] + [-x for x in returns_a[48:]]
    prices_a = pd.concat([pd.Series([100.0]), pd.Series(returns_a).cumsum() + 100.0], ignore_index=True)
    prices_b = pd.concat([pd.Series([100.0]), pd.Series(returns_b).cumsum() + 100.0], ignore_index=True)
    x = pd.DataFrame({"a": prices_a, "b": prices_b})

    assert correlation_stability(x, "a", "b", window=96, segments=4, min_observations=60) < 0.70


def test_pair_correlation_rejects_insufficient_observations():
    import pandas as pd
    from forex_alert.correlation import pair_correlation
    x = pd.DataFrame({"a": range(10), "b": range(10)})
    assert pd.isna(pair_correlation(x, "a", "b", window=96, min_observations=60))


def test_pair_correlation_rejects_constant_series():
    import pandas as pd
    from forex_alert.correlation import pair_correlation
    x = pd.DataFrame({"a": [1.0] * 100, "b": list(range(100))})
    assert pd.isna(pair_correlation(x, "a", "b", window=96, min_observations=60))


def test_confluence_minimum_is_enforced():
    from forex_alert.scoring import confluence_meets_minimum
    assert confluence_meets_minimum("H4_ONLY", "H4_ONLY")
    assert not confluence_meets_minimum("H4_ONLY", "H4_H1")
    assert confluence_meets_minimum("FULL", "H4_H1")


def test_timeframe_windows_preserve_16_day_horizon():
    from forex_alert.config import load_config
    cfg = load_config()
    windows = cfg["correlation"]["timeframe_windows"]
    assert windows["H4"] * 4 == windows["H1"]
    assert windows["H4"] * 16 == windows["M15"]


def test_forward_outcomes_ignore_missing_prices():
    import pandas as pd
    from forex_alert.backtest import _add_forward_outcomes

    h4 = pd.DataFrame(
        {"a": [100.0, float("nan"), 102.0], "b": [100.0, 101.0, float("nan")]}
    )
    rows = [{"pair_a": "a", "pair_b": "b", "correlation_h4": 0.80}]
    _add_forward_outcomes(rows, h4, 0)

    assert rows[0]["relationship_correct_4h"] is None
    assert rows[0]["spread_4h"] is None


def test_fetch_15m_retries_transient_provider_failure(monkeypatch):
    import pandas as pd
    from forex_alert import data

    calls = {"n": 0}

    class Response:
        def raise_for_status(self):
            if calls["n"] == 1:
                raise RuntimeError("temporary failure")

        def json(self):
            return {
                "chart": {
                    "result": [{
                        "timestamp": [1_700_000_000],
                        "indicators": {"quote": [{"close": [1.0]}]},
                    }]
                }
            }

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return Response()

    monkeypatch.setattr(data.requests, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)

    out = data.fetch_15m("EURUSD", period_days=1, timeout=1, retries=2)
    assert calls["n"] == 2
    assert isinstance(out, pd.DataFrame)
    assert out.iloc[0, 0] == 1.0


def test_minimum_available_pairs_is_configured():
    from forex_alert.config import load_config
    cfg = load_config()
    assert cfg["data"]["minimum_available_pairs"] == 8
    assert cfg["data"]["minimum_available_pairs"] <= len(cfg["pairs"])


def test_build_timeframes_excludes_incomplete_bars():
    import pandas as pd
    from forex_alert.data import build_timeframes

    idx = pd.date_range("2026-10-07 00:00:00", periods=9, freq="15min", tz="UTC")
    m15 = pd.DataFrame({"EURUSD": range(100, 109)}, index=idx)
    frames = build_timeframes(m15, as_of=pd.Timestamp("2026-10-07 02:07:00", tz="UTC"))

    assert frames["M15"].index.max() == pd.Timestamp("2026-10-07 01:45:00", tz="UTC")
    assert frames["H1"].index.max() == pd.Timestamp("2026-10-07 01:00:00", tz="UTC")
    assert frames["H4"].index.max() == pd.Timestamp("2026-10-07 00:00:00", tz="UTC")


def test_build_timeframes_normalizes_naive_as_of_to_utc():
    import pandas as pd
    from forex_alert.data import build_timeframes

    idx = pd.date_range("2026-10-07 00:00:00", periods=5, freq="15min", tz="UTC")
    m15 = pd.DataFrame({"EURUSD": range(100, 105)}, index=idx)
    frames = build_timeframes(m15, as_of=pd.Timestamp("2026-10-07 01:07:00"))

    assert frames["M15"].index.max() == pd.Timestamp("2026-10-07 00:45:00", tz="UTC")


def test_main_ignores_empty_pair_frames_for_market_coverage(monkeypatch):
    import pandas as pd
    from forex_alert import main

    empty = pd.DataFrame({"EURUSD": []})
    valid = pd.DataFrame({"GBPUSD": [1.0, 1.1]})
    monkeypatch.setattr(main, "fetch_15m", lambda pair, *args: empty if pair == "EURUSD" else valid)
    monkeypatch.setattr(main, "load_config", lambda: {
        "pairs": ["EURUSD", "GBPUSD"],
        "data": {"period_days": 1, "request_timeout_seconds": 1, "minimum_available_pairs": 2},
    })

    try:
        main.run()
    except RuntimeError as exc:
        assert "Insufficient market coverage: 1/2 pairs available" in str(exc)
    else:
        raise AssertionError("empty provider data must not count as market coverage")


def test_backtest_ignores_empty_pair_frames_for_market_coverage(monkeypatch):
    import pandas as pd
    from forex_alert import backtest

    empty = pd.DataFrame({"EURUSD": []})
    valid = pd.DataFrame({"GBPUSD": [1.0, 1.1]})
    monkeypatch.setattr(backtest, "fetch_15m", lambda pair, *args: empty if pair == "EURUSD" else valid)
    monkeypatch.setattr(backtest, "load_config", lambda: {
        "pairs": ["EURUSD", "GBPUSD"],
        "data": {"period_days": 1, "request_timeout_seconds": 1, "minimum_available_pairs": 2},
    })

    try:
        backtest.run_backtest()
    except RuntimeError as exc:
        assert "Insufficient market coverage: 1/2 pairs available" in str(exc)
    else:
        raise AssertionError("empty provider data must not count as market coverage")
