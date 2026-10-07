import math
import pandas as pd
from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability
from .scoring import trend_direction, score_signal, relationship_alignment, timeframe_confluence, confluence_meets_minimum
from .state import load_state, classify, save_state
from .report import write_reports
from .alerts import emit
from pathlib import Path
import json


def _trade_levels(price, direction, strategy):
    sign = 1 if direction == "LONG" else -1
    sl = price * (1 - sign * strategy["stop_loss_pct"])
    tp1 = price * (1 + sign * strategy["tp1_pct"])
    tp2 = price * (1 + sign * strategy["tp2_pct"])
    return {
        "entry": round(float(price), 6),
        "sl": round(float(sl), 6),
        "tp1": round(float(tp1), 6),
        "tp2": round(float(tp2), 6),
        "trailing_stop": f"{strategy['trailing_stop_pct']:.2%}",
    }


def _strategy_gate_enabled():
    path = Path("results/strategy_gate.json")
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload.get("enabled", False))
    except (OSError, ValueError, TypeError):
        return False


def run():
    cfg = load_config()
    series = []
    for pair in cfg["pairs"]:
        try:
            series.append(fetch_15m(pair, cfg["data"]["period_days"], cfg["data"]["request_timeout_seconds"]))
        except Exception as exc:
            print(f"WARNING: {pair}: {exc}")
    if len(series) < 2:
        raise RuntimeError("Not enough market data.")

    available_pairs = [
        pair for pair in cfg["pairs"]
        if any(pair in frame.columns and not frame[pair].dropna().empty for frame in series)
    ]
    minimum_available_pairs = cfg["data"].get("minimum_available_pairs", 8)
    if len(available_pairs) < minimum_available_pairs:
        raise RuntimeError(
            f"Insufficient market coverage: {len(available_pairs)}/{len(cfg['pairs'])} pairs available; "
            f"minimum is {minimum_available_pairs}."
        )

    frames = build_timeframes(pd.concat(series, axis=1).sort_index())
    corr_cfg = cfg["correlation"]
    window = corr_cfg["window"]
    timeframe_windows = corr_cfg.get("timeframe_windows", {"H4": window, "H1": window, "M15": window})
    threshold = corr_cfg["threshold"]
    min_obs = corr_cfg.get("min_observations", 60)
    weights = {
        "correlation": cfg["scoring"]["correlation_weight"],
        "h4": cfg["scoring"]["h4_weight"],
        "h1": cfg["scoring"]["h1_weight"],
        "m15": cfg["scoring"]["m15_weight"],
    }

    alerts = []
    strategy = cfg["strategy"]
    for i, a in enumerate(available_pairs):
        for b in available_pairs[i + 1:]:
            corrs = {
                tf: pair_correlation(frames[tf], a, b, timeframe_windows[tf], min_obs)
                for tf in ("H4", "H1", "M15")
            }
            if any(pd.isna(corrs[tf]) for tf in corrs) or abs(corrs["H4"]) < threshold:
                continue
            regime = rolling_correlation(frames["H4"], a, b, window, corr_cfg["short_window"], min_obs)
            if not regime:
                continue
            stability = correlation_stability(
                frames["H4"], a, b, window, corr_cfg["stability_segments"], min_obs
            )
            if stability < corr_cfg["minimum_stability"] or abs(regime["spread"]) > corr_cfg["maximum_regime_drift"]:
                continue
            dirs = {
                tf: (trend_direction(frames[tf][a]), trend_direction(frames[tf][b]))
                for tf in ("H4", "H1", "M15")
            }
            aligns = {tf: relationship_alignment(corrs[tf], *dirs[tf]) for tf in dirs}
            confluence = timeframe_confluence((aligns["H4"], aligns["H1"], aligns["M15"]))
            if not confluence_meets_minimum(confluence, cfg["scoring"]["minimum_confluence"]):
                continue
            score = score_signal(
                corrs["H4"], aligns["H4"], aligns["H1"], aligns["M15"],
                weights, threshold, discriminating=True
            )
            if score < cfg["scoring"]["minimum_alert_score"]:
                continue

            direction_a = "LONG" if dirs["H4"][0] > 0 else "SHORT"
            direction_b = direction_a if corrs["H4"] >= 0 else ("SHORT" if direction_a == "LONG" else "LONG")
            price_a = frames["M15"][a].dropna().iloc[-1]
            price_b = frames["M15"][b].dropna().iloc[-1]
            signals = []
            for pair, direction, price in ((a, direction_a, price_a), (b, direction_b, price_b)):
                signals.append({
                    "pair": pair,
                    "signal": direction,
                    **_trade_levels(price, direction, strategy),
                    "timeframe": strategy["timeframe"],
                })

            alerts.append({
                "pair_a": a, "pair_b": b,
                "signals": signals,
                "correlation_h4": round(corrs["H4"], 4),
                "correlation_h1": round(corrs["H1"], 4),
                "correlation_m15": round(corrs["M15"], 4),
                "h4_alignment": aligns["H4"], "h1_alignment": aligns["H1"], "m15_alignment": aligns["M15"],
                "confluence": confluence, "stability": stability,
                "h4_short_correlation": round(regime["short"], 4),
                "h4_long_correlation": round(regime["long"], 4),
                "regime_drift": round(regime["spread"], 4), "score": score,
            })

    alerts, cleared = classify(alerts, load_state())
    alerts.sort(key=lambda x: (x["score"], x["stability"]), reverse=True)
    write_reports(alerts, cleared)
    save_state(alerts)
    alerts_enabled = cfg.get("alerts", {}).get("enabled", True) and _strategy_gate_enabled()
    if alerts_enabled:
        emit(alerts)
    else:
        print("Trade alerts disabled until strategy validation passes.")
    for key in cleared:
        print(f"- CLEARED {key}")


if __name__ == "__main__":
    run()
