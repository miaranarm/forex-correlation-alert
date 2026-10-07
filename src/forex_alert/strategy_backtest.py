from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability
from .scoring import trend_direction, score_signal, relationship_alignment, timeframe_confluence, confluence_meets_minimum


def _signal_at(frames, ts, a, b, cfg, weights):
    cc, sc = cfg["correlation"], cfg["scoring"]
    snap = {tf: frames[tf].loc[:ts] for tf in ("H4", "H1", "M15")}
    windows = cc.get("timeframe_windows", {"H4": cc["window"], "H1": cc["window"], "M15": cc["window"]})
    min_obs = cc.get("min_observations", 60)
    corrs = {tf: pair_correlation(snap[tf], a, b, windows[tf], min_obs) for tf in ("H4", "H1", "M15")}
    if any(pd.isna(corrs[tf]) for tf in ("H4", "H1", "M15")) or abs(corrs["H4"]) < cc["threshold"]:
        return None
    regime = rolling_correlation(snap["H4"], a, b, cc["window"], cc["short_window"], min_obs)
    stability = correlation_stability(snap["H4"], a, b, cc["window"], cc["stability_segments"], min_obs)
    if not regime or stability < cc["minimum_stability"] or abs(regime["spread"]) > cc["maximum_regime_drift"]:
        return None
    dirs = {tf: (trend_direction(snap[tf][a]), trend_direction(snap[tf][b])) for tf in ("H4", "H1", "M15")}
    aligns = {tf: relationship_alignment(corrs[tf], *dirs[tf]) for tf in ("H4", "H1", "M15")}
    confluence = timeframe_confluence((aligns["H4"], aligns["H1"], aligns["M15"]))
    if not confluence_meets_minimum(confluence, sc.get("minimum_confluence", "H4_ONLY")):
        return None
    score = score_signal(corrs["H4"], aligns["H4"], aligns["H1"], aligns["M15"], weights, cc["threshold"], discriminating=True)
    if score < sc["minimum_alert_score"]:
        return None
    return {
        "direction_a": "LONG" if dirs["H4"][0] > 0 else "SHORT",
        "score": score,
        "correlation_h4": corrs["H4"],
        "confluence": confluence,
    }


def _simulate(prices, direction, cfg):
    s = cfg["strategy"]
    entry = float(prices.iloc[0])
    sl_pct, tp1_pct, tp2_pct, trail_pct = map(float, (s["stop_loss_pct"], s["tp1_pct"], s["tp2_pct"], s["trailing_stop_pct"]))
    sign = 1 if direction == "LONG" else -1
    sl, tp1, tp2 = entry * (1 - sign * sl_pct), entry * (1 + sign * tp1_pct), entry * (1 + sign * tp2_pct)
    trail, best, remaining, realized_r, tp1_hit = entry * (1 - sign * trail_pct), entry, 1.0, 0.0, False
    for px in prices.iloc[1:]:
        px = float(px)
        if direction == "LONG":
            best, trail = max(best, px), max(trail, best * (1 - trail_pct))
            if px <= sl:
                return "SL", realized_r + remaining * ((px - entry) / (entry - sl)), tp1_hit
            if not tp1_hit and px >= tp1:
                realized_r += 0.5 * ((tp1 - entry) / (entry - sl)); remaining = 0.5; tp1_hit = True
            if px >= tp2:
                return "TP2", realized_r + remaining * ((tp2 - entry) / (entry - sl)), tp1_hit
            if tp1_hit and px <= trail:
                return "TRAIL", realized_r + remaining * ((px - entry) / (entry - sl)), tp1_hit
        else:
            best, trail = min(best, px), min(trail, best * (1 + trail_pct))
            if px >= sl:
                return "SL", realized_r + remaining * ((entry - px) / (sl - entry)), tp1_hit
            if not tp1_hit and px <= tp1:
                realized_r += 0.5 * ((tp1 - entry) / (sl - entry)); remaining = 0.5; tp1_hit = True
            if px <= tp2:
                return "TP2", realized_r + remaining * ((tp2 - entry) / (sl - entry)), tp1_hit
            if tp1_hit and px >= trail:
                return "TRAIL", realized_r + remaining * ((px - entry) / (sl - entry)), tp1_hit
    last = float(prices.iloc[-1])
    r = realized_r + remaining * ((last - entry) / (entry - sl) if direction == "LONG" else (entry - last) / (sl - entry))
    return "TIMEOUT", r, tp1_hit


def _metrics(df):
    if df.empty:
        return {"trades": 0}
    wins, losses = df["r"] > 0, df["r"] < 0
    gross_loss = -df.loc[losses, "r"].sum()
    equity, dd = df["r"].cumsum(), df["r"].cumsum() - df["r"].cumsum().cummax()
    return {
        "trades": int(len(df)),
        "win_rate": round(float(wins.mean()), 4),
        "profit_factor": round(float(df.loc[wins, "r"].sum() / gross_loss), 4) if gross_loss else None,
        "expectancy_r": round(float(df["r"].mean()), 4),
        "total_r": round(float(df["r"].sum()), 4),
        "max_drawdown_r": round(float(dd.min()), 4),
        "tp1_rate": round(float(df["tp1"].mean()), 4),
    }


def run():
    cfg = load_config()
    series = []
    for pair in cfg["pairs"]:
        try:
            series.append(fetch_15m(pair, cfg["data"]["period_days"], cfg["data"]["request_timeout_seconds"]))
        except Exception as exc:
            print(f"WARNING: {pair}: {exc}")
    available = [p for p in cfg["pairs"] if any(p in f.columns and not f[p].dropna().empty for f in series)]
    if len(available) < cfg["data"].get("minimum_available_pairs", 8):
        raise RuntimeError(f"Insufficient market coverage: {len(available)}/{len(cfg['pairs'])} pairs available")
    prices, frames = pd.concat(series, axis=1).sort_index(), build_timeframes(pd.concat(series, axis=1).sort_index())
    h4, m15, cc = frames["H4"], frames["M15"], cfg["correlation"]
    weights = {k: cfg["scoring"][f"{k}_weight"] for k in ("correlation", "h4", "h1", "m15")}
    timestamps = list(h4.index[cc["window"]:-1])
    split = int(len(timestamps) * 0.70)
    rows = []
    for idx, ts in enumerate(timestamps):
        for a, b in combinations(available, 2):
            sig = _signal_at(frames, ts, a, b, cfg, weights)
            if not sig:
                continue
            entries = m15.loc[m15.index > ts, a].dropna()
            if entries.empty:
                continue
            entry_ts = entries.index[0]
            exit_data = m15.loc[m15.index >= entry_ts, a].dropna()
            exit_type, r, tp1 = _simulate(exit_data, sig["direction_a"], cfg)
            rows.append({
                "signal_ts": ts.isoformat(), "entry_ts": entry_ts.isoformat(), "pair": a,
                "direction": sig["direction_a"], "entry": float(entries.iloc[0]),
                "score": sig["score"], "correlation_h4": round(float(sig["correlation_h4"]), 4),
                "confluence": sig["confluence"], "exit": exit_type, "r": r, "tp1": tp1,
                "sample": "IS" if idx < split else "OOS",
            })
    df = pd.DataFrame(rows)
    out = Path("results"); out.mkdir(exist_ok=True)
    df.to_csv(out / "strategy_backtest_trades.csv", index=False)
    summary = {
        "data_period_days": cfg["data"]["period_days"], "strategy": cfg["strategy"],
        "all": _metrics(df),
        "in_sample": _metrics(df[df["sample"] == "IS"]) if not df.empty else {"trades": 0},
        "out_of_sample": _metrics(df[df["sample"] == "OOS"]) if not df.empty else {"trades": 0},
    }
    (out / "strategy_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Forex Signal Strategy Backtest", "", "Close-based execution on completed M15 bars; no intrabar high/low assumption.", "Chronological 70/30 IS/OOS split; OOS is never used for parameter selection.", ""]
    for name, metrics in (("All", summary["all"]), ("In-sample", summary["in_sample"]), ("Out-of-sample", summary["out_of_sample"])):
        lines += [f"## {name}", ""] + [f"- {k}: {v}" for k, v in metrics.items()] + [""]
    (out / "strategy_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
