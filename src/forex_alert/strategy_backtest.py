from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability, correlation_matrix
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
    # Relative-value signal: estimate a rolling OLS hedge ratio on log-price
    # levels, then trade the residual only when it is statistically stretched.
    # This avoids using a return-covariance beta as a price-level hedge ratio.
    trigger_tf = cfg["strategy"].get("timeframe", "M15")
    trigger = snap[trigger_tf][[a, b]].dropna()
    lookback = int(cfg["strategy"].get("spread_window", cc["window"]))
    h4 = trigger
    if len(h4) < lookback:
        return None
    import numpy as np
    logp = np.log(h4)
    x = logp[b].to_numpy(dtype=float)
    y = logp[a].to_numpy(dtype=float)
    xw, yw = x[-lookback:], y[-lookback:]
    x_mean, y_mean = float(xw.mean()), float(yw.mean())
    denom = float(((xw - x_mean) ** 2).sum())
    if denom <= 0:
        return None
    beta_value = float(((xw - x_mean) * (yw - y_mean)).sum() / denom)
    if not np.isfinite(beta_value) or abs(beta_value) > 3.0 or abs(beta_value) < 0.10:
        return None
    intercept = y_mean - beta_value * x_mean
    residual = logp[a] - (intercept + beta_value * logp[b])
    resid = residual.iloc[-lookback:]
    mean = float(resid.mean())
    std = float(resid.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        return None
    zscore = float((residual.iloc[-1] - mean) / std)
    # Generate the broadest candidate event set; the actual threshold is selected on TRAIN only.
    z_entry = min(float(cfg["strategy"].get("z_entry", 1.5)), 1.25)
    if abs(zscore) < z_entry:
        return None
    signal_mode = cfg["strategy"].get("signal_mode", "mean_reversion")
    if signal_mode == "momentum":
        direction_a = "LONG" if zscore > z_entry else "SHORT"
    else:
        direction_a = "SHORT" if zscore > z_entry else "LONG"
    return {
        "direction_a": direction_a,
        "score": score,
        "correlation_h4": corrs["H4"],
        "confluence": confluence,
        "spread_zscore": zscore,
        "hedge_beta": beta_value,
    }


def _simulate(prices, direction, cfg):
    s = cfg["strategy"]
    entry = float(prices.iloc[0])
    sl_pct, tp1_pct, tp2_pct, trail_pct = map(float, (s["stop_loss_pct"], s["tp1_pct"], s["tp2_pct"], s["trailing_stop_pct"]))
    sign = 1 if direction == "LONG" else -1
    sl, tp1, tp2 = entry * (1 - sign * sl_pct), entry * (1 + sign * tp1_pct), entry * (1 + sign * tp2_pct)
    trail, best, remaining, realized_r, tp1_hit = entry * (1 - sign * trail_pct), entry, 1.0, 0.0, False
    max_bars = int(s.get("max_bars", len(prices)))
    for px in prices.iloc[1:max_bars + 1]:
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
    # Validate the strategy at its real execution timeframe. The H4 relationship
    # remains the primary regime filter, while the spread trigger is evaluated
    # on completed M15 bars exactly as the live alert engine would do.
    strategy_frame = frames[cfg["strategy"].get("timeframe", "M15")]
    warmup = max(cc["window"] * 4, int(cfg["strategy"].get("spread_window", cc["window"])))
    timestamps = list(strategy_frame.index[warmup:-1])
    train_end = int(len(timestamps) * 0.60)
    test_end = int(len(timestamps) * 0.80)
    rows = []
    next_free = {pair: pd.Timestamp.min.tz_localize("UTC") for pair in available}

    # H4 is the mandatory primary filter. Precompute eligible pairs once per
    # completed H4 bar, then evaluate only those candidates on M15. This
    # preserves chronology/no-lookahead while removing redundant calculations.
    h4_index = frames["H4"].index
    h4_candidates = {}
    for h4_ts in h4_index:
        h4_slice = frames["H4"].loc[:h4_ts]
        matrix = correlation_matrix(h4_slice, cc["window"], cc["min_observations"])
        h4_candidates[h4_ts] = [
            (a, b) for a, b in combinations(available, 2)
            if pd.notna(matrix.loc[a, b]) and abs(float(matrix.loc[a, b])) >= cc["threshold"]
        ]

    for idx, ts in enumerate(timestamps):
        h4_pos = h4_index.searchsorted(ts, side="right") - 1
        if h4_pos < 0:
            continue
        h4_ts = h4_index[h4_pos]
        for a, b in h4_candidates.get(h4_ts, []):
            sig = _signal_at(frames, ts, a, b, cfg, weights)
            if not sig:
                continue
            entries = m15.loc[m15.index > ts, a].dropna()
            if entries.empty:
                continue
            entry_ts = entries.index[0]
            if entry_ts < next_free[a]:
                continue
            exit_data = m15.loc[m15.index >= entry_ts, a].dropna()
            exit_type, r, tp1 = _simulate(exit_data, sig["direction_a"], cfg)
            max_bars = int(cfg["strategy"].get("max_bars", len(exit_data)))
            next_free[a] = entry_ts + pd.Timedelta(minutes=15 * max_bars)
            rows.append({
                "signal_ts": ts.isoformat(), "entry_ts": entry_ts.isoformat(), "pair": a,
                "direction": sig["direction_a"], "entry": float(entries.iloc[0]),
                "score": sig["score"], "correlation_h4": round(float(sig["correlation_h4"]), 4),
                "confluence": sig["confluence"], "spread_zscore": sig["spread_zscore"],
                "hedge_beta": sig["hedge_beta"], "exit": exit_type, "r": r, "tp1": tp1,
                "sample": "TRAIN" if idx < train_end else ("TEST" if idx < test_end else "HOLDOUT"),
            })
    raw = pd.DataFrame(rows)
    out = Path("results"); out.mkdir(exist_ok=True)

    # TRAIN-only parameter selection. TEST and FINAL HOLDOUT never participate
    # in choosing parameters. The grid is intentionally small to limit
    # multiple-testing risk; selected parameters are then frozen.
    base = cfg["strategy"].copy()
    grid = product(
        ("mean_reversion", "momentum"),
        (1.25, 1.50, 1.75),
        (0.0020, 0.0025, 0.0030),
        (0.0040, 0.0050),
        (0.0080, 0.0100),
        (0.0020, 0.0025),
    )
    candidates = []
    train_raw = raw[raw["sample"] == "TRAIN"] if not raw.empty else pd.DataFrame()
    for signal_mode, z_entry, sl, tp1, tp2, trail in grid:
        if tp1 <= sl or tp2 <= tp1:
            continue
        candidate_cfg = dict(cfg)
        candidate_cfg["strategy"] = dict(base)
        candidate_cfg["strategy"].update({
            "signal_mode": signal_mode, "z_entry": 1.25, "stop_loss_pct": sl, "tp1_pct": tp1,
            "tp2_pct": tp2, "trailing_stop_pct": trail,
        })
        if train_raw.empty:
            continue
        trade_rows = []
        next_free = {pair: pd.Timestamp.min.tz_localize("UTC") for pair in available}
        for _, sig in train_raw.sort_values("signal_ts").iterrows():
            if abs(float(sig["spread_zscore"])) < z_entry:
                continue
            entry_ts = pd.Timestamp(sig["entry_ts"])
            pair = sig["pair"]
            if entry_ts < next_free[pair]:
                continue
            exit_data = m15.loc[m15.index >= entry_ts, pair].dropna()
            exit_type, r, tp1_hit = _simulate(exit_data, sig["direction"], candidate_cfg)
            max_bars = int(candidate_cfg["strategy"].get("max_bars", len(exit_data)))
            next_free[pair] = entry_ts + pd.Timedelta(minutes=15 * max_bars)
            trade_rows.append({"r": r, "tp1": tp1_hit})
        metrics = _metrics(pd.DataFrame(trade_rows))
        if metrics.get("trades", 0) >= 6:
            candidates.append((metrics.get("expectancy_r", -999), metrics.get("profit_factor") or -999, metrics.get("trades", 0), candidate_cfg["strategy"]))
    if candidates:
        _, _, _, selected_strategy = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    else:
        selected_strategy = dict(base)

    selected_cfg = dict(cfg)
    selected_cfg["strategy"] = dict(selected_strategy)

    final_rows = []
    next_free = {pair: pd.Timestamp.min.tz_localize("UTC") for pair in available}
    for _, sig in raw.sort_values("signal_ts").iterrows():
        if abs(float(sig["spread_zscore"])) < float(selected_strategy.get("z_entry", 1.5)):
            continue
        entry_ts = pd.Timestamp(sig["entry_ts"])
        pair = sig["pair"]
        if entry_ts < next_free[pair]:
            continue
        exit_data = m15.loc[m15.index >= entry_ts, pair].dropna()
        exit_type, r, tp1_hit = _simulate(exit_data, sig["direction"], selected_cfg)
        max_bars = int(selected_strategy.get("max_bars", len(exit_data)))
        next_free[pair] = entry_ts + pd.Timedelta(minutes=15 * max_bars)
        item = sig.to_dict()
        item.update({"exit": exit_type, "r": r, "tp1": tp1_hit})
        final_rows.append(item)

    df = pd.DataFrame(final_rows)
    df.to_csv(out / "strategy_backtest_trades.csv", index=False)
    summary = {
        "data_period_days": cfg["data"]["period_days"],
        "strategy": selected_strategy,
        "train_selection_candidates": len(candidates),
        "all": _metrics(df),
        "train": _metrics(df[df["sample"] == "TRAIN"]) if not df.empty else {"trades": 0},
        "test": _metrics(df[df["sample"] == "TEST"]) if not df.empty else {"trades": 0},
        "holdout": _metrics(df[df["sample"] == "HOLDOUT"]) if not df.empty else {"trades": 0},
    }
    (out / "strategy_backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Forex Signal Strategy Backtest", "", "Relative-value execution on completed H4 relationship signals and next M15 close; no intrabar high/low assumption.", "The strategy trades pair A toward a statistically stretched correlation-implied spread.", "Strict chronological TRAIN/TEST/HOLDOUT split (60/20/20); HOLDOUT is never used for selection or tuning.", ""]
    for name, metrics in (("All", summary["all"]), ("TRAIN", summary["train"]), ("TEST", summary["test"]), ("FINAL HOLDOUT", summary["holdout"])):
        lines += [f"## {name}", ""] + [f"- {k}: {v}" for k, v in metrics.items()] + [""]
    (out / "strategy_backtest_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
