from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability
from .scoring import (
    trend_direction,
    score_signal,
    relationship_alignment,
    timeframe_confluence,
)


def _evaluate_snapshot(
    snap: dict[str, pd.DataFrame],
    h4: pd.DataFrame,
    pos: int,
    a: str,
    b: str,
    cfg: dict,
    weights: dict,
    filtered: bool,
):
    corr_cfg = cfg["correlation"]
    score_cfg = cfg["scoring"]
    window = corr_cfg["window"]
    threshold = corr_cfg["threshold"]

    corrs = {
        tf: pair_correlation(snap[tf], a, b, window)
        for tf in ("H4", "H1", "M15")
    }
    if abs(corrs["H4"]) < threshold:
        return None

    regime = rolling_correlation(
        snap["H4"], a, b, window, corr_cfg["short_window"]
    )
    stability = correlation_stability(
        snap["H4"], a, b, window, corr_cfg["stability_segments"]
    )

    if filtered:
        if not regime:
            return None
        if stability < corr_cfg["minimum_stability"]:
            return None
        if abs(regime["spread"]) > corr_cfg["maximum_regime_drift"]:
            return None

        dirs = {
            tf: (trend_direction(snap[tf][a]), trend_direction(snap[tf][b]))
            for tf in ("H4", "H1", "M15")
        }
        aligns = {
            tf: relationship_alignment(corrs[tf], *dirs[tf])
            for tf in dirs
        }
        confluence = timeframe_confluence(
            (aligns["H4"], aligns["H1"], aligns["M15"])
        )
        if confluence == "WEAK":
            return None

        score = score_signal(
            corrs["H4"],
            aligns["H4"],
            aligns["H1"],
            aligns["M15"],
            weights,
            threshold,
        )
        if score < score_cfg["minimum_alert_score"]:
            return None
    else:
        confluence = "RAW"
        score = round(abs(corrs["H4"]) / threshold * 100.0, 2)

    return {
        "timestamp": str(h4.index[pos]),
        "pair_a": a,
        "pair_b": b,
        "score": score,
        "confluence": confluence,
        "correlation_h4": round(corrs["H4"], 4),
        "correlation_h1": round(corrs["H1"], 4),
        "correlation_m15": round(corrs["M15"], 4),
        "stability": round(stability, 4),
        "regime_drift": round(regime["spread"], 4) if regime else None,
    }


def _add_forward_outcomes(rows: list[dict], h4: pd.DataFrame, pos: int):
    if not rows:
        return
    horizons = {"4h": 1, "8h": 2, "16h": 4, "24h": 6}
    base = h4.iloc[pos]
    for row in rows:
        a, b = row["pair_a"], row["pair_b"]
        expected = 1 if row["correlation_h4"] >= 0 else -1
        for label, steps in horizons.items():
            future_pos = pos + steps
            if future_pos >= len(h4):
                row[f"relationship_correct_{label}"] = None
                row[f"spread_{label}"] = None
                continue
            future = h4.iloc[future_pos]
            pa = float(future[a] / base[a] - 1)
            pb = float(future[b] / base[b] - 1)
            product = pa * pb
            row[f"relationship_correct_{label}"] = bool(
                product > 0 if expected == 1 else product < 0
            )
            row[f"spread_{label}"] = pa - pb


def _aggregate(df: pd.DataFrame, prefix: str = "") -> dict:
    if df.empty:
        return {"observations": 0}
    result = {
        "observations": int(len(df)),
        "unique_timestamps": int(df["timestamp"].nunique()),
        "unique_pair_combinations": int(
            (df["pair_a"] + "|" + df["pair_b"]).nunique()
        ),
        "mean_score": round(float(df["score"].mean()), 2),
        "mean_stability": round(float(df["stability"].mean()), 4),
        "mean_abs_h4_correlation": round(
            float(df["correlation_h4"].abs().mean()), 4
        ),
    }
    for horizon in ("4h", "8h", "16h", "24h"):
        col = f"relationship_correct_{horizon}"
        valid = df[col].dropna()
        if len(valid):
            result[f"relationship_hit_rate_{horizon}"] = round(
                float(valid.mean()), 4
            )
            result[f"correct_relationships_{horizon}"] = int(valid.sum())
    return result


def run_backtest():
    cfg = load_config()
    pairs = cfg["pairs"]
    series = []
    for pair in pairs:
        series.append(
            fetch_15m(
                pair,
                cfg["data"]["period_days"],
                cfg["data"]["request_timeout_seconds"],
            )
        )

    raw = pd.concat(series, axis=1).dropna()
    frames = build_timeframes(raw)
    h4 = frames["H4"]

    corr_cfg = cfg["correlation"]
    window = corr_cfg["window"]
    min_obs = corr_cfg.get("min_observations", 60)
    weights = {
        "correlation": cfg["scoring"]["correlation_weight"],
        "h4": cfg["scoring"]["h4_weight"],
        "h1": cfg["scoring"]["h1_weight"],
        "m15": cfg["scoring"]["m15_weight"],
    }

    filtered_rows = []
    raw_rows = []

    # The signal timestamp is the END of the completed H4 bar.
    # Forward outcomes start at the next H4 close, so no future price
    # information leaks into the signal.
    for pos in range(window, len(h4) - 6):
        ts = h4.index[pos]
        snap = {tf: frames[tf].loc[:ts] for tf in ("H4", "H1", "M15")}

        for i, a in enumerate(pairs):
            for b in pairs[i + 1:]:
                try:
                    if min(
                        len(snap["H4"][[a, b]].dropna()),
                        len(snap["H1"][[a, b]].dropna()),
                        len(snap["M15"][[a, b]].dropna()),
                    ) < min_obs:
                        continue

                    raw_signal = _evaluate_snapshot(
                        snap, h4, pos, a, b, cfg, weights, filtered=False
                    )
                    filtered_signal = _evaluate_snapshot(
                        snap, h4, pos, a, b, cfg, weights, filtered=True
                    )

                    if raw_signal:
                        _add_forward_outcomes([raw_signal], h4, pos)
                        raw_rows.append(raw_signal)

                    if filtered_signal:
                        _add_forward_outcomes([filtered_signal], h4, pos)
                        filtered_rows.append(filtered_signal)

                except (KeyError, TypeError, ValueError):
                    continue

    out = Path("results")
    out.mkdir(exist_ok=True)

    raw_df = pd.DataFrame(raw_rows)
    filtered_df = pd.DataFrame(filtered_rows)

    raw_df.to_csv(out / "backtest_raw_observations.csv", index=False)
    filtered_df.to_csv(out / "backtest_observations.csv", index=False)

    breakdown_rows = []
    if not filtered_df.empty:
        for confluence, group in filtered_df.groupby("confluence"):
            item = {"group": f"confluence:{confluence}"}
            item.update(_aggregate(group))
            breakdown_rows.append(item)

        for sign, group in filtered_df.groupby(
            filtered_df["correlation_h4"] >= 0
        ):
            item = {"group": "positive_correlation" if sign else "negative_correlation"}
            item.update(_aggregate(group))
            breakdown_rows.append(item)

        filtered_df["score_bucket"] = pd.cut(
            filtered_df["score"],
            bins=[-1, 69.999, 79.999, 89.999, 101],
            labels=["65-69", "70-79", "80-89", "90-100"],
        )
        for bucket, group in filtered_df.groupby("score_bucket", observed=True):
            item = {"group": f"score:{bucket}"}
            item.update(_aggregate(group))
            breakdown_rows.append(item)

    breakdown = pd.DataFrame(breakdown_rows)
    breakdown.to_csv(out / "backtest_breakdown.csv", index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_period_days": cfg["data"]["period_days"],
        "h4_bars": int(len(h4)),
        "lookback_h4_bars": int(window),
        "minimum_observations": int(min_obs),
        "raw": _aggregate(raw_df),
        "filtered": _aggregate(filtered_df),
    }

    if not raw_df.empty and not filtered_df.empty:
        summary["filter_effect"] = {
            "observation_retention": round(
                len(filtered_df) / len(raw_df), 4
            ),
            "hit_rate_lift_4h": round(
                _aggregate(filtered_df).get("relationship_hit_rate_4h", 0)
                - _aggregate(raw_df).get("relationship_hit_rate_4h", 0),
                4,
            ),
            "hit_rate_lift_8h": round(
                _aggregate(filtered_df).get("relationship_hit_rate_8h", 0)
                - _aggregate(raw_df).get("relationship_hit_rate_8h", 0),
                4,
            ),
            "hit_rate_lift_16h": round(
                _aggregate(filtered_df).get("relationship_hit_rate_16h", 0)
                - _aggregate(raw_df).get("relationship_hit_rate_16h", 0),
                4,
            ),
            "hit_rate_lift_24h": round(
                _aggregate(filtered_df).get("relationship_hit_rate_24h", 0)
                - _aggregate(raw_df).get("relationship_hit_rate_24h", 0),
                4,
            ),
        }

    (out / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Forex Correlation Historical Audit",
        "",
        "This is a relationship-persistence audit, not a trading P&L backtest.",
        "Signal timestamps use completed right-labelled H4 bars; outcomes start after the signal.",
        "",
        f"- Data window: {summary['data_period_days']} days",
        f"- H4 bars: {summary['h4_bars']}",
        f"- H4 lookback: {summary['lookback_h4_bars']} bars",
        f"- Minimum observations: {summary['minimum_observations']}",
        "",
        "## Raw threshold baseline",
        "",
    ]
    for k, v in summary["raw"].items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Filtered alert model", ""]
    for k, v in summary["filtered"].items():
        lines.append(f"- {k}: {v}")

    if "filter_effect" in summary:
        lines += ["", "## Filter effect", ""]
        for k, v in summary["filter_effect"].items():
            lines.append(f"- {k}: {v}")

    if breakdown_rows:
        lines += ["", "## Breakdown", ""]
        for item in breakdown_rows:
            lines.append(
                f"- **{item['group']}**: "
                f"n={item['observations']}, "
                f"4h={item.get('relationship_hit_rate_4h', 'n/a')}, "
                f"16h={item.get('relationship_hit_rate_16h', 'n/a')}, "
                f"24h={item.get('relationship_hit_rate_24h', 'n/a')}"
            )

    (out / "backtest_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_backtest()
