from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability
from .scoring import trend_direction, score_signal, relationship_alignment, timeframe_confluence

def run_backtest():
    cfg=load_config()
    pairs=cfg["pairs"]
    series=[]
    for pair in pairs:
        series.append(fetch_15m(pair,cfg["data"]["period_days"],cfg["data"]["request_timeout_seconds"]))
    raw=pd.concat(series,axis=1).dropna()
    frames=build_timeframes(raw)
    h4=frames["H4"]
    window=cfg["correlation"]["window"]
    threshold=cfg["correlation"]["threshold"]
    min_stability=cfg["correlation"]["minimum_stability"]
    max_drift=cfg["correlation"]["maximum_regime_drift"]
    short_window=cfg["correlation"]["short_window"]
    segs=cfg["correlation"]["stability_segments"]
    weights={"correlation":cfg["scoring"]["correlation_weight"],"h4":cfg["scoring"]["h4_weight"],"h1":cfg["scoring"]["h1_weight"],"m15":cfg["scoring"]["m15_weight"]}
    rows=[]
    # Evaluate only snapshots for which a full lookback and a 4-H4-bar forward horizon exist.
    for pos in range(window, len(h4)-4):
        ts=h4.index[pos]
        snap={tf:frames[tf].loc[:ts] for tf in ("H4","H1","M15")}
        for i,a in enumerate(pairs):
            for b in pairs[i+1:]:
                try:
                    corrs={tf:pair_correlation(snap[tf],a,b,window) for tf in ("H4","H1","M15")}
                    if abs(corrs["H4"])<threshold: continue
                    regime=rolling_correlation(snap["H4"],a,b,window,short_window)
                    stability=correlation_stability(snap["H4"],a,b,window,segs)
                    if not regime or stability<min_stability or abs(regime["spread"])>max_drift: continue
                    dirs={tf:(trend_direction(snap[tf][a]),trend_direction(snap[tf][b])) for tf in ("H4","H1","M15")}
                    aligns={tf:relationship_alignment(corrs[tf],*dirs[tf]) for tf in dirs}
                    confluence=timeframe_confluence((aligns["H4"],aligns["H1"],aligns["M15"]))
                    if confluence=="WEAK": continue
                    score=score_signal(corrs["H4"],aligns["H4"],aligns["H1"],aligns["M15"],weights,threshold)
                    if score<cfg["scoring"]["minimum_alert_score"]: continue
                    future=h4.iloc[pos+4]
                    pa=float(future[a]/h4.iloc[pos][a]-1)
                    pb=float(future[b]/h4.iloc[pos][b]-1)
                    expected=1 if corrs["H4"]>=0 else -1
                    product=pa*pb
                    correct=(pa*pb>0) if expected==1 else (pa*pb<0)
                    rows.append({"timestamp":str(ts),"pair_a":a,"pair_b":b,"score":score,"confluence":confluence,"correlation_h4":round(corrs["H4"],4),"stability":round(stability,4),"regime_drift":round(regime["spread"],4),"forward_a_16h":pa,"forward_b_16h":pb,"relationship_correct":bool(correct),"relationship_product":product})
                except (KeyError,TypeError,ValueError):
                    continue
    out=Path("results"); out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    summary={"generated_at":pd.Timestamp.utcnow().isoformat(),"observations":len(df)}
    if len(df):
        summary.update({"unique_timestamps":int(df.timestamp.nunique()),"unique_pair_combinations":int((df.pair_a+"|"+df.pair_b).nunique()),"correct_relationships":int(df.relationship_correct.sum()),"relationship_hit_rate":round(float(df.relationship_correct.mean()),4),"mean_score":round(float(df.score.mean()),2),"mean_stability":round(float(df.stability.mean()),4),"mean_abs_h4_correlation":round(float(df.correlation_h4.abs().mean()),4)})
        df.to_csv(out/"backtest_observations.csv",index=False)
    else:
        df.to_csv(out/"backtest_observations.csv",index=False)
    (out/"backtest_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Correlation Alert Backtest","",f"- Observations: {summary['observations']}"]
    for k,v in summary.items():
        if k!="generated_at": lines.append(f"- {k}: {v}")
    (out/"backtest_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    run_backtest()
