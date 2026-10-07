import pandas as pd
from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation
from .scoring import trend_direction, score_signal, relationship_alignment, timeframe_confluence
from .state import load_state, classify, save_state
from .report import write_reports
from .alerts import emit

def run():
    cfg=load_config(); series=[]
    for pair in cfg["pairs"]:
        try:
            series.append(fetch_15m(pair,cfg["data"]["period_days"],cfg["data"]["request_timeout_seconds"]))
        except Exception as exc:
            print(f"WARNING: {pair}: {exc}")
    if len(series)<2: raise RuntimeError("Not enough market data.")
    frames=build_timeframes(pd.concat(series,axis=1).dropna())
    window=cfg["correlation"]["window"]; threshold=cfg["correlation"]["threshold"]
    weights={"correlation":cfg["scoring"]["correlation_weight"],"h4":cfg["scoring"]["h4_weight"],"h1":cfg["scoring"]["h1_weight"],"m15":cfg["scoring"]["m15_weight"]}
    alerts=[]
    for i,a in enumerate(cfg["pairs"]):
        for b in cfg["pairs"][i+1:]:
            corrs={tf:pair_correlation(frames[tf],a,b,window) for tf in ("H4","H1","M15")}
            # H4 is the anchor: a correlation alert is considered actionable only
            # when the primary H4 relationship itself reaches the threshold.
            if abs(corrs["H4"])<threshold:
                continue
            dirs={tf:(trend_direction(frames[tf][a]),trend_direction(frames[tf][b])) for tf in ("H4","H1","M15")}
            aligns={tf:relationship_alignment(corrs[tf],*dirs[tf]) for tf in dirs}
            confluence=timeframe_confluence((aligns["H4"],aligns["H1"],aligns["M15"]))
            if confluence=="WEAK":
                continue
            score=score_signal(corrs["H4"],aligns["H4"],aligns["H1"],aligns["M15"],weights,threshold)
            if score<cfg["scoring"]["minimum_alert_score"]:
                continue
            alerts.append({
                "pair_a":a,"pair_b":b,
                "correlation_h4":round(corrs["H4"],4),"correlation_h1":round(corrs["H1"],4),"correlation_m15":round(corrs["M15"],4),
                "h4_alignment":aligns["H4"],"h1_alignment":aligns["H1"],"m15_alignment":aligns["M15"],
                "confluence":confluence,"score":score
            })
    alerts,cleared=classify(alerts,load_state())
    alerts.sort(key=lambda x:(x["score"],x["correlation_h4"]),reverse=True)
    write_reports(alerts,cleared); save_state(alerts); emit(alerts)
    for key in cleared: print(f"- CLEARED {key}")

if __name__=="__main__":
    run()
