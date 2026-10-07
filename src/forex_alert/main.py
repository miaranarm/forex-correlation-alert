import pandas as pd
from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation, rolling_correlation, correlation_stability
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
    stability_min=cfg["correlation"]["minimum_stability"]
    drift_max=cfg["correlation"]["maximum_regime_drift"]
    weights={"correlation":cfg["scoring"]["correlation_weight"],"h4":cfg["scoring"]["h4_weight"],"h1":cfg["scoring"]["h1_weight"],"m15":cfg["scoring"]["m15_weight"]}
    alerts=[]
    for i,a in enumerate(cfg["pairs"]):
        for b in cfg["pairs"][i+1:]:
            corrs={tf:pair_correlation(frames[tf],a,b,window) for tf in ("H4","H1","M15")}
            if abs(corrs["H4"])<threshold: continue
            regime=rolling_correlation(frames["H4"],a,b,window,cfg["correlation"]["short_window"])
            if not regime: continue
            stability=correlation_stability(frames["H4"],a,b,window,cfg["correlation"]["stability_segments"])
            if stability<stability_min or abs(regime["spread"])>drift_max: continue
            dirs={tf:(trend_direction(frames[tf][a]),trend_direction(frames[tf][b])) for tf in ("H4","H1","M15")}
            aligns={tf:relationship_alignment(corrs[tf],*dirs[tf]) for tf in dirs}
            confluence=timeframe_confluence((aligns["H4"],aligns["H1"],aligns["M15"]))
            if confluence=="WEAK": continue
            score=score_signal(corrs["H4"],aligns["H4"],aligns["H1"],aligns["M15"],weights,threshold)
            if score<cfg["scoring"]["minimum_alert_score"]: continue
            alerts.append({"pair_a":a,"pair_b":b,"correlation_h4":round(corrs["H4"],4),"correlation_h1":round(corrs["H1"],4),"correlation_m15":round(corrs["M15"],4),"h4_alignment":aligns["H4"],"h1_alignment":aligns["H1"],"m15_alignment":aligns["M15"],"confluence":confluence,"stability":stability,"h4_short_correlation":round(regime["short"],4),"h4_long_correlation":round(regime["long"],4),"regime_drift":round(regime["spread"],4),"score":score})
    alerts,cleared=classify(alerts,load_state())
    alerts.sort(key=lambda x:(x["score"],x["stability"]),reverse=True)
    write_reports(alerts,cleared); save_state(alerts); emit(alerts)
    for key in cleared: print(f"- CLEARED {key}")

if __name__=="__main__":
    run()
