import pandas as pd
from .config import load_config
from .data import fetch_15m, build_timeframes
from .correlation import pair_correlation
from .scoring import trend_direction, score_signal, relationship_alignment
from .state import load_state, classify, save_state
from .report import write_reports
from .alerts import emit

def run():
    cfg=load_config()
    series=[]
    for pair in cfg["pairs"]:
        try:
            series.append(fetch_15m(pair,cfg["data"]["period_days"],cfg["data"]["request_timeout_seconds"]))
        except Exception as exc:
            print(f"WARNING: {pair}: {exc}")
    if len(series)<2:
        raise RuntimeError("Not enough market data.")
    raw=pd.concat(series,axis=1).dropna()
    frames=build_timeframes(raw)
    window=cfg["correlation"]["window"]; threshold=cfg["correlation"]["threshold"]
    weights={"correlation":cfg["scoring"]["correlation_weight"],"h4":cfg["scoring"]["h4_weight"],"h1":cfg["scoring"]["h1_weight"],"m15":cfg["scoring"]["m15_weight"]}
    alerts=[]; pairs=cfg["pairs"]
    for i,a in enumerate(pairs):
        for b in pairs[i+1:]:
            corrs={tf:pair_correlation(frames[tf],a,b,window) for tf in ("H4","H1","M15")}
            if all(abs(v)<threshold for v in corrs.values()):
                continue
            dirs={tf:(trend_direction(frames[tf][a]),trend_direction(frames[tf][b])) for tf in ("H4","H1","M15")}
            aligns={tf:relationship_alignment(corrs[tf],*dirs[tf]) for tf in dirs}
            score=score_signal(corrs["H4"],aligns["H4"],aligns["H1"],aligns["M15"],weights)
            if score>=cfg["scoring"]["minimum_alert_score"]:
                alerts.append({"pair_a":a,"pair_b":b,"correlation_h4":corrs["H4"],"correlation_h1":corrs["H1"],"correlation_m15":corrs["M15"],"h4_alignment":aligns["H4"],"h1_alignment":aligns["H1"],"m15_alignment":aligns["M15"],"score":score})
    alerts,cleared=classify(alerts,load_state())
    alerts.sort(key=lambda x:x["score"],reverse=True)
    write_reports(alerts,cleared)
    save_state(alerts)
    emit(alerts)
    for key in cleared:
        print(f"- CLEARED {key}")

if __name__=="__main__":
    run()
