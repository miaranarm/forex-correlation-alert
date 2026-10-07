import json
from pathlib import Path
from datetime import datetime, timezone

def write_reports(alerts, cleared, path="results"):
    p=Path(path); p.mkdir(parents=True,exist_ok=True)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"alerts":alerts,"cleared":cleared}
    (p/"latest_alerts.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lines=["# Forex Correlation Alerts","",f"Generated: {payload['generated_at']}",""]
    if not alerts:
        lines.append("No active alert.")
    for a in alerts:
        lines.append(f"- **{a['status']}** {a['pair_a']} / {a['pair_b']} | H4 {a['correlation_h4']:+.3f} | H1 {a['correlation_h1']:+.3f} | M15 {a['correlation_m15']:+.3f} | score {a['score']:.1f}/100")
    if cleared:
        lines += ["","## Cleared",""]+[f"- {x}" for x in cleared]
    (p/"latest_alerts.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
