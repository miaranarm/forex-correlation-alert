import json
from pathlib import Path
from datetime import datetime, timezone


def write_reports(alerts, cleared, path="results"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alerts": alerts,
        "cleared": cleared,
    }
    (p / "latest_alerts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Forex Signals", "", f"Generated: {payload['generated_at']}", ""]
    if not alerts:
        lines.append("No active signal.")

    for a in alerts:
        for s in a.get("signals", []):
            lines += [
                f"## {s['pair']} — {s['signal']}",
                "",
                f"- **Paire:** {s['pair']}",
                f"- **Signal:** {s['signal']}",
                f"- **Prix d'entrée:** {s['entry']}",
                f"- **SL:** {s['sl']}",
                f"- **TP1:** {s['tp1']}",
                f"- **TP2:** {s['tp2']}",
                f"- **Trailing Stop:** {s['trailing_stop']}",
                f"- **Timeframe:** {s['timeframe']}",
                "",
            ]

    if cleared:
        lines += ["## Cleared", ""] + [f"- {x}" for x in cleared]

    (p / "latest_alerts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
