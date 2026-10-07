import json
from pathlib import Path

def load_state(path="results/alert_state.json"):
    p=Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def classify(current, previous):
    current_keys={f"{a['pair_a']}|{a['pair_b']}" for a in current}
    previous_keys=set(previous)
    for a in current:
        key=f"{a['pair_a']}|{a['pair_b']}"
        a["status"]="MAINTAINED" if key in previous_keys else "NEW"
    return current, sorted(previous_keys-current_keys)

def save_state(alerts, path="results/alert_state.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state={f"{a['pair_a']}|{a['pair_b']}":{"score":a["score"]} for a in alerts}
    Path(path).write_text(json.dumps(state,indent=2),encoding="utf-8")
