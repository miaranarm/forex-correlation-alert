from datetime import datetime, timezone

def emit(alerts):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not alerts:
        print(f"[{now}] No correlation alert.")
        return
    print(f"[{now}] {len(alerts)} correlation alert(s)")
    for a in alerts:
        print(f"- {a['pair_a']} / {a['pair_b']} | r={a['correlation']:+.3f} | score={a['score']:.1f}")
