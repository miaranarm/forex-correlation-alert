from datetime import datetime, timezone

def emit(alerts):
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not alerts:
        print(f"[{now}] No correlation alert.")
        return
    print(f"[{now}] {len(alerts)} correlation alert(s)")
    for a in alerts:
        print(
            f"- {a['pair_a']} / {a['pair_b']} | "
            f"H4={a['correlation_h4']:+.3f} | "
            f"H1={a['correlation_h1']:+.3f} | "
            f"M15={a['correlation_m15']:+.3f} | "
            f"confluence={a['confluence']} | "
            f"stability={a['stability']:.2f} | "
            f"score={a['score']:.1f}"
        )
