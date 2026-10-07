from datetime import datetime, timezone


def emit(alerts):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not alerts:
        print(f"[{now}] No active signal.")
        return
    for alert in alerts:
        for s in alert.get("signals", []):
            print(
                f"{s['pair']} | {s['signal']} | Entry {s['entry']} | "
                f"SL {s['sl']} | TP1 {s['tp1']} | TP2 {s['tp2']} | "
                f"Trailing {s['trailing_stop']} | TF {s['timeframe']}"
            )
