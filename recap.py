"""End-of-day recap worker — pushes a plain-English summary of the day to ntfy after the close.

Reads the committed track record + paper book and the live signals.json, then sends one digest:
the regime, fresh calls today, what resolved (win/loss), the running win-rate/expectancy, and how
many positions are open. Runs as its own scheduled workflow (recap.yml). Never raises.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

SITE = os.getenv("SITE_URL", "https://shak977.github.io/trading-bot").rstrip("/")
NTFY = os.getenv("ALERT_NTFY_TOPIC", "").strip()
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _fetch_signals() -> dict:
    try:
        url = f"{SITE}/signals.json?cb={int(datetime.now().timestamp())}"
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    snap = _fetch_signals()
    sigs = snap.get("signals", []) or []
    regime = (snap.get("regime") or {}).get("label", "n/a")
    fresh = [s for s in sigs if s.get("is_fresh") and s.get("action") in ("BUY", "SHORT")]
    fresh_txt = ", ".join(f"{s['symbol']} {s['action']}" for s in fresh[:8]) or "none"

    tr = _load("track_record.json", [])
    resolved = [t for t in tr if t.get("status") in ("win", "loss", "expired")]
    wins = sum(1 for t in resolved if t["status"] == "win")
    wr = round(wins / len(resolved) * 100) if resolved else None
    today_done = [t for t in resolved if t.get("exit_date") == TODAY]
    done_txt = ", ".join(f"{t['symbol']} {t['status'].upper()} {t.get('return_pct', '?')}%"
                         for t in today_done[:8]) or "none resolved today"
    open_n = sum(1 for t in tr if t.get("status") == "open")

    paper = _load("paper_orders.json", [])
    paper_open = sum(1 for p in paper if isinstance(p, dict) and p.get("status") == "open") if isinstance(paper, list) else 0

    lines = [
        f"📊 Signal Desk — daily recap ({TODAY})",
        f"Regime: {regime}",
        f"Fresh calls today: {len(fresh)} ({fresh_txt})",
        f"Resolved today: {done_txt}",
        (f"Track record: {wr}% win over {len(resolved)} calls · {open_n} still open"
         if wr is not None else f"Track record: building · {open_n} open"),
        f"Paper book: {paper_open} open position(s)",
        SITE,
    ]
    body = "\n".join(lines)
    print(body)
    if NTFY:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY}", data=body.encode("utf-8"),
                headers={"Title": f"Daily recap — {regime}", "Tags": "bar_chart"})
            urllib.request.urlopen(req, timeout=10)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
