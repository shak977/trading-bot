"""Build-health watchdog — pings ntfy if the deployed dashboard has gone stale.

Runs as its OWN scheduled workflow (health.yml), independent of the main build, so it keeps
watching even when the build itself is what broke. It fetches the live signals.json, checks how
old the build is, and if it's stale during US market hours it sends one high-priority ntfy alert.

No Alpaca keys needed — it only reads the public site. Never raises.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SITE = os.getenv("SITE_URL", "https://shak977.github.io/trading-bot").rstrip("/")
NTFY = os.getenv("ALERT_NTFY_TOPIC", "").strip()
STALE_MIN = int(os.getenv("STALE_MINUTES", "25"))


def _market_open() -> bool:
    et = datetime.now(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:                      # Sat/Sun
        return False
    mins = et.hour * 60 + et.minute
    return 570 <= mins < 960                    # 09:30–16:00 ET


def _fetch_generated_ts():
    url = f"{SITE}/signals.json?cb={int(datetime.now().timestamp())}"
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read()).get("generated_ts")


def main() -> None:
    if not _market_open():
        print("watchdog: market closed — nothing to check")
        return
    reason = None
    try:
        ts = _fetch_generated_ts()
        if ts is None:
            reason = "the dashboard has no build timestamp"
        else:
            age = (datetime.now(timezone.utc).timestamp() - float(ts)) / 60
            if age <= STALE_MIN:
                print(f"watchdog: fresh ({age:.0f} min old) — OK")
                return
            reason = f"the last build was {int(age)} min ago (over {STALE_MIN})"
    except Exception as exc:  # noqa: BLE001 - any failure to read the site = it's likely down
        reason = f"the dashboard couldn't be read ({type(exc).__name__})"

    title = "⚠ Signal Desk build is stale"
    body = f"{title}: {reason}. The build may be failing — check GitHub Actions.\n{SITE}"
    print(body)
    if NTFY:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY}", data=body.encode("utf-8"),
                headers={"Title": title, "Priority": "high", "Tags": "warning"})
            urllib.request.urlopen(req, timeout=10)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
