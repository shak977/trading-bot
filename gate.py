"""Print 'true' if today is a US trading day (per Alpaca's calendar), else 'false'.

Used by the GitHub workflow to skip scheduled runs on weekends/holidays.
Fails open: if we can't check, prints 'true' so a run is never wrongly skipped.
"""
from __future__ import annotations

import datetime

import requests

from config import CONFIG


def main() -> None:
    if not CONFIG.api_key or not CONFIG.secret_key:
        print("true")
        return
    base = "https://paper-api.alpaca.markets" if CONFIG.paper else "https://api.alpaca.markets"
    today = datetime.date.today().isoformat()
    try:
        r = requests.get(
            f"{base}/v2/calendar",
            headers={"APCA-API-KEY-ID": CONFIG.api_key, "APCA-API-SECRET-KEY": CONFIG.secret_key},
            params={"start": today, "end": today},
            timeout=12,
        )
        r.raise_for_status()
        days = r.json()
        print("true" if any(d.get("date") == today for d in days) else "false")
    except Exception:  # noqa: BLE001 - never block on an error
        print("true")


if __name__ == "__main__":
    main()
