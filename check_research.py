"""Diagnose the research feeds. Add FINNHUB_API_KEY + FRED_API_KEY to your .env,
then run:  python3 check_research.py
Prints masked keys and the real status of each endpoint so we can see exactly
what your free tier returns."""
from __future__ import annotations

import datetime as dt
import requests
from config import CONFIG


def mask(s):
    return f"{s[:4]}…{s[-4:]} (len {len(s)})" if s else "(empty / not set)"


print("FINNHUB_API_KEY:", mask(CONFIG.finnhub_api_key))
print("FRED_API_KEY   :", mask(CONFIG.fred_api_key))

if CONFIG.finnhub_api_key:
    k = CONFIG.finnhub_api_key
    for name, url, params in [
        ("recommendation", "https://finnhub.io/api/v1/stock/recommendation", {"symbol": "AAPL"}),
        ("price-target", "https://finnhub.io/api/v1/stock/price-target", {"symbol": "AAPL"}),
        ("metric", "https://finnhub.io/api/v1/stock/metric", {"symbol": "AAPL", "metric": "all"}),
        ("earnings cal", "https://finnhub.io/api/v1/calendar/earnings",
         {"symbol": "AAPL", "from": dt.date.today().isoformat(),
          "to": (dt.date.today() + dt.timedelta(days=90)).isoformat()}),
    ]:
        try:
            r = requests.get(url, params=dict(params, token=k), timeout=15)
            body = r.text[:160].replace("\n", " ")
            print(f"\nFinnhub {name}: HTTP {r.status_code}")
            print("  ", body)
        except Exception as e:  # noqa: BLE001
            print(f"\nFinnhub {name}: ERROR {e}")
else:
    print("\n(no Finnhub key in .env — add it to test)")

if CONFIG.fred_api_key:
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": "DGS10", "api_key": CONFIG.fred_api_key,
                                 "file_type": "json", "sort_order": "desc", "limit": 1}, timeout=15)
        print(f"\nFRED DGS10: HTTP {r.status_code}")
        print("  ", r.text[:200].replace("\n", " "))
    except Exception as e:  # noqa: BLE001
        print(f"\nFRED: ERROR {e}")
else:
    print("\n(no FRED key in .env — add it to test)")
