"""Unofficial TradingView Technical Analysis ratings (keyless).

Pulls TradingView's aggregate recommendation — the "Strong Buy → Strong Sell" gauge that
summarizes ~26 moving-average + oscillator signals — for the DAILY and WEEKLY timeframes,
via the public scanner endpoint. We submit each ticker under several exchange prefixes
(NASDAQ/NYSE/AMEX) and keep whichever resolves, so we don't need exact exchange data.

This is an INDEPENDENT cross-check on our own confluence engine and adds a multi-timeframe
view we otherwise lack — never the sole input. Caveats: unofficial (no SLA, can change), and
the endpoint is often blocked from datacenter IPs (GitHub Actions) — in that case route it
through the Cloudflare Worker. Any failure returns {} so the build never breaks.
"""
from __future__ import annotations

import requests

_URL = "https://scanner.tradingview.com/america/scan"
_EXCH = ("NASDAQ", "NYSE", "AMEX")
_HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

_BULL = {"Buy", "Strong Buy"}
_BEAR = {"Sell", "Strong Sell"}


def _bucket(x) -> str | None:
    """Map TradingView's [-1, 1] recommend score to its labelled rating."""
    if x is None:
        return None
    if x >= 0.5:
        return "Strong Buy"
    if x >= 0.1:
        return "Buy"
    if x <= -0.5:
        return "Strong Sell"
    if x <= -0.1:
        return "Sell"
    return "Neutral"


def _parse(data: list) -> dict:
    out: dict = {}
    for row in data or []:
        sym = (row.get("s") or "").split(":")[-1]
        d = row.get("d") or []
        if not sym or sym in out or not d:
            continue
        day = _bucket(d[0]) if len(d) > 0 else None
        wk = _bucket(d[1]) if len(d) > 1 else None
        if day is None and wk is None:
            continue
        out[sym] = {"d": day, "w": wk,
                    "score": round(d[0], 3) if (d and d[0] is not None) else None}
    return out


def ratings(symbols: list[str], proxy: str | None = None, timeout: int = 12) -> dict:
    """Return {symbol: {"d": daily rating, "w": weekly rating, "score": daily float}}.

    If ``proxy`` (the Cloudflare Worker base URL) is given, route via ``{proxy}?tv=...``
    GET — needed because TradingView's endpoint blocks datacenter IPs. Falls back to a
    direct POST otherwise. Any failure returns {} so the build never breaks."""
    syms = [s for s in dict.fromkeys(symbols) if s]
    if not syms:
        return {}
    # 1) via the Worker proxy (residential-ish egress) when configured
    if proxy:
        try:
            import urllib.parse as _up
            u = proxy.rstrip("/") + "/?tv=" + _up.quote(",".join(syms))
            r = requests.get(u, timeout=timeout)
            if r.status_code == 200:
                parsed = _parse((r.json() or {}).get("data", []) or [])
                if parsed:
                    return parsed
        except Exception:  # noqa: BLE001
            pass
    # 2) direct (works locally; usually blocked from CI/datacenter IPs)
    tickers = [f"{e}:{s}" for s in syms for e in _EXCH]
    payload = {"symbols": {"tickers": tickers, "query": {"types": []}},
               "columns": ["Recommend.All", "Recommend.All|1W"]}
    try:
        r = requests.post(_URL, json=payload, headers=_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return {}
        return _parse((r.json() or {}).get("data", []) or [])
    except Exception:  # noqa: BLE001
        return {}


def alignment(tv: dict | None, direction: str) -> str | None:
    """How TradingView's rating lines up with OUR trade direction.
    Returns 'agree' (both timeframes with us), 'oppose' (TV against us), 'mixed', or None."""
    if not tv:
        return None
    want = _BULL if direction != "SHORT" else _BEAR
    opp = _BEAR if direction != "SHORT" else _BULL
    vals = [tv.get("d"), tv.get("w")]
    hits = sum(1 for x in vals if x in want)
    opps = sum(1 for x in vals if x in opp)
    if hits >= 2:
        return "agree"
    if opps >= 1 and hits == 0:
        return "oppose"
    return "mixed"
