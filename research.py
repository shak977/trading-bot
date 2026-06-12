"""Real-world research layer: news sentiment, analyst/fundamentals (Finnhub),
and macro data (FRED). Every piece is OPTIONAL and defensive — if a key is
missing or a call fails, it returns None/empty and the dashboard simply omits
that section. Keys come from the environment (set as GitHub Actions secrets):
  FINNHUB_API_KEY   (free: https://finnhub.io)
  FRED_API_KEY      (free: https://fredaccount.stlouisfed.org)
"""
from __future__ import annotations

import time

import requests

from config import Config

# ---------------------------------------------------------------- news tone
_POS = {"beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
        "upgrade", "upgraded", "raises", "raised", "record", "strong", "jumps",
        "jump", "gains", "gain", "outperform", "bullish", "tops", "growth",
        "profit", "wins", "win", "approval", "approved", "expands", "rises",
        "rise", "boost", "boosts", "high", "higher", "buy", "rebound", "optimistic"}
_NEG = {"miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
        "downgrade", "downgraded", "cuts", "cut", "warning", "warns", "lawsuit",
        "probe", "recall", "weak", "slumps", "slump", "tumbles", "tumble",
        "bearish", "loss", "losses", "layoffs", "investigation", "halts", "halt",
        "slashes", "slash", "lower", "sinks", "sink", "fears", "concerns", "sell"}


def news_sentiment(news: list[dict]) -> dict | None:
    """Light tone score of the headlines (no API key). -1..+1 with a label."""
    if not news:
        return None
    pos = neg = 0
    for n in news:
        words = (n.get("headline", "") or "").lower().replace(",", " ").replace(".", " ").split()
        pos += sum(1 for w in words if w in _POS)
        neg += sum(1 for w in words if w in _NEG)
    total = pos + neg
    if total == 0:
        return {"label": "Neutral", "score": 0.0, "pos": 0, "neg": 0, "n": len(news)}
    score = round((pos - neg) / total, 2)
    label = "Positive" if score >= 0.25 else "Negative" if score <= -0.25 else "Mixed"
    return {"label": label, "score": score, "pos": pos, "neg": neg, "n": len(news)}


# ---------------------------------------------------------------- Finnhub
_FH = "https://finnhub.io/api/v1"


def _fh_get(path: str, key: str, params: dict, timeout: int = 12):
    p = dict(params, token=key)
    r = requests.get(f"{_FH}{path}", params=p, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.json()


def finnhub_snapshot(symbol: str, cfg: Config) -> dict | None:
    """Analyst consensus, price target vs price, key fundamentals, next earnings.
    Returns a partial dict (whatever the free tier allows); None if no key."""
    key = cfg.finnhub_api_key
    if not key:
        return None
    out: dict = {}
    try:
        rec = _fh_get("/stock/recommendation", key, {"symbol": symbol})
        if rec:
            r0 = rec[0]
            buy = (r0.get("strongBuy", 0) or 0) + (r0.get("buy", 0) or 0)
            hold = r0.get("hold", 0) or 0
            sell = (r0.get("strongSell", 0) or 0) + (r0.get("sell", 0) or 0)
            tot = buy + hold + sell
            if tot:
                consensus = "Buy" if buy > hold and buy > sell else "Sell" if sell > buy and sell > hold else "Hold"
                out["analysts"] = {"buy": buy, "hold": hold, "sell": sell,
                                   "consensus": consensus, "period": r0.get("period")}
    except Exception:  # noqa: BLE001
        pass
    try:
        pt = _fh_get("/stock/price-target", key, {"symbol": symbol})
        if pt and pt.get("targetMean"):
            out["target_mean"] = round(float(pt["targetMean"]), 2)
            out["target_high"] = round(float(pt.get("targetHigh") or 0), 2) or None
            out["target_low"] = round(float(pt.get("targetLow") or 0), 2) or None
    except Exception:  # noqa: BLE001
        pass
    try:
        m = _fh_get("/stock/metric", key, {"symbol": symbol, "metric": "all"})
        met = (m or {}).get("metric", {}) if m else {}
        pe = met.get("peTTM") or met.get("peNormalizedAnnual")
        mc = met.get("marketCapitalization")
        if pe:
            out["pe"] = round(float(pe), 1)
        if mc:
            out["market_cap"] = round(float(mc), 0)  # in millions
        hi = met.get("52WeekHigh"); lo = met.get("52WeekLow")
        if hi:
            out["wk52_high"] = round(float(hi), 2)
        if lo:
            out["wk52_low"] = round(float(lo), 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        import datetime as _dt
        today = _dt.date.today()
        cal = _fh_get("/calendar/earnings", key, {
            "symbol": symbol, "from": today.isoformat(),
            "to": (today + _dt.timedelta(days=90)).isoformat()})
        evs = sorted((cal or {}).get("earningsCalendar", []), key=lambda e: e.get("date", ""))
        for e in evs:
            d = e.get("date")
            if d and d >= today.isoformat():
                out["earnings_date"] = d
                out["earnings_days"] = (_dt.date.fromisoformat(d) - today).days
                break
    except Exception:  # noqa: BLE001
        pass
    return out or None


def finnhub_for_symbols(symbols: list[str], cfg: Config, pause: float = 1.05) -> dict:
    """Fetch snapshots for several symbols, throttled to respect the free limit."""
    if not cfg.finnhub_api_key:
        return {}
    out = {}
    for sym in symbols:
        snap = finnhub_snapshot(sym, cfg)
        if snap:
            out[sym] = snap
        time.sleep(pause)  # free tier ~60/min; ~3 calls each -> stay under
    return out


# ---------------------------------------------------- IPOs + pre-IPO buzz
# Notable private/pre-IPO names worth surfacing even before they have a ticker.
_PREIPO = ["spacex", "starlink", "stripe", "databricks", "openai", "anthropic",
           "discord", "klarna", "canva", "revolut", "chime", "shein", "fanatics",
           "ramp", "anduril", "cerebras", "figure", "rippling"]
_IPO_KW = _PREIPO + ["ipo", "going public", "public debut", "files to go public",
                     "confidentially filed", "prospectus", "s-1", "direct listing",
                     "set to debut", "raise in its ipo"]


def ipo_calendar(cfg: Config, days_ahead: int = 90) -> list[dict]:
    """Upcoming IPO calendar from Finnhub (next ~90 days). [] if no key/none."""
    key = cfg.finnhub_api_key
    if not key:
        return []
    try:
        import datetime as _dt
        today = _dt.date.today()
        data = _fh_get("/calendar/ipo", key, {
            "from": today.isoformat(),
            "to": (today + _dt.timedelta(days=days_ahead)).isoformat()})
        rows = (data or {}).get("ipoCalendar", []) or []
        out = [{
            "date": r.get("date"), "name": r.get("name"), "symbol": r.get("symbol"),
            "exchange": r.get("exchange"), "shares": r.get("numberOfShares"),
            "price": r.get("price"), "value": r.get("totalSharesValue"),
            "status": r.get("status"),
        } for r in rows]
        out.sort(key=lambda x: x.get("date") or "")
        return out
    except Exception:  # noqa: BLE001
        return []


def ipo_buzz_news(cfg: Config, limit: int = 14) -> list[dict]:
    """General market headlines mentioning notable pre-IPO names or IPO keywords.

    This is how a private company like SpaceX (no ticker yet) still surfaces — the
    symbol-keyed feed can't see it, but the general news feed can.
    """
    key = cfg.finnhub_api_key
    if not key:
        return []
    try:
        import datetime as _dt
        items = _fh_get("/news", key, {"category": "general"}) or []
        out, seen = [], set()
        for n in items:
            text = ((n.get("headline", "") or "") + " " + (n.get("summary", "") or "")).lower()
            hit = next((k for k in _IPO_KW if k in text), None)
            if not hit:
                continue
            h = n.get("headline", "")
            if not h or h in seen:
                continue
            seen.add(h)
            ts = n.get("datetime")
            try:
                when = _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M") if ts else ""
            except Exception:  # noqa: BLE001
                when = ""
            out.append({"headline": h, "source": n.get("source", ""), "url": n.get("url", ""),
                        "summary": (n.get("summary", "") or "")[:240], "created_at": when,
                        "match": hit.title()})
            if len(out) >= limit:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------- FRED macro
_FRED = "https://api.stlouisfed.org/fred/series/observations"


def _fred_latest(series: str, key: str, limit: int = 1):
    try:
        r = requests.get(_FRED, params={
            "series_id": series, "api_key": key, "file_type": "json",
            "sort_order": "desc", "limit": limit}, timeout=12)
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o.get("value") not in (".", "", None)]
        return vals if limit > 1 else (vals[0] if vals else None)
    except Exception:  # noqa: BLE001
        return None


def fred_macro(cfg: Config) -> dict | None:
    """Key US macro series + a plain-English backdrop read. None if no key."""
    key = cfg.fred_api_key
    if not key:
        return None
    m: dict = {}
    y10 = _fred_latest("DGS10", key)
    y2 = _fred_latest("DGS2", key)
    unrate = _fred_latest("UNRATE", key)
    fedfunds = _fred_latest("FEDFUNDS", key)
    cpi = _fred_latest("CPIAUCSL", key, limit=13)
    if y10 is not None:
        m["y10"] = round(y10, 2)
    if y2 is not None:
        m["y2"] = round(y2, 2)
    if y10 is not None and y2 is not None:
        m["curve"] = round(y10 - y2, 2)
    if unrate is not None:
        m["unemployment"] = round(unrate, 1)
    if fedfunds is not None:
        m["fed_funds"] = round(fedfunds, 2)
    if cpi and len(cpi) >= 13:
        m["cpi_yoy"] = round((cpi[0] / cpi[12] - 1) * 100, 1)
    if not m:
        return None
    curve = m.get("curve")
    if curve is not None and curve < 0:
        m["backdrop"] = "Cautious"
        m["note"] = "The yield curve is inverted (short rates above long) — historically a late-cycle warning sign."
    elif curve is not None and curve > 0.5:
        m["backdrop"] = "Supportive"
        m["note"] = "A normal, positive yield curve — a more supportive backdrop for risk assets."
    else:
        m["backdrop"] = "Mixed"
        m["note"] = "A flattish yield curve — no strong macro signal either way."
    return m
