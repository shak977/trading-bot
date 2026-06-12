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


# Pre-IPO / IPO search queries for the keyless Google News RSS feed.
_IPO_QUERIES = [
    ("SpaceX", "SpaceX IPO"), ("Starlink", "Starlink IPO"), ("Stripe", "Stripe IPO"),
    ("Databricks", "Databricks IPO"), ("OpenAI", "OpenAI IPO"), ("Anthropic", "Anthropic IPO"),
    ("Anduril", "Anduril IPO"), ("IPO", "upcoming tech IPO this week"),
]


def yahoo_quotes(symbols: list[str]) -> dict:
    """Consolidated (full-market) last price + previous close per symbol, via Yahoo.

    Keyless. Used to show card prices/day-change that match Google/Yahoo, since the
    scan itself runs on Alpaca's IEX feed (one exchange) whose close can drift a bit.
    Returns {SYM: {"price": float, "prev_close": float|None}}. Defensive — skips on error.
    """
    out: dict = {}
    for sym in symbols:
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                   "?range=5d&interval=1d")
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            res = ((r.json().get("chart", {}) or {}).get("result") or [None])[0]
            if not res:
                continue
            meta = res.get("meta", {}) or {}
            price = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            if price:
                out[sym] = {"price": round(float(price), 2),
                            "prev_close": round(float(prev), 2) if prev else None}
        except Exception:  # noqa: BLE001
            continue
    return out


def _parse_rss(xml_bytes: bytes, tag: str, default_source: str = "Google News") -> list[dict]:
    """Parse an RSS payload (Google News / Yahoo Finance) into [{headline,url,source,created_at}]."""
    import xml.etree.ElementTree as ET
    out = []
    root = ET.fromstring(xml_bytes)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        source = (src_el.text.strip() if src_el is not None and src_el.text else default_source)
        if title:
            out.append({"headline": title, "url": link, "source": source,
                        "created_at": pub[:16], "match": tag})
    return out


_UA = {"User-Agent": "Mozilla/5.0"}


def gather_symbol_news(symbols: list[str], cfg: Config, per_symbol: int = 6,
                       max_symbols: int = 30) -> list[dict]:
    """Pull recent per-ticker headlines from multiple FREE feeds and merge them.

    Sources (all keyless): Google News (which itself aggregates Reuters, Bloomberg, CNBC,
    MarketWatch, WSJ, etc.) and Yahoo Finance. This widens coverage far beyond the single
    Benzinga feed Alpaca returns, so the tone score that feeds conviction sees more signal.
    Fetched in parallel (threaded) with short timeouts; any source failing is skipped."""
    import concurrent.futures as _cf
    import urllib.parse as _up
    syms = [s for s in dict.fromkeys(symbols) if s][:max_symbols]
    if not syms:
        return []

    def _fetch(sym: str) -> list[dict]:
        items: list[dict] = []
        gq = _up.quote(f"{sym} stock")
        feeds = [
            (f"https://news.google.com/rss/search?q={gq}&hl=en-US&gl=US&ceid=US:en", "Google News"),
            (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US", "Yahoo Finance"),
        ]
        for url, dflt in feeds:
            try:
                r = requests.get(url, timeout=6, headers=_UA)
                if r.status_code == 200:
                    items += _parse_rss(r.content, sym, dflt)[:per_symbol]
            except Exception:  # noqa: BLE001
                continue
        seen, out = set(), []
        for it in items:
            h = (it.get("headline") or "").strip()
            key = h.lower()
            if not h or key in seen:
                continue
            seen.add(key)
            out.append({"headline": h, "source": it.get("source", ""),
                        "created_at": it.get("created_at", ""), "url": it.get("url", ""),
                        "symbols": [sym]})
        return out[: per_symbol * 2]

    merged: list[dict] = []
    try:
        with _cf.ThreadPoolExecutor(max_workers=8) as ex:
            for got in ex.map(_fetch, syms):
                merged.extend(got)
    except Exception:  # noqa: BLE001
        for s in syms:
            merged.extend(_fetch(s))
    return merged


def ipo_buzz_news(cfg: Config, limit: int = 16) -> list[dict]:
    """Headlines about notable pre-IPO names (SpaceX, Stripe, …) via Google News RSS.

    Keyless and reliable for keyword search — this is how a private company with no
    ticker still surfaces, since the symbol-keyed market feed can't see it.
    """
    import urllib.parse
    out, seen = [], set()
    for tag, q in _IPO_QUERIES:
        url = ("https://news.google.com/rss/search?q="
               + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en")
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            for it in _parse_rss(r.content, tag):
                h = it["headline"]
                if h in seen:
                    continue
                seen.add(h)
                out.append(it)
        except Exception:  # noqa: BLE001
            continue
        if len(out) >= limit * 2:
            break
    return out[:limit]


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
