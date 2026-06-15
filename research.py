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


# Source quality tiers — a Reuters headline should count more than an anonymous blog.
_SRC_TOP = ("reuters", "bloomberg", "cnbc", "wall street journal", "wsj", "financial times",
            "associated press", "barron", "the new york times", "ft.com", "dow jones")
_SRC_MID = ("marketwatch", "yahoo finance", "benzinga", "forbes", "business insider",
            "seeking alpha", "the motley fool", "investor's business daily", "thestreet",
            "zacks", "investopedia", "cnn", "fortune",
            # decent analytical aggregators — partial credit (0.7)
            "investing.com", "tipranks", "simply wall", "simplywall", "trefis",
            "stockstory", "marketbeat", "morningstar", "kiplinger", "24/7 wall")


def _source_weight(source: str) -> float:
    s = (source or "").lower()
    if any(k in s for k in _SRC_TOP):
        return 1.0
    if any(k in s for k in _SRC_MID):
        return 0.7
    return 0.5


def _recency_weight(created_at: str) -> float:
    """Newer headlines matter more. Returns ~1.0 (today) down to ~0.3 (a month+ old)."""
    import datetime as _dt
    s = (created_at or "").strip()
    if not s:
        return 0.6
    dt = None
    try:
        dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M", "%a, %d %b %Y", "%Y-%m-%d"):
            try:
                dt = _dt.datetime.strptime(s, fmt)
                break
            except Exception:  # noqa: BLE001
                continue
    if dt is None:
        return 0.6
    now = _dt.datetime.now(_dt.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    age = (now - dt).days
    if age <= 2:
        return 1.0
    if age <= 7:
        return 0.8
    if age <= 14:
        return 0.6
    if age <= 30:
        return 0.45
    return 0.3


def news_sentiment(news: list[dict]) -> dict | None:
    """Tone score of the headlines (no API key), now WEIGHTED by source quality and
    recency: a fresh Reuters headline moves the needle more than a stale blog post.
    Returns -1..+1 with a label, plus the (weighted) pos/neg mass and count."""
    if not news:
        return None
    wpos = wneg = 0.0
    for n in news:
        words = (n.get("headline", "") or "").lower().replace(",", " ").replace(".", " ").split()
        p = sum(1 for w in words if w in _POS)
        q = sum(1 for w in words if w in _NEG)
        if not (p or q):
            continue
        w = _source_weight(n.get("source", "")) * _recency_weight(n.get("created_at", ""))
        wpos += w * p
        wneg += w * q
    total = wpos + wneg
    if total == 0:
        return {"label": "Neutral", "score": 0.0, "pos": 0, "neg": 0, "n": len(news), "weighted": True}
    score = round((wpos - wneg) / total, 2)
    label = "Positive" if score >= 0.25 else "Negative" if score <= -0.25 else "Mixed"
    return {"label": label, "score": score, "pos": round(wpos, 1), "neg": round(wneg, 1),
            "n": len(news), "weighted": True}


# ---------------------------------------------------------------- Finnhub
_FH = "https://finnhub.io/api/v1"


def _fh_get(path: str, key: str, params: dict, timeout: int = 12):
    p = dict(params, token=key)
    r = requests.get(f"{_FH}{path}", params=p, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.json()


def finnhub_news(symbol: str, cfg: Config, days: int = 10, limit: int = 6) -> list[dict]:
    """Recent company news from Finnhub (keyed, reliable from servers). Returns the same
    shape as the other feeds. Source names are real outlets (CNBC, MarketWatch, …)."""
    key = cfg.finnhub_api_key
    if not key:
        return []
    import datetime as _dt
    to = _dt.date.today()
    frm = to - _dt.timedelta(days=days)
    try:
        data = _fh_get("/company-news", key, {"symbol": symbol,
                                              "from": frm.isoformat(), "to": to.isoformat()})
    except Exception:  # noqa: BLE001
        return []
    out = []
    for n in (data or []):
        h = (n.get("headline") or "").strip()
        if not h:
            continue
        ts = n.get("datetime")
        created = (_dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
                   if ts else "")
        out.append({"headline": h, "source": (n.get("source") or "Finnhub"),
                    "created_at": created, "url": n.get("url", ""), "symbols": [symbol]})
        if len(out) >= limit:
            break
    return out


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
        # --- security-level micro: earnings momentum + quality (already in this payload, no extra call) ---
        def _num(*keys):
            for k in keys:
                v = met.get(k)
                if v is not None:
                    try:
                        return round(float(v), 1)
                    except Exception:  # noqa: BLE001
                        pass
            return None
        quality = {
            "eps_growth": _num("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"),
            "rev_growth": _num("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
            "net_margin": _num("netProfitMarginTTM", "netProfitMarginAnnual"),
            "roe": _num("roeTTM", "roeRfy"),
            "debt_equity": _num("totalDebt/totalEquityQuarterly", "longTermDebt/equityQuarterly"),
        }
        if any(v is not None for v in quality.values()):
            out["quality"] = quality
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
    try:
        import datetime as _dt
        ud = _fh_get("/stock/upgrade-downgrade", key, {"symbol": symbol}) or []
        cutoff = (_dt.date.today() - _dt.timedelta(days=60))
        recent = []
        for e in ud:
            gt = e.get("gradeTime")
            if not gt:
                continue
            d = _dt.datetime.utcfromtimestamp(int(gt)).date()
            if d < cutoff:
                continue
            recent.append({"firm": e.get("company"), "from": e.get("fromGrade"),
                           "to": e.get("toGrade"), "action": (e.get("action") or "").lower(),
                           "date": d.isoformat()})
        recent.sort(key=lambda x: x["date"], reverse=True)
        if recent:
            n_up = sum(1 for r in recent if r["action"] == "up")
            n_down = sum(1 for r in recent if r["action"] == "down")
            out["analyst_actions"] = {"recent": recent[:6], "n_up": n_up, "n_down": n_down,
                                      "latest": recent[0]}
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
    syms = [s for s in dict.fromkeys(symbols) if s][:max_symbols]
    if not syms:
        return []

    def _fetch(sym: str) -> list[dict]:
        items: list[dict] = []
        # Yahoo Finance RSS is reachable from servers; Google News RSS is blocked from
        # datacenter IPs (returns empty), so we don't waste a request on it.
        feeds = [
            (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US", "Yahoo Finance"),
        ]
        for url, dflt in feeds:
            try:
                r = requests.get(url, timeout=6, headers=_UA)
                if r.status_code == 200:
                    items += _parse_rss(r.content, sym, dflt)[:per_symbol]
            except Exception:  # noqa: BLE001
                continue
        # Finnhub company-news (keyed, reliable from servers) — real outlet names.
        try:
            items += finnhub_news(sym, cfg, days=10, limit=per_symbol)
        except Exception:  # noqa: BLE001
            pass
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
    # --- cross-asset risk gauges (free FRED series) ---
    vix = _fred_latest("VIXCLS", key, limit=6)      # CBOE Volatility Index — the "fear gauge"
    dxy = _fred_latest("DTWEXBGS", key, limit=23)   # broad trade-weighted US dollar (~1mo trend)
    oil = _fred_latest("DCOILWTICO", key)           # WTI crude
    hy = _fred_latest("BAMLH0A0HYM2", key, limit=23)  # ICE BofA US High-Yield OAS — credit spreads
    if vix:
        m["vix"] = round(vix[0], 1)
        if len(vix) >= 6 and vix[5]:
            m["vix_trend"] = "rising" if vix[0] > vix[5] else "falling"
    if dxy:
        m["dxy"] = round(dxy[0], 1)
        if len(dxy) >= 23 and dxy[22]:
            m["dxy_chg_1mo"] = round((dxy[0] / dxy[22] - 1) * 100, 1)  # % change ~1 month
    if oil is not None:
        m["oil"] = round(oil, 1)
    if hy:
        m["hy_oas"] = round(hy[0], 2)               # current high-yield spread (percentage points)
        if len(hy) >= 23 and hy[22]:
            m["hy_oas_chg_1mo"] = round(hy[0] - hy[22], 2)  # change in spread over ~1 month
            m["hy_trend"] = "widening" if hy[0] > hy[22] else "tightening"
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
    # Volatility read from the VIX — a quick risk-on/off gauge.
    if m.get("vix") is not None:
        v = m["vix"]
        trend = f", {m['vix_trend']}" if m.get("vix_trend") else ""
        m["risk_gauge"] = (f"Fearful — VIX {v} elevated{trend}" if v >= 25 else
                           f"Calm — VIX {v} low{trend}" if v < 16 else
                           f"Normal — VIX {v}{trend}")
    return m


_YF_CRUMB: dict = {"crumb": None, "cookies": None}


def _yahoo_crumb():
    """One-time Yahoo cookie+crumb handshake so we can read quoteSummary (short interest)."""
    if _YF_CRUMB["crumb"]:
        return _YF_CRUMB["crumb"], _YF_CRUMB["cookies"]
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        s.get("https://fc.yahoo.com", timeout=10)
        c = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10)
        crumb = (c.text or "").strip()
        if crumb and "<" not in crumb and len(crumb) < 40:
            _YF_CRUMB["crumb"], _YF_CRUMB["cookies"] = crumb, s.cookies
            return crumb, s.cookies
    except Exception:  # noqa: BLE001
        pass
    return None, None


def short_interest(symbols: list[str], cap: int = 25) -> dict:
    """Short interest per symbol from Yahoo key-stats: {sym: {short_pct_float, days_to_cover,
    shares_short}}. Gated by the crumb handshake; capped + threaded; never raises."""
    crumb, cookies = _yahoo_crumb()
    if not crumb:
        return {}

    def _fetch(sym):
        try:
            url = (f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
                   f"?modules=defaultKeyStatistics&crumb={crumb}")
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, cookies=cookies, timeout=10)
            if r.status_code != 200:
                return None
            ks = ((((r.json().get("quoteSummary") or {}).get("result") or [{}])[0]) or {}).get("defaultKeyStatistics") or {}
            spf = (ks.get("shortPercentOfFloat") or {}).get("raw")
            dtc = (ks.get("shortRatio") or {}).get("raw")
            ss = (ks.get("sharesShort") or {}).get("raw")
            if spf is None and dtc is None:
                return None
            return {"short_pct_float": round(spf * 100, 1) if spf is not None else None,
                    "days_to_cover": round(dtc, 1) if dtc is not None else None,
                    "shares_short": ss}
        except Exception:  # noqa: BLE001
            return None

    out, syms = {}, list(symbols)[:cap]
    try:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=6) as ex:
            for sym, res in zip(syms, ex.map(_fetch, syms)):
                if res:
                    out[sym] = res
    except Exception:  # noqa: BLE001
        pass
    return out


_KEY_RELEASES = ("consumer price index", "employment situation", "producer price",
                 "gross domestic product", "personal income", "retail trade")


def econ_calendar(cfg: Config, days_ahead: int = 12) -> list:
    """Upcoming KEY US macro releases (CPI, jobs, PPI, GDP, PCE, retail) from FRED's free release
    calendar. Returns [{date, name}] within the next ``days_ahead`` days, or [] (never raises)."""
    key = cfg.fred_api_key
    if not key:
        return []
    try:
        from datetime import date, timedelta
        today = date.today().isoformat()
        end = (date.today() + timedelta(days=days_ahead)).isoformat()
        r = requests.get("https://api.stlouisfed.org/fred/releases/dates", params={
            "api_key": key, "file_type": "json", "sort_order": "asc",
            "include_release_dates_with_no_data": "true", "realtime_start": today,
            "limit": 200}, timeout=12)
        if r.status_code != 200:
            return []
        out, seen = [], set()
        for rd in r.json().get("release_dates", []):
            d, nm = rd.get("date"), (rd.get("release_name") or "")
            if not d or d < today or d > end:
                continue
            if any(k in nm.lower() for k in _KEY_RELEASES) and (d, nm) not in seen:
                seen.add((d, nm))
                out.append({"date": d, "name": nm})
        return sorted(out, key=lambda x: x["date"])[:8]
    except Exception:  # noqa: BLE001
        return []
