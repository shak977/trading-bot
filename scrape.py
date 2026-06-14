"""Data scrapers for alternative signal inputs. Phase 1: SEC EDGAR insider transactions.

SEC EDGAR is official, keyless and bot-friendly (it just requires a descriptive User-Agent
and ~10 req/sec politeness) — and crucially it works from GitHub's datacenter IP, unlike most
scraped sites. We pull recent **Form 4** filings (insider transactions) per ticker and tally
open-market **purchases (code P)** vs **sales (code S)**: a cluster of insiders buying their
own stock on the open market is a genuine, well-studied bullish tell.

Everything is best-effort and wrapped so a failure never breaks the build. Returns {} on any
problem (no key, network blocked, parse error).
"""
from __future__ import annotations

import re
import time

import requests

_UA = {"User-Agent": "trading-signals-dashboard research (contact: shakeel.lochun@outlook.com)"}
_CIK: dict[str, str] | None = None  # ticker -> zero-padded CIK, cached per process


def _cik_map() -> dict[str, str]:
    global _CIK
    if _CIK is not None:
        return _CIK
    out: dict[str, str] = {}
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_UA, timeout=15)
        if r.status_code == 200:
            for v in (r.json() or {}).values():
                t = (v.get("ticker") or "").upper()
                if t:
                    out[t] = str(v.get("cik_str")).zfill(10)
    except Exception:  # noqa: BLE001
        out = {}
    _CIK = out
    return out


def _parse_form4(xml: str) -> dict:
    """Tally open-market purchases (P) and sales (S) and their share counts in one Form 4."""
    res = {"P": 0, "S": 0, "p_shares": 0.0, "s_shares": 0.0}
    for blk in re.split(r"<nonDerivativeTransaction>", xml)[1:]:
        code_m = re.search(r"<transactionCode>([A-Z])</transactionCode>", blk)
        sh_m = re.search(r"<transactionShares>\s*<value>([\d.]+)</value>", blk)
        if not code_m:
            continue
        code = code_m.group(1)
        shares = float(sh_m.group(1)) if sh_m else 0.0
        if code == "P":
            res["P"] += 1; res["p_shares"] += shares
        elif code == "S":
            res["S"] += 1; res["s_shares"] += shares
    return res


def _for_symbol(sym: str, cik: str, lookback_days: int, max_forms: int) -> dict | None:
    try:
        s = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_UA, timeout=15)
        if s.status_code != 200:
            return None
        rec = ((s.json() or {}).get("filings") or {}).get("recent") or {}
        forms, dates = rec.get("form", []), rec.get("filingDate", [])
        accs, prims = rec.get("accessionNumber", []), rec.get("primaryDocument", [])
        import datetime as _dt
        cutoff = (_dt.date.today() - _dt.timedelta(days=lookback_days)).isoformat()
        idxs = [i for i in range(len(forms)) if forms[i] == "4" and dates[i] >= cutoff]
        if not idxs:
            return {"symbol": sym, "n_filings": 0, "buys": 0, "sells": 0,
                    "buy_shares": 0, "sell_shares": 0, "cluster_buy": False, "last_date": None}
        buys = sells = 0
        bsh = ssh = 0.0
        for i in idxs[:max_forms]:
            a = accs[i].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/{prims[i]}"
            try:
                x = requests.get(url, headers=_UA, timeout=15)
                time.sleep(0.12)  # be polite to SEC
                if x.status_code != 200 or "<" not in x.text:
                    continue
                p = _parse_form4(x.text)
                if p["P"]:
                    buys += 1; bsh += p["p_shares"]
                if p["S"]:
                    sells += 1; ssh += p["s_shares"]
            except Exception:  # noqa: BLE001
                continue
        return {"symbol": sym, "n_filings": len(idxs), "buys": buys, "sells": sells,
                "buy_shares": int(bsh), "sell_shares": int(ssh),
                "cluster_buy": buys >= 2 and bsh > ssh, "last_date": dates[idxs[0]]}
    except Exception:  # noqa: BLE001
        return None


def _st_parse(data: dict) -> dict | None:
    """Tally StockTwits message volume + tagged Bull/Bear sentiment for one symbol."""
    msgs = (data or {}).get("messages") or []
    if not msgs:
        return None
    bull = bear = 0
    for m in msgs:
        basic = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic")
        if basic == "Bullish":
            bull += 1
        elif basic == "Bearish":
            bear += 1
    tagged = bull + bear
    pct = round(bull / tagged * 100) if tagged else None
    lean = None
    if tagged >= 5:
        lean = "bull" if pct >= 65 else "bear" if pct <= 35 else "mixed"
    return {"n": len(msgs), "bull": bull, "bear": bear, "sentiment_pct": pct, "lean": lean}


def stocktwits_buzz(symbols: list[str], proxy: str | None = None, max_symbols: int = 12) -> dict:
    """Retail chatter per ticker from StockTwits' public stream (message volume + Bull/Bear
    sentiment). Routes via the Worker (`{proxy}/?st=SYM`) when given, since StockTwits often
    blocks datacenter IPs; falls back to a direct call. Best-effort; {} on failure."""
    out: dict[str, dict] = {}
    base_direct = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
    for sym in list(dict.fromkeys(symbols))[:max_symbols]:
        data = None
        if proxy:
            try:
                r = requests.get(proxy.rstrip("/") + "/?st=" + sym, timeout=12)
                if r.status_code == 200:
                    data = r.json()
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            try:
                r = requests.get(base_direct.format(sym), headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
                if r.status_code == 200:
                    data = r.json()
            except Exception:  # noqa: BLE001
                data = None
        parsed = _st_parse(data) if data else None
        if parsed:
            out[sym] = parsed
    return out


def insider_activity(symbols: list[str], lookback_days: int = 90,
                     max_symbols: int = 12, max_forms: int = 5) -> dict:
    """Return {symbol: {buys, sells, buy_shares, sell_shares, cluster_buy, last_date, n_filings}}.
    Best-effort; {} if SEC is unreachable. Scans at most `max_symbols` tickers per run."""
    cik = _cik_map()
    if not cik:
        return {}
    out: dict[str, dict] = {}
    for sym in list(dict.fromkeys(symbols))[:max_symbols]:
        c = cik.get(sym.upper())
        if not c:
            continue
        info = _for_symbol(sym, c, lookback_days, max_forms)
        if info:
            out[sym] = info
    return out
