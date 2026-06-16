"""Generate a self-contained HTML dashboard of the latest weekly signals.

Pipeline:
  1. Scan the market (live: Alpaca movers + most-active; synthetic: demo list).
  2. Run the MA/RSI strategy on each, compute a relative-volume flow proxy.
  3. Pull recent news for the flagged (BUY/SELL) names.
  4. Write dashboard.html (self-contained) and signals.json.

Data source:
  - Real Alpaca data when ALPACA_API_KEY/SECRET are set.
  - Deterministic synthetic data otherwise, clearly labelled SYNTHETIC.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import market
import scanner
from config import CONFIG


def _mode() -> str:
    if CONFIG.api_key and CONFIG.secret_key:
        return "PAPER" if CONFIG.paper else "LIVE"
    return "SYNTHETIC"


def _synthetic_news(symbols: list[str]) -> list[dict]:
    templates = [
        "{s} sees unusual options activity into the close",
        "Analysts revise {s} price target after volume spike",
        "{s} momentum builds as moving averages cross",
        "{s} among most-active names this session",
    ]
    out = []
    for i, s in enumerate(symbols[:6]):
        out.append({
            "headline": templates[i % len(templates)].format(s=s),
            "source": "SyntheticWire", "created_at": "(demo)",
            "url": "", "symbols": [s],
        })
    return out


def _market_regime(rows: list[dict]) -> dict | None:
    """Read the overall tape: breadth (% above trend), average momentum, # buys."""
    if not rows:
        return None
    above = sum(1 for r in rows if (r.get("context", {}).get("vs_slow_ma_pct") or 0) > 0)
    breadth = round(above / len(rows) * 100)
    rsis = [r["rsi"] for r in rows if r.get("rsi") is not None]
    avg_rsi = round(sum(rsis) / len(rsis), 1) if rsis else None
    buys = sum(1 for r in rows if r["action"] == "BUY")
    if breadth >= 60 and (avg_rsi or 0) >= 50:
        label, note = "Risk-on", "Most stocks are trending up — a friendlier backdrop for buying."
    elif breadth <= 40 or (avg_rsi or 100) < 45:
        label, note = "Risk-off", "Most stocks are below trend — be choosier; long signals are fighting the tape."
    else:
        label, note = "Neutral", "Mixed tape — no strong market-wide direction; pick spots carefully."
    return {"label": label, "breadth": breadth, "avg_rsi": avg_rsi,
            "buys": buys, "total": len(rows), "note": note}


def _concentration(shown: list[dict]) -> dict | None:
    """Flag when fresh entries pile into one sector — they're often the same macro bet in
    disguise, so '5 buys' can really be one position's worth of risk. Returns the dominant
    sector and its share when a single sector holds >=50% of fresh BUY (or SHORT) signals."""
    from collections import Counter
    out = None
    for action, word in (("BUY", "buys"), ("SHORT", "shorts")):
        fresh = [s for s in shown if s.get("action") == action]
        if len(fresh) < 3:
            continue
        by = Counter(s.get("sector") or "Other" for s in fresh)
        sector, n = by.most_common(1)[0]
        frac = n / len(fresh)
        if frac >= 0.5:
            cand = {"action": action, "word": word, "sector": sector, "n": n,
                    "total": len(fresh), "pct": round(frac * 100),
                    "symbols": [s["symbol"] for s in fresh if (s.get("sector") or "Other") == sector]}
            # keep the most concentrated of the two sides
            if out is None or cand["pct"] > out["pct"]:
                out = cand
    return out


def _sector_strength(rows: list[dict]) -> list[dict]:
    """Rank sectors by how many of their stocks are above their trend line."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[scanner.sector_of(r["symbol"])].append(r)
    out = []
    for sec, rs in by.items():
        up = sum(1 for r in rs if (r.get("context", {}).get("vs_slow_ma_pct") or 0) > 0)
        out.append({"sector": sec, "count": len(rs), "pct_up": round(up / len(rs) * 100)})
    out.sort(key=lambda x: -x["pct_up"])
    return out


def _compute_changes(shown: list, sectors: list, news_ideas: list, today: str, live: bool) -> list:
    """Worker: surface the MEANINGFUL diffs vs the previous build — new High-conviction calls,
    direction flips, conviction upgrades, sector flips, fresh catalysts. Persists prev_state.json.
    Live-only; never raises. Returns a short list of human-readable change strings."""
    if not live:
        return []
    _rank = {"Low": 0, "Medium": 1, "High": 2}
    path = "prev_state.json"
    try:
        with open(path) as f:
            prev = json.load(f)
    except Exception:  # noqa: BLE001
        prev = {}
    prev_sig, prev_sec = prev.get("signals", {}), prev.get("sectors", {})
    prev_cat = set(prev.get("catalysts", []))
    actionable = ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")
    cur_sig = {}
    for s in shown:
        c = s.get("conviction") or {}
        cur_sig[s.get("symbol")] = {"action": s.get("action"), "label": c.get("label"),
                                    "score": c.get("score_pct"), "dir": s.get("direction")}
    cur_sec = {x["sector"]: x.get("pct_up") for x in (sectors or [])}
    cur_cat = [(i.get("ticker"), i.get("direction")) for i in (news_ideas or []) if i.get("confidence") == "high"]

    def _persist():
        try:
            with open(path, "w") as f:
                json.dump({"signals": cur_sig, "sectors": cur_sec,
                           "catalysts": [t for t, _ in cur_cat if t], "generated_at": today}, f, indent=2)
        except Exception:  # noqa: BLE001
            pass

    if not prev_sig:                       # first run: seed state, don't flood with "new"
        _persist()
        return []
    changes = []
    for sym, c in cur_sig.items():
        p = prev_sig.get(sym)
        if c["label"] == "High" and c["action"] in ("BUY", "SHORT") and (not p or p.get("label") != "High"):
            changes.append(f"🆕 {sym} → {c['action']} (High{', ' + str(c['score']) + '%' if c['score'] else ''})")
        elif p and p.get("dir") in ("LONG", "SHORT") and c["dir"] in ("LONG", "SHORT") \
                and p["dir"] != c["dir"] and c["action"] in actionable:
            changes.append(f"🔄 {sym} flipped to {c['action']}")
        elif p and p.get("label") and c["label"] and _rank.get(c["label"], 0) > _rank.get(p["label"], 0) \
                and c["action"] in actionable:
            changes.append(f"⬆ {sym} conviction now {c['label']} (was {p['label']})")
    for sec, pct in cur_sec.items():
        pp = prev_sec.get(sec)
        if pp is not None and pct is not None:
            if pp < 60 <= pct:
                changes.append(f"📈 {sec} sector turned strong ({pct}% above trend)")
            elif pp > 40 >= pct:
                changes.append(f"📉 {sec} sector turned weak ({pct}% above trend)")
    for tk, d in cur_cat:
        if tk and tk not in prev_cat:
            changes.append(f"🗞 New catalyst: {tk} ({d})")
    _persist()
    return changes[:12]


def build_snapshot() -> dict:
    mode = _mode()
    live = mode != "SYNTHETIC"

    # Pin any symbol we already alerted today so the LATEST dashboard always contains every
    # name you were notified about (alerts and dashboard stay in line as the universe rotates).
    _today0 = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pin = set()
    if live:
        try:
            import notify as _nf0
            pin = _nf0.alerted_today(_today0)
        except Exception:  # noqa: BLE001
            pin = set()
        # News-idea candidates: pull names the LLM flagged from recent news INTO the scan so they
        # get a real technical read. They only surface as signals if the engine confirms them —
        # news widens the net, technicals still decide.
        if getattr(CONFIG, "news_idea_candidates", False):
            try:
                import json as _json
                from datetime import timedelta as _td
                with open("news_candidates.json") as _f:
                    _nc = _json.load(_f)
                _cut = (datetime.now(timezone.utc).date() - _td(days=2)).isoformat()
                pin |= {k for k, v in _nc.items() if v >= _cut}
            except Exception:  # noqa: BLE001
                pass

    rows = scanner.scan(CONFIG, live=live, pin=pin)
    for _r in rows:
        _r["alerted"] = _r["symbol"] in pin

    # split chart data out of each row for compactness
    charts = {r["symbol"]: r.pop("chart") for r in rows}
    # lookup of the full analysis row by symbol (used to make momentum rows clickable)
    rows_by_sym = {r["symbol"]: r for r in rows}

    # Dual-momentum leaderboard over the whole scanned universe (best-validated strategy).
    momentum_rows = _momentum_rank(charts)
    # Drop leaders whose scan price disagrees >15% with the consolidated Yahoo quote —
    # the same bad-feed-price guard we apply to signals (keeps MU-at-$981 junk off the list).
    if live and momentum_rows:
        try:
            import research as _r
            mq = _r.yahoo_quotes([m["symbol"] for m in momentum_rows])
            momentum_rows = [m for m in momentum_rows
                             if not (mq.get(m["symbol"]) and m.get("price")
                                     and abs(m["price"] / mq[m["symbol"]]["price"] - 1) > 0.15)]
        except Exception:  # noqa: BLE001
            pass
    # New-vs-holdover: mark which leaders just entered the list vs the previous run.
    try:
        import json as _json
        import datetime as _dt
        _mpath = "momentum_history.json"
        try:
            with open(_mpath) as _f:
                _prev = set((_json.load(_f) or {}).get("symbols", []))
        except Exception:  # noqa: BLE001
            _prev = set()
        for _m in momentum_rows:
            _m["is_new"] = bool(_prev) and _m["symbol"] not in _prev
        if live and momentum_rows:
            try:
                with open(_mpath, "w") as _f:
                    _json.dump({"as_of": _dt.date.today().isoformat(),
                                "symbols": [_m["symbol"] for _m in momentum_rows]}, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Market regime + sector strength from the FULL scanned set (the "tape").
    regime = _market_regime(rows)
    sectors = _sector_strength(rows)

    shown = rows[: CONFIG.show_top]
    # force-include any alerted-today name that ranking truncated, so it's never missing
    if pin:
        _ss = {r["symbol"] for r in shown}
        for _r in rows:
            if _r.get("alerted") and _r["symbol"] not in _ss:
                shown.append(_r)
                _ss.add(_r["symbol"])
    shown_syms = [r["symbol"] for r in shown]

    # Regime filter: don't initiate against a hostile tape. In Risk-off, demote fresh BUYs to
    # the WATCH tier; symmetrically, in Risk-on demote fresh SHORTs — the tool shouldn't fight
    # a strong market-wide direction with a brand-new entry in the opposite direction.
    if CONFIG.regime_block_buys and regime:
        _lbl = regime.get("label")
        for r in shown:
            if _lbl == "Risk-off" and r.get("action") == "BUY":
                r["action"] = "WATCH LONG"
                r["regime_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, "🛑 Market regime is Risk-off — standing down on new buys; this setup is "
                       "shown as Watch, not a fresh entry.")
            elif _lbl == "Risk-on" and r.get("action") == "SHORT":
                r["action"] = "WATCH SHORT"
                r["regime_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, "🛑 Market regime is Risk-on — standing down on new shorts; this setup is "
                       "shown as Watch, not a fresh entry against a rising market.")

    # Pull news once for everything shown, from MULTIPLE feeds, then bucket per ticker.
    if live:
        import research as _rn
        news = []
        try:                       # Alpaca/Benzinga (symbol-keyed)
            news = market.get_news(shown_syms, CONFIG,
                                   limit=CONFIG.news_per_symbol * max(len(shown_syms), 1))
        except Exception:  # noqa: BLE001
            news = []
        try:                       # + free feeds: Google News (Reuters/Bloomberg/CNBC/…) + Yahoo Finance
            news += _rn.gather_symbol_news(shown_syms, CONFIG,
                                           per_symbol=6, max_symbols=CONFIG.research_top)
        except Exception:  # noqa: BLE001
            pass
        # global dedupe by headline, preserving order (Benzinga → Google → Yahoo)
        _seen, _merged = set(), []
        for _n in news:
            _h = (_n.get("headline") or "").strip().lower()
            if not _h or _h in _seen:
                continue
            _seen.add(_h)
            _merged.append(_n)
        news = _merged or [{"headline": "(no recent news found)", "source": "",
                            "created_at": "", "url": "", "symbols": []}]
        # Interleave by source so the Market news tab leads with a MIX, not 50 Benzinga first.
        from collections import OrderedDict as _OD, deque as _dq
        _g = _OD()
        for _n in news:
            _g.setdefault((_n.get("source") or "").lower(), _dq()).append(_n)
        _inter = []
        while any(_g.values()):
            for _q in _g.values():
                if _q:
                    _inter.append(_q.popleft())
        news = _inter
    else:
        news = _synthetic_news(shown_syms)

    # Company name + exchange for each shown ticker.
    _demo_names = {
        "AAPL": ("Apple Inc.", "NASDAQ"), "MSFT": ("Microsoft Corp.", "NASDAQ"),
        "NVDA": ("NVIDIA Corp.", "NASDAQ"), "AMZN": ("Amazon.com Inc.", "NASDAQ"),
        "TSLA": ("Tesla Inc.", "NASDAQ"), "META": ("Meta Platforms Inc.", "NASDAQ"),
        "GOOGL": ("Alphabet Inc.", "NASDAQ"), "AMD": ("Adv. Micro Devices", "NASDAQ"),
        "SPY": ("SPDR S&P 500 ETF", "NYSE Arca"), "QQQ": ("Invesco QQQ Trust", "NASDAQ"),
    }
    for r in shown:
        if live:
            a = market.get_asset(r["symbol"], CONFIG)
            r["name"], r["exchange"] = a.get("name", ""), a.get("exchange", "")
        else:
            nm, ex = _demo_names.get(r["symbol"], (r["symbol"] + " (demo)", "DEMO"))
            r["name"], r["exchange"] = nm, ex
        r["sector"] = scanner.sector_of(r["symbol"])

    # Attach each ticker's own headlines to its row (for the click-through detail),
    # and fold a plain-English news line into the reasoning so it's news-aware.
    def _interleave_by_source(items):
        """Round-robin across sources so the top headlines show a MIX (not 10 Benzinga first)."""
        from collections import OrderedDict, deque
        groups = OrderedDict()
        for it in items:
            groups.setdefault(it.get("source", ""), deque()).append(it)
        out = []
        while any(groups.values()):
            for q in groups.values():
                if q:
                    out.append(q.popleft())
        return out
    for r in shown:
        # keep more headlines per ticker now that several feeds contribute — richer tone signal,
        # interleaved by source so the modal shows variety, not one outlet stacked on top.
        _matched = [n for n in news if r["symbol"] in (n.get("symbols") or [])]
        r["news"] = _interleave_by_source(_matched)[: max(CONFIG.news_per_symbol, 12)]
        # Catalyst: flag when fresh (<~2 day) news coincides with the signal.
        try:
            import research as _rr
            _fresh = [n for n in r["news"] if _rr._recency_weight(n.get("created_at", "")) >= 0.9]
            if _fresh:
                r["catalyst"] = {"headline": _fresh[0]["headline"],
                                 "source": _fresh[0].get("source", ""), "n": len(_fresh)}
        except Exception:  # noqa: BLE001
            pass
        if r["news"]:
            top = r["news"][0]["headline"]
            n = len(r["news"])
            phrase = "1 recent story mentions" if n == 1 else f"{n} recent stories mention"
            r.setdefault("reasons", []).append(
                f"📰 In the news: {phrase} {r['symbol']}. "
                f"Latest headline — “{top}”. Worth a read for what's driving it."
            )
        else:
            r.setdefault("reasons", []).append(
                f"📰 No recent news found for {r['symbol']} — the move looks technical (chart-driven), "
                f"not headline-driven."
            )

    # --- Research layer: news tone (free), analyst/fundamentals (Finnhub) ---
    import research
    # Consolidated (full-market) price + previous close so the cards match Google/Yahoo,
    # since the scan runs on Alpaca's IEX feed whose close can drift a few cents.
    price_drops = []
    if live:
        try:
            yq = research.yahoo_quotes([r["symbol"] for r in shown])
            for r in shown:
                q = yq.get(r["symbol"])
                if q:
                    r["quote_price"] = q["price"]
                    r["prev_close"] = q.get("prev_close")
            # Cross-check: drop any signal whose scan (Alpaca IEX) price disagrees
            # with the consolidated quote by >15% — that means the feed gave a bad
            # price, so the whole signal (indicators, plan, levels) is untrustworthy.
            bad = [r for r in shown
                   if r.get("quote_price") and r.get("price")
                   and abs(r["price"] / r["quote_price"] - 1) > 0.15]
            if bad:
                price_drops = [f"{r['symbol']} (scan ${r['price']:,.2f} vs ${r['quote_price']:,.2f})"
                               for r in bad]
                drop = {r["symbol"] for r in bad}
                shown = [r for r in shown if r["symbol"] not in drop]
                shown_syms = [r["symbol"] for r in shown]
            # rebase kept signals' price + plan levels onto the consolidated quote so the
            # displayed price and the entry/stop/target are internally consistent.
            for r in shown:
                q = r.get("quote_price")
                if not q or not r.get("price"):
                    continue
                ratio = q / r["price"]
                if abs(ratio - 1) > 0.003:
                    for k in ("stop", "target"):
                        if r.get(k) is not None:
                            r[k] = round(r[k] * ratio, 2)
                    p = r.get("plan") or {}
                    for k in ("entry", "stop", "target"):
                        if p.get(k) is not None:
                            p[k] = round(p[k] * ratio, 2)
                r["price"] = q
        except Exception:  # noqa: BLE001
            pass
    for r in shown:
        r["sentiment"] = research.news_sentiment(r.get("news"))
    # TradingView multi-timeframe TA rating — independent cross-check (keyless, unofficial).
    tv_map = {}
    if live:
        try:
            import tradingview as _tv
            tv_map = _tv.ratings([r["symbol"] for r in shown[: CONFIG.research_top]],
                                 proxy=CONFIG.live_quotes_url or None)
        except Exception:  # noqa: BLE001
            tv_map = {}
    for r in shown:
        if tv_map.get(r["symbol"]):
            r["tv"] = tv_map[r["symbol"]]
    fundamentals = {}
    if live and CONFIG.finnhub_api_key:
        try:
            fundamentals = research.finnhub_for_symbols(
                [r["symbol"] for r in shown[: CONFIG.research_top]], CONFIG)
        except Exception:  # noqa: BLE001
            fundamentals = {}
    # Scraped alt-data for the actionable names: SEC EDGAR insider filings (keyless) +
    # StockTwits retail buzz (routed via the Worker, which has a non-datacenter egress).
    insiders, buzz = {}, {}
    if live:
        _act = [r["symbol"] for r in shown
                if r["action"] in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")]
        try:
            import scrape as _scrape
            insiders = _scrape.insider_activity(_act)
        except Exception:  # noqa: BLE001
            insiders = {}
        try:
            import scrape as _scrape
            buzz = _scrape.stocktwits_buzz(_act, proxy=CONFIG.live_quotes_url or None)
        except Exception:  # noqa: BLE001
            buzz = {}

    # News-driven ideas: one LLM pass over recent headlines -> actionable single-stock reads.
    # Feeds a conviction nudge on scanned names + a standalone list (incl. names not in the scan).
    news_ideas = []
    if live:
        try:
            import llm
            news_ideas = llm.news_ideas(news, CONFIG, universe={r["symbol"] for r in shown})
        except Exception:  # noqa: BLE001
            news_ideas = []
    _idea_map = {i["ticker"]: i for i in news_ideas}
    # Persist material (high/medium-confidence) news-idea tickers so the NEXT build pulls them into
    # the scan for a technical read. Pruned after ~5 days. Live-only; never breaks the build.
    if live and getattr(CONFIG, "news_idea_candidates", False):
        try:
            import json as _json
            from datetime import timedelta as _td
            try:
                with open("news_candidates.json") as _f:
                    _ncw = _json.load(_f)
            except Exception:  # noqa: BLE001
                _ncw = {}
            for _i in news_ideas:
                _tk = (_i.get("ticker") or "").upper().strip().lstrip("$")
                if _tk and _i.get("confidence") in ("high", "medium"):
                    _ncw[_tk] = _today0
            _old = (datetime.now(timezone.utc).date() - _td(days=5)).isoformat()
            _ncw = {k: v for k, v in _ncw.items() if v >= _old}
            with open("news_candidates.json", "w") as _f:
                _json.dump(_ncw, _f, indent=2)
        except Exception:  # noqa: BLE001
            pass

    # --- Intraday layer (gated + graceful): run the SAME engine on intraday bars over the same
    # names. Powers the Intraday tab AND a lower-timeframe confirmation that nudges daily
    # conviction. Any failure leaves it empty — the daily build is never affected.
    intraday_shown: list = []
    intraday_by_sym: dict = {}
    intraday_track: dict = {}
    if live and getattr(CONFIG, "intraday_enabled", False):
        try:
            from dataclasses import replace as _replace
            _icfg = _replace(CONFIG, timeframe=CONFIG.intraday_timeframe,
                             lookback_days=CONFIG.intraday_lookback_days,
                             fast_ma=CONFIG.intraday_fast_ma, slow_ma=CONFIG.intraday_slow_ma)
            _irows = scanner.scan(_icfg, live=live, universe=[r["symbol"] for r in rows])
            for _ir in _irows:
                _ir.pop("chart", None)
                _ir["intraday"] = True
            intraday_by_sym = {_ir["symbol"]: _ir for _ir in _irows}
            _irows.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
            intraday_shown = _irows[: CONFIG.intraday_show_top]
            # Shadow track-record for the intraday layer (NO orders): grade intraday calls against
            # intraday bars, in a SEPARATE log so it never mixes with the daily record/paper book.
            try:
                import tracker as _tracker
                import pandas as _pd
                intraday_track = _tracker.run(
                    list(intraday_by_sym.values()), _icfg, live, _today0,
                    path="track_record_intraday.json", intraday=True,
                    now_ts=_pd.Timestamp.utcnow().tz_localize(None), hold_days=3)
            except Exception as _itexc:  # noqa: BLE001
                intraday_track = {}
                print("INTRADAY TRACK: skipped —", _itexc)
            print(f"INTRADAY: {len(intraday_shown)} signals on {CONFIG.intraday_timeframe} bars")
        except Exception as _iexc:  # noqa: BLE001 - never break the daily build
            intraday_shown, intraday_by_sym, intraday_track = [], {}, {}
            print("INTRADAY: skipped —", _iexc)

    # Short interest (squeeze risk) — Yahoo key-stats for the actionable names. Gated + fail-silent.
    short_int = {}
    if live:
        try:
            _si_syms = [r["symbol"] for r in shown
                        if r.get("action") in ("BUY", "SHORT", "WATCH SHORT", "HOLD SHORT", "WATCH LONG", "HOLD LONG")]
            short_int = research.short_interest(_si_syms)
        except Exception:  # noqa: BLE001
            short_int = {}

    # Retail / social attention (ApeWisdom: Reddit + WSB mentions). One keyless call returns the
    # top-attention names globally; we match by symbol. The noisiest input — only a light nudge.
    retail_map = {}
    if live:
        try:
            import scrape as _scrape
            retail_map = _scrape.retail_attention()
        except Exception:  # noqa: BLE001
            retail_map = {}

    _ACTIONABLE = ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")
    _sector_pct = {s["sector"]: s.get("pct_up") for s in (sectors or [])}   # sector momentum map
    for r in shown:
        r["fundamentals"] = fundamentals.get(r["symbol"])
        r["insider"] = insiders.get(r["symbol"])
        r["buzz"] = buzz.get(r["symbol"])
        r["news_idea"] = _idea_map.get(r["symbol"])
        r["short_interest"] = short_int.get(r["symbol"])
        r["retail"] = retail_map.get(r["symbol"])
        # Lower-timeframe confirmation: does the intraday signal agree with this daily trade?
        _isig = intraday_by_sym.get(r["symbol"])
        if _isig and _isig.get("action") in _ACTIONABLE:
            r["intraday_confirm"] = "agree" if _isig.get("direction") == r.get("direction") else "disagree"
        else:
            r["intraday_confirm"] = "none"
        # Re-score conviction + desk read now that research (news/sector/intraday) is in hand.
        scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"), fundamentals=r.get("fundamentals"),
                        tv=r.get("tv"), regime=regime, insider=r.get("insider"), buzz=r.get("buzz"),
                        news_idea=r.get("news_idea"), intraday=_isig,
                        sector_pct=_sector_pct.get(r.get("sector")), short_interest=r.get("short_interest"),
                        retail=r.get("retail"))

    # LLM structured news scoring: convert recent per-stock headlines into named structured scores
    # (guidance / margin pressure / demand / regulatory risk / …) for the top actionable names only
    # — one cheap batched call. The LLM converts text→numbers; it never decides the trade.
    nlp_scores = {}
    if live and CONFIG.llm_enabled:
        try:
            import llm as _llm_nlp
            nlp_scores = _llm_nlp.structured_scores(shown, CONFIG)
            for r in shown:
                if r["symbol"] in nlp_scores:
                    r["nlp"] = nlp_scores[r["symbol"]]
        except Exception:  # noqa: BLE001
            nlp_scores = {}

    # First-seen dates per signal — powers the "Newest" sort + a date chip on each card.
    # Persisted across runs (like the tracker); a symbol that leaves and returns gets a fresh date.
    try:
        import json as _json2, datetime as _dt2
        _spath = "signals_seen.json"
        try:
            with open(_spath) as _f:
                _seen = (_json2.load(_f) or {}).get("seen", {})
        except Exception:  # noqa: BLE001
            _seen = {}
        _today = _dt2.date.today().isoformat()
        _cur = {}
        for r in shown:
            k = f"{r['symbol']}:{r.get('direction', 'LONG')}"
            first = _seen.get(k) or _today
            _cur[k] = first
            r["first_seen"] = first
            try:
                r["days_old"] = (_dt2.date.fromisoformat(_today) - _dt2.date.fromisoformat(first)).days
            except Exception:  # noqa: BLE001
                r["days_old"] = 0
            r["is_fresh"] = (first == _today)
        if live:  # only the live Action persists (synthetic/dev runs never write)
            try:
                with open(_spath, "w") as _f:
                    _json2.dump({"as_of": _today, "seen": _cur}, _f, indent=2)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Ray Dalio All Weather allocation + backtest vs SPY (keyless Yahoo history).
    try:
        import allweather as _aw
        all_weather = _aw.build(live)
    except Exception:  # noqa: BLE001
        all_weather = None

    # Survivorship-bias-FREE momentum backtest (fixed ETF universe) — the honest performance
    # read for the momentum strategy, vs the survivorship-biased single-stock ranking.
    try:
        import momentum_lab as _ml
        momentum_bt = _ml.build(live)
    except Exception:  # noqa: BLE001
        momentum_bt = None

    # Walk-forward / out-of-sample validation (gated) — the honest "does the edge survive on
    # data it wasn't tuned on?" read. Small basket, read-only; any failure -> None.
    try:
        import walkforward as _wf
        walk_fwd = _wf.validate(CONFIG, live, n_folds=CONFIG.walkforward_folds)
    except Exception:  # noqa: BLE001
        walk_fwd = None

    # Macro backdrop (FRED) — once per run.
    macro = None
    if live and CONFIG.fred_api_key:
        try:
            macro = research.fred_macro(CONFIG)
        except Exception:  # noqa: BLE001
            macro = None

    # Macro regime → exposure: blend the macro backdrop + equity breadth into a posture and an
    # exposure multiplier that scales new-position sizing. Macro controls EXPOSURE, never a direct
    # buy/sell. Gated + fail-silent; None when disabled or no data.
    try:
        import macro_regime as _macro_regime
        macro_posture = _macro_regime.assess(macro, regime, CONFIG)
    except Exception:  # noqa: BLE001
        macro_posture = None
    _exposure_mult = (macro_posture or {}).get("exposure_mult", 1.0)

    # Regime-specific weighting: in a defensive / high-volatility regime, RAISE the conviction bar
    # a fresh entry must clear — so the bot makes fewer, higher-quality trades when the backdrop is
    # hostile. With-tape setups below the regime threshold are demoted to the Watch tier.
    _regime_threshold = 0
    if getattr(CONFIG, "regime_weighting_enabled", True) and macro_posture:
        _lab = macro_posture.get("label")
        _tagset = {t.get("tag") for t in macro_posture.get("tags", [])}
        _regime_threshold = {"Risk-on": 50, "Neutral": 55, "Risk-off": 62}.get(_lab, 50)
        if "High-volatility" in _tagset:
            _regime_threshold += 6
        macro_posture["entry_threshold"] = _regime_threshold
        for r in shown:
            if r.get("action") not in ("BUY", "SHORT"):
                continue
            sc = (r.get("conviction") or {}).get("score_pct") or 0
            if sc < _regime_threshold:
                r["action"] = "WATCH LONG" if r.get("direction") == "LONG" else "WATCH SHORT"
                r["regime_demoted"] = True
                ctx = r.setdefault("context", {})
                r.setdefault("reasons", []).insert(
                    0, f"⚖️ {_lab} regime raises the bar to {_regime_threshold}% — this {sc}% setup is "
                       "demoted to Watch (fewer, higher-quality entries when the backdrop is tough).")

    # Adaptive asset ranking: score actionable names for CAPITAL ALLOCATION (quality + vol-adj
    # reward + macro fit + liquidity + momentum). Mutates rows (rank_score/rank) + returns a list.
    try:
        import rank as _rank
        ranked = _rank.rank_rows(shown, macro_posture, CONFIG)
    except Exception:  # noqa: BLE001
        ranked = []

    # IPO watch: upcoming-IPO calendar + general news mentioning pre-IPO names
    # (e.g. SpaceX). Private names have no ticker, so this is the only way they surface.
    ipos, ipo_news = [], []
    if live:
        try:
            ipo_news = research.ipo_buzz_news(CONFIG)   # keyless (Google News RSS)
        except Exception:  # noqa: BLE001
            ipo_news = []
        if CONFIG.finnhub_api_key:
            try:
                ipos = research.ipo_calendar(CONFIG)
            except Exception:  # noqa: BLE001
                ipos = []

    # Optional AI analyst note — for every High-conviction actionable setup (BUY/SHORT/HOLD).
    # No hard top-N cap (the High-conviction floor already limits the count); a generous safety
    # ceiling guards against a pathological run firing dozens of calls.
    llm_status = {"enabled": bool(CONFIG.llm_enabled)}
    if CONFIG.llm_enabled:
        import llm
        _ai_picks = [r for r in shown
                     if (r.get("conviction") or {}).get("label") == "High"
                     and r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")]
        _ai_picks.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
        llm_status["candidates"] = len(_ai_picks)
        _gen = 0
        for r in _ai_picks[:40]:  # effectively all High-conviction actionable; 40 = safety ceiling
            note = llm.analyst_note(r, CONFIG, regime=regime, macro=macro)
            if note:
                r["ai_read"] = note
                _gen += 1
        llm_status["generated"] = _gen
        # If nothing was produced but candidates existed, probe once to surface WHY.
        if _gen == 0 and _ai_picks:
            try:
                llm_status["probe"] = llm.diagnose(CONFIG)
            except Exception as exc:  # noqa: BLE001
                llm_status["probe"] = {"ok": False, "error": str(exc)[:200]}

    # AI market brief (worker): a plain-English "what's happening / what to watch" summary at the
    # top of the dashboard, built only from data we already have. Live + LLM only; never breaks.
    market_brief = None
    if live and CONFIG.llm_enabled:
        try:
            market_brief = llm.market_brief(regime, shown, sectors, momentum_rows, news_ideas, macro, CONFIG)
        except Exception:  # noqa: BLE001
            market_brief = None

    # What-changed worker: meaningful diffs vs the previous build (new High calls, flips, sector shifts).
    try:
        changes = _compute_changes(shown, sectors, news_ideas, _today0, live)
    except Exception:  # noqa: BLE001
        changes = []

    # Event calendar (worker): earnings this week (from fundamentals already fetched) + key macro
    # releases (FRED). Event-risk awareness; never breaks the build.
    calendar = {"earnings": [], "econ": []}
    try:
        _ew = [{"symbol": r["symbol"], "days": (r.get("fundamentals") or {}).get("earnings_days"),
                "date": (r.get("fundamentals") or {}).get("earnings_date")}
               for r in shown
               if (r.get("fundamentals") or {}).get("earnings_days") is not None
               and 0 <= (r.get("fundamentals") or {}).get("earnings_days") <= 7]
        calendar["earnings"] = sorted(_ew, key=lambda x: x["days"])[:12]
        if live:
            calendar["econ"] = research.econ_calendar(CONFIG)
    except Exception:  # noqa: BLE001
        calendar = {"earnings": [], "econ": []}

    # S&P 500 benchmark (SPY) for chart overlay.
    benchmark = None
    try:
        from data import get_bars, synthetic_bars
        bdf = get_bars("SPY", CONFIG) if live else synthetic_bars("SPY", n=CONFIG.lookback_days)
        if bdf is not None and len(bdf):
            bdf = bdf.tail(300)
            benchmark = {
                "symbol": "SPY", "name": "S&P 500",
                "t": [int(pd.Timestamp(d).timestamp() * 1000) for d in bdf.index],
                "close": [round(float(x), 2) for x in bdf["close"]],
            }
    except Exception:  # noqa: BLE001
        benchmark = None

    # Track record: log new BUYs and grade past calls against real prices.
    import tracker
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        track = tracker.run(shown, CONFIG, live, today, regime=macro_posture)
    except Exception:  # noqa: BLE001
        track = None

    # Meta-signal model: a second-opinion verdict (accept/reduce/delay/reject) on every actionable
    # candidate, from regime fit + liquidity + conflicts + how this regime has paid off historically.
    try:
        import meta as _meta
        for r in shown:
            if r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
                r["meta"] = _meta.evaluate(r, macro_posture=macro_posture, track=track, cfg=CONFIG)
    except Exception:  # noqa: BLE001
        pass

    # Structured signal output: one tidy record per actionable trade (confidence, expected return
    # range, hold, risk/liquidity/uncertainty scores, size rec, kill conditions, meta verdict).
    structured = []
    try:
        import structured as _structured
        for r in shown:
            if r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
                _so = _structured.build(r, macro_posture, CONFIG)
                if _so:
                    r["structured"] = _so
                    structured.append(_so)
    except Exception:  # noqa: BLE001
        structured = []

    # No-trade intelligence layer: one unified "should we be trading right now?" read (macro event,
    # abnormal vol, deteriorating performance, drawdown). Computed before paper so it can gate entries.
    try:
        import notrade as _notrade
        notrade_gate = _notrade.market_gate(CONFIG, macro_posture=macro_posture, macro=macro,
                                            calendar=calendar, track=track, risk=None, today=today)
    except Exception:  # noqa: BLE001
        notrade_gate = {"block_new": False, "reasons": [], "cautions": [], "checks": []}

    # Optional REAL paper-trading record (opt-in via PAPER_TRADE) — submits bracket orders for
    # fresh High-conviction signals and reads the live paper account. Disabled -> None.
    try:
        import paper as _paper
        paper_acct = _paper.run(shown, CONFIG, today, exposure_mult=_exposure_mult,
                                regime=macro_posture, notrade_block=notrade_gate.get("block_new", False))
    except Exception:  # noqa: BLE001
        paper_acct = None

    # Re-evaluate the no-trade gate WITH the risk-engine state (from paper) so the panel unifies it.
    try:
        import notrade as _notrade
        notrade_gate = _notrade.market_gate(CONFIG, macro_posture=macro_posture, macro=macro,
                                            calendar=calendar, track=track,
                                            risk=(paper_acct or {}).get("risk"), today=today)
    except Exception:  # noqa: BLE001
        pass

    # Pairs / mean-reversion diversifier (gated). Market-neutral spread bets on related names —
    # leans in when the tape is trendless. Any failure -> empty list; never breaks the build.
    try:
        import pairs as _pairs
        pairs_data = _pairs.scan(CONFIG, live=live, regime=regime, macro_posture=macro_posture)
    except Exception:  # noqa: BLE001
        pairs_data = {"pairs": [], "regime_fit": False, "note": ""}

    # Alerts: ping configured channels when a NEW high-conviction signal appears (deduped).
    try:
        import notify as _notify
        alerts = _notify.run(shown, today)
    except Exception:  # noqa: BLE001
        alerts = None
    # Pairs alerts: ping when a spread stretches to its ±2σ entry band (deduped, once/day/pair).
    try:
        import notify as _notify
        _notify.run_pairs(pairs_data, today)
    except Exception:  # noqa: BLE001
        pass

    # System status — a live readout of what's actually wired/running (booleans only, no secrets).
    import os as _os
    _has_worker = bool(CONFIG.live_quotes_url)
    # AI is "on" when the key is set AND there's no known API error. (llm_status has no top-level
    # 'ok'; a failed probe — e.g. exhausted credits / bad key — sets probe.ok = False.)
    _llm = llm_status or {}
    _llm_probe_failed = isinstance(_llm.get("probe"), dict) and _llm["probe"].get("ok") is False
    _llm_on = bool(CONFIG.llm_enabled) and not _llm_probe_failed
    _gen = _llm.get("generated")
    if not CONFIG.llm_enabled:
        _ai_note = "ANTHROPIC_API_KEY not set"
    elif _llm_probe_failed:
        _ai_note = "API error — check credits/key"
    elif _gen:
        _ai_note = f"{CONFIG.llm_model} · {_gen} brief(s) this run"
    else:
        _ai_note = f"{CONFIG.llm_model} · no High-conviction names this run"
    system = {
        "mode": mode,
        "feeds": [
            {"name": "Alpaca (prices/quotes)", "on": bool(CONFIG.api_key and CONFIG.secret_key),
             "note": f"{mode} account · IEX feed"},
            {"name": "Yahoo Finance (charts/history)", "on": _has_worker, "note": "via Cloudflare Worker proxy"},
            {"name": "Finnhub (fundamentals/analysts/earnings)", "on": bool(CONFIG.finnhub_api_key), "note": "API key"},
            {"name": "FRED (macro backdrop)", "on": bool(CONFIG.fred_api_key), "note": "API key"},
            {"name": "SEC EDGAR (insider Form 4)", "on": True, "note": "keyless, official"},
            {"name": "StockTwits (retail buzz)", "on": _has_worker, "note": "via Worker"},
            {"name": "TradingView (TA cross-check)", "on": _has_worker, "note": "via Worker"},
            {"name": "News RSS + Benzinga", "on": True, "note": "Yahoo/Finnhub/Benzinga headlines"},
        ],
        "ai": [
            {"name": "AI analyst briefs", "on": _llm_on, "note": _ai_note},
            {"name": "News-idea engine", "on": _llm_on,
             "note": "headlines → ideas + nudge" if _llm_on else "needs ANTHROPIC_API_KEY + credits"},
        ],
        "engine": [
            {"name": "Multi-strategy confluence (7 long + 7 short)", "on": True, "note": "core engine"},
            {"name": "Relative-strength factor", "on": True, "note": f"conviction weight {CONFIG.rs_conviction_weight}"},
            {"name": "Post-earnings drift (PEAD)", "on": bool(CONFIG.pead_enabled), "note": "confluence input"},
            {"name": "Earnings gate", "on": True, "note": "no fresh entry ≤2d to report"},
            {"name": "Regime-alignment tilt", "on": True, "note": "with/against the tape"},
            {"name": "Regime block on buys", "on": bool(CONFIG.regime_block_buys), "note": "demote longs in Risk-off"},
            {"name": "Conviction×vol sizing", "on": True, "note": "backtest + paper"},
        ],
        "execution": [
            {"name": "Auto paper-trading", "on": bool(CONFIG.paper_trade),
             "note": f"risk {CONFIG.paper_risk_pct:.0%}/trade · max {CONFIG.paper_max_open} open" if CONFIG.paper_trade else "set PAPER_TRADE=true"},
            {"name": "Live exit manager (partials/trail)", "on": bool(CONFIG.manage_exits), "note": "amends real OCO legs"},
            {"name": "Wider universe", "on": bool(CONFIG.wide_universe), "note": "expanded scan pool"},
            {"name": "Time-stop", "on": CONFIG.max_hold_days > 0, "note": f"{CONFIG.max_hold_days}d" if CONFIG.max_hold_days else "off"},
        ],
        "scrapers": [
            {"name": "Insider buys (SEC)", "on": True}, {"name": "Retail buzz (StockTwits)", "on": _has_worker},
            {"name": "Analyst rating changes (Finnhub)", "on": bool(CONFIG.finnhub_api_key)},
        ],
        "delivery": [
            {"name": "Phone push (ntfy)", "on": bool(_os.getenv("ALERT_NTFY_TOPIC"))},
            {"name": "Webhook (Discord/Slack)", "on": bool(_os.getenv("ALERT_WEBHOOK_URL"))},
            {"name": "Email (SMTP)", "on": bool(_os.getenv("ALERT_EMAIL_TO") and _os.getenv("SMTP_HOST"))},
            {"name": "Morning digest + intraday alerts", "on": True, "note": "scheduled tasks"},
        ],
        "infra": [
            {"name": "Cloudflare Worker proxy", "on": _has_worker, "note": "quotes/charts/TV/StockTwits"},
            {"name": "GitHub Actions rebuild", "on": True, "note": "every ~30 min, market hours + on push"},
            {"name": "GitHub Pages hosting", "on": True, "note": "static site"},
        ],
    }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT"),
        "generated_ts": int(datetime.now(timezone.utc).timestamp()),
        "mode": mode,
        "scanned": len(rows),
        "diagnostics": list(scanner.LAST_ERRORS),
        "audit_summary": None,  # filled by main() after the audit — kept early so it survives a truncated fetch
        "news_sources": dict(__import__("collections").Counter(
            (n.get("source") or "?") for n in news).most_common(14)),
        "llm": llm_status,
        "benchmark": benchmark,
        "track": track,
        "paper_acct": paper_acct,
        "news_ideas": news_ideas,
        "alerts": alerts,
        "system": system,
        "regime": regime,
        "sectors": sectors,
        "concentration": _concentration(shown),
        "macro": macro,
        "macro_posture": macro_posture,
        "notrade": notrade_gate,
        "price_drops": price_drops,
        "momentum": [dict(m, name=scanner.name_of(
                        m["symbol"], {r["symbol"]: r.get("name", "") for r in shown}.get(m["symbol"], "")))
                     for m in momentum_rows],
        "mom_detail": _mom_detail(momentum_rows, rows_by_sym, shown),
        "allweather": all_weather,
        "momentum_bt": momentum_bt,
        "walkforward": walk_fwd,
        "portfolio": _portfolio(shown),
        "ipos": ipos,
        "ipo_news": ipo_news,
        "params": {
            "fast_ma": CONFIG.fast_ma, "slow_ma": CONFIG.slow_ma,
            "rsi_period": CONFIG.rsi_period, "risk_per_trade": CONFIG.risk_per_trade,
            "stop_loss_pct": CONFIG.stop_loss_pct, "take_profit_pct": CONFIG.take_profit_pct,
            "rel_volume_window": CONFIG.rel_volume_window,
            "intraday_timeframe": CONFIG.intraday_timeframe,
        },
        "signals": shown,
        "ranked": ranked,
        "structured": structured,
        "nlp_scores": nlp_scores,
        "pairs": pairs_data,
        "intraday": intraday_shown,
        "intraday_track": intraday_track,
        "market_brief": market_brief,
        "changes": changes,
        "calendar": calendar,
        "charts": {k: charts[k] for k in shown_syms if k in charts},
        "news": news,
    }


def _regime_html(reg: dict | None) -> str:
    if not reg:
        return ""
    palette = {"Risk-on": ("#15361f", "#7ee2a0"), "Neutral": ("#3a2e12", "#e8c878"),
               "Risk-off": ("#3a1e1e", "#ff9b9b")}
    bg, fg = palette.get(reg["label"], ("#1a212b", "var(--txt2)"))
    return (f'<div class="regime" style="background:{bg};">'
            f'<span class="rlabel" style="color:{fg};">Market: {reg["label"]}</span>'
            f'<span class="rdetail">{reg["breadth"]}% of {reg["total"]} scanned above trend &middot; '
            f'avg momentum {reg["avg_rsi"]}/100 &middot; {reg["buys"]} fresh buys</span>'
            f'<span class="rnote">{reg["note"]}</span></div>')


def _kpi_html(reg: dict | None, snap: dict) -> str:
    """A summary strip of KPI tiles up top — the 'what matters now' inverted pyramid."""
    sigs = snap.get("signals", [])
    n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
    n_short = sum(1 for s in sigs if s.get("action") == "SHORT")
    tone = {"Risk-on": "buy", "Neutral": "warn", "Risk-off": "sell"}.get((reg or {}).get("label"), "")
    tk = snap.get("track") or {}
    wr = tk.get("win_rate")
    wr_txt = f'{wr}%' if isinstance(wr, (int, float)) else "—"

    def tile(label, value, cls="", sub=""):
        v = f'<div class="kpi-v {cls}">{value}</div>'
        s = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return f'<div class="kpi"><div class="kpi-l">{label}</div>{v}{s}</div>'

    tiles = ""
    if reg:
        tiles += (f'<div class="kpi hero"><div class="kpi-l">Market regime</div>'
                  f'<div class="kpi-v {tone}">{reg.get("label", "—")}</div>'
                  f'<div class="kpi-sub">{reg.get("note", "")[:60]}</div></div>')
        tiles += tile("Breadth", f'{reg.get("breadth", "—")}%', "", f'of {reg.get("total","?")} above trend')
        tiles += tile("Avg momentum", f'{reg.get("avg_rsi", "—")}', "", "RSI, 0–100")
    tiles += tile("Fresh buys", str(n_buy), "buy" if n_buy else "", "new long setups")
    tiles += tile("Fresh shorts", str(n_short), "sell" if n_short else "", "new short setups")
    tiles += tile("Track record", wr_txt, "", f'{tk.get("resolved", 0)} calls resolved')
    return f'<div class="kpis">{tiles}</div>'


def _bento_home(snap: dict) -> str:
    """The Signals home as a true HUD bento — a grid of varied-size tiles: a regime hero, the key
    metrics, the market brief and what-changed, all as modular boxes."""
    reg = snap.get("regime") or {}
    mp = snap.get("macro_posture") or {}
    sigs = snap.get("signals", [])
    n_buy = sum(1 for s in sigs if s.get("action") == "BUY")
    n_short = sum(1 for s in sigs if s.get("action") == "SHORT")
    tk = snap.get("track") or {}
    wr = tk.get("win_rate")
    wr_txt = f'{wr}%' if isinstance(wr, (int, float)) else "—"
    ranked = snap.get("ranked") or []
    tone = {"Risk-on": "buy", "Neutral": "warn", "Risk-off": "sell"}.get(reg.get("label"), "")

    def t(label, val, cls="", sub=""):
        sb = f'<div class="bt-sub">{sub}</div>' if sub else ""
        return f'<div class="bt"><div class="bt-l">{label}</div><div class="bt-v {cls}">{val}</div>{sb}</div>'

    if not reg and not sigs:
        return ""
    expo = mp.get("exposure_mult")
    tiles = (
        f'<div class="bt hero"><div class="bt-l">Market regime</div>'
        f'<div class="bt-v {tone}" style="font-size:38px;">{reg.get("label", "—")}</div>'
        f'<div class="bt-sub">{reg.get("note", "")[:80]}</div>'
        + (f'<div class="bt-chip">{expo:.2f}× sizing'
           + (f' · entry bar {mp.get("entry_threshold")}%' if mp.get("entry_threshold") else '')
           + '</div>' if expo else '')
        + '</div>'
    )
    tiles += t("Breadth", f'{reg.get("breadth", "—")}%', sub=f'of {reg.get("total","?")} above trend')
    tiles += t("Avg momentum", f'{reg.get("avg_rsi", "—")}', sub="RSI 0–100")
    tiles += t("Fresh buys", str(n_buy), "buy" if n_buy else "", "new long setups")
    tiles += t("Fresh shorts", str(n_short), "sell" if n_short else "", "new short setups")
    tiles += t("Track record", wr_txt, "", f'{tk.get("resolved", 0)} resolved')
    # top opportunity tile (the #1 ranked name) — quick glance at where capital goes first
    if ranked:
        top = ranked[0]
        tiles += (f'<div class="bt"><div class="bt-l">Top opportunity</div>'
                  f'<div class="bt-v" style="font-size:20px;">{top.get("symbol","")}</div>'
                  f'<div class="bt-sub">rank {top.get("rank_score","—")} · {top.get("action","")}</div></div>')
    brief = (snap.get("market_brief") or "").strip()
    if brief:
        tiles += (f'<div class="bt wide"><div class="bt-l">🧠 Market brief</div>'
                  f'<div class="bt-body">{brief}</div></div>')
    changes = snap.get("changes") or []
    if changes:
        tiles += ('<div class="bt wide"><div class="bt-l">⚡ What changed since last build</div>'
                  '<ul class="bt-list">' + "".join(f"<li>{c}</li>" for c in changes) + "</ul></div>")
    return f'<div class="bento">{tiles}</div>'


def _allweather_html(aw: dict | None) -> str:
    intro = (_strat_badge("All-seasons allocation · static, ~yearly rebalance") +
             '<p style="color:var(--muted);font-size:13px;margin:0 0 14px;max-width:760px;">'
             "Ray Dalio's <b>All Weather</b> portfolio is a static, <b>risk-balanced</b> allocation built to "
             "hold up across all four economic environments — rising and falling growth, rising and falling "
             "inflation — instead of betting on which is next. It's a <b>buy-and-hold</b> mix you rebalance "
             "about once a year, <i>not</i> a trading signal. The trade-off: lower returns than all-stocks in "
             "a bull run, but much shallower drawdowns and steadier compounding.</p>")
    if not aw:
        return intro + '<p style="color:var(--muted);font-size:13px;">All Weather data unavailable.</p>'

    # allocation table
    body = ""
    colors = ["#2ea043", "#58a6ff", "#7aa2f7", "#d29922", "#c08457"]
    bar = ""
    for i, t in enumerate(aw.get("targets", [])):
        px = f'${t["price"]:,.2f}' if t.get("price") is not None else "—"
        body += (f'<tr><td><b>{t["symbol"]}</b> <span style="color:var(--muted);font-weight:400;">{t["name"]}</span></td>'
                 f'<td style="color:var(--muted);">{t["role"]}</td>'
                 f'<td style="color:var(--muted);">{t["env"]}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{px}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;font-weight:700;">{t["weight"]}%</td></tr>')
        bar += (f'<div title="{t["symbol"]} {t["weight"]}%" style="width:{t["weight"]}%;'
                f'background:{colors[i % len(colors)]};">{t["weight"] if t["weight"] >= 7 else ""}</div>')
    table = ('<table class="trackrec"><thead><tr><th>Asset</th><th>Role</th>'
             '<th>Best environment</th><th style="text-align:right;">Price</th>'
             '<th style="text-align:right;">Target weight</th></tr></thead>'
             f'<tbody>{body}</tbody></table>')
    bar_html = (f'<div style="display:flex;height:26px;border-radius:6px;overflow:hidden;'
                f'margin:4px 0 18px;font-size:10px;color:#fff;font-weight:700;text-align:center;'
                f'line-height:26px;">{bar}</div>')

    # backtest vs SPY
    bt_html = ""
    bt = aw.get("backtest")
    if bt:
        a, s = bt["allweather"], bt["spy"]
        def _row(label, m, hot=False):
            w = 'font-weight:700;' if hot else ''
            ddc = 'color:var(--buy);' if m["maxdd"] > s["maxdd"] else ''
            return (f'<tr style="{w}"><td>{label}</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["ret"]:.0f}%</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["cagr"]:.1f}%</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["sharpe"]:.2f}</td>'
                    f'<td style="text-align:right;font-variant-numeric:tabular-nums;{ddc}">{m["maxdd"]:.1f}%</td></tr>')
        bt_html = (f'<h3 style="margin:18px 0 6px;">Backtest vs S&amp;P 500 — {bt["years"]} years '
                   f'({bt["start"]} → {bt["end"]}, monthly rebalance)</h3>'
                   '<table class="trackrec"><thead><tr><th>Portfolio</th>'
                   '<th style="text-align:right;">Total return</th><th style="text-align:right;">CAGR</th>'
                   '<th style="text-align:right;" title="risk-adjusted return — higher is better">Sharpe</th>'
                   '<th style="text-align:right;" title="worst peak-to-trough drop — closer to zero is better">Max drawdown</th>'
                   '</tr></thead><tbody>'
                   + _row("All Weather", a, hot=True) + _row("S&amp;P 500 (buy &amp; hold)", s)
                   + '</tbody></table>'
                   '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">The point isn\'t to beat the '
                   "S&amp;P on raw return — it usually won't in a bull market. It's the <b>shallower max drawdown</b> "
                   "and steadier ride (often a comparable or better Sharpe). If you can't stomach a deep stock-market "
                   "fall, that smoother path is the whole appeal.</p>")
    else:
        bt_html = ('<p style="color:var(--muted);font-size:12px;margin-top:10px;">'
                   f'Backtest unavailable ({aw.get("data_src", "no data")}). Allocation shown above.</p>')

    why = ('<div class="deskread" style="margin-top:16px;border-left-color:#58a6ff;">'
           '<b>Why these five?</b> Each sleeve is the asset that tends to do best in one environment, so '
           'something is usually working: stocks for rising growth, long Treasuries for falling growth/'
           'deflation, intermediate Treasuries as ballast, gold and commodities for rising inflation. They '
           'are weighted so no single one dominates the portfolio\'s <i>risk</i> — which is why bonds get a '
           'big nominal slice (they swing less than stocks).</div>')
    caveat = ('<p style="color:var(--muted);font-size:12px;margin-top:12px;">Educational only, not advice. '
              'Backtest ignores fees, taxes and fund expense ratios; past performance isn\'t a forecast. '
              'The 2022 simultaneous stock+bond drawdown was a notably hard stretch for this mix.</p>')
    return intro + bar_html + table + bt_html + why + caveat


def _ipo_html(ipos: list[dict], ipo_news: list[dict]) -> str:
    """Upcoming-IPO calendar + general headlines mentioning pre-IPO names (SpaceX etc.)."""
    if ipos:
        rows = ""
        for r in ipos[:30]:
            price = r.get("price") or "—"
            val = r.get("value")
            try:
                valtxt = f'${float(val) / 1e6:,.0f}M' if val else "—"
            except Exception:  # noqa: BLE001
                valtxt = "—"
            rows += (f'<tr><td>{r.get("date","")}</td><td><b>{r.get("name","")}</b></td>'
                     f'<td>{r.get("symbol","") or "—"}</td><td>{r.get("exchange","") or ""}</td>'
                     f'<td style="text-align:right;">{price}</td>'
                     f'<td style="text-align:right;color:var(--muted);">{valtxt}</td>'
                     f'<td>{r.get("status","") or ""}</td></tr>')
        cal = ('<table class="trackrec"><thead><tr><th>Date</th><th>Company</th><th>Ticker</th>'
               '<th>Exchange</th><th style="text-align:right;">Price</th>'
               '<th style="text-align:right;">Deal size</th><th>Status</th></tr></thead>'
               f'<tbody>{rows}</tbody></table>')
    else:
        cal = ('<p style="color:var(--muted);font-size:13px;">No companies have formally filed to list in the next '
               '~90 days. Rumoured deals like SpaceX appear in the buzz feed below until they file an S-1.</p>')
    if ipo_news:
        items = ""
        for n in ipo_news:
            t = (f'<a href="{n["url"]}" target="_blank" rel="noopener">{n["headline"]}</a>'
                 if n.get("url") else f'<span class="h">{n["headline"]}</span>')
            items += (f'<li>{t}<div class="src">{n.get("source","")} {n.get("created_at","")} '
                      f'&middot; <span class="chip mini neutral">{n.get("match","")}</span></div></li>')
        news = f'<ul class="news">{items}</ul>'
    else:
        news = '<p style="color:var(--muted);font-size:13px;">No pre-IPO headlines matched right now.</p>'
    return (f'<div class="sech" style="margin-top:0;">Upcoming IPO calendar</div>{cal}'
            '<div class="sech">Pre-IPO buzz — including private names like SpaceX</div>'
            '<p style="color:var(--muted);font-size:12.5px;margin:0 0 8px;">'
            'Private companies have no ticker, so they can\'t be scanned or charted — these are '
            'general-market headlines mentioning notable pre-IPO names and IPO filings.</p>'
            f'{news}')


def _has_bad_bar(cl: list, jump: float = 0.50) -> bool:
    """True if the close series contains a spike-and-revert bad print: a single-day move
    larger than `jump` that is undone (mostly) the very next day. A genuine split or a real
    trend move does NOT immediately reverse, so this isolates corrupt IEX bars without
    flagging legitimate big movers. One such bar anywhere in the ~12-month window can wreck
    the momentum base, so any occurrence disqualifies the name from the leaderboard."""
    for i in range(1, len(cl) - 1):
        p0, p1, p2 = cl[i - 1], cl[i], cl[i + 1]
        if not p0 or not p1:
            continue
        r1 = p1 / p0 - 1.0          # move into the suspect bar
        if abs(r1) <= jump:
            continue
        if not p2:
            return True             # huge move with no valid confirmation bar — distrust it
        r2 = p2 / p1 - 1.0          # move out of it
        # reverses if the next day undoes most of the spike (opposite sign, similar size)
        if r1 * r2 < 0 and abs(r2) >= jump * 0.6:
            return True
    return False


def _momentum_rank(charts: dict, top: int = 15, per_sector: int = 3) -> list[dict]:
    """Dual-momentum leaderboard: 12-1 momentum, kept only if positive AND above the
    200-day average. Then capped to ``per_sector`` names per sector (diversification)
    and assigned inverse-volatility suggested weights (risk-parity, like factor funds)."""
    max_mom = getattr(CONFIG, "max_momentum_pct", 200.0)
    jump = getattr(CONFIG, "bad_bar_jump_pct", 50.0) / 100.0
    cand = []
    for sym, ch in charts.items():
        cl = [c for c in (ch.get("close") or []) if c is not None]
        n = len(cl)
        if n < 230:
            continue
        # Bad-bar guard: a single-day move > `jump` that immediately reverses (spike-and-
        # revert) is the signature of a corrupt IEX print, not a real move or a split. Such
        # a bad bar ~12 months back deflates the momentum base and balloons the score
        # (this is what produced MU +591% / INTC +479%). Drop the name rather than trust it.
        if _has_bad_bar(cl, jump):
            continue
        lb = min(252, n - 1)
        sk = 21 if n > 257 else 0
        base = cl[-lb]
        recent = cl[-1 - sk] if sk else cl[-1]
        if not base:
            continue
        score = recent / base - 1
        sma200 = sum(cl[-200:]) / 200 if n >= 200 else sum(cl) / n
        if not (score > 0 and cl[-1] > sma200):
            continue
        # Score sanity cap: a 12-1 momentum above `max_mom`% on a scanned large/mid-cap is
        # almost certainly a leftover data artifact (a bad base the spike check missed), not
        # a tradeable winner. Drop it so the leaderboard never publishes impossible numbers.
        if score * 100 > max_mom:
            continue
        rets = [cl[i] / cl[i - 1] - 1 for i in range(max(1, n - 21), n) if cl[i - 1]]
        vol = (sum((x - sum(rets) / len(rets)) ** 2 for x in rets) / len(rets)) ** 0.5 if len(rets) > 1 else 0.0
        ext = (cl[-1] / sma200 - 1) * 100 if sma200 else 0.0
        r1m = (cl[-1] / cl[-22] - 1) * 100 if n >= 22 and cl[-22] else None
        cand.append({"symbol": sym, "score": round(score * 100, 1),
                     "price": round(cl[-1], 2), "sector": scanner.sector_of(sym),
                     "ext": round(ext, 1), "r1m": round(r1m, 1) if r1m is not None else None,
                     "_vol": vol})
    cand.sort(key=lambda x: -x["score"])
    out, cnt = [], {}
    for c in cand:
        if cnt.get(c["sector"], 0) >= per_sector:
            continue
        out.append(c)
        cnt[c["sector"]] = cnt.get(c["sector"], 0) + 1
        if len(out) >= top:
            break
    invs = [(1.0 / c["_vol"] if c["_vol"] > 1e-9 else 0.0) for c in out]
    tot = sum(invs) or 1.0
    for c, iv in zip(out, invs):
        c["weight"] = round(iv / tot * 100, 1)
        c.pop("_vol", None)
    return out


def _portfolio(rows: list[dict]) -> dict:
    """Aggregate the actionable signals into a hypothetical book — what you'd be holding
    if you took every BUY/SHORT/HOLD at the model's position size (risk-based, on
    starting_cash). Net/gross exposure, sector mix, total $ at risk, per-position list."""
    longs = [r for r in rows if r.get("action") in ("BUY", "HOLD LONG")]
    shorts = [r for r in rows if r.get("action") in ("SHORT", "HOLD SHORT")]
    def expo(r): return (r.get("plan") or {}).get("exposure") or 0.0
    def risk(r): return (r.get("plan") or {}).get("dollar_risk") or 0.0
    long_e = sum(expo(r) for r in longs)
    short_e = sum(expo(r) for r in shorts)
    sec: dict = {}
    for r in longs:
        sec[r.get("sector", "Other")] = sec.get(r.get("sector", "Other"), 0.0) + expo(r)
    for r in shorts:
        sec[r.get("sector", "Other")] = sec.get(r.get("sector", "Other"), 0.0) - expo(r)
    positions = [{
        "symbol": r["symbol"], "name": r.get("name", ""), "action": r["action"],
        "direction": r.get("direction"), "sector": r.get("sector", ""),
        "shares": (r.get("plan") or {}).get("shares"),
        "exposure": round(expo(r)), "risk": round(risk(r)),
        "conviction": (r.get("conviction") or {}).get("score_pct"),
    } for r in longs + shorts]
    positions.sort(key=lambda p: -(p["conviction"] or 0))
    return {
        "n_long": len(longs), "n_short": len(shorts),
        "long_exposure": round(long_e), "short_exposure": round(short_e),
        "gross": round(long_e + short_e), "net": round(long_e - short_e),
        "at_risk": round(sum(risk(r) for r in longs + shorts)),
        "starting_cash": CONFIG.starting_cash,
        "sectors": sorted(sec.items(), key=lambda kv: -abs(kv[1]))[:10],
        "positions": positions,
    }


def _portfolio_html(p: dict | None) -> str:
    badge = _strat_badge("Hypothetical book · every actionable signal at model size")
    if not p or not p.get("positions"):
        return badge + ('<p style="color:var(--muted);font-size:13px;">No actionable positions right '
                        'now — nothing to assemble into a book.</p>')
    cash = p.get("starting_cash") or 100000
    pct = lambda v: f'{v/cash*100:.0f}%'
    money = lambda v: f'${v:,.0f}'
    def tile(label, value, sub="", cls=""):
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        return f'<div class="stat"><div class="l">{label}</div><div class="v {cls}">{value}</div>{sub_html}</div>'
    netcls = "buy" if p["net"] >= 0 else "sell"
    tiles = (tile("Positions", f'{p["n_long"]}L / {p["n_short"]}S', "long / short") +
             tile("Gross exposure", money(p["gross"]), pct(p["gross"]) + " of book") +
             tile("Net exposure", ("+" if p["net"] >= 0 else "") + money(p["net"]), pct(p["net"]) + " net " + ("long" if p["net"] >= 0 else "short"), netcls) +
             tile("$ at risk", money(p["at_risk"]), pct(p["at_risk"]) + " if all stop out", "sell"))
    # sector mix bar (green long / red short, width by |exposure| share of gross)
    gross = p["gross"] or 1
    secbar = ""
    for name, val in p["sectors"]:
        w = abs(val) / gross * 100
        col = "var(--buy)" if val >= 0 else "var(--sell)"
        secbar += (f'<div class="secrow"><span class="secname">{name}</span>'
                   f'<div class="secbarwrap"><div class="secbarfill" style="width:{w:.0f}%;background:{col};"></div></div>'
                   f'<span class="secval" style="color:{col};">{("+" if val>=0 else "−")}{money(abs(val))}</span></div>')
    rows = ""
    for q in p["positions"]:
        acol = "var(--sell)" if q["direction"] == "SHORT" else "var(--buy)"
        rows += (f'<tr><td><b>{q["symbol"]}</b> <span style="color:var(--muted);font-weight:400;">{q["name"][:22]}</span></td>'
                 f'<td style="color:{acol};">{q["action"]}</td><td style="color:var(--muted);">{q["sector"]}</td>'
                 f'<td style="text-align:right;">{q["shares"] or "—"}</td>'
                 f'<td style="text-align:right;">{money(q["exposure"])}</td>'
                 f'<td style="text-align:right;">{money(q["risk"])}</td>'
                 f'<td style="text-align:right;">{q["conviction"] if q["conviction"] is not None else "—"}</td></tr>')
    table = ('<table class="trackrec"><thead><tr><th>Position</th><th>Side</th><th>Sector</th>'
             '<th style="text-align:right;">Shares</th><th style="text-align:right;">Exposure</th>'
             '<th style="text-align:right;">$ risk</th><th style="text-align:right;">Conv</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>')
    note = ('<p style="color:var(--muted);font-size:12px;margin-top:10px;">Hypothetical: assumes you take '
            'every actionable signal at the model\'s risk-based size on a '
            f'{money(cash)} book. Not advice; sizing is illustrative. Net exposure = long − short.</p>')
    return (badge + f'<div class="trackstats">{tiles}</div>'
            + '<div class="sech">Sector tilt <span style="text-transform:none;color:var(--muted);font-weight:400;">— net exposure by sector (green long · red short)</span></div>'
            + f'<div class="secmix">{secbar}</div>'
            + '<div class="sech">Positions</div>' + table + note)


def _strat_badge(value: str) -> str:
    """A small, consistent 'Strategy type: …' pill so every tab self-labels its approach."""
    return (f'<div class="strat-badge"><span class="k">Strategy type</span>'
            f'<span class="v">{value}</span></div>')


def _mom_detail(momentum_rows: list[dict], rows_by_sym: dict, shown: list[dict]) -> dict:
    """Full analysis row per momentum leader, keyed by symbol, so the leaderboard rows can
    open the same rich detail modal (chart, info, reasoning, conviction) as the signal cards.
    Prefers the already-enriched `shown` row (has research) and prepends a momentum-context line."""
    shown_by_sym = {r["symbol"]: r for r in shown}
    out = {}
    for m in momentum_rows:
        sym = m["symbol"]
        row = shown_by_sym.get(sym) or rows_by_sym.get(sym)
        if not row:
            continue
        d = dict(row)
        d["name"] = scanner.name_of(sym, d.get("name", ""))
        r1m = m.get("r1m")
        r1m_txt = f"{'+' if (r1m or 0) >= 0 else ''}{r1m}%" if r1m is not None else "—"
        mom_note = (f"📈 Momentum leader — 12-1 momentum +{m['score']}% (its return over the last ~12 "
                    f"months, skipping the most recent). It's +{m.get('ext', 0)}% above its 200-day "
                    f"average and did {r1m_txt} over the past month. Suggested risk-parity weight "
                    f"{m.get('weight', '—')}% in a monthly-rebalanced leaders basket — a positional "
                    f"hold, not a day-trade.")
        d["reasons"] = [mom_note] + list(d.get("reasons") or [])
        out[sym] = d
    return out


def _momentum_bt_html(bt: dict | None) -> str:
    """Honest, survivorship-bias-FREE momentum backtest (fixed ETF universe) vs SPY."""
    if not bt or not bt.get("strategy") or not bt.get("spy"):
        return ""
    m, s = bt["strategy"], bt["spy"]
    ddc = 'color:var(--buy);' if m["maxdd"] > s["maxdd"] else ''
    shc = 'color:var(--buy);' if m["sharpe"] > s["sharpe"] else ''
    rows = (f'<tr><td><b>Dual-momentum (ETFs)</b></td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["ret"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{m["cagr"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;{shc}">{m["sharpe"]:.2f}</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;{ddc}">{m["maxdd"]:.1f}%</td></tr>'
            f'<tr><td>SPY buy &amp; hold</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["ret"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["cagr"]:.1f}%</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["sharpe"]:.2f}</td>'
            f'<td style="text-align:right;font-variant-numeric:tabular-nums;">{s["maxdd"]:.1f}%</td></tr>')
    uni = ", ".join(bt.get("universe", [])[:18])
    return ('<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">📐 Honest backtest — '
            'survivorship-bias-free</div>'
            f'<table class="trackrec" style="margin-top:8px;"><thead><tr><th>Strategy</th>'
            '<th style="text-align:right;">Total return</th><th style="text-align:right;">CAGR</th>'
            '<th style="text-align:right;" title="risk-adjusted return — higher is better">Sharpe</th>'
            f'<th style="text-align:right;">Max drawdown</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p style="color:var(--muted);font-size:12px;margin:10px 0 0;">Same dual-momentum rules run on a '
            f'<b>fixed universe of {bt.get("n_universe","~16")} broad ETFs</b> that existed for the whole '
            f'{bt.get("months","")}-month window and never delist ({uni}) — so the result can\'t be flattered '
            'by hindsight stock-picking. This is the honest performance read; the single-stock ranking below is '
            'for idea generation (and is survivorship-biased — it\'s today\'s winners).</p></div>')


def _walkforward_html(wf: dict | None) -> str:
    """Walk-forward / out-of-sample validation panel: IS vs OOS, per-fold, sensitivity sweep."""
    if not wf or wf.get("error") or not wf.get("per_symbol"):
        return ""
    grade = wf.get("grade", "marginal")
    gcol = {"holds up": "var(--buy)", "marginal": "var(--warn)", "fragile": "var(--sell)"}.get(grade, "var(--muted)")
    oos = wf.get("oos_avg_pct")
    pos = wf.get("oos_pos_folds_pct")
    folds = wf.get("oos_total_folds")
    syms = ", ".join(wf.get("symbols", []))

    # per-symbol OOS row table
    prows = ""
    for s in wf.get("per_symbol", []):
        oc = "buy" if (s.get("oos_avg_fold_pct") or 0) > 0 else "sell"
        prows += (f'<tr><td><b>{s["symbol"]}</b></td>'
                  f'<td style="text-align:right;">{s.get("is_avg_sharpe","—")}</td>'
                  f'<td style="text-align:right;" class="{oc}">{s.get("oos_avg_fold_pct",0):+.2f}%</td>'
                  f'<td style="text-align:right;">{s.get("oos_win_folds",0)}/{s.get("oos_n_folds",0)}</td></tr>')
    ptable = ('<table class="tbl" style="margin-top:8px;"><thead><tr><th>Symbol</th>'
              '<th style="text-align:right;" title="avg Sharpe on the in-sample windows the params were tuned on">IS Sharpe</th>'
              '<th style="text-align:right;" title="avg per-fold return on unseen out-of-sample windows">OOS return/fold</th>'
              '<th style="text-align:right;" title="profitable out-of-sample windows">OOS wins</th></tr></thead>'
              f'<tbody>{prows}</tbody></table>')

    # sensitivity sweep
    srows = ""
    for r in wf.get("sensitivity", []):
        srows += (f'<tr><td>{r["params"]}</td>'
                  f'<td style="text-align:right;">{r["total_return_pct"]:+.1f}%</td>'
                  f'<td style="text-align:right;">{r["sharpe"]:.2f}</td>'
                  f'<td style="text-align:right;">{r["max_dd_pct"]:.1f}%</td></tr>')
    stable = ('<h4 style="font-size:13px;margin:14px 0 4px;">Parameter sensitivity (full sample)</h4>'
              '<table class="tbl"><thead><tr><th title="fast/slow moving-average pair">MA pair</th>'
              '<th style="text-align:right;">Return</th><th style="text-align:right;">Sharpe</th>'
              '<th style="text-align:right;">Max DD</th></tr></thead>'
              f'<tbody>{srows}</tbody></table>') if srows else ""

    return ('<div class="ovbox" style="margin:0 0 16px;border-left:4px solid ' + gcol + ';">'
            '<div class="ovhead">🔬 Walk-forward / out-of-sample validation — '
            f'<span style="color:{gcol};text-transform:capitalize;">{grade}</span></div>'
            f'<div style="font-size:12px;color:var(--muted);margin:6px 0 8px;">'
            f'OOS return/fold <b>{oos:+.2f}%</b> &nbsp;·&nbsp; profitable unseen windows <b>{pos}%</b> '
            f'({folds} folds across {len(wf.get("symbols",[]))} names) &nbsp;·&nbsp; basket: {syms}</div>'
            f'<p style="color:var(--txt2);font-size:13px;margin:0 0 4px;">{wf.get("verdict","")}</p>'
            + ptable + stable +
            '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">Each fold tunes the moving-average pair '
            'on past data only, then trades the next unseen window with those frozen settings — so OOS is a fair test '
            'of the edge, not a curve-fit. Net of modeled slippage. Educational; not investment advice.</p></div>')


def _momentum_html(rows: list[dict]) -> str:
    intro = (_strat_badge("Dual momentum · positional, ~monthly rebalance") +
             '<p style="color:var(--muted);font-size:13px;margin:0 0 12px;max-width:680px;">'
             'Ranked by <b>12-1 momentum</b> (return over the last ~12 months, skipping the most '
             'recent month), keeping only names in their own uptrend (above the 200-day average). '
             'This is the dual-momentum approach factor funds use — the one strategy that beat the '
             'index on risk-adjusted terms across a full cycle in our backtest.</p>')
    if not rows:
        return intro + ('<p style="color:var(--muted);font-size:13px;">Not enough price history to '
                        'rank momentum right now.</p>')
    body = ""
    for i, m in enumerate(rows, 1):
        nm = f' <span style="color:var(--muted);font-weight:400;">{m.get("name","")}</span>' if m.get("name") else ""
        r1m = m.get("r1m")
        if r1m is None:
            r1m_cell = '<td style="text-align:right;color:var(--muted);">—</td>'
        else:
            r1m_cell = (f'<td style="text-align:right;font-variant-numeric:tabular-nums;" '
                        f'class="{"win" if r1m >= 0 else "loss"}">{"+" if r1m >= 0 else ""}{r1m}%</td>')
        new = ' <span class="chip mini bull" style="font-size:9px;padding:0 5px;">NEW</span>' if m.get("is_new") else ""
        body += (f'<tr class="momrow" data-sym="{m["symbol"]}" style="cursor:pointer;">'
                 f'<td>{i}</td><td><b>{m["symbol"]}</b>{nm}{new}</td>'
                 f'<td style="color:var(--muted);">{m.get("sector","")}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;">${m["price"]:,.2f}</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;" class="win">+{m["score"]}%</td>'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);">+{m.get("ext",0)}%</td>'
                 f'{r1m_cell}'
                 f'<td style="text-align:right;font-variant-numeric:tabular-nums;font-weight:600;">{m.get("weight","—")}%</td></tr>')
    table = ('<table class="trackrec"><thead><tr><th>#</th><th>Stock</th><th>Sector</th>'
             '<th style="text-align:right;">Price</th>'
             '<th style="text-align:right;" title="return over ~12 months, skipping the last month">12-1 momentum</th>'
             '<th style="text-align:right;" title="how far above its 200-day average — bigger = more extended">vs 200d</th>'
             '<th style="text-align:right;" title="last ~1 month return — negative means the leader is cooling off">1-mo</th>'
             '<th style="text-align:right;" title="suggested risk-parity weight: more to steadier names, less to volatile ones">Wt</th>'
             f'</tr></thead><tbody>{body}</tbody></table>'
             '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">'
             'Columns: <b>12-1 momentum</b> = ranking signal · <b>vs 200d</b> = how extended (high = chasing risk) · '
             '<b>1-mo</b> = recent month (negative = losing steam) · <b>Wt</b> = suggested inverse-volatility weight. '
             'Capped at 3 names per sector. Re-rank monthly; rotate out names that drop off. '
             '<span class="chip mini bull" style="font-size:9px;padding:0 5px;">NEW</span> = entered the list this run.</p>')
    caveats = ('<div class="deskread" style="margin-top:16px;border-left-color:#e8c878;">'
               '<b>Read before using.</b> This is a monthly-rebalanced approach (hold the leaders, '
               're-rank ~monthly) — not a day-trade list. In backtest it earned a higher Sharpe than '
               'the index over ~9 years <i>including</i> the 2022 bear, but with a deeper ~31% drawdown, '
               'and the figures are flattered by survivorship bias (this watchlist is today\'s winners). '
               'Expect a smaller real edge and real drawdowns. Educational only — not financial advice.</div>')
    return intro + table + caveats


def _sectors_html(secs: list[dict]) -> str:
    if not secs:
        return ""
    rows = "".join(
        f'<div class="secrow"><span class="secname">{s["sector"]}</span>'
        f'<div class="secbar"><div class="secfill" style="width:{s["pct_up"]}%;"></div></div>'
        f'<span class="secpct">{s["pct_up"]}% up · {s["count"]}</span></div>'
        for s in secs)
    return ('<div class="ovbox"><div class="ovhead">🧭 Sector strength '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">— share of each sector trending up</span></div>'
            f'{rows}</div>')


def _ranked_html(ranked: list | None, top: int = 12) -> str:
    """Adaptive allocation ranking — the best setups for capital, with a labelled factor breakdown."""
    if not ranked:
        return ""

    def fcell(v, cls=""):
        v = max(0, min(100, int(v or 0)))
        c = "var(--buy)" if v >= 67 else "var(--warn)" if v >= 40 else "var(--sell)"
        return (f'<td class="{cls}" style="min-width:62px;vertical-align:middle;">'
                f'<div style="font-size:12px;color:var(--txt2);margin-bottom:3px;text-align:center;">{v}</div>'
                f'<div style="height:6px;border-radius:3px;background:color-mix(in srgb,var(--accent) 12%,transparent);">'
                f'<div style="height:100%;width:{v}%;border-radius:3px;background:{c};"></div></div></td>')
    rows = ""
    for i, r in enumerate(ranked[:top], 1):
        f = r.get("factors", {})
        d = r.get("direction", "LONG")
        dcol = "buy" if d == "LONG" else "sell"
        nm = (r.get("name") or "")
        nm = (nm[:22] + "…") if len(nm) > 23 else nm
        pr = r.get("price")
        prc = f'${pr:,.2f}' if isinstance(pr, (int, float)) else ""
        rows += (
            f'<tr><td style="text-align:right;color:var(--muted);">{i}</td>'
            f'<td style="min-width:170px;"><b>{r["symbol"]}</b> '
            f'<span style="color:var(--muted);font-size:12px;">{nm}</span>'
            f'<div style="color:var(--muted);font-size:11px;">{prc}</div></td>'
            f'<td class="{dcol}" style="white-space:nowrap;">{r.get("action","")}</td>'
            f'<td style="text-align:center;"><b style="font-size:16px;">{r.get("rank_score","")}</b></td>'
            + fcell(f.get("quality")) + fcell(f.get("vreward")) + fcell(f.get("macrofit"))
            + fcell(f.get("liquidity"), "rkf-sm") + fcell(f.get("momentum"), "rkf-sm") + '</tr>'
        )
    th = ('<th style="text-align:center;font-size:11px;">{}</th>')
    thh = ('<th class="rkf-sm" style="text-align:center;font-size:11px;">{}</th>')
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">🎯 Top opportunities '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— adaptive allocation rank: '
        'where limited capital should go first</span></div>'
        '<table class="tbl" style="margin-top:8px;width:100%;"><thead><tr>'
        '<th style="text-align:right;">#</th><th>Stock</th><th>Action</th>'
        '<th style="text-align:center;" title="0–100 composite allocation score">Rank</th>'
        + th.format("Quality") + th.format("Reward") + th.format("Macro&nbsp;fit")
        + thh.format("Liquidity") + thh.format("Momentum") + '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:12px;margin:12px 0 0;line-height:1.6;">The <b>Rank</b> blends the five '
        'factors into one 0–100 allocation score, and capital (paper entries) goes to the highest first. '
        '<b>Quality</b> = conviction; <b>Reward</b> = volatility-adjusted reward:risk; <b>Macro fit</b> = how well the '
        'trade suits the current regime; <b>Liquidity</b> = how cleanly it trades; <b>Momentum</b> = trend strength. '
        'Greener bars are stronger. Educational; not advice.</p></div>'
    )


def _notrade_html(nt: dict | None) -> str:
    """Unified no-trade panel: the conditions that pause new entries, each ok / caution / block."""
    if not nt or not nt.get("checks"):
        return ""
    blocked = nt.get("block_new")
    head_col = "var(--sell)" if blocked else ("var(--warn)" if nt.get("cautions") else "var(--buy)")
    head_txt = ("Standing down — not opening new positions" if blocked
                else "Caution — trading with reservations" if nt.get("cautions")
                else "Clear to trade — no blocking conditions")
    icon = {"ok": "🟢", "caution": "🟡", "block": "🔴"}
    rows = ""
    for c in nt.get("checks", []):
        rows += (f'<tr><td style="white-space:nowrap;">{icon.get(c["status"],"")} {c["name"]}</td>'
                 f'<td style="color:var(--txt2);font-size:13px;">{c["detail"]}</td></tr>')
    return (
        f'<div class="ovbox" style="border-left:4px solid {head_col};margin:0 0 16px;">'
        f'<div class="ovhead">🚦 No-trade check — <span style="color:{head_col};">{head_txt}</span></div>'
        f'<table class="tbl" style="margin-top:8px;"><tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">The bot sits on its hands when conditions '
        'are poor, even if a signal fires. A 🔴 pauses <b>new</b> entries this run (open positions keep their '
        'stops/targets); 🟡 means trade smaller / be selective. It never overrides the risk engine.</p></div>'
    )


def _nlp_html(scores: dict | None) -> str:
    """LLM structured news-read panel: named text scores per stock (LLM converts text → numbers)."""
    if not scores:
        return ""
    dims = [("guidance", "Guidance"), ("demand_strength", "Demand"), ("management_confidence", "Mgmt"),
            ("margin_pressure", "Margins"), ("regulatory_risk", "Reg risk"),
            ("balance_sheet_concern", "Balance sht"), ("earnings_quality_risk", "Earn qual")]

    def chip(v):
        v = int(v or 0)
        c = "var(--buy)" if v > 0 else "var(--sell)" if v < 0 else "var(--muted)"
        return f'<span style="color:{c};font-weight:600;">{v:+d}</span>'
    rows = ""
    for sym, d in sorted(scores.items(), key=lambda kv: -(kv[1].get("net") or 0)):
        net = d.get("net", 0)
        ncol = "var(--buy)" if net > 0.15 else "var(--sell)" if net < -0.15 else "var(--muted)"
        cells = "".join(f'<td style="text-align:center;">{chip(d.get(k,0))}</td>' for k, _ in dims)
        rows += (f'<tr><td><b>{sym}</b></td>'
                 f'<td style="text-align:center;color:{ncol};font-weight:700;">{net:+.2f}</td>'
                 f'{cells}'
                 f'<td style="color:var(--txt2);font-size:12px;">{d.get("note","")}</td></tr>')
    heads = "".join(f'<th style="text-align:center;" title="{lab}">{lab}</th>' for _, lab in dims)
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">🧠 AI news read '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— the LLM turns recent headlines into '
        'structured scores (−2…+2); it never decides the trade, it just feeds the meta-model</span></div>'
        '<table class="tbl" style="margin-top:8px;"><thead><tr><th>Symbol</th>'
        '<th style="text-align:center;" title="average across dimensions">Net</th>'
        f'{heads}<th>Note</th></tr></thead><tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">+ is favourable for the stock, − is a risk '
        'flag (for risk rows, − means the risk is elevated). Grounded only in the headlines shown on each card. '
        'A strongly opposing read makes the meta-model trim size. Educational; not advice.</p></div>'
    )


def _structured_html(items: list | None, top: int = 14) -> str:
    """Structured signal output: one row per actionable trade with the full signal contract."""
    if not items:
        return ""
    order = {"reject": 0, "delay": 1, "reduce": 2, "accept": 3}
    items = sorted(items, key=lambda s: (-(s.get("confidence") or 0)))
    ucol = {"low": "var(--buy)", "moderate": "var(--warn)", "high": "var(--sell)"}
    dcol = {"accept": "var(--buy)", "reduce": "var(--warn)", "delay": "var(--muted)", "reject": "var(--sell)"}
    rows = ""
    for s in items[:top]:
        rr = s.get("return_range") or {}
        up, dn = rr.get("upside_pct"), rr.get("downside_pct")
        rng = (f'<span class="buy">+{up:.0f}%</span> / <span class="sell">{dn:.0f}%</span>'
               if (up is not None and dn is not None) else "—")
        d = s.get("direction", "LONG")
        ddec = s.get("meta_decision", "accept")
        rows += (
            f'<tr><td><b>{s["symbol"]}</b></td>'
            f'<td class="{"buy" if d=="LONG" else "sell"}">{s.get("action","")}</td>'
            f'<td style="text-align:right;">{s.get("confidence","—")}</td>'
            f'<td style="text-align:right;">{rng}</td>'
            f'<td style="text-align:right;">{("%+.1f%%" % s["expected_value_pct"]) if s.get("expected_value_pct") is not None else "—"}</td>'
            f'<td style="text-align:right;">{s.get("expected_hold_days","—")}d</td>'
            f'<td style="text-align:right;">{s.get("risk_score","—")}</td>'
            f'<td style="text-align:right;color:{ucol.get(s.get("uncertainty_band"),"var(--muted)")};">{s.get("uncertainty","—")}</td>'
            f'<td style="text-align:center;color:{dcol.get(ddec,"var(--muted)")};font-weight:600;">{ddec}</td>'
            f'<td style="text-align:center;">{s.get("size_recommendation","—")}</td></tr>'
        )
    return (
        '<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">🧾 Structured signals '
        '<span style="font-weight:400;color:var(--muted);font-size:12px;">— the full signal contract per trade: '
        'confidence, expected range, risk, uncertainty and the meta verdict</span></div>'
        '<table class="tbl" style="margin-top:8px;"><thead><tr>'
        '<th>Symbol</th><th>Action</th>'
        '<th style="text-align:right;" title="confidence score 0–100">Conf</th>'
        '<th style="text-align:right;" title="target upside / stop downside">Range</th>'
        '<th style="text-align:right;" title="probability-weighted expected return">EV</th>'
        '<th style="text-align:right;" title="expected holding period (sessions)">Hold</th>'
        '<th style="text-align:right;" title="risk score 0–100 (volatility + illiquidity)">Risk</th>'
        '<th style="text-align:right;" title="uncertainty 0–100 (disagreement / mixed macro / thin liquidity)">Unc</th>'
        '<th style="text-align:center;" title="meta-model verdict">Verdict</th>'
        '<th style="text-align:center;" title="recommended size">Size</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<p style="color:var(--muted);font-size:11px;margin:10px 0 0;">EV and the range are rough, '
        'probability-weighted estimates (confidence as win-odds), not promises. High uncertainty is what makes the '
        'meta-model reduce or skip size — so a "reduce/Half" or "delay/Skip" verdict is the system being selective. '
        'Educational; not advice.</p></div>'
    )


def _macro_posture_html(mp: dict | None) -> str:
    """Macro regime → exposure panel: composite posture, exposure multiplier, and the drivers."""
    if not mp:
        return ""
    col = {"Risk-on": "var(--buy)", "Neutral": "var(--muted)", "Risk-off": "var(--sell)"}.get(mp.get("label"), "var(--muted)")
    em = mp.get("exposure_mult", 1.0)
    tilt = mp.get("cash_tilt_pct", 0)
    em_txt = (f"{em:.2f}× sizing" + (f" · ~{tilt}% more cash" if tilt else ""))
    thr = mp.get("entry_threshold")
    thr_txt = (f" · entry bar {thr}%" if thr else "")
    chips = ""
    for d in mp.get("drivers", []):
        dc = "var(--buy)" if d["score"] > 0.1 else "var(--sell)" if d["score"] < -0.1 else "var(--muted)"
        chips += (f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 9px;border-radius:999px;'
                  f'background:color-mix(in srgb,{dc} 14%,transparent);color:{dc};font-size:12px;" '
                  f'title="{d["read"]}">{d["name"]} {d["score"]:+.1f}</span>')
    # secondary regime tags (high-vol / recessionary / inflationary / liquidity-driven)
    tag_html = ""
    for t in mp.get("tags", []):
        tag_html += (f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 10px;border-radius:6px;'
                     f'background:color-mix(in srgb,#e0a82e 16%,transparent);color:#e0a82e;font-size:12px;font-weight:600;" '
                     f'title="{t["why"]}">{t["tag"]}</span>')
    tags_row = (f'<div style="margin:2px 0 8px;">{tag_html}</div>') if tag_html else ""
    # strategy bias (favoured vs caution)
    sb = mp.get("strategy_bias") or {}
    bias_html = ""
    if sb.get("favored") or sb.get("caution"):
        fav = " · ".join(sb.get("favored", []))
        cau = " · ".join(sb.get("caution", []))
        bias_html = (
            '<div style="font-size:12px;margin:6px 0 0;line-height:1.7;">'
            f'<div><span style="color:var(--buy);">▲ Favour:</span> <span style="color:var(--txt2);">{fav}</span></div>'
            f'<div><span style="color:var(--sell);">▼ Ease off:</span> <span style="color:var(--txt2);">{cau}</span></div>'
            '</div>')
    return (
        f'<div class="ovbox" style="border-left:4px solid {col};margin:0 0 16px;">'
        f'<div class="ovhead">🧭 Macro regime → exposure: <span style="color:{col};">{mp.get("label")}</span> '
        f'<span style="font-weight:400;color:var(--muted);font-size:12px;">— composite {mp.get("score"):+.2f}, '
        f'<b style="color:{col};">{em_txt}</b>{thr_txt}</span></div>'
        f'{tags_row}'
        f'<p style="color:var(--txt2);font-size:13px;margin:6px 0 8px;">{mp.get("posture","")}</p>'
        f'<div>{chips}</div>'
        f'{bias_html}'
        '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">Macro sets <b>exposure</b> and <b>strategy emphasis</b>, '
        'not direction: it scales position size and tilts which strategies to lean on — it never directly buys or sells, '
        'and the rules-based risk engine always has the final say. Hover a chip/tag for the read behind it.</p></div>'
    )


def _macro_html(m: dict | None) -> str:
    if not m:
        return ""
    def cell(label, val):
        return (f'<div class="stat"><div class="l">{label}</div>'
                f'<div class="v" style="font-size:15px;">{val}</div></div>')
    cells = ""
    if m.get("vix") is not None:
        _vt = f' ({m["vix_trend"]})' if m.get("vix_trend") else ''
        cells += cell("VIX (fear gauge)", f'{m["vix"]}{_vt}')
    if m.get("dxy") is not None:
        cells += cell("US dollar index", f'{m["dxy"]}')
    if m.get("oil") is not None:
        cells += cell("WTI crude oil", f'${m["oil"]}')
    if m.get("hy_oas") is not None:
        _ht = f' ({m["hy_trend"]})' if m.get("hy_trend") else ''
        cells += cell("Credit spread (HY)", f'{m["hy_oas"]}%{_ht}')
    if m.get("y10") is not None:
        cells += cell("10-yr yield", f'{m["y10"]}%')
    if m.get("curve") is not None:
        cells += cell("Yield curve (10y-2y)", f'{m["curve"]:+.2f}')
    if m.get("cpi_yoy") is not None:
        cells += cell("Inflation (CPI)", f'{m["cpi_yoy"]}%')
    if m.get("unemployment") is not None:
        cells += cell("Unemployment", f'{m["unemployment"]}%')
    if m.get("fed_funds") is not None:
        cells += cell("Fed funds rate", f'{m["fed_funds"]}%')
    return ('<div class="ovbox"><div class="ovhead">🌍 Macro backdrop '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">'
            '— the underlying readings feeding the posture above</span></div>'
            f'<div class="trackstats">{cells}</div></div>')


def _calendar_html(cal: dict | None) -> str:
    if not cal:
        return ""
    ew, ec = cal.get("earnings") or [], cal.get("econ") or []
    if not ew and not ec:
        return ""

    def chip(t):
        return f'<span class="chip mini">{t}</span> '
    e = "".join(chip(f'{x["symbol"]} · {"today" if x["days"] == 0 else str(x["days"]) + "d"}') for x in ew) \
        or '<span style="color:var(--muted);font-size:12px;">none in the next week</span>'
    mm = "".join(chip(f'{x["date"][5:]} · {x["name"][:30]}') for x in ec) \
        or '<span style="color:var(--muted);font-size:12px;">none flagged</span>'
    return ('<div class="ovbox" style="margin-top:14px;"><div class="ovhead">📅 Event calendar '
            '<span style="font-weight:400;color:var(--muted);font-size:12px;">— event risk: avoid fresh entries right before these</span></div>'
            f'<div style="margin:8px 0 4px;"><div class="l" style="margin-bottom:5px;">Earnings this week</div>{e}</div>'
            f'<div style="margin:10px 0 2px;"><div class="l" style="margin-bottom:5px;">Key macro releases</div>{mm}</div></div>')


def _paper_spark(history: dict | None) -> str:
    """Tiny inline SVG equity curve from the paper account's portfolio history."""
    pts = (history or {}).get("points") or []
    vals = [p["v"] for p in pts if p.get("v")]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    w, h = 320, 60
    step = w / (len(vals) - 1)
    coords = " ".join(f"{i*step:.1f},{h - (v-lo)/rng*(h-6) - 3:.1f}" for i, v in enumerate(vals))
    up = vals[-1] >= vals[0]
    col = "var(--buy)" if up else "var(--sell)"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" style="width:100%;max-width:340px;height:60px;">'
            f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="2"/></svg>')


def _system_html(sysd: dict | None) -> str:
    """Live 'under the hood' status: every feed, AI layer, engine feature, execution toggle,
    scraper, alert channel and piece of infra — with an ON/OFF state read from the real config."""
    if not sysd:
        return ""
    groups = [
        ("📡 Data feeds", sysd.get("feeds"), "Where the numbers come from."),
        ("🤖 AI layers", sysd.get("ai"), "Anthropic LLM passes (need API credits)."),
        ("⚙️ Signal engine", sysd.get("engine"), "How signals are generated + scored."),
        ("🎯 Execution", sysd.get("execution"), "What actually places/manages paper orders (mostly opt-in)."),
        ("🔎 Alt-data scrapers", sysd.get("scrapers"), "Extra inputs feeding conviction."),
        ("🔔 Alert delivery", sysd.get("delivery"), "How you get notified."),
        ("🧱 Infrastructure", sysd.get("infra"), "What hosts and rebuilds the site."),
    ]
    blocks = ""
    for title, items, lead in groups:
        if not items:
            continue
        rows = ""
        for it in items:
            on = it.get("on")
            pill = (f'<span class="syspill on">● ON</span>' if on else '<span class="syspill off">○ off</span>')
            note = f'<span class="sysnote">{it.get("note","")}</span>' if it.get("note") else ""
            rows += (f'<div class="sysrow"><span class="sysname">{it["name"]}</span>{note}{pill}</div>')
        blocks += (f'<div class="ovbox" style="margin:0 0 14px;"><div class="ovhead">{title} '
                   f'<span style="font-weight:400;color:var(--muted);text-transform:none;font-size:12px;">— {lead}</span></div>'
                   f'<div class="sysgrid">{rows}</div></div>')
    intro = ('<h2 style="margin-top:0;">System <span style="text-transform:none;font-weight:400;color:var(--muted);'
             'font-size:12px;">— what\'s wired in and running right now</span></h2>'
             '<p style="color:var(--muted);font-size:13px;margin:0 0 16px;">A live readout of every integration and '
             'feature, read straight from the current config each build. <b>● ON</b> = active this run; <b>○ off</b> = '
             'not configured or deliberately disabled. No secrets shown.</p>')
    return intro + blocks


def _news_ideas_html(ideas: list[dict] | None) -> str:
    """Server-rendered 'News-driven ideas' block: the LLM's read of recent headlines."""
    if not ideas:
        return ""
    rows = ""
    for i in ideas:
        d = i.get("direction", "")
        tone = "buy" if d == "bullish" else "sell"
        arrow = "↑" if d == "bullish" else "↓"
        rows += (f'<div class="nidea {tone}"><div class="nidea-top"><b>{i.get("ticker","")}</b> '
                 f'<span class="{tone}" style="font-weight:700;">{arrow} {d}</span> '
                 f'<span class="nidea-conf">{i.get("confidence","")} confidence</span></div>'
                 f'<div class="nidea-why">{i.get("reason","")}</div>'
                 f'<div class="nidea-src">from: “{i.get("headline","")}”</div></div>')
    return ('<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">🗞 News-driven ideas '
            '<span style="font-weight:400;color:var(--muted);text-transform:none;">— an AI read of recent '
            'headlines (sentiment, not the confluence engine)</span></div>'
            f'<div class="nideas">{rows}</div>'
            '<p style="color:var(--muted);font-size:11.5px;margin:9px 0 0;">Extracted by an LLM from recent '
            'news text — directional reads, not verified signals. Treat as leads to research. Tickers also in '
            "today's scan get a small conviction nudge.</p></div>")


def _altdata_html(snap: dict) -> str:
    """Aggregate + explain what the alt-data scrapers (SEC insiders, analyst ratings, StockTwits
    buzz) found across today's signals, and how each is meant to be read."""
    sigs = snap.get("signals", []) or []

    def _dir_tone(s):
        return "buy" if s.get("direction") != "SHORT" else "sell"

    def _rowlink(s, detail):
        nm = (s.get("name") or "")[:26]
        return (f'<tr><td><b>{s["symbol"]}</b> <span style="color:var(--muted);">{nm}</span></td>'
                f'<td><span class="{_dir_tone(s)}" style="font-weight:700;">{s.get("action","")}</span></td>'
                f'<td>{detail}</td></tr>')

    def _block(title, lead, header3, rows, empty):
        body = (f'<table class="trackrec" style="margin-top:8px;"><thead><tr><th>Ticker</th><th>Signal</th>'
                f'<th>{header3}</th></tr></thead><tbody>{rows}</tbody></table>') if rows else \
               f'<p style="color:var(--muted);font-size:13px;margin:6px 0 0;">{empty}</p>'
        return (f'<div class="ovbox" style="margin:0 0 16px;"><div class="ovhead">{title}</div>'
                f'<p style="color:var(--muted);font-size:12.5px;margin:6px 0 0;">{lead}</p>{body}</div>')

    # Insider (SEC Form 4)
    ins_rows = ""
    for s in sigs:
        i = s.get("insider") or {}
        if i.get("cluster_buy"):
            ins_rows += _rowlink(s, f'🏛 {i["buys"]} open-market purchase(s), '
                                    f'{(i.get("buy_shares") or 0):,} shares (last {i.get("last_date","")})')
    ins = _block("🏛 Insider buying (SEC Form 4)",
                 "Insiders (officers/directors) must file a Form 4 within 2 business days of trading their own "
                 "stock. Clusters of <b>open-market purchases</b> are a well-studied bullish tell — they're "
                 "spending real money, unlike option grants. We raise a long's conviction (and cut a short's) when "
                 "we see one.", "Finding", ins_rows,
                 "No insider open-market buy clusters across today's signals (most names have none on a given day).")

    # Analyst rating changes (Finnhub)
    rat_rows = ""
    for s in sigs:
        aa = (s.get("fundamentals") or {}).get("analyst_actions") or {}
        lt = aa.get("latest") or {}
        if lt.get("action") in ("up", "down"):
            arrow = "⬆" if lt["action"] == "up" else "⬇"
            rat_rows += _rowlink(s, f'{arrow} {lt.get("firm","")}: {lt.get("from","") or "?"} → '
                                    f'{lt.get("to","")} ({lt.get("date","")}) · 60d net '
                                    f'{aa.get("n_up",0)}↑/{aa.get("n_down",0)}↓')
    rat = _block("📈 Analyst rating changes (Finnhub)",
                 "Recent upgrades/downgrades and the firm behind them, over the last 60 days. A fresh upgrade is a "
                 "supportive catalyst for a long (headwind for a short); net downgrades lean the other way. It's one "
                 "input, not gospel — analysts lag as often as they lead.", "Latest action", rat_rows,
                 "No analyst rating changes in the last 60 days across today's signals (or Finnhub's free tier "
                 "didn't return them).")

    # Retail buzz (StockTwits)
    buzz_rows = ""
    for s in sigs:
        b = s.get("buzz") or {}
        if b.get("lean"):
            lean = {"bull": "Bullish", "bear": "Bearish", "mixed": "Mixed"}.get(b["lean"], b["lean"])
            buzz_rows += _rowlink(s, f'💬 {lean} — {b.get("sentiment_pct","?")}% bullish of tagged, '
                                     f'{b.get("n","?")} recent posts')
    buzz = _block("💬 Retail buzz (StockTwits)",
                  "Crowd chatter: how many recent posts mention the ticker and the Bull/Bear split among those the "
                  "author tagged. Treat it as a <b>contrarian-tinted attention gauge</b>, not a signal — it's noisy "
                  "and the crowd is often late. We weight it gently.", "Buzz", buzz_rows,
                  "No tickers cleared the buzz threshold (≥5 sentiment-tagged posts) across today's signals.")

    note = ('<p style="color:var(--muted);font-size:12px;margin-top:4px;">These feed the conviction checklist on '
            'each signal (open any card) and show as badges on the Cards/Terminal layouts. Data appears only on live '
            'runs, and is sparse by design — a quiet day here is normal, not a bug. Sources: SEC EDGAR, Finnhub, StockTwits.</p>')
    intro = ('<h2 style="margin-top:0;">Data signals <span style="text-transform:none;font-weight:400;'
             'color:var(--muted);font-size:12px;">— what the scrapers found, and how to read it</span></h2>')
    return intro + ins + rat + buzz + note


def _pairs_html(data: dict | None) -> str:
    """Render the pairs / mean-reversion diversifier tab: spread z-scores, signals, validation."""
    intro = ('<h2 style="margin-top:0;">Pairs &amp; mean-reversion '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             '— market-neutral spread bets on related names; a diversifier for trendless tape</span></h2>')
    explainer = (
        '<details class="ovbox" style="margin:0 0 16px;" open><summary style="cursor:pointer;font-weight:700;'
        'font-size:14px;list-style:none;">📘 What is pairs trading? <span style="font-weight:400;color:var(--muted);'
        'font-size:12px;">(tap to hide)</span></summary>'
        '<div style="margin-top:10px;font-size:13px;line-height:1.7;color:var(--txt2);">'
        '<p style="margin:0 0 8px;">Two stocks in the same business — say <b>Coca-Cola (KO)</b> and <b>Pepsi (PEP)</b> '
        '— normally move together. <b>Pairs trading</b> ignores whether the market goes up or down and instead bets that '
        'when the <i>gap</i> between such a pair stretches unusually wide, it will snap back to normal. You '
        '<b>buy the cheap one and short the expensive one</b> in equal dollar amounts, so you only profit from the gap '
        'closing — not from the market\'s direction. That\'s why it\'s a useful diversifier: it can work when the trend-following '
        'engine is struggling in a sideways market.</p>'
        '<p style="margin:0 0 6px;"><b>How to read the table:</b></p>'
        '<ul style="margin:0 0 8px;padding-left:18px;">'
        '<li><b>Spread z</b> — how far the gap is from normal, in standard deviations. <b>±2σ</b> = unusually stretched '
        '(actionable, marked ★). 0 = at its normal level.</li>'
        '<li><b>Signal</b> — <span class="buy">Long spread</span> = buy the first name, short the second; '
        '<span class="sell">Short spread</span> = the reverse; <b>Watch</b> = not stretched enough yet.</li>'
        '<li><b>β (beta)</b> — the hedge ratio: how many shares of the second name to trade per share of the first so the '
        'two legs cancel out market risk.</li>'
        '<li><b>Corr</b> — how tightly the two normally move together (closer to 1.0 = more reliable pair).</li>'
        '<li><b>Half-life</b> — roughly how many days the gap has historically taken to revert to normal.</li></ul>'
        '<p style="margin:0;"><b>How a trade works:</b> enter when the gap hits ±2σ, take it off as the gap reverts toward '
        '0, and stop out if it stretches past ±3σ (a sign the relationship may have broken). It\'s a paper-money '
        'diversifier here — educational, not investment advice.</p></div></details>')
    if not data or not data.get("pairs"):
        msg = (data or {}).get("note") or "No pairs qualified right now — none are stretched or stably mean-reverting."
        return (intro + explainer + '<div class="ovbox"><div class="ovhead">No active pairs</div>'
                f'<p style="color:var(--muted);font-size:13px;margin:8px 0 0;">{msg}</p>'
                '<p style="color:var(--muted);font-size:12px;margin:8px 0 0;">A pair only appears once its two legs '
                'are correlated, the spread is genuinely mean-reverting (sane half-life), and it has stretched toward '
                '±2σ. Enter at ±2σ, exit toward 0, stop beyond ±3σ.</p></div>')

    fit = data.get("regime_fit")
    fit_col = "var(--buy)" if fit else "var(--muted)"
    fit_txt = data.get("note") or ""
    banner = (f'<div class="ovbox" style="border-left:4px solid {fit_col};margin:0 0 16px;">'
              f'<div class="ovhead">Regime fit: <span style="color:{fit_col};">'
              f'{"favourable" if fit else "lower priority"}</span></div>'
              f'<p style="color:var(--txt2);font-size:13px;margin:6px 0 0;">{fit_txt}</p></div>')

    sig_style = {
        "LONG_SPREAD": ("buy", "Long spread"), "SHORT_SPREAD": ("sell", "Short spread"),
        "STOP": ("sell", "Stop / broken"), "WATCH": ("", "Watch"),
        "FLAT": ("", "At fair value"),
    }
    rows = ""
    hi = ' style="background:color-mix(in srgb,var(--accent) 7%,transparent);"'
    for p in data["pairs"]:
        cls, lab = sig_style.get(p["signal"], ("", p["signal"]))
        star = "★ " if p.get("actionable") else ""
        zc = "sell" if abs(p["z"]) >= p.get("stop_z", 3) else ("buy" if p.get("actionable") else "")
        tr_attr = hi if p.get("actionable") else ""
        rows += (
            f'<tr{tr_attr}>'
            f'<td><b>{p["a"]} / {p["b"]}</b><div style="color:var(--muted);font-size:11px;">'
            f'${p["price_a"]} vs ${p["price_b"]}</div></td>'
            f'<td class="{cls}">{star}{lab}</td>'
            f'<td style="text-align:right;" class="{zc}"><b>{p["z"]:+.2f}σ</b></td>'
            f'<td style="text-align:right;">{p["beta"]:.2f}</td>'
            f'<td style="text-align:right;">{p["corr"]:.2f}</td>'
            f'<td style="text-align:right;">{p["half_life"]:.0f}d</td>'
            f'<td style="color:var(--txt2);font-size:12px;">{p["note"]}</td></tr>'
        )
    table = (
        '<table class="tbl"><thead><tr>'
        '<th>Pair</th><th>Signal</th>'
        '<th style="text-align:right;" title="how many standard deviations the spread sits from its mean">Spread z</th>'
        '<th style="text-align:right;" title="hedge ratio: shares of B per share of A for a neutral spread">β</th>'
        '<th style="text-align:right;" title="return correlation of the two legs">Corr</th>'
        '<th style="text-align:right;" title="how fast the spread reverts to its mean">Half-life</th>'
        '<th>Read</th></tr></thead><tbody>' + rows + '</tbody></table>'
    )
    legend = ('<p style="color:var(--muted);font-size:12px;margin:12px 0 0;">'
              '★ = actionable now (|z| ≥ 2σ). Enter at ±2σ, exit as the spread reverts toward 0, '
              'stop if it stretches past ±3σ (the relationship may have broken). Dollar-neutral: trade β shares of '
              'the second leg per share of the first. Diversifier only — not a core directional position. '
              'Paper money / educational; not investment advice.</p>')
    return intro + explainer + banner + table + legend


def _risk_html(risk: dict | None) -> str:
    """Book-level risk-engine status banner: state, drawdown, day P&L, and any active limits."""
    if not risk or not risk.get("enabled"):
        return ""
    state = risk.get("state", "normal")
    palette = {
        "normal": ("var(--buy)", "🟢", "Normal", "Within all book-level risk limits."),
        "derisk": ("var(--warn)", "🟡", "De-risking", "Drawdown elevated — new positions sized at half."),
        "halt":   ("var(--sell)", "🔴", "Halted", "A book-level limit was hit — no new positions this session."),
        "killed": ("var(--sell)", "🛑", "Kill switch", "Trading paused after repeated run failures."),
        "off":    ("var(--muted)", "⚪", "Off", "Risk engine not evaluated this run."),
    }
    col, dot, lab, default_msg = palette.get(state, palette["normal"])
    dd = risk.get("drawdown_pct")
    dpl = risk.get("day_pl_pct")
    mpp = risk.get("max_position_pct")
    bits = []
    if dd is not None:
        bits.append(f'<span title="peak-to-now equity drawdown">Drawdown <b>{dd:.1f}%</b></span>')
    if dpl is not None:
        bits.append(f'<span title="today\'s P&amp;L vs prior close">Day P&amp;L <b>{dpl:+.1f}%</b></span>')
    if mpp is not None:
        bits.append(f'<span title="max single-position size">Concentration cap <b>{mpp:.0f}%</b></span>')
    metrics = ' &nbsp;·&nbsp; '.join(bits)
    msgs = (risk.get("reasons") or []) + (risk.get("warnings") or [])
    msg = " · ".join(msgs) if msgs else default_msg
    return (
        f'<div class="ovbox" style="border-left:4px solid {col};margin:0 0 16px;">'
        f'<div class="ovhead" style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:15px;">{dot}</span>'
        f'<span>Portfolio risk engine — <span style="color:{col};">{lab}</span></span></div>'
        f'<div style="font-size:12px;color:var(--muted);margin:6px 0 8px;">{metrics}</div>'
        f'<p style="color:var(--txt2);font-size:13px;margin:0;">{msg}</p></div>'
    )


def _paper_html(p: dict | None) -> str:
    """Server-rendered REAL paper-account block (opt-in). Honest, fills-based — distinct from
    the hypothetical tracker."""
    if not p:
        return ""
    intro = ('<h2 style="margin-top:0;">Paper account <span style="text-transform:none;font-weight:400;'
             'color:var(--muted);font-size:12px;">— a real, fills-based record from an Alpaca paper account</span></h2>')
    if not p.get("enabled"):
        return (intro + '<div class="ovbox"><div class="ovhead">Auto paper-trading is off.</div>'
                f'<p style="color:var(--muted);font-size:13px;margin:8px 0 0;">{p.get("reason","Set PAPER_TRADE=true to enable.")}'
                ' Once enabled, fresh High-conviction signals are auto-submitted to your <b>paper</b> account '
                'as bracket orders, and this page shows the real equity, P&amp;L and open positions.</p></div>')

    def tile(label, value, tone="", sub=""):
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v {tone}">{value}</div>{sub_html}</div>'

    dp = p.get("day_pl", 0) or 0
    tone = "buy" if dp > 0 else "sell" if dp < 0 else ""
    rz = p.get("realized") or {}
    wr = rz.get("win_rate")
    wr_v = "—" if wr is None else f"{wr:.0f}%"
    tiles = (tile("Equity", f"${p.get('equity',0):,.0f}", "", "paper account")
             + tile("Day P&amp;L", f"${dp:+,.0f}", tone, f"{p.get('day_pl_pct',0):+.2f}%")
             + tile("Open positions", str(p.get("n_open", 0)), "", f"of {p.get('tracked_total', 0)} tracked")
             + tile("Realized win rate", wr_v, "", f"over {rz.get('n_trades', 0)} closed trades"))
    spark = _paper_spark(p.get("history"))
    spark_html = f'<div style="margin:6px 0 16px;">{spark}</div>' if spark else ""

    # open positions table (live unrealized P&L)
    rows = ""
    for pos in p.get("positions", []):
        pl = pos.get("unrealized_pl", 0) or 0
        c = "buy" if pl > 0 else "sell" if pl < 0 else ""
        rows += (f'<tr><td><b>{pos["symbol"]}</b></td><td>{pos.get("side","") or ""}</td>'
                 f'<td style="text-align:right;">{pos.get("qty","")}</td>'
                 f'<td style="text-align:right;">${pos.get("avg_entry",0):,.2f}</td>'
                 f'<td style="text-align:right;">${pos.get("price",0):,.2f}</td>'
                 f'<td style="text-align:right;" class="{c}">${pl:+,.0f} ({pos.get("unrealized_plpc",0):+.1f}%)</td></tr>')
    postable = (f'<table class="trackrec"><thead><tr><th>Symbol</th><th>Side</th><th style="text-align:right;">Qty</th>'
                f'<th style="text-align:right;">Entry</th><th style="text-align:right;">Last</th>'
                f'<th style="text-align:right;">Unrealized</th></tr></thead><tbody>{rows}</tbody></table>'
                if rows else '<p style="color:var(--muted);font-size:13px;">No open paper positions right now.</p>')

    # --- realized performance: closed round-trips matched from actual fills ---
    def _pct(v, sign=True):
        if v is None:
            return "—"
        return f"{'+' if (sign and v > 0) else ''}{v}%"
    rstats = ""
    if rz.get("n_trades"):
        tp = rz.get("total_pl", 0) or 0
        rstats = ('<div class="kpis" style="margin-top:4px;">'
                  + tile("Closed trades", str(rz.get("n_trades", 0)), "", "matched round-trips")
                  + tile("Avg return / trade", _pct(rz.get("avg_return_pct")),
                         "buy" if (rz.get("avg_return_pct") or 0) > 0 else "sell" if (rz.get("avg_return_pct") or 0) < 0 else "",
                         "per closed trade")
                  + tile("Avg win", _pct(rz.get("avg_win")), "buy", "winners only")
                  + tile("Avg loss", _pct(rz.get("avg_loss")), "sell", "losers only")
                  + tile("Realized P&amp;L", f"${tp:+,.0f}", "buy" if tp > 0 else "sell" if tp < 0 else "", "all closed trades")
                  + '</div>')
        trows = ""
        for t in rz.get("recent", []):
            ret = t.get("return_pct")
            c = "buy" if (ret or 0) > 0 else "sell" if (ret or 0) < 0 else ""
            when = (t.get("exit_time") or "")[:10]
            pl = t.get("pl", 0) or 0
            trows += (f'<tr><td><b>{t.get("symbol","")}</b></td><td>{t.get("direction","")}</td>'
                      f'<td style="text-align:right;">{t.get("qty","")}</td>'
                      f'<td style="text-align:right;">${t.get("entry_price",0):,.2f}</td>'
                      f'<td style="text-align:right;">${t.get("exit_price",0):,.2f}</td>'
                      f'<td style="text-align:right;" class="{c}">{_pct(ret)}</td>'
                      f'<td style="text-align:right;" class="{c}">${pl:+,.0f}</td>'
                      f'<td style="text-align:right;color:var(--muted);">{when}</td></tr>')
        rtable = (f'<table class="trackrec"><thead><tr><th>Symbol</th><th>Side</th>'
                  f'<th style="text-align:right;">Qty</th><th style="text-align:right;">Entry</th>'
                  f'<th style="text-align:right;">Exit</th><th style="text-align:right;">Return</th>'
                  f'<th style="text-align:right;">P&amp;L</th><th style="text-align:right;">Closed</th>'
                  f'</tr></thead><tbody>{trows}</tbody></table>')
        realized_html = ('<h3 style="font-size:15px;margin:18px 0 8px;">Realized performance '
                         '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
                         '— closed round-trips matched from real fills</span></h3>'
                         + rstats + rtable)
    else:
        realized_html = ('<h3 style="font-size:15px;margin:18px 0 8px;">Realized performance</h3>'
                         '<p style="color:var(--muted);font-size:13px;">No closed round-trips yet — '
                         'win rate and per-trade return appear here once paper positions are opened and exited.</p>')

    sub = []
    if p.get("submitted_now"):
        sub.append("Opened this run: " + ", ".join(f'{r["symbol"]} ({r["action"]}, {r["qty"]}sh)' for r in p["submitted_now"]))
    if not p.get("market_open"):
        sub.append("Market is closed — orders fire during market hours.")
    for n in p.get("notes", []):
        sub.append(n)
    subline = ('<p style="color:var(--muted);font-size:12px;margin-top:12px;">' + " · ".join(sub) + "</p>") if sub else ""

    return (intro
            + '<p style="color:var(--muted);font-size:13px;margin:0 0 14px;">These are <b>real fills</b> on a '
            'paper account — actual entry prices, slippage and timing — so they reflect how the calls truly play out, '
            'unlike the hypothetical tracker. Not investment advice; paper money only.</p>'
            + _risk_html(p.get("risk"))
            + f'<div class="kpis">{tiles}</div>{spark_html}'
            + '<h3 style="font-size:15px;margin:6px 0 8px;">Open positions</h3>' + postable
            + realized_html + subline)


def _attribution_html(rep: list[dict] | None) -> str:
    """Panel: which conviction checks actually predicted wins (pass vs fail win rate)."""
    intro = ('<h3 style="font-size:15px;margin:18px 0 8px;">Which checks earn their keep '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             '— win rate when each conviction check passed vs failed, on resolved calls</span></h3>')
    if not rep:
        return intro + ('<p style="color:var(--muted);font-size:13px;">Still accruing — per-check '
                        'win rates appear here once enough tracked calls resolve.</p>')
    rows = ""
    for r in rep:
        wp = "—" if r["win_rate_pass"] is None else f'{r["win_rate_pass"]:.0f}%'
        wf = "—" if r["win_rate_fail"] is None else f'{r["win_rate_fail"]:.0f}%'
        edge = r["edge"]
        ec = "buy" if (edge or 0) > 0 else "sell" if (edge or 0) < 0 else ""
        es = "—" if edge is None else f'{"+" if edge > 0 else ""}{edge:.0f} pts'
        rows += (f'<tr><td>{r["label"]}</td>'
                 f'<td style="text-align:right;">{r["n_pass"]}/{r["n_fail"]}</td>'
                 f'<td style="text-align:right;">{wp}</td><td style="text-align:right;">{wf}</td>'
                 f'<td style="text-align:right;" class="{ec}">{es}</td></tr>')
    return (intro + '<table class="trackrec"><thead><tr><th>Check</th>'
            '<th style="text-align:right;">Pass/Fail n</th><th style="text-align:right;">Win% pass</th>'
            '<th style="text-align:right;">Win% fail</th><th style="text-align:right;">Edge</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>')


def _track_html(track: dict | None) -> str:
    """Server-rendered track-record block (works without JS)."""
    if not track:
        return ""
    def stat(label, value, cls=""):
        return (f'<div class="stat"><div class="l">{label}</div>'
                f'<div class="v {cls}">{value}</div></div>')
    def _pct(v):
        return "—" if v is None else f"{'+' if v > 0 else ''}{v}%"
    wr = "—" if track["win_rate"] is None else f"{track['win_rate']}%"
    stats = (
        stat("Calls advised", track["advised"]) +
        stat("Resolved", track["resolved"]) +
        stat("Still open", track["open"]) +
        stat("Win rate", wr) +
        stat("Expectancy", _pct(track.get("expectancy")), "buy" if (track.get("expectancy") or 0) > 0 else ("sell" if (track.get("expectancy") or 0) < 0 else "")) +
        stat("Avg win", _pct(track.get("avg_win")), "win") +
        stat("Avg loss", _pct(track.get("avg_loss")), "loss")
    )
    # per-direction / per-conviction breakdown (the live-performance read, as data accrues)
    def _brk(title, d, keys):
        cells = ""
        for k in keys:
            g = (d or {}).get(k) or {}
            wrk = "—" if g.get("win_rate") is None else f"{g['win_rate']}%"
            cells += (f'<tr><td>{k}</td><td style="text-align:right;">{g.get("n",0)}</td>'
                      f'<td style="text-align:right;">{wrk}</td>'
                      f'<td style="text-align:right;">{_pct(g.get("avg_return"))}</td></tr>')
        return (f'<div class="sech" style="margin-top:14px;">{title}</div>'
                '<table class="trackrec"><thead><tr><th>'+title.split()[-1]+'</th>'
                '<th style="text-align:right;">Resolved</th><th style="text-align:right;">Win rate</th>'
                '<th style="text-align:right;">Avg return</th></tr></thead><tbody>'+cells+'</tbody></table>')
    breakdown = ""
    if track.get("resolved"):
        breakdown = (_brk("By direction", track.get("by_direction"), ["LONG", "SHORT"]) +
                     _brk("By conviction", track.get("by_conviction"), ["High", "Medium", "Low"]))
        _byreg = track.get("by_regime") or {}
        if _byreg:
            breakdown += _brk("By macro regime", _byreg, list(_byreg.keys()))
        tv = track.get("by_tv") or {}
        if (tv.get("agree", {}) or {}).get("n") or (tv.get("not_agree", {}) or {}).get("n"):
            def _tvrow(label, g):
                g = g or {}
                wrk = "—" if g.get("win_rate") is None else f"{g['win_rate']}%"
                return (f'<tr><td>{label}</td><td style="text-align:right;">{g.get("n",0)}</td>'
                        f'<td style="text-align:right;">{wrk}</td>'
                        f'<td style="text-align:right;">{_pct(g.get("avg_return"))}</td></tr>')
            breakdown += ('<div class="sech" style="margin-top:14px;">Does TradingView help? '
                          '<span style="text-transform:none;color:var(--muted);font-weight:400;">'
                          '— win rate when the TradingView cross-check agreed vs didn\'t</span></div>'
                          '<table class="trackrec"><thead><tr><th>TradingView</th>'
                          '<th style="text-align:right;">Resolved</th><th style="text-align:right;">Win rate</th>'
                          '<th style="text-align:right;">Avg return</th></tr></thead><tbody>'
                          + _tvrow("Agreed", tv.get("agree")) + _tvrow("Disagreed / mixed", tv.get("not_agree"))
                          + '</tbody></table>')
    rows = ""
    icon = {"win": '<span class="win">✅ hit target</span>',
            "loss": '<span class="loss">❌ hit stop</span>',
            "expired": '<span class="exp">⌛ expired</span>'}
    for t in track.get("recent", []):
        ret = t.get("return_pct")
        ret_s = "—" if ret is None else f"{'+' if ret > 0 else ''}{ret}%"
        rows += (f"<tr><td>{t.get('symbol','')}</td><td>{t.get('advised_date','')}</td>"
                 f"<td>{icon.get(t.get('status'), t.get('status',''))}</td>"
                 f"<td>{ret_s}</td><td>{t.get('days_held','—')}d</td></tr>")
    if rows:
        table = (f'<table class="trackrec"><tr><th>Stock</th><th>Advised</th><th>Outcome</th>'
                 f'<th>Return</th><th>Held</th></tr>{rows}</table>')
    elif track.get("advised"):
        table = (f'<p style="color:var(--muted);font-size:13px;">{track["advised"]} call'
                 f'{"s" if track["advised"] != 1 else ""} logged and still open — results appear here as each '
                 'hits its target or stop (usually within a few days).</p>')
    else:
        table = ('<p style="color:var(--muted);font-size:13px;">Building your track record — every BUY the screen '
                 'flags gets logged as it runs each weekday, and the first resolved results land within a day or two '
                 'as trades play out. Nothing to show yet.</p>')
    return f"""
  <div class="track">
    <h2 style="border:0;padding:0;">📊 Track record — how past BUY calls have done</h2>
    <p style="color:var(--muted);font-size:13px;margin:2px 0 0;">Every BUY the tool flags is logged, then
    checked against real prices: did it reach its target (✅) or hit its stop first (❌)? This builds up
    over time into an honest read on how reliable the calls are. It's a hypothetical record — no fees or
    slippage — so treat it as a rough guide, not a brokerage statement.</p>
    <div class="trackstats">{stats}</div>
    {table}
    {breakdown}
  </div>"""


def render_html(snap: dict) -> str:
    data_json = json.dumps(snap)
    mode = snap["mode"]
    mode_note = {
        "LIVE": "Live account data. Real money is at risk if you act on these.",
        "PAPER": "Alpaca paper data and account.",
        "SYNTHETIC": "Synthetic data — NOT real prices or news. Add Alpaca keys for the real thing.",
    }[mode]
    track_html = _track_html(snap.get("track"))
    if track_html:
        try:
            import attribution
            track_html += _attribution_html(attribution.report())
        except Exception:  # noqa: BLE001 - attribution is additive; never break the build
            pass
    _paper_acct = snap.get("paper_acct")
    paper_html = _paper_html(_paper_acct)
    paper_nav = '<button data-page="paper">Paper account</button>' if _paper_acct else ''
    paper_section = f'<section class="page" id="page-paper">{paper_html}</section>' if _paper_acct else ''
    _pairs_data = snap.get("pairs") or {}
    pairs_html = _pairs_html(_pairs_data)
    pairs_nav = '<button data-page="pairs">Pairs</button>' if _pairs_data.get("pairs") else ''
    altdata_html = _altdata_html(snap)
    news_ideas_html = _news_ideas_html(snap.get("news_ideas"))
    system_html = _system_html(snap.get("system"))
    regime_html = _regime_html(snap.get("regime"))
    _pd = snap.get("price_drops") or []
    pdrop_html = (f' &middot; <span style="color:var(--muted);" title="{(" | ".join(_pd))[:300].replace(chr(34), chr(39))}">'
                  f'{len(_pd)} dropped (bad feed price)</span>') if _pd else ""
    kpi_html = _kpi_html(snap.get("regime"), snap)
    bento_home_html = _bento_home(snap)
    _brief = (snap.get("market_brief") or "").strip()
    brief_html = (f'<div class="ai-box" style="margin:2px 0 18px;line-height:1.6;">'
                  f'<span class="ai-h">🧠 Market brief</span> {_brief}</div>') if _brief else ""
    _changes = snap.get("changes") or []
    changes_html = ((f'<div class="ai-box" style="margin:0 0 18px;border-color:color-mix(in srgb,#e0a82e 32%,transparent);'
                     f'background:color-mix(in srgb,#e0a82e 11%,transparent);">'
                     f'<span class="ai-h" style="color:#e0a82e;">⚡ What changed since last build</span>'
                     f'<ul style="margin:7px 0 0;padding-left:18px;line-height:1.8;">'
                     + "".join(f"<li>{_c}</li>" for _c in _changes) + "</ul></div>") if _changes else "")
    momentum_html = (_momentum_bt_html(snap.get("momentum_bt"))
                     + _walkforward_html(snap.get("walkforward"))
                     + _momentum_html(snap.get("momentum") or []))
    allweather_html = _allweather_html(snap.get("allweather"))
    portfolio_html = (_ranked_html(snap.get("ranked")) + _structured_html(snap.get("structured"))
                      + _nlp_html(snap.get("nlp_scores")) + _portfolio_html(snap.get("portfolio")))
    ipo_html = _ipo_html(snap.get("ipos") or [], snap.get("ipo_news") or [])
    sectors_html = _sectors_html(snap.get("sectors"))
    macro_html = (_notrade_html(snap.get("notrade"))
                  + _macro_posture_html(snap.get("macro_posture"))
                  + _macro_html(snap.get("macro")) + _calendar_html(snap.get("calendar")))
    dh = snap.get("data_health")
    if not dh:
        health_html = ""
    else:
        n_err = dh.get("n_err", 0)
        n_warn = dh.get("n_warn", 0)
        if n_err:
            tip = " | ".join(dh.get("errors", []))[:400].replace('"', "'")
            health_html = (f' &middot; <span style="color:var(--sell);" title="{tip}">'
                           f'data check ⚠ {n_err} to review</span>')
        elif n_warn:
            tip = ("Extreme but likely-real movers (volatile names): "
                   + " | ".join(dh.get("warnings", []))[:380]).replace('"', "'")
            health_html = (f' &middot; <span style="color:#2ea043;" title="{tip}">data check ✓</span>'
                           f' <span style="color:var(--muted);font-size:12px;" title="{tip}">'
                           f'· {n_warn} volatile</span>')
        else:
            health_html = (f' &middot; <span style="color:#2ea043;" '
                           f'title="{dh.get("checks",0)} integrity checks passed">data check ✓</span>')
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Trading Signals Dashboard</title>
<meta name="theme-color" content="#0d1117">
<link rel="manifest" href="manifest.webmanifest">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Signal Desk">
<link rel="apple-touch-icon" href="icon-180.png">
<link rel="icon" type="image/png" href="icon-192.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://s3.tradingview.com/tv.js"></script>
<script src="chart_engine.js"></script>
<style>
  /* Light "Capital IQ Pro" palette is the default; dark is a toggle. */
  :root {{ --bg:#f5f7fa; --card:#ffffff; --line:#e4e8ed; --txt:#16202c;
    --muted:#5b6776; --txt2:#3d4757; --buy:#0a7d44; --sell:#d1242f; --hold:#0b5cad; --flat:#6b7785;
    --short:#c2410c; --watch:#475569; --exit:#b45309; --avoid:#6b7280; --warn:#b7791f;
    --accent:#0b5cad; --grid:rgba(120,130,145,0.16); --cross:rgba(60,70,85,0.4);
    --inset:#f1f4f8; --hover:#eef2f7; --ring:rgba(11,92,173,.40);
    --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
    --hud-edge:color-mix(in srgb,var(--accent) 22%,var(--line));
    --shadow:0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
    --shadow-lg:0 6px 20px rgba(16,24,40,0.10); }}
  html[data-theme="dark"] {{ --bg:#090d12; --card:#0f151c; --line:#222b35; --txt:#e6edf3;
    --muted:#8b97a6; --txt2:#c2cad4; --buy:#2ea043; --sell:#f85149; --hold:#58a6ff; --flat:#6e7681;
    --short:#fb7185; --watch:#94a3b8; --exit:#d29922; --avoid:#6e7681; --warn:#e0a82e;
    --accent:#58a6ff; --grid:rgba(42,52,65,0.55); --cross:rgba(139,151,166,0.45);
    --inset:#1c2530; --hover:#243042; --ring:rgba(88,166,255,.45);
    --shadow:0 1px 2px rgba(0,0,0,0.4); --shadow-lg:0 8px 28px rgba(0,0,0,0.5); }}
  * {{ box-sizing:border-box; }}
  /* ---- global polish: motion, focus, numerals, scrollbars ---- */
  button, select, .card, .wl, summary, .tabs button, .ctlgrp button, .ctlbtn, .tc-seg button {{
    transition:background-color .15s ease, border-color .15s ease, color .15s ease,
               transform .15s ease, box-shadow .15s ease; }}
  button {{ font-family:inherit; }}
  :focus-visible {{ outline:2px solid var(--ring); outline-offset:2px; border-radius:6px; }}
  a {{ color:var(--accent); text-underline-offset:2px; }}
  .px, .stat .v, .kv span:last-child, .trackrec td, .secpct, .readout .rprice,
  .wl-px, .wl-chg, .convbadge {{ font-variant-numeric:tabular-nums; font-feature-settings:'tnum' 1; }}
  /* HUD: monospace, tabular data figures */
  .kpi-v, .stat .v, .px, .wl-px, .convbadge, .num {{ font-family:var(--mono); }}
  ::-webkit-scrollbar {{ width:10px; height:10px; }}
  ::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:6px; border:2px solid transparent;
    background-clip:padding-box; }}
  ::-webkit-scrollbar-thumb:hover {{ background:var(--muted); background-clip:padding-box; }}
  ::-webkit-scrollbar-track {{ background:transparent; }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition:none !important; animation:none !important; scroll-behavior:auto !important; }}
    .card:hover {{ transform:none; }} }}
  html, body {{ max-width:100%; overflow-x:hidden; }}
  body {{ margin:0; font:15px/1.5 'Inter',-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--txt);
    -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale; text-rendering:optimizeLegibility; }}
  .wrap {{ width:100%; max-width:1480px; margin:0 auto;
    padding:0 max(24px, env(safe-area-inset-right)) calc(60px + env(safe-area-inset-bottom)) max(24px, env(safe-area-inset-left)); }}
  .grid-stack {{ width:100%; }}
  h1 {{ font-size:25px; font-weight:800; letter-spacing:-.015em; margin:0 0 5px; }}
  h2 {{ font-size:13px; margin:30px 0 12px; color:var(--muted); font-weight:700;
    text-transform:uppercase; letter-spacing:.06em; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:6px; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:12px; font-weight:600; }}
  .m-LIVE {{ background:#5a1e1e; color:#ff9b9b; }}
  .m-PAPER {{ background:#15361f; color:#7ee2a0; }}
  .m-SYNTHETIC {{ background:#3a2e12; color:#e8c878; }}
  .note {{ color:var(--muted); font-size:13px; margin:10px 0 8px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr));
    gap:14px; }}
  .card {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:7px;
    padding:16px; cursor:pointer; box-shadow:var(--shadow); }}
  .card:hover {{ border-color:color-mix(in srgb, var(--accent) 45%, var(--line));
    transform:translateY(-2px); box-shadow:var(--shadow-lg); }}
  .ladder {{ margin-top:12px; border:0.5px solid var(--line); border-radius:8px; overflow:hidden; }}
  .lad-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:6px 11px; font-size:13px; }}
  .lad-row > span:first-child {{ color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; }}
  .lad-row > span:last-child {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .lad-row em {{ font-style:normal; font-size:11px; font-weight:500; margin-left:7px; }}
  .lad-row.ent {{ background:var(--inset); }}
  .lad-row.tgt > span:last-child, .lad-row.tgt em {{ color:var(--buy); }}
  .lad-row.stp > span:last-child, .lad-row.stp em {{ color:var(--sell); }}
  .lad-rr {{ padding:5px 11px; border-top:0.5px solid var(--line); font-size:11px; color:var(--muted); text-align:right; }}
  .card-warn {{ margin-top:9px; font-size:11.5px; color:var(--sell); font-weight:500; }}
  /* scraped alt-data badges (insider / analyst rating / retail buzz) */
  .altrow {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .altpill {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:7px;
    border:1px solid currentColor; background:color-mix(in srgb, currentColor 10%, transparent);
    display:inline-flex; align-items:center; gap:4px; cursor:help; line-height:1.3; }}
  .bbalt {{ margin-top:4px; font-size:10.5px; font-weight:700; display:flex; gap:9px; flex-wrap:wrap; }}
  .bbalt span {{ cursor:help; }}
  .conc-warn {{ margin:0 0 16px; padding:10px 14px; font-size:12.5px; line-height:1.5; border-radius:10px;
    background:color-mix(in srgb, #b8860b 12%, transparent); border:1px solid color-mix(in srgb, #b8860b 38%, transparent);
    color:var(--txt); cursor:help; }}
  .hcell {{ cursor:help; text-decoration:underline dotted var(--muted); text-underline-offset:3px;
    text-decoration-thickness:1px; }}
  .hint {{ cursor:help; }}
  #tip {{ position:fixed; z-index:9999; display:none; max-width:300px; padding:9px 12px;
    background:var(--card); color:var(--txt); border:1px solid var(--line);
    border-radius:8px; box-shadow:var(--shadow-lg); font-size:12px; line-height:1.5;
    pointer-events:none; }}
  #newbuild {{ position:fixed; left:50%; transform:translateX(-50%); bottom:18px; z-index:9998;
    background:var(--accent); color:#fff; border:0; border-radius:999px; cursor:pointer;
    padding:9px 16px; font-size:13px; font-weight:600; box-shadow:var(--shadow-lg); }}
  #newbuild:hover {{ filter:brightness(1.08); }}
  /* ---- alternate layouts ---- */
  .cat-chip {{ margin-top:10px; display:inline-block; font-size:11.5px; font-weight:600;
    color:#b8860b; background:color-mix(in srgb, #e0a82e 16%, transparent);
    border:1px solid color-mix(in srgb, #e0a82e 36%, transparent); padding:3px 9px; border-radius:999px; }}
  .ai-tag {{ color:var(--accent); font-weight:700; }}
  .ai-box {{ margin-top:10px; padding:9px 11px; border-radius:8px; font-size:12px; line-height:1.5;
    color:var(--txt); background:color-mix(in srgb, #9b59b6 12%, transparent);
    border:1px solid color-mix(in srgb, #9b59b6 32%, transparent); }}
  .ai-h {{ color:#9b59b6; font-weight:700; margin-right:5px; }}
  .tv-chip {{ margin-top:8px; display:inline-block; font-size:11.5px; font-weight:600;
    color:var(--accent); background:color-mix(in srgb, var(--accent) 12%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 30%, transparent); padding:3px 9px; border-radius:999px; }}
  .secmix {{ display:flex; flex-direction:column; gap:5px; margin:4px 0 8px; }}
  .secrow {{ display:flex; align-items:center; gap:10px; font-size:12px; }}
  .secname {{ width:120px; color:var(--muted); }}
  .secbarwrap {{ flex:1; height:8px; background:var(--inset); border-radius:4px; overflow:hidden; }}
  .secbarfill {{ height:100%; }}
  .secval {{ width:80px; text-align:right; font-variant-numeric:tabular-nums; }}
  .card-spark {{ margin:8px 0 2px; }}
  .card-spark svg {{ width:100% !important; height:42px; display:block; opacity:.9; }}
  .mono2 {{ display:inline-flex; align-items:center; justify-content:center; border-radius:5px;
    color:#fff; font-weight:700; overflow:hidden; position:relative; flex:none; }}
  .mono2 img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#fff; }}
  .bbwrap {{ background:#000; border:1px solid #2a2a17; border-radius:8px; overflow:hidden;
    font-family:ui-monospace,Menlo,Consolas,monospace; }}
  .bbhead {{ display:flex; align-items:center; gap:18px; padding:8px 12px; background:#13130a;
    border-bottom:1px solid #2a2a17; font-size:11px; color:#8a8a6a; letter-spacing:1px;
    white-space:nowrap; overflow-x:auto; }}
  .bbtitle {{ color:#e8a33d; font-weight:700; }} .bbst b {{ color:#e8a33d; }}
  .bbclock {{ margin-left:auto; color:#5a5a45; }}
  .bbgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:1px; background:#2a2a17; }}
  .bbtile {{ background:#000; padding:9px 11px; cursor:pointer; }}
  .bbtile:hover {{ background:#0c0c06; }}
  .bbtop {{ display:flex; align-items:center; gap:7px; }}
  .bbsym {{ color:#fff; font-weight:700; }} .bbact {{ margin-left:auto; font-size:11px; font-weight:700; }}
  .bbpx {{ color:#fff; font-size:18px; font-weight:700; margin:6px 0 2px; font-variant-numeric:tabular-nums; }}
  .bbmeta {{ color:#8a8a6a; font-size:10.5px; }}
  .bblv {{ color:#8a8a6a; font-size:10.5px; margin-top:3px; font-variant-numeric:tabular-nums; }}
  .bbtv {{ color:#8a8a6a; font-size:10px; margin-top:2px; letter-spacing:.02em; }}
  .gtv {{ color:var(--muted); font-size:10px; margin-top:2px; }}
  .tvwrap {{ position:relative; width:100%; max-width:1100px; padding-bottom:min(56.25%, 620px); height:0;
    border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#000; }}
  .tvwrap iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  /* compact Bloomberg live widget on the Signals page */
  .tvwidget {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:10px 14px; margin:0 0 16px; max-width:460px; box-shadow:var(--shadow); }}
  .tvwidget summary {{ cursor:pointer; font-weight:700; font-size:13px; display:flex; align-items:center;
    gap:8px; list-style:none; }}
  .tvwidget summary::-webkit-details-marker {{ display:none; }}
  .tvwidget .tvw-open {{ margin-left:auto; font-size:11.5px; font-weight:600; color:var(--accent); text-decoration:none; }}
  .tvwidget .tvw-open:hover {{ text-decoration:underline; }}
  .tvw-frame {{ position:relative; width:100%; padding-bottom:56.25%; height:0; margin-top:10px;
    border-radius:8px; overflow:hidden; background:#000; }}
  .tvw-frame iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  .lanes {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; align-items:start; }}
  .lanehd {{ font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }}
  .lcard {{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--muted);
    border-radius:8px; padding:9px 10px; margin-bottom:8px; cursor:pointer; }}
  .lcard:hover {{ background:var(--hover); }}
  .lcard-t {{ display:flex; align-items:center; gap:7px; }} .lsym {{ font-weight:600; }}
  .lconv {{ margin-left:auto; color:var(--muted); font-size:12px; }} .lsub {{ color:var(--muted); font-size:11px; margin-top:3px; }}
  .gauges {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(100px,1fr)); gap:16px; }}
  .gauge {{ text-align:center; cursor:pointer; }}
  .gsvg {{ width:64px; height:64px; display:block; margin:0 auto; }}
  .gnum {{ fill:var(--txt); font-size:17px; font-weight:700; font-family:inherit; }}
  .glab {{ display:flex; align-items:center; justify-content:center; gap:5px; margin-top:7px; font-weight:600; font-size:13px; }}
  .gact {{ font-size:11px; }}
  .feedwrap {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:2px 14px; }}
  .feeditem {{ display:flex; align-items:center; gap:11px; padding:11px 0; border-bottom:1px solid var(--line); cursor:pointer; }}
  .feeditem:last-child {{ border-bottom:0; }} .feedtxt {{ font-size:13px; flex:1; min-width:0; }}
  .feedsub {{ color:var(--muted); font-size:11px; margin-top:2px; }}
  .feedspark {{ flex:none; opacity:.9; }}
  .bento {{ display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:10px; }}
  .bento-regime {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px;
    grid-row:span 2; display:flex; flex-direction:column; justify-content:center; }}
  .bento-feat {{ grid-column:span 2; background:var(--card); border:1px solid var(--line); border-left:3px solid var(--buy);
    border-radius:10px; padding:12px; display:flex; align-items:center; gap:10px; cursor:pointer; }}
  .bento-tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px;
    display:flex; align-items:center; gap:8px; cursor:pointer; }}
  .blab {{ font-size:10px; text-transform:uppercase; color:var(--muted); letter-spacing:.04em; }}
  .bval {{ font-size:20px; font-weight:700; }} .btk {{ font-weight:600; font-size:13px; }}
  .mag {{ display:grid; grid-template-columns:1.5fr 1fr; gap:12px; align-items:start; }}
  .mag-hero {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:13px; cursor:pointer; }}
  .mag-hero-t {{ display:flex; align-items:center; gap:9px; margin-bottom:6px; }}
  .mag-side {{ display:flex; flex-direction:column; gap:6px; }}
  .magrow {{ display:flex; align-items:center; gap:8px; background:var(--card); border:1px solid var(--line);
    border-radius:8px; padding:7px 9px; cursor:pointer; }}
  .magrk {{ color:var(--muted); font-size:11px; width:14px; }} .magsym {{ font-weight:600; }} .magc {{ margin-left:auto; font-size:12px; }}
  .tktape {{ overflow:hidden; background:var(--inset); border:1px solid var(--line); border-radius:8px; white-space:nowrap; }}
  .tktape-in {{ display:inline-block; padding:8px 0; animation:tkscroll 45s linear infinite; }}
  .tkitem {{ margin:0 18px; font-variant-numeric:tabular-nums; font-size:13px; }}
  @keyframes tkscroll {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
  .tkbody {{ margin-top:10px; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  .tkrow {{ display:flex; align-items:center; gap:10px; padding:9px 13px; border-bottom:1px solid var(--line); cursor:pointer; }}
  .tkrow:last-child {{ border-bottom:0; }} .tkrow:hover {{ background:var(--hover); }}
  .tksym {{ font-weight:600; width:58px; }} .tkpx {{ width:88px; font-variant-numeric:tabular-nums; }}
  .tkspark {{ flex:none; opacity:.9; }}
  .tkfam {{ color:var(--muted); font-size:12px; width:120px; }} .tklv {{ margin-left:auto; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
  .splitwrap {{ display:grid; grid-template-columns:172px 1fr; gap:12px; align-items:start; }}
  .splitlist {{ display:flex; flex-direction:column; gap:3px; max-height:540px; overflow:auto; }}
  .splititem {{ display:flex; align-items:center; gap:7px; padding:7px 8px; border-radius:8px; cursor:pointer; }}
  .splititem:hover {{ background:var(--hover); }} .splititem.on {{ background:var(--inset); }}
  .splitsym {{ font-weight:600; font-size:13px; }} .splitact {{ margin-left:auto; font-size:10px; white-space:nowrap; }}
  .splitdetail {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:13px; min-height:160px; }}
  .sd-top {{ display:flex; align-items:center; gap:9px; margin-bottom:8px; }}
  .sd-full {{ margin-top:11px; background:var(--accent); color:#fff; border:0; border-radius:7px; padding:7px 13px; font-size:12px; font-weight:600; cursor:pointer; }}
  @media (max-width:760px) {{ .lanes,.bento,.mag,.splitwrap {{ grid-template-columns:1fr; }} }}
  .card-why {{ margin-top:11px; padding:9px 11px; background:var(--inset);
    border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:8px; }}
  .why-h {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.04em;
    font-weight:800; color:var(--muted); margin-bottom:7px; }}
  .why-fam {{ display:inline-block; max-width:100%; box-sizing:border-box;
    font-weight:700; font-size:11px; margin:0 0 9px; white-space:normal;
    color:var(--accent); background:color-mix(in srgb, var(--accent) 14%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 32%, transparent);
    padding:2px 9px; border-radius:999px; }}
  .strat-badge {{ display:inline-flex; align-items:center; gap:9px; margin:0 0 14px;
    padding:6px 13px; border-radius:999px; background:color-mix(in srgb, var(--accent) 12%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 30%, transparent); }}
  .strat-badge .k {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em;
    font-weight:800; color:var(--muted); }}
  .strat-badge .v {{ font-size:13px; font-weight:700; color:var(--accent); }}
  .why-chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
  .why-chip {{ font-size:11.5px; padding:3px 9px; border-radius:999px; line-height:1.35;
    border:1px solid var(--line); background:var(--card); color:var(--txt); white-space:nowrap; }}
  .why-chip.trig {{ background:var(--accent); color:#fff; border-color:var(--accent); font-weight:700; }}
  .why-chip.more {{ color:var(--muted); }}
  .why-txt {{ font-size:12.5px; color:var(--txt); line-height:1.45; }}
  .more {{ color:var(--muted); font-size:12px; margin-top:10px;
    border-top:1px solid var(--line); padding-top:8px; }}
  .sym {{ font-size:18px; font-weight:700; }}
  .logo {{ width:20px; height:20px; border-radius:4px; vertical-align:middle;
    margin-right:7px; background:#fff; object-fit:contain; }}
  .cname {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .act {{ float:right; padding:2px 10px; border-radius:6px; font-size:12px; font-weight:700; color:#fff; }}
  .a-BUY {{ background:var(--buy); }} .a-SELL {{ background:var(--sell); }}
  .a-HOLDLONG {{ background:var(--hold); }} .a-FLAT {{ background:var(--flat); }}
  .a-SHORT {{ background:var(--short); }} .a-HOLDSHORT {{ background:var(--short); opacity:.82; }}
  .a-WATCHLONG {{ background:var(--watch); }} .a-WATCHSHORT {{ background:var(--watch); }}
  .a-EXIT {{ background:var(--exit); }} .a-AVOID {{ background:var(--avoid); }}
  .px {{ font-size:26px; font-weight:700; margin:8px 0 2px; }}
  .kv {{ display:flex; justify-content:space-between; font-size:13px;
    color:var(--muted); padding:3px 0; }}
  .kv span:last-child {{ color:var(--txt); }}
  .hot span:last-child {{ color:#e8c878; font-weight:700; }}
  select {{ background:var(--card); color:var(--txt); border:1px solid var(--line);
    border-radius:8px; padding:6px 10px; font-size:14px; }}
  .chartbox {{ background:var(--card); border:1px solid var(--line);
    border-radius:12px; padding:16px; margin-top:14px; }}
  .news a, .news span.h {{ color:var(--txt); text-decoration:none; }}
  .news li {{ margin-bottom:10px; }}
  .news .src {{ color:var(--muted); font-size:12px; }}
  .disclaimer {{ color:var(--muted); font-size:12px; margin-top:36px;
    border-top:1px solid var(--line); padding-top:16px; }}
  /* modal */
  .overlay {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.6);
    z-index:50; padding:24px; overflow:auto; }}
  .overlay.open {{ display:block; }}
  .modal {{ max-width:720px; margin:24px auto; background:var(--card);
    border:1px solid var(--line); border-radius:16px; padding:24px; }}
  .modal h3 {{ margin:0; font-size:22px; }}
  .modal .close {{ float:right; cursor:pointer; color:var(--muted);
    font-size:22px; line-height:1; border:none; background:none; }}
  .modal .summary {{ font-size:15px; margin:12px 0 4px; }}
  .reasons {{ list-style:none; padding:0; margin:14px 0; }}
  .reasons li {{ position:relative; padding:8px 0 8px 24px; font-size:14px;
    border-bottom:1px solid var(--line); }}
  .reasons li:before {{ content:'›'; position:absolute; left:6px; color:var(--hold); }}
  .modal .sech {{ color:var(--muted); text-transform:uppercase; font-size:12px;
    letter-spacing:.05em; margin:18px 0 8px; }}
  .modal .chartbox {{ margin-top:0; }}
  .plangrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:10px; }}
  .stat {{ background:var(--inset); border:1px solid var(--hud-edge); border-radius:6px;
    padding:10px 12px; }}
  .stat .l {{ color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.04em; }}
  .stat .v {{ font-size:17px; font-weight:700; margin-top:2px; }}
  .stat .v.buy {{ color:var(--buy); }} .stat .v.sell {{ color:var(--sell); }}
  .stat .sub {{ color:var(--muted); font-size:11px; }}
  /* target scenarios (conservative / base / stretch) */
  .scen {{ grid-column:1/-1; margin-top:8px; border-top:1px solid var(--line); padding-top:12px; }}
  .scen-h {{ font-size:12.5px; font-weight:700; margin-bottom:9px; }}
  .scen-h span {{ font-weight:400; color:var(--muted); text-transform:none; }}
  .scen-row {{ padding:8px 11px; border:1px solid var(--line); border-radius:9px; margin-bottom:7px; background:var(--inset); }}
  .scen-row.higherodds {{ border-left:3px solid var(--buy); }}
  .scen-row.medium {{ border-left:3px solid var(--accent); }}
  .scen-row.lowerodds {{ border-left:3px solid #b8860b; }}
  .scen-top {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; }}
  .scen-px {{ font-variant-numeric:tabular-nums; font-weight:700; }}
  .scen-px em {{ color:var(--muted); font-style:normal; font-weight:500; font-size:12px; }}
  .scen-why {{ color:var(--muted); font-size:12px; margin-top:3px; line-height:1.4; }}
  /* signal-input detail cards (modal Signals sub-tab) */
  .sigdet {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:10px 12px; margin-bottom:9px; }}
  .sigdet.good {{ border-left-color:var(--buy); }}
  .sigdet.bad {{ border-left-color:var(--sell); }}
  .sigdet.warn {{ border-left-color:#b8860b; }}
  .sigdet-h {{ display:flex; align-items:baseline; gap:8px; font-size:13.5px; flex-wrap:wrap; }}
  .sigdet-v {{ margin-left:auto; font-weight:700; font-variant-numeric:tabular-nums; font-size:12.5px; }}
  .sigdet-why {{ color:var(--muted); font-size:12px; margin-top:4px; line-height:1.45; }}
  /* news-driven ideas (Market news tab) */
  .nideas {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:10px; margin-top:10px; }}
  .nidea {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid var(--line);
    border-radius:9px; padding:10px 12px; }}
  .nidea.buy {{ border-left-color:var(--buy); }}
  .nidea.sell {{ border-left-color:var(--sell); }}
  .nidea-top {{ display:flex; align-items:baseline; gap:8px; font-size:14px; flex-wrap:wrap; }}
  .nidea-conf {{ margin-left:auto; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .nidea-why {{ font-size:12.5px; margin-top:4px; line-height:1.45; }}
  .nidea-src {{ color:var(--muted); font-size:11px; margin-top:4px; font-style:italic; }}
  /* system status tab */
  .sysgrid {{ margin-top:8px; }}
  .sysrow {{ display:flex; align-items:center; gap:10px; padding:7px 2px; border-bottom:1px solid var(--line); }}
  .sysrow:last-child {{ border-bottom:0; }}
  .sysname {{ font-size:13px; font-weight:600; }}
  .sysnote {{ color:var(--muted); font-size:11.5px; margin-left:auto; text-align:right; }}
  .syspill {{ font-size:10.5px; font-weight:800; padding:2px 8px; border-radius:999px; white-space:nowrap;
    margin-left:10px; letter-spacing:.03em; }}
  .syspill.on {{ color:var(--buy); background:color-mix(in srgb, var(--buy) 14%, transparent); }}
  .syspill.off {{ color:var(--muted); background:var(--inset); }}
  .deskread {{ background:var(--inset); border:1px solid var(--line); border-left:3px solid var(--hold);
    border-radius:10px; padding:12px 14px; font-size:14px; margin:14px 0; }}
  .convbadge {{ font-size:13px; font-weight:700; padding:2px 10px; border-radius:999px; color:#fff; }}
  .conv-High {{ background:var(--buy); }} .conv-Medium {{ background:#9e6a1e; }}
  .conv-Low {{ background:var(--sell); }}
  .checks {{ list-style:none; padding:0; margin:8px 0; }}
  .checks li {{ display:flex; gap:10px; align-items:flex-start; padding:7px 0;
    border-bottom:1px solid var(--line); font-size:13px; }}
  .checks .ic {{ flex:0 0 18px; font-weight:700; }}
  .checks .pass .ic {{ color:var(--buy); }} .checks .warn .ic {{ color:#e8c878; }}
  .checks .fail .ic {{ color:var(--sell); }}
  .checks .ck-l {{ font-weight:600; }} .checks .ck-n {{ color:var(--muted); }}
  .chartkey {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.6; }}
  .reasons li {{ font-size:14px; line-height:1.5; }}
  .readout {{ display:flex; align-items:baseline; gap:12px; min-height:30px; margin:2px 0 10px; }}
  .readout .rprice {{ font-size:24px; font-weight:700; }}
  .readout .rchg {{ font-size:15px; font-weight:600; }}
  .readout .rdate {{ color:var(--muted); font-size:13px; margin-left:auto; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
  .chip {{ font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px;
    border:1px solid var(--line); }}
  .chip.bull {{ background:color-mix(in srgb, var(--buy) 14%, transparent); color:var(--buy);
    border-color:color-mix(in srgb, var(--buy) 32%, transparent); }}
  .chip.bear {{ background:color-mix(in srgb, var(--sell) 14%, transparent); color:var(--sell);
    border-color:color-mix(in srgb, var(--sell) 32%, transparent); }}
  .chip.neutral {{ background:var(--inset); color:var(--txt2); border-color:var(--line); }}
  .chip.mini {{ font-size:10.5px; padding:1px 7px; }}
  /* primary nav: brand-aligned horizontal scroller flanked by arrow controls */
  .tabsbar {{ display:flex; align-items:stretch; gap:2px; }}
  .tabscroll {{ flex:none; display:none; align-items:center; justify-content:center;
    width:28px; padding:0 0 2px; background:none; border:none; cursor:pointer; color:var(--muted);
    font-size:20px; line-height:1; border-bottom:2px solid transparent;
    transition:color .15s ease, opacity .15s ease; }}
  .tabscroll:hover {{ color:var(--txt); }}
  .tabsbar.scrollable .tabscroll {{ display:inline-flex; }}
  .tabsbar.at-start .tabscroll.left {{ opacity:0; pointer-events:none; }}
  .tabsbar.at-end .tabscroll.right {{ opacity:0; pointer-events:none; }}
  .tabs {{ flex:1 1 auto; min-width:0; display:flex; gap:1px; flex-wrap:nowrap; margin:0; padding:0;
    overflow-x:auto; overflow-y:hidden; scrollbar-width:none; -ms-overflow-style:none;
    -webkit-overflow-scrolling:touch; scroll-snap-type:x proximity; scroll-behavior:smooth; }}
  .tabsbar.scrollable .tabs {{
    -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 calc(100% - 18px),transparent 100%);
            mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 calc(100% - 18px),transparent 100%); }}
  .tabsbar.scrollable.at-start .tabs {{ -webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 18px),transparent 100%);
            mask-image:linear-gradient(90deg,#000 calc(100% - 18px),transparent 100%); }}
  .tabsbar.scrollable.at-end .tabs {{ -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 100%);
            mask-image:linear-gradient(90deg,transparent 0,#000 18px,#000 100%); }}
  .tabs::-webkit-scrollbar {{ height:0; display:none; }}
  .tabs button {{ background:none; border:none; color:var(--muted); font-size:14px;
    font-weight:600; padding:9px 14px 11px; cursor:pointer; white-space:nowrap; flex:none;
    letter-spacing:-.004em; border-bottom:2px solid transparent; scroll-snap-align:start; }}
  .tabs button.on {{ color:var(--txt); border-bottom-color:var(--accent); }}
  .tabs button:hover {{ color:var(--txt); }}
  .ctlbtn:hover, .ctlgrp button:hover {{ color:var(--txt); background:var(--hover); }}
  .page {{ display:none; }} .page.on {{ display:block; }}
  .secthead {{ font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase;
    letter-spacing:.05em; margin:22px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .secthead:first-child {{ margin-top:4px; }}
  .ovbox {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:6px;
    padding:14px 16px; margin-bottom:22px; }}
  .ovhead {{ font-weight:600; font-size:13px; margin-bottom:8px; text-transform:uppercase; letter-spacing:.08em; }}
  .ovwrap {{ display:flex; gap:14px; align-items:stretch; }}
  .ovchart {{ flex:1; min-width:0; }}
  .ovboard {{ width:150px; max-height:300px; overflow-y:auto; border-left:1px solid var(--line); padding-left:10px; }}
  .ovrow {{ display:flex; align-items:center; gap:6px; font-size:12px; padding:3px 2px; cursor:pointer; color:var(--muted); border-radius:4px; }}
  .ovrow:hover {{ color:var(--txt); background:var(--inset); }}
  .ovrow.on {{ color:var(--txt); font-weight:700; }}
  .ovdot {{ width:8px; height:8px; border-radius:50%; flex:0 0 8px; }}
  .ovsym {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .viewctl {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:16px; }}
  .regime {{ border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin:14px 0 4px;
    display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; }}
  .regime .rlabel {{ font-weight:700; font-size:15px; }}
  .regime .rdetail {{ color:var(--txt); font-size:13px; }}
  .regime .rnote {{ color:var(--muted); font-size:12px; width:100%; }}
  .secrow {{ display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }}
  .secname {{ width:130px; color:var(--txt); }}
  .secbar {{ flex:1; height:8px; background:var(--inset); border-radius:5px; overflow:hidden; }}
  .secfill {{ height:100%; background:linear-gradient(90deg,#388bfd,#2ea043); }}
  .secpct {{ width:90px; text-align:right; color:var(--muted); }}
  .track {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px; margin:18px 0; }}
  .track h2 {{ margin:0 0 4px; }}
  .trackstats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
    gap:10px; margin:12px 0; }}
  .trackrec {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  .trackrec th, .trackrec td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
  .trackrec th {{ color:var(--muted); font-weight:600; }}
  .win {{ color:var(--buy); }} .loss {{ color:var(--sell); }} .exp {{ color:var(--muted); }}
  /* shared table style for the intelligence/data panels (was previously unstyled) */
  .tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .tbl th, .tbl td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); vertical-align:middle; }}
  .tbl thead th {{ color:var(--muted); font-weight:600; font-size:11px; white-space:nowrap;
    text-transform:uppercase; letter-spacing:.05em; position:sticky; top:0; background:var(--card); }}
  .tbl tbody tr:hover {{ background:var(--hover); }}
  .tbl tbody tr:last-child td {{ border-bottom:none; }}
  .tbl .buy {{ color:var(--buy); }} .tbl .sell {{ color:var(--sell); }}
  details.ovbox summary {{ list-style:none; }}
  details.ovbox summary::-webkit-details-marker {{ display:none; }}
  .chartctl {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:10px; }}
  .ctlgrp {{ display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  .ctlgrp button {{ background:var(--card); color:var(--muted); border:none; padding:5px 11px;
    font-size:13px; cursor:pointer; border-right:1px solid var(--line); }}
  .ctlgrp button:last-child {{ border-right:none; }}
  .ctlgrp button.on {{ background:var(--hold); color:#fff; }}
  .ctltog {{ font-size:13px; color:var(--muted); cursor:pointer; }}
  .ctlbtn {{ background:var(--card); color:var(--muted); border:1px solid var(--line);
    border-radius:8px; padding:5px 11px; font-size:13px; cursor:pointer; }}
  .method {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:4px 18px; margin:18px 0 8px; }}
  .method summary {{ cursor:pointer; font-weight:700; font-size:15px; padding:12px 0;
    list-style:none; }}
  .method summary::-webkit-details-marker {{ display:none; }}
  .method summary:before {{ content:'▸ '; color:var(--hold); }}
  .method[open] summary:before {{ content:'▾ '; }}
  .method h4 {{ margin:16px 0 6px; font-size:14px; color:var(--txt); }}
  .method p, .method li {{ font-size:14px; color:var(--txt2); line-height:1.6; }}
  .method ol, .method ul {{ padding-left:20px; margin:6px 0; }}
  .method .pill {{ display:inline-block; background:var(--inset); border:1px solid var(--line);
    border-radius:6px; padding:1px 7px; font-size:13px; color:var(--txt); }}
  /* ---- theme toggle ---- */
  .themebtn {{ background:var(--card); color:var(--muted);
    border:1px solid var(--line); border-radius:8px; padding:6px 12px; font-size:13px; cursor:pointer;
    box-shadow:var(--shadow); }}
  .themebtn:hover {{ color:var(--txt); }}
  /* ---- accent colour picker ---- */
  .accent-wrap {{ position:relative; display:inline-block; }}
  .accent-pop {{ position:absolute; right:0; top:calc(100% + 6px); background:var(--card);
    border:1px solid var(--line); border-radius:10px; padding:11px; display:flex; flex-wrap:wrap;
    gap:7px; width:184px; z-index:60; box-shadow:var(--shadow-lg); }}
  .accent-pop[hidden] {{ display:none; }}
  .accent-pop .acsw {{ width:26px; height:26px; border-radius:50%; border:2px solid transparent;
    cursor:pointer; padding:0; }}
  .accent-pop .acsw.on {{ border-color:var(--txt); }}
  .accent-pop .accustom {{ display:flex; align-items:center; gap:6px; font-size:11px;
    color:var(--muted); width:100%; margin-top:2px; }}
  .accent-pop .accustom input {{ width:26px; height:26px; padding:0; border:none; background:none; cursor:pointer; }}
  .accent-pop .acreset {{ width:100%; font-size:11.5px; color:var(--muted); background:var(--inset);
    border:1px solid var(--line); border-radius:7px; padding:6px; cursor:pointer; }}
  .accent-pop .acreset:hover {{ color:var(--txt); }}
  /* ---- sidebar + top-tab shell ---- */
  .shell {{ display:flex; gap:0; align-items:flex-start; }}
  .sidebar {{ width:150px; flex:0 0 150px; position:sticky; top:8px; display:flex; flex-direction:column;
    gap:3px; padding:2px 10px 8px 0; }}
  .sidebar button {{ display:flex; align-items:center; gap:9px; text-align:left; background:none;
    border:none; color:var(--muted); font-size:12px; font-weight:600; padding:9px 11px; border-radius:7px;
    cursor:pointer; text-transform:uppercase; letter-spacing:.07em; }}
  .sidebar button svg {{ width:15px; height:15px; flex:0 0 auto; }}
  .sidebar button:hover {{ background:var(--hover); color:var(--txt); }}
  .sidebar button.on {{ background:color-mix(in srgb,var(--accent) 15%,transparent); color:var(--accent); }}
  .maincol {{ flex:1; min-width:0; padding-left:16px; border-left:1px solid var(--line); }}
  .toptabs {{ display:flex; gap:3px; flex-wrap:wrap; border-bottom:1px solid var(--line); margin:0 0 14px; }}
  .toptabs button {{ background:none; border:none; border-bottom:2px solid transparent; color:var(--muted);
    font-size:13px; font-weight:600; padding:8px 13px; margin-bottom:-1px; cursor:pointer; }}
  .toptabs button:hover {{ color:var(--txt); }}
  .toptabs button.on {{ color:var(--accent); border-bottom-color:var(--accent); }}
  @media (max-width:760px) {{
    .shell {{ flex-direction:column; }}
    .sidebar {{ flex-direction:row; width:auto; flex:none; position:sticky; top:0; z-index:40;
      overflow-x:auto; background:var(--bg); border-bottom:1px solid var(--line); padding:6px 0; gap:2px; }}
    .sidebar button {{ white-space:nowrap; padding:8px 12px; }}
    .maincol {{ padding-left:0; border-left:none; width:100%; }}
  }}
  /* ---- featured chart panel + watchlist ---- */
  .featured {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px;
    margin:6px 0 18px; box-shadow:var(--shadow); }}
  .feat-grid {{ display:grid; grid-template-columns:1fr 256px; gap:16px; }}
  @media (max-width:840px) {{ .feat-grid {{ grid-template-columns:1fr; }} .feat-watch {{ max-height:220px; }} }}
  .feat-wtitle {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
    font-weight:700; margin-bottom:6px; }}
  .feat-watch {{ overflow-y:auto; max-height:560px; border-left:1px solid var(--line); padding-left:12px; }}
  .wl {{ display:flex; align-items:center; gap:9px; padding:6px 8px; border-radius:7px; cursor:pointer; }}
  .wl:hover {{ background:var(--line); }}
  .wl.on {{ background:color-mix(in srgb, var(--accent) 16%, transparent); }}
  .wl-logo {{ position:relative; flex:0 0 auto; width:28px; height:28px; border-radius:6px; color:#fff;
    font-size:10px; font-weight:800; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .wl-logo img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#fff; }}
  .wl-main {{ display:flex; flex-direction:column; min-width:0; flex:1 1 auto; }}
  .wl-sym {{ font-weight:700; font-size:13px; line-height:1.2; }}
  .wl-name {{ color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .wl-r {{ display:flex; flex-direction:column; align-items:flex-end; flex:0 0 auto; }}
  .wl-px {{ font-variant-numeric:tabular-nums; font-size:12px; }}
  .wl-chg {{ font-variant-numeric:tabular-nums; font-size:11px; }}
  /* ---- TradeChart component ---- */
  .tc {{ width:100%; }}
  .tc-bar {{ display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; margin-bottom:8px; }}
  .tc-seg {{ display:inline-flex; align-items:center; gap:2px; background:var(--bg); border:1px solid var(--line);
    border-radius:8px; padding:2px; }}
  .tc-seglab {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
    padding:0 6px; font-weight:700; }}
  .tc-seg button {{ background:none; border:none; color:var(--muted); font-size:12px; padding:4px 9px;
    border-radius:6px; cursor:pointer; font-variant-numeric:tabular-nums; }}
  .tc-seg button:hover {{ color:var(--txt); }}
  .tc-seg button.on {{ background:var(--accent); color:#fff; }}
  .tc-cmp {{ background:var(--bg); border:1px solid var(--line); border-radius:7px; color:var(--txt);
    padding:5px 9px; font-size:12px; }}
  .tc-clr {{ background:var(--bg); border:1px solid var(--line); color:var(--muted); border-radius:7px;
    padding:5px 10px; font-size:12px; cursor:pointer; }}
  .tc-readout {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; min-height:24px; margin-bottom:4px; }}
  .tc-readout .tc-sym {{ font-weight:800; font-size:15px; }}
  .tc-readout .tc-price-lg {{ font-weight:700; font-size:18px; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-chg {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-ohlc {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
  .tc-readout .tc-date {{ color:var(--muted); font-size:12px; margin-left:auto; }}
  .tc-wrap {{ position:relative; height:380px; }}
  .tc-compact .tc-wrap {{ height:300px; }}
  /* TradingView widget containers */
  .tv-wrap {{ position:relative; height:520px; width:100%; }}
  .tv-wrap.tv-compact {{ height:420px; }}
  @media (max-width:760px) {{ .tv-wrap {{ height:380px; }} }}
  .tc-sub {{ position:relative; height:84px; margin-top:6px; }}
  .tc-sublab {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:2px; }}
  .tc-key {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.6; }}
  .tc-chip {{ display:inline-block; background:var(--line); border-radius:10px; padding:1px 8px; cursor:pointer;
    font-size:11px; color:var(--txt); }}
  /* ---- app shell ---- */
  /* unified sticky header: brand + status on top, primary tab nav directly beneath */
  .appbar {{ position:sticky; top:0; z-index:30; margin:0 0 18px;
    padding-top:env(safe-area-inset-top);
    background:color-mix(in srgb, var(--bg) 86%, transparent);
    backdrop-filter:saturate(1.4) blur(12px); -webkit-backdrop-filter:saturate(1.4) blur(12px);
    border-bottom:1px solid var(--line); }}
  .appbar-top {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:13px 2px 9px; }}
  .brand {{ display:flex; align-items:center; gap:11px; font-weight:800; font-size:18px; letter-spacing:-.02em; }}
  .brand-mark {{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
    border-radius:9px; color:#fff; font-size:15px; box-shadow:0 2px 8px color-mix(in srgb, var(--accent) 45%, transparent);
    background:linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 50%, #16c98d)); }}
  .appbar-right {{ display:flex; align-items:center; gap:10px; }}
  .livepill {{ font-size:12px; color:var(--muted); }}
  .subhead {{ color:var(--muted); font-size:12.5px; margin:0 0 16px; }}
  /* ---- KPI summary strip ---- */
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:0 0 18px; }}
  .kpi {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:6px; padding:12px 14px; }}
  .kpi.hero {{ grid-column:span 2; border-top:2px solid var(--accent);
    display:flex; flex-direction:column; justify-content:center; }}
  .kpi.hero .kpi-v {{ font-size:34px; }}
  @media (max-width:600px) {{ .kpi.hero {{ grid-column:span 2; }} .kpi.hero .kpi-v {{ font-size:26px; }} }}
  /* ---- bento home grid ---- */
  .bento {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-auto-flow:row dense;
    gap:10px; margin:0 0 18px; }}
  .bento .bt {{ background:var(--card); border:1px solid var(--hud-edge); border-radius:7px;
    padding:13px 15px; min-width:0; display:flex; flex-direction:column; }}
  .bento .bt.hero {{ grid-column:span 2; grid-row:span 2; border-top:2px solid var(--accent); justify-content:center; }}
  .bento .bt.wide {{ grid-column:span 2; }}
  .bt-l {{ font-size:10px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:600; margin-bottom:3px; }}
  .bt-v {{ font-family:var(--mono); font-weight:700; font-size:22px; }}
  .bt-v.buy {{ color:var(--buy); }} .bt-v.sell {{ color:var(--sell); }} .bt-v.warn {{ color:var(--warn); }}
  .bt-sub {{ font-size:11px; color:var(--muted); margin-top:3px; }}
  .bt-chip {{ display:inline-block; margin-top:9px; font-size:11px; padding:3px 10px; border-radius:999px;
    background:color-mix(in srgb,var(--accent) 16%,transparent); color:var(--accent); align-self:flex-start; }}
  .bt-body {{ font-size:13px; line-height:1.6; color:var(--txt2); max-height:210px; overflow:auto; }}
  .bt-list {{ margin:4px 0 0; padding-left:16px; font-size:13px; line-height:1.7; color:var(--txt2); }}
  @media (max-width:760px) {{
    .bento {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .bento .bt.hero {{ grid-column:span 2; grid-row:auto; }}
    .bento .bt.wide {{ grid-column:span 2; }}
  }}
  .kpi-l {{ font-size:10px; text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-weight:600; }}
  .kpi-v {{ font-size:24px; font-weight:800; margin-top:4px; letter-spacing:-.015em; font-variant-numeric:tabular-nums; }}
  .kpi-v.buy {{ color:var(--buy); }} .kpi-v.sell {{ color:var(--sell); }} .kpi-v.warn {{ color:#b8860b; }}
  .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  /* ---- redesigned signal card ---- */
  .card-top {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
  .card-mono {{ width:36px; height:36px; border-radius:9px; flex:0 0 auto; display:flex; align-items:center;
    justify-content:center; color:#fff; font-weight:800; font-size:12px; position:relative; overflow:hidden; }}
  .card-id {{ min-width:0; flex:1 1 auto; }}
  .card-id .s {{ font-size:16px; font-weight:800; line-height:1.15; }}
  .card-id .n {{ font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .card-age {{ font-size:10.5px; color:var(--muted); margin-top:3px; }}
  .card-age.fresh {{ color:var(--buy); font-weight:700; }}
  .card-px-row {{ display:flex; align-items:baseline; gap:10px; margin:6px 0 4px; }}
  .card-px {{ font-size:24px; font-weight:800; letter-spacing:-.015em; font-variant-numeric:tabular-nums; }}
  .card-day {{ font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .conv-wrap {{ margin:12px 0 10px; }}
  .conv-row {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted);
    margin-bottom:5px; text-transform:uppercase; letter-spacing:.04em; font-weight:700; }}
  .conv-meter {{ height:6px; background:var(--inset); border-radius:4px; overflow:hidden; }}
  .conv-fill {{ height:100%; border-radius:4px; transition:width .3s ease; }}
  .card-stats {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 16px; font-size:12.5px; margin-top:6px; }}
  .card-stat {{ display:flex; justify-content:space-between; color:var(--muted); }}
  .card-stat b {{ color:var(--txt); font-weight:600; font-variant-numeric:tabular-nums; }}
  /* ---- Markets sub-tab layout ---- */
  .mkt {{ display:grid; grid-template-columns:190px minmax(0,1fr); gap:18px; align-items:start; }}
  .mkt-side {{ display:flex; flex-direction:column; gap:4px; position:sticky; top:66px; }}
  .mkt-side button {{ text-align:left; background:none; border:none; color:var(--muted); font-size:14px;
    font-weight:600; padding:10px 13px; border-radius:9px; cursor:pointer; }}
  .mkt-side button:hover {{ background:var(--hover); color:var(--txt); }}
  .mkt-side button.on {{ background:color-mix(in srgb, var(--accent) 14%, transparent); color:var(--accent); }}
  .mkt-view {{ display:none; }} .mkt-view.on {{ display:block; }}
  @media (max-width:760px) {{ .mkt {{ grid-template-columns:1fr; }}
    .mkt-side {{ flex-direction:row; flex-wrap:wrap; position:static; }} }}
  /* ---- modal sub-tab layout ---- */
  .modal-wide {{ max-width:880px; }}
  .mk-top {{ display:flex; gap:4px; flex-wrap:wrap; border-bottom:1px solid var(--line);
    margin:14px 0 0; padding-bottom:0; }}
  .mk-top button {{ background:none; border:none; border-bottom:2px solid transparent; color:var(--muted);
    font-size:14px; font-weight:700; padding:9px 13px; cursor:pointer; margin-bottom:-1px; }}
  .mk-top button:hover {{ color:var(--txt); }}
  .mk-top button.on {{ color:var(--accent); border-bottom-color:var(--accent); }}
  .mk {{ display:grid; grid-template-columns:158px minmax(0,1fr); gap:18px; margin-top:14px; align-items:start; }}
  .mk-side {{ display:flex; flex-direction:column; gap:4px; position:sticky; top:0; }}
  .mk-side button {{ text-align:left; background:none; border:none; color:var(--muted); font-size:13.5px;
    font-weight:600; padding:9px 11px; border-radius:8px; cursor:pointer; }}
  .mk-side button:hover {{ background:var(--hover); color:var(--txt); }}
  .mk-side button.on {{ background:color-mix(in srgb, var(--accent) 14%, transparent); color:var(--accent); }}
  .mk-view {{ display:none; }} .mk-view.on {{ display:block; }}
  @media (max-width:680px) {{ .mk {{ grid-template-columns:1fr; }}
    .mk-side {{ flex-direction:row; flex-wrap:wrap; position:static; }} }}
  /* favorites star */
  .favbtn {{ background:none; border:none; color:var(--flat); cursor:pointer; font-size:17px;
    line-height:1; padding:0 2px; flex:0 0 auto; }}
  .favbtn:hover, .favbtn.on {{ color:#e8a93a; }}
  /* ---- mobile / small screens ---- */
  @media (max-width:600px) {{
    /* rank table: drop to the 3 key factors on a narrow screen */
    .rkf-sm {{ display:none; }}
    /* data/intelligence tables: denser, and let a wide one scroll inside its panel */
    .tbl {{ font-size:12px; }}
    .tbl th, .tbl td {{ padding:5px 6px; }}
    .ovbox {{ overflow-x:auto; }}
    .wrap {{ padding:16px 12px 48px; }}
    h1 {{ font-size:21px; }}
    .appbar-top {{ gap:8px; padding:11px 2px 7px; }}
    .brand {{ font-size:16.5px; }}
    .tabs button {{ padding:9px 12px 11px; font-size:13.5px; }}
    /* tighter KPI tiles so signals aren't pushed a full screen down */
    .kpis {{ grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px; margin-bottom:16px; }}
    .kpi {{ padding:9px 11px; }}
    .kpi-l {{ font-size:10px; }}
    .kpi-v {{ font-size:19px; margin-top:2px; }}
    .kpi-sub {{ font-size:10px; }}
    .trackrec {{ font-size:12px; }}
    .trackrec th, .trackrec td {{ padding:5px 6px; }}
    /* use more of the screen for the detail popup */
    .overlay {{ padding:10px; }}
    .modal {{ padding:16px; margin:8px auto; border-radius:12px; }}
    .modal h3 {{ font-size:19px; }}
    .tv-wrap {{ height:340px; }}
  }}
</style></head>
<body><div class="wrap">
  <header class="appbar">
    <div class="appbar-top">
      <div class="brand"><span class="brand-mark">◈</span><span>Signal Desk</span></div>
      <div class="appbar-right">
        <span class="badge m-{mode}">{mode}</span>
        <span class="livepill" id="liveStatus"></span>
        <button class="themebtn" title="Reload for the latest published build" onclick="location.reload()">⟳ Refresh</button>
        <div class="accent-wrap">
          <button id="accentBtn" class="themebtn" title="Accent colour" aria-label="Accent colour">🎨</button>
          <div id="accentPop" class="accent-pop" hidden>
            <button class="acsw" data-accent="#58a6ff" style="background:#58a6ff;" aria-label="Blue"></button>
            <button class="acsw" data-accent="#2dd4bf" style="background:#2dd4bf;" aria-label="Cyan"></button>
            <button class="acsw" data-accent="#8b5cf6" style="background:#8b5cf6;" aria-label="Violet"></button>
            <button class="acsw" data-accent="#46d08a" style="background:#46d08a;" aria-label="Green"></button>
            <button class="acsw" data-accent="#e0a82e" style="background:#e0a82e;" aria-label="Amber"></button>
            <button class="acsw" data-accent="#ff7a59" style="background:#ff7a59;" aria-label="Coral"></button>
            <button class="acsw" data-accent="#ec4899" style="background:#ec4899;" aria-label="Pink"></button>
            <label class="accustom">Custom <input type="color" id="accentCustom" value="#58a6ff"></label>
            <button id="accentReset" class="acreset">Reset to default</button>
          </div>
        </div>
        <button id="themeToggle" class="themebtn">🌙 Dark</button>
      </div>
    </div>
  </header>
  <div class="shell">
    <aside class="sidebar" id="sideNav">
      <button data-area="signals" class="on"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M1 8h3l2-5 3 10 2-5h4"/></svg> Signals</button>
      <button data-area="markets"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="6.3"/><path d="M1.7 8h12.6M8 1.7c2.4 1.8 2.4 10.8 0 12.6M8 1.7c-2.4 1.8-2.4 10.8 0 12.6"/></svg> Markets</button>
      <button data-area="portfolio"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="5" width="12.4" height="8.5" rx="1.2"/><path d="M5.5 5V3.7a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1V5"/></svg> Portfolio</button>
      <button data-area="intel"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><rect x="4.5" y="4.5" width="7" height="7" rx="1"/><path d="M6.5 1.8v2.7M9.5 1.8v2.7M6.5 11.5v2.7M9.5 11.5v2.7M1.8 6.5h2.7M1.8 9.5h2.7M11.5 6.5h2.7M11.5 9.5h2.7"/></svg> Intel</button>
      <button data-area="news"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="3" width="12.4" height="10" rx="1.2"/><path d="M4.3 6h7.4M4.3 8.3h7.4M4.3 10.6h4.5"/></svg> News</button>
      <button data-area="about"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="6.3"/><path d="M8 7.3v4"/><path d="M8 4.9h.01"/></svg> About</button>
    </aside>
    <div class="maincol">
      <nav class="toptabs" id="topTabs"></nav>
      <div class="subhead">Built {snap['generated_at']} <span id="builtAgo" style="opacity:.7;"></span> &middot; scanned {snap['scanned']} symbols{health_html}{pdrop_html}</div>
      <div class="subhead" id="marketClock" style="margin-top:-9px;opacity:.85;"></div>
      <div class="note" style="margin-top:0;">{mode_note}</div>
      <div id="diag"></div>

  <section class="page" id="page-markets">
    <div class="mkt">
      <nav class="mkt-side" id="mktNav">
        <button data-mview="chart" class="on">Chart</button>
        <button data-mview="sectors">Sector strength</button>
        <button data-mview="macro">Macro backdrop</button>
      </nav>
      <div class="mkt-main">
        <div class="mkt-view on" id="mview-chart">
          <div class="featured">
            <div class="feat-grid">
              <div class="feat-main"><div id="featuredChart"></div></div>
              <aside class="feat-watch"><div class="feat-wtitle">Watchlist · click to load</div><div id="featWatch"></div></aside>
            </div>
          </div>
        </div>
        <div class="mkt-view" id="mview-sectors">{sectors_html}</div>
        <div class="mkt-view" id="mview-macro">{macro_html}</div>
      </div>
    </div>
  </section>

  <section class="page on" id="page-signals">
    {bento_home_html}
    <div class="strat-badge"><span class="k">Strategy type</span><span class="v">Multi-strategy confluence · 7 long + 7 short, trend-gated</span></div>
    <div id="concWarn"></div>
    <details class="tvwidget" open>
      <summary>📺 Live market TV
        <span class="ctlgrp wtvgrp">
          <button data-wtv="KQp-e_XQnDE" class="on" onclick="event.preventDefault();event.stopPropagation();_wtvSet(this);">Yahoo Finance</button>
          <button data-wtv="vKOd3v8VTYo" onclick="event.preventDefault();event.stopPropagation();_wtvSet(this);">Schwab Network</button>
        </span>
        <a class="tvw-open" href="#" onclick="event.preventDefault();event.stopPropagation();_gotoTab('livetv');return false;">more channels →</a></summary>
      <div class="tvw-frame"><iframe id="wtvFrame" src="https://www.youtube.com/embed/KQp-e_XQnDE?autoplay=1&amp;mute=1" title="Live market TV" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    </details>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Layout:</span>
      <span class="ctlgrp" id="layoutBtns"></span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Sort:</span>
      <span class="ctlgrp" id="sortBtns"></span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Show:</span>
      <span class="ctlgrp" id="filterBtns"></span></div>
    <div id="cards"></div>
  </section>

  <section class="page" id="page-momentum">
    <h2 style="margin-top:0;">Momentum leaders <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— dual-momentum ranking (our best backtested strategy)</span></h2>
{momentum_html}
  </section>

  <section class="page" id="page-intraday">
    <h2 style="margin-top:0;">Intraday signals <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— the same engine on intraday bars (faster, noisier; daily signals remain the backbone)</span></h2>
    <div id="intradayCards"></div>
  </section>

  <section class="page" id="page-heatmap">
    <h2 style="margin-top:0;">Market heatmap <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— the whole market by sector, sized by market cap, coloured by today's move (independent of signals). Use the widget's top bar to switch index (S&amp;P 500, Nasdaq 100, TSX…)</span></h2>
    <div id="heatmapHost" style="height:78vh;min-height:520px;width:100%;"></div>
  </section>

  <section class="page" id="page-pairs">
{pairs_html}
  </section>

  <section class="page" id="page-portfolio">
    <h2 style="margin-top:0;">Portfolio <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— hypothetical book from today's actionable signals</span></h2>
{portfolio_html}
  </section>

  <section class="page" id="page-allweather">
    <h2 style="margin-top:0;">All Weather <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— Ray Dalio's risk-balanced all-seasons portfolio</span></h2>
{allweather_html}
  </section>

  <section class="page" id="page-ipos">
    <h2 style="margin-top:0;">IPO watch <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— upcoming listings + pre-IPO buzz (SpaceX, Stripe, etc.)</span></h2>
{ipo_html}
  </section>

  <section class="page" id="page-track">
{track_html}
  </section>

  <section class="page" id="page-altdata">
{altdata_html}
  </section>

  {paper_section}

  <section class="page" id="page-system">
{system_html}
  </section>

  <section class="page" id="page-method">
    <div class="method">
      <h4>The big picture</h4>
      <p>This page is an automated <b>stock screen</b>. Every weekday (after the US close) it scans a
      curated list of major stocks plus the day's biggest movers, and flags the ones that look like
      they're <b>starting to trend upward</b>, using a simple, well-known momentum strategy. It's a
      research tool to tell you <i>where to look</i> — not a tip service.</p>

      <h4>What we're looking for</h4>
      <p>Stocks where a short-term price trend is overtaking the longer-term trend — the classic early
      sign of a move higher — and where momentum and trading activity back that up.</p>

      <h4>The strategy: multi-strategy confluence, both directions</h4>
      <p>Rather than wait for one rare event (a single moving-average crossover), the engine runs a
      panel of well-known, independent strategies on each stock and asks <b>how many agree right now</b>,
      then filters by the trend regime and a conviction floor. This surfaces real setups far more
      often while keeping only the strong ones labelled actionable.</p>
      <ol>
        <li><b>Scan</b> — curated large-caps plus the day's most-active stocks and biggest movers.</li>
        <li><b>Buy signal</b> — price is in an <b>uptrend</b> (above its 200-day average) and <b>3+ independent
        strategies</b> line up long, and the setup clears a Medium-or-better conviction score. Weaker
        setups appear as <span class="pill">Watch</span> rather than a buy.</li>
        <li><b>Short signal</b> — the mirror image: price in a <b>downtrend</b> with 3+ strategies lined up
        short and conviction clearing the bar. Shorts profit if the stock falls — and carry higher risk,
        so they're gated the same way. There are also <span class="pill">Exit</span> (sell a long that
        just rolled over) and <span class="pill">Avoid</span> (weak, stay away) alerts.</li>
        <li><b>Risk first</b> — every setup gets a <b>stop-loss</b> (~{snap['params']['stop_loss_pct']:.0%} the wrong
        way) and an <b>honest target</b> — the nearest real level price has to clear (recent swing high/low),
        bounded by the analyst price target and a volatility-reachable distance, and never more than {snap['params']['take_profit_pct']:.0%} — sized so a stop-out
        costs only about {snap['params']['risk_per_trade']:.0%} of the account. For a long the stop sits below entry and
        the target above; for a short it's inverted.</li>
      </ol>

      <h4>How each signal is graded (multi-factor confluence)</h4>
      <p>A good trade rarely rests on one signal. Each stock is scored on several factors, the way a
      desk trader weighs confluence:</p>
      <ul>
        <li><b>Trend</b> — short-term average above the long-term one (direction).</li>
        <li><b>Momentum</b> — RSI (overbought/oversold) and MACD (is momentum building?).</li>
        <li><b>Trend strength</b> — ADX, to tell a real trend from chop.</li>
        <li><b>Volume</b> — heavier-than-usual trading confirms a move.</li>
        <li><b>Where price sits</b> — Bollinger band position, distance from 1-year highs/lows, and
        whether it's stretched (chasing) or pulling back to the trend.</li>
        <li><b>Risk : reward</b> — the target must pay enough for the risk taken.</li>
        <li><b>Historical edge</b> — we <i>backtest this exact strategy on that stock's own history</i>
        and factor in how often it has actually worked there.</li>
        <li><b>Strategy confluence</b> — the core of the engine. We run seven <i>independent</i> strategies
        in each direction: long (trend crossover, golden cross, Donchian breakout, MACD momentum, RSI-2
        dip-buy, Bollinger squeeze breakout, EMA momentum stack) and their bearish mirrors (death cross,
        breakdowns, RSI-2 rip-sell, etc.). When 3+ agree <i>and</i> price is in the matching trend,
        the setup becomes actionable; 2 agreeing is a Watch. The detail panel shows which are firing.</li>
        <li><b>Independent cross-check (TradingView)</b> — TradingView's own aggregate technical rating
        (daily + weekly), as a second opinion that's separate from our engine.</li>
        <li><b>News &amp; analysts</b> — recent news tone, the analyst consensus and average price target, plus
        <b>recent rating changes</b> (upgrades/downgrades and the firm behind them).</li>
        <li><b>Earnings momentum &amp; quality</b> — EPS and revenue growth, margins and leverage. Growing
        fundamentals back a long (and fight a short); shrinking ones do the reverse.</li>
        <li><b>Liquidity / execution quality</b> — average dollar turnover and an estimated spread. A name
        that's too thin to fill cleanly is flagged, and in paper trading its size is capped (or skipped)
        so the trade is actually practical — microstructure improving <i>execution</i>, not selection.</li>
        <li><b>Insider activity (SEC Form 4)</b> — clusters of open-market insider <i>purchases</i> raise a
        long's conviction (and lower a short's); heavy insider selling leans the other way.</li>
        <li><b>Retail buzz (StockTwits)</b> — crowd chatter and Bull/Bear sentiment, weighted gently since
        it's noisy and often contrarian.</li>
        <li><b>Short interest / squeeze risk (Yahoo)</b> — how heavily a name is shorted (% of float, days-to-cover).
        A crowded short is squeeze fuel for a long (a tailwind) and a real danger for a fresh short.</li>
        <li><b>Retail / social attention (Reddit &amp; WSB, via ApeWisdom)</b> — names the retail crowd is piling into.
        The lightest nudge of all: a mention spike adds momentum and volatility, so it gently helps a long and
        warns a short. Never a primary driver.</li>
        <li><b>Market alignment</b> — is the trade running with the broad tape (Risk-on/off) or against it?
        Counter-trend setups lose points.</li>
        <li><b>Earnings gate</b> — a fresh entry within ~2 days of an earnings report is held back (capped
        out of High conviction): a binary report can gap straight through the stop.</li>
      </ul>
      <p>The Signals page also warns when <b>too many fresh signals cluster in one sector</b> (often the same
      macro bet in disguise), and the <b>Data signals</b> tab explains and lists what the insider / rating /
      buzz scrapers found today.</p>
      <p>The detail panel also flags <b>chart patterns</b> (golden cross, breakouts, pullbacks, MACD
      crosses, oversold bounces…) and reads the <b>market backdrop</b> — overall breadth (how many
      stocks are trending up) and which <b>sectors</b> are strongest — because signals work better when
      the broader tape agrees.</p>

      <h4>How to use it</h4>
      <p>Each card shows the action and a <b>conviction score</b> (how well it fits the rules). Click any
      card for the full breakdown: a plain-English explanation, the trade plan (entry, stop, target,
      risk:reward), a chart marking where the strategy would have bought/sold, and recent news.</p>

      <h4>Proving it out — paper trading &amp; honest backtests</h4>
      <p>The <b>Track record</b> tab logs every call and grades it against real prices (hypothetical — no fees).
      The <b>Paper account</b> tab goes further: when enabled, fresh High-conviction signals are auto-submitted
      as bracket orders to a real Alpaca <b>paper</b> account, so you see actual fills, slippage and P&amp;L — the
      honest counterpart to the hypothetical log. And the <b>Momentum</b> tab leads with a
      <b>survivorship-bias-free</b> backtest (run on a fixed universe of always-alive ETFs) so its headline
      Sharpe/return can't be flattered by today's winners.</p>

      <h4>Trusting the backtest — walk-forward / out-of-sample validation</h4>
      <p>A single backtest sees the whole history, so any setting that happened to fit the past looks good — that's
      curve-fitting. The <b>Momentum</b> tab now also runs a <b>walk-forward</b> test: it tunes the strategy on a
      slice of <i>past</i> data, then trades the <i>next, unseen</i> slice with those frozen settings, and repeats
      rolling forward. The result is an honest <b>out-of-sample</b> read plus a verdict — <i>holds up</i>,
      <i>marginal</i>, or <i>fragile</i> — and a parameter-sensitivity sweep that shows whether the edge depends on
      one lucky setting. If a strategy only shines in-sample, this is where it gets exposed.</p>

      <h4>The intelligence layer — regimes, ranking &amp; learning</h4>
      <p>On top of the per-stock signals sit a few adaptive layers. The <b>macro regime classifier</b> (Markets tab)
      reads the backdrop and labels it risk-on / neutral / risk-off plus secondary tags — <i>high-volatility,
      recessionary, inflationary, liquidity-driven</i> — and uses that to set an exposure dial <i>and</i> tilt which
      strategies to lean on (momentum in risk-on, mean-reversion/pairs when it's choppy). The <b>adaptive ranking</b>
      (Portfolio tab → "Top opportunities") scores every actionable name 0–100 for where capital should go first,
      blending conviction quality, volatility-adjusted reward, macro fit, liquidity and momentum — so a high-conviction
      but illiquid or poorly-paying setup is correctly ranked below a cleaner one. And a <b>feedback loop</b> tags every
      logged trade with the macro regime and score at entry, so the Track record tab can show which regimes each
      strategy actually works in as results accrue. There's also a <b>no-trade layer</b> (Markets tab → "🚦 No-trade
      check") that makes the bot sit on its hands when conditions are poor — a major data release due that day, panic-level
      volatility, a deteriorating track record, or a drawdown breach — even if a signal fires. It's all transparent rules
      today; that logged history is also the foundation for adding machine-learning scoring later — and even then, every
      decision still passes through the rules-based risk engine, which always has the final say.</p>
      <p>Two more layers sit on top. A <b>meta-signal model</b> gives every candidate a second opinion —
      <i>accept, reduce, delay</i> or <i>reject</i> — weighing regime fit, liquidity, conflicting signals and how
      that regime has paid off before; a "reduce" halves the size, a "delay/reject" skips it. Each trade is then
      written out as a <b>structured record</b> (Portfolio tab) with its confidence, expected return range, holding
      period, risk and <b>uncertainty</b> scores — and when uncertainty is high (signals disagree, macro mixed,
      liquidity thin) the meta-model is what trims or skips it. Finally, an <b>AI news read</b> uses the language
      model to turn recent headlines into structured scores (guidance, demand, margins, regulatory risk and so on);
      it never places a trade — it just feeds the meta-model, so genuinely bad news quietly shrinks position size.
      The whole point is to be <i>more selective, not more active</i> — fewer, better trades.</p>

      <h4>Macro sets the exposure dial (not the trades)</h4>
      <p>Macro data — the VIX, the yield curve, credit spreads, the dollar, plus overall market breadth —
      never directly buys or sells anything. Instead it's blended into one <b>risk-on / neutral / risk-off</b>
      posture that sets an <b>exposure multiplier</b>: in a risk-on backdrop new positions are sized a little
      larger and lean into momentum; in risk-off they're sized smaller, with more cash and a defensive tilt.
      You can see the posture, the multiplier, and the drivers behind it on the <b>Markets</b> tab. The core
      principle: macro controls <i>how much</i> you deploy, security-level data controls <i>what</i> you pick,
      and liquidity controls <i>whether the trade is practical</i>.</p>

      <h4>Protecting the whole book — the risk engine &amp; kill switch</h4>
      <p>Stops protect a single trade; the <b>risk engine</b> protects the whole account. Before any new paper order
      it checks the book and can throttle or stand down:</p>
      <ul>
        <li><b>Daily loss limit</b> — once the day is down ~3%, it stops opening new positions (open trades keep their brackets).</li>
        <li><b>Drawdown control</b> — at ~8% peak-to-now drawdown it <b>halves</b> new-position size; at ~10% it
        <b>halts</b> new entries until equity recovers.</li>
        <li><b>Concentration cap</b> — no single position is allowed to exceed ~15% of equity.</li>
        <li><b>Kill switch</b> — repeated run failures (broker/data outages) flip a hard stop, which auto-resets after
        a few clean runs — so a glitch can never trigger runaway trading.</li>
      </ul>
      <p>Its current state shows as a colour-coded banner at the top of the <b>Paper account</b> tab
      (green normal, amber de-risking, red halt, or the kill switch).</p>

      <h4>A diversifier for flat markets — pairs &amp; mean-reversion</h4>
      <p>The core engine trades <i>direction</i> (trend + momentum), which struggles when the market goes sideways.
      The <b>Pairs</b> tab adds a market-neutral complement: it watches economically-related, liquid pairs
      (e.g. KO/PEP, GS/MS, V/MA), and when the price <b>spread</b> between two normally-linked names stretches
      unusually far — about <b>2 standard deviations</b> from its norm — it flags a bet that the gap closes again
      (long the cheap leg, short the rich one). It only lists a pair once the two legs are genuinely correlated and
      the spread is reliably <i>mean-reverting</i>; it exits as the spread reverts toward normal and stops out if the
      relationship breaks (past ~3σ). It leans in when the broad tape is trendless and steps back when there's a
      strong trend to ride. When a spread reaches its entry band you also get a <b>phone alert</b>, just like signal alerts.</p>

      <h4>Honest limits</h4>
      <p>This is an <b>educational tool, not financial advice</b>. Signals are often wrong, the data is
      free and slightly delayed, and the numbers ignore fees and slippage. The extra inputs above (insider,
      buzz, ratings) are <i>context, not certainty</i> — they tilt conviction, they don't guarantee anything.
      Treat it all as a starting point for your own research — never risk money you can't afford to lose.</p>
    </div>
  </section>

  <section class="page" id="page-news">
    <h2 style="margin-top:0;">Market news <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— recent headlines across the scanned stocks</span></h2>
{news_ideas_html}
    <ul class="news" id="news"></ul>
  </section>

  <section class="page" id="page-livetv">
    <h2 style="margin-top:0;">Live TV <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— live financial news streams</span></h2>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Channel:</span>
      <span class="ctlgrp" id="tvBtns"></span></div>
    <div class="tvwrap"><iframe id="tvFrame" title="Live financial news" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <p style="color:var(--muted);font-size:12px;margin-top:10px;">Start muted; unmute in the player. If a stream is blank (it restarted with a new ID) or you want the full experience, <a id="tvLink" href="https://www.youtube.com/@markets/live" target="_blank" rel="noopener">open it on YouTube ↗</a>. Not affiliated with these networks; embedded for convenience.</p>
  </section>

  <div class="disclaimer">
    Strategy: multi-strategy confluence (7 long + 7 short), trend-gated (200-day) with a
    conviction floor; {snap['params']['fast_ma']}/{snap['params']['slow_ma']} SMA + RSI({snap['params']['rsi_period']}) is one input.
    Risk {snap['params']['risk_per_trade']:.0%}/trade, stop {snap['params']['stop_loss_pct']:.0%}, target = nearest structure (bounded by fundamentals &amp; volatility, capped {snap['params']['take_profit_pct']:.0%}).
    Shorts profit if price falls and carry higher risk.
    "Rel vol" = today's volume vs its {snap['params']['rel_volume_window']}-day average — a free
    proxy for unusual activity, NOT real institutional/options order flow.<br>
    Educational tool only. Not financial advice. Signals can be wrong; backtests ignore
    fees and slippage. Verify before acting and never risk money you can't afford to lose.
  </div>
    </div>
  </div>
</div>

<div class="overlay" id="overlay">
  <div class="modal modal-wide">
    <button class="close" id="modalClose">&times;</button>
    <h3 id="mTitle"></h3>
    <div class="summary" id="mSummary"></div>
    <nav class="mk-top" id="mkTop">
      <button data-top="overview" class="on">Overview</button>
      <button data-top="chart">Chart</button>
      <button data-top="trade">Trade</button>
      <button data-top="intel">Intelligence</button>
      <button data-top="research">Research</button>
    </nav>
    <div class="mk">
      <nav class="mk-side" id="mkNav">
        <button data-top="overview" data-mkview="overview" class="on">Summary</button>
        <button data-top="chart" data-mkview="chart">Chart</button>
        <button data-top="trade" data-mkview="plan">Plan</button>
        <button data-top="trade" data-mkview="risk">Risk &amp; sizing</button>
        <button data-top="trade" data-mkview="exec">Execution</button>
        <button data-top="intel" data-mkview="meta">Meta verdict</button>
        <button data-top="intel" data-mkview="regimefit">Regime fit</button>
        <button data-top="intel" data-mkview="newsread">AI news read</button>
        <button data-top="intel" data-mkview="rank">Adaptive rank</button>
        <button data-top="research" data-mkview="strategies">Strategies</button>
        <button data-top="research" data-mkview="signals">Signal inputs</button>
        <button data-top="research" data-mkview="research">Fundamentals &amp; news</button>
      </nav>
      <div class="mk-main">
        <div class="mk-view on" id="mkview-overview">
          <div class="sech" id="mAIHead" style="display:none;">In plain English (AI) 🤖</div>
          <div class="deskread" id="mAI" style="display:none;border-left-color:#9b59b6;"></div>
          <div class="sech">The bottom line</div>
          <div class="deskread" id="mDesk"></div>
          <div class="sech">Should you take it? <span id="mConvScore"></span></div>
          <ul class="checks" id="mChecks"></ul>
          <div class="sech">Patterns spotted</div>
          <div class="chips" id="mPatterns"></div>
        </div>
        <div class="mk-view" id="mkview-chart">
          <div id="modalChart"></div>
        </div>
        <div class="mk-view" id="mkview-plan">
          <div class="sech" style="margin-top:0;">The trade plan <span id="mPlanNote" style="text-transform:none;color:var(--muted);"></span></div>
          <div class="plangrid" id="mPlan"></div>
          <div class="sech">How this strategy has done on this stock <span style="text-transform:none;color:var(--muted);">(backtest)</span></div>
          <div class="plangrid" id="mEdge"></div>
          <div class="sech">Market context</div>
          <div class="plangrid" id="mContext"></div>
        </div>
        <div class="mk-view" id="mkview-strategies">
          <div class="sech" style="margin-top:0;">Strategies in play <span style="text-transform:none;color:var(--muted);">— independent methods + their track record here</span></div>
          <div id="mStrategies"></div>
        </div>
        <div class="mk-view" id="mkview-signals">
          <div class="sech" style="margin-top:0;">Signal inputs <span style="text-transform:none;color:var(--muted);">— the extra data feeding this call, in detail</span></div>
          <div id="mSignals"></div>
        </div>
        <div class="mk-view" id="mkview-research">
          <div class="sech" style="margin-top:0;">Analysts, fundamentals &amp; news tone</div>
          <div class="plangrid" id="mResearch"></div>
          <div class="sech">Latest news on this stock</div>
          <ul class="news" id="mNews"></ul>
          <div class="sech">The details, explained</div>
          <ul class="reasons" id="mReasons"></ul>
        </div>
        <div class="mk-view" id="mkview-risk">
          <div class="sech" style="margin-top:0;">Risk &amp; sizing <span style="text-transform:none;color:var(--muted);">— the structured signal contract</span></div>
          <div class="plangrid" id="mRisk"></div>
          <div class="sech">Kill conditions</div>
          <div id="mKill" style="font-size:13px;color:var(--txt2);"></div>
        </div>
        <div class="mk-view" id="mkview-exec">
          <div class="sech" style="margin-top:0;">Execution quality <span style="text-transform:none;color:var(--muted);">— can this be traded cleanly?</span></div>
          <div class="plangrid" id="mExec"></div>
        </div>
        <div class="mk-view" id="mkview-meta">
          <div class="sech" style="margin-top:0;">Meta-signal verdict <span style="text-transform:none;color:var(--muted);">— the second opinion on this trade</span></div>
          <div id="mMeta"></div>
        </div>
        <div class="mk-view" id="mkview-regimefit">
          <div class="sech" style="margin-top:0;">Macro &amp; regime fit</div>
          <div id="mRegimeFit"></div>
        </div>
        <div class="mk-view" id="mkview-newsread">
          <div class="sech" style="margin-top:0;">AI news read <span style="text-transform:none;color:var(--muted);">— headlines turned into structured scores</span></div>
          <div id="mNewsRead"></div>
        </div>
        <div class="mk-view" id="mkview-rank">
          <div class="sech" style="margin-top:0;">Adaptive allocation rank</div>
          <div id="mRank"></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const LIVE_URL = "{CONFIG.live_quotes_url}";
let LIVE = {{}};  // latest live prices (declared early so renderCards can read it safely)
let featTC = null, modalTC = null;   // Capital IQ-style chart engine instances
window.__APP = {{ DATA: DATA, LIVE_URL: LIVE_URL }};
// read a CSS theme variable (so the overview chart flips with light/dark)
function _cv(n, f) {{ try {{ const v = getComputedStyle(document.documentElement).getPropertyValue(n).trim(); return v || f; }} catch (e) {{ return f; }} }}
function _shortDate(s) {{ try {{ return new Date(s + 'T00:00:00').toLocaleDateString([], {{month:'short', day:'numeric'}}); }} catch (e) {{ return s; }} }}
let _curSort = 'sector';
try {{ _curSort = localStorage.getItem('sort') || 'sector'; }} catch(e) {{}}
let _curFilter = 'all';
try {{ _curFilter = localStorage.getItem('filter') || 'all'; }} catch(e) {{}}
let FAVS = new Set();
try {{ FAVS = new Set(JSON.parse(localStorage.getItem('tb-favs') || '[]')); }} catch (e) {{}}
function _toggleFav(sym) {{
  if (FAVS.has(sym)) FAVS.delete(sym); else FAVS.add(sym);
  try {{ localStorage.setItem('tb-favs', JSON.stringify([...FAVS])); }} catch (e) {{}}
}}
// Plain-English explanations shown on hover for every strategy + type, across the app.
const STRAT_INFO = {{
  'Trend crossover': 'A short-term average price crosses ABOVE a longer-term one — a classic early sign an uptrend is starting.',
  'Golden cross': 'The 50-day average rises above the 200-day — a slow, big-picture signal the long-term trend has turned up.',
  'Donchian breakout': 'Price pushes above its highest level of the last 20 days — buyers breaking it out to fresh short-term highs.',
  'MACD momentum': 'A popular momentum gauge turns positive — the upward speed of the move is building.',
  'Dip buy (RSI-2)': 'Inside an existing uptrend, price dips hard for a day or two — a chance to buy the pullback before it resumes.',
  'Squeeze breakout': 'After a quiet, low-volatility stretch, price pops out of its range — pent-up energy releasing into a move up.',
  'EMA momentum stack': 'Fast averages line up above slow ones (8 > 21 > 50) — a tidy, healthy uptrend with momentum behind it.',
  'Trend cross-down': 'A short-term average crosses BELOW a longer-term one — a classic early sign a downtrend is starting.',
  'Death cross': 'The 50-day average falls below the 200-day — a slow, big-picture signal the long-term trend has turned down.',
  'Donchian breakdown': 'Price breaks below its lowest level of the last 20 days — sellers pushing it to fresh short-term lows.',
  'MACD momentum (down)': 'The momentum gauge turns negative — the downward speed of the move is building.',
  'Rip-sell (RSI-2)': 'Inside a downtrend, price spikes up sharply for a day or two — a chance to short the bounce before it rolls back over.',
  'Squeeze breakdown': 'After a quiet stretch, price drops out of its range to the downside — pent-up energy releasing into a fall.',
  'EMA momentum stack (down)': 'Fast averages line up below slow ones (8 < 21 < 50) — a clean downtrend with momentum behind it.',
}};
const TYPE_INFO = {{
  'trend': 'Trend-following: aims to ride a sustained move in one direction. Great in trending markets, whipsaws in choppy ones.',
  'momentum': 'Momentum: bets that recent strength (or weakness) keeps going a while longer.',
  'breakout': 'Breakout: acts when price escapes its recent range to a new high, expecting the move to continue.',
  'breakdown': 'Breakdown: acts when price escapes its recent range to a new low, expecting the fall to continue.',
  'mean-reversion': 'Mean-reversion: bets a short, sharp move snaps back toward the average — buy dips in uptrends, sell spikes in downtrends.',
}};
const FAMILY_INFO = {{
  'Trend-following': TYPE_INFO['trend'], 'Momentum': TYPE_INFO['momentum'],
  'Breakout': 'Breakout: acts when price escapes its recent range (a new high for longs, new low for shorts), expecting the move to continue.',
  'Mean-reversion': TYPE_INFO['mean-reversion'], 'Trend filter': TYPE_INFO['trend'],
}};
const _esc = t => String(t||'').replace(/"/g, '&quot;');

// Custom tooltip: reliable + instant (native title= is slow and easy to miss). Any element
// with a non-empty data-tip shows it on hover, positioned by the cursor.
const _tipEl = document.createElement('div'); _tipEl.id = 'tip'; document.body.appendChild(_tipEl);
function _placeTip(e) {{
  const pad = 14, w = _tipEl.offsetWidth, h = _tipEl.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
  _tipEl.style.left = Math.max(8, x) + 'px';
  _tipEl.style.top = Math.max(8, y) + 'px';
}}
document.addEventListener('mouseover', e => {{
  const t = e.target.closest && e.target.closest('[data-tip],[data-tiphtml]');
  if (!t) return;
  const h = t.getAttribute('data-tiphtml');
  if (h) {{ _tipEl.innerHTML = h; _tipEl.classList.add('rich'); _tipEl.style.display = 'block'; _placeTip(e); return; }}
  const txt = t.getAttribute('data-tip');
  if (txt) {{ _tipEl.textContent = txt; _tipEl.classList.remove('rich'); _tipEl.style.display = 'block'; _placeTip(e); }}
}});
document.addEventListener('mousemove', e => {{ if (_tipEl.style.display === 'block') _placeTip(e); }});
document.addEventListener('mouseout', e => {{
  if (e.target.closest && e.target.closest('[data-tip],[data-tiphtml]')) _tipEl.style.display = 'none';
}});

const diag = document.getElementById('diag');
if ((DATA.diagnostics||[]).length) {{
  diag.innerHTML = '<div style="background:#3a1e1e;border:1px solid #5a1e1e;color:#ff9b9b;'
    + 'border-radius:10px;padding:14px;margin:8px 0 18px;font-size:13px;">'
    + '<b>No signals to show.</b> Diagnostic:<br>'
    + DATA.diagnostics.map(e => '&bull; '+e).join('<br>') + '</div>';
}}
const cards = document.getElementById('cards');
// Small "first seen" age chip for a card (powered by the persisted first_seen date).
function _ageBit(s) {{
  if (!s.first_seen) return '';
  const txt = s.is_fresh ? 'New today' : (s.days_old === 1 ? '1 day old' : (s.days_old||0) + ' days old');
  return `<div class="card-age${{s.is_fresh ? ' fresh' : ''}}" title="First flagged ${{s.first_seen}}">🕒 ${{txt}}</div>`;
}}
function _alertBit(s) {{
  return s.alerted ? `<div class="card-age fresh" title="An ntfy alert fired for this name today — pinned here so your alerts and the dashboard stay in line">🔔 Alerted today</div>` : '';
}}
function _intradayBit(s) {{
  if (!s.intraday_confirm || s.intraday_confirm === 'none') return '';
  const ok = s.intraday_confirm === 'agree';
  const tf = (DATA.params && DATA.params.intraday_timeframe) || '5m';
  return `<div class="card-age${{ok ? ' fresh' : ''}}"${{ok ? '' : ' style="color:var(--sell);"'}} title="Lower-timeframe (${{tf}}) momentum ${{ok ? 'agrees with' : 'is against'}} this trade">⚡ Intraday ${{ok ? '✓' : '✗'}}</div>`;
}}
function makeCard(s) {{
  const el = document.createElement('div'); el.className='card';
  const cls = (s.action||'').replace(' ','');
  const conv = s.conviction || {{}};
  const cpct = conv.score_pct || 0;
  const ccol = _rag(cpct);   // RAG: green ≥70, amber ≥50, red below
  // Prefer Yahoo's consolidated price + previous close so the card matches Google.
  const _px = (s.quote_price != null) ? s.quote_price : s.price;
  const _base = (s.prev_close != null) ? s.prev_close : s.price;
  const _hasQ = (s.quote_price != null && s.prev_close);
  const _dc = _hasQ ? (s.quote_price / s.prev_close - 1) * 100 : (s.context && s.context.day_change_pct);
  const _lab = _hasQ ? 'today' : ('on ' + _shortDate(s.as_of));
  // refreshLive() recomputes this live vs the previous close and relabels it "today".
  const dchg = (_dc != null)
    ? `<span class="card-day" data-chg="${{s.symbol}}" data-base="${{_base}}" style="color:${{_dc>=0?'var(--buy)':'var(--sell)'}};">${{_dc>=0?'+':''}}${{_dc.toFixed(2)}}% ${{_lab}}</span>`
    : '';
  const initials = (s.symbol.replace(/[^A-Za-z]/g,'').slice(0,2) || s.symbol.slice(0,2)).toUpperCase();
  const logo = `<span class="card-mono" style="background:hsl(${{_symHue(s.symbol)}},42%,42%);">${{initials}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{s.symbol}}.png" alt="" loading="lazy" onerror="this.remove()" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#fff;">`
    + `</span>`;
  const _isShort = (s.direction === 'SHORT');
  const _cn = (s.strategies&&s.strategies.now) ? s.strategies.now : null;
  const _cs = (s.strategies&&s.strategies.short) ? s.strategies.short : null;
  const ed = (s.fundamentals||{{}}).earnings_days;
  const edGated = (s.conviction||{{}}).earnings_gated;
  const edWarn = edGated
    ? `<div class="card-warn">⛔ Earnings in ${{ed}}d — held back from a fresh entry (a report this close can gap through the stop)</div>`
    : (ed!=null && ed<=7)
    ? `<div class="card-warn">⚠ Earnings in ${{ed}}d — event risk around the report</div>` : '';
  // direction-aware price ladder: Target / Entry / Stop, ordered so higher price sits higher.
  const _p = s.plan || {{}};
  let ladder = '';
  if (_p.entry!=null && _p.stop!=null && _p.target!=null) {{
    const _m = v => '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}});
    const tgt = `<div class="lad-row tgt hint" data-tip="${{_esc('Target basis: ' + (_p.target_basis || 'nearest structural level, bounded by fundamentals & volatility'))}}"><span>Target</span><span>${{_m(_p.target)}}<em>${{_isShort?'−':'+'}}${{_p.target_pct}}%</em></span></div>`;
    const ent = `<div class="lad-row ent"><span>Entry</span><span>${{_m(_p.entry)}}</span></div>`;
    const stp = `<div class="lad-row stp"><span>Stop</span><span>${{_m(_p.stop)}}<em>${{_isShort?'+':'−'}}${{_p.stop_pct}}%</em></span></div>`;
    const rr = (_p.rr!=null) ? `<div class="lad-rr">Reward : risk &nbsp; 1 : ${{_p.rr}}</div>` : '';
    ladder = `<div class="ladder">${{_isShort ? (stp+ent+tgt) : (tgt+ent+stp)}}${{rr}}</div>`;
  }}
  // "Why this signal" — a clear panel naming the strategies behind the decision.
  // Trigger strategies (the catalyst) are filled pills; supporting ones are outlined.
  const _co = _isShort ? _cs : _cn;
  const _actWord = (s.action==='SHORT'||s.action==='HOLD SHORT'||s.action==='WATCH SHORT') ? 'short'
                 : (s.action==='BUY'||s.action==='HOLD LONG'||s.action==='WATCH LONG') ? 'buy' : 'signal';
  // strategy FAMILY (approach) behind this card — derived from the firing strategies' kind.
  const _FAMILY = {{trend:'Trend-following', momentum:'Momentum', breakout:'Breakout',
                    breakdown:'Breakout', 'mean-reversion':'Mean-reversion'}};
  let whyBody = '', famLabel = '';
  if (_co) {{
    const agree = (_isShort ? (_co.short||[]) : (_co.long||[]));
    const fresh = _co.fresh || [];
    if (agree.length) {{
      const pills = agree.slice(0,6).map(n => {{
        const tip = (STRAT_INFO[n]||'') + (fresh.includes(n) ? '  (this is the fresh trigger)' : '');
        return `<span class="why-chip${{fresh.includes(n)?' trig':''}} hint" data-tip="${{_esc(tip)}}">${{n}}</span>`;
      }});
      const extra = agree.length>6 ? `<span class="why-chip more">+${{agree.length-6}}</span>` : '';
      whyBody = `<div class="why-chips">${{pills.join('')}}${{extra}}</div>`;
      // collect the families of the agreeing strategies (most common first)
      const counts = {{}};
      if (_co.results) Object.values(_co.results).forEach(r => {{
        if (_isShort ? r.short : r.long) {{ const f=_FAMILY[r.kind]||r.kind; counts[f]=(counts[f]||0)+1; }}
      }});
      famLabel = Object.keys(counts).sort((a,b)=>counts[b]-counts[a]).slice(0,2).join(' + ') || 'Multi-strategy';
    }}
  }}
  if (!whyBody && s.action==='EXIT') {{ whyBody = `<div class="why-txt">Trend break — its uptrend just rolled over.</div>`; famLabel='Trend-following'; }}
  if (!whyBody && s.action==='AVOID') {{ whyBody = `<div class="why-txt">Below trend with a weak/bearish setup — stay away.</div>`; famLabel='Trend filter'; }}
  const famTip = (famLabel || '').split(' + ').map(f => FAMILY_INFO[f]).filter(Boolean).join('  •  ')
                 || 'the strategy approach behind this signal';
  const famTag = famLabel ? `<span class="why-fam hint" data-tip="${{_esc(famTip)}}">${{famLabel}}</span>` : '';
  const whyHtml = whyBody
    ? `<div class="card-why"><div class="why-h">📋 Why this ${{_actWord}}</div>${{famTag}}${{whyBody}}</div>` : '';
  const nNews = (s.news||[]).length;
  el.innerHTML = `
    <div class="card-top">${{logo}}
      <div class="card-id"><div class="s">${{s.symbol}}</div><div class="n">${{s.name||s.exchange||''}}</div>${{_ageBit(s)}}${{_alertBit(s)}}${{_intradayBit(s)}}</div>
      <span class="act a-${{cls}}">${{s.action}}</span>
      <button class="favbtn ${{FAVS.has(s.symbol)?'on':''}}" title="Save to favorites">${{FAVS.has(s.symbol)?'★':'☆'}}</button></div>
    <div class="card-px-row"><span class="card-px" data-px="${{s.symbol}}">$${{_px.toLocaleString()}}</span>${{dchg}}</div>
    <div class="card-spark">${{_spark2(s.symbol, _dirCol(s), 300, 42)}}</div>
    ${{conv.label ? `<div class="conv-wrap"><div class="conv-row"><span>Conviction · ${{conv.label}}</span><span style="color:${{ccol}};font-weight:700;">${{cpct}}%</span></div>`
      + `<div class="conv-meter"><div class="conv-fill" style="width:${{cpct}}%;background:${{ccol}};"></div></div></div>` : ''}}
    ${{ladder}}
    ${{whyHtml}}
    ${{s.catalyst ? `<div class="cat-chip hint" data-tip="${{_esc(s.catalyst.headline)}}">⚡ Catalyst — fresh news${{s.catalyst.source?' · '+s.catalyst.source:''}}</div>` : ''}}
    ${{_altPills(s)}}
    ${{s.tv ? `<div class="tv-chip hint" data-tip="TradingView's aggregate technical rating (independent of our engine) — daily ${{s.tv.d||'n/a'}}, weekly ${{s.tv.w||'n/a'}}">TradingView: ${{s.tv.d||'—'}} <span style="opacity:.7;">· 1W ${{s.tv.w||'—'}}</span></div>` : ''}}
    ${{s.ai_read ? `<div class="ai-box hint" data-tip="${{_esc(s.ai_read.slice(0,600))}}"><span class="ai-h">🤖 AI analyst</span> ${{_esc(s.ai_read.split('. ')[0]).slice(0,130)}}…</div>` : ''}}
    ${{edWarn}}
    <div class="more">${{nNews ? nNews+' news &middot; ':''}}click for chart, RSI, patterns + full breakdown →</div>`;
  const _fb = el.querySelector('.favbtn');
  if (_fb) _fb.addEventListener('click', (e) => {{
    e.stopPropagation(); _toggleFav(s.symbol);
    _fb.textContent = FAVS.has(s.symbol) ? '★' : '☆'; _fb.classList.toggle('on', FAVS.has(s.symbol));
    if (_curFilter === 'favs') renderCards();
  }});
  el.addEventListener('click', () => openModal(s));
  return el;
}}
// --- views: filter / sort the signal cards ---
const _ACT_ORDER = {{'BUY':0, 'SHORT':1, 'HOLD LONG':2, 'HOLD SHORT':3, 'EXIT':4,
                     'WATCH LONG':5, 'WATCH SHORT':6, 'AVOID':7, 'SELL':7, 'FLAT':8}};
const _conv = s => (s.conviction ? s.conviction.score_pct : -1);

// ===== Alternate layouts (view switcher) ===========================================
let _layout = 'terminal';
try {{ _layout = localStorage.getItem('layout') || 'terminal'; }} catch(e) {{}}
function _pxOf(s) {{ return (s.quote_price != null) ? s.quote_price : s.price; }}
function _rag(pct) {{ pct = pct||0; return pct>=70 ? 'var(--buy)' : (pct>=50 ? '#c08a1e' : 'var(--sell)'); }}
function _ragT(pct) {{ pct = pct||0; return pct>=70 ? '#33d17a' : (pct>=50 ? '#e8a33d' : '#ff5c4d'); }}
function _tvBit(s) {{ return (s.tv && s.tv.d) ? (' · TV ' + s.tv.d) : ''; }}  // short tail for compact rows

// Jump to a top-level tab programmatically (used by clickable alt-data badges).
function _gotoTab(p) {{ if (window._showPage) window._showPage(p); }}

// Signals-page TV widget: switch the embedded player between reliably-embeddable channels.
function _wtvSet(btn) {{
  const f = document.getElementById('wtvFrame');
  if (f) f.src = `https://www.youtube.com/embed/${{btn.dataset.wtv}}?autoplay=1&mute=1`;
  document.querySelectorAll('.wtvgrp button').forEach(b => b.classList.toggle('on', b === btn));
}}

// Scraped alt-data (SEC insiders / analyst rating change / StockTwits buzz) as a normalised list.
function _altData(s) {{
  const out = [];
  const ins = s.insider;
  if (ins && ins.cluster_buy)
    out.push({{icon:'🏛', txt:'Insider buys', extra:ins.buys, col:'var(--buy)',
      tip:`SEC Form 4: ${{ins.buys}} recent open-market insider purchase(s), ${{(ins.buy_shares||0).toLocaleString()}} shares.`}});
  const aa = (s.fundamentals||{{}}).analyst_actions, lt = aa && aa.latest;
  if (lt && (lt.action==='up' || lt.action==='down'))
    out.push({{icon: lt.action==='up'?'⬆':'⬇', txt: lt.action==='up'?'Upgrade':'Downgrade', extra: lt.firm||'',
      col: lt.action==='up'?'var(--buy)':'var(--sell)',
      tip:`Analyst: ${{_esc(lt.firm||'')}} ${{lt.from?lt.from+' → ':''}}${{_esc(lt.to||'')}} on ${{lt.date}}.`}});
  const b = s.buzz;
  if (b && b.lean)
    out.push({{icon:'💬', txt: b.lean==='bull'?'Bullish buzz':b.lean==='bear'?'Bearish buzz':'Mixed buzz',
      extra: b.sentiment_pct!=null?b.sentiment_pct+'%':'',
      col: b.lean==='bull'?'var(--buy)':b.lean==='bear'?'var(--sell)':'var(--muted)',
      tip:`StockTwits: ${{b.n}} recent posts, ${{b.sentiment_pct}}% bullish of tagged. Crowd sentiment — noisy/contrarian.`}});
  const rs = (s.factors||{{}}).rs;
  if (rs && rs.pct!=null)
    out.push({{icon:'📈', txt:'RS '+rs.pct, extra:'', nojump:true,
      col: rs.pct>=70?'var(--buy)':rs.pct<=40?'var(--sell)':'var(--muted)',
      tip:`Relative-strength percentile ${{rs.pct}} vs the market over recent months — higher = leading, lower = lagging.`}});
  return out;
}}
// Clear, labelled pills for the Cards layout.
function _altPills(s) {{
  const d = _altData(s); if (!d.length) return '';
  return '<div class="altrow">' + d.map(x => x.nojump
    ? `<span class="altpill hint" style="color:${{x.col}};" data-tip="${{x.tip}}">${{x.icon}} ${{x.txt}}${{x.extra!==''&&x.extra!=null?' '+x.extra:''}}</span>`
    : `<span class="altpill hint" style="color:${{x.col}};cursor:pointer;" data-tip="${{x.tip}} · Click for all findings →" onclick="event.stopPropagation();_gotoTab('altdata')">${{x.icon}} ${{x.txt}}${{x.extra!==''&&x.extra!=null?' '+x.extra:''}}</span>`
  ).join('') + '</div>';
}}
// Compact coloured icon strip for dense layouts (Terminal etc.).
function _altMini(s) {{
  const d = _altData(s); if (!d.length) return '';
  return '<div class="bbalt">' + d.map(x => x.nojump
    ? `<span class="hint" style="color:${{x.col}};" data-tip="${{x.tip}}">${{x.icon}} ${{x.extra!==''&&x.extra!=null?x.extra:x.txt}}</span>`
    : `<span class="hint" style="color:${{x.col}};cursor:pointer;" data-tip="${{x.tip}} · Click for all findings →" onclick="event.stopPropagation();_gotoTab('altdata')">${{x.icon}} ${{x.extra!==''&&x.extra!=null?x.extra:x.txt}}</span>`
  ).join('') + '</div>';
}}
function _dirCol(s) {{
  if (s.direction === 'SHORT') return 'var(--sell)';
  if (s.action === 'BUY' || s.action === 'HOLD LONG' || s.action === 'WATCH LONG') return 'var(--buy)';
  return 'var(--muted)';
}}
function _logo2(sym, px) {{
  const ini = (sym.replace(/[^A-Za-z]/g,'').slice(0,2) || sym.slice(0,2)).toUpperCase();
  return `<span class="mono2" style="width:${{px}}px;height:${{px}}px;font-size:${{Math.round(px*0.4)}}px;background:hsl(${{_symHue(sym)}},42%,42%);">${{ini}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{sym}}.png" alt="" loading="lazy" onerror="this.remove()">` + `</span>`;
}}
function _spark2(sym, color, w, h) {{
  const ch = (DATA.charts||{{}})[sym]; const c = (ch && ch.close ? ch.close : []).filter(x=>x!=null).slice(-40);
  if (c.length < 3) return `<svg viewBox="0 0 ${{w}} ${{h}}" style="width:${{w}}px;height:${{h}}px;"></svg>`;
  const mn=Math.min(...c), mx=Math.max(...c), rng=(mx-mn)||1;
  const pts = c.map((v,i)=>`${{(i/(c.length-1)*w).toFixed(1)}},${{(h-((v-mn)/rng)*(h-3)-1.5).toFixed(1)}}`).join(' ');
  return `<svg viewBox="0 0 ${{w}} ${{h}}" style="width:${{w}}px;height:${{h}}px;"><polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="1.5"/></svg>`;
}}
function _famOf(s) {{
  const co = (s.direction==='SHORT') ? (s.strategies&&s.strategies.short) : (s.strategies&&s.strategies.now);
  const F={{trend:'Trend',momentum:'Momentum',breakout:'Breakout',breakdown:'Breakout','mean-reversion':'Mean-rev'}};
  if(!co||!co.results) return '';
  const cnt={{}}; Object.values(co.results).forEach(r=>{{ if(s.direction==='SHORT'?r.short:r.long){{const f=F[r.kind]||r.kind;cnt[f]=(cnt[f]||0)+1;}}}});
  return Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a]).slice(0,2).join(' + ');
}}
function _levelsInline(s) {{
  const p=s.plan||{{}}; if(p.entry==null||p.stop==null||p.target==null) return '';
  return `${{p.entry}} · <span style="color:var(--sell);">${{p.stop}}</span> · <span style="color:var(--buy);">${{p.target}}</span>`;
}}
function _empty() {{ return '<div style="color:var(--muted);padding:14px;">Nothing matches this view right now.</div>'; }}
function _seenTs(s) {{ return Date.parse((s.first_seen || s.as_of || '') + 'T00:00:00') || 0; }}
function _applyFilter(list, f) {{
  if (f==='favs') return list.filter(s=>FAVS.has(s.symbol));
  if (f==='buys') return list.filter(s=>s.action==='BUY'||s.action==='HOLD LONG');
  if (f==='shorts') return list.filter(s=>s.action==='SHORT'||s.action==='HOLD SHORT');
  if (f==='watch') return list.filter(s=>s.action==='WATCH LONG'||s.action==='WATCH SHORT');
  if (f==='actionable') return list.filter(s=>['BUY','SHORT','HOLD LONG','HOLD SHORT','EXIT'].includes(s.action));
  return list;  // 'all'
}}
function _applySort(list, sort) {{
  if (sort==='conviction') list.sort((a,b)=>_conv(b)-_conv(a));
  else if (sort==='movers') list.sort((a,b)=>(b.rel_volume||0)-(a.rel_volume||0));
  else if (sort==='newest') list.sort((a,b)=>(_seenTs(b)-_seenTs(a))||(_conv(b)-_conv(a)));
  else list.sort((a,b)=>(_ACT_ORDER[a.action]-_ACT_ORDER[b.action])||(_conv(b)-_conv(a)));  // 'order' / 'sector' groups
  return list;
}}
function _reapplyLive() {{
  document.querySelectorAll('[data-px]').forEach(el => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[el.dataset.px] : null;
    if (p != null) el.textContent = _fmtPx(p);
  }});
}}
function _bindAll(container, list) {{
  const bySym = {{}}; list.forEach(s=>bySym[s.symbol]=s);
  container.querySelectorAll('[data-open]').forEach(el => {{
    el.addEventListener('click', () => {{ const s = bySym[el.getAttribute('data-open')]; if (s) openModal(s); }});
  }});
  return container;
}}
const _wrap = (cls, html) => {{ const d=document.createElement('div'); d.className=cls; d.innerHTML=html||_empty(); return d; }};

function _bbAct(s) {{
  if (s.direction==='SHORT') return '#ff5c4d';
  if (['BUY','HOLD LONG','WATCH LONG'].includes(s.action)) return '#33d17a';
  return '#9a9a78';
}}
function L_terminal(list, grouped) {{
  const r = DATA.regime || {{}};
  const buys = list.filter(s=>['BUY','HOLD LONG'].includes(s.action)).length;
  const shorts = list.filter(s=>['SHORT','HOLD SHORT'].includes(s.action)).length;
  const head = grouped ? '' : (`<div class="bbhead"><span class="bbtitle">SIGNAL DESK ▮</span>`
    + `<span class="bbst">REGIME <b>${{(r.label||'—').toUpperCase()}}</b></span>`
    + `<span class="bbst">BREADTH <b style="color:#fff;">${{r.breadth!=null?r.breadth+'%':'—'}}</b></span>`
    + `<span class="bbst">BUYS <b style="color:#33d17a;">${{buys}}</b></span>`
    + `<span class="bbst">SHORTS <b style="color:#ff5c4d;">${{shorts}}</b></span>`
    + `<span class="bbclock" id="bbclock"></span></div>`);
  const tiles = list.map(s=>{{
    const p = s.plan||{{}};
    const dc = (s.quote_price!=null && s.prev_close) ? (s.quote_price/s.prev_close-1)*100 : (s.context&&s.context.day_change_pct);
    const dcs = (dc!=null) ? `<span style="color:${{dc<0?'#ff5c4d':'#33d17a'}};">${{dc>=0?'+':''}}${{dc.toFixed(1)}}%</span>` : '';
    const lv = (p.entry!=null)
      ? `<span style="color:#33d17a;">T${{Math.round(p.target)}}</span> <span style="color:#cfcfcf;">E${{Math.round(p.entry)}}</span> <span style="color:#ff5c4d;">S${{Math.round(p.stop)}}</span>`
      : '—';
    return `<div class="bbtile" data-open="${{s.symbol}}"><div class="bbtop">${{_logo2(s.symbol,24)}}`
      + `<span class="bbsym">${{s.symbol}}</span><span class="bbact" style="color:${{_bbAct(s)}};">${{s.action}}</span></div>`
      + `<div class="bbpx"><span data-px="${{s.symbol}}">${{_pxOf(s).toLocaleString()}}</span> ${{dcs}}</div>`
      + `<div class="bbmeta">CONV <b style="color:${{_conv(s)>=0?_ragT(_conv(s)):'#8a8a6a'}};">${{_conv(s)>=0?_conv(s):'—'}}</b> · ${{_famOf(s)||'—'}}</div>`
      + `<div class="bblv">${{lv}}</div>`
      + (s.tv && s.tv.d ? `<div class="bbtv">TV <span style="color:#cbb88a;">${{s.tv.d}}</span> · 1W ${{s.tv.w||'—'}}</div>` : '')
      + _altMini(s)
      + `</div>`;
  }}).join('');
  return _bindAll(_wrap('bbwrap', head + `<div class="bbgrid">${{tiles}}</div>`), list);
}}
function _laneTip(s) {{
  const p = s.plan||{{}}, conv=_conv(s);
  const reason = (s.reasons||[]).find(r=>!/^📰|^📈/.test(r)) || (s.reasons||[])[0] || '';
  const lvl = (p.entry!=null) ? `Entry ${{p.entry}} · Stop ${{p.stop}} · Target ${{p.target}}${{p.rr!=null?' · R:R 1:'+p.rr:''}}` : '';
  return `<div style='font-weight:600;'>${{s.symbol}} · <span style='color:${{_dirCol(s)}};'>${{s.action}}</span></div>`
    + `<div style='color:var(--muted);font-size:11px;margin-bottom:5px;'>${{s.name||''}}</div>`
    + `<div>$${{_pxOf(s).toLocaleString()}} · Conviction ${{conv>=0?conv+'%':'—'}}</div>`
    + (_famOf(s)?`<div style='color:var(--muted);margin:3px 0;'>${{_famOf(s)}}</div>`:'')
    + (lvl?`<div>${{lvl}}</div>`:'')
    + (reason?`<div style='margin-top:5px;color:var(--muted);'>${{String(reason).slice(0,150)}}</div>`:'');
}}
function L_lanes(list) {{
  const lane = (title, col, items) => `<div class="lane"><div class="lanehd" style="color:${{col}};">${{title}} · ${{items.length}}</div>`
    + (items.map(s=>`<div class="lcard hint" data-open="${{s.symbol}}" data-tiphtml="${{_esc(_laneTip(s))}}" style="border-left-color:${{col}};">`
      + `<div class="lcard-t">${{_logo2(s.symbol,18)}}<span class="lsym">${{s.symbol}}</span><span class="lconv" style="color:${{_conv(s)>=0?_rag(_conv(s)):'var(--muted)'}};font-weight:700;">${{_conv(s)>=0?_conv(s)+'%':'—'}}</span></div>`
      + `${{_spark2(s.symbol, col, 150, 26)}}`
      + `<div class="lsub">${{_pxOf(s).toLocaleString ? '$'+_pxOf(s).toLocaleString() : ''}} · ${{_famOf(s)}}${{_tvBit(s)}}</div></div>`).join('') || '<div class="lsub" style="padding:6px;">—</div>') + '</div>';
  const buys = list.filter(s=>['BUY','HOLD LONG'].includes(s.action));
  const shorts = list.filter(s=>['SHORT','HOLD SHORT'].includes(s.action));
  const watch = list.filter(s=>['WATCH LONG','WATCH SHORT','EXIT','AVOID','FLAT'].includes(s.action));
  return _bindAll(_wrap('lanes', lane('↑ Long','var(--buy)',buys)+lane('↓ Short','var(--sell)',shorts)+lane('◷ Watch','var(--muted)',watch)), list);
}}
function L_gauges(list) {{
  const R=26, C=2*Math.PI*R;
  const cells = list.map(s=>{{
    const raw=_conv(s); const pc=Math.max(0,Math.min(100, raw>=0?raw:0)); const col=_rag(pc);
    const dash=(C*pc/100).toFixed(1);
    return `<div class="gauge" data-open="${{s.symbol}}"><svg viewBox="0 0 64 64" class="gsvg">`
      + `<circle cx="32" cy="32" r="${{R}}" fill="none" stroke="var(--inset)" stroke-width="6"/>`
      + `<circle cx="32" cy="32" r="${{R}}" fill="none" stroke="${{col}}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${{dash}} ${{(C-dash).toFixed(1)}}" transform="rotate(-90 32 32)"/>`
      + `<text x="32" y="38" text-anchor="middle" class="gnum">${{raw>=0?pc:'—'}}</text></svg>`
      + `<div class="glab">${{_logo2(s.symbol,16)}}<span>${{s.symbol}}</span></div>`
      + `<div class="gact" style="color:${{_dirCol(s)}};">${{s.action}}</div>`
      + (s.tv && s.tv.d ? `<div class="gtv">TV ${{s.tv.d}}</div>` : '') + `</div>`;
  }}).join('');
  return _bindAll(_wrap('gauges', cells), list);
}}
function L_feed(list) {{
  const rows = list.map(s=>{{
    const co=(s.direction==='SHORT')?(s.strategies&&s.strategies.short):(s.strategies&&s.strategies.now);
    const trig=(co&&co.fresh&&co.fresh[0])||_famOf(s)||'multiple methods';
    const verb = s.action.indexOf('WATCH')>=0?'is building a':(s.action.indexOf('HOLD')>=0?'is holding a':'triggered a');
    return `<div class="feeditem" data-open="${{s.symbol}}">${{_logo2(s.symbol,26)}}`
      + `<div class="feedtxt"><div><b>${{s.symbol}}</b> ${{verb}} <span style="color:${{_dirCol(s)}};font-weight:600;">${{s.action}}</span> — ${{trig}}</div>`
      + `<div class="feedsub">${{_conv(s)>=0?_conv(s)+'% conviction · ':''}}$${{_pxOf(s).toLocaleString()}}${{_tvBit(s)}} · as of ${{s.as_of||''}}</div></div>`
      + `<span class="feedspark">${{_spark2(s.symbol,_dirCol(s),96,30)}}</span></div>`;
  }}).join('');
  return _bindAll(_wrap('feedwrap', rows), list);
}}
function L_ticker(list, grouped) {{
  const tape = list.map(s=>`<span class="tkitem"><b>${{s.symbol}}</b> <span data-px="${{s.symbol}}" style="color:${{_dirCol(s)}};">$${{_pxOf(s).toLocaleString()}}</span></span>`).join('');
  const rows = list.map(s=>`<div class="tkrow" data-open="${{s.symbol}}">${{_logo2(s.symbol,22)}}`
    + `<span class="tksym">${{s.symbol}}</span><span style="color:${{_dirCol(s)}};font-weight:600;width:80px;">${{s.action}}</span>`
    + `<span class="tkpx" data-px="${{s.symbol}}">$${{_pxOf(s).toLocaleString()}}</span>`
    + `<span class="tkspark">${{_spark2(s.symbol,_dirCol(s),72,22)}}</span>`
    + `<span class="tkfam">${{_famOf(s)}}${{s.tv&&s.tv.d?' · TV '+s.tv.d:''}}</span><span class="tklv">${{_levelsInline(s)}}</span></div>`).join('');
  const tapeHtml = grouped ? '' : `<div class="tktape"><div class="tktape-in">${{tape}}${{tape}}</div></div>`;
  return _bindAll(_wrap('', `${{tapeHtml}}<div class="tkbody">${{rows}}</div>`), list);
}}
const LAYOUT_RENDER = {{terminal:L_terminal, lanes:L_lanes, gauges:L_gauges, feed:L_feed, ticker:L_ticker}};
// ===================================================================================

function _renderConcWarn() {{
  const el = document.getElementById('concWarn'); if (!el) return;
  const c = DATA.concentration;
  if (!c) {{ el.innerHTML = ''; return; }}
  el.innerHTML = `<div class="conc-warn" data-tip="${{c.symbols.join(', ')}}">`
    + `⚠ Concentration: ${{c.n}} of ${{c.total}} fresh ${{c.word}} are in <b>${{c.sector}}</b> (${{c.pct}}%). `
    + `These can be the same macro bet in disguise — sizing them as separate trades understates your real risk.</div>`;
}}
function renderCards() {{
  _renderConcWarn();
  cards.innerHTML = '';
  const useLayout = _layout && _layout !== 'cards' && LAYOUT_RENDER[_layout];
  const base = _applyFilter(DATA.signals.slice(), _curFilter);
  const emptyMsg = (_curFilter === 'favs')
    ? 'No favorites yet — tap the ☆ on any card to save it here.'
    : 'Nothing matches this view right now.';
  const flatGrid = (list) => {{
    const grid = document.createElement('div'); grid.className = 'grid';
    if (!list.length) grid.innerHTML = '<div style="color:var(--muted);">' + emptyMsg + '</div>';
    list.forEach(s => grid.appendChild(makeCard(s)));
    return grid;
  }};
  if (_curSort === 'sector') {{
    const by = {{}}, order = [];
    base.forEach(s => {{ const sec = s.sector || 'Other / Movers';
      if (!by[sec]) {{ by[sec] = []; order.push(sec); }} by[sec].push(s); }});
    if (!order.length) {{
      cards.appendChild(useLayout ? LAYOUT_RENDER[_layout]([]) : flatGrid([]));
    }}
    order.forEach(sec => {{
      const grp = _applySort(by[sec].slice(), 'order');
      const h = document.createElement('div'); h.className = 'secthead';
      h.textContent = sec + ' · ' + grp.length; cards.appendChild(h);
      cards.appendChild(useLayout ? LAYOUT_RENDER[_layout](grp, true) : flatGrid(grp));
    }});
  }} else {{
    const list = _applySort(base, _curSort);
    cards.appendChild(useLayout ? LAYOUT_RENDER[_layout](list) : flatGrid(list));
  }}
  // re-apply any live prices to the freshly rendered cards
  document.querySelectorAll('[data-px]').forEach(el => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[el.dataset.px] : null;
    if (p != null) el.textContent = _fmtPx(p);
  }});
}}
(function setupViews() {{
  // Sort = how the SAME set of cards is ordered (sector grouping, conviction, etc.)
  const sortBar = document.getElementById('sortBtns');
  const sorts = [['sector','By sector'],['order','Actionable first'],['newest','🕒 Newest'],
                 ['conviction','Highest conviction'],['movers','Biggest movers']];
  sorts.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.sort = v;
    if (v === _curSort) b.className = 'on';
    b.onclick = () => {{
      _curSort = v;
      try {{ localStorage.setItem('sort', v); }} catch(e) {{}}
      sortBar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.sort === v));
      renderCards();
    }};
    sortBar.appendChild(b);
  }});
  // Show = which cards to include (narrows the set); composes with Sort
  const filterBar = document.getElementById('filterBtns');
  const filters = [['all','All'],['buys','Longs'],['shorts','Shorts'],['watch','Watch'],
                   ['actionable','Actionable'],['favs','★ Favorites']];
  filters.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.filter = v;
    if (v === _curFilter) b.className = 'on';
    b.onclick = () => {{
      _curFilter = v;
      try {{ localStorage.setItem('filter', v); }} catch(e) {{}}
      filterBar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.filter === v));
      renderCards();
    }};
    filterBar.appendChild(b);
  }});
  // --- layout switcher (visual form: cards / terminal / lanes / …) ---
  const lbar = document.getElementById('layoutBtns');
  const layouts = [['cards','Cards'],['lanes','Lanes'],['terminal','Terminal'],
                   ['gauges','Gauges'],['feed','Feed'],['ticker','Ticker']];
  layouts.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.layout = v;
    if (v === _layout) b.className = 'on';
    b.onclick = () => {{
      _layout = v;
      try {{ localStorage.setItem('layout', v); }} catch(e) {{}}
      lbar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.layout === v));
      renderCards();
    }};
    lbar.appendChild(b);
  }});
  renderCards();
}})();

// --- Intraday tab: render the intraday-bar signals (reuses the same card UI) ---
function renderIntraday() {{
  const el = document.getElementById('intradayCards'); if (!el) return;
  const list = DATA.intraday || [];
  const tf = (DATA.params && DATA.params.intraday_timeframe) || '5Min';
  const it = DATA.intraday_track || {{}};
  const rec = it.resolved
    ? `Shadow record: <b>${{it.win_rate ?? '—'}}%</b> win over ${{it.resolved}} resolved · expectancy ${{(it.expectancy >= 0 ? '+' : '')}}${{it.expectancy ?? '—'}}% · ${{it.open || 0}} open`
    : `Shadow record: building — grades these ${{tf}} calls against real prices over the next few days (no orders placed)`;
  el.innerHTML = `<div class="strat-badge"><span class="k">Layer</span><span class="v">Intraday · ${{tf}} bars — faster &amp; noisier than the daily signals; confirm before acting</span></div>`
    + `<div class="note" style="margin:0 0 12px;">📊 ${{rec}}</div>`;
  const grid = document.createElement('div'); grid.className = 'grid';
  if (!list.length) {{
    grid.innerHTML = '<div style="color:var(--muted);font-size:13px;">No intraday signals this build (or intraday data was unavailable — it falls back silently, so the daily view is never affected).</div>';
  }} else {{
    list.forEach(s => grid.appendChild(makeCard(s)));
  }}
  el.appendChild(grid);
  document.querySelectorAll('#page-intraday [data-px]').forEach(elp => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[elp.dataset.px] : null;
    if (p != null) elp.textContent = _fmtPx(p);
  }});
}}
renderIntraday();

// --- momentum leaderboard rows open the same rich detail modal as the cards ---
(function bindMomentumRows() {{
  const det = DATA.mom_detail || {{}};
  document.querySelectorAll('tr.momrow').forEach(tr => {{
    tr.style.cursor = det[tr.dataset.sym] ? 'pointer' : 'default';
    tr.addEventListener('click', () => {{
      const s = det[tr.dataset.sym];
      if (s) openModal(s);
    }});
  }});
}})();

// ---- live prices (via Cloudflare Worker proxy) ----
const LIVE_SYMS = [...new Set(DATA.signals.map(s => s.symbol).concat('SPY'))];
function _fmtPx(v) {{ return '$' + Number(v).toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}}); }}
async function refreshLive() {{
  if (!LIVE_URL) return;
  const st = document.getElementById('liveStatus');
  try {{
    const r = await fetch(LIVE_URL + '?symbols=' + encodeURIComponent(LIVE_SYMS.join(',')));
    if (!r.ok) throw new Error('bad');
    const d = await r.json();
    LIVE = d.prices || {{}};
    document.querySelectorAll('[data-px]').forEach(el => {{
      const p = LIVE[el.dataset.px];
      if (p != null) el.textContent = _fmtPx(p);
    }});
    // live, correctly-labelled "today" change: live quote vs the last daily close
    document.querySelectorAll('[data-chg]').forEach(el => {{
      const p = LIVE[el.dataset.chg], base = parseFloat(el.dataset.base);
      if (p != null && base) {{
        const pct = (p / base - 1) * 100;
        el.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '% today';
        el.style.color = pct >= 0 ? 'var(--buy)' : 'var(--sell)';
      }}
    }});
    if (featTC) featTC.onLive(LIVE);
    if (modalTC) modalTC.onLive(LIVE);
    const tm = new Date(d.at || Date.now()).toLocaleTimeString('en-GB', {{timeZone:'GMT', hour12:false}}) + ' GMT';
    st.innerHTML = '&middot; <span style="color:#2ea043;">● Live</span> <span style="color:#8b97a6;">'+tm+'</span>';
  }} catch (e) {{
    st.innerHTML = '&middot; <span style="color:#8b97a6;">live prices unavailable</span>';
  }}
}}
if (LIVE_URL) {{ refreshLive(); setInterval(refreshLive, 15000); }}
// "last built X min ago" ticker for the build time (shown in GMT in the subhead).
(function builtAgo() {{
  const el = document.getElementById('builtAgo');
  const ts = (typeof DATA !== 'undefined' && DATA.generated_ts) ? DATA.generated_ts * 1000 : null;
  if (!el || !ts) return;
  function upd() {{
    const m = Math.max(0, Math.round((Date.now() - ts) / 60000));
    el.textContent = '· ' + (m < 1 ? 'just now' : (m === 1 ? '1 min ago' : m + ' min ago'));
  }}
  upd(); setInterval(upd, 30000);
}})();
// Live clock: current time in GMT and GMT+4, plus the US market window + open/closed status.
(function marketClock() {{
  const el = document.getElementById('marketClock'); if (!el) return;
  const t = (tz) => new Date().toLocaleTimeString('en-GB', {{timeZone:tz, hour12:false, hour:'2-digit', minute:'2-digit'}});
  function isOpen() {{
    try {{
      const et = new Date(new Date().toLocaleString('en-US', {{timeZone:'America/New_York'}}));
      const d = et.getDay(), m = et.getHours()*60 + et.getMinutes();
      return d >= 1 && d <= 5 && m >= 570 && m < 960;   // Mon–Fri, 09:30–16:00 ET
    }} catch (e) {{ return false; }}
  }}
  function upd() {{
    const open = isOpen();
    el.innerHTML = '🕒 ' + t('GMT') + ' GMT &middot; ' + t('Asia/Dubai') + ' GMT+4'
      + ' &middot; NYSE 09:30–16:00 ET ('
      + (open ? '<span style="color:var(--buy);font-weight:600;">open</span>'
              : '<span style="color:var(--muted);font-weight:600;">closed</span>') + ')';
  }}
  upd(); setInterval(upd, 1000);
}})();
// live ET clock for the terminal header (only present when Terminal layout is active)
setInterval(() => {{
  const c = document.getElementById('bbclock');
  if (c) try {{ c.textContent = new Date().toLocaleTimeString('en-US', {{hour12:false, timeZone:'America/New_York'}}) + ' ET'; }} catch(e) {{}}
}}, 1000);

// Notice when a newer build has been published (the Action rebuilds every ~30 min in market
// hours) and offer a one-click refresh — never reloads from under the user.
(function watchForNewBuild() {{
  const cur = (typeof DATA !== 'undefined' && DATA.generated_at) || '';
  async function check() {{
    if (document.hidden) return;
    try {{
      const r = await fetch('signals.json?cb=' + Date.now(), {{cache: 'no-store'}});
      if (!r.ok) return;
      const d = await r.json();
      if (d.generated_at && d.generated_at !== cur && !document.getElementById('newbuild')) {{
        const b = document.createElement('button');
        b.id = 'newbuild';
        b.textContent = '↻ New data available — refresh';
        b.onclick = () => location.reload();
        document.body.appendChild(b);
      }}
    }} catch (e) {{}}
  }}
  setInterval(check, 300000);  // check every 5 minutes
}})();

// ---- detail modal ----
const overlay = document.getElementById('overlay');
// Detailed, readable breakdown of every alt-data input behind a signal (for the Signals sub-tab).
function _signalsDetail(s) {{
  const short = s.direction === 'SHORT';
  const cards = [];
  const card = (icon, title, value, tone, why) =>
    `<div class="sigdet ${{tone||''}}"><div class="sigdet-h">${{icon}} <b>${{title}}</b>`
    + `<span class="sigdet-v">${{value}}</span></div><div class="sigdet-why">${{why}}</div></div>`;

  // Relative strength
  const rs = (s.factors||{{}}).rs;
  if (rs && rs.pct!=null) {{
    const lead = rs.pct>=70, lag = rs.pct<=40;
    cards.push(card('📈','Relative strength', 'RS '+rs.pct+' percentile',
      lead?'good':lag?'bad':'warn',
      lead ? 'Outrunning most of the market — leadership, which tends to persist.'
           : lag ? 'Lagging the market — relative weakness.'
                 : 'Middle of the pack versus the market — no strong lead either way.'));
  }}
  // TradingView
  if (s.tv && (s.tv.d||s.tv.w)) {{
    const agree = (!short && /Buy/.test(s.tv.d||'')) || (short && /Sell/.test(s.tv.d||''));
    cards.push(card('🟦','TradingView rating', (s.tv.d||'—')+' daily · '+(s.tv.w||'—')+' weekly',
      agree?'good':'warn',
      'An independent technical read (≈26 indicators), separate from our engine. '
      + (agree ? 'It lines up with this '+(short?'short':'long')+'.' : 'It does not strongly confirm this side — weigh it.')));
  }}
  // Insider (SEC Form 4)
  const ins = s.insider;
  if (ins && ins.n_filings) {{
    if (ins.cluster_buy)
      cards.push(card('🏛','Insider activity (SEC Form 4)', ins.buys+' open-market buy(s)', short?'bad':'good',
        ins.buys+' insider purchase(s) of '+(ins.buy_shares||0).toLocaleString()+' shares recently — real money down. '
        + (short?'A headwind for a short.':'A bullish vote of confidence.')));
    else if (ins.sells>=2)
      cards.push(card('🏛','Insider activity (SEC Form 4)', ins.sells+' sale(s)', short?'good':'warn',
        ins.sells+' insider sale(s) recently' + (short?' — supports the short.':' and little buying — mild caution.')));
    else
      cards.push(card('🏛','Insider activity (SEC Form 4)', 'no clear cluster', '',
        'No notable cluster of insider buys or sells in recent filings.'));
  }}
  // Analyst rating changes
  const aa = (s.fundamentals||{{}}).analyst_actions, lt = aa && aa.latest;
  if (lt && (lt.action==='up'||lt.action==='down')) {{
    const up = lt.action==='up';
    cards.push(card(up?'⬆':'⬇','Analyst rating change',
      (lt.firm||'analyst')+': '+(lt.from?lt.from+' → ':'')+(lt.to||''), up?(short?'bad':'good'):(short?'good':'warn'),
      '60-day net: '+(aa.n_up||0)+' upgrades / '+(aa.n_down||0)+' downgrades. Latest on '+(lt.date||'')+'. '
      + (up?'Fresh upgrades are a supportive catalyst for a long.':'Net downgrades lean bearish.')));
  }}
  // Retail buzz
  const b = s.buzz;
  if (b && b.lean) {{
    const lean = b.lean==='bull'?'Bullish':b.lean==='bear'?'Bearish':'Mixed';
    cards.push(card('💬','Retail buzz (StockTwits)', lean+' · '+(b.sentiment_pct)+'% bullish',
      b.lean==='mixed'?'warn':'',
      (b.n)+' recent posts tagged; '+(b.sentiment_pct)+'% bullish. Crowd sentiment is noisy and often contrarian — weighted gently.'));
  }}
  // News-driven idea (LLM read of headlines)
  const ni = s.news_idea;
  if (ni && ni.direction) {{
    const aligns = (ni.direction==='bearish') === short;
    cards.push(card('🗞','News-driven read', ni.direction+' · '+(ni.confidence||'')+' conf',
      aligns?'good':'warn',
      _esc(ni.reason||'') + (ni.headline?` (from: “${{_esc(ni.headline)}}”)`:'')));
  }}
  // Catalyst (fresh news)
  if (s.catalyst) {{
    cards.push(card('⚡','News catalyst', (s.catalyst.source||'news'),
      '', 'Fresh headline driving attention: “'+_esc(s.catalyst.headline||'')+'”.'));
  }}
  // News tone
  const sent = s.sentiment;
  if (sent && sent.label && sent.label!=='Neutral') {{
    cards.push(card('📰','News tone', sent.label, sent.label==='Positive'?(short?'bad':'good'):(short?'good':'warn'),
      'Overall tone across '+(sent.n||'recent')+' headlines reads '+sent.label.toLowerCase()+'.'));
  }}
  return cards.length ? cards.join('')
    : '<div class="sigdet"><div class="sigdet-why">No extra signal data for this name on this run '
      + '(insider/analyst/buzz data is sparse and only appears on live runs).</div></div>';
}}

// ---- per-headline news implication (heuristic keyword read; no API cost) ----
const _NEWS_POS = ['beat','beats','surge','surges','soar','soars','rally','rallies','gain','gains','jump','jumps',
  'upgrade','upgraded','upgrades','outperform','growth','grow','grows','profit','profits','record','strong',
  'bullish','rebound','rebounds','firepower','tops','wins','win','approval','approved',
  'expansion','breakthrough','momentum','boost','boosts','optimistic','upside','accelerate','rise','rises',
  'soaring','green','greenlight','demand','partnership','buyback','dividend'];
const _NEWS_NEG = ['miss','misses','plunge','plunges','fall','falls','drop','drops','decline','declines','sink',
  'sinks','slump','downgrade','downgraded','downgrades','lawsuit','probe','investigation','fraud','defeat','risk',
  'risks','loss','losses','weak','bearish','warning','warns','recall','halt','ban','fine','fined','slash','slashes',
  'layoff','layoffs','bankruptcy','default','concern','concerns','fears','plummet','tumble','selloff','sue','sued',
  'delay','delays','disappoint','disappoints','crash','slows','slowing','cuts','subpoena','dispute'];
function _newsLean(h) {{
  const words = (h||'').toLowerCase().replace(/[^a-z0-9\s-]/g,' ').split(/\s+/);
  let p=0,q=0; const hit=[];
  words.forEach(w => {{ if (_NEWS_POS.indexOf(w)>=0) {{ p++; if (hit.indexOf(w)<0 && hit.length<3) hit.push(w); }}
                       else if (_NEWS_NEG.indexOf(w)>=0) {{ q++; if (hit.indexOf(w)<0 && hit.length<3) hit.push(w); }} }});
  return {{ lean: p>q ? 'bull' : q>p ? 'bear' : 'flat', hit }};
}}
function _newsSent(h) {{ const l=_newsLean(h).lean;
  return l==='bull'?{{t:'Bullish',c:'var(--buy)'}}:l==='bear'?{{t:'Bearish',c:'var(--sell)'}}:{{t:'Neutral',c:'var(--muted)'}}; }}
function _newsImpact(h, s) {{ const l=_newsLean(h).lean, short=s.direction==='SHORT';
  if (l==='flat') return {{t:'Neutral for your',c:'var(--muted)',g:'•'}};
  const helps = (l==='bull') !== short;  // bullish helps a long; bearish helps a short
  return helps ? {{t:'Supports your',c:'var(--buy)',g:'▲'}} : {{t:'Works against your',c:'var(--sell)',g:'▼'}}; }}
function _newsTip(n, s) {{
  const r=_newsLean(n.headline), sym=s.symbol, short=s.direction==='SHORT', dirw=short?'short':'long';
  let lead, rel;
  if (r.lean==='bull') {{ lead=`Reads bullish for ${{sym}} — positive coverage / potential catalyst.`;
    rel = short ? `Bullish news pushes the price up, which works against your short.` : `Bullish news supports your long.`; }}
  else if (r.lean==='bear') {{ lead=`Reads bearish for ${{sym}} — negative coverage / risk flagged.`;
    rel = short ? `Bearish news pushes the price down, which backs your short.` : `Bearish news is a headwind for your long.`; }}
  else {{ lead=`Neutral / unclear for ${{sym}} — context, not a clear catalyst.`; rel=`No clear push for or against this ${{dirw}}.`; }}
  const flags = r.hit.length ? ` Flags: ${{r.hit.join(', ')}}.` : '';
  return `${{lead}}${{flags}} ${{rel}} (Heuristic read of the headline text — verify before acting.)`;
}}
function openModal(s) {{
  const cls = (s.action||'').replace(' ','');
  document.getElementById('mTitle').innerHTML =
    `<img class="logo" style="width:26px;height:26px;" src="https://assets.parqet.com/logos/symbol/${{s.symbol}}?format=png" alt="" onerror="this.style.display='none'"> ${{s.symbol}} <span class="act a-${{cls}}" style="float:none;font-size:13px;">${{s.action}}</span> &nbsp; <span data-px="${{s.symbol}}" style="color:var(--muted);font-size:15px;">$${{s.price.toLocaleString()}}</span>`
    + (s.name ? `<div class="cname" style="font-size:13px;margin-top:4px;">${{s.name}}${{s.exchange?` · ${{s.exchange}}`:''}}</div>` : '');
  document.getElementById('mSummary').textContent = s.summary || '';
  document.getElementById('mDesk').textContent = s.desk_read || '';
  const pel = document.getElementById('mPatterns');
  pel.innerHTML = (s.patterns||[]).length
    ? (s.patterns||[]).map(p => `<span class="chip ${{p.kind}}">${{p.label}}</span>`).join('')
    : '<span style="color:var(--muted);font-size:13px;">No standout chart patterns right now.</span>';
  // research: analysts, fundamentals, news tone
  const rel = document.getElementById('mResearch');
  const short = s.direction === 'SHORT';
  // color by whether a signal HELPS this trade's direction, not by raw bullishness:
  // on a short, bullish news/ratings/upside are headwinds (red), bearish are supportive (green).
  const dirCls = (bullish) => bullish === null ? '' : ((bullish !== short) ? 'buy' : 'sell');
  const fu = s.fundamentals || {{}}, an = fu.analysts, sen = s.sentiment;
  let rcells = '';
  const statc = (l,v,cls)=>`<div class="stat"><div class="l">${{l}}</div><div class="v ${{cls||''}}" style="font-size:15px;">${{v}}</div></div>`;
  if (an) {{
    const cc = dirCls(an.consensus==='Buy'?true:an.consensus==='Sell'?false:null);
    rcells += statc('Analyst consensus', an.consensus, cc) + statc('Buy / Hold / Sell', `${{an.buy}} / ${{an.hold}} / ${{an.sell}}`);
  }}
  if (fu.target_mean) {{
    const up = ((fu.target_mean/s.price-1)*100);
    rcells += statc('Avg price target', '$'+fu.target_mean.toLocaleString(), dirCls(up>=0))
            + statc('Upside to target', (up>=0?'+':'')+up.toFixed(0)+'%', dirCls(up>=0));
  }}
  if (fu.pe) rcells += statc('P/E ratio', fu.pe);
  if (fu.earnings_date) {{
    const ed = fu.earnings_days;
    rcells += statc('Next earnings', fu.earnings_date + (ed!=null?` (${{ed}}d)`:''), (ed!=null && ed<=7)?'sell':'');
  }}
  if (sen && sen.label) rcells += statc('News tone', sen.label, dirCls(sen.label==='Positive'?true:sen.label==='Negative'?false:null));
  rel.innerHTML = rcells || '<div style="color:var(--muted);font-size:13px;">No analyst/fundamental data available'
    + (sen ? '' : ' (add a Finnhub key to enable it)') + '.</div>';
  const eel = document.getElementById('mEdge'), e = s.edge;
  if (e && e.n_trades) {{
    const money = v => (v==null?'–':(v>0?'+':'')+v+'%');
    eel.innerHTML =
      `<div class="stat"><div class="l">Win rate</div><div class="v">${{e.win_rate==null?'–':e.win_rate+'%'}}</div></div>`
      + `<div class="stat"><div class="l">Past trades</div><div class="v">${{e.n_trades}}</div></div>`
      + `<div class="stat"><div class="l">Total return</div><div class="v ${{e.total_return>=0?'buy':'sell'}}">${{money(e.total_return)}}</div></div>`
      + `<div class="stat"><div class="l">Worst drawdown</div><div class="v sell">${{money(e.max_drawdown)}}</div></div>`;
  }} else {{
    eel.innerHTML = '<div style="color:var(--muted);font-size:13px;">Not enough past trades on this stock to measure an edge yet.</div>';
  }}
  // strategies in play: which independent methods are long now + their edge here
  const sel = document.getElementById('mStrategies'), sd = s.strategies || {{}};
  if (sel) {{
    const now = sd.now || {{}}, res = now.results || {{}}, edges = ((sd.edges || {{}}).by) || {{}};
    let chips = '';
    Object.keys(res).forEach(k => {{
      const r = res[k]; const cls = r.long ? (r.fresh ? 'bull' : 'neutral') : '';
      chips += `<span class="chip mini ${{cls}} hint" data-tip="${{_esc(STRAT_INFO[r.label] || r.kind)}}">${{r.long ? '●' : '○'}} ${{r.label}}</span>`;
    }});
    let rows = '';
    Object.keys(edges).forEach(k => {{
      const e = edges[k]; const wr = e.win_rate == null ? '–' : e.win_rate + '%';
      const ret = (e.total_return >= 0 ? '+' : '') + e.total_return + '%';
      const dir = e.side==='short' ? '<span style="color:var(--sell);" title="short strategy">▼</span> ' : '<span style="color:var(--buy);" title="long strategy">▲</span> ';
      rows += `<tr><td class="hcell" data-tip="${{_esc(STRAT_INFO[e.label] || '')}}">${{dir}}${{e.label}}</td>`
        + `<td class="hcell" style="color:var(--muted);" data-tip="${{_esc(TYPE_INFO[e.kind] || '')}}">${{e.kind}}</td>`
        + `<td style="text-align:right;">${{wr}}</td><td style="text-align:right;color:var(--muted);">${{e.n_trades}}</td>`
        + `<td style="text-align:right;" class="${{e.total_return >= 0 ? 'win' : 'loss'}}">${{ret}}</td></tr>`;
    }});
    const head = `<div style="margin-bottom:6px;font-size:13px;"><b>${{now.count || 0}}</b> of ${{now.total || 0}} strategies are long here right now (● long · ○ flat). <span style="color:var(--muted);">Hover any name for what it means.</span></div>`
      + `<div class="chips" style="margin-bottom:12px;">${{chips}}</div>`;
    const table = rows
      ? `<table class="trackrec"><thead><tr><th class="hcell" data-tip="The method being tested. Hover each row's name for a plain-English description.">Strategy</th><th class="hcell" data-tip="The family the method belongs to — hover for what each type means.">Type</th><th class="hcell" style="text-align:right;" data-tip="How often this method has been profitable on THIS stock historically.">Win</th><th class="hcell" style="text-align:right;" data-tip="How many completed trades that win rate is based on — more = more reliable.">Trades</th><th class="hcell" style="text-align:right;" data-tip="Total hypothetical return of this method on this stock (no fees/slippage).">Return</th></tr></thead><tbody>${{rows}}</tbody></table>`
      : '<div style="color:var(--muted);font-size:13px;">Per-strategy backtests are computed for the shown signals.</div>';
    sel.innerHTML = head + table;
  }}
  const aiHead = document.getElementById('mAIHead'), aiBox = document.getElementById('mAI');
  if (s.ai_read) {{
    aiBox.textContent = s.ai_read; aiBox.style.display = 'block'; aiHead.style.display = 'block';
  }} else {{
    aiBox.style.display = 'none'; aiHead.style.display = 'none';
  }}

  const conv = s.conviction || {{}};
  document.getElementById('mConvScore').innerHTML = conv.label
    ? `<span class="convbadge conv-${{conv.label}}">${{conv.label}} · ${{conv.score_pct}}% · ${{conv.passes}}/${{conv.total}} checks</span>`
    : '';
  const icon = {{pass:'✓', warn:'!', fail:'✗'}};
  document.getElementById('mChecks').innerHTML = (conv.checks||[]).map(c =>
    `<li class="${{c.status}}"><span class="ic">${{icon[c.status]}}</span>`
    + `<span><span class="ck-l">${{c.label}}</span> — <span class="ck-n">${{c.note}}</span></span></li>`
  ).join('');

  const p = s.plan || {{}}, ctx = s.context || {{}};
  const money = v => (v==null ? '–' : '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}}));
  const pct = v => (v==null ? '–' : (v>0?'+':'')+v+'%');
  const stat = (label, value, sub, cls) =>
    `<div class="stat"><div class="l">${{label}}</div><div class="v ${{cls||''}}">${{value}}</div>${{sub?`<div class="sub">${{sub}}</div>`:''}}</div>`;
  const _short = (s.direction === 'SHORT');
  const _active = (s.action==='BUY'||s.action==='HOLD LONG'||s.action==='SHORT'||s.action==='HOLD SHORT');
  const _dirWord = _short ? 'short' : 'long';
  document.getElementById('mPlanNote').textContent =
    _active ? `(${{_dirWord}} — active)` : `— levels if you took this ${{_dirWord}}`;
  const _scen = (Array.isArray(p.targets) && p.targets.length)
    ? `<div class="scen"><div class="scen-h">🎯 Target scenarios <span>— the order uses the Base case; the others are where you could scale out or run it</span></div>`
      + p.targets.map(t => {{
          const cls = (t.odds||'').replace(/ /g,'');
          return `<div class="scen-row ${{cls}}"><div class="scen-top"><b>${{t.label}}</b>`
            + `<span class="scen-px">${{money(t.price)}} <em>${{_short?'−':'+'}}${{Math.abs(t.pct)}}% · ${{t.r}}R · ${{t.odds}}</em></span></div>`
            + `<div class="scen-why">${{t.basis}}</div></div>`;
        }}).join('') + `</div>`
    : '';
  document.getElementById('mPlan').innerHTML =
    stat('Entry', money(p.entry), _short ? 'short here' : 'current price') +
    stat('Stop-loss', money(p.stop), `${{_short?'+':'−'}}${{p.stop_pct}}%  ·  ATR-based`, 'sell') +
    stat(_short ? 'Cover target' : 'Take-profit', money(p.target), `${{_short?'−':'+'}}${{p.target_pct}}%  ·  ${{p.target_basis||'base target'}}`, 'buy') +
    stat('Risk : Reward', p.rr!=null ? ('1 : '+p.rr) : '–', 'reward per $1 risked') +
    stat('Position size', (p.shares||0)+' sh', money(p.exposure)+' exposure') +
    stat('$ at risk', money(p.dollar_risk), `${{p.shares||0}} sh to stop`, 'sell') +
    _scen;
  document.getElementById('mContext').innerHTML =
    stat('Today', pct(ctx.day_change_pct)) +
    stat('Volatility (ATR)', money(ctx.atr), (ctx.atr_pct!=null?ctx.atr_pct+'% of price':'')) +
    stat('Vs trend line', pct(ctx.vs_slow_ma_pct), 'price vs slow MA') +
    stat('From recent high', pct(ctx.pct_from_high), money(ctx.period_high)) +
    stat('From recent low', pct(ctx.pct_from_low), money(ctx.period_low)) +
    stat('History', (ctx.history_bars||0)+' bars', 'data depth');

  document.getElementById('mReasons').innerHTML =
    (s.reasons||[]).map(r => `<li>${{r}}</li>`).join('') || '<li>No details available.</li>';
  const nl = document.getElementById('mNews');
  nl.innerHTML = (s.news||[]).length
    ? (s.news||[]).map(n => {{
        const t = n.url ? `<a href="${{n.url}}" target="_blank" rel="noopener">${{n.headline}}</a>`
                        : `<span class="h">${{n.headline}}</span>`;
        const se = _newsSent(n.headline), im = _newsImpact(n.headline, s);
        return `<li class="hint" data-tip="${{_esc(_newsTip(n, s))}}">${{t}}`
          + `<div class="src"><span style="color:${{se.c}};font-weight:600;">${{se.t}} for ${{s.symbol}}</span>`
          + ` &middot; <span style="color:${{im.c}};font-weight:600;">${{im.g}} ${{im.t}} ${{s.direction==='SHORT'?'short':'long'}}</span>`
          + ` &middot; ${{n.source||''}} ${{n.created_at||''}}</div></li>`;
      }}).join('')
    : '<li class="src">No recent news tagged for this symbol.</li>';
  // ---- Signals sub-tab: every alt-data input in detail, with plain-English reasoning ----
  document.getElementById('mSignals').innerHTML = _signalsDetail(s);

  // ===== Intelligence + Trade sub-views (meta / structured / nlp / rank / liquidity) =====
  const so = s.structured || {{}};
  const _rng = so.return_range || {{}};
  const _fmtPct = v => (v==null?'—':((v>0?'+':'')+v+'%'));
  // Risk & sizing
  const mRisk = document.getElementById('mRisk');
  if (mRisk) mRisk.innerHTML =
    stat('Confidence', (so.confidence!=null?so.confidence:'—'), '0–100 conviction') +
    stat('Expected value', _fmtPct(so.expected_value_pct), 'probability-weighted') +
    stat('Return range', (_rng.upside_pct!=null? (_fmtPct(_rng.upside_pct)+' / '+_fmtPct(_rng.downside_pct)) : '—'), 'target / stop') +
    stat('Reward : risk', (so.rr!=null?('1 : '+so.rr):'—'), '') +
    stat('Hold (est.)', (so.expected_hold_days!=null?(so.expected_hold_days+' sessions'):'—'), 'to target at typical move') +
    stat('Risk score', (so.risk_score!=null?(so.risk_score+'/100'):'—'), 'volatility + illiquidity', (so.risk_score>=66?'sell':'')) +
    stat('Uncertainty', (so.uncertainty!=null?(so.uncertainty+'/100 · '+(so.uncertainty_band||'')):'—'), 'disagreement / mixed macro / thin liq', (so.uncertainty_band==='high'?'sell':'')) +
    stat('Size rec.', (so.size_recommendation||'—'), 'after meta + regime', (so.size_recommendation==='Skip'?'sell':so.size_recommendation==='Full'?'buy':''));
  const kc = so.kill_conditions || {{}};
  const mKill = document.getElementById('mKill');
  if (mKill) mKill.innerHTML = 'Exit if the <b>stop</b> ('+(kc.stop_pct!=null?kc.stop_pct+'%':'—')+') is hit. The whole book de-risks at <b>'
    + (kc.book_drawdown_halt_pct!=null?kc.book_drawdown_halt_pct:'—')+'% drawdown</b> or a <b>'+(kc.daily_loss_limit_pct!=null?kc.daily_loss_limit_pct:'—')
    + '% daily loss</b>, and the kill switch halts trading after repeated run failures.';
  // Execution / liquidity
  const lq = s.liquidity || {{}};
  const _dv = lq.dollar_volume;
  const mExec = document.getElementById('mExec');
  if (mExec) mExec.innerHTML =
    stat('Liquidity tier', (lq.tier||'—'), 'by daily $ turnover', (lq.tier==='illiquid'||lq.tier==='thin'?'sell':'')) +
    stat('Avg $ volume', (_dv!=null? ('$'+(_dv>=1e9?(_dv/1e9).toFixed(1)+'B':(_dv/1e6).toFixed(0)+'M')+'/day') : '—'), 'how much trades hands') +
    stat('Est. spread', (lq.spread_bps!=null?(lq.spread_bps+' bps'):'—'), 'modeled half-spread') +
    stat('Liquidity score', (so.liquidity_score!=null?(so.liquidity_score+'/100'):'—'), 'execution quality');
  // Meta verdict
  const mv = s.meta;
  const mMeta = document.getElementById('mMeta');
  if (mMeta) {{
    if (!mv) mMeta.innerHTML = '<div class="deskread">No meta verdict for this name on this run.</div>';
    else {{
      const _dc = ({{accept:'var(--buy)',reduce:'var(--warn)',delay:'var(--muted)',reject:'var(--sell)'}})[mv.decision] || 'var(--muted)';
      mMeta.innerHTML = '<div class="deskread" style="border-left-color:'+_dc+';"><b style="color:'+_dc+';text-transform:capitalize;">'+mv.decision+'</b>'
        + ((mv.decision==='reduce'&&mv.size_factor!=null)?(' — size × '+mv.size_factor):'')
        + '<ul style="margin:8px 0 0;padding-left:18px;line-height:1.7;">'+(mv.reasons||[]).map(r=>'<li>'+r+'</li>').join('')+'</ul></div>';
    }}
  }}
  // Macro & regime fit
  const mp = DATA.macro_posture || {{}};
  const _fit = (s.rank_factors||{{}}).macrofit;
  const mRF = document.getElementById('mRegimeFit');
  if (mRF) {{
    let g = '<div class="plangrid">'
      + stat('Macro regime', (mp.label||'—'), (mp.score!=null?('composite '+mp.score):''))
      + stat('Exposure dial', (mp.exposure_mult!=null?(mp.exposure_mult+'×'):'—'), 'new-position sizing')
      + stat('This trade’s fit', (_fit!=null?(_fit+'/100'):'—'), 'direction vs regime', (_fit!=null&&_fit<40?'sell':_fit>=70?'buy':''))
      + (mp.entry_threshold?stat('Entry bar', mp.entry_threshold+'%', 'raised by regime'):'')
      + '</div>';
    const _tags = (mp.tags||[]).map(t=>'<span class="chip" title="'+_esc(t.why||'')+'">'+t.tag+'</span>').join(' ');
    if (_tags) g += '<div class="sech">Regime tags</div><div class="chips">'+_tags+'</div>';
    const _sb = mp.strategy_bias || {{}};
    if (_sb.favored) g += '<div class="sech">Favoured now</div><div style="font-size:13px;color:var(--txt2);">'+_sb.favored.join(' · ')+'</div>';
    mRF.innerHTML = g;
  }}
  // AI news read (LLM structured scores)
  const nlp = s.nlp;
  const mNR = document.getElementById('mNewsRead');
  if (mNR) {{
    if (!nlp) mNR.innerHTML = '<div class="deskread">No AI news read for this name this run (top actionable names only, live runs).</div>';
    else {{
      const dims = [['guidance','Guidance'],['demand_strength','Demand'],['management_confidence','Mgmt confidence'],['margin_pressure','Margin pressure'],['regulatory_risk','Regulatory risk'],['balance_sheet_concern','Balance-sheet'],['earnings_quality_risk','Earnings quality']];
      const cells = dims.map(d => {{ const v = nlp[d[0]]||0; const c = v>0?'var(--buy)':v<0?'var(--sell)':'var(--muted)';
        return '<div class="stat"><div class="l">'+d[1]+'</div><div class="v" style="color:'+c+';font-size:17px;">'+(v>0?'+':'')+v+'</div></div>'; }}).join('');
      const net = nlp.net; const nc = net>0.15?'var(--buy)':net<-0.15?'var(--sell)':'var(--muted)';
      mNR.innerHTML = '<div class="deskread">Net read: <b style="color:'+nc+';">'+(net>0?'+':'')+net+'</b>'+(nlp.note?(' — '+_esc(nlp.note)):'')+'</div>'
        + '<div class="plangrid">'+cells+'</div>'
        + '<p style="color:var(--muted);font-size:11px;margin:8px 0 0;">+ favourable, − a risk flag. Built from headlines only; it feeds the meta-model, it never places the trade.</p>';
    }}
  }}
  // Adaptive rank
  const rf = s.rank_factors, rs = s.rank_score;
  const mRk = document.getElementById('mRank');
  if (mRk) {{
    if (rs==null || !rf) mRk.innerHTML = '<div class="deskread">Not ranked (only actionable names get an allocation rank).</div>';
    else {{
      const bar = (lab,v) => {{ v=Math.max(0,Math.min(100,Math.round(v||0)));
        return '<div style="margin:7px 0;"><div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);"><span>'+lab+'</span><span>'+v+'</span></div>'
          + '<div style="height:7px;border-radius:4px;background:color-mix(in srgb,var(--accent) 14%,transparent);"><div style="height:100%;width:'+v+'%;border-radius:4px;background:var(--accent);"></div></div></div>'; }};
      mRk.innerHTML = '<div class="deskread">Allocation rank <b>'+rs+'</b>/100'+(s.rank?(' · #'+s.rank+' today'):'')+'</div>'
        + bar('Quality',rf.quality)+bar('Vol-adjusted reward',rf.vreward)+bar('Macro fit',rf.macrofit)+bar('Liquidity',rf.liquidity)+bar('Momentum',rf.momentum);
    }}
  }}

  // load this symbol into the modal's Capital IQ-style chart engine
  if (modalTC) modalTC.setSymbol(s.symbol, s.plan || {{}});
  if (window._mkShow) window._mkShow('overview');   // every open starts on Overview
  overlay.classList.add('open');
  try {{ history.replaceState(null, '', '#' + s.symbol); }} catch (e) {{}}   // shareable deep link
}}

// ---- Capital IQ-style chart engine: featured panel + watchlist + theme ----
function _symHue(s) {{ let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360; return h; }}
function _logoHTML(sym) {{
  const initials = (sym.replace(/[^A-Za-z]/g, '').slice(0, 2) || sym.slice(0, 2)).toUpperCase();
  const bg = `hsl(${{_symHue(sym)}},42%,42%)`;
  return `<span class="wl-logo" style="background:${{bg}};">${{initials}}`
    + `<img src="https://financialmodelingprep.com/image-stock/${{sym}}.png" alt="" loading="lazy" onerror="this.remove()">`
    + `</span>`;
}}
function buildWatchlist() {{
  const box = document.getElementById('featWatch'); if (!box) return;
  box.innerHTML = '';
  (DATA.signals || []).forEach((s, i) => {{
    const row = document.createElement('div');
    row.className = 'wl' + (i === 0 ? ' on' : ''); row.dataset.sym = s.symbol;
    const px = s.price != null ? ('$' + Number(s.price).toFixed(2)) : '';
    const dc = s.context && s.context.day_change_pct;
    const chg = (dc != null) ? `<span class="wl-chg" style="color:${{dc >= 0 ? 'var(--buy)' : 'var(--sell)'}};">${{dc >= 0 ? '+' : ''}}${{dc.toFixed(1)}}%</span>` : '';
    row.innerHTML = _logoHTML(s.symbol)
      + `<span class="wl-main"><span class="wl-sym">${{s.symbol}}</span>`
      + `<span class="wl-name">${{s.name || ''}}</span></span>`
      + `<span class="wl-r"><span class="wl-px" data-px="${{s.symbol}}">${{px}}</span>${{chg}}</span>`;
    row.onclick = () => {{
      box.querySelectorAll('.wl').forEach(x => x.classList.remove('on'));
      row.classList.add('on');
      if (featTC) featTC.setSymbol(s.symbol, s.plan || {{}});
    }};
    box.appendChild(row);
  }});
}}
function _initCharts() {{
  if (!window.TradeChart) {{ console.warn('chart engine not loaded'); return; }}
  const fEl = document.getElementById('featuredChart');
  const mEl = document.getElementById('modalChart');
  if (fEl) featTC = new TradeChart(fEl, {{ app: window.__APP, range: '6M', type: 'candle' }});
  if (mEl) modalTC = new TradeChart(mEl, {{ app: window.__APP, range: '6M', type: 'candle', compact: true }});
  buildWatchlist();
  const first = DATA.signals && DATA.signals[0];
  if (featTC && first) featTC.setSymbol(first.symbol, first.plan || {{}});
}}
// ---- light / dark theme ----
(function themeSetup() {{
  const KEY = 'tb-theme-v2';  // bumped: drops stale 'light' prefs so dark is the real default
  const btn = document.getElementById('themeToggle');
  function apply(t) {{
    document.documentElement.dataset.theme = t;
    if (btn) btn.textContent = (t === 'dark') ? '☀ Light' : '🌙 Dark';
    if (featTC) featTC.applyTheme();
    if (modalTC) modalTC.applyTheme();
  }}
  let cur = 'dark';
  try {{ cur = localStorage.getItem(KEY) || 'dark'; }} catch (e) {{}}
  document.documentElement.dataset.theme = cur;
  if (btn) {{
    btn.textContent = (cur === 'dark') ? '☀ Light' : '🌙 Dark';
    btn.onclick = () => {{
      const next = (document.documentElement.dataset.theme === 'dark') ? 'light' : 'dark';
      try {{ localStorage.setItem(KEY, next); }} catch (e) {{}}
      apply(next);
    }};
  }}
}})();
// ---- accent colour picker (persists; overrides --accent for both themes) ----
(function accentSetup() {{
  const KEY = 'tb-accent';
  const root = document.documentElement;
  const pop = document.getElementById('accentPop');
  const btn = document.getElementById('accentBtn');
  const cust = document.getElementById('accentCustom');
  if (!pop || !btn) return;
  function apply(c) {{ if (c) root.style.setProperty('--accent', c); else root.style.removeProperty('--accent'); }}
  function mark(c) {{ pop.querySelectorAll('.acsw').forEach(x => x.classList.toggle('on', x.dataset.accent === c)); }}
  let saved = null;
  try {{ saved = localStorage.getItem(KEY); }} catch (e) {{}}
  if (saved) {{ apply(saved); if (cust) cust.value = saved; mark(saved); }}
  btn.onclick = (e) => {{ e.stopPropagation(); pop.hidden = !pop.hidden; }};
  document.addEventListener('click', (e) => {{ if (!pop.hidden && !pop.contains(e.target) && e.target !== btn) pop.hidden = true; }});
  pop.querySelectorAll('.acsw').forEach(s => s.onclick = () => {{
    apply(s.dataset.accent); try {{ localStorage.setItem(KEY, s.dataset.accent); }} catch (e) {{}}
    if (cust) cust.value = s.dataset.accent; mark(s.dataset.accent); pop.hidden = true;
  }});
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') pop.hidden = true; }});
  if (cust) cust.oninput = () => {{ apply(cust.value); try {{ localStorage.setItem(KEY, cust.value); }} catch (e) {{}} mark(cust.value); }};
  const rst = document.getElementById('accentReset');
  if (rst) rst.onclick = () => {{ apply(null); try {{ localStorage.removeItem(KEY); }} catch (e) {{}} mark(null); pop.hidden = true; }};
}})();
// resize the featured chart when its panel becomes visible
function _refitCharts() {{
  try {{ if (featTC) featTC.resize(); }} catch (e) {{}}
}}
_initCharts();

function closeModal() {{ overlay.classList.remove('open'); try {{ history.replaceState(null, '', location.pathname + location.search); }} catch (e) {{}} }}
(function deepLink() {{
  function openFromHash() {{
    const h = (location.hash || '').replace('#', '').trim().toUpperCase();
    if (!h) return;
    const s = (DATA.signals || []).find(x => x.symbol === h);
    if (s) openModal(s);
  }}
  window.addEventListener('hashchange', openFromHash);
  openFromHash();   // open a shared #SYMBOL link on load
}})();
document.getElementById('modalClose').addEventListener('click', closeModal);
overlay.addEventListener('click', e => {{ if (e.target === overlay) closeModal(); }});
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

// ---- Live TV (pinned embeds) ----
// [key, label, videoId, watch-link]. We embed a SPECIFIC video id directly (no channel
// auto-resolve) because that reliably plays the exact feed we want — Bloomberg runs several
// concurrent streams, so auto-resolve grabbed the wrong one. Trade-off: a 24/7 stream that
// restarts gets a new id, so the embed can go blank until the id is refreshed -> the
// "open on YouTube" link below always works as the escape hatch.
const TV_CHANNELS = [
  ['bloomberg', 'Bloomberg', 'iEpJwprxDdk', 'https://www.youtube.com/@markets/live'],
  ['yahoo', 'Yahoo Finance', 'KQp-e_XQnDE', 'https://www.youtube.com/@YahooFinance/live'],
  ['schwab', 'Schwab Network', 'vKOd3v8VTYo', 'https://www.youtube.com/@SchwabNetwork/live'],
  ['cnbc', 'CNBC', '', 'https://www.youtube.com/@CNBC/live'],
];
let _tvCur = 'bloomberg';
try {{ _tvCur = localStorage.getItem('tvch') || 'bloomberg'; }} catch (e) {{}}
function _tvSet(key) {{
  const ch = TV_CHANNELS.find(c => c[0] === key) || TV_CHANNELS[0];
  _tvCur = ch[0];
  const f = document.getElementById('tvFrame');
  // No pinned id (e.g. CNBC, whose live is login-gated) -> blank the player; the link below covers it.
  if (f) f.src = ch[2] ? `https://www.youtube.com/embed/${{ch[2]}}?autoplay=1&mute=1` : 'about:blank';
  const lk = document.getElementById('tvLink'); if (lk) lk.href = ch[3];
  document.querySelectorAll('#tvBtns button').forEach(b => b.classList.toggle('on', b.dataset.tv === _tvCur));
  try {{ localStorage.setItem('tvch', _tvCur); }} catch (e) {{}}
}}
let _tvLoaded = false;
// Market sector heatmap — official TradingView Stock Heatmap widget, lazy-loaded on first open.
let _heatmapLoaded = false;
function _heatmapInit() {{
  if (_heatmapLoaded) return;
  const host = document.getElementById('heatmapHost');
  if (!host) return;
  _heatmapLoaded = true;
  const cont = document.createElement('div');
  cont.className = 'tradingview-widget-container';
  cont.style.height = '100%';
  const widget = document.createElement('div');
  widget.className = 'tradingview-widget-container__widget';
  widget.style.height = '100%';
  cont.appendChild(widget);
  const s = document.createElement('script');
  s.type = 'text/javascript';
  s.async = true;
  s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js';
  s.innerHTML = JSON.stringify({{
    dataSource: 'SPX500', exchanges: [], grouping: 'sector',
    blockSize: 'market_cap_basic', blockColor: 'change', locale: 'en',
    hasTopBar: true, isDataSetEnabled: true, isZoomEnabled: true, hasSymbolTooltip: true,
    colorTheme: (document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'),
    width: '100%', height: '100%'
  }});
  cont.appendChild(s);
  host.appendChild(cont);
}}
function _tvInit() {{
  const bar = document.getElementById('tvBtns');
  if (!bar || bar.childElementCount) return;
  TV_CHANNELS.forEach(c => {{
    const b = document.createElement('button'); b.textContent = c[1]; b.dataset.tv = c[0];
    if (c[0] === _tvCur) b.className = 'on';
    b.onclick = () => _tvSet(c[0]);
    bar.appendChild(b);
  }});
}}

// ---- tab navigation ----
(function setupTabs() {{
  // primary areas (sidebar) -> pages (top tabs). Pages not present in the DOM are filtered out.
  const AREAS = [
    ['signals', [['signals','Signals'],['intraday','Intraday'],['pairs','Pairs']]],
    ['markets', [['markets','Markets'],['heatmap','Heatmap'],['momentum','Momentum']]],
    ['portfolio', [['portfolio','Portfolio'],['paper','Paper account'],['allweather','All Weather']]],
    ['intel', [['altdata','Data signals'],['track','Track record']]],
    ['news', [['news','Market news'],['ipos','IPO watch'],['livetv','Live TV']]],
    ['about', [['method','How it works'],['system','System']]]
  ];
  AREAS.forEach(a => a[1] = a[1].filter(p => document.getElementById('page-' + p[0])));
  const sideNav = document.getElementById('sideNav');
  const topTabs = document.getElementById('topTabs');
  if (!sideNav || !topTabs) return;
  const areaOf = page => AREAS.find(a => a[1].some(p => p[0] === page)) || AREAS[0];
  function renderTop(area) {{
    topTabs.innerHTML = area[1].map(p => `<button data-page="${{p[0]}}">${{p[1]}}</button>`).join('');
  }}
  function show(page) {{
    const area = areaOf(page);
    if (!area[1].some(p => p[0] === page)) page = (area[1][0] || ['signals'])[0];
    sideNav.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.area === area[0]));
    if (!topTabs.querySelector(`[data-page="${{page}}"]`)) renderTop(area);
    topTabs.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.page === page));
    document.querySelectorAll('.page').forEach(s => s.classList.toggle('on', s.id === 'page-' + page));
    try {{ localStorage.setItem('tab', page); }} catch (e) {{}}
    window.scrollTo(0, 0);
    if (page === 'markets') setTimeout(() => {{ try {{ window.dispatchEvent(new Event('resize')); }} catch (e) {{}} _refitCharts(); }}, 60);
    if (page === 'livetv') {{ _tvInit(); if (!_tvLoaded) {{ _tvLoaded = true; _tvSet(_tvCur); }} }}
    if (page === 'heatmap') _heatmapInit();
  }}
  window._showPage = show;
  sideNav.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {{
    const area = AREAS.find(a => a[0] === b.dataset.area);
    if (area && area[1].length) {{ renderTop(area); show(area[1][0][0]); }}
  }}));
  topTabs.addEventListener('click', e => {{ const b = e.target.closest('button'); if (b && b.dataset.page) show(b.dataset.page); }});
  let saved = 'signals';
  try {{ saved = localStorage.getItem('tab') || 'signals'; }} catch (e) {{}}
  if (!document.getElementById('page-' + saved)) saved = 'signals';
  renderTop(areaOf(saved));
  show(saved);
}})();

// ---- modal sub-views (left rail) ----
(function setupModalViews() {{
  const top = document.getElementById('mkTop');
  const nav = document.getElementById('mkNav');
  const mk = document.querySelector('.mk');
  if (!top || !nav) return;
  const topBtns = top.querySelectorAll('button');
  const sideBtns = nav.querySelectorAll('button');
  function showView(v) {{
    sideBtns.forEach(b => b.classList.toggle('on', b.dataset.mkview === v));
    document.querySelectorAll('.mk-view').forEach(p => p.classList.toggle('on', p.id === 'mkview-' + v));
    if (v === 'chart') setTimeout(() => {{ try {{ if (modalTC) modalTC.resize(); }} catch (e) {{}} }}, 50);
  }}
  function showTop(t) {{
    topBtns.forEach(b => b.classList.toggle('on', b.dataset.top === t));
    let first = null, count = 0;
    sideBtns.forEach(b => {{
      const inGroup = b.dataset.top === t;
      b.style.display = inGroup ? '' : 'none';
      if (inGroup) {{ count++; if (!first) first = b; }}
    }});
    // single-view top tabs (Overview / Chart) go full width with no side rail
    if (nav) nav.style.display = count > 1 ? '' : 'none';
    if (mk) mk.style.gridTemplateColumns = count > 1 ? '' : '1fr';
    if (first) showView(first.dataset.mkview);
  }}
  topBtns.forEach(b => b.addEventListener('click', () => showTop(b.dataset.top)));
  sideBtns.forEach(b => b.addEventListener('click', () => showView(b.dataset.mkview)));
  // _mkShow(viewId): jump straight to a sub-view, activating its parent top tab too
  window._mkShow = function(v) {{
    const btn = nav.querySelector('[data-mkview="' + v + '"]');
    const t = btn ? btn.dataset.top : 'overview';
    showTop(t);
    showView(v);
  }};
}})();

// ---- Markets sub-views (left rail) ----
(function setupMarketViews() {{
  const nav = document.getElementById('mktNav'); if (!nav) return;
  const btns = nav.querySelectorAll('button');
  function show(v) {{
    btns.forEach(b => b.classList.toggle('on', b.dataset.mview === v));
    document.querySelectorAll('.mkt-view').forEach(p => p.classList.toggle('on', p.id === 'mview-' + v));
    setTimeout(() => {{ try {{ window.dispatchEvent(new Event('resize')); }} catch (e) {{}} _refitCharts(); }}, 50);
  }}
  btns.forEach(b => b.addEventListener('click', () => show(b.dataset.mview)));
}})();

const news = document.getElementById('news');
(DATA.news||[]).forEach(n => {{
  const li=document.createElement('li');
  const title = n.url ? `<a href="${{n.url}}" target="_blank" rel="noopener">${{n.headline}}</a>`
                      : `<span class="h">${{n.headline}}</span>`;
  const tag = (n.symbols&&n.symbols.length)?` [${{n.symbols.join(', ')}}]`:'';
  li.innerHTML = `${{title}}<div class="src">${{n.source}} ${{n.created_at}}${{tag}}</div>`;
  news.appendChild(li);
}});
if (!(DATA.news||[]).length) news.innerHTML = '<li class="src">No news for flagged symbols.</li>';

// build the overview chart last so nothing else can be blocked by it
</script></body></html>"""


def main() -> None:
    snap = build_snapshot()
    # Self-audit the data and record a health badge for the dashboard.
    try:
        import audit
        checks, flags = audit.audit_data(snap)
        errs = [f["msg"] for f in flags if f.get("level") == "error"]
        warns = [f["msg"] for f in flags if f.get("level") == "warn"]
        snap["data_health"] = {"ok": not errs, "checks": checks,
                               "errors": errs[:20], "warnings": warns[:20],
                               "n_err": len(errs), "n_warn": len(warns)}
        # Categorise the errors so we can see at a glance WHAT is failing (and surface it
        # early in the JSON via audit_summary so a truncated fetch still shows it).
        def _cat(msg):
            m = msg.lower()
            if "one-day jump" in m or "split" in m: return "split/jump >50%"
            if "ohlc" in m: return "OHLC violation"
            if "timestamp" in m: return "timestamps"
            if "bollinger" in m: return "bollinger order"
            if "3mo return" in m or "return" in m: return "window return"
            if "macro" in m: return "macro range"
            return "field/range"
        by_type = {}
        for m in errs:
            k = _cat(m); by_type[k] = by_type.get(k, 0) + 1
        snap["audit_summary"] = {"n_err": len(errs), "n_warn": len(warns),
                                 "by_type": by_type, "errors": errs[:25]}
        if errs or warns:
            print(f"DATA AUDIT: {len(errs)} error(s), {len(warns)} warning(s) over {checks} checks:")
            for fl in (errs + warns)[:25]:
                print("  -", fl)
        else:
            print(f"DATA AUDIT: clean ({checks} checks).")
    except Exception as exc:  # noqa: BLE001
        snap["data_health"] = None
        print("DATA AUDIT: skipped —", exc)
    with open("signals.json", "w") as f:
        json.dump(snap, f, indent=2)
    with open("dashboard.html", "w") as f:
        f.write(render_html(snap))
    print(f"[{snap['mode']}] scanned {snap['scanned']}, showing {len(snap['signals'])} "
          f"-> dashboard.html / signals.json @ {snap['generated_at']}")
    for s in snap["signals"]:
        rv = f" relvol {s['rel_volume']}x" if s.get("rel_volume") else ""
        print(f"  {s['symbol']}: {s['action']} @ ${s['price']} (RSI {s['rsi']}){rv}")


if __name__ == "__main__":
    main()
