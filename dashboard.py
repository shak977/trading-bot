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


def build_snapshot() -> dict:
    mode = _mode()
    live = mode != "SYNTHETIC"

    rows = scanner.scan(CONFIG, live=live)

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
    for r in shown:
        r["fundamentals"] = fundamentals.get(r["symbol"])
        # Re-score conviction + desk read now that research is in hand.
        scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"), fundamentals=r.get("fundamentals"), tv=r.get("tv"))

    # Ray Dalio All Weather allocation + backtest vs SPY (keyless Yahoo history).
    try:
        import allweather as _aw
        all_weather = _aw.build(live)
    except Exception:  # noqa: BLE001
        all_weather = None

    # Macro backdrop (FRED) — once per run.
    macro = None
    if live and CONFIG.fred_api_key:
        try:
            macro = research.fred_macro(CONFIG)
        except Exception:  # noqa: BLE001
            macro = None

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

    # Optional AI analyst note — only for the strongest, actionable setups (High conviction
    # BUY/SHORT/HOLD), capped, so we spend the API budget where it matters and keep builds fast.
    llm_status = {"enabled": bool(CONFIG.llm_enabled)}
    if CONFIG.llm_enabled:
        import llm
        _ai_picks = [r for r in shown
                     if (r.get("conviction") or {}).get("label") == "High"
                     and r.get("action") in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT")]
        _ai_picks.sort(key=lambda r: -((r.get("conviction") or {}).get("score_pct") or 0))
        llm_status["candidates"] = len(_ai_picks)
        _gen = 0
        for r in _ai_picks[:8]:
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
        track = tracker.run(shown, CONFIG, live, today)
    except Exception:  # noqa: BLE001
        track = None

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": mode,
        "scanned": len(rows),
        "diagnostics": list(scanner.LAST_ERRORS),
        "audit_summary": None,  # filled by main() after the audit — kept early so it survives a truncated fetch
        "news_sources": dict(__import__("collections").Counter(
            (n.get("source") or "?") for n in news).most_common(14)),
        "llm": llm_status,
        "benchmark": benchmark,
        "track": track,
        "regime": regime,
        "sectors": sectors,
        "macro": macro,
        "price_drops": price_drops,
        "momentum": [dict(m, name=scanner.name_of(
                        m["symbol"], {r["symbol"]: r.get("name", "") for r in shown}.get(m["symbol"], "")))
                     for m in momentum_rows],
        "mom_detail": _mom_detail(momentum_rows, rows_by_sym, shown),
        "allweather": all_weather,
        "portfolio": _portfolio(shown),
        "ipos": ipos,
        "ipo_news": ipo_news,
        "params": {
            "fast_ma": CONFIG.fast_ma, "slow_ma": CONFIG.slow_ma,
            "rsi_period": CONFIG.rsi_period, "risk_per_trade": CONFIG.risk_per_trade,
            "stop_loss_pct": CONFIG.stop_loss_pct, "take_profit_pct": CONFIG.take_profit_pct,
            "rel_volume_window": CONFIG.rel_volume_window,
        },
        "signals": shown,
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
        tiles += tile("Market regime", reg.get("label", "—"), tone, reg.get("note", "")[:46])
        tiles += tile("Breadth", f'{reg.get("breadth", "—")}%', "", f'of {reg.get("total","?")} above trend')
        tiles += tile("Avg momentum", f'{reg.get("avg_rsi", "—")}', "", "RSI, 0–100")
    tiles += tile("Fresh buys", str(n_buy), "buy" if n_buy else "", "new long setups")
    tiles += tile("Fresh shorts", str(n_short), "sell" if n_short else "", "new short setups")
    tiles += tile("Track record", wr_txt, "", f'{tk.get("resolved", 0)} calls resolved')
    return f'<div class="kpis">{tiles}</div>'


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


def _macro_html(m: dict | None) -> str:
    if not m:
        return ""
    def cell(label, val):
        return (f'<div class="stat"><div class="l">{label}</div>'
                f'<div class="v" style="font-size:15px;">{val}</div></div>')
    cells = ""
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
    return ('<div class="ovbox"><div class="ovhead">🌍 Macro backdrop: '
            f'{m.get("backdrop","")} <span style="font-weight:400;color:var(--muted);font-size:12px;">'
            f'— {m.get("note","")}</span></div>'
            f'<div class="trackstats">{cells}</div></div>')


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
    regime_html = _regime_html(snap.get("regime"))
    _pd = snap.get("price_drops") or []
    pdrop_html = (f' &middot; <span style="color:var(--muted);" title="{(" | ".join(_pd))[:300].replace(chr(34), chr(39))}">'
                  f'{len(_pd)} dropped (bad feed price)</span>') if _pd else ""
    kpi_html = _kpi_html(snap.get("regime"), snap)
    momentum_html = _momentum_html(snap.get("momentum") or [])
    allweather_html = _allweather_html(snap.get("allweather"))
    portfolio_html = _portfolio_html(snap.get("portfolio"))
    ipo_html = _ipo_html(snap.get("ipos") or [], snap.get("ipo_news") or [])
    sectors_html = _sectors_html(snap.get("sectors"))
    macro_html = _macro_html(snap.get("macro"))
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
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Signals Dashboard</title>
<script src="https://s3.tradingview.com/tv.js"></script>
<script src="chart_engine.js"></script>
<style>
  /* Light "Capital IQ Pro" palette is the default; dark is a toggle. */
  :root {{ --bg:#f5f7fa; --card:#ffffff; --line:#e4e8ed; --txt:var(--inset);
    --muted:#5b6776; --txt2:#3d4757; --buy:#0f9d58; --sell:#d1242f; --hold:#0b5cad; --flat:#8a96a3;
    --short:#c2410c; --watch:#475569; --exit:#b45309; --avoid:#6b7280;
    --accent:#0b5cad; --grid:rgba(120,130,145,0.16); --cross:rgba(60,70,85,0.4);
    --inset:#f1f4f8; --hover:#eef2f7; --ring:rgba(11,92,173,.40);
    --shadow:0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
    --shadow-lg:0 6px 20px rgba(16,24,40,0.10); }}
  html[data-theme="dark"] {{ --bg:#0d1117; --card:#161b22; --line:#262d36; --txt:#e6edf3;
    --muted:#8b97a6; --txt2:var(--txt2); --buy:#2ea043; --sell:#f85149; --hold:#58a6ff; --flat:#6e7681;
    --short:#fb7185; --watch:#94a3b8; --exit:#d29922; --avoid:#6e7681;
    --accent:#58a6ff; --grid:rgba(42,52,65,0.55); --cross:rgba(139,151,166,0.45);
    --inset:var(--inset); --hover:#1c2530; --ring:rgba(88,166,255,.45);
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
  ::-webkit-scrollbar {{ width:10px; height:10px; }}
  ::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:6px; border:2px solid transparent;
    background-clip:padding-box; }}
  ::-webkit-scrollbar-thumb:hover {{ background:var(--muted); background-clip:padding-box; }}
  ::-webkit-scrollbar-track {{ background:transparent; }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ transition:none !important; animation:none !important; scroll-behavior:auto !important; }}
    .card:hover {{ transform:none; }} }}
  html, body {{ max-width:100%; overflow-x:hidden; }}
  body {{ margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--txt); }}
  .wrap {{ width:100%; max-width:1480px; margin:0 auto; padding:28px 24px 60px; }}
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
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
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
  .stat {{ background:var(--inset); border:1px solid var(--line); border-radius:10px;
    padding:10px 12px; }}
  .stat .l {{ color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.04em; }}
  .stat .v {{ font-size:17px; font-weight:700; margin-top:2px; }}
  .stat .v.buy {{ color:var(--buy); }} .stat .v.sell {{ color:var(--sell); }}
  .stat .sub {{ color:var(--muted); font-size:11px; }}
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
  .tabs {{ display:flex; gap:4px; flex-wrap:wrap; border-bottom:1px solid var(--line);
    margin:18px 0 22px; position:sticky; top:0; z-index:10; padding-top:6px;
    background:color-mix(in srgb, var(--bg) 85%, transparent); backdrop-filter:saturate(1.3) blur(10px); }}
  .tabs button {{ background:none; border:none; color:var(--muted); font-size:15px;
    font-weight:600; padding:10px 16px; cursor:pointer; border-bottom:2px solid transparent; }}
  .tabs button.on {{ color:var(--txt); border-bottom-color:var(--accent); }}
  .tabs button:hover {{ color:var(--txt); }}
  .ctlbtn:hover, .ctlgrp button:hover {{ color:var(--txt); background:var(--hover); }}
  .page {{ display:none; }} .page.on {{ display:block; }}
  .secthead {{ font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase;
    letter-spacing:.05em; margin:22px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .secthead:first-child {{ margin-top:4px; }}
  .ovbox {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 16px; margin-bottom:22px; }}
  .ovhead {{ font-weight:700; font-size:14px; margin-bottom:8px; }}
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
  .appbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:14px 0; border-bottom:1px solid var(--line); margin-bottom:16px;
    position:sticky; top:0; z-index:20;
    background:color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter:saturate(1.3) blur(10px); }}
  .brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:18px; letter-spacing:-.01em; }}
  .brand-mark {{ display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px;
    border-radius:9px; background:var(--accent); color:#fff; font-size:15px; }}
  .appbar-right {{ display:flex; align-items:center; gap:10px; }}
  .livepill {{ font-size:12px; color:var(--muted); }}
  .subhead {{ color:var(--muted); font-size:13px; margin:0 0 14px; }}
  /* ---- KPI summary strip ---- */
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:12px; margin:0 0 20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; box-shadow:var(--shadow); }}
  .kpi-l {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:700; }}
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
    .wrap {{ padding:16px 12px 48px; }}
    h1 {{ font-size:21px; }}
    .appbar {{ flex-wrap:wrap; gap:8px; }}
    .tabs {{ overflow-x:auto; flex-wrap:nowrap; -webkit-overflow-scrolling:touch; }}
    .tabs button {{ white-space:nowrap; padding:10px 12px; font-size:14px; }}
    .kpis {{ grid-template-columns:repeat(2, minmax(0,1fr)); }}
    .trackrec {{ font-size:12px; }}
    .trackrec th, .trackrec td {{ padding:5px 6px; }}
    .modal {{ padding:16px; margin:12px auto; }}
    .tv-wrap {{ height:340px; }}
  }}
</style></head>
<body><div class="wrap">
  <header class="appbar">
    <div class="brand"><span class="brand-mark">◈</span><span>Signal Desk</span></div>
    <div class="appbar-right">
      <span class="badge m-{mode}">{mode}</span>
      <span class="livepill" id="liveStatus"></span>
      <button id="themeToggle" class="themebtn">🌙 Dark</button>
    </div>
  </header>
  <div class="subhead">Generated {snap['generated_at']} &middot; scanned {snap['scanned']} symbols{health_html}{pdrop_html}</div>
  {kpi_html}
  <div class="note" style="margin-top:0;">{mode_note}</div>
  <div id="diag"></div>

  <nav class="tabs" id="tabs">
    <button data-page="signals" class="on">Signals</button>
    <button data-page="markets">Markets</button>
    <button data-page="momentum">Momentum</button>
    <button data-page="portfolio">Portfolio</button>
    <button data-page="allweather">All Weather</button>
    <button data-page="ipos">IPO watch</button>
    <button data-page="track">Track record</button>
    <button data-page="method">How it works</button>
    <button data-page="news">Market news</button>
    <button data-page="livetv">Live TV</button>
  </nav>

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
    <div class="strat-badge"><span class="k">Strategy type</span><span class="v">Multi-strategy confluence · 7 long + 7 short, trend-gated</span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Layout:</span>
      <span class="ctlgrp" id="layoutBtns"></span></div>
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Sort &amp; filter:</span>
      <span class="ctlgrp" id="viewBtns"></span></div>
    <div id="cards"></div>
  </section>

  <section class="page" id="page-momentum">
    <h2 style="margin-top:0;">Momentum leaders <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— dual-momentum ranking (our best backtested strategy)</span></h2>
{momentum_html}
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
        way) and a <b>target</b> (~{snap['params']['take_profit_pct']:.0%} the right way), sized so a stop-out
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
      </ul>
      <p>The detail panel also flags <b>chart patterns</b> (golden cross, breakouts, pullbacks, MACD
      crosses, oversold bounces…) and reads the <b>market backdrop</b> — overall breadth (how many
      stocks are trending up) and which <b>sectors</b> are strongest — because signals work better when
      the broader tape agrees.</p>

      <h4>How to use it</h4>
      <p>Each card shows the action and a <b>conviction score</b> (how well it fits the rules). Click any
      card for the full breakdown: a plain-English explanation, the trade plan (entry, stop, target,
      risk:reward), a chart marking where the strategy would have bought/sold, and recent news.</p>

      <h4>Honest limits</h4>
      <p>This is an <b>educational tool, not financial advice</b>. Signals are often wrong, the data is
      free and slightly delayed, and the numbers ignore fees and slippage. Treat it as a starting point
      for your own research — never risk money you can't afford to lose.</p>
    </div>
  </section>

  <section class="page" id="page-news">
    <h2 style="margin-top:0;">Market news <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— recent headlines across the scanned stocks</span></h2>
    <ul class="news" id="news"></ul>
  </section>

  <section class="page" id="page-livetv">
    <h2 style="margin-top:0;">Live TV <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— Bloomberg Television, 24/7 markets stream</span></h2>
    <div class="tvwrap"><iframe id="tvFrame" data-src="https://www.youtube.com/embed/live_stream?channel=UCIALMKvObZNtJ6AmdCLP7Lg&autoplay=1&mute=1" title="Bloomberg Television live" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe></div>
    <p style="color:var(--muted);font-size:12px;margin-top:10px;">Live 24/7 markets stream from <b>Bloomberg Television</b> (@markets) via YouTube — starts muted; unmute in the player. If it doesn't load (occasional geo/embedding limits), <a href="https://www.youtube.com/@markets/live" target="_blank" rel="noopener">watch on YouTube ↗</a>. Not affiliated with Bloomberg; embedded for convenience.</p>
  </section>

  <div class="disclaimer">
    Strategy: multi-strategy confluence (7 long + 7 short), trend-gated (200-day) with a
    conviction floor; {snap['params']['fast_ma']}/{snap['params']['slow_ma']} SMA + RSI({snap['params']['rsi_period']}) is one input.
    Risk {snap['params']['risk_per_trade']:.0%}/trade, stop {snap['params']['stop_loss_pct']:.0%}, target {snap['params']['take_profit_pct']:.0%}.
    Shorts profit if price falls and carry higher risk.
    "Rel vol" = today's volume vs its {snap['params']['rel_volume_window']}-day average — a free
    proxy for unusual activity, NOT real institutional/options order flow.<br>
    Educational tool only. Not financial advice. Signals can be wrong; backtests ignore
    fees and slippage. Verify before acting and never risk money you can't afford to lose.
  </div>
</div>

<div class="overlay" id="overlay">
  <div class="modal modal-wide">
    <button class="close" id="modalClose">&times;</button>
    <h3 id="mTitle"></h3>
    <div class="summary" id="mSummary"></div>
    <div class="mk">
      <nav class="mk-side" id="mkNav">
        <button data-mkview="overview" class="on">Overview</button>
        <button data-mkview="chart">Chart</button>
        <button data-mkview="plan">Trade plan</button>
        <button data-mkview="strategies">Strategies</button>
        <button data-mkview="research">Research &amp; news</button>
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
        <div class="mk-view" id="mkview-research">
          <div class="sech" style="margin-top:0;">Analysts, fundamentals &amp; news tone</div>
          <div class="plangrid" id="mResearch"></div>
          <div class="sech">Latest news on this stock</div>
          <ul class="news" id="mNews"></ul>
          <div class="sech">The details, explained</div>
          <ul class="reasons" id="mReasons"></ul>
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
let _curView = 'sector';
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
  const edWarn = (ed!=null && ed<=7)
    ? `<div class="card-warn">⚠ Earnings in ${{ed}}d — event risk around the report</div>` : '';
  // direction-aware price ladder: Target / Entry / Stop, ordered so higher price sits higher.
  const _p = s.plan || {{}};
  let ladder = '';
  if (_p.entry!=null && _p.stop!=null && _p.target!=null) {{
    const _m = v => '$'+Number(v).toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}});
    const tgt = `<div class="lad-row tgt"><span>Target</span><span>${{_m(_p.target)}}<em>${{_isShort?'−':'+'}}${{_p.target_pct}}%</em></span></div>`;
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
      <div class="card-id"><div class="s">${{s.symbol}}</div><div class="n">${{s.name||s.exchange||''}}</div></div>
      <span class="act a-${{cls}}">${{s.action}}</span>
      <button class="favbtn ${{FAVS.has(s.symbol)?'on':''}}" title="Save to favorites">${{FAVS.has(s.symbol)?'★':'☆'}}</button></div>
    <div class="card-px-row"><span class="card-px" data-px="${{s.symbol}}">$${{_px.toLocaleString()}}</span>${{dchg}}</div>
    <div class="card-spark">${{_spark2(s.symbol, _dirCol(s), 300, 42)}}</div>
    ${{conv.label ? `<div class="conv-wrap"><div class="conv-row"><span>Conviction · ${{conv.label}}</span><span style="color:${{ccol}};font-weight:700;">${{cpct}}%</span></div>`
      + `<div class="conv-meter"><div class="conv-fill" style="width:${{cpct}}%;background:${{ccol}};"></div></div></div>` : ''}}
    ${{ladder}}
    ${{whyHtml}}
    ${{s.catalyst ? `<div class="cat-chip hint" data-tip="${{_esc(s.catalyst.headline)}}">⚡ Catalyst — fresh news${{s.catalyst.source?' · '+s.catalyst.source:''}}</div>` : ''}}
    ${{s.tv ? `<div class="tv-chip hint" data-tip="TradingView's aggregate technical rating (independent of our engine) — daily ${{s.tv.d||'n/a'}}, weekly ${{s.tv.w||'n/a'}}">TradingView: ${{s.tv.d||'—'}} <span style="opacity:.7;">· 1W ${{s.tv.w||'—'}}</span></div>` : ''}}
    ${{s.ai_read ? `<div class="ai-box hint" data-tip="${{_esc(s.ai_read.slice(0,600))}}"><span class="ai-h">🤖 AI analyst</span> ${{_esc(s.ai_read.split('. ')[0]).slice(0,130)}}…</div>` : ''}}
    ${{edWarn}}
    <div class="more">${{nNews ? nNews+' news &middot; ':''}}click for chart, RSI, patterns + full breakdown →</div>`;
  const _fb = el.querySelector('.favbtn');
  if (_fb) _fb.addEventListener('click', (e) => {{
    e.stopPropagation(); _toggleFav(s.symbol);
    _fb.textContent = FAVS.has(s.symbol) ? '★' : '☆'; _fb.classList.toggle('on', FAVS.has(s.symbol));
    if (_curView === 'favs') renderCards('favs');
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
function _applyView(list, view) {{
  if (view==='favs') list = list.filter(s=>FAVS.has(s.symbol));
  else if (view==='buys') list = list.filter(s=>s.action==='BUY'||s.action==='HOLD LONG');
  else if (view==='shorts') list = list.filter(s=>s.action==='SHORT'||s.action==='HOLD SHORT');
  else if (view==='watch') list = list.filter(s=>s.action==='WATCH LONG'||s.action==='WATCH SHORT');
  else if (view==='actionable') list = list.filter(s=>['BUY','SHORT','HOLD LONG','HOLD SHORT','EXIT'].includes(s.action));
  if (view==='conviction') list.sort((a,b)=>_conv(b)-_conv(a));
  else if (view==='movers') list.sort((a,b)=>(b.rel_volume||0)-(a.rel_volume||0));
  else list.sort((a,b)=>(_ACT_ORDER[a.action]-_ACT_ORDER[b.action])||(_conv(b)-_conv(a)));
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
function L_terminal(list) {{
  const r = DATA.regime || {{}};
  const buys = list.filter(s=>['BUY','HOLD LONG'].includes(s.action)).length;
  const shorts = list.filter(s=>['SHORT','HOLD SHORT'].includes(s.action)).length;
  const head = `<div class="bbhead"><span class="bbtitle">SIGNAL DESK ▮</span>`
    + `<span class="bbst">REGIME <b>${{(r.label||'—').toUpperCase()}}</b></span>`
    + `<span class="bbst">BREADTH <b style="color:#fff;">${{r.breadth!=null?r.breadth+'%':'—'}}</b></span>`
    + `<span class="bbst">BUYS <b style="color:#33d17a;">${{buys}}</b></span>`
    + `<span class="bbst">SHORTS <b style="color:#ff5c4d;">${{shorts}}</b></span>`
    + `<span class="bbclock" id="bbclock"></span></div>`;
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
function L_ticker(list) {{
  const tape = list.map(s=>`<span class="tkitem"><b>${{s.symbol}}</b> <span data-px="${{s.symbol}}" style="color:${{_dirCol(s)}};">$${{_pxOf(s).toLocaleString()}}</span></span>`).join('');
  const rows = list.map(s=>`<div class="tkrow" data-open="${{s.symbol}}">${{_logo2(s.symbol,22)}}`
    + `<span class="tksym">${{s.symbol}}</span><span style="color:${{_dirCol(s)}};font-weight:600;width:80px;">${{s.action}}</span>`
    + `<span class="tkpx" data-px="${{s.symbol}}">$${{_pxOf(s).toLocaleString()}}</span>`
    + `<span class="tkspark">${{_spark2(s.symbol,_dirCol(s),72,22)}}</span>`
    + `<span class="tkfam">${{_famOf(s)}}${{s.tv&&s.tv.d?' · TV '+s.tv.d:''}}</span><span class="tklv">${{_levelsInline(s)}}</span></div>`).join('');
  return _bindAll(_wrap('', `<div class="tktape"><div class="tktape-in">${{tape}}${{tape}}</div></div><div class="tkbody">${{rows}}</div>`), list);
}}
const LAYOUT_RENDER = {{terminal:L_terminal, lanes:L_lanes, gauges:L_gauges, feed:L_feed, ticker:L_ticker}};
// ===================================================================================

function renderCards(view) {{
  _curView = view;
  cards.innerHTML = '';
  if (_layout && _layout !== 'cards' && LAYOUT_RENDER[_layout]) {{
    const l = _applyView(DATA.signals.slice(), view);
    cards.appendChild(LAYOUT_RENDER[_layout](l));
    _reapplyLive();
    return;
  }}
  let list = DATA.signals.slice();
  if (view === 'favs') {{
    list = list.filter(s => FAVS.has(s.symbol));
    const grid = document.createElement('div'); grid.className = 'grid';
    if (!list.length) grid.innerHTML = '<div style="color:var(--muted);">No favorites yet — tap the ☆ on any card to save it here.</div>';
    list.forEach(s => grid.appendChild(makeCard(s)));
    cards.appendChild(grid);
  }} else if (view === 'sector') {{
    const by = {{}}, order = [];
    list.forEach(s => {{ const sec = s.sector || 'Other / Movers';
      if (!by[sec]) {{ by[sec] = []; order.push(sec); }} by[sec].push(s); }});
    order.forEach(sec => {{
      const h = document.createElement('div'); h.className='secthead';
      h.textContent = sec + ' · ' + by[sec].length; cards.appendChild(h);
      const grid = document.createElement('div'); grid.className='grid';
      by[sec].forEach(s => grid.appendChild(makeCard(s))); cards.appendChild(grid);
    }});
  }} else {{
    if (view === 'buys') list = list.filter(s => s.action === 'BUY' || s.action === 'HOLD LONG');
    else if (view === 'shorts') list = list.filter(s => s.action === 'SHORT' || s.action === 'HOLD SHORT');
    else if (view === 'watch') list = list.filter(s => s.action === 'WATCH LONG' || s.action === 'WATCH SHORT');
    else if (view === 'actionable') list = list.filter(s => ['BUY','SHORT','HOLD LONG','HOLD SHORT','EXIT'].includes(s.action));
    else if (view === 'conviction') list.sort((a,b) => _conv(b) - _conv(a));
    else if (view === 'movers') list.sort((a,b) => (b.rel_volume||0) - (a.rel_volume||0));
    else if (view === 'order') list.sort((a,b) =>
      (_ACT_ORDER[a.action]-_ACT_ORDER[b.action]) || (_conv(b)-_conv(a)));
    const grid = document.createElement('div'); grid.className='grid';
    if (!list.length) grid.innerHTML = '<div style="color:var(--muted);">Nothing matches this view right now.</div>';
    list.forEach(s => grid.appendChild(makeCard(s)));
    cards.appendChild(grid);
  }}
  // re-apply any live prices to the freshly rendered cards
  document.querySelectorAll('[data-px]').forEach(el => {{
    const p = (typeof LIVE !== 'undefined') ? LIVE[el.dataset.px] : null;
    if (p != null) el.textContent = _fmtPx(p);
  }});
}}
(function setupViews() {{
  const bar = document.getElementById('viewBtns');
  const views = [['sector','By sector'],['order','Actionable first'],['conviction','Highest conviction'],
                 ['buys','Longs'],['shorts','Shorts'],['watch','Watch'],['actionable','Actionable'],
                 ['movers','Biggest movers'],['favs','★ Favorites']];
  let cur = 'sector';
  try {{ cur = localStorage.getItem('view') || 'sector'; }} catch(e) {{}}
  views.forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.view = v;
    if (v === cur) b.className = 'on';
    b.onclick = () => {{
      try {{ localStorage.setItem('view', v); }} catch(e) {{}}
      bar.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.view === v));
      renderCards(v);
    }};
    bar.appendChild(b);
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
      renderCards(_curView || cur);
    }};
    lbar.appendChild(b);
  }});
  renderCards(cur);
}})();

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
    const tm = new Date(d.at || Date.now()).toLocaleTimeString();
    st.innerHTML = '&middot; <span style="color:#2ea043;">● Live</span> <span style="color:#8b97a6;">'+tm+'</span>';
  }} catch (e) {{
    st.innerHTML = '&middot; <span style="color:#8b97a6;">live prices unavailable</span>';
  }}
}}
if (LIVE_URL) {{ refreshLive(); setInterval(refreshLive, 30000); }}
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
  const fu = s.fundamentals || {{}}, an = fu.analysts, sen = s.sentiment;
  let rcells = '';
  const statc = (l,v,cls)=>`<div class="stat"><div class="l">${{l}}</div><div class="v ${{cls||''}}" style="font-size:15px;">${{v}}</div></div>`;
  if (an) {{
    const cc = an.consensus==='Buy'?'buy':an.consensus==='Sell'?'sell':'';
    rcells += statc('Analyst consensus', an.consensus, cc) + statc('Buy / Hold / Sell', `${{an.buy}} / ${{an.hold}} / ${{an.sell}}`);
  }}
  if (fu.target_mean) {{
    const up = ((fu.target_mean/s.price-1)*100);
    rcells += statc('Avg price target', '$'+fu.target_mean.toLocaleString(), up>=0?'buy':'sell')
            + statc('Upside to target', (up>=0?'+':'')+up.toFixed(0)+'%', up>=0?'buy':'sell');
  }}
  if (fu.pe) rcells += statc('P/E ratio', fu.pe);
  if (fu.earnings_date) {{
    const ed = fu.earnings_days;
    rcells += statc('Next earnings', fu.earnings_date + (ed!=null?` (${{ed}}d)`:''), (ed!=null && ed<=7)?'sell':'');
  }}
  if (sen && sen.label) rcells += statc('News tone', sen.label, sen.label==='Positive'?'buy':sen.label==='Negative'?'sell':'');
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
  document.getElementById('mPlan').innerHTML =
    stat('Entry', money(p.entry), _short ? 'short here' : 'current price') +
    stat('Stop-loss', money(p.stop), `${{_short?'+':'−'}}${{p.stop_pct}}%  ·  ATR-based`, 'sell') +
    stat(_short ? 'Cover target' : 'Take-profit', money(p.target), `${{_short?'−':'+'}}${{p.target_pct}}%`, 'buy') +
    stat('Risk : Reward', p.rr!=null ? ('1 : '+p.rr) : '–', 'reward per $1 risked') +
    stat('Position size', (p.shares||0)+' sh', money(p.exposure)+' exposure') +
    stat('$ at risk', money(p.dollar_risk), `${{p.shares||0}} sh to stop`, 'sell');
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
        return `<li>${{t}}<div class="src">${{n.source||''}} ${{n.created_at||''}}</div></li>`;
      }}).join('')
    : '<li class="src">No recent news tagged for this symbol.</li>';
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
  const KEY = 'tb-theme';
  const btn = document.getElementById('themeToggle');
  function apply(t) {{
    document.documentElement.dataset.theme = t;
    if (btn) btn.textContent = (t === 'dark') ? '☀ Light' : '🌙 Dark';
    if (featTC) featTC.applyTheme();
    if (modalTC) modalTC.applyTheme();
  }}
  let cur = 'light';
  try {{ cur = localStorage.getItem(KEY) || 'light'; }} catch (e) {{}}
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

// ---- tab navigation ----
(function setupTabs() {{
  const tabs = document.querySelectorAll('#tabs button');
  function show(page) {{
    tabs.forEach(b => b.classList.toggle('on', b.dataset.page === page));
    document.querySelectorAll('.page').forEach(s => s.classList.toggle('on', s.id === 'page-' + page));
    try {{ localStorage.setItem('tab', page); }} catch (e) {{}}
    window.scrollTo(0, 0);
    // the Markets grid + charts are laid out while hidden; size them on first reveal
    if (page === 'markets') setTimeout(() => {{
      try {{ window.dispatchEvent(new Event('resize')); }} catch (e) {{}}
      _refitCharts();
    }}, 60);
    // lazy-load the Live TV stream only when the tab is first opened (saves bandwidth)
    if (page === 'livetv') {{
      const f = document.getElementById('tvFrame');
      if (f && !f.src && f.dataset.src) f.src = f.dataset.src;
    }}
  }}
  tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.page)));
  let saved = 'signals';
  try {{ saved = localStorage.getItem('tab') || 'signals'; }} catch (e) {{}}
  show(saved);
}})();

// ---- modal sub-views (left rail) ----
(function setupModalViews() {{
  const nav = document.getElementById('mkNav'); if (!nav) return;
  const btns = nav.querySelectorAll('button');
  window._mkShow = function(v) {{
    btns.forEach(b => b.classList.toggle('on', b.dataset.mkview === v));
    document.querySelectorAll('.mk-view').forEach(p => p.classList.toggle('on', p.id === 'mkview-' + v));
    if (v === 'chart') setTimeout(() => {{ try {{ if (modalTC) modalTC.resize(); }} catch (e) {{}} }}, 50);
  }};
  btns.forEach(b => b.addEventListener('click', () => window._mkShow(b.dataset.mkview)));
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
