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

    # Market regime + sector strength from the FULL scanned set (the "tape").
    regime = _market_regime(rows)
    sectors = _sector_strength(rows)

    shown = rows[: CONFIG.show_top]
    shown_syms = [r["symbol"] for r in shown]

    # Regime filter: in a Risk-off tape, stand down on NEW buys — demote fresh BUYs
    # to HOLD so the tool isn't initiating longs against a hostile market backdrop.
    if CONFIG.regime_block_buys and regime and regime.get("label") == "Risk-off":
        for r in shown:
            if r.get("action") == "BUY":
                r["action"] = "HOLD LONG"
                r["regime_blocked"] = True
                r.setdefault("reasons", []).insert(
                    0, "🛑 Market regime is Risk-off — standing down on new buys; this fresh "
                       "crossover is shown as HOLD, not a fresh entry.")

    # Pull news once for everything shown, then bucket per ticker.
    if live:
        try:
            news = market.get_news(shown_syms, CONFIG,
                                   limit=CONFIG.news_per_symbol * max(len(shown_syms), 1))
        except Exception as exc:  # noqa: BLE001
            news = [{"headline": f"(news unavailable: {exc})", "source": "",
                     "created_at": "", "url": "", "symbols": []}]
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
    for r in shown:
        r["news"] = [n for n in news if r["symbol"] in (n.get("symbols") or [])][: CONFIG.news_per_symbol]
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
    if live:
        try:
            yq = research.yahoo_quotes([r["symbol"] for r in shown])
            for r in shown:
                q = yq.get(r["symbol"])
                if q:
                    r["quote_price"] = q["price"]
                    r["prev_close"] = q.get("prev_close")
        except Exception:  # noqa: BLE001
            pass
    for r in shown:
        r["sentiment"] = research.news_sentiment(r.get("news"))
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
        scanner.rescore(r, CONFIG, sentiment=r.get("sentiment"), fundamentals=r.get("fundamentals"))

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

    # Optional AI analyst note per signal (silent no-op if no key).
    if CONFIG.llm_enabled:
        import llm
        for r in shown:
            note = llm.analyst_note(r, CONFIG, regime=regime, macro=macro)
            if note:
                r["ai_read"] = note

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
        "benchmark": benchmark,
        "track": track,
        "regime": regime,
        "sectors": sectors,
        "macro": macro,
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
    tiles += tile("Fresh buys", str(n_buy), "buy" if n_buy else "", "new signals today")
    tiles += tile("Signals shown", str(len(sigs)), "", "ranked candidates")
    tiles += tile("Track record", wr_txt, "", f'{tk.get("resolved", 0)} calls resolved')
    return f'<div class="kpis">{tiles}</div>'


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
    wr = "—" if track["win_rate"] is None else f"{track['win_rate']}%"
    ar = "—" if track["avg_return"] is None else f"{'+' if track['avg_return'] > 0 else ''}{track['avg_return']}%"
    stats = (
        stat("Calls advised", track["advised"]) +
        stat("Resolved", track["resolved"]) +
        stat("Still open", track["open"]) +
        stat("Hit target", track["wins"], "win") +
        stat("Hit stop", track["losses"], "loss") +
        stat("Win rate", wr) +
        stat("Avg return", ar)
    )
    rows = ""
    icon = {"win": '<span class="win">✅ hit target</span>',
            "loss": '<span class="loss">❌ hit stop</span>',
            "expired": '<span class="exp">⌛ expired</span>'}
    for t in track.get("recent", []):
        ret = t.get("return_pct")
        ret_s = "—" if ret is None else f"{'+' if ret > 0 else ''}{ret}%"
        rows += (f"<tr><td>{t['symbol']}</td><td>{t['advised_date']}</td>"
                 f"<td>{icon.get(t['status'], t['status'])}</td>"
                 f"<td>{ret_s}</td><td>{t.get('days_held','—')}d</td></tr>")
    table = (f'<table class="trackrec"><tr><th>Stock</th><th>Advised</th><th>Outcome</th>'
             f'<th>Return</th><th>Held</th></tr>{rows}</table>') if rows else \
        '<p style="color:var(--muted);font-size:13px;">No calls have resolved yet — check back as trades play out.</p>'
    return f"""
  <div class="track">
    <h2 style="border:0;padding:0;">📊 Track record — how past BUY calls have done</h2>
    <p style="color:var(--muted);font-size:13px;margin:2px 0 0;">Every BUY the tool flags is logged, then
    checked against real prices: did it reach its target (✅) or hit its stop first (❌)? This builds up
    over time into an honest read on how reliable the calls are. It's a hypothetical record — no fees or
    slippage — so treat it as a rough guide, not a brokerage statement.</p>
    <div class="trackstats">{stats}</div>
    {table}
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
    kpi_html = _kpi_html(snap.get("regime"), snap)
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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<script src="chart_engine.js"></script>
<style>
  /* Light "Capital IQ Pro" palette is the default; dark is a toggle. */
  :root {{ --bg:#f5f7fa; --card:#ffffff; --line:#e4e8ed; --txt:var(--inset);
    --muted:#5b6776; --txt2:#3d4757; --buy:#0f9d58; --sell:#d1242f; --hold:#0b5cad; --flat:#8a96a3;
    --accent:#0b5cad; --grid:rgba(120,130,145,0.16); --cross:rgba(60,70,85,0.4);
    --inset:#f1f4f8; --hover:#eef2f7; --ring:rgba(11,92,173,.40);
    --shadow:0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
    --shadow-lg:0 6px 20px rgba(16,24,40,0.10); }}
  html[data-theme="dark"] {{ --bg:#0d1117; --card:#161b22; --line:#262d36; --txt:#e6edf3;
    --muted:#8b97a6; --txt2:var(--txt2); --buy:#2ea043; --sell:#f85149; --hold:#58a6ff; --flat:#6e7681;
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
  .more {{ color:var(--muted); font-size:12px; margin-top:10px;
    border-top:1px solid var(--line); padding-top:8px; }}
  .sym {{ font-size:18px; font-weight:700; }}
  .logo {{ width:20px; height:20px; border-radius:4px; vertical-align:middle;
    margin-right:7px; background:#fff; object-fit:contain; }}
  .cname {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .act {{ float:right; padding:2px 10px; border-radius:6px; font-size:12px; font-weight:700; color:#fff; }}
  .a-BUY {{ background:var(--buy); }} .a-SELL {{ background:var(--sell); }}
  .a-HOLDLONG {{ background:var(--hold); }} .a-FLAT {{ background:var(--flat); }}
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
  <div class="subhead">Generated {snap['generated_at']} &middot; scanned {snap['scanned']} symbols{health_html}</div>
  {kpi_html}
  <div class="note" style="margin-top:0;">{mode_note}</div>
  <div id="diag"></div>

  <nav class="tabs" id="tabs">
    <button data-page="signals" class="on">Signals</button>
    <button data-page="markets">Markets</button>
    <button data-page="ipos">IPO watch</button>
    <button data-page="track">Track record</button>
    <button data-page="method">How it works</button>
    <button data-page="news">Market news</button>
  </nav>

  <section class="page" id="page-markets">
    <div class="mkt">
      <nav class="mkt-side" id="mktNav">
        <button data-mview="chart" class="on">Featured chart</button>
        <button data-mview="overview">Overview vs S&amp;P</button>
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
        <div class="mkt-view" id="mview-overview">
          <div class="ovbox">
            <div class="ovhead">Live overview — % change vs S&amp;P 500 <span id="ovStatus"></span>
              <span class="ctlgrp" id="ovRangeBtns" style="margin-left:8px;"></span>
              <button class="ctlbtn" id="ovColorBtn" style="margin-left:6px;">Colour all</button>
              <span style="font-weight:400;color:var(--muted);font-size:12px;"> · click a name to highlight</span></div>
            <div class="ovwrap">
              <div class="ovchart"><canvas id="overviewChart" height="150"></canvas></div>
              <div class="ovboard" id="ovBoard"></div>
            </div>
          </div>
        </div>
        <div class="mkt-view" id="mview-sectors">{sectors_html}</div>
        <div class="mkt-view" id="mview-macro">{macro_html}</div>
      </div>
    </div>
  </section>

  <section class="page on" id="page-signals">
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">Sort &amp; filter:</span>
      <span class="ctlgrp" id="viewBtns"></span></div>
    <div id="cards"></div>
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

      <h4>The strategy: moving-average crossover (trend-following)</h4>
      <p>A "moving average" is just the average price over the last N days, which smooths out the daily
      noise so the underlying direction is visible. We track two:
      a fast one (<span class="pill">{snap['params']['fast_ma']}-day</span>) and a slow one
      (<span class="pill">{snap['params']['slow_ma']}-day</span>).</p>
      <ol>
        <li><b>Scan</b> — curated large-caps plus the day's most-active stocks and biggest movers.</li>
        <li><b>Buy signal</b> — the fast average crosses <b>above</b> the slow one (an uptrend is starting),
        as long as momentum (RSI) isn't already overheated.</li>
        <li><b>Sell signal</b> — the fast average crosses back <b>below</b> the slow one, or momentum gets
        overbought (the move looks exhausted).</li>
        <li><b>Risk first</b> — every trade gets a <b>stop-loss</b> (a safety exit ~{snap['params']['stop_loss_pct']:.0%}
        below entry) and a <b>take-profit</b> target (~{snap['params']['take_profit_pct']:.0%} above), with the
        position sized so a stop-out costs only about {snap['params']['risk_per_trade']:.0%} of the account.</li>
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
        <li><b>Strategy confluence</b> — we also run several <i>independent</i> strategies on the same
        stock (trend crossover, golden cross, Donchian breakout, MACD momentum, RSI-2 dip-buy,
        Bollinger squeeze breakout, EMA momentum stack). When more of them agree the setup is long,
        conviction rises. The detail panel shows which are firing and how each has historically
        performed on that stock.</li>
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

  <div class="disclaimer">
    Strategy: {snap['params']['fast_ma']}/{snap['params']['slow_ma']} SMA crossover with
    RSI({snap['params']['rsi_period']}) filter. Risk {snap['params']['risk_per_trade']:.0%}/trade,
    stop {snap['params']['stop_loss_pct']:.0%}, target {snap['params']['take_profit_pct']:.0%}.
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
  const ccol = conv.label==='High' ? 'var(--buy)' : (conv.label==='Low' ? 'var(--sell)' : '#c08a1e');
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
  const st = [];
  st.push(`<div class="card-stat"><span>RSI</span><b>${{s.rsi}}</b></div>`);
  if (s.rel_volume!=null) st.push(`<div class="card-stat"><span>Rel vol</span><b>${{s.rel_volume}}×</b></div>`);
  if (s.stop!=null) st.push(`<div class="card-stat"><span>Stop</span><b>$${{s.stop}}</b></div>`);
  if (s.target!=null) st.push(`<div class="card-stat"><span>Target</span><b>$${{s.target}}</b></div>`);
  if ((s.plan||{{}}).rr!=null) st.push(`<div class="card-stat"><span>Reward:risk</span><b>${{s.plan.rr}}</b></div>`);
  st.push(`<div class="card-stat"><span>As of</span><b>${{s.as_of}}</b></div>`);
  const ed = (s.fundamentals||{{}}).earnings_days;
  const ch = (s.patterns||[]).slice(0,2).map(p=>`<span class="chip mini ${{p.kind}}">${{p.label}}</span>`);
  if (ed!=null && ed<=7) ch.unshift(`<span class="chip mini bear">⚠ Earnings ${{ed}}d</span>`);
  const _cn = (s.strategies&&s.strategies.now) ? s.strategies.now : null;
  if (_cn && _cn.count>=2) ch.unshift(`<span class="chip mini bull" title="independent strategies agreeing">▲ ${{_cn.count}}/${{_cn.total}} strategies</span>`);
  const nNews = (s.news||[]).length;
  el.innerHTML = `
    <div class="card-top">${{logo}}
      <div class="card-id"><div class="s">${{s.symbol}}</div><div class="n">${{s.name||s.exchange||''}}</div></div>
      <span class="act a-${{cls}}">${{s.action}}</span></div>
    <div class="card-px-row"><span class="card-px" data-px="${{s.symbol}}">$${{_px.toLocaleString()}}</span>${{dchg}}</div>
    ${{conv.label ? `<div class="conv-wrap"><div class="conv-row"><span>Conviction · ${{conv.label}}</span><span>${{cpct}}%</span></div>`
      + `<div class="conv-meter"><div class="conv-fill" style="width:${{cpct}}%;background:${{ccol}};"></div></div></div>` : ''}}
    <div class="card-stats">${{st.join('')}}</div>
    ${{ch.length ? `<div class="chips" style="margin-top:10px;">${{ch.join('')}}</div>` : ''}}
    <div class="more">${{nNews ? nNews+' news &middot; ':''}}click for full plan + reasoning →</div>`;
  el.addEventListener('click', () => openModal(s));
  return el;
}}
// --- views: filter / sort the signal cards ---
const _ACT_ORDER = {{'BUY':0, 'SELL':1, 'HOLD LONG':2, 'FLAT':3}};
const _conv = s => (s.conviction ? s.conviction.score_pct : -1);
function renderCards(view) {{
  cards.innerHTML = '';
  let list = DATA.signals.slice();
  if (view === 'sector') {{
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
    if (view === 'buys') list = list.filter(s => s.action === 'BUY');
    else if (view === 'actionable') list = list.filter(s => s.action !== 'FLAT');
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
  const views = [['sector','By sector'],['order','Buys first'],['conviction','Highest conviction'],
                 ['movers','Biggest movers'],['buys','Buys only'],['actionable','Actionable']];
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
  renderCards(cur);
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
    _updateOverviewLive();
    const tm = new Date(d.at || Date.now()).toLocaleTimeString();
    st.innerHTML = '&middot; <span style="color:#2ea043;">● Live</span> <span style="color:#8b97a6;">'+tm+'</span>';
    const ov = document.getElementById('ovStatus');
    if (ov) ov.innerHTML = '<span style="color:#2ea043;font-size:12px;">● live ' + tm + '</span>';
  }} catch (e) {{
    st.innerHTML = '&middot; <span style="color:#8b97a6;">live prices unavailable</span>';
  }}
}}
if (LIVE_URL) {{ refreshLive(); setInterval(refreshLive, 30000); }}

// ---- live overview: dim-by-default % chart + sorted leaderboard (click to highlight) ----
let ovChart = null;
const OV_WIN = 60;      // ~3 months of daily bars
const OV_PALETTE = ['#388bfd','#2ea043','#f0883e','#f85149','#a371f7','#e8c878',
                    '#56d4dd','#db61a2','#6cc644','#bd8b00','#ff7b72','#79c0ff'];
const OV = {{ items: [], pinned: new Set(), range: 'D', intra: {{}}, colorAll: false }};  // range: D=daily, 1W, 1D
function _rebase(t, close) {{
  const n = close.length, st = Math.max(0, n - OV_WIN);
  const T = t.slice(st), C = close.slice(st);
  let base = null;
  for (const v of C) {{ if (v != null) {{ base = v; break; }} }}
  if (!base) return {{ pts: [], base: null }};
  return {{ pts: T.map((tt, i) => ({{x: tt, y: (C[i]/base - 1) * 100}})), base }};
}}
// draw the series name at the end of each *active* (benchmark/pinned) line
const _ovEndLabel = {{
  id: 'ovEndLabel',
  afterDatasetsDraw(chart) {{
    const xs = chart.scales.x, ys = chart.scales.y, ctx = chart.ctx;
    chart.data.datasets.forEach(ds => {{
      if (!ds._active || !ds.data.length) return;
      const last = ds.data[ds.data.length - 1];
      const px = xs.getPixelForValue(last.x), py = ys.getPixelForValue(last.y);
      if (px == null || py == null) return;
      ctx.save(); ctx.fillStyle = ds.borderColor; ctx.font = '600 11px -apple-system,sans-serif';
      ctx.textBaseline = 'middle'; ctx.fillText(' ' + ds.label, px + 2, py); ctx.restore();
    }});
  }}
}};
function _ovChart(plot, unit, intraday) {{
  const cv = document.getElementById('overviewChart'); if (!cv) return;
  const datasets = plot.map(it => ({{
    label: it.label, data: it.pts, _sym: it.sym, _active: it.active,
    borderColor: it.bench ? _cv('--txt','#e6edf3')
                 : (it.active ? it.color : (OV.colorAll ? it.color : 'rgba(139,151,166,0.22)')),
    borderWidth: it.bench ? 2.4 : (it.active ? 2 : (OV.colorAll ? 1.3 : 1)), pointRadius: 0, fill: false,
    order: it.active ? 1 : 5 }}));
  // correct min/max: fit the axis so no line is ever clipped. When the user has
  // pinned/highlighted names, scale to those (+ the benchmark); otherwise fit all.
  const anyPin = OV.pinned && OV.pinned.size > 0;
  const src = anyPin ? plot.filter(it => it.active) : plot;
  const allY = [0];
  src.forEach(it => it.pts.forEach(p => {{ if (p.y != null) allY.push(p.y); }}));
  const lo = Math.min(...allY), hi = Math.max(...allY);
  const pad = Math.max((hi - lo) * 0.05, 1);
  const ymin = Math.floor(lo - pad), ymax = Math.ceil(hi + pad);
  if (ovChart) ovChart.destroy();
  ovChart = new Chart(cv, {{
    type:'line', data:{{datasets}},
    options:{{responsive:true, parsing:false, interaction:{{mode:'index',intersect:false}},
      layout:{{padding:{{right:46}}}}, elements:{{line:{{tension:0.15}}}},
      plugins:{{
        legend:{{display:false}},
        tooltip:{{mode:'index', intersect:false, itemSort:(a,b)=>b.parsed.y-a.parsed.y,
          callbacks:{{
            title:(its)=> {{ const d=new Date(its[0].parsed.x); return intraday
              ? d.toLocaleString([], {{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}})
              : d.toLocaleDateString([], {{year:'numeric',month:'short',day:'numeric'}}); }},
            label:(it)=>` ${{it.dataset.label}}: ${{it.parsed.y>=0?'+':''}}${{it.parsed.y.toFixed(1)}}%`}}}},
        zoom:{{zoom:{{wheel:{{enabled:true}}, pinch:{{enabled:true}}, mode:'x'}}, pan:{{enabled:true, mode:'x'}}}}
      }},
      scales:{{
        x:{{type:'time', time:{{unit}}, ticks:{{color:_cv('--muted','#8b97a6'),maxTicksLimit:7}}, grid:{{display:false}}}},
        y:{{position:'right', min:ymin, max:ymax, ticks:{{color:_cv('--muted','#8b97a6'), maxTicksLimit:8, callback:v=>(v>0?'+':'')+Math.round(v)+'%'}},
           grid:{{ color:(c)=> c.tick.value===0 ? _cv('--muted','rgba(139,151,166,0.6)') : _cv('--grid','rgba(42,52,65,0.5)'),
                   lineWidth:(c)=> c.tick.value===0 ? 1.5 : 1 }}}}
      }}
    }},
    plugins:[_ovEndLabel]
  }});
}}
function buildOverview() {{
 try {{
  if (typeof Chart === 'undefined') return;
  // daily items power the leaderboard (and daily mode)
  const items = [];
  if (DATA.benchmark) {{
    const r = _rebase(DATA.benchmark.t, DATA.benchmark.close);
    if (r.pts.length) items.push({{sym:'SPY', label:'S&P 500', pts:r.pts, base:r.base, color:_cv('--txt','#e6edf3'), bench:true}});
  }}
  DATA.signals.forEach((s, i) => {{
    const c = DATA.charts[s.symbol]; if (!c) return;
    const r = _rebase(c.t, c.close); if (!r.pts.length) return;
    items.push({{sym:s.symbol, label:s.symbol, pts:r.pts, base:r.base, color:OV_PALETTE[i % OV_PALETTE.length]}});
  }});
  OV.items = items;
  _buildBoard();
  const key = document.getElementById('mChartKey');  // (unused here)
  if (OV.range === 'D') {{
    const plot = items.map(it => Object.assign({{}}, it, {{active: it.bench || OV.pinned.has(it.sym)}}));
    _ovChart(plot, 'month', false);
  }} else {{
    _ovIntra();
  }}
 }} catch (e) {{ console.error('overview chart failed', e); }}
}}
function _ovIntra() {{
  if (!LIVE_URL) {{ OV.range = 'D'; buildOverview(); return; }}
  const active = ['SPY', ...OV.pinned];
  const tf = OV.range === '1D' ? '15Min' : '1Hour';
  const days = OV.range === '1D' ? 3 : 8;
  const colorOf = sym => sym === 'SPY' ? _cv('--txt','#e6edf3') : ((OV.items.find(it => it.sym === sym) || {{}}).color || '#388bfd');
  Promise.all(active.map(sym => {{
    const ck = sym + ':' + OV.range;
    if (OV.intra[ck]) return Promise.resolve({{sym, bars: OV.intra[ck]}});
    return fetch(LIVE_URL + '?bars=' + encodeURIComponent(sym) + '&tf=' + tf + '&days=' + days)
      .then(r => r.json()).then(d => {{
        let b = d.bars || [];
        if (OV.range === '1D' && b.length) {{
          const ld = new Date(b[b.length-1].t).toDateString();
          const t = b.filter(x => new Date(x.t).toDateString() === ld);
          b = t.length >= 2 ? t : b.slice(-26);
        }}
        OV.intra[ck] = b; return {{sym, bars: b}};
      }}).catch(() => ({{sym, bars: []}}));
  }})).then(res => {{
    const plot = res.filter(r => r.bars.length).map(r => {{
      const base = r.bars[0].c;
      return {{sym:r.sym, label: r.sym==='SPY'?'S&P 500':r.sym, bench: r.sym==='SPY', active:true,
        color: colorOf(r.sym), pts: r.bars.map(b => ({{x:b.t, y:(b.c/base - 1)*100}}))}};
    }});
    _ovChart(plot, OV.range === '1D' ? 'hour' : 'day', true);
  }});
}}
function _setupOvRange() {{
  const bar = document.getElementById('ovRangeBtns'); if (!bar || bar.childElementCount) return;
  [['D','Daily'],['1W','1W'],['1D','1D']].forEach(([v,lab]) => {{
    const b = document.createElement('button'); b.textContent = lab; b.dataset.r = v;
    if (v === OV.range) b.className = 'on';
    b.onclick = () => {{ OV.range = v; bar.querySelectorAll('button').forEach(x=>x.classList.toggle('on', x.dataset.r===v)); buildOverview(); }};
    bar.appendChild(b);
  }});
  const cb = document.getElementById('ovColorBtn');
  if (cb) {{ cb.classList.toggle('on', OV.colorAll);
    cb.onclick = () => {{ OV.colorAll = !OV.colorAll; cb.classList.toggle('on', OV.colorAll); buildOverview(); }}; }}
}}
function _buildBoard() {{
  const board = document.getElementById('ovBoard'); if (!board || !OV.items.length) return;
  const rows = OV.items.map(it => ({{sym:it.sym, label:it.label,
    val: it.pts[it.pts.length-1].y, color: it.color, bench: it.bench}}));
  rows.sort((a, b) => b.val - a.val);
  board.innerHTML = rows.map(r => {{
    const on = r.bench || OV.pinned.has(r.sym);
    return `<div class="ovrow ${{on?'on':''}}" data-sym="${{r.sym}}">`
      + `<span class="ovdot" style="background:${{(on||OV.colorAll) ? r.color : 'rgba(139,151,166,0.4)'}};"></span>`
      + `<span class="ovsym">${{r.label}}</span>`
      + `<span class="ovval" style="color:${{r.val>=0?'#2ea043':'#f85149'}};">${{r.val>=0?'+':''}}${{r.val.toFixed(1)}}%</span></div>`;
  }}).join('');
  board.querySelectorAll('.ovrow').forEach(el => {{
    el.onclick = () => {{
      const sym = el.dataset.sym; if (sym === 'SPY') return;  // benchmark always on
      if (OV.pinned.has(sym)) OV.pinned.delete(sym); else OV.pinned.add(sym);
      buildOverview();
    }};
  }});
}}
function _updateOverviewLive() {{
  if (!ovChart || !OV.items.length || OV.range !== 'D') return;  // live-nudge daily mode only
  OV.items.forEach(it => {{
    const lp = LIVE[it.sym];
    if (lp != null && it.base) it.pts[it.pts.length-1].y = (lp / it.base - 1) * 100;
  }});
  ovChart.update('none');
  _buildBoard();
}}

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
      chips += `<span class="chip mini ${{cls}}" title="${{r.kind}}">${{r.long ? '●' : '○'}} ${{r.label}}</span>`;
    }});
    let rows = '';
    Object.keys(edges).forEach(k => {{
      const e = edges[k]; const wr = e.win_rate == null ? '–' : e.win_rate + '%';
      const ret = (e.total_return >= 0 ? '+' : '') + e.total_return + '%';
      rows += `<tr><td>${{e.label}}</td><td style="color:var(--muted);">${{e.kind}}</td>`
        + `<td style="text-align:right;">${{wr}}</td><td style="text-align:right;color:var(--muted);">${{e.n_trades}}</td>`
        + `<td style="text-align:right;" class="${{e.total_return >= 0 ? 'win' : 'loss'}}">${{ret}}</td></tr>`;
    }});
    const head = `<div style="margin-bottom:6px;font-size:13px;"><b>${{now.count || 0}}</b> of ${{now.total || 0}} strategies are long here right now (● long · ○ flat):</div>`
      + `<div class="chips" style="margin-bottom:12px;">${{chips}}</div>`;
    const table = rows
      ? `<table class="trackrec"><thead><tr><th>Strategy</th><th>Type</th><th style="text-align:right;">Win</th><th style="text-align:right;">Trades</th><th style="text-align:right;">Return</th></tr></thead><tbody>${{rows}}</tbody></table>`
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
  document.getElementById('mPlanNote').textContent =
    (s.action==='BUY'||s.action==='HOLD LONG') ? '(long — active)' : '— levels if you took this long';
  document.getElementById('mPlan').innerHTML =
    stat('Entry', money(p.entry), 'current price') +
    stat('Stop-loss', money(p.stop), `−${{p.stop_pct}}%  ·  ATR-based`, 'sell') +
    stat('Take-profit', money(p.target), `+${{p.target_pct}}%`, 'buy') +
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
    if (typeof buildOverview === 'function') {{ try {{ buildOverview(); }} catch (e) {{}} }}
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
// resize the featured + overview charts when their panel becomes visible
function _refitCharts() {{
  try {{ if (featTC) featTC.resize(); }} catch (e) {{}}
  try {{ if (typeof ovChart !== 'undefined' && ovChart) ovChart.resize(); }} catch (e) {{}}
}}
_initCharts();

function closeModal() {{ overlay.classList.remove('open'); }}
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
_setupOvRange();
buildOverview();
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
