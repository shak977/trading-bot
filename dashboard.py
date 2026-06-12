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
    bg, fg = palette.get(reg["label"], ("#1a212b", "#c4ccd6"))
    return (f'<div class="regime" style="background:{bg};">'
            f'<span class="rlabel" style="color:{fg};">Market: {reg["label"]}</span>'
            f'<span class="rdetail">{reg["breadth"]}% of {reg["total"]} scanned above trend &middot; '
            f'avg momentum {reg["avg_rsi"]}/100 &middot; {reg["buys"]} fresh buys</span>'
            f'<span class="rnote">{reg["note"]}</span></div>')


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
    sectors_html = _sectors_html(snap.get("sectors"))
    macro_html = _macro_html(snap.get("macro"))
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
<style>
  :root {{ --bg:#0f1419; --card:#1a212b; --line:#2a3441; --txt:#e6edf3;
    --muted:#8b97a6; --buy:#2ea043; --sell:#f85149; --hold:#388bfd; --flat:#6e7681; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--txt); }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px 60px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:30px 0 12px; color:var(--muted);
    text-transform:uppercase; letter-spacing:.05em; }}
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
    padding:16px; cursor:pointer; transition:border-color .12s, transform .12s; }}
  .card:hover {{ border-color:#3d4d5f; transform:translateY(-2px); }}
  .more {{ color:var(--muted); font-size:12px; margin-top:10px;
    border-top:1px solid var(--line); padding-top:8px; }}
  .sym {{ font-size:18px; font-weight:700; }}
  .logo {{ width:20px; height:20px; border-radius:4px; vertical-align:middle;
    margin-right:7px; background:#fff; object-fit:contain; }}
  .cname {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .act {{ float:right; padding:2px 10px; border-radius:6px; font-size:12px; font-weight:700; }}
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
  .stat {{ background:#0f1722; border:1px solid var(--line); border-radius:10px;
    padding:10px 12px; }}
  .stat .l {{ color:var(--muted); font-size:11px; text-transform:uppercase;
    letter-spacing:.04em; }}
  .stat .v {{ font-size:17px; font-weight:700; margin-top:2px; }}
  .stat .v.buy {{ color:var(--buy); }} .stat .v.sell {{ color:var(--sell); }}
  .stat .sub {{ color:var(--muted); font-size:11px; }}
  .deskread {{ background:#0f1722; border:1px solid var(--line); border-left:3px solid var(--hold);
    border-radius:10px; padding:12px 14px; font-size:14px; margin:14px 0; }}
  .convbadge {{ font-size:13px; font-weight:700; padding:2px 10px; border-radius:999px; }}
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
  .chip.bull {{ background:#15361f; color:#7ee2a0; border-color:#1d4a2b; }}
  .chip.bear {{ background:#3a1e1e; color:#ff9b9b; border-color:#5a1e1e; }}
  .chip.neutral {{ background:#1a212b; color:#c4ccd6; }}
  .chip.mini {{ font-size:10.5px; padding:1px 7px; }}
  .tabs {{ display:flex; gap:4px; flex-wrap:wrap; border-bottom:1px solid var(--line);
    margin:18px 0 22px; position:sticky; top:0; background:var(--bg); z-index:10; padding-top:6px; }}
  .tabs button {{ background:none; border:none; color:var(--muted); font-size:15px;
    font-weight:600; padding:10px 16px; cursor:pointer; border-bottom:2px solid transparent; }}
  .tabs button.on {{ color:var(--txt); border-bottom-color:var(--hold); }}
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
  .ovrow:hover {{ color:var(--txt); background:#0f1722; }}
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
  .secbar {{ flex:1; height:8px; background:#0f1722; border-radius:5px; overflow:hidden; }}
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
  .method p, .method li {{ font-size:14px; color:#c4ccd6; line-height:1.6; }}
  .method ol, .method ul {{ padding-left:20px; margin:6px 0; }}
  .method .pill {{ display:inline-block; background:#0f1722; border:1px solid var(--line);
    border-radius:6px; padding:1px 7px; font-size:13px; color:var(--txt); }}
</style></head>
<body><div class="wrap">
  <h1>Trading Signals Dashboard</h1>
  <div class="meta">Generated {snap['generated_at']} &middot;
    <span class="badge m-{mode}">{mode}</span> &middot;
    scanned {snap['scanned']} symbols <span id="liveStatus"></span></div>
  <div class="note">{mode_note}</div>
{regime_html}
  <div id="diag"></div>

  <nav class="tabs" id="tabs">
    <button data-page="signals" class="on">Signals</button>
    <button data-page="track">Track record</button>
    <button data-page="method">How it works</button>
    <button data-page="news">Market news</button>
  </nav>

  <section class="page on" id="page-signals">
    <h2 style="margin-top:0;">Signals <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— grouped by sector · click any card for the full reasoning</span></h2>
    <div class="ovbox">
      <div class="ovhead">📈 Live overview — % change vs S&amp;P 500 <span id="ovStatus"></span>
        <span class="ctlgrp" id="ovRangeBtns" style="margin-left:8px;"></span>
        <button class="ctlbtn" id="ovColorBtn" style="margin-left:6px;">Colour all</button>
        <span style="font-weight:400;color:var(--muted);font-size:12px;"> · click a name to highlight</span></div>
      <div class="ovwrap">
        <div class="ovchart"><canvas id="overviewChart" height="150"></canvas></div>
        <div class="ovboard" id="ovBoard"></div>
      </div>
    </div>
{sectors_html}
{macro_html}
    <div class="viewctl"><span style="color:var(--muted);font-size:13px;">View:</span>
      <span class="ctlgrp" id="viewBtns"></span></div>
    <div id="cards"></div>
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
  <div class="modal">
    <button class="close" id="modalClose">&times;</button>
    <h3 id="mTitle"></h3>
    <div class="summary" id="mSummary"></div>
    <div class="sech" id="mAIHead" style="display:none;">In plain English (AI) 🤖</div>
    <div class="deskread" id="mAI" style="display:none;border-left-color:#9b59b6;"></div>
    <div class="sech">The bottom line</div>
    <div class="deskread" id="mDesk"></div>
    <div class="sech">Patterns spotted</div>
    <div class="chips" id="mPatterns"></div>
    <div class="sech">Should you take it? <span id="mConvScore"></span></div>
    <ul class="checks" id="mChecks"></ul>
    <div class="sech">Analysts, fundamentals &amp; news tone</div>
    <div class="plangrid" id="mResearch"></div>
    <div class="sech">How this strategy has done on this stock <span style="text-transform:none;color:var(--muted);">(backtest)</span></div>
    <div class="plangrid" id="mEdge"></div>
    <div class="sech">The trade plan <span id="mPlanNote" style="text-transform:none;color:var(--muted);"></span></div>
    <div class="plangrid" id="mPlan"></div>
    <div class="sech">Price chart</div>
    <div class="readout" id="mReadout"></div>
    <div class="chartctl">
      <span class="ctlgrp" id="rangeBtns"></span>
      <span class="ctlgrp" id="typeBtns"></span>
      <span class="ctlgrp" id="indBtns"></span>
      <label class="ctltog"><input type="checkbox" id="benchToggle"> vs S&amp;P 500</label>
      <span class="ctlgrp">
        <button id="zoomOut" title="Zoom out">&minus;</button>
        <button id="zoomIn" title="Zoom in">+</button>
      </span>
      <button class="ctlbtn" id="zoomReset">Reset</button>
    </div>
    <div class="chartbox"><canvas id="mChart" height="140"></canvas></div>
    <div class="chartbox" id="macdBox" style="display:none; margin-top:8px;">
      <div style="color:var(--muted);font-size:11px;margin-bottom:4px;">MACD — momentum (above 0 = bullish)</div>
      <canvas id="mMacd" height="70"></canvas>
    </div>
    <div class="chartkey" id="mChartKey"></div>
    <div class="sech">The details, explained</div>
    <ul class="reasons" id="mReasons"></ul>
    <div class="sech">Latest news on this stock</div>
    <ul class="news" id="mNews"></ul>
    <div class="sech">Market context</div>
    <div class="plangrid" id="mContext"></div>
  </div>
</div>
<script>
const DATA = {data_json};
const LIVE_URL = "{CONFIG.live_quotes_url}";
let LIVE = {{}};  // latest live prices (declared early so renderCards can read it safely)
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
  const hot = (s.rel_volume!=null && s.rel_volume>=1.5) ? ' hot' : '';
  const nNews = (s.news||[]).length;
  el.innerHTML = `
    <div><img class="logo" src="https://assets.parqet.com/logos/symbol/${{s.symbol}}?format=png" alt="" onerror="this.style.display='none'">
      <span class="sym">${{s.symbol}}</span>
      <span class="act a-${{cls}}">${{s.action}}</span></div>
    ${{s.name ? `<div class="cname">${{s.name}}${{s.exchange?` · ${{s.exchange}}`:''}}</div>`:''}}
    <div class="px" data-px="${{s.symbol}}">$${{s.price.toLocaleString()}}</div>`;
  el.innerHTML += `
    <div class="kv"><span>As of</span><span>${{s.as_of}}</span></div>
    <div class="kv"><span>RSI</span><span>${{s.rsi}}</span></div>
    <div class="kv"><span>Fast / Slow MA</span><span>${{s.fast_ma}} / ${{s.slow_ma}}</span></div>
    ${{s.rel_volume!=null ? `<div class="kv${{hot}}"><span>Rel vol (flow proxy)</span><span>${{s.rel_volume}}x</span></div>`:''}}
    ${{s.stop!=null ? `<div class="kv"><span>Stop / Target</span><span>$${{s.stop}} / $${{s.target}}</span></div>`:''}}
    ${{s.suggested_shares ? `<div class="kv"><span>Suggested size</span><span>${{s.suggested_shares}} sh</span></div>`:''}}
    ${{s.conviction ? `<div class="kv"><span>Conviction</span><span><span class="convbadge conv-${{s.conviction.label}}" style="font-size:11px;">${{s.conviction.label}} ${{s.conviction.score_pct}}%</span></span></div>`:''}}
    ${{(() => {{ const ed=(s.fundamentals||{{}}).earnings_days; const ch=(s.patterns||[]).slice(0,3).map(p=>`<span class="chip mini ${{p.kind}}">${{p.label}}</span>`); if(ed!=null && ed<=7) ch.unshift(`<span class="chip mini bear">⚠ Earnings ${{ed}}d</span>`); return ch.length?`<div class="chips" style="margin-top:8px;">${{ch.join('')}}</div>`:''; }})()}}
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
function _pushLiveToChart() {{
  // Update the most recent point of the open modal chart with the live price,
  // preserving the user's current zoom/pan (update('none')).
  if (!mChart || !MODAL) return;
  const lp = LIVE[MODAL.symbol];
  if (lp == null) return;
  const price = mChart.data.datasets.find(d => d.label === 'Price' || d.label === MODAL.symbol);
  if (!price || !price.data || !price.data.length) return;
  const last = price.data[price.data.length - 1];
  if (last && typeof last === 'object') {{
    if ('c' in last) {{ last.c = lp; if (lp > last.h) last.h = lp; if (lp < last.l) last.l = lp; }}
    else last.y = lp;
    mChart.update('none');
  }}
}}
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
    _pushLiveToChart();
    if (overlay && overlay.classList.contains('open') && CUR.prices.length) {{
      CUR.prices[CUR.prices.length-1] = LIVE[MODAL && MODAL.symbol] != null ? LIVE[MODAL.symbol] : CUR.prices[CUR.prices.length-1];
      _updateReadout();
    }}
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
    borderColor: it.bench ? '#e6edf3'
                 : (it.active ? it.color : (OV.colorAll ? it.color : 'rgba(139,151,166,0.16)')),
    borderWidth: it.bench ? 2.4 : (it.active ? 2 : (OV.colorAll ? 1.3 : 1)), pointRadius: 0, fill: false,
    order: it.active ? 1 : 5 }}));
  const allY = []; plot.forEach(it => it.pts.forEach(p => allY.push(p.y)));
  allY.sort((a, b) => a - b);
  const q = p => allY.length ? allY[Math.min(allY.length-1, Math.max(0, Math.round(p*(allY.length-1))))] : 0;
  const ymin = Math.min(q(0.04), 0) - 2, ymax = Math.max(q(0.96), 0) + 3;
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
        x:{{type:'time', time:{{unit}}, ticks:{{color:'#8b97a6',maxTicksLimit:7}}, grid:{{display:false}}}},
        y:{{position:'right', min:ymin, max:ymax, ticks:{{color:'#8b97a6', callback:v=>(v>0?'+':'')+v+'%'}},
           grid:{{ color:(c)=> c.tick.value===0 ? 'rgba(139,151,166,0.6)' : 'rgba(42,52,65,0.5)',
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
    if (r.pts.length) items.push({{sym:'SPY', label:'S&P 500', pts:r.pts, base:r.base, color:'#e6edf3', bench:true}});
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
  const colorOf = sym => sym === 'SPY' ? '#e6edf3' : ((OV.items.find(it => it.sym === sym) || {{}}).color || '#388bfd');
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
let mChart;
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
  // chart is stateful (range / type / benchmark) — render via renderModalChart
  MODAL = s;
  renderModalChart();
  overlay.classList.add('open');
}}

// ---- stateful modal chart ----
let MODAL = null;
const CSTATE = {{ range:'6M', type:'line', bench:false, avgs:true, boll:true, macd:false }};
const INTRA = {{}};  // cache for intraday bars (5D only)
let macdChart = null;
let CUR = {{ prices:[], times:[], base:null, intraday:false }};  // for the hover readout
let _finOK = false;
try {{ _finOK = !!(window.Chart && Chart.registry.getController('candlestick')); }} catch(e) {{ _finOK = false; }}

// crosshair: vertical line + dot at the hovered point (Robinhood/Google style)
const _crosshair = {{
  id:'crosshair',
  afterDatasetsDraw(chart) {{
    const act = chart.getActiveElements ? chart.getActiveElements() : (chart._active||[]);
    if (!act || !act.length) return;
    const x = act[0].element.x, y = act[0].element.y;
    const {{top, bottom}} = chart.chartArea, c = chart.ctx;
    c.save();
    c.beginPath(); c.moveTo(x, top); c.lineTo(x, bottom);
    c.lineWidth = 1; c.strokeStyle = 'rgba(139,151,166,0.45)'; c.stroke();
    c.beginPath(); c.arc(x, y, 4, 0, Math.PI*2); c.fillStyle = '#e6edf3'; c.fill();
    c.restore();
  }}
}};
try {{ Chart.register(_crosshair); }} catch(e) {{}}

function _priceColor(arr) {{
  const a = arr.find(v => v != null);
  let b = null; for (let i = arr.length-1; i >= 0; i--) {{ if (arr[i] != null) {{ b = arr[i]; break; }} }}
  return (b != null && a != null && b < a) ? '#f85149' : '#2ea043';
}}
function _areaFill(color) {{
  return (ctx) => {{
    const ch = ctx.chart, area = ch.chartArea; if (!area) return color + '00';
    const g = ch.ctx.createLinearGradient(0, area.top, 0, area.bottom);
    g.addColorStop(0, color + '44'); g.addColorStop(1, color + '00'); return g;
  }};
}}
function _updateReadout(idx) {{
  const r = document.getElementById('mReadout'); if (!r || !CUR.prices.length) return;
  if (idx == null || CUR.prices[idx] == null) idx = CUR.prices.length - 1;
  const p = CUR.prices[idx]; if (p == null) return;
  const chg = CUR.base ? (p/CUR.base - 1)*100 : 0;
  const up = chg >= 0, col = up ? '#2ea043' : '#f85149';
  const d = new Date(CUR.times[idx]);
  const ds = CUR.intraday
    ? d.toLocaleString([], {{month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}})
    : d.toLocaleDateString([], {{year:'numeric', month:'short', day:'numeric'}});
  r.innerHTML = `<span class="rprice">$${{p.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}})}}</span>`
    + `<span class="rchg" style="color:${{col}};">${{up?'+':''}}${{chg.toFixed(2)}}% over range</span>`
    + `<span class="rdate">${{ds}}</span>`;
}}
function _cleanOpts(unit, y2) {{
  const df = {{hour:'HH:mm', day:'MMM d', week:'MMM d', month:'MMM yyyy'}};
  const scales = {{
    x:{{type:'time', time:{{unit, displayFormats:df}}, ticks:{{color:'#8b97a6', maxTicksLimit:6}}, grid:{{display:false}}}},
    y:{{position:'right', ticks:{{color:'#8b97a6'}}, grid:{{color:'rgba(42,52,65,0.55)'}}}}
  }};
  if (y2) scales.y2 = {{position:'left', ticks:{{color:'#a371f7', callback:v=>v+'%'}}, grid:{{display:false}}}};
  return {{
    responsive:true, parsing:false, interaction:{{mode:'index', intersect:false}},
    onHover:(e, act) => {{ _updateReadout(act && act.length ? act[0].index : null); }},
    elements:{{line:{{tension:0.15}}}},
    plugins:{{
      legend:{{labels:{{color:'#8b97a6', usePointStyle:true, boxWidth:8,
        filter:(it)=> it.text && it.text[0] !== '_'}}}},
      tooltip:{{enabled:false}},
      zoom:{{ zoom:{{wheel:{{enabled:true}}, pinch:{{enabled:true}}, mode:'x'}}, pan:{{enabled:true, mode:'x'}} }}
    }},
    scales
  }};
}}

function _winStart(c, range) {{
  const n = c.t.length;
  if (range==='1M') return Math.max(0, n-21);
  if (range==='3M') return Math.max(0, n-63);
  if (range==='6M') return Math.max(0, n-126);
  if (range==='1Y') return 0;
  if (range==='YTD') {{
    const y = new Date().getUTCFullYear();
    const i = c.dates.findIndex(d => parseInt(d.slice(0,4),10) === y);
    return i < 0 ? 0 : i;
  }}
  return 0;
}}

function renderModalChart() {{
  const s = MODAL; if (!s) return;
  if (CSTATE.range === '1D' || CSTATE.range === '5D') return _renderIntraday(s);
  const c = DATA.charts[s.symbol], p = s.plan || {{}};
  const key = document.getElementById('mChartKey');
  if (mChart) mChart.destroy();
  if (!c) {{ key.innerHTML = ''; _hideMacd(); return; }}
  const st = _winStart(c, CSTATE.range);
  const T = c.t.slice(st), close=c.close.slice(st), fast=c.fast.slice(st), slow=c.slow.slice(st);
  const op=c.open.slice(st), hi=c.high.slice(st), lo=c.low.slice(st);
  const buys=c.buys.slice(st), sells=c.sells.slice(st);
  // reflect today's live price on the latest bar, if we have it
  const _lp = LIVE[s.symbol];
  if (_lp != null && close.length) {{
    close[close.length-1] = _lp;
    if (_lp > hi[hi.length-1]) hi[hi.length-1] = _lp;
    if (_lp < lo[lo.length-1]) lo[lo.length-1] = _lp;
  }}
  const useCandle = CSTATE.type==='candle' && _finOK;
  const pcol = _priceColor(close);
  const ds = [];
  if (useCandle) {{
    ds.push({{type:'candlestick', label:s.symbol, order:3,
      data:T.map((t,i)=>({{x:t,o:op[i],h:hi[i],l:lo[i],c:close[i]}})),
      color:{{up:'#2ea043',down:'#f85149',unchanged:'#8b97a6'}}}});
  }} else {{
    ds.push({{type:'line', label:'Price', order:3, borderColor:pcol, borderWidth:2,
      pointRadius:0, fill:true, backgroundColor:_areaFill(pcol), data:T.map((t,i)=>({{x:t,y:close[i]}}))}});
  }}
  if (CSTATE.avgs) {{
    ds.push({{type:'line', label:'20-day avg', order:4, borderColor:'rgba(56,139,253,0.65)', borderWidth:1,
      pointRadius:0, fill:false, data:T.map((t,i)=>({{x:t,y:fast[i]}}))}});
    ds.push({{type:'line', label:'50-day avg', order:4, borderColor:'rgba(240,136,62,0.65)', borderWidth:1,
      pointRadius:0, fill:false, data:T.map((t,i)=>({{x:t,y:slow[i]}}))}});
  }}
  if (CSTATE.boll && c.bb_up) {{
    const bu=c.bb_up.slice(st), bl=c.bb_lo.slice(st);
    ds.push({{type:'line', label:'Normal range (Bollinger)', order:5, borderColor:'rgba(163,113,247,0.5)',
      borderWidth:1, borderDash:[3,3], pointRadius:0, fill:false, data:T.map((t,i)=>({{x:t,y:bu[i]}}))}});
    ds.push({{type:'line', label:'_bblo', order:5, borderColor:'rgba(163,113,247,0.5)',
      borderWidth:1, borderDash:[3,3], pointRadius:0, fill:'-1', backgroundColor:'rgba(163,113,247,0.06)',
      data:T.map((t,i)=>({{x:t,y:bl[i]}}))}});
  }}
  const buyPts = T.map((t,i)=> buys[i]!=null ? {{x:t,y:buys[i]}} : null).filter(Boolean);
  const sellPts = T.map((t,i)=> sells[i]!=null ? {{x:t,y:sells[i]}} : null).filter(Boolean);
  ds.push({{type:'scatter', label:'Buy signal', data:buyPts, backgroundColor:'#2ea043',
    pointStyle:'triangle', radius:8, hoverRadius:10, order:1}});
  ds.push({{type:'scatter', label:'Sell signal', data:sellPts, backgroundColor:'#f85149',
    pointStyle:'triangle', rotation:180, radius:8, hoverRadius:10, order:1}});
  const tA=T[0], tB=T[T.length-1];
  const hline=(v,col,lab)=>({{type:'line',label:lab,borderColor:col,borderDash:[5,4],borderWidth:1,
    pointRadius:0,order:2,fill:false,data:[{{x:tA,y:v}},{{x:tB,y:v}}]}});
  if (p.entry!=null) ds.push(hline(p.entry,'rgba(139,151,166,0.7)','Buy near'));
  if (p.target!=null) ds.push(hline(p.target,'rgba(46,160,67,0.7)','Take-profit'));
  if (p.stop!=null) ds.push(hline(p.stop,'rgba(248,81,73,0.7)','Stop-loss'));
  let benchNote = '', y2 = false;
  if (CSTATE.bench && DATA.benchmark) {{
    const b = DATA.benchmark, bmap = {{}};
    b.t.forEach((t,i)=> bmap[t]=b.close[i]);
    let base=null; const bpts=[];
    T.forEach(t=>{{ const v=bmap[t]; if(v!=null){{ if(base==null) base=v; bpts.push({{x:t,y:(v/base-1)*100}}); }} }});
    if (bpts.length) {{
      ds.push({{type:'line', label:'S&P 500 (%)', data:bpts, borderColor:'#a371f7',
        borderWidth:1.3, pointRadius:0, fill:false, yAxisID:'y2', order:2}});
      y2 = true; benchNote = ' Purple = S&P 500 (% change, left axis) to compare.';
    }}
  }}
  CUR = {{ prices:close, times:T, base: close.find(v=>v!=null), intraday:false }};
  mChart = new Chart(document.getElementById('mChart'),
    {{ data:{{datasets:ds}}, options:_cleanOpts(CSTATE.range==='1M'?'week':'month', y2) }});
  _updateReadout();
  _renderMacd(s, T, st);
  const nB=buyPts.length, nS=sellPts.length;
  key.innerHTML = `Hover to scrub. ▲ ${{nB}} past buys &nbsp; ▼ ${{nS}} past sells &nbsp; dashed = entry / target / stop.` + benchNote;
}}

function _hideMacd() {{
  if (macdChart) {{ macdChart.destroy(); macdChart = null; }}
  const b = document.getElementById('macdBox'); if (b) b.style.display = 'none';
}}
function _renderMacd(s, T, st) {{
  const c = DATA.charts[s.symbol];
  if (!CSTATE.macd || !c || !c.macd) {{ _hideMacd(); return; }}
  document.getElementById('macdBox').style.display = 'block';
  const ml=c.macd.slice(st), msig=c.macd_sig.slice(st), mh=c.macd_hist.slice(st);
  const ds = [
    {{type:'bar', label:'_h', data:T.map((t,i)=>({{x:t,y:mh[i]}})), order:3, borderWidth:0,
      backgroundColor:T.map((t,i)=> (mh[i]>=0?'rgba(46,160,67,0.6)':'rgba(248,81,73,0.6)'))}},
    {{type:'line', label:'_m', data:T.map((t,i)=>({{x:t,y:ml[i]}})), borderColor:'#388bfd', borderWidth:1.2, pointRadius:0, order:1}},
    {{type:'line', label:'_s', data:T.map((t,i)=>({{x:t,y:msig[i]}})), borderColor:'#f0883e', borderWidth:1.2, pointRadius:0, order:1}},
  ];
  if (macdChart) macdChart.destroy();
  macdChart = new Chart(document.getElementById('mMacd'), {{
    data:{{datasets:ds}},
    options:{{responsive:true, parsing:false, interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{display:false}}, tooltip:{{enabled:false}}}},
      scales:{{x:{{type:'time', time:{{unit: CSTATE.range==='1M'?'week':'month'}}, ticks:{{display:false}}, grid:{{display:false}}}},
               y:{{position:'right', ticks:{{color:'#8b97a6', maxTicksLimit:3}}, grid:{{color:'rgba(42,52,65,0.5)'}}}}}}}}
  }});
}}

// ---- intraday (1D / 5D) — fetched live from the Worker proxy ----
function _renderIntraday(s) {{
  _hideMacd();
  const key = document.getElementById('mChartKey');
  if (mChart) {{ mChart.destroy(); mChart = null; }}
  if (!LIVE_URL) {{ key.innerHTML = 'Intraday view needs the live data proxy (LIVE_QUOTES_URL).'; return; }}
  const cacheKey = s.symbol + ':' + CSTATE.range;
  if (CSTATE.range === '5D' && INTRA[cacheKey]) {{ _drawIntra(s, INTRA[cacheKey]); return; }}
  key.innerHTML = 'Loading intraday…';
  const tf = CSTATE.range === '1D' ? '15Min' : '1Hour';
  const days = CSTATE.range === '1D' ? 3 : 8;
  fetch(LIVE_URL + '?bars=' + encodeURIComponent(s.symbol) + '&tf=' + tf + '&days=' + days)
    .then(r => r.json())
    .then(d => {{
      let bars = d.bars || [];
      if (CSTATE.range === '1D' && bars.length) {{
        const lastDay = new Date(bars[bars.length-1].t).toDateString();
        const today = bars.filter(b => new Date(b.t).toDateString() === lastDay);
        bars = today.length >= 2 ? today : bars.slice(-26);
      }}
      if (CSTATE.range === '5D') INTRA[cacheKey] = bars;
      if (MODAL && MODAL.symbol === s.symbol) _drawIntra(s, bars);
    }})
    .catch(() => {{ key.innerHTML = 'Intraday unavailable right now (market may be closed).'; }});
}}
function _drawIntra(s, bars) {{
  const p = s.plan || {{}};
  const key = document.getElementById('mChartKey');
  if (mChart) {{ mChart.destroy(); mChart = null; }}
  if (!bars || !bars.length) {{ key.innerHTML = 'No intraday data (market may be closed).'; return; }}
  const T = bars.map(b=>b.t), close = bars.map(b=>b.c);
  const lp = LIVE[s.symbol]; if (lp != null) close[close.length-1] = lp;
  const useCandle = CSTATE.type === 'candle' && _finOK;
  const pcol = _priceColor(close);
  const ds = [];
  if (useCandle) ds.push({{type:'candlestick', label:s.symbol,
    data:bars.map(b=>({{x:b.t,o:b.o,h:b.h,l:b.l,c:b.c}})),
    color:{{up:'#2ea043',down:'#f85149',unchanged:'#8b97a6'}}}});
  else ds.push({{type:'line', label:'Price', borderColor:pcol, borderWidth:2, pointRadius:0,
    fill:true, backgroundColor:_areaFill(pcol), data:T.map((t,i)=>({{x:t,y:close[i]}}))}});
  // NOTE: no stop/target lines here — they sit ±5–15% away and would flatten an
  // intraday move into a sliver. Intraday is about the day's shape, so we fit
  // the y-axis tightly to the actual price range instead.
  const ys = (useCandle ? bars.flatMap(b=>[b.h,b.l]) : close).filter(v=>v!=null);
  const lo = Math.min(...ys), hi = Math.max(...ys), pad = Math.max((hi-lo)*0.08, hi*0.001);
  CUR = {{ prices:close, times:T, base: close[0], intraday:true }};
  const opts = _cleanOpts(CSTATE.range==='1D'?'hour':'day', false);
  opts.scales.y.min = lo - pad; opts.scales.y.max = hi + pad;
  mChart = new Chart(document.getElementById('mChart'), {{ data:{{datasets:ds}}, options:opts }});
  _updateReadout();
  // honest, accurate label from the data itself (handles weekends / closed days)
  const fmt = ms => new Date(ms).toLocaleDateString([], {{month:'short', day:'numeric'}});
  const span = (CSTATE.range==='1D') ? ('Session of ' + fmt(T[T.length-1]))
                                     : (fmt(T[0]) + ' – ' + fmt(T[T.length-1]));
  key.innerHTML = `${{span}} · ${{CSTATE.range==='1D'?'15-min':'1-hour'}} bars, live. Hover to scrub. `
    + `(Stop/target levels are on the daily ranges.)`;
}}

// build chart controls once
(function setupChartControls() {{
  const rb = document.getElementById('rangeBtns');
  ['1D','5D','1M','3M','6M','YTD','1Y'].forEach(r => {{
    const b=document.createElement('button'); b.textContent=r; b.dataset.range=r;
    if (r===CSTATE.range) b.className='on';
    b.onclick=()=>{{ CSTATE.range=r; rb.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.range===r)); renderModalChart(); }};
    rb.appendChild(b);
  }});
  const tb = document.getElementById('typeBtns');
  [['line','Line'],['candle','Candles']].forEach(([v,lab]) => {{
    const b=document.createElement('button'); b.textContent=lab; b.dataset.type=v;
    if (v===CSTATE.type) b.className='on';
    b.onclick=()=>{{
      if (v==='candle' && !_finOK) {{ alert('Candlestick view is unavailable (chart library not loaded).'); return; }}
      CSTATE.type=v; tb.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.type===v)); renderModalChart();
    }};
    tb.appendChild(b);
  }});
  const bt = document.getElementById('benchToggle');
  bt.checked = CSTATE.bench;
  bt.onchange = () => {{ CSTATE.bench = bt.checked; renderModalChart(); }};
  // indicator toggles
  const ib = document.getElementById('indBtns');
  [['avgs','Averages'],['boll','Bollinger'],['macd','MACD']].forEach(([k,lab]) => {{
    const b=document.createElement('button'); b.textContent=lab; b.dataset.ind=k;
    if (CSTATE[k]) b.className='on';
    b.onclick=()=>{{ CSTATE[k]=!CSTATE[k]; b.classList.toggle('on', CSTATE[k]); renderModalChart(); }};
    ib.appendChild(b);
  }});
  // zoom buttons (x-axis)
  const zi=document.getElementById('zoomIn'), zo=document.getElementById('zoomOut'), zr=document.getElementById('zoomReset');
  if (zi) zi.onclick=()=>{{ if(mChart&&mChart.zoom) mChart.zoom(1.3); }};
  if (zo) zo.onclick=()=>{{ if(mChart&&mChart.zoom) mChart.zoom(0.77); }};
  if (zr) zr.onclick=()=>{{ if(mChart&&mChart.resetZoom) mChart.resetZoom(); }};
}})();

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
  }}
  tabs.forEach(b => b.addEventListener('click', () => show(b.dataset.page)));
  let saved = 'signals';
  try {{ saved = localStorage.getItem('tab') || 'signals'; }} catch (e) {{}}
  show(saved);
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
