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


def build_snapshot() -> dict:
    mode = _mode()
    live = mode != "SYNTHETIC"

    rows = scanner.scan(CONFIG, live=live)

    # split chart data out of each row for compactness
    charts = {r["symbol"]: r.pop("chart") for r in rows}

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

    # Optional AI analyst note per signal (silent no-op if no key).
    if CONFIG.llm_enabled:
        import llm
        for r in shown:
            note = llm.analyst_note(r, CONFIG)
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
      <div class="ovhead">📈 Live overview — S&amp;P 500 vs your top signals <span style="font-weight:400;color:var(--muted);">(% change)</span> <span id="ovStatus"></span></div>
      <canvas id="overviewChart" height="92"></canvas>
    </div>
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

      <h4>The three things each signal checks</h4>
      <ul>
        <li><b>Trend</b> — is the short-term average above the long-term one? (direction)</li>
        <li><b>Momentum (RSI)</b> — a 0–100 gauge of how fast/far price has moved; we want room to rise,
        not already overheated.</li>
        <li><b>Volume</b> — are more people trading it than usual? Heavy volume gives a move more conviction.</li>
      </ul>

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
    <div class="sech">Should you take it? <span id="mConvScore"></span></div>
    <ul class="checks" id="mChecks"></ul>
    <div class="sech">The trade plan <span id="mPlanNote" style="text-transform:none;color:var(--muted);"></span></div>
    <div class="plangrid" id="mPlan"></div>
    <div class="sech">Chart: where it would buy &amp; sell</div>
    <div class="chartctl">
      <span class="ctlgrp" id="rangeBtns"></span>
      <span class="ctlgrp" id="typeBtns"></span>
      <label class="ctltog"><input type="checkbox" id="benchToggle" checked> Compare to S&amp;P 500</label>
      <span class="ctlgrp">
        <button id="zoomOut" title="Zoom out">&minus;</button>
        <button id="zoomIn" title="Zoom in">+</button>
      </span>
      <button class="ctlbtn" id="zoomReset">Reset</button>
    </div>
    <div class="chartkey" style="margin:0 0 6px;">Tip: use &minus;/+ or scroll / pinch to zoom, drag to pan, Reset to fit.</div>
    <div class="chartbox"><canvas id="mChart" height="130"></canvas></div>
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
    <div><span class="sym">${{s.symbol}}</span>
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
    <div class="more">${{nNews ? nNews+' news &middot; ':''}}click for full plan + reasoning →</div>`;
  el.addEventListener('click', () => openModal(s));
  return el;
}}
// group by sector, preserving the ranked order (sector ordered by its best member)
const _bySector = {{}}, _sectorOrder = [];
DATA.signals.forEach(s => {{
  const sec = s.sector || 'Other / Movers';
  if (!_bySector[sec]) {{ _bySector[sec] = []; _sectorOrder.push(sec); }}
  _bySector[sec].push(s);
}});
_sectorOrder.forEach(sec => {{
  const h = document.createElement('div'); h.className='secthead';
  h.textContent = sec + ' · ' + _bySector[sec].length;
  cards.appendChild(h);
  const grid = document.createElement('div'); grid.className='grid';
  _bySector[sec].forEach(s => grid.appendChild(makeCard(s)));
  cards.appendChild(grid);
}});

// ---- live prices (via Cloudflare Worker proxy) ----
const LIVE_SYMS = [...new Set(DATA.signals.map(s => s.symbol).concat('SPY'))];
let LIVE = {{}};  // latest live prices, shared with the charts
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

// ---- live overview chart: S&P 500 vs top signals, normalised to % change ----
let ovChart = null;
const OV_BASE = {{}};   // symbol -> base price for rebasing
const OV_WIN = 60;      // ~3 months of daily bars
const OV_PALETTE = ['#388bfd','#2ea043','#f0883e','#f85149','#a371f7','#e8c878',
                    '#56d4dd','#db61a2','#6cc644','#bd8b00','#8b97a6','#ff7b72'];
function _rebase(t, close) {{
  const n = close.length, st = Math.max(0, n - OV_WIN);
  const T = t.slice(st), C = close.slice(st);
  let base = null;
  for (const v of C) {{ if (v != null) {{ base = v; break; }} }}
  if (!base) return {{ pts: [], base: null }};
  return {{ pts: T.map((tt, i) => ({{x: tt, y: (C[i]/base - 1) * 100}})), base }};
}}
function buildOverview() {{
  const cv = document.getElementById('overviewChart');
  if (!cv || typeof Chart === 'undefined') return;
  // pick the index + the most actionable signals (BUY/SELL/HOLD), capped for readability
  let picks = DATA.signals.filter(s => s.action !== 'FLAT').slice(0, 10).map(s => s.symbol);
  if (picks.length < 4) picks = DATA.signals.slice(0, 8).map(s => s.symbol);
  const ds = [];
  if (DATA.benchmark) {{
    const r = _rebase(DATA.benchmark.t, DATA.benchmark.close);
    if (r.pts.length) {{ OV_BASE['SPY'] = r.base;
      ds.push({{label:'S&P 500', data:r.pts, borderColor:'#e6edf3', borderWidth:2.4, pointRadius:0, order:1}}); }}
  }}
  picks.forEach((sym, i) => {{
    const c = DATA.charts[sym]; if (!c) return;
    const r = _rebase(c.t, c.close); if (!r.pts.length) return;
    OV_BASE[sym] = r.base;
    ds.push({{label:sym, data:r.pts, borderColor:OV_PALETTE[i % OV_PALETTE.length],
      borderWidth:1.3, pointRadius:0, order:2}});
  }});
  if (ovChart) ovChart.destroy();
  ovChart = new Chart(cv, {{
    data:{{datasets:ds}},
    options:{{responsive:true, parsing:false, interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{color:'#8b97a6', usePointStyle:true, boxWidth:8, font:{{size:11}}}}}},
        zoom:{{zoom:{{wheel:{{enabled:true}}, pinch:{{enabled:true}}, mode:'x'}}, pan:{{enabled:true, mode:'x'}}}}}},
      scales:{{x:{{type:'time', time:{{unit:'month'}}, ticks:{{color:'#8b97a6',maxTicksLimit:8}}, grid:{{color:'#2a3441'}}}},
               y:{{ticks:{{color:'#8b97a6', callback:v=>(v>0?'+':'')+v+'%'}}, grid:{{color:'#2a3441'}}}}}}}}
  }});
}}
function _updateOverviewLive() {{
  if (!ovChart) return;
  ovChart.data.datasets.forEach(d => {{
    const sym = d.label === 'S&P 500' ? 'SPY' : d.label;
    const lp = LIVE[sym], base = OV_BASE[sym];
    if (lp != null && base && d.data.length) d.data[d.data.length - 1].y = (lp/base - 1) * 100;
  }});
  ovChart.update('none');
}}
buildOverview();

// ---- detail modal ----
const overlay = document.getElementById('overlay');
let mChart;
function openModal(s) {{
  const cls = (s.action||'').replace(' ','');
  document.getElementById('mTitle').innerHTML =
    `${{s.symbol}} <span class="act a-${{cls}}" style="float:none;font-size:13px;">${{s.action}}</span> &nbsp; <span data-px="${{s.symbol}}" style="color:var(--muted);font-size:15px;">$${{s.price.toLocaleString()}}</span>`
    + (s.name ? `<div class="cname" style="font-size:13px;margin-top:4px;">${{s.name}}${{s.exchange?` · ${{s.exchange}}`:''}}</div>` : '');
  document.getElementById('mSummary').textContent = s.summary || '';
  document.getElementById('mDesk').textContent = s.desk_read || '';
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
const CSTATE = {{ range:'6M', type:'line', bench:true }};
let _finOK = false;
try {{ _finOK = !!(window.Chart && Chart.registry.getController('candlestick')); }} catch(e) {{ _finOK = false; }}

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
  const c = DATA.charts[s.symbol], p = s.plan || {{}};
  const key = document.getElementById('mChartKey');
  if (mChart) mChart.destroy();
  if (!c) {{ key.innerHTML = ''; return; }}
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
  const ds = [];
  if (useCandle) {{
    ds.push({{type:'candlestick', label:s.symbol, order:3,
      data:T.map((t,i)=>({{x:t,o:op[i],h:hi[i],l:lo[i],c:close[i]}})),
      color:{{up:'#2ea043',down:'#f85149',unchanged:'#8b97a6'}}}});
  }} else {{
    ds.push({{type:'line', label:'Price', order:3, borderColor:'#e6edf3', borderWidth:1.5,
      pointRadius:0, data:T.map((t,i)=>({{x:t,y:close[i]}}))}});
  }}
  ds.push({{type:'line', label:'Short-term avg', order:3, borderColor:'#388bfd', borderWidth:1.2,
    pointRadius:0, data:T.map((t,i)=>({{x:t,y:fast[i]}}))}});
  ds.push({{type:'line', label:'Long-term avg', order:3, borderColor:'#f0883e', borderWidth:1.2,
    pointRadius:0, data:T.map((t,i)=>({{x:t,y:slow[i]}}))}});
  const buyPts = T.map((t,i)=> buys[i]!=null ? {{x:t,y:buys[i]}} : null).filter(Boolean);
  const sellPts = T.map((t,i)=> sells[i]!=null ? {{x:t,y:sells[i]}} : null).filter(Boolean);
  ds.push({{type:'scatter', label:'Buy signal', data:buyPts, backgroundColor:'#2ea043',
    pointStyle:'triangle', radius:9, hoverRadius:11, order:1}});
  ds.push({{type:'scatter', label:'Sell signal', data:sellPts, backgroundColor:'#f85149',
    pointStyle:'triangle', rotation:180, radius:9, hoverRadius:11, order:1}});
  const tA=T[0], tB=T[T.length-1];
  const hline=(v,col,lab)=>({{type:'line',label:lab,borderColor:col,borderDash:[6,4],borderWidth:1,
    pointRadius:0,order:2,data:[{{x:tA,y:v}},{{x:tB,y:v}}]}});
  if (p.entry!=null) ds.push(hline(p.entry,'#8b97a6','Buy near'));
  if (p.target!=null) ds.push(hline(p.target,'#2ea043','Take-profit'));
  if (p.stop!=null) ds.push(hline(p.stop,'#f85149','Stop-loss'));
  const scales = {{
    x:{{type:'time', time:{{unit: CSTATE.range==='1M'?'week':'month'}},
       ticks:{{color:'#8b97a6',maxTicksLimit:8}}, grid:{{color:'#2a3441'}}}},
    y:{{ticks:{{color:'#8b97a6'}}, grid:{{color:'#2a3441'}}}}
  }};
  let benchNote = '';
  if (CSTATE.bench && DATA.benchmark) {{
    const b = DATA.benchmark, bmap = {{}};
    b.t.forEach((t,i)=> bmap[t]=b.close[i]);
    let base=null; const bpts=[];
    T.forEach(t=>{{ const v=bmap[t]; if(v!=null){{ if(base==null) base=v; bpts.push({{x:t,y:(v/base-1)*100}}); }} }});
    if (bpts.length) {{
      ds.push({{type:'line', label:'S&P 500 (% change)', data:bpts, borderColor:'#a371f7',
        borderWidth:1.3, pointRadius:0, yAxisID:'y2', order:2}});
      scales.y2 = {{position:'right', ticks:{{color:'#a371f7', callback:v=>v+'%'}}, grid:{{drawOnChartArea:false}}}};
      benchNote = ' Purple line = S&P 500 over the same window (right axis, % change) so you can compare.';
    }}
  }}
  mChart = new Chart(document.getElementById('mChart'), {{
    data:{{datasets:ds}},
    options:{{responsive:true, parsing:false, interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{labels:{{color:'#8b97a6', usePointStyle:true, boxWidth:8}}}},
        zoom:{{
          zoom:{{wheel:{{enabled:true}}, pinch:{{enabled:true}}, drag:{{enabled:false}}, mode:'xy'}},
          pan:{{enabled:true, mode:'xy'}}
        }}
      }}, scales}}
  }});
  const nB=buyPts.length, nS=sellPts.length;
  key.innerHTML = `▲ green = past buy signals (${{nB}}) &nbsp; ▼ red = past sell signals (${{nS}}) &nbsp; `
    + `dashed lines = your buy / take-profit / stop levels.` + benchNote;
}}

// build chart controls once
(function setupChartControls() {{
  const rb = document.getElementById('rangeBtns');
  ['1M','3M','6M','YTD','1Y'].forEach(r => {{
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
  const zr = document.getElementById('zoomReset');
  if (zr) zr.onclick = () => {{ if (mChart && mChart.resetZoom) mChart.resetZoom(); }};
  const zi = document.getElementById('zoomIn');
  if (zi) zi.onclick = () => {{ if (mChart && mChart.zoom) mChart.zoom(1.3); }};
  const zo = document.getElementById('zoomOut');
  if (zo) zo.onclick = () => {{ if (mChart && mChart.zoom) mChart.zoom(0.77); }};
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
