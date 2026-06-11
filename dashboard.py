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

    flagged = [r["symbol"] for r in rows if r["action"] in ("BUY", "SELL")][:8]
    if live:
        try:
            news = market.get_news(flagged or [r["symbol"] for r in rows[:5]], CONFIG)
        except Exception as exc:  # noqa: BLE001
            news = [{"headline": f"(news unavailable: {exc})", "source": "",
                     "created_at": "", "url": "", "symbols": []}]
    else:
        news = _synthetic_news([r["symbol"] for r in rows])

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "mode": mode,
        "scanned": len(rows),
        "diagnostics": list(scanner.LAST_ERRORS),
        "params": {
            "fast_ma": CONFIG.fast_ma, "slow_ma": CONFIG.slow_ma,
            "rsi_period": CONFIG.rsi_period, "risk_per_trade": CONFIG.risk_per_trade,
            "stop_loss_pct": CONFIG.stop_loss_pct, "take_profit_pct": CONFIG.take_profit_pct,
            "rel_volume_window": CONFIG.rel_volume_window,
        },
        "signals": rows[: CONFIG.show_top],
        "charts": {k: charts[k] for k in (r["symbol"] for r in rows[: CONFIG.show_top]) if k in charts},
        "news": news,
    }


def render_html(snap: dict) -> str:
    data_json = json.dumps(snap)
    mode = snap["mode"]
    mode_note = {
        "LIVE": "Live account data. Real money is at risk if you act on these.",
        "PAPER": "Alpaca paper data and account.",
        "SYNTHETIC": "Synthetic data — NOT real prices or news. Add Alpaca keys for the real thing.",
    }[mode]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Signals Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
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
    padding:16px; }}
  .sym {{ font-size:18px; font-weight:700; }}
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
</style></head>
<body><div class="wrap">
  <h1>Trading Signals Dashboard</h1>
  <div class="meta">Generated {snap['generated_at']} &middot;
    <span class="badge m-{mode}">{mode}</span> &middot;
    scanned {snap['scanned']} symbols</div>
  <div class="note">{mode_note}</div>
  <div id="diag"></div>

  <h2>Signals</h2>
  <div class="grid" id="cards"></div>

  <h2>Price &amp; moving averages</h2>
  <label for="symsel">Symbol:</label> <select id="symsel"></select>
  <div class="chartbox"><canvas id="chart" height="110"></canvas></div>

  <h2>News</h2>
  <ul class="news" id="news"></ul>

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
<script>
const DATA = {data_json};
const diag = document.getElementById('diag');
if ((DATA.diagnostics||[]).length) {{
  diag.innerHTML = '<div style="background:#3a1e1e;border:1px solid #5a1e1e;color:#ff9b9b;'
    + 'border-radius:10px;padding:14px;margin:8px 0 18px;font-size:13px;">'
    + '<b>No signals to show.</b> Diagnostic:<br>'
    + DATA.diagnostics.map(e => '&bull; '+e).join('<br>') + '</div>';
}}
const cards = document.getElementById('cards');
DATA.signals.forEach(s => {{
  const el = document.createElement('div'); el.className='card';
  const cls = (s.action||'').replace(' ','');
  const hot = (s.rel_volume!=null && s.rel_volume>=1.5) ? ' hot' : '';
  el.innerHTML = `
    <div><span class="sym">${{s.symbol}}</span>
      <span class="act a-${{cls}}">${{s.action}}</span></div>
    <div class="px">$${{s.price.toLocaleString()}}</div>
    <div class="kv"><span>As of</span><span>${{s.as_of}}</span></div>
    <div class="kv"><span>RSI</span><span>${{s.rsi}}</span></div>
    <div class="kv"><span>Fast / Slow MA</span><span>${{s.fast_ma}} / ${{s.slow_ma}}</span></div>
    ${{s.rel_volume!=null ? `<div class="kv${{hot}}"><span>Rel vol (flow proxy)</span><span>${{s.rel_volume}}x</span></div>`:''}}
    ${{s.stop!=null ? `<div class="kv"><span>Stop / Target</span><span>$${{s.stop}} / $${{s.target}}</span></div>`:''}}
    ${{s.suggested_shares ? `<div class="kv"><span>Suggested size</span><span>${{s.suggested_shares}} sh</span></div>`:''}}`;
  cards.appendChild(el);
}});

const sel = document.getElementById('symsel');
Object.keys(DATA.charts).forEach(sym => {{ const o=document.createElement('option'); o.value=o.textContent=sym; sel.appendChild(o); }});
let chart;
function draw(sym) {{
  const c = DATA.charts[sym]; if (!c) return;
  const ds = [
    {{label:'Close', data:c.close, borderColor:'#e6edf3', borderWidth:1.5, pointRadius:0}},
    {{label:'Fast MA', data:c.fast, borderColor:'#388bfd', borderWidth:1.5, pointRadius:0}},
    {{label:'Slow MA', data:c.slow, borderColor:'#f0883e', borderWidth:1.5, pointRadius:0}},
  ];
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chart'), {{
    type:'line', data:{{labels:c.dates, datasets:ds}},
    options:{{responsive:true, interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{color:'#8b97a6'}}}}}},
      scales:{{x:{{ticks:{{color:'#8b97a6',maxTicksLimit:8}},grid:{{color:'#2a3441'}}}},
               y:{{ticks:{{color:'#8b97a6'}},grid:{{color:'#2a3441'}}}}}}}}
  }});
}}
sel.addEventListener('change', e => draw(e.target.value));
if (sel.options.length) draw(sel.value);

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
