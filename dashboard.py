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

    # Attach each ticker's own headlines to its row (for the click-through detail),
    # and fold a plain-English news line into the reasoning so it's news-aware.
    for r in shown:
        r["news"] = [n for n in news if r["symbol"] in (n.get("symbols") or [])][: CONFIG.news_per_symbol]
        if r["news"]:
            top = r["news"][0]["headline"]
            r.setdefault("reasons", []).append(
                f"📰 In the news: {len(r['news'])} recent stor"
                f"{'y' if len(r['news']) == 1 else 'ies'} mention {r['symbol']}. "
                f"Latest headline — “{top}”. Check these for a reason behind the move."
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
        "signals": shown,
        "charts": {k: charts[k] for k in shown_syms if k in charts},
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
    padding:16px; cursor:pointer; transition:border-color .12s, transform .12s; }}
  .card:hover {{ border-color:#3d4d5f; transform:translateY(-2px); }}
  .more {{ color:var(--muted); font-size:12px; margin-top:10px;
    border-top:1px solid var(--line); padding-top:8px; }}
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
</style></head>
<body><div class="wrap">
  <h1>Trading Signals Dashboard</h1>
  <div class="meta">Generated {snap['generated_at']} &middot;
    <span class="badge m-{mode}">{mode}</span> &middot;
    scanned {snap['scanned']} symbols</div>
  <div class="note">{mode_note}</div>
  <div id="diag"></div>

  <h2>Signals <span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">— click any card for the full reasoning</span></h2>
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
  const nNews = (s.news||[]).length;
  el.innerHTML = `
    <div><span class="sym">${{s.symbol}}</span>
      <span class="act a-${{cls}}">${{s.action}}</span></div>
    <div class="px">$${{s.price.toLocaleString()}}</div>
    <div class="kv"><span>As of</span><span>${{s.as_of}}</span></div>
    <div class="kv"><span>RSI</span><span>${{s.rsi}}</span></div>
    <div class="kv"><span>Fast / Slow MA</span><span>${{s.fast_ma}} / ${{s.slow_ma}}</span></div>
    ${{s.rel_volume!=null ? `<div class="kv${{hot}}"><span>Rel vol (flow proxy)</span><span>${{s.rel_volume}}x</span></div>`:''}}
    ${{s.stop!=null ? `<div class="kv"><span>Stop / Target</span><span>$${{s.stop}} / $${{s.target}}</span></div>`:''}}
    ${{s.suggested_shares ? `<div class="kv"><span>Suggested size</span><span>${{s.suggested_shares}} sh</span></div>`:''}}
    ${{s.conviction ? `<div class="kv"><span>Conviction</span><span><span class="convbadge conv-${{s.conviction.label}}" style="font-size:11px;">${{s.conviction.label}} ${{s.conviction.score_pct}}%</span></span></div>`:''}}
    <div class="more">${{nNews ? nNews+' news &middot; ':''}}click for full plan + reasoning →</div>`;
  el.addEventListener('click', () => openModal(s));
  cards.appendChild(el);
}});

// ---- detail modal ----
const overlay = document.getElementById('overlay');
let mChart;
function openModal(s) {{
  const cls = (s.action||'').replace(' ','');
  document.getElementById('mTitle').innerHTML =
    `${{s.symbol}} <span class="act a-${{cls}}" style="float:none;font-size:13px;">${{s.action}}</span> &nbsp; <span style="color:var(--muted);font-size:15px;">$${{s.price.toLocaleString()}}</span>`;
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
  // chart with simulated entries/exits + plan levels
  const c = DATA.charts[s.symbol];
  const key = document.getElementById('mChartKey');
  if (mChart) mChart.destroy();
  if (c) {{
    const flat = v => (v==null ? null : c.dates.map(()=>v));
    const ds = [
      {{label:'Price', data:c.close, borderColor:'#e6edf3', borderWidth:1.5, pointRadius:0, order:3}},
      {{label:'Short-term avg', data:c.fast, borderColor:'#388bfd', borderWidth:1.2, pointRadius:0, order:3}},
      {{label:'Long-term avg', data:c.slow, borderColor:'#f0883e', borderWidth:1.2, pointRadius:0, order:3}},
      {{label:'Buy signal', data:c.buys, borderColor:'transparent', backgroundColor:'#2ea043',
        pointStyle:'triangle', pointRadius:9, pointHoverRadius:11, showLine:false, order:1}},
      {{label:'Sell signal', data:c.sells, borderColor:'transparent', backgroundColor:'#f85149',
        pointStyle:'triangle', rotation:180, pointRadius:9, pointHoverRadius:11, showLine:false, order:1}},
    ];
    if (p.entry!=null) ds.push({{label:'Buy near', data:flat(p.entry), borderColor:'#8b97a6',
      borderDash:[4,4], borderWidth:1, pointRadius:0, order:2}});
    if (p.target!=null) ds.push({{label:'Take-profit', data:flat(p.target), borderColor:'#2ea043',
      borderDash:[6,4], borderWidth:1, pointRadius:0, order:2}});
    if (p.stop!=null) ds.push({{label:'Stop-loss', data:flat(p.stop), borderColor:'#f85149',
      borderDash:[6,4], borderWidth:1, pointRadius:0, order:2}});
    mChart = new Chart(document.getElementById('mChart'), {{
      type:'line', data:{{labels:c.dates, datasets:ds}},
      options:{{responsive:true, interaction:{{mode:'index',intersect:false}},
        plugins:{{legend:{{labels:{{color:'#8b97a6', usePointStyle:true, boxWidth:8}}}}}},
        scales:{{x:{{ticks:{{color:'#8b97a6',maxTicksLimit:8}},grid:{{color:'#2a3441'}}}},
                 y:{{ticks:{{color:'#8b97a6'}},grid:{{color:'#2a3441'}}}}}}}}
    }});
    const nB = c.buys.filter(x=>x!=null).length, nS = c.sells.filter(x=>x!=null).length;
    key.innerHTML = `▲ green = past buy signals (${{nB}}) &nbsp; ▼ red = past sell signals (${{nS}}) &nbsp; `
      + `dashed lines = your buy / take-profit / stop levels. This shows where the strategy `
      + `would have entered and exited over the last few months.`;
  }} else {{ key.innerHTML = ''; }}
  overlay.classList.add('open');
}}
function closeModal() {{ overlay.classList.remove('open'); }}
document.getElementById('modalClose').addEventListener('click', closeModal);
overlay.addEventListener('click', e => {{ if (e.target === overlay) closeModal(); }});
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});

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
