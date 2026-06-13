// Cloudflare Worker - live-quote proxy for the trading dashboard.
//
// It holds your Alpaca keys as Worker SECRETS (never sent to the browser) and
// returns the latest trade price for the requested symbols, with CORS enabled
// so the static GitHub Pages site can call it safely.
//
// Request:  GET https://<your-worker>.workers.dev/?symbols=AAPL,MSFT,SPY
// Response: {"prices":{"AAPL":201.23,"MSFT":410.1,...},"at":"<ISO time>"}
//
// Set two secrets on the Worker (see LIVE_SETUP.md):
//   ALPACA_API_KEY, ALPACA_SECRET_KEY

export default {
  async fetch(request, env) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "*",
      "Cache-Control": "no-store",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    const url = new URL(request.url);
    const headers = {
      "APCA-API-KEY-ID": env.ALPACA_API_KEY,
      "APCA-API-SECRET-KEY": env.ALPACA_SECRET_KEY,
    };

    // --- Yahoo Finance OHLCV chart: /?chart=AAPL&range=6mo&interval=1d ---
    // Full-market consolidated data (cleaner than IEX), keyless, all ranges.
    const chSym = (url.searchParams.get("chart") || "").trim();
    if (chSym) {
      const range = (url.searchParams.get("range") || "6mo").trim();
      const interval = (url.searchParams.get("interval") || "1d").trim();
      const yurl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(chSym)}`
        + `?range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}&includePrePost=false`;
      try {
        const up = await fetch(yurl, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              + "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json",
          },
        });
        if (!up.ok) return json({ error: "yahoo " + up.status }, 502, cors);
        const d = await up.json();
        const r = d && d.chart && d.chart.result && d.chart.result[0];
        if (!r) return json({ error: "no data" }, 502, cors);
        const ts = r.timestamp || [];
        const q = (r.indicators && r.indicators.quote && r.indicators.quote[0]) || {};
        const bars = [];
        for (let i = 0; i < ts.length; i++) {
          const c = q.close ? q.close[i] : null;
          if (c == null) continue;
          bars.push({
            t: ts[i] * 1000,
            o: q.open ? q.open[i] : null,
            h: q.high ? q.high[i] : null,
            l: q.low ? q.low[i] : null,
            c,
            v: q.volume ? (q.volume[i] || 0) : 0,
          });
        }
        const meta = r.meta || {};
        return json({ symbol: chSym, range, interval, bars,
          exchange: meta.exchangeName, currency: meta.currency }, 200, cors);
      } catch (e) {
        return json({ error: "fetch failed" }, 502, cors);
      }
    }

    // --- intraday bars: /?bars=AAPL&tf=15Min&days=2 ---
    const barsSym = (url.searchParams.get("bars") || "").trim();
    if (barsSym) {
      const tf = (url.searchParams.get("tf") || "15Min").trim();
      const days = Math.min(parseInt(url.searchParams.get("days") || "2", 10) || 2, 10);
      const start = new Date(Date.now() - days * 24 * 3600 * 1000).toISOString();
      const api = `https://data.alpaca.markets/v2/stocks/${encodeURIComponent(barsSym)}/bars`
        + `?timeframe=${encodeURIComponent(tf)}&feed=iex&adjustment=split&limit=1000&start=${encodeURIComponent(start)}`;
      try {
        const up = await fetch(api, { headers });
        if (!up.ok) return json({ error: "upstream " + up.status }, 502, cors);
        const d = await up.json();
        const bars = (d.bars || []).map(b => ({ t: Date.parse(b.t), o: b.o, h: b.h, l: b.l, c: b.c }));
        return json({ symbol: barsSym, tf, bars }, 200, cors);
      } catch (e) {
        return json({ error: "fetch failed" }, 502, cors);
      }
    }

    // --- TradingView TA ratings: /?tv=AAPL,NVDA,SPY ---
    // Proxies the (unofficial) TradingView scanner from the Worker's egress, since the
    // endpoint blocks datacenter IPs (GitHub Actions). Returns the raw scanner payload.
    const tvSyms = (url.searchParams.get("tv") || "").trim();
    if (tvSyms) {
      const exch = ["NASDAQ", "NYSE", "AMEX"];
      const syms = tvSyms.split(",").map(s => s.trim()).filter(Boolean).slice(0, 60);
      const tickers = [];
      for (const s of syms) for (const e of exch) tickers.push(`${e}:${s}`);
      const body = JSON.stringify({
        symbols: { tickers, query: { types: [] } },
        columns: ["Recommend.All", "Recommend.All|1W"],
      });
      try {
        const up = await fetch("https://scanner.tradingview.com/america/scan", {
          method: "POST",
          headers: { "User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Accept": "application/json" },
          body,
        });
        if (!up.ok) return json({ error: "tv " + up.status }, 502, cors);
        return json(await up.json(), 200, cors);
      } catch (e) {
        return json({ error: "fetch failed" }, 502, cors);
      }
    }

    // --- latest quotes: /?symbols=AAPL,MSFT ---
    const symbols = (url.searchParams.get("symbols") || "").trim();
    if (!symbols) return json({ error: "pass ?symbols=AAPL,MSFT or ?bars=AAPL" }, 400, cors);

    const api = "https://data.alpaca.markets/v2/stocks/trades/latest?feed=iex&symbols="
      + encodeURIComponent(symbols);
    let upstream;
    try {
      upstream = await fetch(api, { headers });
    } catch (e) {
      return json({ error: "fetch failed" }, 502, cors);
    }
    if (!upstream.ok) return json({ error: "upstream " + upstream.status }, 502, cors);

    const data = await upstream.json();
    const trades = data.trades || {};
    const prices = {};
    for (const sym in trades) prices[sym] = trades[sym].p;  // .p = trade price
    return json({ prices, at: new Date().toISOString() }, 200, cors);
  },
};

function json(obj, status, cors) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { ...cors, "content-type": "application/json" },
  });
}
