// Cloudflare Worker — live-quote proxy for the trading dashboard.
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
