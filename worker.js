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
  // Cron Trigger: reliably kick the GitHub rebuild (GitHub's own cron is best-effort and
  // skips runs under load). Configure the schedule in Cloudflare → Worker → Triggers → Cron,
  // and set secrets GH_TOKEN (fine-grained PAT, Actions: read+write) + optional GH_REPO /
  // GH_WORKFLOW. No-ops quietly if GH_TOKEN isn't set.
  async scheduled(event, env, ctx) {
    if (!env.GH_TOKEN) return;
    const repo = env.GH_REPO || "shak977/trading-bot";
    const wf = env.GH_WORKFLOW || "weekly-signals.yml";
    const url = `https://api.github.com/repos/${repo}/actions/workflows/${wf}/dispatches`;
    ctx.waitUntil(fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GH_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "trading-bot-cron",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: env.GH_BRANCH || "main" }),
    }).catch(() => {}));
  },

  async fetch(request, env, ctx) {
    // --- Telegram bot control: POST /?telegram=1 (the webhook) ---
    // Text /status, /scan, /analyst, /test, /help to your bot. Locked to YOUR chat id, and (if set)
    // verified by a secret token so only real Telegram updates are accepted. Needs Worker secrets:
    //   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TG_WEBHOOK_SECRET  (+ the existing GH_TOKEN).
    if (request.method === "POST") {
      if (env.TG_WEBHOOK_SECRET &&
          request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.TG_WEBHOOK_SECRET) {
        return new Response("forbidden", { status: 403 });
      }
      let update;
      try { update = await request.json(); } catch (e) { return new Response("ok"); }
      const msg = update.message || update.edited_message || {};
      const chatId = msg.chat && String(msg.chat.id);
      const text = (msg.text || "").trim();
      if (!chatId || (env.TELEGRAM_CHAT_ID && chatId !== String(env.TELEGRAM_CHAT_ID))) {
        return new Response("ok");            // not the owner — ignore silently
      }
      ctx.waitUntil(handleTelegram(env, chatId, text));
      return new Response("ok");              // ack Telegram immediately (reply is sent async)
    }

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

    // --- StockTwits retail buzz: /?st=AAPL ---
    // Proxies StockTwits' public stream from the Worker's egress (it often blocks datacenter IPs).
    const stSym = (url.searchParams.get("st") || "").trim();
    if (stSym) {
      const su = `https://api.stocktwits.com/api/2/streams/symbol/${encodeURIComponent(stSym)}.json`;
      try {
        const up = await fetch(su, { headers: { "User-Agent": "Mozilla/5.0", "Accept": "application/json" } });
        if (!up.ok) return json({ error: "stocktwits " + up.status }, 502, cors);
        return json(await up.json(), 200, cors);
      } catch (e) {
        return json({ error: "fetch failed" }, 502, cors);
      }
    }

    // --- resolve a channel's CURRENT live video id: /?ytlive=CHANNEL_ID ---
    // YouTube's legacy embed/live_stream?channel= auto-resolve is deprecated and often
    // shows "video unavailable". Instead we fetch the channel's /live page server-side
    // (with a consent cookie + desktop UA so YouTube serves the real HTML, not a consent
    // wall) and pull out the live video id. The browser then embeds that exact video.
    const ytlive = (url.searchParams.get("ytlive") || "").trim();
    if (ytlive) {
      const tryUrls = [
        `https://www.youtube.com/channel/${encodeURIComponent(ytlive)}/live?hl=en&gl=US`,
        `https://www.youtube.com/embed/live_stream?channel=${encodeURIComponent(ytlive)}`,
      ];
      for (const yurl of tryUrls) {
        try {
          const up = await fetch(yurl, {
            headers: {
              "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                + "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
              "Accept-Language": "en-US,en;q=0.9",
              "Cookie": "CONSENT=YES+cb; PREF=hl=en",
            },
          });
          if (!up.ok) continue;
          const html = await up.text();
          const isLive = /"isLiveContent":true|"isLive":true|"liveBroadcastDetails"|"isLiveNow":true/.test(html);
          // The canonical <link> only resolves to a watch?v= when /live points at a real
          // video — trust it even through YouTube's bot-protection page (which still leaks it).
          const can = html.match(/<link rel="canonical" href="https:\/\/www\.youtube\.com\/watch\?v=([\w-]{11})"/);
          if (can) return json({ videoId: can[1], live: isLive }, 200, cors);
          // Embed page, or a page explicitly flagged live: take the first videoId we see.
          if (yurl.includes("/embed/") || isLive) {
            const vid = html.match(/"videoId":"([\w-]{11})"/);
            if (vid) return json({ videoId: vid[1], live: isLive }, 200, cors);
          }
        } catch (e) { /* try next */ }
      }
      return json({ error: "no live video" }, 404, cors);
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

// ---------------- Telegram bot control ----------------
const TG_HELP =
  "🤖 Trading-bot commands:\n" +
  "/status — latest signals + book health\n" +
  "/scan — run a fresh scan now (~2 min)\n" +
  "/analyst — run the nightly analyst review\n" +
  "/test — send yourself a test alert\n" +
  "/help — this menu";

async function handleTelegram(env, chatId, text) {
  const cmd = (text.split(/\s+/)[0] || "").toLowerCase().replace(/@.*$/, "");  // strip @botname
  try {
    if (cmd === "/scan") {
      await dispatchWorkflow(env, env.GH_WORKFLOW || "weekly-signals.yml");
      return tgSend(env, chatId, "🔄 Scan started — fresh signals in ~2 min. I'll ping you if anything strong shows up.");
    }
    if (cmd === "/analyst") {
      await dispatchWorkflow(env, "analyst.yml");
      return tgSend(env, chatId, "🧠 Analyst run started — it'll review the book and refresh its proposals.");
    }
    if (cmd === "/test") {
      await dispatchWorkflow(env, "alert-test.yml");
      return tgSend(env, chatId, "🔔 Alert test fired — a test message should arrive shortly.");
    }
    if (cmd === "/status") {
      return tgSend(env, chatId, await buildStatus(env));
    }
    return tgSend(env, chatId, TG_HELP);      // /help, /start, or anything unrecognised
  } catch (e) {
    return tgSend(env, chatId, "⚠️ Couldn't run that just now — try again in a moment.");
  }
}

async function dispatchWorkflow(env, wf) {
  if (!env.GH_TOKEN) throw new Error("no GH_TOKEN");
  const repo = env.GH_REPO || "shak977/trading-bot";
  const url = `https://api.github.com/repos/${repo}/actions/workflows/${wf}/dispatches`;
  const r = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GH_TOKEN}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "trading-bot-tg",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GH_BRANCH || "main" }),
  });
  if (!r.ok) throw new Error("dispatch " + r.status);
}

async function tgSend(env, chatId, text) {
  if (!env.TELEGRAM_BOT_TOKEN) return;
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, text, disable_web_page_preview: true }),
  }).catch(() => {});
}

async function buildStatus(env) {
  const site = (env.SITE_URL || "https://shak977.github.io/trading-bot").replace(/\/$/, "");
  try {
    const r = await fetch(site + "/signals.json", { cf: { cacheTtl: 0 } });
    if (!r.ok) return "Couldn't read the latest data (site returned " + r.status + ").";
    const d = await r.json();
    const regime = (d.regime && d.regime.label) || "—";
    const rows = (d.signals || []).filter(s => s.action === "BUY" || s.action === "SHORT");
    rows.sort((a, b) => ((b.conviction && b.conviction.score_pct) || 0) - ((a.conviction && a.conviction.score_pct) || 0));
    const top = rows.slice(0, 6).map(s => {
      const cp = (s.conviction && s.conviction.score_pct);
      const arrow = s.action === "BUY" ? "🟢" : "🔴";
      return `${arrow} ${s.symbol} ${s.action}${cp ? " — " + cp + "%" : ""}`;
    }).join("\n") || "no fresh BUY/SHORT signals right now";
    const wr = d.track && d.track.win_rate;
    const dd = d.paper_acct && d.paper_acct.day_pl_pct;
    const bits = [`📊 Market: ${regime}`];
    if (wr != null) bits.push(`track: ${wr}% win rate`);
    if (dd != null) bits.push(`today: ${dd > 0 ? "+" : ""}${dd}%`);
    return bits.join("  ·  ") + "\n\nTop signals:\n" + top + "\n\nBuilt " + (d.generated_at || "");
  } catch (e) {
    return "Couldn't read the latest data right now.";
  }
}
