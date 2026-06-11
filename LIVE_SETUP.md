# Live prices — Cloudflare Worker setup (one-time, ~15 min, free)

This makes the dashboard show **live prices** by adding a tiny proxy that holds
your Alpaca keys server-side. Your keys are NEVER exposed in the web page.

## 1. Create a Cloudflare account
- Go to https://dash.cloudflare.com/sign-up and sign up (free).

## 2. Create the Worker
1. In the dashboard, go to **Workers & Pages → Create → Create Worker**.
2. Give it a name like `trading-quotes`. Click **Deploy** (it deploys a hello-world).
3. Click **Edit code**. Delete what's there, paste the entire contents of
   `worker.js` from this project, then **Deploy** again.

## 3. Add your Alpaca keys as Worker secrets
1. On the Worker's page: **Settings → Variables and Secrets**.
2. Add two **secrets** (type: Secret, encrypted):
   - `ALPACA_API_KEY` = your Alpaca key
   - `ALPACA_SECRET_KEY` = your Alpaca secret
3. **Deploy** so the secrets take effect.

## 4. Get the Worker URL & test it
- The URL looks like `https://trading-quotes.<your-subdomain>.workers.dev`.
- Test in a browser: open `https://.../?symbols=AAPL,MSFT` — you should see
  `{"prices":{"AAPL":...,"MSFT":...},"at":"..."}`.

## 5. Tell the dashboard about it
Add the URL in two places (no quotes, no trailing slash):

**a) Locally** — add to your `.env`:
```
LIVE_QUOTES_URL=https://trading-quotes.<your-subdomain>.workers.dev
```

**b) On GitHub** — repo **Settings → Secrets and variables → Actions → New
repository secret**: name `LIVE_QUOTES_URL`, value the same URL.

Then push (the workflow already passes `LIVE_QUOTES_URL` to the build). After the
next deploy, the page fetches live prices on load and refreshes every 30 seconds,
showing a green "● Live" indicator with the last-updated time.

## Notes
- Free Cloudflare Workers allow 100,000 requests/day — far more than this needs.
- Prices use Alpaca's free IEX feed (real-time IEX trades). It's a price refresh
  only — the buy/sell **signals** still come from daily bars and don't change
  intraday.
- If `LIVE_QUOTES_URL` is not set, the dashboard simply shows prices as of the
  last build (nothing breaks).
