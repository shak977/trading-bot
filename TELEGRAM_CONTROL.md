# Telegram bot control — setup

Text commands to the bot from your phone. The always-on Cloudflare Worker receives them, runs the
action, and replies. Locked to YOUR chat id, and verified by a secret so only real Telegram updates
are accepted.

**Commands:** `/status` (latest signals + book health) · `/scan` (fresh scan now) ·
`/analyst` (run the review) · `/test` (test alert) · `/help`.

## One-time setup (~5 min)

### 1. Push the code
`worker.js` now has the Telegram handler. Deploy it (wrangler, or paste into the Cloudflare
dashboard editor for your Worker and Save/Deploy).

### 2. Add three Worker secrets
Cloudflare dashboard → **Workers & Pages** → your worker → **Settings → Variables and Secrets** →
add each as an **encrypted** secret (`GH_TOKEN` is already there from the cron):

| Name | Value |
|------|-------|
| `TELEGRAM_BOT_TOKEN` | your bot token (the long one with a colon) |
| `TELEGRAM_CHAT_ID`   | your chat id (the number from RawDataBot) |
| `TG_WEBHOOK_SECRET`  | any long random string you invent (e.g. a 30-char password) |

Optional: `SITE_URL` (defaults to `https://shak977.github.io/trading-bot`), `GH_REPO`.

### 3. Point Telegram at the Worker
Your Worker's address is the same URL you set as `LIVE_QUOTES_URL` for the dashboard
(e.g. `https://trading-bot.<you>.workers.dev`). Open this in a browser, filling in the three blanks:

```
https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=<YOUR_WORKER_URL>&secret_token=<YOUR_TG_WEBHOOK_SECRET>
```

You should see `{"ok":true,"result":true,"description":"Webhook was set"}`.

### 4. Test
Text your bot `/help` — you should get the command menu back. Then `/status` for a live read.

## Notes
- Only your chat id is answered; anyone else is ignored.
- `/scan`, `/analyst`, `/test` trigger GitHub Actions via the Worker's existing `GH_TOKEN`.
- To turn it off: delete the webhook — `https://api.telegram.org/bot<TOKEN>/deleteWebhook`.
