# Running weekly & taking the dashboard online

You have two layers: (1) run the bot weekly to refresh signals, and
(2) optionally publish `dashboard.html` to a URL you can open anywhere.

---

## A. Get real data (required for real signals)

The bot uses **Alpaca** for market data. Free paper keys, no funding needed:

1. Sign up at https://alpaca.markets and open the **Paper** dashboard.
2. Generate an API key + secret.
3. `cp .env.example .env` and paste them in. Keep `ALPACA_PAPER=true`.

Without keys the dashboard still builds, but uses **synthetic** prices (clearly
labelled) — fine for testing the plumbing, useless as real signals.

---

## B. Run it weekly on your own computer (simplest)

**macOS / Linux (cron):**
```bash
chmod +x weekly.sh
crontab -e
# add this line — Mondays at 08:00 local:
0 8 * * 1  /full/path/to/trading_bot/weekly.sh
```
`dashboard.html` is rewritten each run. Double-click it to view; logs land in
`logs/weekly.log`.

**macOS (launchd, survives reboots better):** create
`~/Library/LaunchAgents/com.you.tradingbot.plist` pointing at `weekly.sh` with a
`StartCalendarInterval` of Weekday 1, Hour 8, then `launchctl load` it.

**Windows (Task Scheduler):** Create Task → Trigger: Weekly, Monday 08:00 →
Action: Start a program → `python` with argument `dashboard.py` and "Start in"
set to the project folder.

This keeps everything on your machine. The dashboard is just a local file.

---

## C. Take it fully online (free, automatic, public URL)

Use **GitHub Actions + GitHub Pages** — included as
`.github/workflows/weekly-signals.yml`. It runs weekly in the cloud, builds the
dashboard, and publishes it to a URL.

1. Create a new **private** GitHub repo and push this folder:
   ```bash
   git init && git add . && git commit -m "trading bot"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Repo **Settings → Secrets and variables → Actions** → add
   `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. (Never commit `.env`.)
3. Repo **Settings → Pages** → Source: **GitHub Actions**.
4. **Actions** tab → run *Weekly Trading Signals* once (workflow_dispatch) to
   confirm. After that it runs every Monday automatically.

Your dashboard lives at `https://<you>.github.io/<repo>/`, refreshed weekly.
`signals.json` is also published for programmatic access.

> Pages sites are public even from a private repo. This dashboard shows only
> signals, never your keys (keys stay in Actions secrets). Still, treat the URL
> as semi-public.

**Other hosts:** the same `dashboard.py` output works on Netlify, Cloudflare
Pages, or any static host. For always-on intraday signals instead of weekly,
move the run to a small cloud VM or a scheduled container and keep the cron idea.

---

## D. About actually placing trades

`run.py trade` can submit orders to your Alpaca **paper** account. Going from
signals to automated live orders is a deliberate, high-risk step: test on paper
for a long time, add order/fill error handling, and only switch to live keys
when you fully trust it. This project will not place live orders for you by
default.
