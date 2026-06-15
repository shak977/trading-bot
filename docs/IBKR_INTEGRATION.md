# IBKR integration — setup & architecture

Goal: enrich the dashboard's signals with Interactive Brokers data — intraday/tick bars,
options (IV + greeks), your real account positions, and futures/FX/global instruments —
without breaking the existing Alpaca-powered cloud build.

This is opt-in. Nothing here runs until you stand up a gateway and set `IBKR_ENABLED=true`.
The bot **never logs in, places trades, or sees your password** — it only reads data from a
gateway *you* authenticate.

---

## 1. Why IBKR needs extra plumbing (vs Alpaca)

Alpaca gives you a keyless REST API — perfect for a stateless GitHub Action. IBKR does **not**.
Every IBKR data path needs a **persistent, authenticated gateway session**:

- **TWS / IB Gateway + `ib_async`** — desktop app or Gateway must be running and logged in.
- **Client Portal (CP) Web API** — a small gateway process you run, authenticated via browser
  login + 2FA. Sessions expire and must be kept alive.

A GitHub Action is ephemeral (no persistent process) and can't safely hold IBKR credentials or
pass 2FA. So the gateway lives **outside** the build, and the build (or a local job) queries it.

We use the **Client Portal Web API**, kept alive headlessly by **IBeam** on an always-on VPS.

---

## 2. Target architecture (always-on VPS)

```
  IBKR servers
       ▲  (authenticated session, kept alive)
       │
  ┌────┴───────────────────────┐
  │  VPS (always on)           │
  │   Docker: IBeam            │  ← auto-logs in, refreshes session, exposes CP Web API
  │   → CP Web API :5000       │     (https, behind a reverse proxy + token)
  └────┬───────────────────────┘
       │  HTTPS + bearer token (your secret)
       ▼
  GitHub Action build  ── ibkr.py ──►  scanner / strategies / dashboard
  (every 30 min)                        (Alpaca stays the base; IBKR enriches)
```

Key idea: **IBeam keeps the session authenticated 24/7** so the stateless build can just make
HTTPS calls to your gateway, the same way it calls Alpaca.

---

## 3. What you need to provision (your side)

1. **IBKR account** with API access enabled, and **market-data subscriptions** for what you want:
   - US equities real-time (NASDAQ/NYSE) — small monthly fee (waived above some commissions).
   - **OPRA** for options (IV/greeks).
   - CME/ICE etc. for **futures**; FX is generally included.
   - International exchanges as needed. (Delayed data is cheaper/free but lags ~15 min.)
2. **A small VPS** (1 vCPU / 2 GB is enough): e.g. a $5–6/mo cloud box. Docker installed.
3. **IBeam** running in Docker on the VPS (auto-login). It needs your IBKR username/password as
   env vars **on the VPS** and your phone for the IBKEY/2FA approval on (re)login.
4. **A reverse proxy (Caddy/nginx) with TLS + a bearer token** in front of IBeam so only your
   build can reach it. Never expose the raw gateway to the internet.

> Security: your IBKR credentials live only on your VPS (for IBeam). The GitHub build only ever
> holds the **gateway URL + bearer token** (GitHub secrets), never your IBKR password.

### IBeam quick start (on the VPS)

```bash
# docker-compose.yml (sketch — see Voyz/ibeam docs for current options)
services:
  ibeam:
    image: voyz/ibeam
    environment:
      IBEAM_ACCOUNT: ${IBKR_USERNAME}
      IBEAM_PASSWORD: ${IBKR_PASSWORD}
      IBEAM_GATEWAY_BASE_URL: https://localhost:5000
    ports: ["5000:5000"]
    restart: unless-stopped
```

Then put Caddy/nginx in front with `Authorization: Bearer <token>` enforcement + Let's Encrypt TLS.

---

## 4. What the bot needs (my side — built, off by default)

- `ibkr.py` — a thin client over the CP Web API using the maintained **`ibind`** library, with:
  - `available()` / `diagnose()` — connectivity + auth self-test (feeds the System tab).
  - `intraday_bars(symbol, bar, lookback)` — finer-grained price history.
  - `option_summary(symbol)` — ATM IV, IV-rank, basic greeks from the chain.
  - `positions()` — your real account holdings (portfolio-aware signals).
  - `find_contract(symbol, sec_type)` — resolve futures/FX/global contracts (conid).
  - Every call **fails soft** (returns `None`/`[]`, never raises) so a gateway hiccup can't break
    the build.
- `config.py` flags: `ibkr_enabled`, `ibkr_gateway_url`, `ibkr_account_id`, `ibkr_timeout`.
- `requirements`: `ibind` added as an **optional** extra (not installed unless you enable IBKR).

Wiring into `scanner.py` / `strategies.py` / the dashboard happens **after** your gateway is live
and we've validated the real endpoints together — so we don't destabilize the working build.

---

## 5. Phased rollout (recommended order)

1. **Stand up gateway + connectivity.** VPS + IBeam + proxy. Set `IBKR_GATEWAY_URL` + token as
   GitHub secrets. Confirm `ibkr.diagnose()` is green in the System tab. *(No signals change yet.)*
2. **Real positions** → portfolio-aware tweaks: flag over-concentration vs your actual book, mark
   "you already hold this," surface exits. Lowest risk, high value.
3. **Intraday bars** → an intraday momentum/confirmation layer feeding conviction (kept gated so
   daily-bar signals stay the backbone).
4. **Options + IV/greeks** → IV-rank + unusual-options-activity as conviction inputs; an options
   view on the card modal.
5. **Futures / FX / global** → extend the universe via contract resolution; new tabs/sections.

Each phase is a small, reversible change behind the same feature flag.

---

## 6. Costs & caveats

- **Market data isn't free** — budget for the per-exchange subscriptions above.
- **One session per login** — IBeam holds the CP session; running TWS with the same login
  elsewhere can bump it. Use a dedicated login or second username for the gateway.
- **Re-auth** — IBKR forces periodic re-login (often daily, and a weekly hard reset); IBeam
  automates most of it but you may approve 2FA on your phone.
- **Latency/SLA** — your VPS is now in the critical path for the IBKR-enriched parts; the build
  degrades gracefully to Alpaca-only if the gateway is down.

---

## 7. Next action

Stand up the VPS + IBeam (Phase 1) and give me the **gateway URL + bearer token** (as GitHub
secrets, not pasted in chat). Then we turn on `IBKR_ENABLED`, confirm the System-tab health check,
and start Phase 2 (real positions).

Refs: ib_async (github.com/ib-api-reloaded/ib_async), IBeam (github.com/Voyz/ibeam),
`ibind` (Python client for the IBKR Web API).
