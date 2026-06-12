# Research feeds — free API keys (optional)

The dashboard now blends in real research. **News sentiment works with no key.**
Two extra feeds light up with free keys — add them the same safe way as your other
secrets (GitHub repo → Settings → Secrets and variables → Actions). If a key is
absent, that section is simply hidden; nothing breaks.

## 1. Finnhub — analyst ratings, price targets, fundamentals
- Sign up free at **https://finnhub.io** → copy your API key from the dashboard.
- Add a GitHub Actions secret: `FINNHUB_API_KEY` = your key.
- Unlocks per stock: analyst Buy/Hold/Sell consensus, average price target (and
  upside vs current price), P/E, market cap — and these factor into conviction.
- Free tier is ~60 calls/min, so the bot fetches research for the **top 25 shown**
  names per run (tune `research_top` in `config.py`).

## 2. FRED — macro data (US Federal Reserve)
- Sign up free at **https://fredaccount.stlouisfed.org** → **My Account → API Keys
  → Request API Key**.
- Add a GitHub Actions secret: `FRED_API_KEY` = your key.
- Unlocks the **Macro backdrop** panel: 10-yr & 2-yr Treasury yields + the yield
  curve, inflation (CPI YoY), unemployment, fed funds rate — plus a plain-English
  read (Supportive / Mixed / Cautious).

## After adding the secrets
Re-run the workflow (Actions → Weekly Trading Signals → Run workflow), or just wait
for the next daily run. Locally, add the same keys to your `.env` to test.

## Honest limits
- News sentiment is a **headline-tone** estimate, not a deep read.
- Finnhub gives the analyst **consensus, targets and fundamentals** — not the full
  paywalled report text.
- All of it is **context to inform your own judgement**, not financial advice.
