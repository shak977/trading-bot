# Roadmap / future options

Deferred ideas, captured so they're not lost. None are required for the current
dashboard to work — it's a daily-bar, free-data tool by design.

## Data & refresh

- **Intraday signals + real-time feed (biggest upgrade).** Today the engine is built on
  **daily bars** from Alpaca's free **IEX** feed (~15 min delayed), and the Action rebuilds
  every ~30 min during market hours. That means intraday rebuilds mostly refresh the
  *forming* day's price/RSI/levels and the news — the BUY/SHORT calls intentionally don't
  whipsaw every 30 min. To get **true intraday signals** you'd:
    1. switch the scan timeframe to intraday (e.g. 5-min / 15-min bars) in `config.py`,
    2. move to a **paid real-time data feed** (Alpaca SIP, or Polygon/IEX Cloud) so prices
       aren't 15 min delayed,
    3. re-tune the strategies/stops for the shorter timeframe (daily-bar params won't
       transfer), and
    4. run the rebuild far more often (a small always-on worker rather than GitHub cron,
       since cron's practical floor is ~15–30 min and runs can be delayed/skipped).
  This is a meaningfully bigger build and changes the risk profile — treat as a project,
  not a tweak.

## News

- **Google News RSS is blocked from GitHub's datacenter IPs** (returns empty), so it was
  removed. Current live feeds: Yahoo Finance RSS + Finnhub company-news + Benzinga (Alpaca).
  A server-side proxy (e.g. the Cloudflare Worker) could re-enable Google News if desired.

## Strategy validation (honesty work)

- **Reduce momentum survivorship bias** with a point-in-time universe (hard on free data).
- **Walk-forward-tune momentum params** (lookback/skip/top-K) — carefully, to avoid overfit.
- **Run the short-side bake-off on real data** (`edge_hunt.py`) once enough history is cached.
- **Paper-trade live through Alpaca** to measure true forward performance vs. the (rebuilding)
  hypothetical track record.

## UX

- **Mobile tooltips:** the strategy explanations are hover-based (desktop). Convert to a
  tap-to-reveal popover for phones.
- **Externalize the big inline `<script>`** in `dashboard.py` to a served `.js` file to
  shrink `dashboard.html` and ease editing.
