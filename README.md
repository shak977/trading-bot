# Simple US-Equities Trading Bot (Alpaca)

A clean, dependency-light starting point: indicator-based strategy, an
event-driven backtester with risk management, and a paper-trading runner.
**Defaults to paper/dry-run.** It never places live orders unless you supply
live keys and explicitly set `ALPACA_PAPER=false`.

## What's inside

| File            | Role                                                            |
|-----------------|-----------------------------------------------------------------|
| `config.py`     | All settings + secrets from env. Paper trading on by default.   |
| `indicators.py` | SMA, RSI, ATR, MACD, Bollinger, ADX, rolling highs/lows.        |
| `strategy.py`   | MA-crossover entry, gated by an RSI filter. Emits 0/1 signals.  |
| `analytics.py`  | Chart-pattern detection + per-stock historical edge (backtest). |
| `data.py`       | Real Alpaca bars **or** deterministic synthetic bars (no keys). |
| `market.py`     | Alpaca REST helpers: most-active, movers, news, assets.         |
| `scanner.py`    | Dynamic scan + multi-factor conviction + reasoning/desk read.   |
| `tracker.py`    | Logs every BUY call and grades it vs real prices over time.     |
| `risk.py`       | Risk-based position sizing, stop-loss, take-profit.             |
| `backtest.py`   | Bar-by-bar backtester → equity curve + metrics.                |
| `broker.py`     | Alpaca order wrapper (paper by default).                        |
| `llm.py`        | Optional Anthropic analyst note (desk-trader style).            |
| `dashboard.py`  | Builds `dashboard.html` + `signals.json` (scan/analytics/UI).   |
| `selftest.py`   | `python3 selftest.py` — checks indicators/strategy/analytics.   |
| `run.py`        | CLI: `backtest` and `trade`.                                    |

## What the analysis does (overnight upgrade)

Beyond the moving-average cross, each stock now gets a **multi-factor read** like a
desk trader would: trend + RSI/MACD momentum + ADX trend-strength + volume + Bollinger
position + distance from highs/lows + risk:reward, **plus a per-stock historical edge**
(the strategy is backtested on that stock's own history to see how often it has worked).
The detail panel flags **chart patterns** (golden/death cross, breakouts, pullbacks,
oversold bounces, MACD crosses) and the dashboard shows a **market-regime banner**
(breadth/Risk-on–off) and **sector strength** ranking.

## Dashboard & weekly signals

`python dashboard.py` scans the market (live: Alpaca movers + most-active;
no keys: a synthetic demo list), runs the strategy on each name, ranks them,
attaches recent news for flagged tickers, and writes a self-contained
`dashboard.html` you can open in any browser. See **DEPLOY.md** to run it
weekly and publish it online (GitHub Actions + Pages).

"Rel vol" on the dashboard is today's volume vs its 20-day average — a free
proxy for unusual activity, **not** real institutional/options order flow
(that's paid data; can be added later).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Alpaca PAPER keys
```

Get free paper keys at https://app.alpaca.markets/paper/dashboard/overview

## Usage

```bash
# Backtest on synthetic data — no keys required
python run.py backtest

# Backtest on real Alpaca history
python run.py backtest --live-data

# See intended orders without placing any
python run.py trade --dry-run

# Place orders on your PAPER account
python run.py trade
```

`trade` runs a single pass over `CONFIG.symbols`. Schedule it on your bar
interval (cron, e.g. `30 13 * * 1-5` for a daily check just after the US open)
rather than running a long-lived loop.

## The strategy

Long-only. Enter when the fast SMA (20) crosses above the slow SMA (50) and RSI
is below the overbought line (70). Exit on the reverse crossover, on RSI hitting
overbought, or when the stop-loss / take-profit triggers intrabar. Tune any of
this in `config.py`.

## Risk management

Each position is sized so that hitting the stop-loss costs about
`risk_per_trade` (default 2%) of equity. A 5% stop and 15% target are attached
at entry. Adjust `risk_per_trade`, `stop_loss_pct`, `take_profit_pct`,
`max_positions` in `config.py`.

## Important caveats

- This is an educational scaffold, **not** financial advice or a profitable
  system. Backtest results on synthetic data are meaningless for real returns.
- Backtests here don't model commissions, slippage, or partial fills. Validate
  on real history and paper-trade for a long time before risking real money.
- Keep `ALPACA_PAPER=true` until you fully understand the code and the risk.
