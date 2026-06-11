"""Entry point.

Usage:
  python run.py backtest            # run backtest on synthetic data (no keys)
  python run.py backtest --live-data  # backtest on real Alpaca history (needs keys)
  python run.py trade --dry-run     # compute signals, print intended orders, place nothing
  python run.py trade               # paper-trade for real (needs PAPER keys)

`trade` runs ONE pass over the symbol list. Schedule it (cron / Task Scheduler)
to run on your bar interval — this avoids a long-lived process and is easier to
reason about. It will refuse to run if the market data looks insufficient.
"""
from __future__ import annotations

import argparse
import sys

from config import CONFIG
from data import get_bars, synthetic_bars
from strategy import latest_signal


def cmd_backtest(args) -> None:
    from backtest import run_backtest

    for symbol in CONFIG.symbols:
        if args.live_data:
            df = get_bars(symbol, CONFIG)
        else:
            df = synthetic_bars(symbol, n=CONFIG.lookback_days)
        result = run_backtest(df, CONFIG)
        print(f"\n=== {symbol} ===")
        print(result.summary())


def cmd_trade(args) -> None:
    from broker import Broker
    from risk import position_size

    dry = args.dry_run
    broker = None if dry else Broker(CONFIG)
    equity = CONFIG.starting_cash if dry else broker.equity()
    print(f"[{'DRY-RUN' if dry else ('PAPER' if CONFIG.paper else 'LIVE')}] "
          f"equity=${equity:,.2f}")

    for symbol in CONFIG.symbols:
        df = synthetic_bars(symbol, n=CONFIG.lookback_days) if dry else get_bars(symbol, CONFIG)
        if len(df) < CONFIG.slow_ma + 2:
            print(f"{symbol}: not enough data, skipping")
            continue

        signal = latest_signal(df, CONFIG)
        price = float(df["close"].iloc[-1])
        held = 0 if dry else broker.position_qty(symbol)

        if signal == 1 and held == 0:
            qty = position_size(equity, price, CONFIG)
            if qty <= 0:
                print(f"{symbol}: BUY signal but size=0, skipping")
                continue
            print(f"{symbol}: BUY {qty} @ ~${price:,.2f}")
            if not dry:
                broker.submit_market(symbol, qty, "buy")
        elif signal == 0 and held > 0:
            print(f"{symbol}: SELL {held} @ ~${price:,.2f}")
            if not dry:
                broker.submit_market(symbol, held, "sell")
        else:
            print(f"{symbol}: HOLD (signal={signal}, held={held})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simple US-equities trading bot (Alpaca).")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backtest", help="run the strategy over history")
    b.add_argument("--live-data", action="store_true", help="use real Alpaca bars")
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser("trade", help="run one trading pass")
    t.add_argument("--dry-run", action="store_true", help="print orders, place none")
    t.set_defaults(func=cmd_trade)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
