"""Inspect a stock's daily closes under different adjustments to diagnose
implausible % moves. Usage:  python3 check_prices.py INTC

Prints the close ~60 trading days ago vs now (the overview's 3-month window)
for raw / split / all adjustments, plus the last few closes and any big jumps.
"""
from __future__ import annotations

import sys
import pandas as pd
from config import CONFIG


def fetch(symbol, adjustment):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    client = StockHistoricalDataClient(CONFIG.api_key, CONFIG.secret_key)
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=160)
    end = pd.Timestamp.utcnow() - pd.Timedelta(minutes=16)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                           start=start.to_pydatetime(), end=end.to_pydatetime(),
                           feed=DataFeed.IEX, adjustment=adjustment)
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return df


def main(symbol):
    from alpaca.data.enums import Adjustment
    print(f"=== {symbol} — IEX daily closes ===\n")
    for name, adj in [("raw", Adjustment.RAW), ("split", Adjustment.SPLIT), ("all", Adjustment.ALL)]:
        try:
            df = fetch(symbol, adj)
            c = df["close"]
            n = len(c)
            if n < 61:
                print(f"{name:6}: only {n} bars"); continue
            base, last = float(c.iloc[-60]), float(c.iloc[-1])
            ret = (last / base - 1) * 100
            # biggest one-day move
            chg = c.pct_change().abs()
            j = chg.idxmax()
            print(f"{name:6}: 60d-ago ${base:.2f} ({c.index[-60].date()}) -> now ${last:.2f} "
                  f"= {ret:+.1f}%  | biggest 1-day move {chg.max()*100:.0f}% on {j.date()}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:6}: ERROR {e}")
    # show the last 6 closes (split-adjusted)
    try:
        df = fetch(symbol, Adjustment.SPLIT)
        tail = df["close"].tail(6)
        print("\nlast 6 closes (split-adj):")
        for ts, v in tail.items():
            print(f"  {ts.date()}  ${float(v):.2f}")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "INTC")
