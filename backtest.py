"""Single-symbol event-driven backtester.

Walks bar by bar:
  - enters long on signal==1 (sized by risk.py), sets stop & target
  - exits on signal->0, stop-loss, or take-profit (intrabar via high/low)
Produces an equity curve and summary metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config
from risk import position_size, stop_loss_price, take_profit_price
from strategy import generate_signals


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Total return : {m['total_return']:>8.2%}\n"
            f"CAGR         : {m['cagr']:>8.2%}\n"
            f"Max drawdown : {m['max_drawdown']:>8.2%}\n"
            f"Sharpe       : {m['sharpe']:>8.2f}\n"
            f"Trades       : {m['n_trades']:>8d}\n"
            f"Win rate     : {m['win_rate']:>8.2%}\n"
            f"Final equity : ${m['final_equity']:>,.2f}"
        )


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    sig = generate_signals(df, cfg)
    cash = cfg.starting_cash
    shares = 0
    entry = stop = target = 0.0
    equity_hist, trades = [], []

    for ts, row in sig.iterrows():
        price = row["close"]
        signal = row["signal"]

        # --- manage open position ---
        if shares > 0:
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
            exit_signal = signal == 0
            if hit_stop or hit_target or exit_signal:
                exit_px = stop if hit_stop else target if hit_target else price
                cash += shares * exit_px
                trades.append(
                    {"exit_time": ts, "exit_px": exit_px, "shares": shares,
                     "pnl": shares * (exit_px - entry),
                     "reason": "stop" if hit_stop else "target" if hit_target else "signal"}
                )
                shares = 0

        # --- open new position ---
        if shares == 0 and signal == 1 and not np.isnan(price):
            qty = position_size(cash, price, cfg)
            if qty > 0:
                shares = qty
                entry = price
                stop = stop_loss_price(entry, cfg)
                target = take_profit_price(entry, cfg)
                cash -= shares * price
                trades.append({"entry_time": ts, "entry_px": entry, "shares": shares})

        equity_hist.append(cash + shares * price)

    equity = pd.Series(equity_hist, index=sig.index, name="equity")
    return BacktestResult(equity, pd.DataFrame(trades), _metrics(equity, trades, cfg))


def _metrics(equity: pd.Series, trades: list, cfg: Config) -> dict:
    eq = equity.dropna()
    if eq.empty:
        return {k: 0 for k in
                ["total_return", "cagr", "max_drawdown", "sharpe",
                 "n_trades", "win_rate", "final_equity"]}
    rets = eq.pct_change().fillna(0.0)
    total_return = eq.iloc[-1] / cfg.starting_cash - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / cfg.starting_cash) ** (1 / years) - 1
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    closed = [t for t in trades if "pnl" in t]
    wins = [t for t in closed if t["pnl"] > 0]
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "n_trades": len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "final_equity": float(eq.iloc[-1]),
    }
