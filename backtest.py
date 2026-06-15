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
from risk import honest_target, position_size, stop_loss_price
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
    """Backtest the house strategy (SMA crossover + RSI filter)."""
    sig = generate_signals(df, cfg)
    return backtest_positions(sig, sig["signal"], cfg)


def backtest_positions(df: pd.DataFrame, positions: pd.Series, cfg: Config,
                       side: str = "long") -> BacktestResult:
    """Backtest an arbitrary 0/1 position series with the same risk handling.

    ``df`` must carry high/low/close columns aligned to ``positions``. This lets
    every strategy in ``strategies.py`` be graded identically. ``side='short'``
    grades the series as short positions (1 = short on): stop above entry, target
    below, profit when price falls — the mirror of the long path.
    """
    if side == "short":
        return _backtest_short(df, positions, cfg)
    pos = positions.reindex(df.index).fillna(0.0)
    cash = cfg.starting_cash
    shares = 0
    entry = stop = target = 0.0
    held_bars = 0
    took_partial = False
    equity_hist, trades = [], []
    # optional trailing ATR stop ("let winners run")
    from indicators import atr as _atr
    atr_s = _atr(df, cfg.atr_period)                   # needed for both trailing stop & honest target
    trail = cfg.trail_atr_mult > 0
    slip = getattr(cfg, "slippage_bps", 0.0) / 1e4     # cost per fill (each side)
    comm = getattr(cfg, "commission_per_trade", 0.0)

    def _atr_at(ts):
        a = atr_s.loc[ts]
        if isinstance(a, pd.Series):
            a = a.iloc[-1]
        return float(a) if (a is not None and not np.isnan(a)) else None

    for ts, row in df.iterrows():
        price = row["close"]
        signal = pos.loc[ts]
        if isinstance(signal, pd.Series):  # guard against duplicate index labels
            signal = signal.iloc[-1]

        # --- manage open position ---
        if shares > 0:
            # ratchet the stop up under a rising price (never down)
            if trail:
                a = atr_s.loc[ts]
                if isinstance(a, pd.Series):
                    a = a.iloc[-1]
                if a is not None and not np.isnan(a):
                    stop = max(stop, float(price) - cfg.trail_atr_mult * float(a))
            held_bars += 1
            reached_1r = row["high"] >= entry + (entry - stop)
            time_stop = cfg.max_hold_days > 0 and held_bars >= cfg.max_hold_days and not reached_1r
            # partial profit-take at partial_take_r (once); move the remainder to breakeven
            if (cfg.partial_take_r > 0 and not took_partial and shares >= 2
                    and row["high"] >= entry + cfg.partial_take_r * (entry - stop)):
                half = shares // 2
                px = (entry + cfg.partial_take_r * (entry - stop)) * (1 - slip)
                cash += half * px - comm
                trades.append({"exit_time": ts, "exit_px": px, "shares": half,
                               "pnl": half * (px - entry), "reason": "partial"})
                shares -= half
                took_partial = True
                stop = max(stop, entry)
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
            exit_signal = signal == 0
            if hit_stop or hit_target or exit_signal or time_stop:
                exit_px = stop if hit_stop else target if hit_target else price
                fill = exit_px * (1 - slip)              # sell into slippage
                cash += shares * fill - comm
                trades.append(
                    {"exit_time": ts, "exit_px": fill, "shares": shares,
                     "pnl": shares * (fill - entry),
                     "reason": "stop" if hit_stop else "target" if hit_target else ("time" if time_stop else "signal")}
                )
                shares = 0

        # --- open new position ---
        if shares == 0 and signal == 1 and not np.isnan(price):
            qty = position_size(cash, price, cfg)
            if qty > 0:
                shares = qty
                entry = price * (1 + slip)              # buy fill incl. slippage
                stop = stop_loss_price(price, cfg)
                target, _ = honest_target(price, df.loc[:ts], _atr_at(ts), stop, cfg, short=False)
                held_bars = 0
                took_partial = False
                cash -= shares * entry + comm
                trades.append({"entry_time": ts, "entry_px": entry, "shares": shares})

        equity_hist.append(cash + shares * price)

    equity = pd.Series(equity_hist, index=df.index, name="equity")
    return BacktestResult(equity, pd.DataFrame(trades), _metrics(equity, trades, cfg))


def _backtest_short(df: pd.DataFrame, positions: pd.Series, cfg: Config) -> BacktestResult:
    """Short mirror of ``backtest_positions``: 1 = short on. Sell to open (into slippage),
    cover on signal->0, stop (above entry) or target (below). Profit = entry - exit."""
    pos = positions.reindex(df.index).fillna(0.0)
    cash = cfg.starting_cash
    shares = 0
    entry = stop = target = 0.0
    held_bars = 0
    took_partial = False
    equity_hist, trades = [], []
    from indicators import atr as _atr
    atr_s = _atr(df, cfg.atr_period)                   # needed for both trailing stop & honest target
    trail = cfg.trail_atr_mult > 0
    slip = getattr(cfg, "slippage_bps", 0.0) / 1e4
    comm = getattr(cfg, "commission_per_trade", 0.0)

    def _atr_at(ts):
        a = atr_s.loc[ts]
        if isinstance(a, pd.Series):
            a = a.iloc[-1]
        return float(a) if (a is not None and not np.isnan(a)) else None

    for ts, row in df.iterrows():
        price = row["close"]
        signal = pos.loc[ts]
        if isinstance(signal, pd.Series):
            signal = signal.iloc[-1]

        # --- manage open short ---
        if shares > 0:
            # ratchet the stop DOWN over a falling price (never up)
            if trail:
                a = atr_s.loc[ts]
                if isinstance(a, pd.Series):
                    a = a.iloc[-1]
                if a is not None and not np.isnan(a):
                    stop = min(stop, float(price) + cfg.trail_atr_mult * float(a))
            held_bars += 1
            reached_1r = row["low"] <= entry - (stop - entry)
            time_stop = cfg.max_hold_days > 0 and held_bars >= cfg.max_hold_days and not reached_1r
            # partial cover at partial_take_r (once); move the remainder to breakeven
            if (cfg.partial_take_r > 0 and not took_partial and shares >= 2
                    and row["low"] <= entry - cfg.partial_take_r * (stop - entry)):
                half = shares // 2
                px = (entry - cfg.partial_take_r * (stop - entry)) * (1 + slip)
                cash += half * (entry - px) - comm
                trades.append({"exit_time": ts, "exit_px": px, "shares": half,
                               "pnl": half * (entry - px), "reason": "partial"})
                shares -= half
                took_partial = True
                stop = min(stop, entry)
            hit_stop = row["high"] >= stop          # price rose into the stop
            hit_target = row["low"] <= target        # price fell to the cover target
            exit_signal = signal == 0
            if hit_stop or hit_target or exit_signal or time_stop:
                exit_px = stop if hit_stop else target if hit_target else price
                fill = exit_px * (1 + slip)           # buy to cover into slippage
                cash += shares * (entry - fill) - comm
                trades.append(
                    {"exit_time": ts, "exit_px": fill, "shares": shares,
                     "pnl": shares * (entry - fill),
                     "reason": "stop" if hit_stop else "target" if hit_target else ("time" if time_stop else "signal")}
                )
                shares = 0

        # --- open new short ---
        if shares == 0 and signal == 1 and not np.isnan(price):
            qty = position_size(cash, price, cfg)
            if qty > 0:
                shares = qty
                entry = price * (1 - slip)            # sell to open incl. slippage
                stop = entry * (1 + cfg.stop_loss_pct)
                target, _ = honest_target(entry, df.loc[:ts], _atr_at(ts), stop, cfg, short=True)
                held_bars = 0
                took_partial = False
                cash -= comm
                trades.append({"entry_time": ts, "entry_px": entry, "shares": shares})

        unreal = shares * (entry - price) if shares > 0 else 0.0
        equity_hist.append(cash + unreal)

    equity = pd.Series(equity_hist, index=df.index, name="equity")
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
    cagr = (max(eq.iloc[-1], 0.0) / cfg.starting_cash) ** (1 / years) - 1
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
