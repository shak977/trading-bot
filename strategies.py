"""A library of well-known, independent trading strategies.

Each strategy is expressed as a **position series** in {0, 1} (long / flat),
forward-filled with the same state-machine convention as
``strategy.generate_signals`` — so ``backtest.backtest_positions`` can grade any
of them with identical risk handling (ATR/percent stop, target, sizing).

Two uses:
  1. **Confluence** — how many independent strategies are long on a stock *right
     now*. Multiple unrelated methods agreeing is a stronger tell than one.
  2. **Per-strategy edge** — backtest each on the stock's own history to show
     which approach has actually worked there (see ``analytics.strategy_edges``).

Everything is derived from price data we already have — no extra API calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from strategy import generate_signals


def _state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Turn boolean entry/exit triggers into a held 0/1 position.

    Enter on an entry trigger, flatten on an exit trigger, hold otherwise.
    Entry wins ties (matches strategy.generate_signals).
    """
    entry = entry.fillna(False).astype(bool)
    exit_ = exit_.fillna(False).astype(bool)
    pos, out = 0.0, []
    for e, x in zip(entry.to_numpy(), exit_.to_numpy()):
        if e:
            pos = 1.0
        elif x:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=entry.index, dtype="float64")


# ---- individual strategies: each returns a 0/1 position Series ----

def trend_cross(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """The house strategy: fast/slow SMA crossover gated by an RSI filter."""
    return generate_signals(df, cfg)["signal"].astype("float64")


def golden_cross(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Classic 50/200 SMA regime — slow and positional."""
    c = df["close"]
    f, s = ind.sma(c, 50), ind.sma(c, 200)
    entry = (f > s) & (f.shift(1) <= s.shift(1))
    exit_ = (f < s) & (f.shift(1) >= s.shift(1))
    return _state(entry, exit_)


def donchian_breakout(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Turtle-style: buy a new 20-day high, exit on a 10-day low."""
    prior_high = df["high"].shift(1).rolling(20, min_periods=20).max()
    prior_low = df["low"].shift(1).rolling(10, min_periods=10).min()
    entry = df["close"] > prior_high
    exit_ = df["close"] < prior_low
    return _state(entry, exit_)


def macd_trend(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """MACD line crossing up through its signal while above zero (momentum)."""
    line, sigl, _ = ind.macd(df["close"])
    entry = (line > sigl) & (line.shift(1) <= sigl.shift(1)) & (line > 0)
    exit_ = line < sigl
    return _state(entry, exit_)


def rsi2_meanrev(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Connors RSI(2): buy a sharp dip *within* an uptrend, exit on the bounce."""
    c = df["close"]
    trend = c > ind.sma(c, 200)
    r2 = ind.rsi(c, 2)
    entry = trend & (r2 < 10)
    exit_ = c > ind.sma(c, 5)
    return _state(entry, exit_)


def bollinger_breakout(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Volatility breakout: close pops above the upper band after a squeeze."""
    c = df["close"]
    mid, up, lo, _ = ind.bollinger(c, 20, 2.0)
    bandwidth = (up - lo) / mid.replace(0.0, np.nan)
    # a "squeeze" = bandwidth in the bottom quartile of the last ~6 months
    floor = bandwidth.rolling(120, min_periods=30).quantile(0.25)
    squeezed = (bandwidth <= floor)
    recent_squeeze = squeezed.rolling(10, min_periods=1).max().astype(bool)
    entry = (c > up) & (c.shift(1) <= up.shift(1)) & recent_squeeze.shift(1, fill_value=False)
    exit_ = c < mid
    return _state(entry, exit_)


def ema_stack(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Momentum stack: 8 > 21 > 50 EMAs aligned up; exit when 8 loses 21."""
    c = df["close"]
    e8, e21, e50 = ind.ema(c, 8), ind.ema(c, 21), ind.ema(c, 50)
    stacked = (e8 > e21) & (e21 > e50)
    entry = stacked & (~stacked.shift(1, fill_value=False))
    exit_ = e8 < e21
    return _state(entry, exit_)


# registry: key -> (label, fn, kind, blurb)
STRATEGIES: dict[str, tuple] = {
    "trend_cross": ("Trend crossover", trend_cross, "trend",
                    "20/50-day moving-average crossover with an RSI filter."),
    "golden_cross": ("Golden cross", golden_cross, "trend",
                     "50-day average above the 200-day — a long, positional regime."),
    "donchian": ("Donchian breakout", donchian_breakout, "breakout",
                 "Buys a fresh 20-day high; exits on a 10-day low (turtle-style)."),
    "macd_trend": ("MACD momentum", macd_trend, "momentum",
                   "MACD crossing up through its signal line while above zero."),
    "rsi2_meanrev": ("Dip buy (RSI-2)", rsi2_meanrev, "mean-reversion",
                     "Buys a sharp oversold dip inside an uptrend (Connors RSI-2)."),
    "bollinger": ("Squeeze breakout", bollinger_breakout, "breakout",
                  "Breaks above the Bollinger band after a low-volatility squeeze."),
    "ema_stack": ("EMA momentum stack", ema_stack, "momentum",
                  "Fast EMAs stacked above slow ones (8 > 21 > 50)."),
}


def positions(df: pd.DataFrame, cfg: Config, key: str) -> pd.Series:
    return STRATEGIES[key][1](df, cfg)


def _bars_since_entry(pos: pd.Series) -> int | None:
    """How many bars since the position last flipped from 0 -> 1 (None if flat)."""
    arr = pos.to_numpy()
    if len(arr) == 0 or arr[-1] != 1.0:
        return None
    n = 0
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == 1.0:
            n += 1
        else:
            break
    return n


def evaluate(df: pd.DataFrame, cfg: Config) -> dict:
    """Run every strategy and report which are LONG right now (confluence).

    Returns {results, long, fresh, count, total} where ``fresh`` are strategies
    that turned long within the last ``buy_window`` bars (a new setup).
    """
    results: dict[str, dict] = {}
    long_labels, fresh_labels = [], []
    for key, (label, fn, kind, _blurb) in STRATEGIES.items():
        try:
            pos = fn(df, cfg)
            is_long = bool(len(pos)) and bool(pos.iloc[-1] == 1.0)
            since = _bars_since_entry(pos)
            is_fresh = bool(is_long and since is not None and since <= cfg.buy_window)
        except Exception:  # noqa: BLE001
            is_long = is_fresh = False
            since = None
        results[key] = {"label": label, "kind": kind, "long": is_long,
                        "fresh": is_fresh, "bars_held": since}
        if is_long:
            long_labels.append(label)
        if is_fresh:
            fresh_labels.append(label)
    return {"results": results, "long": long_labels, "fresh": fresh_labels,
            "count": len(long_labels), "total": len(STRATEGIES)}
