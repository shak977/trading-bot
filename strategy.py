"""Signal generation: moving-average crossover gated by an RSI filter.

Returns a DataFrame with the input plus columns:
  fast, slow, rsi, signal  (signal in {-1, 0, 1})

Rule:
  - Go long (1) when fast MA crosses above slow MA AND RSI is not overbought.
  - Exit / flat (0) when fast MA crosses below slow MA OR RSI is overbought.
  - We hold the position between a buy and the next exit (forward-filled).
"""
from __future__ import annotations

import pandas as pd

from config import Config
from indicators import rsi, sma


def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["fast"] = sma(close, cfg.fast_ma)
    out["slow"] = sma(close, cfg.slow_ma)
    out["rsi"] = rsi(close, cfg.rsi_period)

    cross_up = (out["fast"] > out["slow"]) & (out["fast"].shift(1) <= out["slow"].shift(1))
    cross_down = (out["fast"] < out["slow"]) & (out["fast"].shift(1) >= out["slow"].shift(1))

    raw = pd.Series(0, index=out.index, dtype="float64")
    raw[cross_up & (out["rsi"] < cfg.rsi_overbought)] = 1.0
    raw[cross_down | (out["rsi"] >= cfg.rsi_overbought)] = 0.0

    # Position state machine: enter on 1, exit on explicit 0, hold otherwise.
    position = 0.0
    states = []
    for val, is_up, is_exit in zip(
        raw, cross_up & (out["rsi"] < cfg.rsi_overbought),
        cross_down | (out["rsi"] >= cfg.rsi_overbought),
    ):
        if is_up:
            position = 1.0
        elif is_exit:
            position = 0.0
        states.append(position)

    out["signal"] = states
    return out


def latest_signal(df: pd.DataFrame, cfg: Config) -> int:
    """Convenience for the live runner: signal for the most recent bar."""
    sig = generate_signals(df, cfg)
    if sig.empty:
        return 0
    return int(sig["signal"].iloc[-1])
