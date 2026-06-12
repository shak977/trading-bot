"""Dual momentum — the most robustly-documented equity edge, distilled.

Sources: Jegadeesh & Titman (1993, cross-sectional momentum); Moskowitz, Ooi &
Pedersen (2012, time-series momentum); Antonacci (dual momentum). What AQR / DFA /
Alpha Architect actually run, in spirit:

  1. Cross-sectional: rank the universe by 12-1 momentum (total return over the
     last ~12 months, skipping the most recent ~1 month to dodge short-term
     reversal). Winners tend to keep winning.
  2. Absolute (trend) filter: only hold a name if it's ALSO in its own uptrend —
     positive 12-1 momentum AND above its 200-day average. This is the overlay
     that sidesteps the worst of bear markets / momentum crashes.

Modest and real (~0.3 information ratio long-only, per AQR), not magic — and it
crashes hard in sharp reversals. Use with diversification and risk sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_score(close: pd.Series, lookback: int = 252, skip: int = 21) -> float:
    """12-1 momentum: total return from ~12 months ago to ~1 month ago."""
    if close is None or len(close) < lookback + 1:
        return float("nan")
    past = close.iloc[-lookback]
    recent = close.iloc[-skip - 1]
    if not past or np.isnan(past) or np.isnan(recent):
        return float("nan")
    return recent / past - 1.0


def above_trend(close: pd.Series, win: int = 200) -> bool:
    if close is None or len(close) < win:
        return False
    return bool(close.iloc[-1] > close.iloc[-win:].mean())


def has_bad_bar(close: pd.Series, jump: float = 0.50) -> bool:
    """True if the series contains a spike-and-revert bad print: a >`jump` single-day move
    undone the next day. Isolates corrupt data bars (which wreck the momentum base) without
    flagging real trends or splits, which don't immediately reverse."""
    if close is None or len(close) < 3:
        return False
    r = close.pct_change()
    rev = (r.shift(-1) * r < 0) & (r.abs() > jump) & (r.shift(-1).abs() >= jump * 0.6)
    return bool(rev.fillna(False).any())


def rank(bars: dict, lookback: int = 252, skip: int = 21,
         max_score_pct: float = 200.0, bad_bar_jump: float = 0.50) -> list[tuple]:
    """Return [(symbol, momentum%)] for names passing the dual-momentum filter,
    ranked best-first. Filter = positive 12-1 momentum AND above the 200-day MA.

    Data-quality guards: drop names with a spike-and-revert bad bar in their history, and
    cap the score at `max_score_pct`% — a 12-1 return above that on a large/mid-cap almost
    always means a corrupt price inflated the base (e.g. the MU +591% IEX artifact), not edge."""
    out = []
    for sym, df in bars.items():
        c = df["close"]
        if has_bad_bar(c, bad_bar_jump):
            continue
        s = momentum_score(c, lookback, skip)
        if np.isnan(s):
            continue
        if s > 0 and s * 100 <= max_score_pct and above_trend(c):
            out.append((sym, round(s * 100, 1)))
    out.sort(key=lambda x: -x[1])
    return out
