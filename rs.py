"""Relative strength vs a benchmark — cross-sectional leadership ranking.

Momentum tells you a name is rising; relative strength tells you it's rising
*faster than the market*. Leadership (high RS) is one of the most robust
momentum factors (O'Neil's RS line, AQR cross-sectional momentum). We blend a
few trailing windows, subtract the benchmark's return over each, then
percentile-rank the whole scanned universe so the conviction model can reward
leaders and penalise laggards. Pure functions, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_return(close: pd.Series, lookback: int) -> float | None:
    if close is None or len(close) < lookback + 1:
        return None
    past = close.iloc[-lookback - 1]
    last = close.iloc[-1]
    if not past or np.isnan(past) or np.isnan(last):
        return None
    return float(last / past - 1.0)


def rs_score(close: pd.Series, bench_close: pd.Series,
             lookbacks=(21, 63, 126), weights=(0.2, 0.3, 0.5)) -> float | None:
    """Blended excess return of `close` over `bench_close` across `lookbacks`.

    Returns None if either series lacks enough history for the longest lookback.
    Positive = the name outran the benchmark; negative = it lagged.
    """
    if bench_close is None or len(bench_close) < max(lookbacks) + 1:
        return None
    total, wsum = 0.0, 0.0
    for lb, w in zip(lookbacks, weights):
        r = _trailing_return(close, lb)
        b = _trailing_return(bench_close, lb)
        if r is None or b is None:
            return None
        total += w * (r - b)
        wsum += w
    if wsum <= 0:
        return None
    return total / wsum


def rank_universe(scores: dict) -> dict:
    """Percentile-rank RS scores across the universe (0-100, best = 100).

    Symbols whose score is None/NaN are dropped. Ties share the average percentile.
    Returns {symbol: {"rs": float, "pct": int}}.
    """
    clean = {s: float(v) for s, v in scores.items()
             if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if not clean:
        return {}
    ser = pd.Series(clean)
    pct = ser.rank(pct=True, method="average") * 100.0
    return {s: {"rs": round(clean[s], 4), "pct": int(round(pct[s]))} for s in clean}
