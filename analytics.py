"""Trader-style analytics: chart-pattern detection and the per-stock historical
edge of the strategy (via the backtester). All derived from price data we
already have — no extra API calls.

Everything here is descriptive/historical, not predictive. It exists to help a
human reason about a setup the way a desk trader would: confluence of signals,
trend strength, where price sits, and how the rules have actually performed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as ind
from backtest import run_backtest
from config import Config


def backtest_edge(df: pd.DataFrame, cfg: Config) -> dict | None:
    """How this exact strategy has historically performed on THIS stock."""
    try:
        res = run_backtest(df, cfg)
        m = res.metrics
        return {
            "win_rate": round(m["win_rate"] * 100, 1) if m["n_trades"] else None,
            "n_trades": int(m["n_trades"]),
            "total_return": round(m["total_return"] * 100, 1),
            "max_drawdown": round(m["max_drawdown"] * 100, 1),
            "sharpe": round(m["sharpe"], 2),
        }
    except Exception:  # noqa: BLE001
        return None


def _f(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def detect(df: pd.DataFrame, sig: pd.DataFrame, cfg: Config, relvol):
    """Return (patterns, extra_factors).

    patterns: list of {label, kind in 'bull'/'bear'/'neutral'}
    extra_factors: {adx, macd_hist, bb_pct} numeric snapshot for the UI/scoring.
    """
    pats: list[dict] = []
    close = df["close"]
    n = len(close)
    last = sig.iloc[-1]
    fast, slow, rsi_v = _f(last["fast"]), _f(last["slow"]), _f(last["rsi"])
    px = _f(close.iloc[-1])

    # 1) Moving-average crossover recency
    rel = (sig["fast"] > sig["slow"]).astype(int)
    flips = rel.diff().fillna(0)
    idx = flips[flips != 0].index
    if len(idx):
        days = (sig.index[-1] - idx[-1]).days
        if rel.loc[idx[-1]] == 1 and days <= cfg.buy_window * 4:
            pats.append({"label": f"Trend flipped up {days} days ago", "kind": "bull"})
        elif rel.loc[idx[-1]] == 0 and days <= cfg.buy_window * 4:
            pats.append({"label": f"Trend flipped down {days} days ago", "kind": "bear"})

    # 2) 20-day breakout (closing above the prior 20-day high)
    if n > 21:
        prior_high = _f(df["high"].iloc[-21:-1].max())
        if px >= prior_high:
            pats.append({"label": "Broke to a 1-month high", "kind": "bull"})

    # 3) Proximity to 1-year high/low
    if n >= 60:
        look = min(n, 252)
        hi = _f(df["high"].iloc[-look:].max())
        lo = _f(df["low"].iloc[-look:].min())
        if hi and px >= 0.98 * hi:
            pats.append({"label": "Near its 1-year high", "kind": "bull"})
        if lo and px <= 1.03 * lo:
            pats.append({"label": "Near its 1-year low", "kind": "bear"})

    # 4) Pullback to a rising trend line (buy-the-dip in an uptrend)
    slow_series = sig["slow"]
    if not np.isnan(slow) and slow and fast > slow:
        dist = (px - slow) / slow * 100
        slope_up = len(slow_series) > 6 and _f(slow_series.iloc[-1]) > _f(slow_series.iloc[-6])
        if 0 <= dist <= 3 and slope_up:
            pats.append({"label": "Dipped back to its rising trend", "kind": "bull"})

    # 5) RSI extremes + oversold turn-up
    if rsi_v >= 70:
        pats.append({"label": "Run up fast (may need a breather)", "kind": "bear"})
    elif rsi_v <= 30:
        pats.append({"label": "Beaten down (could bounce)", "kind": "bull"})
    rsi_series = sig["rsi"]
    if len(rsi_series) > 5 and _f(rsi_series.iloc[-5:].min()) <= 32 and rsi_v > _f(rsi_series.iloc[-2]):
        pats.append({"label": "Bouncing back from a dip", "kind": "bull"})

    # 6) MACD cross
    _, _, hist = ind.macd(close)
    hv = _f(hist.iloc[-1]) if len(hist) else float("nan")
    if len(hist.dropna()) > 2:
        h1, h0 = _f(hist.iloc[-1]), _f(hist.iloc[-2])
        if h1 > 0 and h0 <= 0:
            pats.append({"label": "Momentum just turned up", "kind": "bull"})
        elif h1 < 0 and h0 >= 0:
            pats.append({"label": "Momentum just turned down", "kind": "bear"})

    # 7) Bollinger position
    _, _, _, pb = ind.bollinger(close)
    pbv = _f(pb.iloc[-1]) if len(pb) else float("nan")
    if not np.isnan(pbv):
        if pbv >= 1:
            pats.append({"label": "Stretched above its normal range", "kind": "bear"})
        elif pbv <= 0:
            pats.append({"label": "Below its normal range (cheap)", "kind": "bull"})

    # 8) Trend strength (ADX) and unusual volume
    adx_series = ind.adx(df)
    adxv = _f(adx_series.iloc[-1]) if len(adx_series) else float("nan")
    if not np.isnan(adxv) and adxv >= 25:
        pats.append({"label": "Strong, steady trend", "kind": "neutral"})
    if relvol is not None and relvol >= 1.5:
        pats.append({"label": "Unusually heavy trading", "kind": "neutral"})

    extra = {
        "adx": None if np.isnan(adxv) else round(adxv, 1),
        "macd_hist": None if np.isnan(hv) else round(hv, 3),
        "bb_pct": None if np.isnan(pbv) else round(pbv, 2),
    }
    return pats, extra
