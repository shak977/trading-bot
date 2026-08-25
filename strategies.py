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


def pullback_uptrend(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Swing pullback: in a healthy uptrend, buy the controlled dip back to the 20-day EMA rather
    than chasing strength. The complement to the breakout strategies — they buy highs, this buys the
    rest *within* a trend, which is where the data says the edge is (non-extended entries won 64%
    vs 40% when chasing). Holds while the trend holds: exits only when the 50-day is lost.

    Entry: price above the 50-day AND 200-day (trend intact), pulls back to within ~1% of the
    20-day EMA (or briefly under it) without breaking the 50-day, then closes back up.
    Exit: close below the 50-day SMA — a genuine trend break, not noise. That long exit is what
    makes this a SWING strategy rather than a 2-5 day mean-reversion scalp.
    """
    c = df["close"]
    e20, s50, s200 = ind.ema(c, 20), ind.sma(c, 50), ind.sma(c, 200)
    uptrend = (c > s50) & (s50 > s200)
    near20 = c <= e20 * 1.01                      # dipped to/through the 20-day
    reclaim = c > c.shift(1)                      # and turned back up
    entry = uptrend & near20.shift(1, fill_value=False) & reclaim & (c > s50)
    exit_ = c < s50
    return _state(entry, exit_)


# ---- short-side mirrors: each returns a 0/1 position Series (1 = short active) ----
# Same state-machine convention as the long side, but the triggers are the bearish
# mirror image. "1" means "a short is on" so confluence counting stays uniform.

def trend_cross_down(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mirror of the house strategy: fast crosses BELOW slow, RSI not yet oversold."""
    c = df["close"]
    f, s = ind.sma(c, cfg.fast_ma), ind.sma(c, cfg.slow_ma)
    r = ind.rsi(c, cfg.rsi_period)
    entry = (f < s) & (f.shift(1) >= s.shift(1)) & (r > cfg.rsi_oversold)
    exit_ = (f > s) & (f.shift(1) <= s.shift(1)) | (r <= cfg.rsi_oversold)
    return _state(entry, exit_)


def death_cross(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Classic 50/200 death cross — slow, positional downtrend regime."""
    c = df["close"]
    f, s = ind.sma(c, 50), ind.sma(c, 200)
    entry = (f < s) & (f.shift(1) >= s.shift(1))
    exit_ = (f > s) & (f.shift(1) <= s.shift(1))
    return _state(entry, exit_)


def donchian_breakdown(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Turtle mirror: short a new 20-day low, cover on a 10-day high."""
    prior_low = df["low"].shift(1).rolling(20, min_periods=20).min()
    prior_high = df["high"].shift(1).rolling(10, min_periods=10).max()
    entry = df["close"] < prior_low
    exit_ = df["close"] > prior_high
    return _state(entry, exit_)


def macd_trend_down(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """MACD line crossing DOWN through its signal while below zero (negative momentum)."""
    line, sigl, _ = ind.macd(df["close"])
    entry = (line < sigl) & (line.shift(1) >= sigl.shift(1)) & (line < 0)
    exit_ = line > sigl
    return _state(entry, exit_)


def rsi2_short(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Connors RSI(2) mirror: short a sharp overbought RIP inside a downtrend, cover on the drop."""
    c = df["close"]
    downtrend = c < ind.sma(c, 200)
    r2 = ind.rsi(c, 2)
    entry = downtrend & (r2 > 90)
    exit_ = c < ind.sma(c, 5)
    return _state(entry, exit_)


def bollinger_breakdown(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Volatility breakdown: close drops below the lower band after a squeeze."""
    c = df["close"]
    mid, up, lo, _ = ind.bollinger(c, 20, 2.0)
    bandwidth = (up - lo) / mid.replace(0.0, np.nan)
    floor = bandwidth.rolling(120, min_periods=30).quantile(0.25)
    squeezed = (bandwidth <= floor)
    recent_squeeze = squeezed.rolling(10, min_periods=1).max().astype(bool)
    entry = (c < lo) & (c.shift(1) >= lo.shift(1)) & recent_squeeze.shift(1, fill_value=False)
    exit_ = c > mid
    return _state(entry, exit_)


def ema_stack_down(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Momentum stack mirror: 8 < 21 < 50 EMAs aligned down; cover when 8 regains 21."""
    c = df["close"]
    e8, e21, e50 = ind.ema(c, 8), ind.ema(c, 21), ind.ema(c, 50)
    stacked = (e8 < e21) & (e21 < e50)
    entry = stacked & (~stacked.shift(1, fill_value=False))
    exit_ = e8 > e21
    return _state(entry, exit_)


# ---- post-earnings drift (PEAD): ride the drift after a strong, high-volume reaction ----

def _pead(df: pd.DataFrame, cfg: Config, *, short: bool) -> pd.Series:
    """Drift after an earnings-like reaction, inferred from price+volume (no date feed).

    A "reaction" bar = a close-to-close move of at least ``pead_gap_min`` in the trade's
    direction, on volume >= ``pead_vol_mult`` x the rolling-median volume (an earnings-style
    surprise). We then ride the drift while price holds the reaction (above its close for a
    long, below for a short) and within ``pead_window`` bars, exiting when price breaks back
    through the reaction bar's extreme. Bars-only, so it backtests like every other strategy.
    """
    if not getattr(cfg, "pead_enabled", True):
        return pd.Series(0.0, index=df.index, dtype="float64")
    c = df["close"]
    ret = c.pct_change()
    vol = df["volume"] if "volume" in df else pd.Series(1.0, index=df.index)
    med = vol.rolling(cfg.pead_vol_window, min_periods=5).median()
    surge = vol >= (cfg.pead_vol_mult * med)
    if short:
        reaction = (ret <= -cfg.pead_gap_min) & surge
    else:
        reaction = (ret >= cfg.pead_gap_min) & surge
    reaction = reaction.fillna(False)
    # carry forward the most recent reaction bar's close / extreme
    ref_close = c.where(reaction).ffill()
    ref_ext = (df["low"] if short else df["high"]).where(reaction).ffill()
    within = reaction.rolling(cfg.pead_window, min_periods=1).max().astype(bool)
    if short:
        holding = c <= ref_close
        entry = within & holding & reaction.cumsum().gt(0)
        exit_ = c > ref_ext
    else:
        holding = c >= ref_close
        entry = within & holding & reaction.cumsum().gt(0)
        exit_ = c < ref_ext
    return _state(entry.fillna(False), exit_.fillna(False))


def pead_long(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Long the drift after a strong, high-volume up reaction (post-earnings drift)."""
    return _pead(df, cfg, short=False)


def pead_short(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Short the drift after a strong, high-volume down reaction (mirror)."""
    return _pead(df, cfg, short=True)


# registry: key -> (label, fn, kind, blurb)
STRATEGIES: dict[str, tuple] = {
    "trend_cross": ("Trend crossover", trend_cross, "trend",
                    "20/50-day moving-average crossover with an RSI filter."),
    "golden_cross": ("Golden cross", golden_cross, "trend",
                     "50-day average above the 200-day — a long, positional regime."),
    "donchian": ("Donchian breakout", donchian_breakout, "breakout",
                 "Buys a fresh 20-day high; exits on a 10-day low (turtle-style)."),
    # RETIRED Aug 2026, both on their own evidence (see strategy_backtest.py):
    #  · "macd_trend" (MACD momentum) — 34.8% win / -1.13% avg return over 347 trades.
    #  · "pullback" (Pullback in uptrend) — added as a swing entry, then backtested under the live
    #    exit model across 59 symbols: -1.12% expectancy, payoff 1.09, -85% max drawdown on its own
    #    equity curve, and flagged FRAGILE (expectancy worsened when the trail was nudged either way).
    #    The function is kept below for reference//re-testing but is NOT in the live panel.
    "rsi2_meanrev": ("Dip buy (RSI-2)", rsi2_meanrev, "mean-reversion",
                     "Buys a sharp oversold dip inside an uptrend (Connors RSI-2)."),
    "bollinger": ("Squeeze breakout", bollinger_breakout, "breakout",
                  "Breaks above the Bollinger band after a low-volatility squeeze."),
    "ema_stack": ("EMA momentum stack", ema_stack, "momentum",
                  "Fast EMAs stacked above slow ones (8 > 21 > 50)."),
    "pead": ("Post-earnings drift", pead_long, "event",
             "Rides the drift after a strong, high-volume earnings-style reaction."),
}


SHORT_STRATEGIES: dict[str, tuple] = {
    "trend_cross_dn": ("Trend cross-down", trend_cross_down, "trend",
                       "20/50-day moving-average cross to the downside with an RSI filter."),
    "death_cross": ("Death cross", death_cross, "trend",
                    "50-day average below the 200-day — a long, positional downtrend."),
    "donchian_dn": ("Donchian breakdown", donchian_breakdown, "breakdown",
                    "Shorts a fresh 20-day low; covers on a 10-day high (turtle mirror)."),
    # "macd_trend_dn" retired alongside its long twin (see note above) to keep the panels symmetric.
    "rsi2_short": ("Rip-sell (RSI-2)", rsi2_short, "mean-reversion",
                   "Shorts a sharp overbought rip inside a downtrend (Connors RSI-2 mirror)."),
    "bollinger_dn": ("Squeeze breakdown", bollinger_breakdown, "breakdown",
                     "Breaks below the Bollinger band after a low-volatility squeeze."),
    "ema_stack_dn": ("EMA momentum stack (down)", ema_stack_down, "momentum",
                     "Fast EMAs stacked below slow ones (8 < 21 < 50)."),
    "pead_dn": ("Post-earnings drift (down)", pead_short, "event",
                "Rides the down-drift after a strong, high-volume negative reaction."),
}


def regime_confluence(kinds: list[str], regime_label: str | None, cfg: Config,
                      short: bool = False) -> dict | None:
    """Regime-weighted confluence 'fit' (adapted from Scientia REGIME_WEIGHTS): given the KINDS of
    the strategies firing right now, how well-suited are they to the current tape? Returns a 0-1
    fit score (average regime-weight of the firing strategies, normalized to the best possible),
    plus the raw weighted sum. A short reads the regime mirror-image (a short in Risk-off is 'with
    the tape' like a long in Risk-on). None when disabled / no signals / no weights configured."""
    if not getattr(cfg, "regime_confluence_enabled", True) or not kinds or not regime_label:
        return None
    weights = getattr(cfg, "regime_kind_weights", {}) or {}
    eff = regime_label
    if short:
        eff = {"Risk-on": "Risk-off", "Risk-off": "Risk-on"}.get(regime_label, regime_label)
    table = weights.get(eff) or weights.get("Neutral") or {}
    if not table:
        return None
    raw = sum(table.get(k, 0.5) for k in kinds)
    best = max(table.values()) or 1.0
    fit = raw / (len(kinds) * best)                     # average fit of the firing strategies, 0-1
    return {"fit": round(fit, 3), "raw": round(raw, 2), "n": len(kinds), "regime": eff}


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


def evaluate_short(df: pd.DataFrame, cfg: Config) -> dict:
    """Bearish mirror of ``evaluate``: which short strategies are active right now.

    Returns {results, short, fresh, count, total} where ``short`` are strategies
    currently positioned short and ``fresh`` turned short within the last
    ``buy_window`` bars (a new bearish setup)."""
    results: dict[str, dict] = {}
    short_labels, fresh_labels = [], []
    for key, (label, fn, kind, _blurb) in SHORT_STRATEGIES.items():
        try:
            pos = fn(df, cfg)
            is_short = bool(len(pos)) and bool(pos.iloc[-1] == 1.0)
            since = _bars_since_entry(pos)
            is_fresh = bool(is_short and since is not None and since <= cfg.buy_window)
        except Exception:  # noqa: BLE001
            is_short = is_fresh = False
            since = None
        results[key] = {"label": label, "kind": kind, "short": is_short,
                        "fresh": is_fresh, "bars_held": since}
        if is_short:
            short_labels.append(label)
        if is_fresh:
            fresh_labels.append(label)
    return {"results": results, "short": short_labels, "fresh": fresh_labels,
            "count": len(short_labels), "total": len(SHORT_STRATEGIES)}
