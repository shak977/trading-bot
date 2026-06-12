"""Pure-pandas technical indicators. No external TA dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — a volatility measure in price units."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # Use np.nan (float) not pd.NA so the series stays float64 — avoids the
    # pandas downcasting FutureWarning on fillna.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).astype("float64")  # no losses -> RSI 100


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram). Momentum/trend gauge."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(series: pd.Series, window: int = 20, mult: float = 2.0):
    """Returns (mid, upper, lower, pct_b). pct_b: where price sits in the band (0-1)."""
    mid = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std()
    upper, lower = mid + mult * sd, mid - mult * sd
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (series - lower) / width
    return mid, upper, lower, pct_b


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend STRENGTH (not direction). >25 = trending."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)) * down.clip(lower=0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()
