"""Market data access.

`get_bars` pulls real history from Alpaca when keys are present.
`synthetic_bars` generates a random-walk series so the backtester and tests
run with zero credentials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def synthetic_bars(symbol: str = "TEST", n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Deterministic random-walk OHLCV for testing/backtests without an API."""
    rng = np.random.default_rng(seed + hash(symbol) % 1000)
    rets = rng.normal(loc=0.0005, scale=0.015, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def sanitize_bars(df: pd.DataFrame, spike: float = 0.5) -> tuple[pd.DataFrame, int]:
    """Repair corrupt bars from the free IEX feed before they reach the indicators.

    Two fixes, both conservative (they only touch clearly-bad bars, never real moves):
      1. **Spike-and-revert bad prints** — a single bar whose close jumps >`spike` and is
         (mostly) undone the next day. That's a bad tick, not a real move (real moves don't
         instantly reverse), so we replace that bar with the average of its neighbours.
      2. **OHLC violations** — bars where high isn't the max or low isn't the min (e.g.
         close > high). We clamp high/low to the true extremes of {open, high, low, close}.

    Persistent jumps (a real gap or an already-split-adjusted move) are left alone — only
    the reverting spikes are treated as data errors. Returns (clean_df, n_repairs)."""
    if df is None or len(df) < 3:
        return df, 0
    df = df.copy()
    o = df["open"].to_numpy(float).copy()
    h = df["high"].to_numpy(float).copy()
    lo_ = df["low"].to_numpy(float).copy()
    c = df["close"].to_numpy(float).copy()
    repairs = 0
    # 1) spike-and-revert single-bar bad prints (on close)
    for i in range(1, len(c) - 1):
        p0, p1, p2 = c[i - 1], c[i], c[i + 1]
        if not (p0 and p1 and p2):
            continue
        r1 = p1 / p0 - 1.0
        if abs(r1) <= spike:
            continue
        r2 = p2 / p1 - 1.0
        if r1 * r2 < 0 and abs(r2) >= spike * 0.6:   # reverses → bad print
            mid = (p0 + p2) / 2.0
            o[i] = h[i] = lo_[i] = c[i] = mid
            repairs += 1
    # 2) OHLC integrity: high must be the max, low the min of the bar
    for i in range(len(c)):
        mx = max(o[i], h[i], lo_[i], c[i])
        mn = min(o[i], h[i], lo_[i], c[i])
        if h[i] != mx or lo_[i] != mn:
            h[i], lo_[i] = mx, mn
            repairs += 1
    df["open"], df["high"], df["low"], df["close"] = o, h, lo_, c
    return df, repairs


def get_bars(symbol: str, cfg: Config) -> pd.DataFrame:
    """Fetch daily/intraday bars from Alpaca. Requires valid keys."""
    cfg.validate_for_live()
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=cfg.lookback_days)
    # Free Alpaca accounts only have the IEX feed; SIP (the default) returns
    # empty/permission errors. Also keep `end` ~16 min back to dodge the free
    # plan's recent-data restriction.
    end = pd.Timestamp.utcnow() - pd.Timedelta(minutes=16)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map.get(cfg.timeframe, tf_map["1Day"]),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=DataFeed.IEX,
        adjustment=Adjustment.SPLIT,  # split-adjust so reverse splits (e.g. SPCE) don't fabricate huge moves
    )
    # Resilience: retry transient failures (rate-limit 429 / timeout / network) with backoff so a
    # brief Alpaca blip can't thin out or zero a whole scan. Re-raises after the last attempt, so
    # the scanner still records + skips a genuinely dead symbol.
    import time as _time
    bars = None
    for _attempt in range(3):
        try:
            bars = client.get_stock_bars(req).df
            break
        except Exception:  # noqa: BLE001
            if _attempt == 2:
                raise
            _time.sleep(0.5 * (2 ** _attempt))   # 0.5s, then 1.0s before retrying
    if bars.empty:
        return bars
    # Multi-index (symbol, timestamp) -> single symbol frame
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    bars = bars[["open", "high", "low", "close", "volume"]]
    # Clean IEX bad prints / OHLC glitches before indicators are computed on them.
    bars, _ = sanitize_bars(bars)
    return bars
