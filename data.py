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


def get_bars(symbol: str, cfg: Config) -> pd.DataFrame:
    """Fetch daily/intraday bars from Alpaca. Requires valid keys."""
    cfg.validate_for_live()
    from alpaca.data.enums import DataFeed
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
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        return bars
    # Multi-index (symbol, timestamp) -> single symbol frame
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars[["open", "high", "low", "close", "volume"]]
