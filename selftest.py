"""Lightweight self-tests (no pytest needed): python3 selftest.py

Exercises indicators, strategy, analytics and the backtester on synthetic data
so we can verify the analytics pipeline without any API keys.
"""
from __future__ import annotations

import numpy as np

from config import CONFIG
from data import synthetic_bars
import indicators as ind


def _ok(name, cond):
    print(("  ok  " if cond else " FAIL ") + name)
    assert cond, name


def test_indicators():
    print("indicators:")
    df = synthetic_bars("TEST", n=300)
    c = df["close"]
    _ok("sma length", len(ind.sma(c, 20)) == len(c))
    rsi = ind.rsi(c, 14)
    _ok("rsi in 0..100", float(rsi.dropna().min()) >= 0 and float(rsi.dropna().max()) <= 100)
    m, s, h = ind.macd(c)
    _ok("macd hist = line-signal", np.allclose((m - s).dropna(), h.dropna(), equal_nan=True))
    mid, up, lo, pb = ind.bollinger(c)
    _ok("bollinger upper>=lower", bool((up.dropna() >= lo.dropna()).all()))
    adx = ind.adx(df, 14)
    _ok("adx finite", np.isfinite(adx.dropna()).all())
    _ok("rolling_high>=close", bool((ind.rolling_high(c, 20) >= c).all()))


def test_strategy_backtest():
    print("strategy + backtest:")
    from strategy import generate_signals
    from backtest import run_backtest
    df = synthetic_bars("TEST", n=300)
    sig = generate_signals(df, CONFIG)
    _ok("signal is 0/1", set(sig["signal"].unique()).issubset({0.0, 1.0}))
    res = run_backtest(df, CONFIG)
    m = res.metrics
    _ok("backtest has metrics", all(k in m for k in ("win_rate", "n_trades", "total_return", "max_drawdown")))
    _ok("win_rate 0..1", 0 <= m["win_rate"] <= 1)


def test_analytics():
    print("analytics:")
    import analytics
    from strategy import generate_signals
    df = synthetic_bars("AAPL", n=300)
    sig = generate_signals(df, CONFIG)
    pats, extra = analytics.detect(df, sig, CONFIG, 1.6)
    _ok("patterns is list of dicts", isinstance(pats, list) and all("kind" in p for p in pats))
    _ok("extra has adx/macd/bb", all(k in extra for k in ("adx", "macd_hist", "bb_pct")))
    edge = analytics.backtest_edge(df, CONFIG)
    _ok("edge has win_rate key", edge is None or "win_rate" in edge)


def test_llm_prompt():
    print("llm prompt:")
    import llm
    df = synthetic_bars("MSFT", n=300)
    from scanner import _analyse
    row = _analyse("MSFT", df, CONFIG, CONFIG.starting_cash)
    txt = llm._prompt(row, {"label": "Neutral", "breadth": 50})
    _ok("prompt mentions ticker", "MSFT" in txt)
    _ok("prompt mentions patterns/indicators", "Indicators:" in txt and "Chart patterns:" in txt)


def test_regime():
    print("regime + sectors:")
    import dashboard, scanner
    rows = scanner.scan(CONFIG, live=False)
    reg = dashboard._market_regime(rows)
    secs = dashboard._sector_strength(rows)
    _ok("regime label valid", reg is None or reg["label"] in ("Risk-on", "Neutral", "Risk-off"))
    _ok("sectors sorted desc", all(secs[i]["pct_up"] >= secs[i+1]["pct_up"] for i in range(len(secs)-1)))


def main():
    test_indicators()
    test_strategy_backtest()
    test_analytics()
    test_llm_prompt()
    test_regime()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
