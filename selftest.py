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


def test_research():
    print("research:")
    import research
    sent = research.news_sentiment([{"headline": "Stock surges as profit beats, analysts upgrade"},
                                    {"headline": "Shares rally to record high"}])
    _ok("positive sentiment detected", sent and sent["label"] == "Positive")
    neg = research.news_sentiment([{"headline": "Stock plunges on weak guidance and downgrade"}])
    _ok("negative sentiment detected", neg and neg["label"] == "Negative")
    _ok("finnhub none without key", research.finnhub_snapshot("AAPL", CONFIG) is None or CONFIG.finnhub_api_key)
    _ok("fred none without key", research.fred_macro(CONFIG) is None or CONFIG.fred_api_key)


def test_rescore():
    print("rescore with research:")
    import scanner
    from data import synthetic_bars
    row = scanner._analyse("MSFT", synthetic_bars("MSFT", n=300), CONFIG, CONFIG.starting_cash)
    before = row["conviction"]["total"]
    scanner.rescore(row, CONFIG,
                    sentiment={"label": "Positive", "n": 3, "score": 0.5},
                    fundamentals={"analysts": {"buy": 8, "hold": 2, "sell": 0, "consensus": "Buy"},
                                  "target_mean": row["price"] * 1.2, "pe": 25})
    _ok("conviction gained research checks", row["conviction"]["total"] > before)
    _ok("desk read mentions analysts", "analyst" in row["desk_read"].lower())


def test_strategies():
    print("strategies + confluence:")
    import strategies
    import analytics
    df = synthetic_bars("TEST", n=CONFIG.lookback_days)
    for k in strategies.STRATEGIES:
        p = strategies.positions(df, CONFIG, k)
        _ok(f"{k} is 0/1 series", len(p) == len(df) and set(p.dropna().unique()).issubset({0.0, 1.0}))
    ev = strategies.evaluate(df, CONFIG)
    _ok("confluence count within range", 0 <= ev["count"] <= ev["total"] == len(strategies.STRATEGIES))
    se = analytics.strategy_edges(df, CONFIG)
    _ok("edges cover every strategy", set(se["by"].keys()) == set(strategies.STRATEGIES.keys()))
    _ok("each edge has win/trade fields", all(("win_rate" in v and "n_trades" in v) for v in se["by"].values()))


def test_confluence_in_scan():
    print("scan integration:")
    import scanner
    row = scanner._analyse("AAPL", synthetic_bars("AAPL", n=CONFIG.lookback_days), CONFIG, CONFIG.starting_cash)
    _ok("row carries strategies.now", "strategies" in row and "now" in row["strategies"])
    _ok("factors carry confluence", row["factors"].get("confluence") is not None)
    labels = [c["label"] for c in row["conviction"]["checks"]]
    _ok("conviction includes strategies check", "Strategies agree?" in labels)
    row.pop("_df", None)  # transient frame _analyse stashes for scan()


def main():
    test_indicators()
    test_strategy_backtest()
    test_analytics()
    test_strategies()
    test_confluence_in_scan()
    test_research()
    test_rescore()
    test_llm_prompt()
    test_regime()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
