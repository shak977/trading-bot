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


def test_tracker_no_lookahead():
    print("tracker (no look-ahead):")
    import tempfile, json
    import pandas as pd
    import tracker
    idx = pd.to_datetime(["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"])
    df = pd.DataFrame({"open": [100, 101, 102, 103, 104], "high": [101, 102, 116, 105, 106],
                       "low": [99, 100, 101, 102, 103], "close": [100, 101, 110, 104, 105]}, index=idx)
    tracker.synthetic_bars = lambda *a, **k: df
    tracker.get_bars = lambda *a, **k: df
    tf = tempfile.mktemp(suffix=".json"); tracker.PATH = tf
    json.dump([{"id": "X:2026-06-09", "symbol": "X", "name": "X", "advised_date": "2026-06-09",
                "entry": 100, "stop": 95, "target": 115, "rr": 3, "conviction": "High", "status": "open"}],
              open(tf, "w"))
    tracker.run([], CONFIG, live=False, today="2026-06-12")
    t = json.load(open(tf))[0]
    _ok("past trade resolves on a completed later day", t["status"] == "win" and t["exit_date"] == "2026-06-10" and t["days_held"] == 1)
    json.dump([{"id": "Y:2026-06-12", "symbol": "Y", "name": "Y", "advised_date": "2026-06-12",
                "entry": 100, "stop": 95, "target": 115, "rr": 3, "conviction": "High", "status": "open"}],
              open(tf, "w"))
    tracker.run([], CONFIG, live=False, today="2026-06-12")
    _ok("same-day trade stays open (no look-ahead)", json.load(open(tf))[0]["status"] == "open")


def test_momentum_dataquality():
    print("momentum data-quality guards:")
    import pandas as pd
    import momentum as mom
    idx = pd.date_range("2024-01-01", periods=300)
    clean = list(np.linspace(100, 220, 300))                 # real winner, ~+77% 12-1
    _ok("clean series not flagged", not mom.has_bad_bar(pd.Series(clean, index=idx)))
    bad = clean.copy(); bad[40] = clean[40] * 6; bad[41] = clean[41]  # spike-and-revert print
    _ok("spike-and-revert flagged", mom.has_bad_bar(pd.Series(bad, index=idx)))
    _ok("bad-bar name dropped from rank",
        mom.rank({"B": pd.DataFrame({"close": bad}, index=idx)}) == [])
    hot = list(np.linspace(30, 300, 300))                    # smooth but >200% 12-1
    _ok("over-cap score dropped",
        mom.rank({"H": pd.DataFrame({"close": hot}, index=idx)}, max_score_pct=200.0) == [])
    _ok("over-cap kept when cap lifted",
        mom.rank({"H": pd.DataFrame({"close": hot}, index=idx)}, max_score_pct=999.0) != [])
    import dashboard as d
    _ok("dashboard helper agrees (clean)", not d._has_bad_bar(clean))
    _ok("dashboard helper agrees (bad)", d._has_bad_bar(bad))


def test_short_engine():
    print("short engine (classifier + plan + conviction):")
    import numpy as np
    import pandas as pd
    import strategies, scanner
    from data import synthetic_bars
    # build a clean downtrend
    df = synthetic_bars("DN", n=320)
    base = pd.Series(np.linspace(300, 140, 320), index=df.index)
    for c in ["open", "high", "low", "close"]:
        df[c] = base * (1 + (df[c] / df["close"] - 1).fillna(0))
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    sh = strategies.evaluate_short(df, CONFIG)
    _ok("downtrend fires short strategies", sh["count"] >= 1)
    _ok("evaluate_short shape", set(sh) >= {"short", "fresh", "count", "total"})
    # classifier: 3+ bear + downtrend = SHORT/HOLD SHORT; 2 = WATCH SHORT
    bull0 = {"count": 0, "fresh": [], "long": [], "total": 7}
    a, d = scanner._classify(bull0, {"count": 3, "fresh": ["x"], "short": [], "total": 7}, False, True, False, CONFIG)
    _ok("3 bear + downtrend -> SHORT", a == "SHORT" and d == "SHORT")
    a, d = scanner._classify(bull0, {"count": 2, "fresh": [], "short": [], "total": 7}, False, True, False, CONFIG)
    _ok("2 bear + downtrend -> WATCH SHORT", a == "WATCH SHORT")
    a, d = scanner._classify({"count": 3, "fresh": ["x"], "long": [], "total": 7},
                             {"count": 0, "fresh": [], "short": [], "total": 7}, True, False, False, CONFIG)
    _ok("3 bull + uptrend -> BUY", a == "BUY" and d == "LONG")
    a, d = scanner._classify(bull0, {"count": 0, "fresh": [], "short": [], "total": 7}, False, True, True, CONFIG)
    _ok("recent long exit -> EXIT", a == "EXIT")
    # SHORT plan geometry: stop above entry, target below
    from strategy import generate_signals
    sig = generate_signals(df, CONFIG)
    plan, _ctx = scanner._trade_plan(df, sig, CONFIG, 100.0, CONFIG.starting_cash, "SHORT")
    _ok("short stop above entry", plan["stop"] > plan["entry"])
    _ok("short target below entry", plan["target"] < plan["entry"])
    _ok("short rr positive", (plan["rr"] or 0) > 0)


def test_short_tracker():
    print("tracker (short grading):")
    import tempfile, json
    import pandas as pd
    import tracker
    idx = pd.to_datetime(["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"])
    # price falls after advice -> a short should WIN at its (below-entry) target
    df = pd.DataFrame({"open": [100, 99, 92, 90, 89], "high": [101, 100, 95, 91, 90],
                       "low": [99, 96, 84, 88, 87], "close": [100, 97, 85, 89, 88]}, index=idx)
    tracker.synthetic_bars = lambda *a, **k: df
    tracker.get_bars = lambda *a, **k: df
    tf = tempfile.mktemp(suffix=".json"); tracker.PATH = tf
    json.dump([{"id": "S:2026-06-09", "symbol": "S", "name": "S", "direction": "SHORT",
                "advised_date": "2026-06-09", "entry": 100, "stop": 105, "target": 88, "rr": 2.4,
                "conviction": "High", "status": "open"}], open(tf, "w"))
    tracker.run([], CONFIG, live=False, today="2026-06-12")
    t = json.load(open(tf))[0]
    _ok("short wins when price falls to target", t["status"] == "win" and t["return_pct"] > 0)


def test_short_backtest():
    print("short backtest (direction correct):")
    import pandas as pd
    from backtest import backtest_positions
    # a steady decline: a held short should make money; the same series long should lose
    idx = pd.date_range("2024-01-01", periods=60)
    close = pd.Series([100 * (0.99 ** i) for i in range(60)], index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close})
    pos = pd.Series(1.0, index=idx)  # always in the position
    short_ret = backtest_positions(df, pos, CONFIG, side="short").metrics["total_return"]
    long_ret = backtest_positions(df, pos, CONFIG, side="long").metrics["total_return"]
    _ok("short profits on a falling series", short_ret > 0)
    _ok("long loses on the same falling series", long_ret < 0)


def test_sanitize_bars():
    print("data sanitizer (bad-print repair):")
    import pandas as pd
    from data import sanitize_bars
    idx = pd.date_range("2024-01-01", periods=8)
    close = [100, 101, 102, 260, 103, 104, 105, 106]  # bar 3 is a spike-and-revert bad print
    df = pd.DataFrame({"open": close, "high": [x * 1.01 for x in close],
                       "low": [x * 0.99 for x in close], "close": close,
                       "volume": [1e6] * 8}, index=idx)
    # also inject an OHLC violation on bar 5 (high below close)
    df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["close"] * 0.5
    clean, reps = sanitize_bars(df)
    _ok("repairs detected", reps >= 2)
    _ok("spike bar repaired (~102.5)", 100 < clean.iloc[3]["close"] < 110)
    _ok("no >50% one-day jumps remain",
        not any(abs(clean["close"].iloc[i] / clean["close"].iloc[i-1] - 1) > 0.5
                for i in range(1, len(clean)) if clean["close"].iloc[i-1]))
    _ok("OHLC integrity restored",
        bool((clean["high"] >= clean[["open","low","close"]].max(axis=1) - 1e-6).all()
             and (clean["low"] <= clean[["open","high","close"]].min(axis=1) + 1e-6).all()))


def test_audit_direction_aware():
    print("audit (direction-aware plans):")
    import audit
    short_sig = {"symbol": "X", "price": 100, "direction": "SHORT",
                 "plan": {"direction": "SHORT", "entry": 100, "stop": 105, "target": 88, "rr": 2.4}}
    long_sig = {"symbol": "Y", "price": 100, "direction": "LONG",
                "plan": {"direction": "LONG", "entry": 100, "stop": 95, "target": 115, "rr": 3.0}}
    _, flags = audit.audit_data({"signals": [short_sig, long_sig], "charts": {}})
    errs = [f["msg"] for f in flags if f["level"] == "error"]
    _ok("valid short plan not flagged", not any("X" in m for m in errs))
    _ok("valid long plan not flagged", not any("Y" in m for m in errs))
    # a long with short-style inverted levels SHOULD still be caught
    bad = {"symbol": "Z", "price": 100, "direction": "LONG",
           "plan": {"direction": "LONG", "entry": 100, "stop": 105, "target": 88}}
    _, flags2 = audit.audit_data({"signals": [bad], "charts": {}})
    _ok("genuinely inverted long plan still flagged",
        any("Z" in f["msg"] for f in flags2 if f["level"] == "error"))


def main():
    test_audit_direction_aware()
    test_sanitize_bars()
    test_indicators()
    test_strategy_backtest()
    test_analytics()
    test_strategies()
    test_confluence_in_scan()
    test_tracker_no_lookahead()
    test_momentum_dataquality()
    test_short_engine()
    test_short_tracker()
    test_short_backtest()
    test_research()
    test_rescore()
    test_llm_prompt()
    test_regime()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
