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
    _expect = set(strategies.STRATEGIES.keys()) | set(strategies.SHORT_STRATEGIES.keys())
    _ok("edges cover every long + short strategy", set(se["by"].keys()) == _expect)
    _ok("edges tagged with side", all(v.get("side") in ("long", "short") for v in se["by"].values()))
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
    tracker.run([], CONFIG, live=True, today="2026-06-12")  # live=True: persists (bars are patched)
    t = json.load(open(tf))[0]
    _ok("past trade resolves on a completed later day", t["status"] == "win" and t["exit_date"] == "2026-06-10" and t["days_held"] == 1)
    json.dump([{"id": "Y:2026-06-12", "symbol": "Y", "name": "Y", "advised_date": "2026-06-12",
                "entry": 100, "stop": 95, "target": 115, "rr": 3, "conviction": "High", "status": "open"}],
              open(tf, "w"))
    tracker.run([], CONFIG, live=True, today="2026-06-12")
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
    tracker.run([], CONFIG, live=True, today="2026-06-12")  # live=True: persists (bars are patched)
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


def test_tradingview():
    print("tradingview cross-check:")
    import tradingview as tv
    _ok("bucket strong buy", tv._bucket(0.7) == "Strong Buy")
    _ok("bucket buy", tv._bucket(0.2) == "Buy")
    _ok("bucket neutral", tv._bucket(0.0) == "Neutral")
    _ok("bucket strong sell", tv._bucket(-0.8) == "Strong Sell")
    _ok("long agrees w/ buy ratings", tv.alignment({"d": "Strong Buy", "w": "Buy"}, "LONG") == "agree")
    _ok("long opposes sell ratings", tv.alignment({"d": "Sell", "w": "Strong Sell"}, "LONG") == "oppose")
    _ok("short agrees w/ sell ratings", tv.alignment({"d": "Sell", "w": "Strong Sell"}, "SHORT") == "agree")
    _ok("mixed when split", tv.alignment({"d": "Buy", "w": "Neutral"}, "LONG") == "mixed")
    _ok("none when no data", tv.alignment(None, "LONG") is None)


def test_rs():
    print("relative strength:")
    import numpy as np
    import pandas as pd
    import rs
    idx = pd.date_range("2024-01-01", periods=300)
    bench = pd.Series(np.linspace(100, 110, 300), index=idx)        # benchmark +10%
    strong = pd.Series(np.linspace(100, 200, 300), index=idx)       # way ahead of bench
    weak = pd.Series(np.linspace(100, 95, 300), index=idx)          # behind bench
    s_strong = rs.rs_score(strong, bench, (21, 63, 126), (0.2, 0.3, 0.5))
    s_weak = rs.rs_score(weak, bench, (21, 63, 126), (0.2, 0.3, 0.5))
    _ok("strong name has positive RS", s_strong > 0)
    _ok("weak name has negative RS", s_weak < 0)
    _ok("strong beats weak", s_strong > s_weak)
    _ok("short history -> None", rs.rs_score(strong.iloc[:50], bench.iloc[:50], (21, 63, 126), (0.2, 0.3, 0.5)) is None)
    ranks = rs.rank_universe({"A": s_strong, "B": s_weak, "C": 0.0})
    _ok("ranks cover all symbols", set(ranks) == {"A", "B", "C"})
    _ok("strong is top percentile", ranks["A"]["pct"] == 100)
    _ok("weak is bottom percentile", ranks["B"]["pct"] <= ranks["C"]["pct"] <= ranks["A"]["pct"])
    _ok("None scores excluded from ranking", "Z" not in rs.rank_universe({"A": s_strong, "Z": None}))


def test_conviction_weighting():
    print("conviction weighting:")
    import scanner
    from data import synthetic_bars
    from config import CONFIG
    row = scanner._analyse("MSFT", synthetic_bars("MSFT", n=CONFIG.lookback_days), CONFIG, CONFIG.starting_cash)
    checks = row["conviction"]["checks"]
    _ok("every check carries an explicit weight", all("weight" in c for c in checks))
    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    manual = round(sum(pts[c["status"]] for c in checks) / len(checks) * 100)
    _ok("unweighted score matches simple average", row["conviction"]["score_pct"] == manual)


def test_rs_conviction():
    print("rs conviction check:")
    import scanner
    from data import synthetic_bars
    from config import CONFIG

    def _rs_status(direction, pct):
        # Direction is forced explicitly: synthetic_bars seeds off hash(symbol), which is
        # process-randomised, so a named synthetic's direction is not stable across runs.
        row = scanner._analyse("MSFT", synthetic_bars("MSFT", n=CONFIG.lookback_days), CONFIG, CONFIG.starting_cash)
        row["direction"] = direction
        row.setdefault("factors", {})["rs"] = {"rs": 0.25 if pct >= 50 else -0.2, "pct": pct}
        scanner.rescore(row, CONFIG)
        labels = [c["label"] for c in row["conviction"]["checks"]]
        assert "Leading the market?" in labels, "RS check missing"
        c = next(c for c in row["conviction"]["checks"] if c["label"] == "Leading the market?")
        return c

    c = _rs_status("LONG", 92)
    _ok("RS check appears once RS factor present", c["label"] == "Leading the market?")
    _ok("RS check is weighted heavily", c["weight"] == CONFIG.rs_conviction_weight)
    _ok("high-RS long passes", c["status"] == "pass")
    _ok("low-RS long fails", _rs_status("LONG", 8)["status"] == "fail")
    # for a short, a laggard (low percentile) is the favourable case
    _ok("low-RS short passes", _rs_status("SHORT", 8)["status"] == "pass")
    _ok("high-RS short fails", _rs_status("SHORT", 92)["status"] == "fail")


def test_rs_in_scan():
    print("rs wired into scan:")
    import scanner
    from config import CONFIG
    rows = scanner.scan(CONFIG, live=False)
    _ok("scan returns rows", len(rows) > 0)
    rs_rows = [r for r in rows if (r.get("factors") or {}).get("rs")]
    _ok("at least one row carries an RS factor", len(rs_rows) > 0)
    pcts = [r["factors"]["rs"]["pct"] for r in rs_rows]
    _ok("RS percentiles are 0..100", all(0 <= p <= 100 for p in pcts))
    sample = rs_rows[0]
    _ok("RS check present in conviction", any(c["label"] == "Leading the market?" for c in sample["conviction"]["checks"]))


def test_universe_cap():
    print("universe cap:")
    import dataclasses
    import scanner
    from config import CONFIG
    cfg = dataclasses.replace(CONFIG, wide_universe=True, max_candidates=50)
    u = scanner.build_universe(cfg)
    _ok("universe respects max_candidates", len(u) <= cfg.max_candidates)
    _ok("universe is de-duplicated", len(u) == len(set(u)))
    _ok("core names still present", "AAPL" in u or "MSFT" in u)


def test_tracker_checks_snapshot():
    print("tracker stores conviction checks:")
    import tempfile, json
    import tracker
    from config import CONFIG
    tf = tempfile.mktemp(suffix=".json"); tracker.PATH = tf
    json.dump([], open(tf, "w"))
    sig = {"symbol": "X", "name": "X Co", "action": "BUY", "direction": "LONG",
           "price": 100.0,
           "plan": {"entry": 100.0, "stop": 95.0, "target": 115.0, "rr": 3.0},
           "conviction": {"label": "High", "checks": [
               {"label": "Is it trending up?", "status": "pass", "weight": 1.0},
               {"label": "Leading the market?", "status": "pass", "weight": 2.0}]}}
    tracker.run([sig], CONFIG, live=True, today="2026-06-12")
    rec = json.load(open(tf))[0]
    _ok("record stores conviction checks", isinstance(rec.get("checks"), list) and len(rec["checks"]) == 2)
    _ok("snapshot keeps label+status", set(rec["checks"][0]) >= {"label", "status"})


def test_attribution():
    print("factor attribution:")
    import attribution
    log = [
        {"status": "win",  "checks": [{"label": "Leading the market?", "status": "pass"},
                                       {"label": "Room to rise?", "status": "fail"}]},
        {"status": "win",  "checks": [{"label": "Leading the market?", "status": "pass"},
                                       {"label": "Room to rise?", "status": "pass"}]},
        {"status": "loss", "checks": [{"label": "Leading the market?", "status": "fail"},
                                       {"label": "Room to rise?", "status": "pass"}]},
        {"status": "open", "checks": [{"label": "Leading the market?", "status": "pass"}]},
    ]
    rep = attribution.attribute(log)
    by = {r["label"]: r for r in rep}
    _ok("only resolved trades counted", by["Leading the market?"]["n_pass"] == 2)
    _ok("RS pass win rate computed", by["Leading the market?"]["win_rate_pass"] == 100.0)
    _ok("RS fail win rate computed", by["Leading the market?"]["win_rate_fail"] == 0.0)
    _ok("edge = pass minus fail win rate", by["Leading the market?"]["edge"] == 100.0)
    _ok("report sorted by edge desc", rep == sorted(rep, key=lambda r: -(r["edge"] if r["edge"] is not None else -999)))


def test_attribution_panel():
    print("attribution panel renders:")
    import dashboard
    rep = [{"label": "Leading the market?", "n_pass": 12, "n_fail": 5,
            "win_rate_pass": 64.0, "win_rate_fail": 30.0, "edge": 34.0},
           {"label": "Retail buzz?", "n_pass": 8, "n_fail": 3,
            "win_rate_pass": 40.0, "win_rate_fail": 45.0, "edge": -5.0}]
    html = dashboard._attribution_html(rep)
    _ok("panel names the best factor", "Leading the market?" in html)
    _ok("panel shows the edge value", "+34" in html or "34.0" in html)
    _ok("empty report yields a friendly note", "accruing" in dashboard._attribution_html([]).lower())


def test_risk_multiplier():
    print("conviction/vol sizing:")
    import risk
    import dataclasses
    from config import CONFIG
    cfg = CONFIG
    m_hi = risk.risk_multiplier("High", cfg.vol_target_atr_pct, cfg)
    m_md = risk.risk_multiplier("Medium", cfg.vol_target_atr_pct, cfg)
    m_lo = risk.risk_multiplier("Low", cfg.vol_target_atr_pct, cfg)
    _ok("High >= Medium >= Low", m_hi >= m_md >= m_lo)
    _ok("at target ATR%, High mult == conv_mult_high", abs(m_hi - cfg.conv_mult_high) < 1e-9)
    m_calm = risk.risk_multiplier("High", cfg.vol_target_atr_pct, cfg)
    m_wild = risk.risk_multiplier("High", cfg.vol_target_atr_pct * 2, cfg)
    _ok("higher vol -> smaller size", m_wild < m_calm)
    _ok("never exceeds base ceiling", risk.risk_multiplier("High", 0.1, cfg) <= 1.0 + 1e-9)
    _ok("floored at min_size_mult", risk.risk_multiplier("Low", 100.0, cfg) >= cfg.min_size_mult - 1e-9)
    off = dataclasses.replace(cfg, size_by_conviction=False)
    _ok("disabled => multiplier 1.0", risk.risk_multiplier("Low", 100.0, off) == 1.0)


def test_paper_sizing():
    print("paper conviction sizing:")
    import paper
    from config import CONFIG
    base = paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=1.0)
    half = paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=0.5)
    _ok("half multiplier ~ half the shares", half <= base and half >= base // 2 - 1)
    _ok("min multiplier shrinks size", paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct, mult=CONFIG.min_size_mult) < base)
    _ok("mult defaults to 1.0", paper._qty(100_000, 100_000, 100.0, 95.0, CONFIG.paper_risk_pct) == base)


def test_time_stop():
    print("backtest time-stop:")
    import dataclasses
    import pandas as pd
    from backtest import backtest_positions
    from config import CONFIG
    idx = pd.date_range("2024-01-01", periods=20)
    close = pd.Series([100 + (i % 2) * 0.2 for i in range(20)], index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close})
    pos = pd.Series(1.0, index=idx)
    cfg = dataclasses.replace(CONFIG, max_hold_days=5, trail_atr_mult=0.0)
    res = backtest_positions(df, pos, cfg, side="long")
    reasons = [t.get("reason") for t in res.trades.to_dict("records") if t.get("reason")]
    _ok("a time-stop exit is recorded", "time" in reasons)
    cfg_off = dataclasses.replace(CONFIG, max_hold_days=0, trail_atr_mult=0.0)
    res_off = backtest_positions(df, pos, cfg_off, side="long")
    _ok("no time exit when disabled", "time" not in [t.get("reason") for t in res_off.trades.to_dict("records")])


def test_partial_take():
    print("backtest partial-take:")
    import dataclasses
    import pandas as pd
    from backtest import backtest_positions
    from config import CONFIG
    idx = pd.date_range("2024-01-01", periods=12)
    close = pd.Series([100, 101, 103, 106, 110, 113, 116, 118, 120, 121, 122, 123], index=idx, dtype=float)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close})
    pos = pd.Series(1.0, index=idx)
    cfg = dataclasses.replace(CONFIG, partial_take_r=1.0, trail_atr_mult=0.0)
    res = backtest_positions(df, pos, cfg, side="long")
    recs = res.trades.to_dict("records")
    _ok("a partial exit is recorded", any(t.get("reason") == "partial" for t in recs))
    cfg_off = dataclasses.replace(CONFIG, partial_take_r=0.0, trail_atr_mult=0.0)
    res_off = backtest_positions(df, pos, cfg_off, side="long")
    _ok("no partial when disabled", not any(t.get("reason") == "partial" for t in res_off.trades.to_dict("records")))


def test_manage_exits_mock():
    print("live exit manager (mock broker, flagged):")
    import dataclasses
    import paper
    from config import CONFIG

    class MockBroker:
        def __init__(self):
            self.replaced = []; self.partials = []
            self._positions = [{"symbol": "X", "qty": 10, "side": "long",
                                "avg_entry": 100.0, "price": 110.0, "unrealized_plpc": 10.0}]
            self._stops = {"X": 95.0}
            self.fail = False
        def positions_detail(self):
            return list(self._positions)
        def open_orders_for(self, sym):
            return [{"id": "stop-" + sym, "type": "stop", "stop_price": self._stops.get(sym)}]
        def replace_stop(self, order_id, new_stop):
            if self.fail:
                raise RuntimeError("broker down")
            self.replaced.append((order_id, round(new_stop, 2)))
        def partial_close(self, sym, qty, side):
            if self.fail:
                raise RuntimeError("broker down")
            self.partials.append((sym, qty, side))

    cfg = dataclasses.replace(CONFIG, manage_exits=True, partial_take_r=1.0, trail_atr_mult=3.0)
    log = [{"client_id": "sd-X", "symbol": "X", "direction": "LONG", "qty": 10,
            "entry_plan": 100.0, "stop": 95.0, "target": 130.0, "status": "open"}]
    mb = MockBroker()
    notes = paper.manage_open_positions(mb, cfg, log)
    _ok("manager returns a list of notes", isinstance(notes, list))
    _ok("partial taken once at >=1R (price 110, 1R=105)", len(mb.partials) == 1)
    paper.manage_open_positions(mb, cfg, log)
    _ok("partial is idempotent across a second run", len(mb.partials) == 1)
    _ok("log marks the partial as taken", log[0].get("partial_done") is True)
    _ok("stop was tightened (toward breakeven) after partial", len(mb.replaced) >= 1)
    mb.fail = True
    log2 = [{"client_id": "sd-Y", "symbol": "X", "direction": "LONG", "qty": 10,
             "entry_plan": 100.0, "stop": 95.0, "target": 130.0, "status": "open"}]
    notes2 = paper.manage_open_positions(mb, cfg, log2)
    _ok("manager never raises on broker failure", isinstance(notes2, list))


def test_pead():
    print("post-earnings drift:")
    import numpy as np
    import pandas as pd
    import strategies
    from config import CONFIG
    n = 60
    idx = pd.date_range("2024-01-01", periods=n)
    close = np.linspace(100, 102, n).astype(float)
    vol = np.full(n, 1_000_000.0)
    close[40] = close[39] * 1.08          # +8% reaction (earnings-like)
    vol[40] = 5_000_000.0                 # volume surge
    close[41:46] = close[40] * 1.01       # drift holds above the reaction close
    df = pd.DataFrame({"open": np.concatenate([[close[0]], close[:-1]]),
                       "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": vol}, index=idx)
    pos = strategies.pead_long(df, CONFIG)
    _ok("pead is a 0/1 series", set(pos.dropna().unique()).issubset({0.0, 1.0}) and len(pos) == n)
    _ok("pead long active during the held drift", bool(pos.iloc[44] == 1.0))
    _ok("pead long flat before the reaction", bool(pos.iloc[35] == 0.0))
    close2 = close.copy(); close2[47] = close[40] * 0.90
    df2 = df.copy(); df2["close"] = close2; df2["low"] = close2 * 0.99
    pos2 = strategies.pead_long(df2, CONFIG)
    _ok("pead long exits when the drift breaks down", bool(pos2.iloc[48] == 0.0))
    import dataclasses
    off = dataclasses.replace(CONFIG, pead_enabled=False)
    _ok("pead disabled => all flat", float(strategies.pead_long(df, off).sum()) == 0.0)
    closed = np.linspace(100, 98, n).astype(float)
    closed[40] = closed[39] * 0.92; closed[41:46] = closed[40] * 0.99
    vold = np.full(n, 1_000_000.0); vold[40] = 5_000_000.0
    dfd = pd.DataFrame({"open": np.concatenate([[closed[0]], closed[:-1]]),
                        "high": closed * 1.01, "low": closed * 0.99,
                        "close": closed, "volume": vold}, index=idx)
    _ok("pead short active during a held down-drift", bool(strategies.pead_short(dfd, CONFIG).iloc[44] == 1.0))


def test_pead_in_pipeline():
    print("pead in pipeline:")
    import strategies, analytics
    from data import synthetic_bars
    from config import CONFIG
    df = synthetic_bars("TEST", n=CONFIG.lookback_days)
    ev = strategies.evaluate(df, CONFIG)
    _ok("confluence total now counts PEAD", ev["total"] == len(strategies.STRATEGIES))
    _ok("PEAD registered long + short", "pead" in strategies.STRATEGIES and "pead_dn" in strategies.SHORT_STRATEGIES)
    se = analytics.strategy_edges(df, CONFIG)
    _ok("per-strategy edges include PEAD", "pead" in se["by"] and "pead_dn" in se["by"])
    _ok("PEAD edge tagged with a side", se["by"]["pead"]["side"] == "long" and se["by"]["pead_dn"]["side"] == "short")


def test_timing_engine():
    print("market timing (FTD + distribution):")
    import numpy as np, pandas as pd
    import timing
    def bars(c, v=None):
        c = np.asarray(c, float); v = np.asarray(v if v is not None else [1e6] * len(c), float)
        return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": v})
    # correction (>=3% drop, >=3 down days) then a day-4 FTD (+>=1.25% on higher volume)
    up = [100 + i for i in range(20)]
    corr = [119, 116, 113, 110, 108, 106]
    rally = [107, 108, 109, 111.5]
    v = [1e6] * len(up) + [1.1e6] * (len(corr) - 1) + [1e6, 1e6, 1e6, 2e6]
    ftd = timing._ftd_state(bars(up + corr[1:] + rally, v))
    _ok("FTD confirmed on a valid follow-through", ftd["state"] == "confirmed" and ftd["quality"] > 0)
    # a run of down-on-higher-volume days is a distribution cluster
    c = [100.0]; vv = [1e6]
    for _ in range(10):
        c.append(c[-1] * 0.995); vv.append(vv[-1] * 1.05)
    dd = timing._distribution_days(bars(c, vv))
    _ok("distribution cluster flagged as correction risk", dd["risk"] == "correction" and dd["count"] >= 6)
    _ok("state_at combines FTD + distribution", timing._single_state("confirmed", "correction") == "correction")
    quiet = [100 + i * 0.3 for i in range(60)]
    _ok("quiet uptrend is neutral", timing._ftd_state(bars(quiet))["state"] == "neutral")


def test_setups_screens():
    print("ported setups (burst / EP / parabolic short):")
    import numpy as np, pandas as pd
    import screens
    def b(o, h, l, c, v): return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    n = 30
    # momentum burst: tight base then +5% breakout on 2.4x volume closing near high
    c = [100 + np.sin(i) * 0.5 for i in range(n - 1)] + [105.0]
    o = [x for x in c[:-1]] + [100.2]; h = [x + 0.3 for x in c[:-1]] + [105.2]
    l = [x - 0.3 for x in c[:-1]] + [100.0]; v = [1e6] * (n - 1) + [2.4e6]
    burst = screens.momentum_burst(b(o, h, l, c, v))
    _ok("momentum burst fires on a valid breakout", burst["valid"] and burst["score"] >= 55)
    _q = [50 + i * .05 for i in range(n)]
    _ok("burst quiet name stays silent",
        not screens.momentum_burst(b(_q, _q, _q, _q, [1e6] * n))["valid"])
    # episodic pivot: quiet base then +8% gap on 4x volume; catalyst enrichment lifts the family
    c2 = [50 + np.sin(i) * .3 for i in range(n - 1)] + [54.0]
    o2 = [x for x in c2[:-1]] + [53.8]; h2 = [x + .2 for x in c2[:-1]] + [54.2]
    l2 = [x - .2 for x in c2[:-1]] + [53.6]; v2 = [8e5] * (n - 1) + [3.4e6]
    ep = screens.episodic_pivot(b(o2, h2, l2, c2, v2), has_news=False)
    _ok("EP fires technical-only", ep["valid"] and ep["family"] == "TECHNICAL_EP")
    enr = screens.reclassify_ep(ep, True, "Q3 earnings beat and guidance raise")
    _ok("EP reclassify lifts catalyst family + score",
        enr["family"] == "EARNINGS_EP" and enr["score"] >= ep["score"] and enr["base_score"] == ep["base_score"])
    _ok("EP reclassify leaves malformed input untouched", screens.reclassify_ep({"x": 1}, True, "y") == {"x": 1})
    # parabolic short: watch_only while climbing, actionable once it cracks
    cp = [20.0] * 20
    for g in (0.06, 0.07, 0.09, 0.12, 0.16):
        cp.append(cp[-1] * (1 + g))
    op = [x * .995 for x in cp]; hp = [x * 1.01 for x in cp]; lp = [x * .99 for x in cp]
    vp = [1e6] * 20 + [2e6, 2.4e6, 2.9e6, 3.5e6, 4.2e6]
    hp[-1] = cp[-1] * 1.005; lp[-1] = cp[-2]
    _ok("parabolic short is watch_only while still in markup",
        screens.parabolic_short(b(op, hp, lp, cp, vp))["state"] == "watch_only")
    peak = cp[-1]
    crack = screens.parabolic_short(b(op + [peak * 1.0], hp + [peak * 1.005], lp + [peak * .925],
                                      cp + [peak * .93], vp + [6e6]))
    _ok("parabolic short is actionable once it cracks", crack["valid"] and crack["state"] == "actionable")


def test_setups_walkforward():
    print("setup self-validation (direction-aware):")
    import numpy as np
    import setups_backtest as sb
    _ok("short agg counts down-moves as hits",
        sb._agg([-0.05, -0.03, 0.02, -0.04], short=True)["hit_rate"] == 75.0)
    _ok("long agg counts up-moves as hits",
        sb._agg([0.05, 0.03, -0.02, 0.04], short=False)["hit_rate"] == 75.0)
    bk = {k: {5: [], 10: [], 20: []} for k in ("baseline", "burst", "ep", "vcp", "pshort")}
    bk["baseline"][10] = [0.01] * 100
    bk["pshort"][10] = [-0.03, -0.02, -0.05, -0.04, -0.01, 0.01, -0.03] * 8   # mostly down = good short
    s = sb.summarize(bk, horizons=(5, 10, 20))
    ps = s["setups"]["pshort"]
    _ok("short setup tagged with direction", ps["direction"] == "short")
    _ok("short that falls > baseline shows a positive edge", ps["edge_pct"] > 0)


def test_setup_weighting_and_findings():
    print("setup self-weighting + analyst edge findings:")
    import dashboard
    w = dashboard._setup_check_weights({"setups": {
        "burst": {"edge_pct": 1.2, "n": 340}, "pshort": {"edge_pct": -0.9, "n": 80},
        "ep": {"edge_pct": 3.0, "n": 12}}})   # ep under-sampled -> excluded
    _ok("proven-edge setup weighted up", w["Momentum burst?"] > 1.0)
    _ok("lagging setup weighted down", w["Parabolic exhaustion?"] < 1.0)
    _ok("under-sampled setup excluded", "Episodic pivot?" not in w)
    _ok("weight clamps at 1.5", dashboard._setup_check_weights(
        {"setups": {"burst": {"edge_pct": 9.0, "n": 100}}})["Momentum burst?"] == 1.5)
    _ok("no study => no weights", dashboard._setup_check_weights(None) == {})
    import analyst
    orig = analyst._load_json
    analyst._load_json = lambda p: {
        "setups_study.json": {"primary_horizon": 10, "setups": {
            "burst": {"label": "Momentum Burst", "edge_pct": 1.4, "n": 300, "stats": {"10": {"hit_rate": 61}}},
            "pshort": {"label": "Parabolic Short", "edge_pct": -1.1, "n": 80, "stats": {"10": {"hit_rate": 40}}}}},
        "timing_study.json": {"horizons": [5, 10, 20], "states": {
            "confirmed": {"10": {"mean_pct": 2.1}}, "correction": {"10": {"mean_pct": -1.2}}}},
    }.get(p)
    try:
        f = {x["area"]: x["severity"] for x in analyst._edge_findings()}
    finally:
        analyst._load_json = orig
    _ok("analyst keeps a proven setup (info)", f["setup:burst"] == "info")
    _ok("analyst flags a lagging setup to act", f["setup:pshort"] == "act")
    _ok("analyst confirms a predictive timing gate", f["timing"] == "info")


def test_loss_streak_cooldown():
    print("losing-streak cooldown (risk engine):")
    import tempfile, json
    import portfolio_risk as pr
    tf = tempfile.mktemp(suffix=".json")
    json.dump([{"status": "loss", "advised_ts": f"2026-01-{i:02d}"} for i in range(1, 6)], open(tf, "w"))
    _ok("counts a trailing losing streak", pr._recent_loss_streak(tf) == 5)
    json.dump([{"status": "loss", "advised_ts": "2026-01-01"},
               {"status": "win", "advised_ts": "2026-01-02"},
               {"status": "loss", "advised_ts": "2026-01-03"}], open(tf, "w"))
    _ok("streak resets after a win", pr._recent_loss_streak(tf) == 1)
    _ok("missing file => no streak", pr._recent_loss_streak("/no/such/file.json") == 0)


def main():
    test_tradingview()
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
    test_pead()
    test_pead_in_pipeline()
    test_risk_multiplier()
    test_paper_sizing()
    test_time_stop()
    test_partial_take()
    test_manage_exits_mock()
    test_rs()
    test_conviction_weighting()
    test_rs_conviction()
    test_rs_in_scan()
    test_universe_cap()
    test_tracker_checks_snapshot()
    test_attribution()
    test_attribution_panel()
    test_timing_engine()
    test_setups_screens()
    test_setups_walkforward()
    test_setup_weighting_and_findings()
    test_loss_streak_cooldown()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
