"""Backtest + STRESS TEST every strategy in the panel, on real history, under the LIVE exit model.

Why this exists: the per-strategy numbers on the dashboard come from each strategy's own entry/exit
rules. But the engine no longer exits that way — since the Aug 2026 swing rebuild it takes a partial
at 2R, then trails an ATR stop and rides past the target. This walks each strategy's real entries
through *that* exit model, so you see what the strategy is actually worth AS TRADED.

It also stress-tests, which a single average hides:
  · regime split      — does the edge survive when the market ISN'T going up?
  · worst drawdown    — the deepest peak-to-trough on the equity curve of its trades
  · worst losing run  — the longest string of losses you'd have had to sit through
  · tail risk         — the worst single trade, and the 5th-percentile trade
  · robustness        — the same test re-run with the trail 25% tighter / looser, to check the edge
                        isn't an artifact of one lucky parameter

Run (needs market-data keys in .env — it uses the same data path as the engine):
    python3 strategy_backtest.py                 # full panel, default universe
    python3 strategy_backtest.py --quick         # fewer names, faster
    python3 strategy_backtest.py --json          # also write strategy_study.json

Pure/offline core (`simulate`) so the exit math unit-tests on synthetic data without fetching.
"""
from __future__ import annotations

import json
import statistics as st
import sys

import numpy as np

STUDY_PATH = "strategy_study.json"


# ---------------------------------------------------------------- core (offline, testable)
def simulate(df, entries, cfg, *, trail_scale: float = 1.0, hold_cap: int = 60) -> list[dict]:
    """Walk a strategy's entry signals through the LIVE exit model and return one record per trade.

    Exit model (mirrors tracker.py): initial ATR stop → book `partial_exit_frac` at `partial_exit_r`
    and move to breakeven → trail `swing_trail_atr` x ATR under the high-water mark, tightening to
    `swing_trail_tight_atr` once the base target is exceeded. No look-ahead: every decision on bar i
    uses only data up to bar i.
    """
    import indicators as ind
    c, hi_s, lo_s = df["close"], df["high"], df["low"]
    atr_s = ind.atr(df, getattr(cfg, "atr_period", 14))
    stop_mult = getattr(cfg, "atr_stop_mult", 2.0)
    tp_cap = getattr(cfg, "take_profit_pct", 0.30)
    p_on = getattr(cfg, "partial_exit_enabled", True)
    p_r = getattr(cfg, "partial_exit_r", 2.0)
    p_f = getattr(cfg, "partial_exit_frac", 0.5)
    t_on = getattr(cfg, "swing_trail_enabled", True)
    t_m = getattr(cfg, "swing_trail_atr", 3.0) * trail_scale
    t_t = getattr(cfg, "swing_trail_tight_atr", 1.8) * trail_scale
    t_act = getattr(cfg, "swing_trail_activate_r", 1.0)

    out: list[dict] = []
    n = len(df)
    for i in np.flatnonzero(np.asarray(entries)):
        i = int(i)
        if i + 2 >= n:
            continue
        a = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else None
        if not a or a <= 0:
            continue
        entry = float(c.iloc[i])
        stop = entry - stop_mult * a
        risk = entry - stop
        if risk <= 0:
            continue
        target = entry * (1 + tp_cap)
        t1 = entry + p_r * risk if p_on else None
        eff, hw, t1_hit, tex = stop, entry, False, False
        exit_px, bars = None, 0
        for j in range(i + 1, min(n, i + 1 + hold_cap)):
            h, l = float(hi_s.iloc[j]), float(lo_s.iloc[j])
            bars = j - i
            if l <= eff:                                   # stop (conservative: checked first)
                exit_px = eff
                break
            if t1 and not t1_hit and h >= t1:
                t1_hit = True
                eff = max(eff, entry)                      # breakeven on the remainder
            if t_on:
                hw = max(hw, h)
                if h >= target:
                    tex = True
                m = t_t if tex else t_m
                if hw >= entry + t_act * risk:
                    eff = max(eff, hw - m * a)
            elif h >= target:
                exit_px = target
                break
        if exit_px is None:
            exit_px = float(c.iloc[min(n - 1, i + hold_cap)])
        gain = exit_px - entry
        if t1_hit and t1:
            gain = p_f * (t1 - entry) + (1 - p_f) * gain
        out.append({"ret": gain / entry * 100, "bars": bars, "t1": bool(t1_hit),
                    "i": i, "date": str(df.index[i].date())})
    return out


def _entries_from_positions(pos) -> np.ndarray:
    """A strategy returns a 0/1 position series; entries are the 0→1 transitions."""
    p = np.asarray(pos, dtype=float)
    prev = np.concatenate(([0.0], p[:-1]))
    return (p == 1) & (prev == 0)


# ---------------------------------------------------------------- stress metrics
def _max_drawdown(rets: list[float]) -> float:
    """Deepest peak-to-trough on the compounding equity curve of these trades, in %."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        eq *= (1 + r / 100.0)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return mdd * 100


def _worst_streak(rets: list[float]) -> int:
    worst = run = 0
    for r in rets:
        run = run + 1 if r <= 0 else 0
        worst = max(worst, run)
    return worst


def stats(trades: list[dict]) -> dict:
    rets = [t["ret"] for t in trades]
    if not rets:
        return {"n": 0}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    aw = st.mean(wins) if wins else 0.0
    al = abs(st.mean(losses)) if losses else 0.0
    return {
        "n": len(rets),
        "win_rate": round(100 * len(wins) / len(rets), 1),
        "expectancy": round(st.mean(rets), 3),
        "avg_win": round(aw, 2),
        "avg_loss": round(-al, 2),
        "payoff": round(aw / al, 2) if al else None,
        "total_ret": round(sum(rets), 1),
        "max_dd": round(_max_drawdown(rets), 1),
        "worst_trade": round(min(rets), 2),
        "p05": round(float(np.percentile(rets, 5)), 2),
        "best_trade": round(max(rets), 2),
        "worst_losing_streak": _worst_streak(rets),
        "avg_bars_held": round(st.mean([t["bars"] for t in trades]), 1),
        "partial_hit_pct": round(100 * sum(1 for t in trades if t["t1"]) / len(trades), 1),
    }


def _regime_split(df, trades: list[dict]) -> dict:
    """Split trades by whether the BROADER trend was up at entry (price > 200-day) — a cheap proxy
    for 'was the market helping?'. An edge that only exists in the up-regime is fragile."""
    import indicators as ind
    s200 = ind.sma(df["close"], 200)
    up, dn = [], []
    for t in trades:
        i = t["i"]
        v = s200.iloc[i]
        if np.isnan(v):
            continue
        (up if float(df["close"].iloc[i]) > float(v) else dn).append(t)
    return {"uptrend": stats(up), "not_uptrend": stats(dn)}


# ---------------------------------------------------------------- driver
def study(cfg, symbols=None, path: str | None = None, quick: bool = False) -> dict:
    import scanner
    import strategies as S
    from data import get_bars

    syms = symbols or (scanner.CORE_WATCHLIST[:12] if quick
                       else list(dict.fromkeys(scanner.CORE_WATCHLIST + scanner.WIDE_POOL))[:60])
    frames = {}
    for s in syms:
        try:
            df = get_bars(s, cfg)
            if df is not None and len(df) > 260:
                frames[s] = df
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        print("[strategy_backtest] no price data — are ALPACA keys set in .env?")
        return {}
    print(f"[strategy_backtest] {len(frames)} symbols, {len(S.STRATEGIES)} strategies, live exit model\n")

    results = {}
    for key, (label, fn, kind, _blurb) in S.STRATEGIES.items():
        allt, per_regime_up, per_regime_dn = [], [], []
        tight, loose = [], []
        for sym, df in frames.items():
            try:
                pos = fn(df, cfg)
                ent = _entries_from_positions(pos)
                if not ent.any():
                    continue
                tr = simulate(df, ent, cfg)
                allt += tr
                rs = _regime_split(df, tr)
                per_regime_up.append(rs["uptrend"])
                per_regime_dn.append(rs["not_uptrend"])
                tight += simulate(df, ent, cfg, trail_scale=0.75)
                loose += simulate(df, ent, cfg, trail_scale=1.25)
            except Exception:  # noqa: BLE001
                continue
        if not allt:
            continue
        base = stats(allt)
        results[key] = {
            "label": label, "kind": kind, "base": base,
            "regime": {"uptrend": _merge(per_regime_up), "not_uptrend": _merge(per_regime_dn)},
            "robustness": {"trail_tight_75pct": stats(tight), "trail_loose_125pct": stats(loose)},
        }
    out = {"generated": _now(), "n_symbols": len(frames), "strategies": results,
           "exit_model": {"partial_r": getattr(cfg, "partial_exit_r", 2.0),
                          "partial_frac": getattr(cfg, "partial_exit_frac", 0.5),
                          "trail_atr": getattr(cfg, "swing_trail_atr", 3.0),
                          "trail_tight": getattr(cfg, "swing_trail_tight_atr", 1.8),
                          "target_cap": getattr(cfg, "take_profit_pct", 0.30)}}
    if path:
        json.dump(out, open(path, "w"), indent=1)
        print(f"[wrote {path}]")
    return out


def _merge(list_of_stats: list[dict]) -> dict:
    """Weighted-merge per-symbol stat dicts into one (approximate; weights by n)."""
    tot = sum(s.get("n", 0) for s in list_of_stats)
    if not tot:
        return {"n": 0}
    def w(k):
        vals = [(s.get(k), s.get("n", 0)) for s in list_of_stats if s.get(k) is not None and s.get("n")]
        return round(sum(v * n for v, n in vals) / sum(n for _, n in vals), 2) if vals else None
    return {"n": tot, "win_rate": w("win_rate"), "expectancy": w("expectancy"), "payoff": w("payoff")}


def report(out: dict) -> str:
    if not out.get("strategies"):
        return "no results"
    em = out["exit_model"]
    L = [f"STRATEGY BACKTEST — as traded (partial {em['partial_r']:g}R, trail {em['trail_atr']:g}x ATR, "
         f"cap {em['target_cap']:.0%}) · {out['n_symbols']} symbols", "=" * 104,
         f"{'strategy':24} {'kind':15} {'n':>5} {'win%':>6} {'exp%':>7} {'payoff':>7} {'maxDD%':>8} "
         f"{'wrstRun':>8} {'held':>6}", "-" * 104]
    rows = sorted(out["strategies"].items(), key=lambda kv: -(kv[1]["base"].get("expectancy") or -99))
    for _k, r in rows:
        b = r["base"]
        L.append(f"{r['label'][:24]:24} {r['kind'][:15]:15} {b['n']:5} {b['win_rate']:6.1f} "
                 f"{b['expectancy']:+7.2f} {str(b['payoff'] or '—'):>7} {b['max_dd']:8.1f} "
                 f"{b['worst_losing_streak']:8} {b['avg_bars_held']:6.1f}")
    L += ["", "STRESS — does the edge survive a weaker tape, and is it parameter-robust?", "-" * 104,
          f"{'strategy':24} {'exp uptrend':>13} {'exp NOT-up':>12} {'trail -25%':>12} {'trail +25%':>12} {'verdict':>22}"]
    for _k, r in rows:
        up = r["regime"]["uptrend"].get("expectancy")
        dn = r["regime"]["not_uptrend"].get("expectancy")
        ti = r["robustness"]["trail_tight_75pct"].get("expectancy")
        lo = r["robustness"]["trail_loose_125pct"].get("expectancy")
        fine = [x for x in (r["base"].get("expectancy"), ti, lo) if x is not None]
        robust = all(x > 0 for x in fine) if fine else False
        both = (dn is not None and dn > 0)
        verdict = ("robust · all regimes" if (robust and both)
                   else "robust · needs uptrend" if robust
                   else "fragile — param-sensitive")
        f = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) else "—"
        L.append(f"{r['label'][:24]:24} {f(up):>13} {f(dn):>12} {f(ti):>12} {f(lo):>12} {verdict:>22}")
    L += ["", "maxDD = deepest peak-to-trough on that strategy's own trade equity curve.",
          "wrstRun = longest consecutive losing streak. held = avg bars in the trade.",
          "'fragile' = expectancy flips negative when the trail is nudged — treat with suspicion."]
    return "\n".join(L)


def _now() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT")


if __name__ == "__main__":
    from config import CONFIG
    quick = "--quick" in sys.argv
    out = study(CONFIG, path=(STUDY_PATH if "--json" in sys.argv else None), quick=quick)
    if out:
        print(report(out))
