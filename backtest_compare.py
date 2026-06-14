"""A/B backtest harness for the strategy refinements.

Run locally (uses your Alpaca keys if set, else synthetic data for a sanity check):
    python3 backtest_compare.py

It backtests the core watchlist on the SAME historical data under four configs and
prints a comparison so you can see whether each refinement actually helps before
trusting it live:
    baseline       — current rules, fixed stop
    adx_gate(20)   — only take longs when ADX >= 20 (trend-strength filter)
    trail_atr(3)   — trailing 3x-ATR stop ("let winners run") instead of fixed
    both           — ADX gate + trailing stop together

Look at win%, avg return, avg drawdown and Sharpe across the universe. Higher
win%/return and a shallower (less negative) drawdown is better.
"""
from __future__ import annotations

import copy
import statistics as st

from backtest import run_backtest
from config import CONFIG
from data import get_bars, synthetic_bars
from scanner import CORE_WATCHLIST

VARIANTS = {
    "baseline":      {"adx_min": 0.0,  "trail_atr_mult": 0.0},
    "adx_gate(20)":  {"adx_min": 20.0, "trail_atr_mult": 0.0},
    "trail_atr(3)":  {"adx_min": 0.0,  "trail_atr_mult": 3.0},
    "both":          {"adx_min": 20.0, "trail_atr_mult": 3.0},
    "partial+trail": {"adx_min": 0.0,  "trail_atr_mult": 3.0, "partial_take_r": 1.0},
    "time_stop(10)": {"adx_min": 0.0,  "trail_atr_mult": 0.0, "max_hold_days": 10},
}


def _cfg(overrides: dict):
    c = copy.copy(CONFIG)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def main():
    live = bool(CONFIG.api_key and CONFIG.secret_key)
    syms = list(dict.fromkeys(CORE_WATCHLIST))[:40]
    bars = {}
    for s in syms:
        try:
            df = get_bars(s, CONFIG) if live else synthetic_bars(s, n=CONFIG.lookback_days)
            if df is not None and len(df) > CONFIG.slow_ma + 5:
                bars[s] = df
        except Exception:  # noqa: BLE001
            continue

    print(f"\nBacktest comparison over {len(bars)} symbols "
          f"({'LIVE Alpaca' if live else 'SYNTHETIC'} data)\n")
    print(f"{'variant':14} {'trades':>7} {'win%':>6} {'avgRet%':>8} {'avgDD%':>7} {'sharpe':>7}")
    print("-" * 52)
    for name, ov in VARIANTS.items():
        cfg = _cfg(ov)
        wr, rets, dds, shp, ntr = [], [], [], [], 0
        for df in bars.values():
            try:
                m = run_backtest(df, cfg).metrics
            except Exception:  # noqa: BLE001
                continue
            if m["n_trades"]:
                wr.append(m["win_rate"] * 100)
                ntr += m["n_trades"]
            rets.append(m["total_return"] * 100)
            dds.append(m["max_drawdown"] * 100)
            shp.append(m["sharpe"])

        def avg(x):
            return round(st.mean(x), 1) if x else 0.0
        print(f"{name:14} {ntr:>7} {avg(wr):>6} {avg(rets):>8} {avg(dds):>7} "
              f"{round(st.mean(shp), 2) if shp else 0:>7}")
    print("\nHigher win%/return + shallower drawdown = better. Tune adx_min / "
          "trail_atr_mult in config.py to the winner, then rebuild.")


if __name__ == "__main__":
    main()
