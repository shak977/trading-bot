"""Honest strategy bake-off with costs + walk-forward validation.

Run locally (uses Alpaca keys if set, else synthetic data for a sanity check):
    python3 edge_hunt.py

For each of the 7 strategies it backtests across the watchlist **net of slippage +
commission**, and splits every stock's history into:
    in-sample  (IS)  — the older 60% of bars
    out-of-sample (OOS) — the newer 40%, never used to choose anything

OOS is the honest forward estimate. The overfit tell: a strategy that looks great
IS but falls apart OOS is curve-fit, not real. Trust the OOS columns.
"""
from __future__ import annotations

import statistics as st

import strategies
from backtest import backtest_positions
from config import CONFIG
from data import get_bars, synthetic_bars
from scanner import CORE_WATCHLIST


def _agg(ms: list[dict]) -> dict:
    if not ms:
        return {"ret": 0, "win": 0, "trades": 0, "prof": 0, "sharpe": 0}
    rets = [m["total_return"] * 100 for m in ms]
    wins = [m["win_rate"] * 100 for m in ms if m["n_trades"]]
    return {
        "ret": round(st.mean(rets), 2),
        "win": round(st.mean(wins), 1) if wins else 0,
        "trades": sum(m["n_trades"] for m in ms),
        "prof": round(100 * sum(1 for m in ms if m["total_return"] > 0) / len(ms)),
        "sharpe": round(st.mean([m["sharpe"] for m in ms]), 2),
    }


def main():
    live = bool(CONFIG.api_key and CONFIG.secret_key)
    syms = list(dict.fromkeys(CORE_WATCHLIST))[:40]
    bars = {}
    for s in syms:
        try:
            df = get_bars(s, CONFIG) if live else synthetic_bars(s, n=CONFIG.lookback_days)
            if df is not None and len(df) > CONFIG.slow_ma + 60:
                bars[s] = df
        except Exception:  # noqa: BLE001
            continue

    print(f"\nStrategy bake-off — net of {CONFIG.slippage_bps:.0f}bps/side slippage + "
          f"${CONFIG.commission_per_trade:.0f} commission — over {len(bars)} symbols "
          f"({'LIVE Alpaca' if live else 'SYNTHETIC'} data)\n")
    print(f"{'strategy':20} {'OOSret%':>8} {'OOSwin%':>8} {'OOStr':>6} {'OOSprof%':>9} {'OOSshrp':>8}   {'ISret%':>7}")
    print("-" * 80)
    results = []
    for key, (label, fn, kind, _blurb) in strategies.STRATEGIES.items():
        is_m, oos_m = [], []
        for df in bars.values():
            try:
                pos = fn(df, CONFIG)
            except Exception:  # noqa: BLE001
                continue
            split = int(len(df) * 0.6)
            try:
                is_m.append(backtest_positions(df.iloc[:split], pos.iloc[:split], CONFIG).metrics)
                oos_m.append(backtest_positions(df.iloc[split:], pos.iloc[split:], CONFIG).metrics)
            except Exception:  # noqa: BLE001
                continue
        oos, ins = _agg(oos_m), _agg(is_m)
        results.append((oos["ret"], label, oos, ins))

    for _ret, label, oos, ins in sorted(results, reverse=True):
        print(f"{label:20} {oos['ret']:>8} {oos['win']:>8} {oos['trades']:>6} "
              f"{oos['prof']:>9} {oos['sharpe']:>8}   {ins['ret']:>7}")
    print("\nOOS = out-of-sample (newer 40%, never used to pick). Trust OOS, not IS.")
    print("Real edge = positive OOS return, OOS roughly tracks IS (not collapsing), and "
          ">50% of stocks profitable. If nothing clears that bar, the honest read is "
          "'use it as a screen, not an edge.'")


if __name__ == "__main__":
    main()
