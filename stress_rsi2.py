"""Stress-test the RSI-2 dip-buy strategy before trusting it. Run locally:
    python3 stress_rsi2.py

The bake-off flagged RSI-2 as the standout. This checks whether that's a real (if
small) effect or a fluke, three ways — all out-of-sample and cost-aware:

  1) PARAMETER SWEEP  — entry RSI threshold x exit SMA. A real effect is positive
     across most settings; a mirage works at only one lucky combo.
  2) COST SENSITIVITY — slippage 0 / 5 / 10 / 20 bps. If the edge dies by 10-20bps,
     it isn't tradeable.
  3) REGIME WINDOWS   — 3 consecutive time slices. A real effect shows up in most
     periods, not just one bull run.

If it holds across all three, it's worth promoting/validating further. If not, the
honest answer is 'interesting, not tradeable'.
"""
from __future__ import annotations

import copy
import statistics as st

import indicators as ind
from backtest import backtest_positions
from config import CONFIG
from data import get_bars, synthetic_bars
from scanner import CORE_WATCHLIST
from strategies import _state


def rsi2_pos(df, entry_th, exit_sma):
    c = df["close"]
    entry = (c > ind.sma(c, 200)) & (ind.rsi(c, 2) < entry_th)
    exit_ = c > ind.sma(c, exit_sma)
    return _state(entry, exit_)


def _agg(ms):
    if not ms:
        return {"ret": 0, "win": 0, "trades": 0, "prof": 0, "sharpe": 0}
    rets = [m["total_return"] * 100 for m in ms]
    wins = [m["win_rate"] * 100 for m in ms if m["n_trades"]]
    return {"ret": round(st.mean(rets), 2), "win": round(st.mean(wins), 1) if wins else 0,
            "trades": sum(m["n_trades"] for m in ms),
            "prof": round(100 * sum(1 for m in ms if m["total_return"] > 0) / len(ms)),
            "sharpe": round(st.mean([m["sharpe"] for m in ms]), 2)}


def _cfg(**kw):
    c = copy.copy(CONFIG)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _oos(bars, c, et, ex):
    ms = []
    for df in bars.values():
        pos = rsi2_pos(df, et, ex)
        split = int(len(df) * 0.6)
        try:
            ms.append(backtest_positions(df.iloc[split:], pos.iloc[split:], c).metrics)
        except Exception:  # noqa: BLE001
            pass
    return _agg(ms)


def main():
    live = bool(CONFIG.api_key and CONFIG.secret_key)
    bars = {}
    for s in dict.fromkeys(CORE_WATCHLIST):
        try:
            df = get_bars(s, CONFIG) if live else synthetic_bars(s, n=CONFIG.lookback_days)
            if df is not None and len(df) > 260:
                bars[s] = df
        except Exception:  # noqa: BLE001
            continue

    print(f"\nRSI-2 stress test — {len(bars)} symbols ({'LIVE Alpaca' if live else 'SYNTHETIC'} data)\n")

    print("1) PARAMETER SWEEP  (out-of-sample, 5bps cost)")
    print(f"   {'entry<':>7}{'exitSMA':>9}{'ret%':>8}{'win%':>7}{'prof%':>7}{'sharpe':>8}{'trades':>8}")
    for et in (5, 10, 15):
        for ex in (5, 10):
            a = _oos(bars, _cfg(slippage_bps=5), et, ex)
            print(f"   {et:>7}{ex:>9}{a['ret']:>8}{a['win']:>7}{a['prof']:>7}{a['sharpe']:>8}{a['trades']:>8}")

    print("\n2) COST SENSITIVITY  (entry<10, exit SMA5, OOS)")
    print(f"   {'slip_bps':>9}{'ret%':>8}{'win%':>7}{'prof%':>7}{'sharpe':>8}")
    for bps in (0, 5, 10, 20):
        a = _oos(bars, _cfg(slippage_bps=bps), 10, 5)
        print(f"   {bps:>9}{a['ret']:>8}{a['win']:>7}{a['prof']:>7}{a['sharpe']:>8}")

    print("\n3) REGIME WINDOWS  (entry<10, exit SMA5, 5bps) — 3 consecutive slices")
    print(f"   {'window':>8}{'ret%':>8}{'win%':>7}{'prof%':>7}{'sharpe':>8}{'trades':>8}")
    for i, (lo, hi) in enumerate([(0.0, 0.34), (0.34, 0.67), (0.67, 1.0)]):
        ms = []
        for df in bars.values():
            pos = rsi2_pos(df, 10, 5)
            n = len(df)
            try:
                ms.append(backtest_positions(df.iloc[int(n * lo):int(n * hi)],
                                             pos.iloc[int(n * lo):int(n * hi)],
                                             _cfg(slippage_bps=5)).metrics)
            except Exception:  # noqa: BLE001
                pass
        a = _agg(ms)
        print(f"   {'W' + str(i + 1):>8}{a['ret']:>8}{a['win']:>7}{a['prof']:>7}{a['sharpe']:>8}{a['trades']:>8}")

    print("\nVerdict guide: a real (if small) edge stays positive on ret + Sharpe across MOST")
    print("parameter combos, survives 10-20bps cost, and is positive in 2-3 of the 3 windows.")
    print("If it only works at one setting / zero cost / one window, it's noise — keep it as a screen.")


if __name__ == "__main__":
    main()
