"""Backtest the dual-momentum portfolio vs simply buying SPY. Run locally:
    python3 momentum_lab.py

Monthly rebalance: rank the universe by 12-1 momentum, keep names above their
200-day MA with positive momentum, hold the top K equal-weight, repeat. Reports
total return / CAGR / Sharpe / max-drawdown against SPY buy-and-hold over the
same window, net of a rough rebalance cost.

The honest bar: beating SPY on **risk-adjusted** terms (Sharpe + shallower
drawdown) over a full window — not just raw return in a bull run. Momentum wins
in trends and crashes in reversals; expect both.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import momentum as mom
from config import CONFIG
from data import get_bars, synthetic_bars
from scanner import CORE_WATCHLIST

TOP_K = 10
LOOKBACK, SKIP = 252, 21


def _load():
    live = bool(CONFIG.api_key and CONFIG.secret_key)
    cfg = copy.copy(CONFIG)
    cfg.lookback_days = 1100  # ~4.3y so there are enough monthly rebalances to mean something
    bars = {}
    for s in dict.fromkeys(list(CORE_WATCHLIST) + ["SPY"]):
        try:
            df = get_bars(s, cfg) if live else synthetic_bars(s, n=cfg.lookback_days)
            if df is not None and len(df) > LOOKBACK + 40:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
                bars[s] = df
        except Exception:  # noqa: BLE001
            continue
    return bars, live


def _metrics(eq: list) -> dict:
    s = pd.Series(eq).dropna()
    if len(s) < 3:
        return {"ret": 0, "cagr": 0, "sharpe": 0, "maxdd": 0}
    rets = s.pct_change().dropna()
    yrs = max(len(s) / 12.0, 1e-9)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0.0
    dd = ((s - s.cummax()) / s.cummax()).min()
    return {"ret": (s.iloc[-1] / s.iloc[0] - 1) * 100, "cagr": cagr * 100,
            "sharpe": sharpe, "maxdd": dd * 100}


def main():
    bars, live = _load()
    syms = [s for s in bars if s != "SPY"]
    if not syms:
        print("No data."); return
    monthly = pd.DataFrame({s: df["close"] for s, df in bars.items()}).resample("ME").last()
    months = monthly.index
    start = LOOKBACK // 21 + 1
    slip = CONFIG.slippage_bps / 1e4
    port, spy = [1.0], [1.0]

    for i in range(start, len(months) - 1):
        asof = months[i]
        upto = {s: bars[s][bars[s].index <= asof] for s in syms
                if len(bars[s][bars[s].index <= asof]) > LOOKBACK}
        picks = [s for s, _ in mom.rank(upto, LOOKBACK, SKIP)[:TOP_K]]
        r = []
        for s in picks:
            p0, p1 = monthly[s].iloc[i], monthly[s].iloc[i + 1]
            if p0 and p1 and not np.isnan(p0) and not np.isnan(p1):
                r.append(p1 / p0 - 1)
        mret = (float(np.mean(r)) if r else 0.0) - 2 * slip  # rough rebalance cost
        port.append(port[-1] * (1 + mret))
        sp0, sp1 = monthly.get("SPY", pd.Series()).iloc[i] if "SPY" in monthly else np.nan, \
            monthly.get("SPY", pd.Series()).iloc[i + 1] if "SPY" in monthly else np.nan
        sret = (sp1 / sp0 - 1) if (sp0 and sp1 and not np.isnan(sp0) and not np.isnan(sp1)) else 0.0
        spy.append(spy[-1] * (1 + sret))

    pm, sm = _metrics(port), _metrics(spy)
    print(f"\nDual-momentum portfolio (top {TOP_K}, monthly) vs SPY — {len(syms)} names, "
          f"{len(port)} months, {'LIVE Alpaca' if live else 'SYNTHETIC'} data\n")
    print(f"{'':16}{'totRet%':>9}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}")
    print(f"{'Dual-momentum':16}{pm['ret']:>9.1f}{pm['cagr']:>8.1f}{pm['sharpe']:>8.2f}{pm['maxdd']:>8.1f}")
    print(f"{'SPY buy & hold':16}{sm['ret']:>9.1f}{sm['cagr']:>8.1f}{sm['sharpe']:>8.2f}{sm['maxdd']:>8.1f}")
    print("\nBar to clear: higher Sharpe AND shallower (less negative) MaxDD than SPY.")
    print("If it only beats SPY on raw return with a deeper drawdown, it's just more risk, not edge.")


if __name__ == "__main__":
    main()
