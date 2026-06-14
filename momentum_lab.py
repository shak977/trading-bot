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
from scanner import CORE_WATCHLIST, sector_of

TOP_K = 10
LOOKBACK, SKIP = 252, 21


def _yahoo_closes(sym: str, rng: str = "10y"):
    """~10 years of daily closes from Yahoo (keyless) — long enough to span the
    2020 crash and 2022 bear, so momentum gets tested through a real downturn."""
    import requests
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range={rng}&interval=1d")
    r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return None
    res = ((r.json().get("chart", {}) or {}).get("result") or [None])[0]
    if not res:
        return None
    ts = res.get("timestamp") or []
    q = (res.get("indicators", {}).get("quote") or [{}])[0]
    cl = q.get("close") or []
    rows = [(ts[i], cl[i]) for i in range(min(len(ts), len(cl))) if cl[i] is not None]
    if len(rows) < 300:
        return None
    idx = pd.to_datetime([r[0] for r in rows], unit="s")
    return pd.DataFrame({"close": [r[1] for r in rows]}, index=idx)


def _load():
    """Prefer ~10y of keyless Yahoo history (spans 2020 + 2022 downturns); fall
    back to synthetic when offline."""
    bars, src = {}, "Yahoo ~10y"
    for s in dict.fromkeys(list(CORE_WATCHLIST) + ["SPY"]):
        df = None
        try:
            df = _yahoo_closes(s)
        except Exception:  # noqa: BLE001
            df = None
        if df is None:
            try:
                d = synthetic_bars(s, n=1100)
                df = d[["close"]].copy()
                df.index = pd.to_datetime(df.index)
                src = "SYNTHETIC"
            except Exception:  # noqa: BLE001
                df = None
        if df is not None and len(df) > LOOKBACK + 40:
            bars[s] = df
    return bars, src


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


# Survivorship-bias-free universe: broad ETFs that existed for the whole window and
# never delist — sector SPDRs (9 originals, present since 1998) + asset classes. The
# strategy can't be flattered by hindsight stock-picking because the universe is fixed
# and contains every member, winners and losers alike.
ETF_UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
                "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "IYR"]


def _run(bars: dict, cap_by_sector: bool = True) -> dict | None:
    """Monthly dual-momentum backtest over `bars` (must include 'SPY'). Returns metrics for
    the portfolio and SPY, or None if there isn't enough data."""
    syms = [s for s in bars if s != "SPY"]
    if not syms or "SPY" not in bars:
        return None
    monthly = pd.DataFrame({s: df["close"] for s, df in bars.items()}).resample("ME").last()
    months = monthly.index
    start = LOOKBACK // 21 + 1
    if len(months) <= start + 6:
        return None
    slip = CONFIG.slippage_bps / 1e4
    port, spy = [1.0], [1.0]
    for i in range(start, len(months) - 1):
        asof = months[i]
        upto = {s: bars[s][bars[s].index <= asof] for s in syms
                if len(bars[s][bars[s].index <= asof]) > LOOKBACK}
        ranked = mom.rank(upto, LOOKBACK, SKIP)
        picks, cnt = [], {}
        for s, _ in ranked:
            if cap_by_sector:
                sec = sector_of(s)
                if cnt.get(sec, 0) >= 3:
                    continue
                cnt[sec] = cnt.get(sec, 0) + 1
            picks.append(s)
            if len(picks) >= TOP_K:
                break
        ws = {}
        for s in picks:
            v = float(upto[s]["close"].pct_change().dropna().tail(21).std())
            ws[s] = (1.0 / v) if v > 1e-9 else 0.0
        tw = sum(ws.values()) or 1.0
        mret = -2 * slip
        for s in picks:
            p0, p1 = monthly[s].iloc[i], monthly[s].iloc[i + 1]
            if p0 and p1 and not np.isnan(p0) and not np.isnan(p1):
                mret += (ws[s] / tw) * (p1 / p0 - 1)
        port.append(port[-1] * (1 + mret))
        sp0, sp1 = monthly["SPY"].iloc[i], monthly["SPY"].iloc[i + 1]
        sret = (sp1 / sp0 - 1) if (sp0 and sp1 and not np.isnan(sp0) and not np.isnan(sp1)) else 0.0
        spy.append(spy[-1] * (1 + sret))
    pm, sm = _metrics(port), _metrics(spy)
    return {"strategy": pm, "spy": sm, "months": len(port), "n_universe": len(syms)}


def build(live: bool = True) -> dict | None:
    """Survivorship-bias-FREE momentum backtest for the dashboard: runs the same dual-momentum
    rules on a fixed ETF universe (no hindsight stock selection). Returns metrics vs SPY, or
    None if history can't be fetched. Mirrors allweather.build()."""
    try:
        bars = {}
        for s in dict.fromkeys(ETF_UNIVERSE + ["SPY"]):
            df = None
            try:
                df = _yahoo_closes(s)
            except Exception:  # noqa: BLE001
                df = None
            if df is not None and len(df) > LOOKBACK + 40:
                bars[s] = df
        if len(bars) < 6 or "SPY" not in bars:
            return None
        res = _run(bars, cap_by_sector=False)
        if not res:
            return None
        res["survivorship_free"] = True
        res["universe"] = [s for s in ETF_UNIVERSE if s in bars]
        return res
    except Exception:  # noqa: BLE001
        return None


def main():
    bars, src = _load()
    res = _run(bars, cap_by_sector=True)
    if not res:
        print("No data."); return
    pm, sm = res["strategy"], res["spy"]
    print(f"\nDual-momentum portfolio (top {TOP_K}, monthly) vs SPY — {res['n_universe']} names, "
          f"{res['months']} months, {src} data")
    print("NOTE: this stock universe is today's CORE_WATCHLIST (survivorship-biased — optimistic).")
    print("      For an honest read, see momentum_lab.build() (fixed ETF universe).\n")
    print(f"{'':16}{'totRet%':>9}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>8}")
    print(f"{'Dual-momentum':16}{pm['ret']:>9.1f}{pm['cagr']:>8.1f}{pm['sharpe']:>8.2f}{pm['maxdd']:>8.1f}")
    print(f"{'SPY buy & hold':16}{sm['ret']:>9.1f}{sm['cagr']:>8.1f}{sm['sharpe']:>8.2f}{sm['maxdd']:>8.1f}")
    print("\nBar to clear: higher Sharpe AND shallower (less negative) MaxDD than SPY.")
    print("If it only beats SPY on raw return with a deeper drawdown, it's just more risk, not edge.")


if __name__ == "__main__":
    main()
