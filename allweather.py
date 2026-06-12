"""Ray Dalio's All Weather portfolio — a static, risk-balanced asset allocation.

The idea (Bridgewater): you can't predict which economic environment is coming, so
hold a mix that does OK in all four — rising growth, falling growth, rising inflation,
falling inflation — balanced by *risk* rather than dollars. The canonical retail
version (popularised by Tony Robbins' "MONEY") is:

    30%  US stocks                 (VTI)  — does well when growth surprises up
    40%  long-term Treasuries      (TLT)  — does well when growth/inflation fall
    15%  intermediate Treasuries   (IEF)  — ballast, less rate-sensitive than TLT
    7.5% gold                      (GLD)  — hedges inflation + currency debasement
    7.5% broad commodities         (DBC)  — does well when inflation surprises up

It is NOT a trading signal — it's a buy-and-hold allocation you rebalance ~yearly.
The honest pitch: lower returns than 100% stocks in a bull run, but much shallower
drawdowns and steadier compounding. We verify that with a real backtest vs SPY below.

`build(live)` fetches ~10y of keyless Yahoo history, reports each ETF's latest price,
and backtests the (monthly-rebalanced) allocation against SPY buy-and-hold. Offline it
returns the allocation only (no backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# symbol -> (name, target weight %, role, which environment it covers)
ALLOCATION = [
    ("VTI", "Vanguard Total US Stock Market ETF", 30.0, "US stocks", "Rising growth"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", 40.0, "Long-term Treasuries", "Falling growth / deflation"),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", 15.0, "Intermediate Treasuries", "Ballast / falling rates"),
    ("GLD", "SPDR Gold Shares", 7.5, "Gold", "Rising inflation"),
    ("DBC", "Invesco DB Commodity Index ETF", 7.5, "Broad commodities", "Rising inflation / supply shocks"),
]


def _yahoo_closes(sym: str, rng: str = "10y"):
    """~10 years of daily closes from Yahoo (keyless)."""
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
    if len(rows) < 200:
        return None
    idx = pd.to_datetime([x[0] for x in rows], unit="s")
    return pd.Series([x[1] for x in rows], index=idx, name=sym)


def _metrics(eq: pd.Series) -> dict:
    s = eq.dropna()
    if len(s) < 6:
        return {"ret": 0.0, "cagr": 0.0, "sharpe": 0.0, "maxdd": 0.0}
    rets = s.pct_change().dropna()
    yrs = max(len(s) / 12.0, 1e-9)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else 0.0
    dd = ((s - s.cummax()) / s.cummax()).min()
    return {"ret": float((s.iloc[-1] / s.iloc[0] - 1) * 100), "cagr": float(cagr * 100),
            "sharpe": float(sharpe), "maxdd": float(dd * 100)}


def build(live: bool = True) -> dict:
    """Return the allocation (with latest prices when available) + a backtest vs SPY."""
    targets = [{"symbol": s, "name": n, "weight": w, "role": role, "env": env, "price": None}
               for (s, n, w, role, env) in ALLOCATION]
    out = {"targets": targets, "backtest": None, "data_src": None}
    if not live:
        out["data_src"] = "offline"
        return out

    series = {}
    for s, *_ in ALLOCATION:
        try:
            series[s] = _yahoo_closes(s)
        except Exception:  # noqa: BLE001
            series[s] = None
    try:
        spy = _yahoo_closes("SPY")
    except Exception:  # noqa: BLE001
        spy = None

    for t in targets:
        sv = series.get(t["symbol"])
        if sv is not None and len(sv):
            t["price"] = round(float(sv.iloc[-1]), 2)

    # Need every sleeve + SPY to run an honest backtest over a common window.
    if any(series.get(s) is None for s, *_ in ALLOCATION) or spy is None:
        out["data_src"] = "prices only (insufficient history for backtest)"
        return out

    cols = {s: series[s] for s, *_ in ALLOCATION}
    cols["SPY"] = spy
    monthly = pd.DataFrame(cols).resample("ME").last().dropna()
    if len(monthly) < 12:
        out["data_src"] = "prices only (insufficient overlap for backtest)"
        return out

    w = {s: wt / 100.0 for (s, _n, wt, *_r) in ALLOCATION}
    rets = monthly.pct_change().dropna()
    aw_ret = sum(rets[s] * w[s] for s in w)          # monthly rebalanced to target weights
    aw_eq = (1 + aw_ret).cumprod()
    spy_eq = (1 + rets["SPY"]).cumprod()

    out["backtest"] = {
        "allweather": _metrics(aw_eq),
        "spy": _metrics(spy_eq),
        "months": int(len(rets)),
        "years": round(len(rets) / 12.0, 1),
        "start": str(rets.index[0].date()),
        "end": str(rets.index[-1].date()),
    }
    out["data_src"] = "Yahoo ~10y"
    return out
