"""Ported screening methodologies — automatable versions of the tradermonty/claude-trading-skills
methods, rewritten to run headless inside the pipeline off the OHLCV we already fetch.

First method: Mark Minervini's VCP (Volatility Contraction Pattern), faithful to the vendored
vcp-screener skill:
  * trend_template() — the 7-point Stage-2 uptrend gate (a stock must be in a confirmed uptrend
    before a base even counts).
  * vcp() — successive contractions, each tighter than the last, coiling under a breakout pivot,
    with volatility (ATR) compression.
  * vcp_setup() — combines the two into a single long-only read the conviction engine can score.

Works on a chronological OHLCV DataFrame (oldest→newest) with columns open/high/low/close/volume,
exactly what data.get_bars returns. Pure/no-network; never raises (returns a not-valid dict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(closes: np.ndarray, w: int) -> float | None:
    if len(closes) < w:
        return None
    return float(np.mean(closes[-w:]))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float | None:
    n = len(close)
    if n < period + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])
    if len(tr) < period:
        return None
    return float(np.mean(tr[-period:]))


def trend_template(df: pd.DataFrame, rs_pct: float | None = None) -> dict:
    """Minervini's 7-point Stage-2 trend template. Returns {score 0-100, passed, criteria, n}.

    1) price > SMA150 and > SMA200; 2) SMA150 > SMA200; 3) SMA200 rising for ~22 days;
    4) price > SMA50; 5) >=25% above the 52-week low; 6) within 25% of the 52-week high;
    7) relative-strength percentile > 70 (only scored when rs_pct is supplied)."""
    if df is None or len(df) < 50:
        return {"score": 0, "passed": False, "criteria": {}, "n": 0}
    close = df["close"].to_numpy(float)
    price = float(close[-1])
    win = close[-252:] if len(close) >= 252 else close
    year_high, year_low = float(np.max(win)), float(np.min(win))

    sma50, sma150, sma200 = _sma(close, 50), _sma(close, 150), _sma(close, 200)
    crit: dict[str, bool] = {}
    if sma150 is not None and sma200 is not None:
        crit["price_above_150_200"] = price > sma150 and price > sma200
        crit["sma150_above_200"] = sma150 > sma200
    if len(close) >= 222 and sma200 is not None:
        sma200_22d = _sma(close[:-22], 200)
        if sma200_22d is not None:
            crit["sma200_rising"] = sma200 > sma200_22d
    if sma50 is not None:
        crit["price_above_50"] = price > sma50
    if year_low > 0:
        crit["25pct_above_low"] = (price - year_low) / year_low * 100 >= 25
    if year_high > 0:
        crit["within_25pct_high"] = price >= year_high * 0.75
    if rs_pct is not None:
        crit["rs_above_70"] = rs_pct > 70

    evaluated = len(crit)
    passed_n = sum(1 for v in crit.values() if v)
    score = round(passed_n / evaluated * 100) if evaluated else 0
    # Minervini pass: essentially all criteria (allow one miss on a full 6-7 set)
    passed = evaluated >= 5 and passed_n >= evaluated - 1
    return {"score": score, "passed": passed, "criteria": crit, "n": evaluated,
            "sma50": sma50, "sma150": sma150, "sma200": sma200,
            "year_high": year_high, "year_low": year_low}


def _swings(high: np.ndarray, low: np.ndarray, window: int = 5):
    """Local swing highs/lows: a bar that is the max/min of a ±window neighbourhood.
    Returns (highs, lows) as lists of (idx, price), chronological."""
    n = len(high)
    hs, ls = [], []
    for i in range(window, n - window):
        seg_h, seg_l = high[i - window:i + window + 1], low[i - window:i + window + 1]
        if high[i] == seg_h.max():
            hs.append((i, float(high[i])))
        if low[i] == seg_l.min():
            ls.append((i, float(low[i])))
    return hs, ls


def vcp(df: pd.DataFrame, lookback: int = 120, min_contractions: int = 2,
        t1_depth_min: float = 8.0, contraction_ratio: float = 0.75,
        near_pivot_pct: float = 6.0) -> dict:
    """Detect a Volatility Contraction Pattern in the recent price path.

    Walks successive swing-high→swing-low legs from the highest recent pivot; each contraction's
    depth (% off its local high) must be at least `contraction_ratio` tighter than the previous,
    with a first correction of >= `t1_depth_min`%. A valid VCP coils under a breakout `pivot`
    (the last swing high) with contracting volatility (ATR10/ATR50 < 1). Never raises."""
    out = {"valid": False, "num_contractions": 0, "contractions": [], "pivot": None,
           "atr_compression": None, "near_pivot": False, "state": "None", "score": 0}
    if df is None or len(df) < 40:
        return out
    d = df.tail(lookback)
    high, low, close = d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float)
    n = len(close)
    hs, ls = _swings(high, low, window=5)
    if len(hs) < 2 or len(ls) < 1:
        return out

    # Build a contraction chain from the highest swing high: high -> next lower low -> ...
    start = max(hs, key=lambda x: x[1])
    chain, cur_hi = [], start
    while True:
        # first low after this high
        lows_after = [l for l in ls if l[0] > cur_hi[0]]
        if not lows_after:
            break
        lo = min(lows_after, key=lambda x: x[1])
        depth = (cur_hi[1] - lo[1]) / cur_hi[1] * 100 if cur_hi[1] else 0
        if depth <= 0:
            break
        chain.append({"high_idx": cur_hi[0], "high": cur_hi[1], "low_idx": lo[0],
                      "low": lo[1], "depth_pct": round(depth, 2)})
        # next swing high after this low (lower than / near the prior high)
        highs_after = [h for h in hs if h[0] > lo[0]]
        if not highs_after:
            break
        cur_hi = min(highs_after, key=lambda x: x[0])  # earliest next high
        if len(chain) >= 4:
            break

    if len(chain) < min_contractions:
        return out

    depths = [c["depth_pct"] for c in chain]
    # T1 deep enough, and each successive contraction tighter than the last by the ratio
    tightening = all(depths[i] <= depths[i - 1] * contraction_ratio + 1e-9 for i in range(1, len(depths)))
    t1_ok = depths[0] >= t1_depth_min
    atr10, atr50 = _atr(high, low, close, 10), _atr(high, low, close, 50)
    atr_comp = (atr10 / atr50) if (atr10 and atr50 and atr50 > 0) else None
    atr_ok = atr_comp is not None and atr_comp < 1.0

    pivot = float(max(c["high"] for c in chain[-1:]) if chain else 0) or None
    price = float(close[-1])
    near = pivot is not None and (pivot - price) / pivot * 100 <= near_pivot_pct and price <= pivot * 1.02
    breakout = pivot is not None and price > pivot

    valid = t1_ok and tightening and len(chain) >= min_contractions and (atr_ok or len(chain) >= 3)
    # score: contractions + tightness + compression + proximity
    score = 0
    if valid:
        score = min(100, 40 + 12 * len(chain)
                    + (15 if atr_ok else 0)
                    + (15 if near else 0)
                    + (10 if depths[-1] < 10 else 0))
    state = ("Breakout" if breakout else "Pre-breakout" if near else "Base") if valid else "None"
    out.update({"valid": valid, "num_contractions": len(chain), "contractions": chain,
                "pivot": round(pivot, 2) if pivot else None,
                "atr_compression": round(atr_comp, 3) if atr_comp is not None else None,
                "near_pivot": bool(near), "state": state, "score": int(score),
                "final_depth_pct": depths[-1]})
    return out


def vcp_setup(df: pd.DataFrame, rs_pct: float | None = None) -> dict:
    """Combined long-only VCP read for the conviction engine: a valid VCP only counts inside a
    Minervini Stage-2 uptrend. Returns a compact dict the scanner can score & display."""
    tt = trend_template(df, rs_pct=rs_pct)
    pat = vcp(df)
    stage2 = tt.get("passed", False)
    valid = bool(stage2 and pat.get("valid"))
    if valid and pat["state"] in ("Pre-breakout", "Breakout"):
        status = "pass"
    elif stage2 and (pat.get("valid") or tt["score"] >= 85):
        status = "warn"
    else:
        status = "fail"
    return {
        "valid": valid, "status": status, "state": pat.get("state", "None"),
        "score": pat.get("score", 0), "pivot": pat.get("pivot"),
        "num_contractions": pat.get("num_contractions", 0),
        "stage2": stage2, "trend_score": tt.get("score", 0),
        "near_pivot": pat.get("near_pivot", False),
        "atr_compression": pat.get("atr_compression"),
    }


# ---------------------------------------------------------------------------------------------------
# Stockbee Momentum Burst  (ported from stockbee-momentum-burst-screener)
# ---------------------------------------------------------------------------------------------------
def momentum_burst(df: pd.DataFrame) -> dict:
    """Detect a Stockbee-style short-term momentum burst on the latest bar: a sharp expansion out
    of a prior range-contraction, meant to run 3-5 sessions. Three trigger families (4% breakout /
    dollar breakout / range expansion), then a 0-100 setup-quality score from volume expansion,
    prior-base tightness, close-location and risk-distance, minus failure filters. Long-only; pure;
    never raises. Returns a compact dict the conviction engine scores & the modal displays."""
    out = {"valid": False, "triggers": [], "score": 0, "rating": "none",
           "gain_pct": None, "vol_ratio": None, "close_loc": None, "risk_pct": None}
    try:
        if df is None or len(df) < 25:
            return out
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        pc = c[-2]
        if not (pc and c[-1]):
            return out
        gain = c[-1] / pc - 1.0
        rng = h[-1] - lo[-1]
        prior3 = [h[i] - lo[i] for i in range(-4, -1)]
        vol_ratio = v[-1] / v[-2] if v[-2] else 0.0
        vol_avg20 = float(np.mean(v[-21:-1])) if len(v) > 21 else float(np.mean(v[:-1]))
        close_loc = (c[-1] - lo[-1]) / rng if rng > 0 else 0.0

        triggers = []
        if gain >= 0.04 and v[-1] > v[-2]:
            triggers.append("4% breakout")
        if (c[-1] - o[-1]) >= 0.90 and v[-1] > vol_avg20 * 0.8:
            triggers.append("dollar breakout")
        prior_not_extended = max(prior3) < np.mean(prior3) * 1.8 if prior3 else True
        if rng > max(prior3) and prior_not_extended:
            triggers.append("range expansion")
        if not triggers:
            return out

        # prior base tightness (range contraction over the ~10 bars before the trigger)
        base = c[-12:-1]
        base_width = (float(np.max(base)) / float(np.min(base)) - 1.0) if len(base) and np.min(base) else 1.0
        # risk distance to the trigger-day low (the Stockbee stop)
        risk_pct = (c[-1] - lo[-1]) / c[-1] if c[-1] else 1.0
        # 3-day prior run-up (failure filter: already extended into the trigger)
        runup3 = c[-2] / c[-5] - 1.0 if len(c) >= 5 and c[-5] else 0.0

        score = 0
        score += min(30, gain * 100 * 3)                    # day gain (capped)
        score += min(25, (vol_ratio - 1.0) * 25) if vol_ratio > 1 else 0   # volume expansion
        score += 20 if base_width <= 0.12 else (10 if base_width <= 0.20 else 0)  # tight prior base
        score += 15 * close_loc                              # close near high of day
        score += 10 if risk_pct <= 0.06 else (5 if risk_pct <= 0.09 else 0)  # manageable stop
        # failure filters
        reject = []
        if runup3 > 0.20:
            score -= 20; reject.append("already +20% in 3 days")
        if risk_pct > 0.12:
            score -= 15; reject.append("stop too far")
        score = int(max(0, min(100, score)))
        rating = "A" if score >= 75 else "B" if score >= 55 else "C"
        return {
            "valid": score >= 55, "triggers": triggers, "score": score, "rating": rating,
            "gain_pct": float(round(gain * 100, 2)), "vol_ratio": float(round(vol_ratio, 2)),
            "close_loc": float(round(close_loc * 100, 1)), "risk_pct": float(round(risk_pct * 100, 2)),
            "base_width": float(round(base_width * 100, 1)), "entry": float(round(c[-1], 2)),
            "stop": float(round(lo[-1], 2)), "reject": reject,
        }
    except Exception:  # noqa: BLE001 - advisory screen must never break the scan
        return out


# ---------------------------------------------------------------------------------------------------
# Episodic Pivot  (ported from stockbee-episodic-pivot-analyzer)
# ---------------------------------------------------------------------------------------------------
def episodic_pivot(df: pd.DataFrame, has_news: bool = False, headline: str | None = None) -> dict:
    """Detect a Day-1 Episodic Pivot: a discrete catalyst repricing a *neglected* name — a large
    gap/expansion on a volume shock out of a quiet base, ideally with fresh news. Scores gap size,
    volume shock, prior neglect (tight base) and catalyst quality. `has_news`/`headline` come from
    our news pipeline so the catalyst component is real. Pure; never raises."""
    out = {"valid": False, "score": 0, "family": None, "gap_pct": None, "vol_x": None,
           "catalyst": False}
    try:
        if df is None or len(df) < 25:
            return out
        o = df["open"].to_numpy(float)
        c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        pc = c[-2]
        if not (pc and c[-1]):
            return out
        gap = o[-1] / pc - 1.0 if o[-1] else 0.0            # opening gap vs prior close
        day = c[-1] / pc - 1.0                              # full-day move
        move = max(gap, day)
        vol_avg20 = float(np.mean(v[-21:-1])) if len(v) > 21 else float(np.mean(v[:-1]))
        vol_x = v[-1] / vol_avg20 if vol_avg20 else 0.0
        # neglect: was the prior ~15-bar base quiet (tight range, unremarkable volume)?
        base = c[-16:-1]
        base_width = (float(np.max(base)) / float(np.min(base)) - 1.0) if len(base) and np.min(base) else 1.0
        neglected = base_width <= 0.18

        # an EP needs a real move + a real volume shock; news makes it a *catalyst* EP
        if move < 0.04 or vol_x < 1.8:
            return out
        # infer family from the headline text (best-effort; falls back to TECHNICAL_EP)
        family = "TECHNICAL_EP"
        cat_pts = 8
        htxt = (headline or "").lower()
        if has_news and htxt:
            if any(k in htxt for k in ("earnings", "beat", "guidance", "raise", "revenue", "profit")):
                family, cat_pts = "EARNINGS_EP", 35
            elif any(k in htxt for k in ("fda", "approval", "phase 3", "pdufa", "trial")):
                family, cat_pts = "FDA_EP", 30
            elif any(k in htxt for k in ("acqui", "merger", "buyout", "takeover", "deal")):
                family, cat_pts = "M_AND_A_EP", 22
            elif any(k in htxt for k in ("contract", "order", "award", "partnership", "customer")):
                family, cat_pts = "CONTRACT_EP", 24
            elif any(k in htxt for k in ("upgrade", "price target", "initiat")):
                family, cat_pts = "ANALYST_EP", 15
            else:
                family, cat_pts = "NEWS_EP", 18
        elif has_news:
            family, cat_pts = "NEWS_EP", 16

        score = 0
        score += min(30, move * 100 * 3)                    # gap/move size
        score += min(25, (vol_x - 1.0) * 12)                # volume shock
        score += 20 if neglected else 5                     # revaluation from neglect
        score += cat_pts                                    # catalyst quality (max 35)
        score = int(max(0, min(100, score)))
        return {
            "valid": score >= 55 and (has_news or move >= 0.06), "score": score, "family": family,
            "gap_pct": float(round(move * 100, 2)), "vol_x": float(round(vol_x, 2)),
            "catalyst": bool(has_news), "neglected": neglected,
            "entry": float(round(c[-1], 2)), "stop": float(round(df["low"].to_numpy(float)[-1], 2)),
        }
    except Exception:  # noqa: BLE001
        return out
