"""Market-timing engine — O'Neil Follow-Through Day (FTD) + IBD distribution-day count.

Ported (and adapted to our Alpaca/Yahoo daily bars) from the tradermonty trading-skills
`ftd-detector` and `ibd-distribution-day-monitor` methodologies.

The macro engine (`macro_regime.py`) already scores VIX / curve / credit / breadth into an
exposure posture. This module adds the piece it lacked: the classic institutional-timing pair —

  * **Follow-Through Day** — after a 3%+ correction, the first high-volume >=1.25% up day in the
    day 4-10 window of a rally attempt. O'Neil's "green light" to add exposure. ~25% go on to a
    sustained trend, so it's necessary-not-sufficient — we treat it as a tilt, never an auto-buy.
  * **Distribution days** — index down >=0.2% on higher volume than the prior day = institutional
    selling. A cluster (5+ in 25 sessions) is a "market under pressure / correction" warning.

`assess(cfg)` runs both on SPY and NASDAQ (QQQ) and returns a compact posture dict the snapshot
stores under `snap["timing"]` and folds into the exposure multiplier. Everything is best-effort:
any data hiccup returns None and the build carries on untouched.
"""
from __future__ import annotations

import dataclasses

import numpy as np


# ---- data ---------------------------------------------------------------------------------------
def _daily_bars(sym: str, cfg):
    """~90 daily bars for an index proxy. Forces a daily timeframe regardless of the caller's
    (the intraday pass runs on 5-min bars). Returns a DataFrame or None — never raises."""
    try:
        from data import get_bars
        dcfg = dataclasses.replace(cfg, timeframe="1Day", lookback_days=130)
        df = get_bars(sym, dcfg)
        if df is None or len(df) < 25:
            return None
        return df
    except Exception:  # noqa: BLE001 - timing is advisory; never break the build
        return None


# ---- distribution days --------------------------------------------------------------------------
def _distribution_days(df, dd_thresh: float = -0.002, expire: int = 25,
                       invalidate_gain: float = 0.05) -> dict:
    """Count active distribution days: close <= -0.2% vs prior close AND volume > prior volume.
    A DD drops out of the count after `expire` sessions or once the index closes `invalidate_gain`
    (5%) above the DD's close. Returns counts by 5/15/25-session buckets + an IBD risk label."""
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    active = []                                    # indices of live DDs
    for i in range(1, n):
        if not (c[i - 1] and v[i - 1]):
            continue
        if (c[i] / c[i - 1] - 1.0) <= dd_thresh + 1e-9 and v[i] > v[i - 1]:
            active.append(i)
    last = n - 1
    live = []
    for i in active:
        age = last - i
        if age > expire:
            continue                               # expired
        # invalidated if any *later* close (within the window) is 5%+ above the DD close
        window = c[i + 1: i + 1 + expire]
        if len(window) and float(np.max(window)) >= c[i] * (1.0 + invalidate_gain):
            continue
        live.append(age)
    d5 = sum(1 for a in live if a <= 5)
    d25 = len(live)
    if d25 >= 6:
        risk = "correction"
    elif d25 >= 4:
        risk = "pressure"
    elif d25 >= 3:
        risk = "caution"
    else:
        risk = "normal"
    return {"count": d25, "d5": d5, "risk": risk}


# ---- follow-through day state machine -----------------------------------------------------------
def _ftd_state(df, look: int = 60) -> dict:
    """Approximate O'Neil FTD state on the recent `look` sessions. Returns a dict with the current
    state, the FTD quality (if any), and rally day-count. States:
      correction   — 3%+ off a recent high / below the swing low, no valid rally yet
      rally_attempt— a Day-1 up day has printed off the low, not yet (or never) confirmed
      confirmed    — a valid FTD fired recently and the index still holds above the rally low
      neutral      — none of the above (quiet uptrend / not enough signal)
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    if n < 20:
        return {"state": "neutral", "quality": 0, "day": 0}
    w0 = max(1, n - look)
    peak_i = w0 + int(np.argmax(c[w0:]))            # recent swing high (by close)
    # swing low = lowest close AFTER the peak
    if peak_i >= n - 2:
        return {"state": "neutral", "quality": 0, "day": 0}
    seg = c[peak_i:]
    low_off = int(np.argmin(seg))
    low_i = peak_i + low_off
    decline = 1.0 - c[low_i] / c[peak_i] if c[peak_i] else 0.0
    down_days = int(np.sum(np.diff(c[peak_i:low_i + 1]) < 0)) if low_i > peak_i else 0
    off_high = 1.0 - c[-1] / c[peak_i] if c[peak_i] else 0.0
    # not a qualifying correction yet
    if decline < 0.03 or down_days < 3:
        return {"state": "neutral", "quality": 0, "day": 0, "off_high": float(round(off_high * 100, 2))}
    # find Day 1: first day after the swing low that's an up day OR closes in the top half of range
    day1 = None
    for i in range(low_i + 1, n):
        if c[i] < c[low_i]:                          # undercut the low -> reset (new low, no rally)
            return {"state": "correction", "quality": 0, "day": 0,
                    "off_high": float(round(off_high * 100, 2))}
        rng = h[i] - lo[i]
        top_half = rng > 0 and (c[i] - lo[i]) / rng >= 0.5
        if c[i] > c[i - 1] or top_half:
            day1 = i
            break
    if day1 is None:
        return {"state": "correction", "quality": 0, "day": 0, "off_high": float(round(off_high * 100, 2))}
    day1_low = lo[day1]
    # walk the attempt: invalidate if a later close breaks Day-1's low; look for the FTD in day 4-10
    best = {"state": "rally_attempt", "quality": 0, "day": n - day1}
    for i in range(day1 + 1, n):
        dnum = i - day1 + 1                          # Day number (Day1 == 1)
        if c[i] < day1_low:                          # integrity break -> attempt fails, back to corr.
            return {"state": "correction", "quality": 0, "day": 0,
                    "off_high": float(round(off_high * 100, 2))}
        gain = c[i] / c[i - 1] - 1.0
        vol_up = v[i] > v[i - 1]
        if 4 <= dnum <= 10 and gain >= 0.0125 and vol_up:
            q = 60 if dnum <= 7 else 50
            if gain >= 0.02:
                q += 25
            elif gain >= 0.015:
                q += 12
            age = n - 1 - i
            state = "confirmed" if age <= 12 else "neutral"
            return {"state": state, "quality": int(min(q, 100)), "day": dnum,
                    "gain": float(round(gain * 100, 2)), "age": age,
                    "off_high": float(round(off_high * 100, 2))}
    return {"state": best["state"], "quality": 0, "day": n - day1,
            "off_high": float(round(off_high * 100, 2))}


# ---- combine ------------------------------------------------------------------------------------
_STATE_RANK = {"correction": 0, "rally_attempt": 1, "neutral": 2, "confirmed": 3}


def assess(cfg) -> dict | None:
    """Run FTD + distribution on SPY and QQQ, combine into one market-timing posture.
    Returns None if neither index could be fetched (build proceeds unaffected)."""
    idx = {}
    for name, sym in (("S&P 500", "SPY"), ("NASDAQ", "QQQ")):
        df = _daily_bars(sym, cfg)
        if df is None:
            continue
        idx[name] = {"ftd": _ftd_state(df), "dist": _distribution_days(df), "sym": sym}
    if not idx:
        return None

    # Worst distribution risk + best FTD state across the two indexes (O'Neil: either index suffices
    # to confirm, but distribution on either is a warning).
    dist_rank = {"normal": 0, "caution": 1, "pressure": 2, "correction": 3}
    worst_dist = max(idx.values(), key=lambda x: dist_rank[x["dist"]["risk"]])["dist"]
    best_ftd = max(idx.values(), key=lambda x: _STATE_RANK[x["ftd"]["state"]])["ftd"]
    dd_total = sum(x["dist"]["count"] for x in idx.values())

    state = best_ftd["state"]
    # A fresh, decisive distribution cluster overrides a stale "confirmed" — institutions are selling.
    if worst_dist["risk"] == "correction":
        state = "correction"
    elif worst_dist["risk"] == "pressure" and state in ("neutral", "rally_attempt"):
        state = "pressure"

    posture = {
        "correction":    ("In correction",     0.5, "Indexes are in a correction / heavy distribution — defense first; new longs fight the tape."),
        "pressure":      ("Under pressure",     0.7, "Distribution is building on the indexes — trim risk, demand higher conviction."),
        "rally_attempt": ("Rally attempt",      0.85, "A rally attempt is underway but unconfirmed — wait for a Follow-Through Day before adding."),
        "confirmed":     ("Confirmed uptrend",  1.15, "A Follow-Through Day confirmed the uptrend — the green light to add exposure into strength."),
        "neutral":       ("Uptrend / neutral",  1.0, "No fresh timing signal — no confirmed FTD and distribution is light; carry on at normal exposure."),
    }
    label, mult, note = posture.get(state, posture["neutral"])
    out = {
        "label": label,
        "state": state,
        "exposure_mult": mult,
        "note": note,
        "dd_total": dd_total,
        "dd_risk": worst_dist["risk"],
        "ftd_quality": best_ftd.get("quality", 0),
        "ftd_day": best_ftd.get("day", 0),
        "off_high": best_ftd.get("off_high"),
        "indexes": {
            name: {
                "state": x["ftd"]["state"],
                "ftd_quality": x["ftd"].get("quality", 0),
                "dd": x["dist"]["count"],
                "dd_risk": x["dist"]["risk"],
                "off_high": x["ftd"].get("off_high"),
            }
            for name, x in idx.items()
        },
    }
    return out
