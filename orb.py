"""Opening Range Breakout (ORB) + VWAP strategy — intraday, stocks-in-play.

A modular, auditable day-trading signal generator, per the design brief. NOT a black box: every
signal carries the levels and reasons behind it, and risk gates can veto it.

Rules
-----
  • Opening range (OR) = the high/low of the first N minutes after the open. The window is
    parameterised (5 / 15 / 30) so the backtest can compare them.
  • Entry: price breaks the OR high (long) or OR low (short) AFTER the window closes — CONFIRMED by
    two gates: VWAP (longs only above VWAP, shorts only below) and market alignment (the index, SPY,
    must agree with the side). Either gate failing vetoes the trade.
  • Stop = the opposite side of the opening range, with an ATR floor so a razor-thin range can't
    create absurd size. Targets = R multiples of the entry-to-stop risk.
  • Position size by risk: a fixed fraction of equity to the stop, capped by a max position size.
  • Realistic costs: half-spread + slippage in basis points, charged per side.

Design
------
The signal logic is pure: it takes a bars DataFrame (+ the index's bars) and parameters and returns
structured records. Data fetching, gating and persistence live in the caller. ``build`` and
``backtest`` never raise — they return ([], {}) / None on any problem — so a bad symbol or feed blip
can never break the dashboard build. Gated by ``cfg.orb_enabled``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_ET = "America/New_York"
_SESSION_OPEN = (9, 30)
_SESSION_CLOSE = (16, 0)
_LIQ_SCORE = {"mega": 100, "very high": 92, "high": 80, "moderate": 55, "thin": 28, "illiquid": 8}

# Signal-score weights (sum = 1.0) — the brief's 7-factor model. Kept SEPARATE from the swing/
# intraday conviction engine: ORB has its own score, threshold, tracker and learning bucket.
_SCORE_W = {"breakout": 0.20, "rvol": 0.20, "vwap": 0.15, "catalyst": 0.15,
            "market": 0.10, "liquidity": 0.10, "volatility": 0.10}
# component key -> the pass/fail check label used by the learning bucket (keep in sync with score_checks)
_COMP_LABEL = {"breakout": "Clean breakout candle?", "rvol": "Unusual volume?",
               "vwap": "Confirmed by VWAP?", "catalyst": "Real catalyst?",
               "market": "Market aligned?", "liquidity": "Liquid enough?",
               "volatility": "OR width tradable?"}


def _weights(learned: dict | None) -> dict:
    """The 7 factor weights, optionally scaled by the ORB bucket's learned multipliers, renormalised
    to sum to 1 so the score stays 0-100. Empty/None learned → the default weights unchanged."""
    if not learned:
        return _SCORE_W
    adj = {k: _SCORE_W[k] * float(learned.get(_COMP_LABEL[k], 1.0)) for k in _SCORE_W}
    tot = sum(adj.values()) or 1.0
    return {k: v / tot for k, v in adj.items()}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _saturate(x: float, full: float) -> float:
    """0..100 reaching ~100 as x approaches `full`, then plateauing (diminishing returns)."""
    if x is None or x <= 0:
        return 0.0
    return round(100.0 * (1.0 - 1.0 / (1.0 + x / max(full, 1e-9))), 1)


def _hhmm_to_min(s: str) -> int:
    try:
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- helpers

def _to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Index -> America/New_York tz so 09:30 means the cash open regardless of DST."""
    if df is None or df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
        df = df.set_index(idx)
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(_ET)


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-hours bars (09:30–16:00 ET); drops pre/post-market prints."""
    if df is None or df.empty:
        return df
    t = df.index
    mins = t.hour * 60 + t.minute
    open_m = _SESSION_OPEN[0] * 60 + _SESSION_OPEN[1]
    close_m = _SESSION_CLOSE[0] * 60 + _SESSION_CLOSE[1]
    return df[(mins >= open_m) & (mins < close_m)]


def _session_vwap(day: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP: cumulative (typical price × volume) / cumulative volume."""
    tp = (day["high"] + day["low"] + day["close"]) / 3.0
    vol = day["volume"].replace(0, np.nan)
    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    return (cum_pv / cum_v).ffill()


def _atr(day: pd.DataFrame, period: int = 14) -> float:
    """ATR on the session's own bars (a volatility floor for the stop). Falls back gracefully."""
    if len(day) < 2:
        return float("nan")
    h, l, c = day["high"], day["low"], day["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    n = min(period, len(tr.dropna()))
    return float(tr.tail(n).mean()) if n else float("nan")


def _sessions(df: pd.DataFrame):
    """Yield (date, day_df) for each regular-hours session, in order."""
    et = _regular_session(_to_et(df))
    if et is None or et.empty:
        return
    for day, grp in et.groupby(et.index.normalize()):
        if not grp.empty:
            yield day.date(), grp.sort_index()


def opening_range(day: pd.DataFrame, window_min: int):
    """(or_high, or_low, or_end_ts) for the first `window_min` minutes of a session."""
    if day is None or day.empty:
        return None
    start = day.index[0]
    or_end = start + pd.Timedelta(minutes=window_min)
    or_bars = day[day.index < or_end]
    if or_bars.empty:
        return None
    return float(or_bars["high"].max()), float(or_bars["low"].min()), or_end


def _market_bias(spy_day: pd.DataFrame, window_min: int):
    """Index tilt for the morning: +1 supports longs, -1 supports shorts, 0 = no clear lean.

    Measured at the moment the opening range closes: is SPY above or below its own VWAP, and on the
    right side of its own opening range? Both agreeing = a clear lean."""
    if spy_day is None or spy_day.empty:
        return 0, "no index data"
    orr = opening_range(spy_day, window_min)
    if not orr:
        return 0, "no index OR"
    sh, sl, or_end = orr
    upto = spy_day[spy_day.index <= or_end]
    if upto.empty:
        return 0, "no index bars"
    vw = _session_vwap(spy_day)
    px = float(upto["close"].iloc[-1])
    vwap_now = float(vw.reindex(upto.index, method="ffill").iloc[-1])
    above_vwap = px > vwap_now
    if above_vwap and px >= sh:
        return 1, "SPY above VWAP & breaking up"
    if (not above_vwap) and px <= sl:
        return -1, "SPY below VWAP & breaking down"
    return (1 if above_vwap else -1), ("SPY above VWAP" if above_vwap else "SPY below VWAP")


def cost_pct(cfg) -> float:
    """Round-trip transaction cost as a fraction of notional (half-spread + slippage, both sides)."""
    half_spread = getattr(cfg, "orb_half_spread_bps", 2.0)
    slip = getattr(cfg, "orb_slippage_bps", 3.0)
    return 2.0 * (half_spread + slip) / 10_000.0


def position_size(entry: float, stop: float, equity: float, cfg) -> tuple[int, float]:
    """Shares sized so a stop-out costs ~`paper_risk_pct` of equity, capped by max position %."""
    risk_frac = getattr(cfg, "paper_risk_pct", 0.005)
    max_pos_pct = getattr(cfg, "max_position_pct", 12.0) / 100.0
    per_share = abs(entry - stop)
    if per_share <= 0 or entry <= 0 or equity <= 0:
        return 0, 0.0
    qty = math.floor((equity * risk_frac) / per_share)
    cap = math.floor((equity * max_pos_pct) / entry)
    qty = max(0, min(qty, cap))
    return qty, round(qty * entry, 2)


# --------------------------------------------------------------------------- signal

def _score_components(bar, vwap_now, or_high, or_low, atr, bias, ctx) -> dict:
    """The brief's 7 factors, each 0-100. Intrinsic factors come from the bars; rvol/catalyst/
    liquidity come from `ctx` (the caller's live data), defaulting to neutral when absent."""
    ctx = ctx or {}
    rng = max(float(bar["high"] - bar["low"]), 1e-9)
    body = float(bar["close"] - bar["open"])
    upper_wick = float(bar["high"] - max(bar["open"], bar["close"]))
    breakout = _clamp(100.0 * (0.5 + 0.5 * (body / rng) - 0.6 * (upper_wick / rng)), 0, 100)

    rvol = ctx.get("rel_volume")
    rvol_s = _saturate(max(0.0, (rvol - 1.0)), 2.0) if rvol is not None else 50.0

    ext = (float(bar["close"]) - vwap_now) / atr if (atr and not math.isnan(atr) and atr > 0) else 0.0
    vwap_s = 0.0 if float(bar["close"]) <= vwap_now else _clamp(100.0 - max(0.0, ext - 1.0) * 30.0, 40, 100)

    catalyst_s = float(ctx.get("catalyst_score") if ctx.get("catalyst_score") is not None else 0.0)
    market_s = {1: 100.0, 0: 50.0, -1: 0.0}.get(bias, 50.0)
    liq_s = float(_LIQ_SCORE.get((ctx.get("liquidity_tier") or "").lower(), 60))

    orw_atr = (or_high - or_low) / atr if (atr and not math.isnan(atr) and atr > 0) else 1.0
    vol_s = _clamp(100.0 - abs(orw_atr - 1.0) * 40.0, 0, 100)   # best when OR width ≈ 1 ATR

    return {"breakout": round(breakout, 1), "rvol": round(rvol_s, 1), "vwap": round(vwap_s, 1),
            "catalyst": round(catalyst_s, 1), "market": round(market_s, 1),
            "liquidity": round(liq_s, 1), "volatility": round(vol_s, 1)}


def signal_for_session(day: pd.DataFrame, spy_day: pd.DataFrame, window_min: int, cfg,
                       equity: float, ctx: dict | None = None, learned: dict | None = None) -> dict | None:
    """First long ORB breakout of the session that clears the hard gates, scored 0-100 per the brief.

    Hard gates (final authority, score can't override): long-only (v1), entry inside the morning
    trade window, close above VWAP, market not leaning against, OR-width within ATR band, RR ≥ floor,
    and (live only) spread under cap. Returns the first qualifying breakout with its score + action
    band, or None. One trade per session."""
    orr = opening_range(day, window_min)
    if not orr:
        return None
    or_high, or_low, or_end = orr
    post = day[day.index >= or_end]
    if post.empty or or_high <= or_low:
        return None
    vw = _session_vwap(day)
    atr = _atr(day)
    bias, bias_why = _market_bias(spy_day, window_min)
    target_rs = list(getattr(cfg, "orb_target_r", [1.0, 2.0]) or [1.0, 2.0])
    long_only = getattr(cfg, "orb_long_only", True)
    win_lo = _hhmm_to_min(getattr(cfg, "orb_window_start", "09:45"))
    win_hi = _hhmm_to_min(getattr(cfg, "orb_window_end", "10:30"))
    orw_lo = getattr(cfg, "orb_orw_atr_min", 0.3)
    orw_hi = getattr(cfg, "orb_orw_atr_max", 3.0)
    min_rr = getattr(cfg, "orb_min_rr", 2.0)
    max_spread = getattr(cfg, "orb_max_spread_pct", 0.25)
    spread_pct = (ctx or {}).get("spread_pct")

    # OR-width / ATR no-trade filter (too wide = stop too far; too narrow = noise)
    if atr and not math.isnan(atr) and atr > 0:
        orw_atr = (or_high - or_low) / atr
        if orw_atr < orw_lo or orw_atr > orw_hi:
            return None

    for ts, bar in post.iterrows():
        tmin = ts.hour * 60 + ts.minute
        if tmin < win_lo:
            continue
        if tmin > win_hi:
            break                                   # past the trade window — done for the day
        vwap_now = float(vw.reindex([ts], method="ffill").iloc[0])
        broke_up = bar["high"] >= or_high
        broke_dn = bar["low"] <= or_low
        if not (broke_up or (broke_dn and not long_only)):
            continue
        direction = "LONG" if broke_up else "SHORT"
        if direction == "LONG" and not (bar["close"] > vwap_now):
            continue                                # VWAP confirmation failed
        if direction == "SHORT" and not (bar["close"] < vwap_now):
            continue
        if direction == "LONG" and bias < 0:
            continue                                # market leaning against
        if direction == "SHORT" and bias > 0:
            continue
        if spread_pct is not None and spread_pct > max_spread:
            return None                             # spread too wide — execution cost kills the edge
        sign = 1 if direction == "LONG" else -1
        entry = or_high if direction == "LONG" else or_low
        stop = or_low if direction == "LONG" else or_high
        if not math.isnan(atr) and abs(entry - stop) < 0.5 * atr:   # ATR floor on the stop
            stop = entry - sign * 0.5 * atr
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        targets = [round(entry + sign * r * risk, 2) for r in target_rs]
        rr = abs(targets[-1] - entry) / risk if targets else 0.0
        if rr < min_rr:                             # hard reward:risk floor
            continue
        comp = _score_components(bar, vwap_now, or_high, or_low, atr, bias, ctx)
        wts = _weights(learned)
        score = round(sum(wts[k] * comp[k] for k in wts), 1)
        thr = getattr(cfg, "orb_score_threshold", 75.0)
        alert = getattr(cfg, "orb_alert_threshold", 65.0)
        action = ("paper_trade" if score >= thr else "alert" if score >= alert else "reject")
        band = ("eligible" if score >= thr else "alert_only" if score >= alert
                else "watch" if score >= 50 else "reject")
        qty, notional = position_size(entry, stop, equity, cfg)
        rt_cost = cost_pct(cfg)
        return {
            "strategy": "ORB", "direction": direction, "window_min": window_min,
            "entry": round(entry, 2), "stop": round(stop, 2),
            "targets": targets, "target": targets[-1] if targets else None,
            "rr": round(rr, 2), "risk_per_share": round(risk, 4),
            "risk_pct": round(risk / entry * 100, 2) if entry else None,
            "or_high": round(or_high, 2), "or_low": round(or_low, 2),
            "or_width_atr": round((or_high - or_low) / atr, 2) if (atr and not math.isnan(atr) and atr > 0) else None,
            "vwap_at_entry": round(vwap_now, 2),
            "atr": round(atr, 4) if not math.isnan(atr) else None,
            "entry_time": ts.isoformat(), "qty": qty, "notional": notional,
            "est_cost_pct": round(rt_cost * 100, 3), "est_cost_usd": round(rt_cost * notional, 2),
            "market_bias": bias, "market_bias_note": bias_why,
            "spread_pct": spread_pct,
            "score": score, "score_components": comp, "score_band": band,
            "recommended_action": action,
            "reasons": [f"Broke {window_min}m OR {'high' if direction=='LONG' else 'low'}",
                        f"{'Above' if direction=='LONG' else 'Below'} VWAP {vwap_now:.2f}",
                        bias_why, f"RR {rr:.1f} ≥ {min_rr}"],
        }
    return None


def build(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame, cfg,
          equity: float | None = None, ctx: dict | None = None, learned: dict | None = None) -> dict | None:
    """Latest-session ORB signal for one symbol (the live entry point). Uses the primary window;
    the backtest compares all windows. `ctx` carries the live stocks-in-play data (rel_volume,
    catalyst_score, liquidity_tier, spread_pct) that completes the 0-100 score. Returns None on no
    signal / any error (fail-silent)."""
    if not getattr(cfg, "orb_enabled", False):
        return None
    try:
        equity = equity if equity is not None else getattr(cfg, "starting_cash", 100_000.0)
        sessions = list(_sessions(df))
        if not sessions:
            return None
        date, day = sessions[-1]
        spy_by_date = dict(_sessions(spy_df)) if spy_df is not None else {}
        spy_day = spy_by_date.get(date)
        window = int(getattr(cfg, "orb_primary_window", 15))
        sig = signal_for_session(day, spy_day, window, cfg, equity, ctx=ctx, learned=learned)
        if sig:
            sig["symbol"] = symbol
            sig["session"] = str(date)
        return sig
    except Exception:  # noqa: BLE001 — ORB must never break the build
        return None


# --------------------------------------------------------------------------- backtest

def backtest(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame, window_min: int, cfg,
             equity: float | None = None) -> dict:
    """Replay every historical session for one window, simulating entry→stop/target/EOD WITH costs.

    Returns {window, trades, n, wins, losses, win_rate, avg_r, expectancy_r, total_r, gross_pct,
    net_pct} — net is after round-trip transaction costs. Never raises."""
    out = {"window": window_min, "symbol": symbol, "trades": [], "n": 0}
    try:
        equity = equity if equity is not None else getattr(cfg, "starting_cash", 100_000.0)
        spy_by_date = dict(_sessions(spy_df)) if spy_df is not None else {}
        rt_cost = cost_pct(cfg)
        trades = []
        for date, day in _sessions(df):
            sig = signal_for_session(day, spy_by_date.get(date), window_min, cfg, equity)
            if not sig:
                continue
            entry, stop = sig["entry"], sig["stop"]
            tgt = sig["target"]
            long = sig["direction"] == "LONG"
            risk = abs(entry - stop)
            after = day[day.index > pd.Timestamp(sig["entry_time"])]
            exit_px, outcome = None, "eod"
            for _ts, b in after.iterrows():
                hit_stop = (b["low"] <= stop) if long else (b["high"] >= stop)
                hit_tgt = (b["high"] >= tgt) if long else (b["low"] <= tgt)
                if hit_stop and hit_tgt:           # both in one bar — assume stop first (conservative)
                    exit_px, outcome = stop, "loss"; break
                if hit_stop:
                    exit_px, outcome = stop, "loss"; break
                if hit_tgt:
                    exit_px, outcome = tgt, "win"; break
            if exit_px is None:
                exit_px = float(day["close"].iloc[-1])
            gross = (exit_px - entry) / entry * (1 if long else -1)
            net = gross - rt_cost
            r_mult = ((exit_px - entry) * (1 if long else -1)) / risk if risk else 0.0
            trades.append({"date": str(date), "direction": sig["direction"], "entry": entry,
                           "stop": stop, "target": tgt, "exit": round(exit_px, 2),
                           "outcome": outcome, "gross_pct": round(gross * 100, 3),
                           "net_pct": round(net * 100, 3), "r": round(r_mult, 2)})
        out["trades"] = trades
        out["n"] = len(trades)
        if trades:
            wins = [t for t in trades if t["net_pct"] > 0]
            out["wins"] = len(wins)
            out["losses"] = out["n"] - len(wins)
            out["win_rate"] = round(len(wins) / out["n"] * 100, 1)
            out["avg_r"] = round(sum(t["r"] for t in trades) / out["n"], 2)
            out["total_r"] = round(sum(t["r"] for t in trades), 2)
            out["gross_pct"] = round(sum(t["gross_pct"] for t in trades), 2)
            out["net_pct"] = round(sum(t["net_pct"] for t in trades), 2)
            out["expectancy_pct"] = round(out["net_pct"] / out["n"], 3)
    except Exception:  # noqa: BLE001
        pass
    return out


_FLATTEN = (15, 45)   # EOD flatten time (ET) — no overnight exposure (brief)


def simulate_exit(sig: dict, day: pd.DataFrame):
    """Replay a single signal forward through its own session: stop, target, or EOD flatten.

    Returns (exit_price, outcome, exit_ts, r_multiple). `outcome` ∈ {win, loss, eod, open}. 'open'
    means the trade hasn't resolved yet on the bars available so far (live, mid-session)."""
    try:
        entry, stop, tgt = sig["entry"], sig["stop"], sig["target"]
        long = sig["direction"] == "LONG"
        risk = abs(entry - stop)
        et = _to_et(day) if (day is not None and not day.empty) else day
        after = et[et.index > pd.Timestamp(sig["entry_time"])] if et is not None else None
        if after is None or after.empty:
            return None, "open", None, 0.0
        flat_m = _FLATTEN[0] * 60 + _FLATTEN[1]
        for ts, b in after.iterrows():
            if ts.hour * 60 + ts.minute >= flat_m:                      # EOD flatten
                px = float(b["open"])
                r = ((px - entry) * (1 if long else -1)) / risk if risk else 0.0
                return round(px, 4), "eod", ts.isoformat(), round(r, 2)
            hit_stop = (b["low"] <= stop) if long else (b["high"] >= stop)
            hit_tgt = (b["high"] >= tgt) if long else (b["low"] <= tgt)
            if hit_stop:                                                # stop first (conservative)
                return round(stop, 4), "loss", ts.isoformat(), round(-risk / risk if risk else -1, 2)
            if hit_tgt:
                r = ((tgt - entry) * (1 if long else -1)) / risk if risk else 0.0
                return round(tgt, 4), "win", ts.isoformat(), round(r, 2)
        return None, "open", None, 0.0          # still live, not yet resolved
    except Exception:  # noqa: BLE001
        return None, "open", None, 0.0


def score_checks(sig: dict, pass_at: float = 60.0) -> list[dict]:
    """The 7 score factors as pass/fail checks, so ORB can feed the same attribution/learning
    machinery as the other strategies (in its OWN bucket). A factor 'passes' when it scored ≥ `pass_at`."""
    comp = sig.get("score_components") or {}
    labels = {"breakout": "Clean breakout candle?", "rvol": "Unusual volume?",
              "vwap": "Confirmed by VWAP?", "catalyst": "Real catalyst?",
              "market": "Market aligned?", "liquidity": "Liquid enough?",
              "volatility": "OR width tradable?"}
    return [{"label": labels.get(k, k), "status": "pass" if (v or 0) >= pass_at else "fail"}
            for k, v in comp.items()]


def gap_pct(df: pd.DataFrame):
    """Overnight gap %: today's regular-session OPEN vs the prior session's last close. This is the
    real gap the stocks-in-play filter wants (vs the placeholder 0%). None if <2 sessions."""
    try:
        sess = list(_sessions(df))
        if len(sess) < 2:
            return None
        prev_close = float(sess[-2][1]["close"].iloc[-1])
        today_open = float(sess[-1][1]["open"].iloc[0])
        if prev_close <= 0:
            return None
        return round((today_open - prev_close) / prev_close * 100, 2)
    except Exception:  # noqa: BLE001
        return None


def aggregate_backtest(by_window_lists: dict) -> dict:
    """Pool per-symbol backtests into one per-window read across all stocks-in-play. Input:
    {window: [per-symbol stats dicts]}. Output per window: n, win_rate, expectancy, net%, avg_r,
    profit_factor — net of the cost model. The honest 'does this edge survive costs' summary."""
    out = {}
    for w, lst in (by_window_lists or {}).items():
        trades = [t for s in (lst or []) for t in (s.get("trades") or [])]
        n = len(trades)
        if not n:
            out[w] = {"window": w, "n": 0}
            continue
        wins = [t for t in trades if t["net_pct"] > 0]
        gross_w = sum(t["net_pct"] for t in trades if t["net_pct"] > 0)
        gross_l = -sum(t["net_pct"] for t in trades if t["net_pct"] < 0)
        out[w] = {"window": w, "n": n, "win_rate": round(len(wins) / n * 100, 1),
                  "expectancy_pct": round(sum(t["net_pct"] for t in trades) / n, 3),
                  "net_pct": round(sum(t["net_pct"] for t in trades), 2),
                  "avg_r": round(sum(t["r"] for t in trades) / n, 2),
                  "profit_factor": round(gross_w / gross_l, 2) if gross_l > 0 else None}
    return out


def best_window(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame, cfg,
                windows=None) -> dict:
    """Run the backtest across the parameterised windows and pick the best by net expectancy.

    Returns {by_window: {w: stats}, best: w, ranked: [...] } — the 'test several' selection."""
    windows = windows or list(getattr(cfg, "orb_windows", [5, 15, 30]))
    by = {w: backtest(symbol, df, spy_df, w, cfg) for w in windows}
    ranked = sorted(by.values(),
                    key=lambda s: (s.get("expectancy_pct", -99) if s.get("n", 0) >= 5 else -99),
                    reverse=True)
    best = ranked[0]["window"] if ranked and ranked[0].get("n", 0) >= 5 else int(
        getattr(cfg, "orb_primary_window", 15))
    return {"by_window": by, "best": best, "ranked": ranked}
