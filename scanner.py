"""Dynamic universe scan + ranking.

Builds a candidate list from Alpaca's most-active and movers screeners, runs the
strategy on each, and computes a relative-volume "flow proxy". Returns ranked
analysis dicts. Synthetic mode produces deterministic fake data so the whole
pipeline (and the dashboard) works with no keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import market
from config import Config
from data import get_bars, synthetic_bars
from indicators import atr
from risk import position_size, stop_loss_price, take_profit_price
from strategy import generate_signals

# A small static fallback universe used if the screener is unavailable.
_FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "SPY", "QQQ"]


def relative_volume(df: pd.DataFrame, window: int) -> float:
    """Latest volume divided by its trailing average. >1.5 ~ unusual activity."""
    if "volume" not in df or len(df) < window + 1:
        return float("nan")
    avg = df["volume"].iloc[-(window + 1):-1].mean()
    if avg <= 0:
        return float("nan")
    return float(df["volume"].iloc[-1] / avg)


def build_universe(cfg: Config) -> list[str]:
    syms: list[str] = []
    try:
        syms += market.most_actives(cfg)
        syms += market.movers(cfg)
    except Exception:  # noqa: BLE001 - screener optional; fall back gracefully
        syms = list(_FALLBACK)
    if not syms:
        syms = list(_FALLBACK)
    # dedupe preserving order, cap
    seen, uniq = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[: cfg.max_candidates]


def _analyse(symbol: str, df: pd.DataFrame, cfg: Config, equity: float) -> dict | None:
    if df is None or len(df) < cfg.slow_ma + 2:
        return None
    sig = generate_signals(df, cfg)
    last, prev = sig.iloc[-1], sig.iloc[-2]
    price = float(last["close"])
    if price < cfg.min_price:
        return None
    signal, prev_signal = int(last["signal"]), int(prev["signal"])
    if signal == 1 and prev_signal == 0:
        action = "BUY"
    elif signal == 0 and prev_signal == 1:
        action = "SELL"
    elif signal == 1:
        action = "HOLD LONG"
    else:
        action = "FLAT"
    relvol = relative_volume(df, cfg.rel_volume_window)
    qty = position_size(equity, price, cfg) if signal == 1 else 0
    tail = sig.tail(120)
    rv_rounded = None if np.isnan(relvol) else round(relvol, 2)
    summary, reasons = _reasoning(sig, cfg, action, price, rv_rounded)
    plan, context = _trade_plan(df, sig, cfg, price, equity)
    conviction = _conviction(action, float(last["rsi"]), rv_rounded, plan, context, cfg)
    desk_read = _desk_read(action, plan, context, conviction)
    return {
        "symbol": symbol,
        "action": action,
        "price": round(price, 2),
        "rsi": round(float(last["rsi"]), 1),
        "fast_ma": round(float(last["fast"]), 2),
        "slow_ma": round(float(last["slow"]), 2),
        "rel_volume": rv_rounded,
        "stop": round(stop_loss_price(price, cfg), 2) if signal == 1 else None,
        "target": round(take_profit_price(price, cfg), 2) if signal == 1 else None,
        "suggested_shares": qty,
        "as_of": str(sig.index[-1].date()),
        "summary": summary,
        "reasons": reasons,
        "plan": plan,
        "context": context,
        "conviction": conviction,
        "desk_read": desk_read,
        "chart": {
            "dates": [str(d.date()) for d in tail.index],
            "close": [round(float(x), 2) for x in tail["close"]],
            "fast": [None if np.isnan(x) else round(float(x), 2) for x in tail["fast"]],
            "slow": [None if np.isnan(x) else round(float(x), 2) for x in tail["slow"]],
        },
    }


def _reasoning(sig, cfg: Config, action: str, price: float, relvol):
    """Build a plain-English justification from the actual indicator state.

    Everything here is derived from the same numbers that produced the signal —
    no guessing. Returns (summary, [reason, ...]).
    """
    last = sig.iloc[-1]
    fast, slow, rsi_v = float(last["fast"]), float(last["slow"]), float(last["rsi"])
    above = fast > slow
    reasons = []

    # When did the MA relationship last flip?
    rel = (sig["fast"] > sig["slow"]).astype(int)
    flips = rel.diff().fillna(0)
    flip_idx = flips[flips != 0].index
    if len(flip_idx):
        last_flip = flip_idx[-1]
        days = (sig.index[-1] - last_flip).days
        direction = "above" if rel.loc[last_flip] == 1 else "below"
        reasons.append(
            f"The {cfg.fast_ma}-day average crossed {direction} the {cfg.slow_ma}-day "
            f"average about {days} days ago ({last_flip.date()}) — this crossover is the core trigger."
        )
    reasons.append(
        f"Right now the fast MA (${fast:,.2f}) is {'above' if above else 'below'} the slow MA "
        f"(${slow:,.2f}), so the trend bias is {'bullish' if above else 'bearish'}."
    )

    # RSI momentum filter
    if rsi_v >= cfg.rsi_overbought:
        reasons.append(
            f"RSI is {rsi_v:.0f}, at/above the overbought line ({cfg.rsi_overbought:.0f}) — "
            f"the strategy blocks new buys here and treats it as exit pressure."
        )
    elif rsi_v <= cfg.rsi_oversold:
        reasons.append(
            f"RSI is {rsi_v:.0f}, at/below oversold ({cfg.rsi_oversold:.0f}) — stretched to the downside."
        )
    else:
        reasons.append(
            f"RSI is {rsi_v:.0f}, in the neutral zone ({cfg.rsi_oversold:.0f}–{cfg.rsi_overbought:.0f}) — "
            f"momentum isn't blocking an entry."
        )

    # Relative volume (flow proxy)
    if relvol is not None:
        if relvol >= 1.5:
            reasons.append(
                f"Volume is {relvol}× its {cfg.rel_volume_window}-day average — unusually active, "
                f"which often accompanies a real move (volume proxy, not true order flow)."
            )
        elif relvol < 0.8:
            reasons.append(f"Volume is {relvol}× average — quiet, below-normal participation.")
        else:
            reasons.append(f"Volume is {relvol}× average — roughly normal participation.")

    if action == "BUY":
        summary = ("Fresh bullish crossover with RSI clear of overbought — the strategy "
                   "just entered a long here.")
    elif action == "SELL":
        summary = ("Bearish crossover or RSI hitting overbought — the strategy is exiting / "
                   "stepping aside.")
    elif action == "HOLD LONG":
        summary = "Uptrend still intact (fast MA above slow), so an existing long is held."
    else:
        summary = "No active bullish crossover, so the strategy is staying out for now."

    return summary, reasons


def _trade_plan(df, sig, cfg: Config, price: float, equity: float):
    """Concrete long-side trade plan + market context, in price/dollar terms.

    The strategy is long-only, so the plan describes entering/holding a long.
    For non-BUY signals it still shows the levels you'd use *if* you took the
    trade, clearly labelled.
    """
    atr_series = atr(df, cfg.atr_period)
    atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None

    entry = price
    stop_pct = stop_loss_price(entry, cfg)                     # flat % stop
    stop_atr = (entry - cfg.atr_stop_mult * atr_val) if atr_val else None
    # use the tighter (higher) of the two as the working stop, but show both
    working_stop = max(stop_pct, stop_atr) if stop_atr else stop_pct
    target = take_profit_price(entry, cfg)

    shares = position_size(equity, entry, cfg)
    per_share_risk = entry - working_stop
    dollar_risk = shares * per_share_risk if shares else 0.0
    exposure = shares * entry if shares else 0.0
    reward = target - entry
    rr = (reward / per_share_risk) if per_share_risk > 0 else None

    def r(x):
        return None if x is None else round(float(x), 2)

    plan = {
        "direction": "LONG",
        "entry": r(entry),
        "stop": r(working_stop),
        "stop_pct": round((entry - working_stop) / entry * 100, 1),
        "stop_flat": r(stop_pct),
        "stop_atr": r(stop_atr),
        "target": r(target),
        "target_pct": round((target - entry) / entry * 100, 1),
        "rr": None if rr is None else round(rr, 2),
        "shares": shares,
        "dollar_risk": r(dollar_risk),
        "exposure": r(exposure),
    }

    # --- market context ---
    closes = df["close"]
    day_change = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1) * 100 if len(closes) > 1 else None
    hi = float(df["high"].max())
    lo = float(df["low"].min())
    slow = float(sig["slow"].iloc[-1])
    context = {
        "atr": r(atr_val),
        "atr_pct": round(atr_val / entry * 100, 1) if atr_val else None,
        "day_change_pct": None if day_change is None else round(day_change, 2),
        "period_high": r(hi),
        "period_low": r(lo),
        "pct_from_high": round((entry / hi - 1) * 100, 1) if hi else None,
        "pct_from_low": round((entry / lo - 1) * 100, 1) if lo else None,
        "vs_slow_ma_pct": round((entry / slow - 1) * 100, 1) if slow else None,
        "history_bars": int(len(df)),
    }
    return plan, context


def _conviction(action, rsi, relvol, plan, context, cfg: Config):
    """Auto-scored pre-entry checklist for a long. Each check is pass/warn/fail."""
    checks = []

    def add(label, status, note):
        checks.append({"label": label, "status": status, "note": note})

    vs = context.get("vs_slow_ma_pct")
    bullish = vs is not None and vs > 0
    add("Trend alignment",
        "pass" if bullish else "fail",
        "Price above the trend line (slow MA)." if bullish else "Price below the trend line — counter-trend.")

    if rsi >= cfg.rsi_overbought:
        add("Momentum room", "warn", f"RSI {rsi:.0f} is overbought — limited upside room.")
    elif rsi < 40:
        add("Momentum room", "warn", f"RSI {rsi:.0f} is weak — momentum not confirming.")
    else:
        add("Momentum room", "pass", f"RSI {rsi:.0f} leaves room to run without being overbought.")

    if relvol is None:
        add("Volume confirmation", "warn", "Volume data unavailable.")
    elif relvol >= 1.2:
        add("Volume confirmation", "pass", f"Volume {relvol}× average — participation confirms the move.")
    elif relvol < 0.8:
        add("Volume confirmation", "fail", f"Volume {relvol}× average — move lacks conviction.")
    else:
        add("Volume confirmation", "warn", f"Volume {relvol}× average — only normal participation.")

    rr = plan.get("rr")
    if rr is None:
        add("Risk : reward", "fail", "No valid stop/target — can't size the trade.")
    elif rr >= 2:
        add("Risk : reward", "pass", f"1:{rr} — reward clears the 1:2 minimum.")
    elif rr >= 1:
        add("Risk : reward", "warn", f"1:{rr} — marginal; below the 1:2 you'd want.")
    else:
        add("Risk : reward", "fail", f"1:{rr} — risk outweighs reward.")

    if vs is not None and vs > 12:
        add("Not chasing", "warn", f"Price is {vs}% above the trend line — extended, chase risk.")
    else:
        add("Not chasing", "pass", "Entry isn't stretched far above the trend line.")

    atrp = context.get("atr_pct")
    if atrp is not None and atrp > 7:
        add("Volatility sane", "warn", f"ATR is {atrp}% of price — large swings, expect a wide stop.")
    else:
        add("Volatility sane", "pass", f"ATR is {atrp}% of price — manageable volatility." if atrp else "Volatility manageable.")

    if action == "BUY":
        add("Signal freshness", "pass", "Fresh crossover — entering at the start of the move.")
    elif action == "HOLD LONG":
        add("Signal freshness", "warn", "Trend is aged — you'd be entering mid-move.")
    else:
        add("Signal freshness", "fail", "No active long trigger right now.")

    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = sum(pts[c["status"]] for c in checks) / len(checks)
    label = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    return {"score_pct": round(score * 100), "label": label,
            "passes": sum(1 for c in checks if c["status"] == "pass"),
            "total": len(checks), "checks": checks}


def _desk_read(action, plan, context, conviction) -> str:
    """A short risk-strategist read, composed from the computed levels."""
    stop, target = plan.get("stop"), plan.get("target")
    rr = plan.get("rr")
    stop_pct, tgt_pct = plan.get("stop_pct"), plan.get("target_pct")
    conv = conviction["label"]
    bits = []
    if action == "BUY":
        bits.append("Setup: a fresh long trigger.")
    elif action == "HOLD LONG":
        bits.append("Setup: an established long still in its trend.")
    elif action == "SELL":
        bits.append("Setup: a long exit — trend/momentum has rolled over.")
    else:
        bits.append("Setup: no active long; this is a watch, not a trade.")
    if stop is not None and target is not None:
        bits.append(
            f"The trade is defined: risk to ${stop:,.2f} (−{stop_pct}%), reward to ${target:,.2f} "
            f"(+{tgt_pct}%), about 1:{rr} risk:reward. Invalidation is a close below the stop — "
            f"if that breaks, the thesis is wrong and you're out."
        )
    weak = [c["label"].lower() for c in conviction["checks"] if c["status"] != "pass"]
    if weak:
        bits.append(f"Conviction is {conv}; the soft spots are {', '.join(weak)} — "
                    f"tightening those (or waiting for them to confirm) raises the edge.")
    else:
        bits.append(f"Conviction is {conv} with every check passing — a clean setup by the rules.")
    return " ".join(bits)


def _rank_key(row: dict) -> tuple:
    # Actionable first (BUY/SELL), then by unusual volume, then RSI extremity.
    action_rank = {"BUY": 0, "SELL": 0, "HOLD LONG": 1, "FLAT": 2}.get(row["action"], 3)
    relvol = row["rel_volume"] or 0
    return (action_rank, -relvol, -abs(row["rsi"] - 50))


# Populated by scan(); the dashboard surfaces these if nothing was found.
LAST_ERRORS: list[str] = []


def scan(cfg: Config, live: bool) -> list[dict]:
    LAST_ERRORS.clear()
    equity = cfg.starting_cash
    if live and cfg.scan_market:
        symbols = build_universe(cfg)
    elif live:
        symbols = list(cfg.symbols)
    else:
        symbols = list(_FALLBACK)  # synthetic demo universe
    rows, empty, errs = [], 0, 0
    for sym in symbols:
        try:
            df = get_bars(sym, cfg) if live else synthetic_bars(sym, n=cfg.lookback_days)
        except Exception as exc:  # noqa: BLE001 - record, keep scanning
            errs += 1
            if len(LAST_ERRORS) < 3:
                LAST_ERRORS.append(f"{sym}: {type(exc).__name__}: {exc}")
            continue
        if df is None or len(df) == 0:
            empty += 1
            continue
        row = _analyse(sym, df, cfg, equity)
        if row:
            rows.append(row)
    if not rows:
        LAST_ERRORS.append(
            f"Scanned {len(symbols)} symbols: {errs} errored, {empty} returned no data, "
            f"0 usable. If on a free Alpaca plan, confirm market-data access (IEX feed)."
        )
    rows.sort(key=_rank_key)
    return rows
