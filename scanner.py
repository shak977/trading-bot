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
    tail = sig.tail(300)  # ~1+ year of daily bars, sliced client-side by range
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
        "chart": _chart_data(tail),
    }


def _chart_data(tail) -> dict:
    """Full OHLC series + timestamps + moving averages + simulated buy/sell
    markers (where the strategy would have entered/exited historically)."""
    dates = [str(d.date()) for d in tail.index]
    t = [int(pd.Timestamp(d).timestamp() * 1000) for d in tail.index]
    o = [round(float(x), 2) for x in tail["open"]]
    h = [round(float(x), 2) for x in tail["high"]]
    low = [round(float(x), 2) for x in tail["low"]]
    close = [round(float(x), 2) for x in tail["close"]]
    sig_vals = list(tail["signal"])
    n = len(dates)
    buys = [None] * n
    sells = [None] * n
    for i in range(1, n):
        if sig_vals[i] == 1 and sig_vals[i - 1] == 0:
            buys[i] = close[i]        # entry: flat -> long
        elif sig_vals[i] == 0 and sig_vals[i - 1] == 1:
            sells[i] = close[i]       # exit: long -> flat
    return {
        "dates": dates,
        "t": t,
        "open": o, "high": h, "low": low, "close": close,
        "fast": [None if np.isnan(x) else round(float(x), 2) for x in tail["fast"]],
        "slow": [None if np.isnan(x) else round(float(x), 2) for x in tail["slow"]],
        "buys": buys,
        "sells": sells,
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

    # 1) The crossover — the core trigger, in plain terms.
    rel = (sig["fast"] > sig["slow"]).astype(int)
    flips = rel.diff().fillna(0)
    flip_idx = flips[flips != 0].index
    if len(flip_idx):
        last_flip = flip_idx[-1]
        days = (sig.index[-1] - last_flip).days
        if rel.loc[last_flip] == 1:
            reasons.append(
                f"📈 Trend turned up {days} days ago. The recent average price climbed above the "
                f"longer-term average — a classic sign a stock may be starting to trend higher. "
                f"This is the main reason the strategy flagged it."
            )
        else:
            reasons.append(
                f"📉 Trend turned down {days} days ago. The recent average price dropped below the "
                f"longer-term average — usually a sign upward momentum has faded."
            )
    reasons.append(
        f"Direction check: the recent price trend is currently {'ABOVE' if above else 'BELOW'} its "
        f"longer-term trend, so the stock is leaning {'upward' if above else 'downward'} right now."
    )

    # 2) RSI = how 'stretched' the price is (0-100).
    if rsi_v >= cfg.rsi_overbought:
        reasons.append(
            f"Overbought: momentum reads {rsi_v:.0f}/100 (above {cfg.rsi_overbought:.0f}). The stock has "
            f"run up fast and may be due for a pause or pullback, so the strategy won't start a new buy here."
        )
    elif rsi_v <= cfg.rsi_oversold:
        reasons.append(
            f"Oversold: momentum reads {rsi_v:.0f}/100 (below {cfg.rsi_oversold:.0f}). The stock has been "
            f"beaten down and could be near a bounce — but also still falling."
        )
    else:
        reasons.append(
            f"Healthy momentum: reads {rsi_v:.0f}/100 (the calm middle zone). Not overheated, not oversold — "
            f"there's room to move higher without looking stretched."
        )

    # 3) Volume = how much interest there is vs normal.
    if relvol is not None:
        if relvol >= 1.5:
            reasons.append(
                f"Busy: about {relvol}× more shares are trading than usual. Heavy volume means lots of "
                f"people are paying attention, which can give a move more staying power."
            )
        elif relvol < 0.8:
            reasons.append(
                f"Quiet: only about {relvol}× the usual volume. Light trading means weaker conviction "
                f"behind any move."
            )
        else:
            reasons.append(f"Normal activity: about {relvol}× the usual volume — nothing unusual.")

    if action == "BUY":
        summary = ("The trend just turned up and momentum has room to run, so the strategy would "
                   "open a new position (buy) here.")
    elif action == "SELL":
        summary = ("The trend has turned down (or the stock got overheated), so the strategy would "
                   "close the position (sell) and step aside.")
    elif action == "HOLD LONG":
        summary = "The uptrend is still going, so the strategy stays in the position it already holds."
    else:
        summary = "There's no clear uptrend right now, so the strategy stays out and waits."

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
    add("Is it trending up?",
        "pass" if bullish else "fail",
        "Yes — price is above its longer-term trend." if bullish
        else "No — price is below its trend, so you'd be betting against the current direction.")

    if rsi >= cfg.rsi_overbought:
        add("Room to rise?", "warn", f"Careful — momentum ({rsi:.0f}/100) is high; it's run up fast and may pull back.")
    elif rsi < 40:
        add("Room to rise?", "warn", f"Weak — momentum ({rsi:.0f}/100) is soft; buyers aren't in control yet.")
    else:
        add("Room to rise?", "pass", f"Good — momentum ({rsi:.0f}/100) is healthy with space to climb.")

    if relvol is None:
        add("Are people trading it?", "warn", "Volume data unavailable.")
    elif relvol >= 1.2:
        add("Are people trading it?", "pass", f"Yes — about {relvol}× the usual volume; strong interest backs the move.")
    elif relvol < 0.8:
        add("Are people trading it?", "fail", f"Not really — only {relvol}× usual volume; little conviction behind it.")
    else:
        add("Are people trading it?", "warn", f"So-so — about {relvol}× usual volume; nothing special.")

    rr = plan.get("rr")
    if rr is None:
        add("Worth the risk?", "fail", "Can't set a stop/target, so the trade can't be sized.")
    elif rr >= 2:
        add("Worth the risk?", "pass", f"Yes — you'd aim to make about ${rr} for every $1 you risk.")
    elif rr >= 1:
        add("Worth the risk?", "warn", f"Borderline — only about ${rr} reward per $1 risked (you'd want $2+).")
    else:
        add("Worth the risk?", "fail", f"No — the risk outweighs the reward (under $1 back per $1 risked).")

    if vs is not None and vs > 12:
        add("Not chasing?", "warn", f"It's already {vs}% above its trend — buying this late can mean chasing.")
    else:
        add("Not chasing?", "pass", "Price isn't stretched far above its trend, so you're not buying late.")

    atrp = context.get("atr_pct")
    if atrp is not None and atrp > 7:
        add("Calm enough?", "warn", f"Jumpy — it swings ~{atrp}% a day, so expect a wider stop and bigger moves.")
    else:
        add("Calm enough?", "pass", f"Yes — day-to-day swings (~{atrp}%) are manageable." if atrp else "Day-to-day swings are manageable.")

    if action == "BUY":
        add("Good timing?", "pass", "Yes — the up-trend just started, so you'd be getting in early.")
    elif action == "HOLD LONG":
        add("Good timing?", "warn", "The trend started a while ago — you'd be joining partway in.")
    else:
        add("Good timing?", "fail", "No buy trigger right now — nothing to act on yet.")

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
        bits.append("In short: the trend just turned up, so this is a possible spot to buy.")
    elif action == "HOLD LONG":
        bits.append("In short: the up-trend is still going, so a position you already hold would stay open.")
    elif action == "SELL":
        bits.append("In short: the trend has turned down, so this is where you'd sell and step aside.")
    else:
        bits.append("In short: there's no clear up-trend, so this is one to watch, not buy yet.")
    if stop is not None and target is not None:
        bits.append(
            f"The plan: buy near ${plan.get('entry'):,.2f}, and if you're wrong, get out at ${stop:,.2f} "
            f"(a {stop_pct}% loss — your safety exit). If it works, aim for ${target:,.2f} (a {tgt_pct}% gain). "
            f"That's about ${rr} of potential reward for every $1 you put at risk."
        )
    weak = [c["label"].rstrip('?').lower() for c in conviction["checks"] if c["status"] != "pass"]
    if weak:
        bits.append(f"Overall confidence is {conv}. The weaker points are: {', '.join(weak)}. "
                    f"Waiting for those to improve would make it a stronger setup.")
    else:
        bits.append(f"Overall confidence is {conv} — every check passed, which is a clean setup by these rules.")
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
