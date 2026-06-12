"""Dynamic universe scan + ranking.

Builds a candidate list from Alpaca's most-active and movers screeners, runs the
strategy on each, and computes a relative-volume "flow proxy". Returns ranked
analysis dicts. Synthetic mode produces deterministic fake data so the whole
pipeline (and the dashboard) works with no keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analytics
import market
import strategies
from config import Config
from data import get_bars, synthetic_bars
from indicators import atr
from risk import position_size, stop_loss_price, take_profit_price
from strategy import generate_signals

# A small static fallback universe used if the screener is unavailable.
_FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "SPY", "QQQ"]

# A curated core of liquid large-caps + sector leaders, always scanned alongside
# the day's movers so there are real uptrends to catch (the movers list alone is
# mostly volatile/declining names that can't produce a bullish crossover).
CORE_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "ADBE", "CRM", "ORCL", "CSCO", "QCOM", "TXN", "INTC", "IBM", "MU", "PLTR",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK", "SCHW", "PYPL",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "HD", "LOW", "MCD", "SBUX", "NKE", "TGT", "COST", "WMT", "DIS", "PG", "KO", "PEP",
    "XOM", "CVX", "CAT", "DE", "BA", "GE", "HON", "UPS", "LIN",
    "CMCSA", "T", "VZ", "UBER", "ABNB", "SHOP", "COIN", "SNOW",
    # --- broader large/mid-cap set (reduces survivorship bias, improves breadth) ---
    "NOW", "INTU", "AMAT", "LRCX", "PANW", "CRWD", "ANET", "MRVL", "TMUS",
    "BKNG", "CMG", "MAR", "GM", "LULU",
    "MDLZ", "CL", "MO", "PM",
    "SPGI", "CB", "PGR", "USB", "KKR",
    "AMGN", "BMY", "GILD", "ISRG", "VRTX", "MDT",
    "COP", "SLB", "RTX", "LMT", "UNP", "ETN",
    "SHW", "FCX", "NEE", "DUK", "SO", "PLD", "AMT",
    "SPY", "QQQ", "DIA", "IWM",
]

# Sector for grouping on the dashboard. Anything not listed falls under "Other".
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology",
    "AMD": "Technology", "ADBE": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "CSCO": "Technology", "QCOM": "Technology", "TXN": "Technology", "INTC": "Technology",
    "IBM": "Technology", "MU": "Technology", "PLTR": "Technology", "SNOW": "Technology",
    # Communication
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "CMCSA": "Communication", "T": "Communication", "VZ": "Communication", "DIS": "Communication",
    # Consumer Discretionary
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "LOW": "Consumer",
    "MCD": "Consumer", "SBUX": "Consumer", "NKE": "Consumer", "TGT": "Consumer",
    "UBER": "Consumer", "ABNB": "Consumer", "SHOP": "Consumer",
    # Consumer Staples
    "COST": "Staples", "WMT": "Staples", "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials", "PYPL": "Financials",
    "COIN": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "PFE": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare",
    # Energy / Industrials / Materials
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "LIN": "Materials",
    # --- broader set ---
    "NOW": "Technology", "INTU": "Technology", "AMAT": "Technology", "LRCX": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "ANET": "Technology", "MRVL": "Technology",
    "TMUS": "Communication",
    "BKNG": "Consumer", "CMG": "Consumer", "MAR": "Consumer", "GM": "Consumer", "LULU": "Consumer",
    "MDLZ": "Staples", "CL": "Staples", "MO": "Staples", "PM": "Staples",
    "SPGI": "Financials", "CB": "Financials", "PGR": "Financials", "USB": "Financials", "KKR": "Financials",
    "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare", "ISRG": "Healthcare",
    "VRTX": "Healthcare", "MDT": "Healthcare",
    "COP": "Energy", "SLB": "Energy",
    "RTX": "Industrials", "LMT": "Industrials", "UNP": "Industrials", "ETN": "Industrials",
    "SHW": "Materials", "FCX": "Materials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate",
    # Index ETFs
    "SPY": "Index ETFs", "QQQ": "Index ETFs", "DIA": "Index ETFs", "IWM": "Index ETFs",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Other / Movers")


def relative_volume(df: pd.DataFrame, window: int) -> float:
    """Latest volume divided by its trailing average. >1.5 ~ unusual activity."""
    if "volume" not in df or len(df) < window + 1:
        return float("nan")
    avg = df["volume"].iloc[-(window + 1):-1].mean()
    if avg <= 0:
        return float("nan")
    return float(df["volume"].iloc[-1] / avg)


def build_universe(cfg: Config) -> list[str]:
    # Core quality names first (so they always get analysed), then the day's movers.
    syms: list[str] = list(CORE_WATCHLIST)
    try:
        syms += market.most_actives(cfg)
        syms += market.movers(cfg)
    except Exception:  # noqa: BLE001 - screener optional; core list still works
        pass
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
    last = sig.iloc[-1]
    price = float(last["close"])
    if price < cfg.min_price:
        return None
    # Quality gate: drop hyper-volatile penny-chaos before the heavy analysis.
    # (Sub-$5 is already filtered; this catches the $5+ names with absurd swings —
    # e.g. +47%/day movers with ~28% ATR that aren't tradeable trend signals and
    # only add noise / trip the data auditor.)
    _atr_s = atr(df, cfg.atr_period)
    _atrp = (float(_atr_s.iloc[-1]) / price * 100) if (len(_atr_s) and not np.isnan(_atr_s.iloc[-1])) else 0.0
    _prev = float(df["close"].iloc[-2]) if len(df) > 1 else price
    _dmove = abs(price / _prev - 1) * 100 if _prev else 0.0
    if _atrp > cfg.max_atr_pct or _dmove > cfg.max_day_move_pct:
        return None
    signal = int(last["signal"])

    # How many bars since the signal last flipped (i.e. how fresh is this state)?
    svals = list(sig["signal"])
    bars_since_flip = 0
    for i in range(len(svals) - 1, 0, -1):
        if svals[i] == svals[-1]:
            bars_since_flip += 1
        else:
            break
    fresh = bars_since_flip <= cfg.buy_window  # crossover within the last few days

    if signal == 1 and fresh:
        action = "BUY"          # entered long within the buy window
    elif signal == 1:
        action = "HOLD LONG"    # long, but the cross was a while ago
    elif signal == 0 and fresh and bars_since_flip < len(svals):
        action = "SELL"         # just dropped out of a long
    else:
        action = "FLAT"
    relvol = relative_volume(df, cfg.rel_volume_window)
    qty = position_size(equity, price, cfg) if signal == 1 else 0
    tail = sig.tail(300)  # ~1+ year of daily bars, sliced client-side by range
    rv_rounded = None if np.isnan(relvol) else round(relvol, 2)
    patterns, factors = analytics.detect(df, sig, cfg, rv_rounded)
    edge = analytics.backtest_edge(df, cfg)
    # Strategy confluence (cheap — no backtests): how many independent methods
    # are long here right now. Stuffed into `factors` so _conviction can score it.
    confl = strategies.evaluate(df, cfg)
    factors["confluence"] = confl["count"]
    factors["confluence_total"] = confl["total"]
    factors["strategies_long"] = confl["long"]
    summary, reasons = _reasoning(sig, cfg, action, price, rv_rounded)
    if confl["count"] >= 2:
        reasons.append(
            f"Strategy confluence — {confl['count']} of {confl['total']} independent strategies "
            f"are long here: {', '.join(confl['long'][:5])}.")
    plan, context = _trade_plan(df, sig, cfg, price, equity)
    conviction = _conviction(action, float(last["rsi"]), rv_rounded, plan, context, cfg,
                             factors, patterns, edge)
    desk_read = _desk_read(action, plan, context, conviction, patterns, edge)
    return {
        "_df": df,  # kept transiently so scan() can backtest per-strategy edges for shown rows
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
        "patterns": patterns,
        "factors": factors,
        "edge": edge,
        "strategies": {"now": confl, "edges": None},  # edges filled for shown rows in scan()
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

    # Bollinger bands (20,2) + MACD histogram for the depth view
    from indicators import bollinger as _bb, macd as _macd
    _, bb_up, bb_lo, _ = _bb(tail["close"], 20, 2.0)
    macd_line, macd_sig, macd_hist = _macd(tail["close"])

    def _ser(s):
        return [None if np.isnan(x) else round(float(x), 3) for x in s]

    return {
        "dates": dates,
        "t": t,
        "open": o, "high": h, "low": low, "close": close,
        "fast": [None if np.isnan(x) else round(float(x), 2) for x in tail["fast"]],
        "slow": [None if np.isnan(x) else round(float(x), 2) for x in tail["slow"]],
        "bb_up": _ser(bb_up), "bb_lo": _ser(bb_lo),
        "macd": _ser(macd_line), "macd_sig": _ser(macd_sig), "macd_hist": _ser(macd_hist),
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

    # 1) The crossover — the core trigger.
    rel = (sig["fast"] > sig["slow"]).astype(int)
    flips = rel.diff().fillna(0)
    flip_idx = flips[flips != 0].index
    if len(flip_idx):
        last_flip = flip_idx[-1]
        days = (sig.index[-1] - last_flip).days
        if rel.loc[last_flip] == 1:
            reasons.append(f"📈 It started trending up about {days} days ago. That's the main reason it was flagged.")
        else:
            reasons.append(f"📉 It started trending down about {days} days ago.")
    reasons.append(
        f"Right now it's heading {'up' if above else 'down'}."
    )

    # 2) RSI = how stretched the price is.
    if rsi_v >= cfg.rsi_overbought:
        reasons.append("It's risen fast and looks a bit overheated, so it could pause or dip soon.")
    elif rsi_v <= cfg.rsi_oversold:
        reasons.append("It's been sold off hard — it could bounce, but it's also still falling.")
    else:
        reasons.append("It's not overheated, so there's still room to climb.")

    # 3) Volume = how much interest vs normal.
    if relvol is not None:
        if relvol >= 1.5:
            reasons.append(f"Lots of people are trading it today — about {relvol}× the usual. That's real interest.")
        elif relvol < 0.8:
            reasons.append(f"It's quiet today — only about {relvol}× the usual trading. Not much interest behind the move.")
        else:
            reasons.append("Trading activity is about normal today.")

    if action == "BUY":
        summary = "It just started trending up, so this could be a spot to buy."
    elif action == "SELL":
        summary = "It's turning down, so this is where the strategy would sell and step back."
    elif action == "HOLD LONG":
        summary = "It's still trending up, so a position you already own would stay open."
    else:
        summary = "There's no clear up-trend yet, so this one's just worth watching for now."

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


def _conviction(action, rsi, relvol, plan, context, cfg: Config,
                factors=None, patterns=None, edge=None,
                sentiment=None, fundamentals=None, price=None):
    """Auto-scored pre-entry checklist for a long. Each check is pass/warn/fail.

    v3: weighs technicals + MACD momentum + the strategy's historical win rate on
    this stock + (when available) news tone, analyst consensus and price-target
    upside — multi-factor confluence, the way a desk would frame it.
    """
    factors = factors or {}
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

    mh = factors.get("macd_hist")
    if mh is None:
        add("Momentum building?", "warn", "Momentum (MACD) reading unavailable.")
    elif mh > 0:
        add("Momentum building?", "pass", "Yes — MACD momentum is positive, so buyers have the upper hand.")
    else:
        add("Momentum building?", "warn", "Not yet — MACD momentum is negative, so sellers still have control.")

    wr = (edge or {}).get("win_rate")
    nt = (edge or {}).get("n_trades") or 0
    if not edge or wr is None or nt < 3:
        add("Worked here before?", "warn", "Not enough past trades on this stock to judge the edge yet.")
    elif wr >= 50:
        add("Worked here before?", "pass", f"Yes — this strategy won {wr}% of its {nt} past trades on this stock.")
    elif wr >= 35:
        add("Worked here before?", "warn", f"Mixed — a {wr}% win rate across {nt} past trades on this stock.")
    else:
        add("Worked here before?", "fail", f"Weak — only {wr}% of {nt} past trades on this stock worked out.")

    cf = factors.get("confluence")
    cft = factors.get("confluence_total")
    if cf is not None and cft:
        longs = factors.get("strategies_long") or []
        names = (": " + ", ".join(longs[:4])) if longs else ""
        if cf >= 3:
            add("Strategies agree?", "pass", f"Strong — {cf} of {cft} independent strategies are long here{names}.")
        elif cf == 2:
            add("Strategies agree?", "warn", f"Some support — 2 of {cft} strategies are long{names}.")
        else:
            add("Strategies agree?", "warn", f"Thin — only {cf} of {cft} strategies are long; limited cross-confirmation.")

    # --- research-driven checks (only added when the data is available) ---
    if sentiment:
        lbl = sentiment.get("label")
        if lbl == "Positive":
            add("News on side?", "pass", "Recent headlines lean positive.")
        elif lbl == "Negative":
            add("News on side?", "fail", "Recent headlines lean negative — a headwind.")
        elif lbl == "Mixed":
            add("News on side?", "warn", "Recent headlines are mixed.")
    if fundamentals:
        an = fundamentals.get("analysts")
        if an:
            c = an.get("consensus")
            if c == "Buy":
                add("Analysts on side?", "pass", f"Wall St leans Buy ({an['buy']} buy / {an['hold']} hold / {an['sell']} sell).")
            elif c == "Sell":
                add("Analysts on side?", "fail", f"Wall St leans Sell ({an['sell']} sell / {an['hold']} hold / {an['buy']} buy).")
            else:
                add("Analysts on side?", "warn", "Wall St is mostly on Hold — no strong analyst conviction.")
        tm = fundamentals.get("target_mean")
        if tm and price:
            up = (tm / price - 1) * 100
            if up >= 10:
                add("Upside to target?", "pass", f"Avg analyst target ${tm:,.0f} is {up:.0f}% above today's price.")
            elif up >= 0:
                add("Upside to target?", "warn", f"Avg target ${tm:,.0f} is only {up:.0f}% above today — limited room.")
            else:
                add("Upside to target?", "fail", f"Price is already above the avg analyst target (${tm:,.0f}).")
        ed = fundamentals.get("earnings_days")
        if ed is not None:
            if ed <= 7:
                add("Earnings clear?", "warn",
                    f"Earnings in {ed} day{'s' if ed != 1 else ''} — a binary event that can gap the price either way.")
            else:
                add("Earnings clear?", "pass", f"No earnings for ~{ed} days, so no imminent event risk.")

    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = sum(pts[c["status"]] for c in checks) / len(checks)
    label = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    return {"score_pct": round(score * 100), "label": label,
            "passes": sum(1 for c in checks if c["status"] == "pass"),
            "total": len(checks), "checks": checks}


def rescore(row: dict, cfg: Config, sentiment=None, fundamentals=None) -> None:
    """Recompute conviction + desk read for a shown row once research is fetched."""
    conv = _conviction(row["action"], row["rsi"], row["rel_volume"], row["plan"], row["context"], cfg,
                       row.get("factors"), row.get("patterns"), row.get("edge"),
                       sentiment=sentiment, fundamentals=fundamentals, price=row.get("price"))
    row["conviction"] = conv
    row["desk_read"] = _desk_read(row["action"], row["plan"], row["context"], conv,
                                  row.get("patterns"), row.get("edge"),
                                  sentiment=sentiment, fundamentals=fundamentals, price=row.get("price"))


def _desk_read(action, plan, context, conviction, patterns=None, edge=None,
               sentiment=None, fundamentals=None, price=None) -> str:
    """A short risk-strategist read, composed from the computed levels + patterns."""
    stop, target = plan.get("stop"), plan.get("target")
    rr = plan.get("rr")
    stop_pct, tgt_pct = plan.get("stop_pct"), plan.get("target_pct")
    conv = conviction["label"]
    bits = []
    if action == "BUY":
        bits.append("In short: it just started trending up — a possible spot to buy.")
    elif action == "HOLD LONG":
        bits.append("In short: it's still trending up, so you'd keep a position you already own.")
    elif action == "SELL":
        bits.append("In short: it's turning down — this is where you'd sell.")
    else:
        bits.append("In short: no clear trend yet — one to watch, not buy.")
    if stop is not None and target is not None:
        bits.append(
            f"The plan in plain terms: buy around ${plan.get('entry'):,.2f}. If it drops to ${stop:,.2f} "
            f"(−{stop_pct}%), sell to cut the loss. If it climbs to ${target:,.2f} (+{tgt_pct}%), take the win. "
            f"So you'd be risking a little to aim for about {rr}× as much."
        )
    bull = [p["label"] for p in (patterns or []) if p["kind"] == "bull"]
    bear = [p["label"] for p in (patterns or []) if p["kind"] == "bear"]
    if bull or bear:
        parts = []
        if bull:
            parts.append("in its favour — " + ", ".join(bull[:3]))
        if bear:
            parts.append("watch-outs — " + ", ".join(bear[:3]))
        bits.append("On the chart, " + "; ".join(parts) + ".")
    if edge and edge.get("win_rate") is not None and (edge.get("n_trades") or 0) >= 3:
        bits.append(f"History check: this strategy has won {edge['win_rate']}% of its "
                    f"{edge['n_trades']} past trades on this stock (hypothetical, no fees).")
    if sentiment and sentiment.get("label") and sentiment["label"] != "Neutral":
        bits.append(f"News tone reads {sentiment['label'].lower()} across {sentiment['n']} recent headlines.")
    if fundamentals:
        seg = []
        an = fundamentals.get("analysts")
        if an:
            seg.append(f"analysts lean {an['consensus'].lower()}")
        tm = fundamentals.get("target_mean")
        if tm and price:
            seg.append(f"avg target ${tm:,.0f} ({(tm/price-1)*100:+.0f}%)")
        if fundamentals.get("pe"):
            seg.append(f"P/E {fundamentals['pe']}")
        if seg:
            bits.append("Research: " + ", ".join(seg) + ".")
        ed = fundamentals.get("earnings_days")
        if ed is not None and ed <= 10:
            bits.append(f"⚠️ Earnings are {ed} day{'s' if ed != 1 else ''} away — expect a possible sharp move around the report.")
    weak = [c["label"].rstrip('?').lower() for c in conviction["checks"] if c["status"] != "pass"]
    if weak:
        bits.append(f"Overall confidence: {conv}. Weaker spots: {', '.join(weak)}.")
    else:
        bits.append(f"Overall confidence: {conv} — everything checks out.")
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
    # Per-strategy backtests are the expensive part, so only run them for the
    # rows that will actually be shown; drop the stashed frame from the rest.
    for row in rows[:cfg.show_top]:
        df = row.pop("_df", None)
        if df is None:
            continue
        try:
            edges = analytics.strategy_edges(df, cfg)
            row["strategies"]["edges"] = edges
            best = edges.get("best")
            if best:
                row.setdefault("reasons", []).append(
                    f"Best historical edge here: {best['label']} — {best['win_rate']}% win over "
                    f"{best['n_trades']} past trades (hypothetical, no fees).")
        except Exception:  # noqa: BLE001
            pass
    for row in rows[cfg.show_top:]:
        row.pop("_df", None)
    return rows
