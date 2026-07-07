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
import rs as rs_mod
import strategies
from config import Config
from data import get_bars, synthetic_bars
from indicators import atr
from risk import honest_target, position_size, stop_loss_price
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

# Expanded liquid pool — large/mid-cap names BEYOND the core list, used when
# cfg.wide_universe is set. Static (checked in) so offline scans stay deterministic.
# Deliberately additive: none of these appear in CORE_WATCHLIST. Keep to liquid,
# optionable names; refresh manually.
WIDE_POOL = [
    "ACN", "ADI", "SNPS", "CDNS", "KLAC", "NXPI", "MCHP", "FTNT", "ADSK", "WDAY",
    "DDOG", "NET", "ZS", "TTD", "DELL", "HPQ", "WDC", "ON", "GLW", "KEYS",
    "REGN", "MRNA", "DXCM", "IDXX", "ZTS", "SYK", "BSX", "BDX", "HCA", "CI",
    "ELV", "CVS", "HUM", "CNC", "MCK",
    "COF", "TFC", "PNC", "BK", "AIG", "MET", "PRU", "TRV", "ALL", "ICE", "CME", "MCO",
    "PSX", "MPC", "VLO", "EOG", "OXY", "KMI", "WMB", "HAL", "DVN", "FANG",
    "ORLY", "AZO", "ROST", "DG", "DLTR", "YUM", "F", "FDX", "NSC", "CSX",
    "WM", "ITW", "PH", "ROK", "MMC", "AON", "EMR", "GD", "NOC", "PCAR",
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


# Company names for the core universe + common ETFs, so every row (signals AND the
# momentum leaderboard) can show a name without an extra API call. Falls back to the
# live asset name, then to the bare ticker.
NAME_MAP = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com Inc.", "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "AVGO": "Broadcom Inc.", "AMD": "Advanced Micro Devices",
    "NFLX": "Netflix Inc.", "ADBE": "Adobe Inc.", "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corp.", "CSCO": "Cisco Systems Inc.", "QCOM": "Qualcomm Inc.",
    "TXN": "Texas Instruments", "INTC": "Intel Corp.", "IBM": "IBM Corp.",
    "MU": "Micron Technology", "PLTR": "Palantir Technologies", "SNOW": "Snowflake Inc.",
    "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America", "WFC": "Wells Fargo & Co.",
    "GS": "Goldman Sachs Group", "MS": "Morgan Stanley", "C": "Citigroup Inc.",
    "V": "Visa Inc.", "MA": "Mastercard Inc.", "AXP": "American Express",
    "BLK": "BlackRock Inc.", "SCHW": "Charles Schwab", "PYPL": "PayPal Holdings",
    "COIN": "Coinbase Global", "UNH": "UnitedHealth Group", "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly & Co.", "ABBV": "AbbVie Inc.", "MRK": "Merck & Co.",
    "PFE": "Pfizer Inc.", "TMO": "Thermo Fisher Scientific", "ABT": "Abbott Laboratories",
    "DHR": "Danaher Corp.", "HD": "Home Depot Inc.", "LOW": "Lowe's Companies",
    "MCD": "McDonald's Corp.", "SBUX": "Starbucks Corp.", "NKE": "Nike Inc.",
    "TGT": "Target Corp.", "COST": "Costco Wholesale", "WMT": "Walmart Inc.",
    "DIS": "Walt Disney Co.", "PG": "Procter & Gamble", "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.", "XOM": "Exxon Mobil Corp.", "CVX": "Chevron Corp.",
    "CAT": "Caterpillar Inc.", "DE": "Deere & Co.", "BA": "Boeing Co.",
    "GE": "GE Aerospace", "HON": "Honeywell International", "UPS": "United Parcel Service",
    "LIN": "Linde plc", "CMCSA": "Comcast Corp.", "T": "AT&T Inc.", "VZ": "Verizon Communications",
    "UBER": "Uber Technologies", "ABNB": "Airbnb Inc.", "SHOP": "Shopify Inc.",
    "NOW": "ServiceNow Inc.", "INTU": "Intuit Inc.", "AMAT": "Applied Materials",
    "LRCX": "Lam Research", "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike Holdings",
    "ANET": "Arista Networks", "MRVL": "Marvell Technology", "TMUS": "T-Mobile US",
    "BKNG": "Booking Holdings", "CMG": "Chipotle Mexican Grill", "MAR": "Marriott International",
    "GM": "General Motors", "LULU": "Lululemon Athletica", "MDLZ": "Mondelez International",
    "CL": "Colgate-Palmolive", "MO": "Altria Group", "PM": "Philip Morris International",
    "SPGI": "S&P Global Inc.", "CB": "Chubb Ltd.", "PGR": "Progressive Corp.",
    "USB": "U.S. Bancorp", "KKR": "KKR & Co.", "AMGN": "Amgen Inc.",
    "BMY": "Bristol-Myers Squibb", "GILD": "Gilead Sciences", "ISRG": "Intuitive Surgical",
    "VRTX": "Vertex Pharmaceuticals", "MDT": "Medtronic plc", "COP": "ConocoPhillips",
    "SLB": "Schlumberger NV", "RTX": "RTX Corp.", "LMT": "Lockheed Martin",
    "UNP": "Union Pacific", "ETN": "Eaton Corp.", "SHW": "Sherwin-Williams",
    "FCX": "Freeport-McMoRan", "NEE": "NextEra Energy", "DUK": "Duke Energy",
    "SO": "Southern Co.", "PLD": "Prologis Inc.", "AMT": "American Tower",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust", "DIA": "SPDR Dow Jones ETF",
    "IWM": "iShares Russell 2000 ETF", "VTI": "Vanguard Total Stock Market ETF",
    "TLT": "iShares 20+ Yr Treasury ETF", "IEF": "iShares 7-10 Yr Treasury ETF",
    "GLD": "SPDR Gold Shares", "DBC": "Invesco DB Commodity ETF",
}


def name_of(symbol: str, fallback: str = "") -> str:
    return NAME_MAP.get(symbol) or fallback or ""


# Curated recent listings to always include, even before they have the months of history the
# daily/swing engine needs (they still get scanned by ORB + intraday, which only need a session
# or two of intraday bars). Plus auto-pulled recent IPOs from the calendar. IPOs are textbook
# stocks-in-play — fresh, high-volume, catalyst-driven.
NEW_LISTINGS = ["SPCX"]   # SpaceX (Nasdaq, IPO'd 2026-06-12)


def recent_listings(cfg: Config) -> list[str]:
    """Curated new listings + auto-pulled recently-listed IPOs (last ~60 days, already trading).
    Deduped, uppercased. Never raises — falls back to the curated list on any error."""
    out = list(NEW_LISTINGS)
    try:
        import research
        for r in research.recent_ipos(cfg):
            s = (r.get("symbol") or "").strip().upper()
            if s and s not in out:
                out.append(s)
    except Exception:  # noqa: BLE001 — calendar is best-effort
        pass
    return out


def relative_volume(df: pd.DataFrame, window: int) -> float:
    """Latest volume divided by its trailing average. >1.5 ~ unusual activity."""
    if "volume" not in df or len(df) < window + 1:
        return float("nan")
    avg = df["volume"].iloc[-(window + 1):-1].mean()
    if avg <= 0:
        return float("nan")
    return float(df["volume"].iloc[-1] / avg)


import re as _re

# A tradable US-equity symbol is uppercase letters, optionally with a dot class/unit
# suffix (BRK.B, AAC.U). Screeners occasionally emit junk like "AAC'U" (units/warrants
# with a stray apostrophe) that the market-data API rejects as "invalid symbol" — filter
# those out here so one bad ticker can't pollute the run's diagnostics.
_SYM_RE = _re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")


def _clean_symbol(s: str) -> str:
    s = (s or "").strip().upper()
    return s if _SYM_RE.match(s) else ""


def build_universe(cfg: Config) -> list[str]:
    # Core quality names first (so they always get analysed), then the day's movers.
    syms: list[str] = list(CORE_WATCHLIST)
    if getattr(cfg, "wide_universe", False):
        syms += WIDE_POOL
    try:
        syms += market.most_actives(cfg)
        syms += market.movers(cfg)
    except Exception:  # noqa: BLE001 - screener optional; core list still works
        pass
    try:
        syms += recent_listings(cfg)   # SpaceX-style fresh IPOs (also fed to ORB/intraday directly)
    except Exception:  # noqa: BLE001
        pass
    if not syms:
        syms = list(_FALLBACK)
    # dedupe preserving order, drop malformed tickers, cap
    seen, uniq = set(), []
    for s in syms:
        s = _clean_symbol(s)
        if s and s not in seen:
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

    # How many bars since the house signal last flipped (state freshness, for the chart).
    svals = list(sig["signal"])
    bars_since_flip = 0
    for i in range(len(svals) - 1, 0, -1):
        if svals[i] == svals[-1]:
            bars_since_flip += 1
        else:
            break

    # --- precision engine: multi-strategy confluence, BOTH directions ---
    # A BUY/SHORT is no longer a single 20/50 crossover (rare → no signals). Instead we ask
    # how many independent, well-known methods agree right now, gated by the trend regime and
    # (downstream) a conviction floor. This surfaces real setups far more often while keeping
    # only the high-quality ones labelled actionable.
    bull = strategies.evaluate(df, cfg)
    bear = strategies.evaluate_short(df, cfg)

    # Trend backbone: above/below the 200-day average.
    closes = df["close"]
    sma200 = float(closes.tail(200).mean()) if len(closes) >= 50 else float(closes.mean())
    uptrend, downtrend = price > sma200, price < sma200

    # "Exit your long" alert: the house long just flipped to flat within the buy window.
    recent_long_exit = any(
        svals[i] == 0 and svals[i - 1] == 1
        for i in range(len(svals) - 1, max(0, len(svals) - 1 - cfg.buy_window), -1))

    action, direction = _classify(bull, bear, uptrend, downtrend, recent_long_exit, cfg)

    relvol = relative_volume(df, cfg.rel_volume_window)
    rv_rounded = None if np.isnan(relvol) else round(relvol, 2)
    # Average daily dollar turnover (liquidity / execution-quality input).
    try:
        _dv = float((df["close"] * df["volume"]).tail(cfg.rel_volume_window).mean()) if "volume" in df else None
        dollar_vol = round(_dv) if _dv and not np.isnan(_dv) else None
    except Exception:  # noqa: BLE001
        dollar_vol = None
    tail = sig.tail(300)  # ~1+ year of daily bars, sliced client-side by range
    patterns, factors = analytics.detect(df, sig, cfg, rv_rounded)
    edge = analytics.backtest_edge(df, cfg)
    factors["confluence"] = bull["count"]
    factors["confluence_total"] = bull["total"]
    factors["strategies_long"] = bull["long"]
    factors["short_confluence"] = bear["count"]
    factors["short_total"] = bear["total"]
    factors["strategies_short"] = bear["short"]
    # Minervini VCP (ported from the vcp-screener skill): a long-only quality confirmation, stored
    # on factors so it survives re-scoring and flows into the self-adapting conviction checklist.
    try:
        import screens
        factors["vcp"] = screens.vcp_setup(df)
    except Exception:  # noqa: BLE001 - additive; never fail a scan on the screen
        factors["vcp"] = None
    # Stockbee Momentum Burst + Episodic Pivot (ported): short-term expansion setups. Momentum
    # burst is a pure price/volume read; EP is computed technical-only here (has_news enrichment
    # happens later once the news pipeline attaches headlines). Both long-only, additive, fail-safe.
    try:
        import screens
        factors["burst"] = screens.momentum_burst(df)
        factors["ep"] = screens.episodic_pivot(df, has_news=False)
        factors["pshort"] = screens.parabolic_short(df)      # short-only exhaustion candidate
    except Exception:  # noqa: BLE001
        factors["burst"] = factors["ep"] = factors["pshort"] = None

    plan, context = _trade_plan(df, sig, cfg, price, equity, direction)
    conviction = _conviction(action, direction, float(last["rsi"]), rv_rounded, plan, context, cfg,
                             factors, patterns, edge)

    # Quality gate (per the "confluence, quality-gated" choice): a fresh BUY/SHORT must clear
    # Medium conviction (≥50). Otherwise the setup is real but not strong enough to call —
    # demote it to the WATCH tier instead of issuing a weak signal.
    if action == "BUY" and conviction["score_pct"] < 50:
        action = "WATCH LONG"
    elif action == "SHORT" and conviction["score_pct"] < 50:
        action = "WATCH SHORT"

    summary, reasons = _reasoning(sig, cfg, action, direction, price, rv_rounded, bull, bear)
    if direction == "LONG" and bull["count"] >= 2:
        reasons.append(
            f"Strategy confluence — {bull['count']} of {bull['total']} independent strategies "
            f"are long here: {', '.join(bull['long'][:5])}.")
    if direction == "SHORT" and bear["count"] >= 2:
        reasons.append(
            f"Bearish confluence — {bear['count']} of {bear['total']} independent strategies are "
            f"short here: {', '.join(bear['short'][:5])}.")

    desk_read = _desk_read(action, direction, plan, context, conviction, patterns, edge)
    actionable = action in ("BUY", "HOLD LONG", "SHORT", "HOLD SHORT")
    return {
        "_df": df,  # kept transiently so scan() can backtest per-strategy edges for shown rows
        "symbol": symbol,
        "action": action,
        "direction": direction,
        "price": round(price, 2),
        "rsi": round(float(last["rsi"]), 1),
        "fast_ma": round(float(last["fast"]), 2),
        "slow_ma": round(float(last["slow"]), 2),
        "rel_volume": rv_rounded,
        "dollar_volume": dollar_vol,
        "liquidity": (__import__("liquidity").classify(dollar_vol, price) if dollar_vol else None),
        "stop": plan.get("stop") if actionable else None,
        "target": plan.get("target") if actionable else None,
        "suggested_shares": plan.get("shares") or 0,
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
        "strategies": {"now": bull, "short": bear, "edges": None},  # edges filled for shown rows
        "chart": _chart_data(tail),
    }


def _classify(bull: dict, bear: dict, uptrend: bool, downtrend: bool,
              recent_long_exit: bool, cfg: Config) -> tuple[str, str]:
    """Turn confluence + trend into an action + direction. The conviction floor (applied by
    the caller) decides whether a fresh BUY/SHORT survives or drops to the WATCH tier.

    Tiers: 3+ agreeing strategies in-trend = actionable (BUY/SHORT, or HOLD if not fresh);
    exactly 2 = WATCH; a long that just broke = EXIT; weak bearish = AVOID; nothing = FLAT."""
    bc, bf = bull["count"], len(bull["fresh"])
    sc, sf = bear["count"], len(bear["fresh"])
    if uptrend and bc >= 3:
        return ("BUY", "LONG") if bf > 0 else ("HOLD LONG", "LONG")
    if downtrend and sc >= 3:
        return ("SHORT", "SHORT") if sf > 0 else ("HOLD SHORT", "SHORT")
    if uptrend and bc == 2:
        return ("WATCH LONG", "LONG")
    if downtrend and sc == 2:
        return ("WATCH SHORT", "SHORT")
    if recent_long_exit:
        return ("EXIT", "LONG")
    if downtrend and sc >= 1:
        return ("AVOID", "SHORT")
    return ("FLAT", "LONG" if uptrend else "SHORT")


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


def _reasoning(sig, cfg: Config, action: str, direction: str, price: float, relvol, bull=None, bear=None):
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

    summaries = {
        "BUY": "Several methods line up to the upside, so this could be a spot to buy.",
        "HOLD LONG": "It's still trending up, so a long you already own would stay open.",
        "WATCH LONG": "A bullish setup is building but not confirmed — one to watch, not buy yet.",
        "SHORT": "Several methods line up to the downside, so this could be a spot to short.",
        "HOLD SHORT": "It's still trending down, so a short you already have would stay open.",
        "WATCH SHORT": "A bearish setup is building but not confirmed — one to watch, not short yet.",
        "EXIT": "It just rolled over out of its up-trend — this is where you'd sell a long.",
        "AVOID": "It's below trend and weak — better to stay away than to buy here.",
        "FLAT": "There's no clear trend yet, so this one's just worth watching for now.",
    }
    summary = summaries.get(action, summaries["FLAT"])
    return summary, reasons


def _target_scenarios(direction: str, entry: float, stop: float, base_target: float, df) -> list[dict]:
    """Three labelled exit scenarios with a plain-English rationale each, so the user sees a
    spectrum (quick / base / stretch) instead of one flat number. Display-only.

    Conservative = ~1.5x risk, or the recent swing level if it sits nearer (resistance/support).
    Base = the actual order target. Stretch = ~5x risk (toward the period high/low when relevant).
    """
    R = abs(entry - stop)
    if R <= 0 or entry <= 0:
        return []
    short = direction == "SHORT"
    try:
        swing = float(df["low"].tail(20).min()) if short else float(df["high"].tail(20).max())
        extreme = float(df["low"].min()) if short else float(df["high"].max())
    except Exception:  # noqa: BLE001
        swing = extreme = None

    def mk(label, price, basis, odds):
        if price is None or price <= 0:
            return None
        gain = (entry - price) if short else (price - entry)
        return {"label": label, "price": round(price, 2),
                "pct": round(gain / entry * 100, 1), "r": round(gain / R, 1),
                "basis": basis, "odds": odds}

    out = []
    # --- conservative: ~1.5R, unless a recent swing level sits between entry and 1.5R (nearer) ---
    if short:
        cons, cb = entry - 1.5 * R, "about 1.5x your risk — a quick, higher-odds cover"
        if swing is not None and entry > swing > cons:
            cons, cb = swing, "just above the recent 20-day low, where buyers stepped in before"
    else:
        cons, cb = entry + 1.5 * R, "about 1.5x your risk — a quick, higher-odds exit"
        if swing is not None and entry < swing < cons:
            cons, cb = swing, "just under the recent 20-day high, where sellers showed up before"
    out.append(mk("Conservative", cons, cb, "higher odds"))

    # --- base: the actual order target ---
    out.append(mk("Base (order target)", base_target,
                  "the honest base target — nearest real level price must clear, bounded by fundamentals & volatility", "medium"))

    # --- stretch: ~5R, noting the period extreme when it's in that neighbourhood ---
    if short:
        stretch = entry - 5 * R
        sb = "about 5x your risk — lower odds, bigger payoff"
        if extreme is not None and extreme <= stretch:
            sb = "down toward the period low (~5x+ risk) — lower odds, bigger payoff"
    else:
        stretch = entry + 5 * R
        sb = "about 5x your risk — lower odds, bigger payoff"
        if extreme is not None and extreme >= stretch:
            sb = "up toward the period high (~5x+ risk) — lower odds, bigger payoff"
    out.append(mk("Stretch", stretch, sb, "lower odds"))
    return [o for o in out if o]


def _apply_fundamental_cap(plan: dict, fundamentals, cfg: Config) -> None:
    """Once fundamentals are known, bound the structural target by the analyst mean price
    target (fundamental fair value). Only ever trims the target closer — never extends it."""
    if not (plan and fundamentals):
        return
    tm = fundamentals.get("target_mean")
    entry, stop, tgt = plan.get("entry"), plan.get("stop"), plan.get("target")
    if not (tm and entry and tgt):
        return
    short = plan.get("direction") == "SHORT"
    new = tm if ((not short and entry < tm < tgt) or (short and entry > tm > tgt)) else tgt
    if new == tgt:
        return
    plan["target"] = round(float(new), 2)
    plan["target_pct"] = round(abs(new - entry) / entry * 100, 1)
    if stop and abs(entry - stop) > 0:
        plan["rr"] = round(abs(new - entry) / abs(entry - stop), 2)
    plan["target_basis"] = "analyst mean price target — fundamentals cap the chart objective"
    for sc in plan.get("targets", []):
        if str(sc.get("label", "")).startswith("Base"):
            sc["price"], sc["pct"] = plan["target"], plan["target_pct"]
            if stop and abs(entry - stop) > 0:
                sc["r"] = round(abs(new - entry) / abs(entry - stop), 1)


def _trade_plan(df, sig, cfg: Config, price: float, equity: float, direction: str = "LONG"):
    """Concrete trade plan + market context, in price/dollar terms, for either direction.

    LONG: stop below entry, target above. SHORT: the mirror — stop above entry (where the
    bet is wrong), target below (where you'd cover). For non-actionable rows it still shows
    the levels you'd use *if* you took the trade, clearly labelled in the UI.
    """
    atr_series = atr(df, cfg.atr_period)
    atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None
    entry = price

    if direction == "SHORT":
        stop_pct_lvl = entry * (1 + cfg.stop_loss_pct)              # flat % stop, above entry
        stop_atr = (entry + cfg.atr_stop_mult * atr_val) if atr_val else None
        # tighter stop = the LOWER one (closer to entry) when shorting
        working_stop = min(stop_pct_lvl, stop_atr) if stop_atr else stop_pct_lvl
        target, target_basis = honest_target(entry, df, atr_val, working_stop, cfg, short=True)
        per_share_risk = working_stop - entry
        reward = entry - target
    else:
        stop_pct_lvl = stop_loss_price(entry, cfg)                 # flat % stop, below entry
        stop_atr = (entry - cfg.atr_stop_mult * atr_val) if atr_val else None
        # tighter stop = the HIGHER one (closer to entry) when long
        working_stop = max(stop_pct_lvl, stop_atr) if stop_atr else stop_pct_lvl
        target, target_basis = honest_target(entry, df, atr_val, working_stop, cfg, short=False)
        per_share_risk = entry - working_stop
        reward = target - entry

    shares = position_size(equity, entry, cfg)
    dollar_risk = shares * per_share_risk if shares else 0.0
    exposure = shares * entry if shares else 0.0
    rr = (reward / per_share_risk) if per_share_risk > 0 else None

    def r(x):
        return None if x is None else round(float(x), 2)

    plan = {
        "direction": direction,
        "entry": r(entry),
        "stop": r(working_stop),
        "stop_pct": round(abs(working_stop - entry) / entry * 100, 1),
        "stop_flat": r(stop_pct_lvl),
        "stop_atr": r(stop_atr),
        "target": r(target),
        "target_pct": round(abs(target - entry) / entry * 100, 1),
        "target_basis": target_basis,
        "rr": None if rr is None else round(rr, 2),
        "shares": shares,
        "dollar_risk": r(dollar_risk),
        "exposure": r(exposure),
    }
    # Three exit scenarios (display only — the order still uses plan["target"] as the base case).
    plan["targets"] = _target_scenarios(direction, entry, working_stop, target, df)

    # Estimated time-to-play-out: how many sessions to reach the target at the stock's typical daily
    # move (ATR). Presented as a RANGE — the low end if it trends straight there, the high end if it
    # chops (price rarely moves in a straight line). Bounded to a sane swing-trade window.
    _atrp = (atr_val / entry * 100) if atr_val else None
    _tgtp = abs(target - entry) / entry * 100 if (target and entry) else None
    if _atrp and _atrp > 0 and _tgtp:
        _base = _tgtp / _atrp
        plan["hold_lo"] = int(max(1, min(60, round(_base))))
        plan["hold_hi"] = int(max(plan["hold_lo"] + 1, min(90, round(_base * 2.4))))
        # bars are in the SIGNAL's timeframe — daily = sessions, intraday = 5-min bars, etc.
        plan["hold_tf"] = getattr(cfg, "timeframe", "1Day")

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


def _conviction(action, direction, rsi, relvol, plan, context, cfg: Config,
                factors=None, patterns=None, edge=None,
                sentiment=None, fundamentals=None, price=None, tv=None, regime=None, insider=None, buzz=None, news_idea=None, intraday=None, sector_pct=None, short_interest=None, retail=None, liquidity=None, learned=None):
    """Auto-scored pre-entry checklist, direction-aware. Each check is pass/warn/fail.

    For a LONG it asks the bullish questions (trending up? room to rise?); for a SHORT it
    asks the mirror (trending down? room to fall?). Weighs technicals + MACD momentum + the
    strategy's historical win rate + (when available) news tone, analyst consensus and
    price-target room — multi-factor confluence, the way a desk would frame it.
    """
    factors = factors or {}
    checks = []
    short = direction == "SHORT"

    def add(label, status, note, weight=1.0):
        if learned:                       # adaptive: scale by what this check has historically earned
            weight = round(weight * learned.get(label, 1.0), 3)
        checks.append({"label": label, "status": status, "note": note, "weight": weight})

    vs = context.get("vs_slow_ma_pct")
    if short:
        bearish = vs is not None and vs < 0
        add("Is it trending down?",
            "pass" if bearish else "fail",
            "Yes — price is below its longer-term trend, so the bet runs with the downtrend." if bearish
            else "No — price is above its trend, so a short would fight the current direction.")
    else:
        bullish = vs is not None and vs > 0
        add("Is it trending up?",
            "pass" if bullish else "fail",
            "Yes — price is above its longer-term trend." if bullish
            else "No — price is below its trend, so you'd be betting against the current direction.")

    # Relative strength — is this name leading or lagging the market?
    rs_f = factors.get("rs")
    if rs_f and rs_f.get("pct") is not None:
        pct = rs_f["pct"]
        w = cfg.rs_conviction_weight
        if short:
            # for a short we want a laggard: low percentile is good
            if pct <= (100 - cfg.rs_pass_pct):
                add("Leading the market?", "pass",
                    f"Yes (for a short) — RS percentile {pct}; this name lags the market, so weakness has company.", w)
            elif pct >= (100 - cfg.rs_fail_pct):
                add("Leading the market?", "fail",
                    f"No — RS percentile {pct}; you'd be shorting a relative leader.", w)
            else:
                add("Leading the market?", "warn",
                    f"Mixed — RS percentile {pct}; no clear relative weakness yet.", w)
        else:
            if pct >= cfg.rs_pass_pct:
                add("Leading the market?", "pass",
                    f"Yes — RS percentile {pct}; this name is outrunning the market (leadership).", w)
            elif pct <= cfg.rs_fail_pct:
                add("Leading the market?", "fail",
                    f"No — RS percentile {pct}; this name lags the market (laggard).", w)
            else:
                add("Leading the market?", "warn",
                    f"Mixed — RS percentile {pct}; roughly in line with the market.", w)

    # Regime alignment — is this trade running with the broad market or against it?
    reg_lbl = (regime or {}).get("label")
    if reg_lbl and action in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
        br = (regime or {}).get("breadth")
        brtxt = f"{br}% of stocks above trend" if br is not None else "mixed breadth"
        if short:
            if reg_lbl == "Risk-off":
                add("With the market?", "pass", f"Yes — the tape is Risk-off ({brtxt}); a short runs with the broad market.")
            elif reg_lbl == "Risk-on":
                add("With the market?", "fail", f"No — the tape is Risk-on ({brtxt}); a short fights a rising market.")
            else:
                add("With the market?", "warn", f"Mixed — neutral tape ({brtxt}); no market tailwind for a short.")
        else:
            if reg_lbl == "Risk-on":
                add("With the market?", "pass", f"Yes — the tape is Risk-on ({brtxt}); a long runs with the broad market.")
            elif reg_lbl == "Risk-off":
                add("With the market?", "fail", f"No — the tape is Risk-off ({brtxt}); a long fights a falling market.")
            else:
                add("With the market?", "warn", f"Mixed — neutral tape ({brtxt}); no market tailwind for a long.")

    if short:
        if rsi <= cfg.rsi_oversold:
            add("Room to fall?", "warn", f"Careful — momentum ({rsi:.0f}/100) is already very low; it's been sold hard and may bounce.")
        elif rsi > 60:
            add("Room to fall?", "pass", f"Good — momentum ({rsi:.0f}/100) is elevated, so there's room to drop.")
        else:
            add("Room to fall?", "pass", f"OK — momentum ({rsi:.0f}/100) isn't washed out yet, so there's room lower.")
    elif rsi >= cfg.rsi_overbought:
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

    if short:
        if vs is not None and vs < -12:
            add("Not chasing?", "warn", f"It's already {vs}% below its trend — shorting this late can mean chasing the move down.")
        else:
            add("Not chasing?", "pass", "Price isn't stretched far below its trend, so you're not shorting late.")
    elif vs is not None and vs > 12:
        add("Not chasing?", "warn", f"It's already {vs}% above its trend — buying this late can mean chasing.")
    else:
        add("Not chasing?", "pass", "Price isn't stretched far above its trend, so you're not buying late.")

    atrp = context.get("atr_pct")
    if atrp is not None and atrp > 7:
        add("Calm enough?", "warn", f"Jumpy — it swings ~{atrp}% a day, so expect a wider stop and bigger moves.")
    else:
        add("Calm enough?", "pass", f"Yes — day-to-day swings (~{atrp}%) are manageable." if atrp else "Day-to-day swings are manageable.")

    if action in ("BUY", "SHORT"):
        add("Good timing?", "pass",
            "Yes — the down-trend setup just triggered, so you'd be getting in early." if short
            else "Yes — the up-trend just started, so you'd be getting in early.")
    elif action in ("HOLD LONG", "HOLD SHORT"):
        add("Good timing?", "warn", "The move started a while ago — you'd be joining partway in.")
    elif action in ("WATCH LONG", "WATCH SHORT"):
        add("Good timing?", "warn", "Setup is building but not confirmed yet — one to watch, not act on.")
    else:
        add("Good timing?", "fail", "No trigger right now — nothing to act on yet.")

    mh = factors.get("macd_hist")
    if mh is None:
        add("Momentum building?", "warn", "Momentum (MACD) reading unavailable.")
    elif (mh < 0) if short else (mh > 0):
        add("Momentum building?", "pass",
            "Yes — MACD momentum is negative, so sellers have the upper hand." if short
            else "Yes — MACD momentum is positive, so buyers have the upper hand.")
    else:
        add("Momentum building?", "warn",
            "Not yet — MACD momentum is still positive, so buyers haven't given up." if short
            else "Not yet — MACD momentum is negative, so sellers still have control.")

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

    if short:
        cf = factors.get("short_confluence")
        cft = factors.get("short_total")
        names_list = factors.get("strategies_short") or []
        side = "short"
    else:
        cf = factors.get("confluence")
        cft = factors.get("confluence_total")
        names_list = factors.get("strategies_long") or []
        side = "long"
    if cf is not None and cft:
        names = (": " + ", ".join(names_list[:4])) if names_list else ""
        if cf >= 3:
            add("Strategies agree?", "pass", f"Strong — {cf} of {cft} independent strategies are {side} here{names}.")
        elif cf == 2:
            add("Strategies agree?", "warn", f"Some support — 2 of {cft} strategies are {side}{names}.")
        else:
            add("Strategies agree?", "warn", f"Thin — only {cf} of {cft} strategies are {side}; limited cross-confirmation.")

    # --- research-driven checks (only added when the data is available) ---
    if tv:
        try:
            import tradingview as _tv
            al = _tv.alignment(tv, direction)
        except Exception:  # noqa: BLE001
            al = None
        d, w = tv.get("d"), tv.get("w")
        side_word = "short" if short else "long"
        if al == "agree":
            add("TradingView agrees?", "pass",
                f"TradingView rates it {d} daily / {w} weekly — an independent read that lines up with this {side_word}.")
        elif al == "oppose":
            add("TradingView agrees?", "fail",
                f"TradingView rates it {d} daily / {w} weekly — that's against this {side_word}; a real disagreement to weigh.")
        elif al == "mixed":
            add("TradingView agrees?", "warn",
                f"TradingView is mixed/neutral here ({d} daily / {w} weekly) — no strong confirmation either way.")

    if sentiment:
        lbl = sentiment.get("label")
        _nw = getattr(cfg, "news_conviction_weight", 1.0)
        if lbl == "Positive":
            add("News on side?", "fail" if short else "pass",
                "Recent headlines lean positive (bullish) — that pushes the price up, which works against your short." if short
                else "Recent headlines lean positive (bullish) — supports your long.", _nw)
        elif lbl == "Negative":
            add("News on side?", "pass" if short else "fail",
                "Recent headlines lean negative (bearish) — downward pressure that backs your short." if short
                else "Recent headlines lean negative (bearish) — a headwind for your long.", _nw)
        elif lbl == "Mixed":
            add("News on side?", "warn", "Recent headlines are mixed.", _nw)
    if fundamentals:
        an = fundamentals.get("analysts")
        if an:
            c = an.get("consensus")
            # For a short, a Sell consensus is *supportive*; a Buy consensus is the headwind.
            if c == "Buy":
                add("Analysts on side?", "fail" if short else "pass",
                    f"Wall St leans Buy ({an['buy']} buy / {an['hold']} hold / {an['sell']} sell) — "
                    + ("they expect it to RISE, which works against your short." if short
                       else "they expect it to rise, which supports your long."))
            elif c == "Sell":
                add("Analysts on side?", "pass" if short else "fail",
                    f"Wall St leans Sell ({an['sell']} sell / {an['hold']} hold / {an['buy']} buy) — "
                    + ("they expect it to FALL, which backs your short." if short
                       else "they expect it to fall — a headwind for your long."))
            else:
                add("Analysts on side?", "warn", "Wall St is mostly on Hold — no strong analyst conviction.")
        tm = fundamentals.get("target_mean")
        if tm and price:
            up = (tm / price - 1) * 100
            if short:
                if up <= -10:
                    add("Room to target?", "pass", f"Avg analyst target ${tm:,.0f} is {abs(up):.0f}% BELOW today — room to fall.")
                elif up <= 0:
                    add("Room to target?", "warn", f"Avg target ${tm:,.0f} is only {abs(up):.0f}% below today — limited room down.")
                else:
                    add("Room to target?", "fail", f"Avg analyst target ${tm:,.0f} is {up:.0f}% ABOVE today — analysts see upside, against a short.")
            elif up >= 10:
                add("Upside to target?", "pass", f"Avg analyst target ${tm:,.0f} is {up:.0f}% above today's price.")
            elif up >= 0:
                add("Upside to target?", "warn", f"Avg target ${tm:,.0f} is only {up:.0f}% above today — limited room.")
            else:
                add("Upside to target?", "fail", f"Price is already above the avg analyst target (${tm:,.0f}).")
        aa = fundamentals.get("analyst_actions")
        if aa and (aa.get("n_up") or aa.get("n_down")):
            nu, nd, lt = aa.get("n_up", 0), aa.get("n_down", 0), aa.get("latest", {})
            firm = lt.get("firm") or "an analyst"
            if nu > nd:
                add("Recent rating change?", "fail" if short else "pass",
                    f"{nu} upgrade{'s' if nu != 1 else ''} vs {nd} downgrade{'s' if nd != 1 else ''} in 60d "
                    f"(latest: {firm} → {lt.get('to','')})" + (" — a headwind for a short." if short else "."))
            elif nd > nu:
                add("Recent rating change?", "pass" if short else "warn",
                    f"{nd} downgrade{'s' if nd != 1 else ''} vs {nu} upgrade{'s' if nu != 1 else ''} in 60d "
                    f"(latest: {firm} → {lt.get('to','')})" + ("." if short else " — analysts cooling, a caution."))
            else:
                add("Recent rating change?", "warn", f"Mixed analyst actions lately ({nu} up / {nd} down).")
        ed = fundamentals.get("earnings_days")
        if ed is not None:
            fresh = action in ("BUY", "SHORT")
            if ed <= 2 and fresh:
                add("Earnings clear?", "fail",
                    f"Earnings in {ed} day{'s' if ed != 1 else ''} — too close to open a fresh position; "
                    "a binary report can gap straight through your stop.")
            elif ed <= 7:
                add("Earnings clear?", "warn",
                    f"Earnings in {ed} day{'s' if ed != 1 else ''} — a binary event that can gap the price either way.")
            else:
                add("Earnings clear?", "pass", f"No earnings for ~{ed} days, so no imminent event risk.")

        # Earnings momentum / quality (security-level micro): growing EPS + revenue and healthy
        # margins support a long; shrinking/negative growth supports a short. Direction-aware.
        q = fundamentals.get("quality")
        if q and (q.get("eps_growth") is not None or q.get("rev_growth") is not None):
            eg, rg = q.get("eps_growth"), q.get("rev_growth")
            nm = q.get("net_margin")
            _bits = []
            if eg is not None:
                _bits.append(f"EPS {eg:+.0f}% YoY")
            if rg is not None:
                _bits.append(f"revenue {rg:+.0f}% YoY")
            if nm is not None:
                _bits.append(f"{nm:.0f}% net margin")
            _txt = ", ".join(_bits)
            growing = (eg or 0) > 5 or (rg or 0) > 5
            shrinking = (eg if eg is not None else 0) < -5 or (rg if rg is not None else 0) < -5
            if short:
                if shrinking:
                    add("Earnings momentum?", "pass", f"{_txt} — deteriorating fundamentals support the short.", 0.6)
                elif growing:
                    add("Earnings momentum?", "fail", f"{_txt} — improving fundamentals fight a short here.", 0.6)
                else:
                    add("Earnings momentum?", "warn", f"{_txt} — flattish fundamentals, no strong fundamental edge.", 0.4)
            else:
                if growing:
                    add("Earnings momentum?", "pass", f"{_txt} — growing fundamentals back the long.", 0.6)
                elif shrinking:
                    add("Earnings momentum?", "fail", f"{_txt} — shrinking fundamentals undercut the long.", 0.6)
                else:
                    add("Earnings momentum?", "warn", f"{_txt} — flattish fundamentals, no strong fundamental edge.", 0.4)

    # Insider transactions (SEC Form 4) — open-market buys by insiders are a bullish tell;
    # heavy selling leans bearish. Direction-aware, only when we actually scraped data.
    if insider and insider.get("n_filings"):
        b, sl = insider.get("buys", 0), insider.get("sells", 0)
        if insider.get("cluster_buy"):
            add("Insiders buying?", "fail" if short else "pass",
                f"{b} recent insider open-market purchase{'s' if b != 1 else ''} — insiders buying their own "
                "stock" + (" is a headwind for a short." if short else "; a bullish vote of confidence."))
        elif sl >= 2 and sl > b:
            add("Insiders buying?", "pass" if short else "warn",
                f"{sl} recent insider sale{'s' if sl != 1 else ''}" + (" — leans bearish, supports the short."
                if short else " and little buying — insiders trimming, a mild caution."))
        else:
            add("Insiders buying?", "warn", "No clear insider buying/selling cluster in recent Form 4 filings.")

    # Retail buzz (StockTwits) — a soft, noisy crowd-sentiment read; informative, not decisive.
    if buzz and buzz.get("lean"):
        lean, pct = buzz["lean"], buzz.get("sentiment_pct")
        if lean == "mixed":
            add("Retail buzz?", "warn", f"StockTwits chatter is split ({pct}% bullish) — no clear crowd lean.")
        else:
            with_us = (lean == "bear") if short else (lean == "bull")
            add("Retail buzz?", "pass" if with_us else "warn",
                f"StockTwits leans {lean}ish ({pct}% bullish of tagged posts)"
                + (" — crowd is on this side." if with_us else " — crowd leans the other way (often contrarian)."))

    # News-driven idea (LLM read of recent headlines) — does a fresh catalyst back this direction?
    if news_idea and news_idea.get("direction"):
        nd = news_idea["direction"]  # 'bullish' | 'bearish'
        conf = news_idea.get("confidence", "medium")
        aligns = (nd == "bearish") if short else (nd == "bullish")
        reason = (news_idea.get("reason") or "").strip()
        _nw = getattr(cfg, "news_conviction_weight", 1.0)
        # Fresh, aligned, HIGH-confidence catalyst gets the heaviest news weight (still one check
        # among ~15, so it lifts a setup but can't carry a technically-weak one on its own).
        _cw = getattr(cfg, "catalyst_boost_weight", _nw) if (aligns and conf == "high") else _nw
        if aligns and conf in ("high", "medium"):
            add("News catalyst aligns?", "pass",
                f"A {conf}-confidence news read backs this {'short' if short else 'long'}: {reason}", _cw)
        elif not aligns and conf in ("high", "medium"):
            add("News catalyst aligns?", "fail",
                f"Recent news leans the other way ({nd}) — a headwind: {reason}", _nw)
        else:
            add("News catalyst aligns?", "warn",
                f"News read is {nd} but low-confidence — weak/ambiguous catalyst: {reason}", _nw)

    # Intraday confirmation — does the lower-timeframe chart agree with this daily trade?
    # A light nudge (weight 0.6), not a dominant factor; only when an intraday signal exists.
    if intraday and intraday.get("direction"):
        tf = getattr(cfg, "intraday_timeframe", "5min")
        iact = intraday.get("action", "")
        if iact not in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
            add("Intraday on side?", "warn",
                f"No clear {tf} intraday signal — no lower-timeframe confirmation either way.", 0.6)
        elif intraday.get("direction") == direction:
            add("Intraday on side?", "pass",
                f"The {tf} chart agrees — lower-timeframe momentum points the same way, confirming the entry.", 0.6)
        else:
            add("Intraday on side?", "fail",
                f"The {tf} chart disagrees — lower-timeframe momentum is currently against this "
                + ("short." if short else "long."), 0.6)

    # Sector momentum — is the name's sector moving the right way right now (an ongoing market
    # development, not a single headline)? Long wants a strong sector; short wants a weak one.
    if sector_pct is not None and getattr(cfg, "sector_conviction_weight", 0) > 0:
        sw = cfg.sector_conviction_weight
        strong, weak = sector_pct >= 60, sector_pct <= 40
        if (strong and not short) or (weak and short):
            add("Sector trending right?", "pass",
                f"Its sector has momentum with you — {sector_pct}% of the sector is above trend.", sw)
        elif (weak and not short) or (strong and short):
            add("Sector trending right?", "fail",
                f"Its sector is moving against you — {sector_pct}% of the sector is above trend.", sw)
        else:
            add("Sector trending right?", "warn",
                f"Its sector is mixed ({sector_pct}% above trend) — no clear sector momentum either way.", sw)

    # Short interest / squeeze risk. For a SHORT, a crowded short (high % of float / days-to-cover)
    # is a real squeeze danger; for a LONG, heavy shorting is potential squeeze FUEL to the upside.
    if short_interest:
        spf = short_interest.get("short_pct_float")
        dtc = short_interest.get("days_to_cover")
        if spf is not None or dtc is not None:
            crowded = (spf is not None and spf >= 15) or (dtc is not None and dtc >= 5)
            _si = (f"{spf}% of float short" if spf is not None else "") + \
                  (f"{', ' if spf is not None else ''}{dtc}d to cover" if dtc is not None else "")
            if short:
                if crowded:
                    add("Crowd already short?", "fail",
                        f"{_si} — a crowded short with real squeeze risk if it reverses.", 0.8)
                else:
                    add("Crowd already short?", "pass",
                        f"{_si} — not a crowded short, so lower squeeze risk.", 0.5)
            elif crowded:
                add("Short-squeeze fuel?", "pass",
                    f"{_si} — heavy shorting can fuel a squeeze higher, a tailwind for this long.", 0.5)

    # Retail / social attention (ApeWisdom: Reddit + WSB mentions). The noisiest input we
    # use, so it is only ever a *light* nudge (weight 0.3-0.4). A crowd piling into a name
    # adds momentum AND volatility: for a LONG that's mild fuel (with froth caution); for a
    # SHORT, crowding into the same name you're fading is a yellow flag (they can squeeze it).
    if retail:
        rk = retail.get("rank")
        chg = retail.get("change_pct")
        ment = retail.get("mentions")
        hot = (rk is not None and rk <= 15) or (chg is not None and chg >= 75)
        if hot and ment:
            _rt = f"#{rk} on retail boards" if rk is not None else f"{ment} mentions"
            if chg is not None:
                _rt += f", {'+' if chg >= 0 else ''}{chg}% vs yesterday"
            if short:
                add("Retail crowding in?", "warn",
                    f"{_rt} — a retail crowd piling into the name you're fading can squeeze it; size carefully.", 0.4)
            else:
                add("Retail wind behind it?", "pass",
                    f"{_rt} — rising retail attention can add momentum (watch for froth at the top).", 0.3)

    # Liquidity / execution quality (microstructure): is it practical to trade without paying up?
    if liquidity:
        tier = liquidity.get("tier")
        dv = liquidity.get("dollar_volume")
        _dvm = f"~${dv/1e6:.0f}M/day" if dv else ""
        if tier in ("mega", "very high", "high"):
            add("Liquid enough to trade?", "pass",
                f"Yes — {tier} liquidity ({_dvm}); tight spreads, easy fills.", 0.4)
        elif tier == "moderate":
            add("Liquid enough to trade?", "warn",
                f"So-so — {tier} liquidity ({_dvm}); wider spreads, mind size and slippage.", 0.4)
        else:
            add("Liquid enough to trade?", "fail",
                f"Thin — {tier} liquidity ({_dvm}); hard to fill cleanly, slippage will bite.", 0.6)

    # Reward:risk — with honest (structural) targets, a cramped target vs the stop is a weak trade.
    # This is the number to judge a small target by: payoff relative to what you risk.
    rr = (plan or {}).get("rr")
    if rr is not None:
        if rr >= 2.0:
            add("Reward:risk worth it?", "pass",
                f"Target is {rr:.1f}x the risk — a healthy payoff for the stop distance.")
        elif rr >= 1.0:
            add("Reward:risk worth it?", "warn",
                f"Target is only {rr:.1f}x the risk — modest room to the nearest level; size carefully.")
        else:
            add("Reward:risk worth it?", "fail",
                f"Target is just {rr:.1f}x the risk — the nearest level is closer than the stop, a poor payoff.")

    # Minervini VCP confirmation (long only). Only fires for names already in a Stage-2 uptrend, so
    # it REWARDS a tight breakout-ready base rather than penalising every non-VCP long. Like every
    # check it's subject to the learned-weight loop, so if it doesn't predict winners it self-retires.
    vcpd = (factors or {}).get("vcp")
    if not short and isinstance(vcpd, dict) and vcpd.get("stage2"):
        if vcpd.get("status") == "pass":
            add("VCP base setup?", "pass",
                f"Yes — a Minervini VCP in a Stage-2 uptrend, {str(vcpd.get('state','')).lower()} under "
                f"the ${vcpd.get('pivot')} pivot across {vcpd.get('num_contractions')} contractions.")
        else:
            add("VCP base setup?", "warn",
                f"Building — Stage-2 uptrend (trend score {vcpd.get('trend_score')}) but not yet a tight, "
                "breakout-ready VCP base.")

    # Parabolic exhaustion (short only): a blow-off that's cracking. Only rewards conviction once the
    # move actually breaks (state=actionable) — while it's still in markup it's a WARN, never a green
    # light, because shorting a still-climbing parabola is the fastest way to lose. Self-retires via
    # the learned-weight loop if it doesn't predict.
    psh = (factors or {}).get("pshort")
    if short and isinstance(psh, dict) and psh.get("state"):
        if psh.get("state") == "actionable":
            add("Parabolic exhaustion?", "pass",
                f"Yes — a parabolic blow-off (+{psh.get('ret5_pct')}% in 5d, {psh.get('atr_ext')} ATR above "
                f"its 20-day line) is cracking on {psh.get('vol_ratio')}x volume — textbook exhaustion short.")
        else:
            add("Parabolic exhaustion?", "warn",
                f"Extended (+{psh.get('ret5_pct')}% in 5d, {psh.get('atr_ext')} ATR stretched) but still in "
                "markup — wait for the first crack before shorting a climbing parabola.")

    # Stockbee Momentum Burst (long only): a sharp 3-5 day expansion out of a tight base. Rewards a
    # clean A/B breakout on volume; self-retires via the learned-weight loop if it stops predicting.
    burst = (factors or {}).get("burst")
    if not short and isinstance(burst, dict) and burst.get("valid"):
        _tg = ", ".join(burst.get("triggers", [])) or "expansion"
        add("Momentum burst?", "pass" if burst.get("rating") == "A" else "warn",
            f"{burst.get('rating')}-grade burst (+{burst.get('gain_pct')}% on {burst.get('vol_ratio')}x volume, "
            f"{_tg}) out of a {burst.get('base_width')}% base — stop {burst.get('risk_pct')}% away at the trigger low.")

    # Episodic Pivot (long only): a discrete-catalyst repricing out of a neglected base. Fires on a
    # gap + volume shock; catalyst family enriched when the news pipeline has a headline.
    ep = (factors or {}).get("ep")
    if not short and isinstance(ep, dict) and ep.get("valid"):
        _fam = str(ep.get("family", "EP")).replace("_", " ").title()
        add("Episodic pivot?", "pass" if ep.get("score", 0) >= 70 else "warn",
            f"{_fam} — +{ep.get('gap_pct')}% on {ep.get('vol_x')}x volume"
            + (" out of a neglected base" if ep.get("neglected") else "")
            + (", fresh catalyst" if ep.get("catalyst") else "") + ".")

    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    wsum = sum(c.get("weight", 1.0) for c in checks)
    score = sum(pts[c["status"]] * c.get("weight", 1.0) for c in checks) / wsum if wsum else 0.0
    label = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
    # Earnings gate: a fresh entry within ~2 days of a report is a coin-flip event, so never
    # let it carry High conviction — that keeps it out of the digest/alerts (which key on High).
    ed = (fundamentals or {}).get("earnings_days")
    gated = ed is not None and ed <= 2 and action in ("BUY", "SHORT")
    if gated and label == "High":
        label = "Medium"
    return {"score_pct": round(score * 100), "label": label,
            "passes": sum(1 for c in checks if c["status"] == "pass"),
            "total": len(checks), "checks": checks,
            "earnings_days": ed, "earnings_gated": gated}


def rescore(row: dict, cfg: Config, sentiment=None, fundamentals=None, tv=None, regime=None, insider=None, buzz=None, news_idea=None, intraday=None, sector_pct=None, short_interest=None, retail=None, liquidity=None, learned=None) -> None:
    """Recompute conviction + desk read for a shown row once research is fetched."""
    direction = row.get("direction", "LONG")
    _apply_fundamental_cap(row.get("plan"), fundamentals, cfg)
    conv = _conviction(row["action"], direction, row["rsi"], row["rel_volume"], row["plan"], row["context"], cfg,
                       row.get("factors"), row.get("patterns"), row.get("edge"),
                       sentiment=sentiment, fundamentals=fundamentals, price=row.get("price"), tv=tv,
                       regime=regime, insider=insider, buzz=buzz, news_idea=news_idea, intraday=intraday,
                       sector_pct=sector_pct, short_interest=short_interest, retail=retail,
                       liquidity=liquidity if liquidity is not None else row.get("liquidity"), learned=learned)
    row["conviction"] = conv
    row["desk_read"] = _desk_read(row["action"], direction, row["plan"], row["context"], conv,
                                  row.get("patterns"), row.get("edge"),
                                  sentiment=sentiment, fundamentals=fundamentals, price=row.get("price"))


def _desk_read(action, direction, plan, context, conviction, patterns=None, edge=None,
               sentiment=None, fundamentals=None, price=None) -> str:
    """A short risk-strategist read, composed from the computed levels + patterns."""
    stop, target = plan.get("stop"), plan.get("target")
    rr = plan.get("rr")
    stop_pct, tgt_pct = plan.get("stop_pct"), plan.get("target_pct")
    short = direction == "SHORT"
    conv = conviction["label"]
    bits = []
    intros = {
        "BUY": "In short: several methods line up to the upside — a possible spot to buy.",
        "HOLD LONG": "In short: it's still trending up, so you'd keep a long you already own.",
        "WATCH LONG": "In short: a bullish setup is forming but isn't confirmed — watch, don't buy yet.",
        "SHORT": "In short: several methods line up to the downside — a possible spot to short.",
        "HOLD SHORT": "In short: it's still trending down, so you'd keep a short you already have.",
        "WATCH SHORT": "In short: a bearish setup is forming but isn't confirmed — watch, don't short yet.",
        "EXIT": "In short: it just rolled over — this is where you'd sell a long.",
        "AVOID": "In short: below trend and weak — better to stay away than buy.",
        "FLAT": "In short: no clear trend yet — one to watch.",
    }
    bits.append(intros.get(action, intros["FLAT"]))
    if stop is not None and target is not None:
        if short:
            bits.append(
                f"The plan in plain terms: short around ${plan.get('entry'):,.2f}. If it rises to ${stop:,.2f} "
                f"(+{stop_pct}%), buy to cover and cut the loss. If it falls to ${target:,.2f} (−{tgt_pct}%), "
                f"cover for the win. So you'd be risking a little to aim for about {rr}× as much.")
        else:
            bits.append(
                f"The plan in plain terms: buy around ${plan.get('entry'):,.2f}. If it drops to ${stop:,.2f} "
                f"(−{stop_pct}%), sell to cut the loss. If it climbs to ${target:,.2f} (+{tgt_pct}%), take the win. "
                f"So you'd be risking a little to aim for about {rr}× as much.")
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
    # Actionable first (BUY/SHORT), then holds, exits, the watch tier, avoid, flat.
    # Within a tier, higher conviction then unusual volume float to the top.
    order = {"BUY": 0, "SHORT": 1, "HOLD LONG": 2, "HOLD SHORT": 3, "EXIT": 4,
             "WATCH LONG": 5, "WATCH SHORT": 6, "AVOID": 7, "FLAT": 8}
    action_rank = order.get(row["action"], 9)
    conv = (row.get("conviction") or {}).get("score_pct") or 0
    relvol = row["rel_volume"] or 0
    return (action_rank, -conv, -relvol)


# Populated by scan(); the dashboard surfaces these if nothing was found.
LAST_ERRORS: list[str] = []


def scan(cfg: Config, live: bool, pin: set | None = None, universe: list | None = None) -> list[dict]:
    LAST_ERRORS.clear()
    equity = cfg.starting_cash
    if universe is not None:           # explicit list (e.g. the intraday pass reuses daily names)
        symbols = list(universe)
    elif live and cfg.scan_market:
        symbols = build_universe(cfg)
    elif live:
        symbols = list(cfg.symbols)
    else:
        symbols = list(_FALLBACK)  # synthetic demo universe
    if pin:  # always analyse names we alerted today so the dashboard stays in line with alerts
        symbols = [*pin, *symbols]
    # Sanitize every path (movers, pins, intraday reuse): drop malformed tickers such as
    # SPAC units like "AAC'U" that the data API rejects, then de-dupe. Runs after the pin
    # merge so alerted names can't bypass the universe cleaner.
    symbols = list(dict.fromkeys(c for s in symbols if (c := _clean_symbol(s))))
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
        try:
            row = _analyse(sym, df, cfg, equity)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must never fail the whole build
            errs += 1
            if len(LAST_ERRORS) < 5:
                LAST_ERRORS.append(f"{sym}: analyse failed: {type(exc).__name__}: {exc}")
            continue
        if row:
            rows.append(row)
    if not rows:
        LAST_ERRORS.append(
            f"Scanned {len(symbols)} symbols: {errs} errored, {empty} returned no data, "
            f"0 usable. If on a free Alpaca plan, confirm market-data access (IEX feed)."
        )
    # --- relative strength across the just-scanned universe (vs the benchmark) ---
    try:
        if live:
            bench_df = get_bars(cfg.benchmark, cfg)
        else:
            bench_df = synthetic_bars(cfg.benchmark, n=cfg.lookback_days)
        bench_close = bench_df["close"] if bench_df is not None and len(bench_df) else None
        if bench_close is not None:
            scores = {}
            for row in rows:
                df = row.get("_df")
                if df is not None and len(df):
                    scores[row["symbol"]] = rs_mod.rs_score(
                        df["close"], bench_close, cfg.rs_lookbacks, cfg.rs_weights)
            ranked = rs_mod.rank_universe(scores)
            for row in rows:
                rf = ranked.get(row["symbol"])
                if rf:
                    row.setdefault("factors", {})["rs"] = rf
                    rescore(row, cfg)   # fold the RS check into conviction
    except Exception:  # noqa: BLE001 - RS is additive; never break the scan
        pass

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
