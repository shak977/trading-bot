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
    return {
        "symbol": symbol,
        "action": action,
        "price": round(price, 2),
        "rsi": round(float(last["rsi"]), 1),
        "fast_ma": round(float(last["fast"]), 2),
        "slow_ma": round(float(last["slow"]), 2),
        "rel_volume": None if np.isnan(relvol) else round(relvol, 2),
        "stop": round(stop_loss_price(price, cfg), 2) if signal == 1 else None,
        "target": round(take_profit_price(price, cfg), 2) if signal == 1 else None,
        "suggested_shares": qty,
        "as_of": str(sig.index[-1].date()),
        "chart": {
            "dates": [str(d.date()) for d in tail.index],
            "close": [round(float(x), 2) for x in tail["close"]],
            "fast": [None if np.isnan(x) else round(float(x), 2) for x in tail["fast"]],
            "slow": [None if np.isnan(x) else round(float(x), 2) for x in tail["slow"]],
        },
    }


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
