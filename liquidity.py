"""Execution / liquidity gate — market microstructure controls EXECUTION.

A signal can be perfect and still be a bad *trade* if you can't get in and out without paying a
big spread or pushing the price. This module estimates how practical a name is to trade and gates
sizing accordingly. Two halves:

  • classify(dollar_volume, price)  — size-independent: a liquidity tier + an estimated spread
    (bps), from average daily dollar turnover. Used as a soft flag on every signal.
  • practical(position_value, …)    — size-dependent: estimated market impact for the intended
    position (square-root model on % of ADV) + a yes/no on whether the trade is practical.

We can't see the real order book from daily bars, so the spread is a calibrated proxy by turnover
tier — good enough to separate "trades like water" from "you'll move it." Everything is best-effort
and never raises. Gated by cfg.liquidity_enabled.
"""
from __future__ import annotations

import math

# (min avg $ volume/day, tier label, estimated half-spread in bps)
_TIERS = [
    (1_000_000_000, "mega",      1.5),
    (200_000_000,   "very high", 3.0),
    (50_000_000,    "high",      6.0),
    (10_000_000,    "moderate",  15.0),
    (2_000_000,     "thin",      35.0),
    (0,             "illiquid",  80.0),
]


def classify(dollar_volume: float | None, price: float | None = None) -> dict | None:
    """Size-independent liquidity read. Returns {dollar_volume, tier, spread_bps} or None."""
    if not dollar_volume or dollar_volume <= 0:
        return None
    for floor, tier, spread in _TIERS:
        if dollar_volume >= floor:
            return {"dollar_volume": round(dollar_volume), "tier": tier, "spread_bps": spread}
    return None


def practical(position_value: float, dollar_volume: float | None, cfg,
              spread_bps: float | None = None) -> dict:
    """Size-dependent practicality read for a position of `position_value` dollars.
    Returns {ok, pct_adv, impact_bps, cost_bps, note}. `ok` is False if the name is too thin or
    the position would be too large a share of average daily volume."""
    out = {"ok": True, "pct_adv": None, "impact_bps": None, "cost_bps": None, "note": ""}
    if not getattr(cfg, "liquidity_enabled", True):
        return out
    if not dollar_volume or dollar_volume <= 0:
        out["ok"] = False
        out["note"] = "No volume data — can't confirm it's liquid enough to trade."
        return out
    if dollar_volume < float(getattr(cfg, "min_dollar_volume", 5_000_000)):
        out["ok"] = False
        out["note"] = f"Thin — only ~${dollar_volume/1e6:.1f}M/day turnover, below the liquidity floor."
        return out
    pct_adv = (position_value / dollar_volume) if position_value else 0.0
    out["pct_adv"] = round(pct_adv * 100, 2)
    # square-root market-impact model: ~10 bps at 1% of ADV, scaling with sqrt(size)
    impact = 100.0 * math.sqrt(max(pct_adv, 0.0))
    out["impact_bps"] = round(impact, 1)
    out["cost_bps"] = round((spread_bps or 0.0) + impact, 1)
    max_pct = float(getattr(cfg, "max_pct_of_adv", 0.02))
    if pct_adv > max_pct:
        out["ok"] = False
        out["note"] = (f"Position would be {pct_adv*100:.1f}% of daily volume (cap {max_pct*100:.0f}%) — "
                       "too large to fill cleanly; trimming.")
    else:
        out["note"] = f"OK — ~{pct_adv*100:.2f}% of ADV, est. cost ~{out['cost_bps']:.0f} bps."
    return out


def cap_qty_to_adv(qty: int, entry: float, dollar_volume: float | None, cfg) -> int:
    """Trim a position so its notional stays within max_pct_of_adv of average daily $ volume."""
    if not getattr(cfg, "liquidity_enabled", True):
        return qty
    if not dollar_volume or not entry or entry <= 0:
        return qty
    max_notional = float(getattr(cfg, "max_pct_of_adv", 0.02)) * dollar_volume
    cap = int(max_notional / entry)
    return max(0, min(qty, cap))
