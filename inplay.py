"""Stocks-in-play selector — which names are worth running the ORB strategy on today.

ORB only works on names that are actually *in play*: gapping, trading heavy volume, with a catalyst,
and liquid enough to get in and out cleanly. This module turns those signals into one transparent
in-play score so the morning watchlist is ranked, not arbitrary.

It is a pure scorer: the caller passes candidates with the fields it already computes elsewhere
(gap %, relative volume, a news-catalyst flag, a liquidity tier), and this returns them ranked with
a 0–100 score and the component breakdown. No fetching, never raises.

Score = weighted blend of:
  • gap     — overnight gap size (abs %), saturating (a 3% gap and a 30% gap are both "gapping").
  • rvol    — relative volume vs the name's own average (the single best "in play" tell).
  • catalyst— a fresh news catalyst today (binary nudge).
  • liquid  — liquidity tier (penalises thin names you can't trade cleanly).
"""
from __future__ import annotations

_LIQ_SCORE = {"mega": 100, "very high": 92, "high": 80, "moderate": 55, "thin": 28, "illiquid": 8}

# component weights (sum = 1.0)
_W = {"gap": 0.30, "rvol": 0.40, "catalyst": 0.15, "liquid": 0.15}


def _saturate(x: float, full: float) -> float:
    """0..100 that reaches ~100 as x approaches `full`, then plateaus (diminishing returns)."""
    if x is None or x <= 0:
        return 0.0
    return round(100.0 * (1.0 - 1.0 / (1.0 + x / max(full, 1e-9))), 1)


def score_one(c: dict) -> dict:
    """Score a single candidate dict → {symbol, in_play, components, in_play_band}.

    Expected keys (all optional, default to neutral/zero):
      symbol, gap_pct, rel_volume, has_news (bool), liquidity_tier (str)."""
    gap = abs(c.get("gap_pct") or 0.0)
    rvol = c.get("rel_volume") or 0.0
    has_news = bool(c.get("has_news"))
    tier = (c.get("liquidity_tier") or "").lower()

    comp = {
        "gap": _saturate(gap, 5.0),            # ~5% gap ≈ strongly in play
        "rvol": _saturate(max(0.0, rvol - 1.0), 2.0),  # rvol 3x ≈ strongly in play; 1x = baseline 0
        "catalyst": 100.0 if has_news else 0.0,
        "liquid": float(_LIQ_SCORE.get(tier, 50)),
    }
    score = round(sum(_W[k] * comp[k] for k in _W), 1)
    band = "hot" if score >= 65 else "warm" if score >= 40 else "quiet"
    return {"symbol": c.get("symbol"), "in_play": score, "in_play_band": band,
            "components": {k: round(v, 1) for k, v in comp.items()},
            "gap_pct": round(gap, 2), "rel_volume": round(rvol, 2),
            "has_news": has_news, "liquidity_tier": tier or None}


def rank(candidates: list[dict], cfg=None, top: int | None = None,
         min_score: float = 35.0) -> list[dict]:
    """Rank candidates by in-play score; keep those clearing `min_score`, cap at `top`.

    `top` defaults to cfg.orb_inplay_top (or all). Liquid names with no edge are filtered out so ORB
    only runs on genuinely active names. Never raises — returns [] on bad input."""
    try:
        scored = [score_one(c) for c in (candidates or []) if c.get("symbol")]
        scored = [s for s in scored if s["in_play"] >= min_score]
        scored.sort(key=lambda s: -s["in_play"])
        if top is None and cfg is not None:
            top = getattr(cfg, "orb_inplay_top", None)
        return scored[:top] if top else scored
    except Exception:  # noqa: BLE001 — selection must never break the build
        return []
