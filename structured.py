"""Structured signal output + uncertainty scoring.

Turns each actionable trade into one tidy, machine-readable record — the "signal contract" the
spec asks for — so every idea carries the same explicit fields instead of being scattered across
the card. It assembles values already computed elsewhere (conviction, plan, liquidity, macro fit,
meta verdict) and adds three derived reads:

  • expected return range + expected value  — from the target/stop and the conviction as a rough
    win probability (a ballpark, not a promise).
  • expected holding period  — how many sessions the target is, at the stock's typical daily move.
  • risk score + uncertainty score  — uncertainty rises when signals disagree, the macro regime is
    mixed, liquidity is thin or conviction is low. High uncertainty is *already* what makes the
    meta-model reduce or skip size, so this score explains that, it doesn't double-count it.

Pure assembly + arithmetic; never raises. Gated by cfg.structured_enabled.
"""
from __future__ import annotations

_TIER_SCORE = {"mega": 100, "very high": 90, "high": 80, "moderate": 55, "thin": 30, "illiquid": 10}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build(row: dict, macro_posture: dict | None, cfg) -> dict | None:
    """Return the structured signal record for one actionable row, or None if disabled."""
    if cfg is not None and not getattr(cfg, "structured_enabled", True):
        return None
    try:
        direction = row.get("direction", "LONG")
        conv = (row.get("conviction") or {}).get("score_pct")
        plan = row.get("plan") or {}
        ctx = row.get("context") or {}
        tgt = plan.get("target_pct")
        stop = plan.get("stop_pct")
        rr = plan.get("rr")
        atr = ctx.get("atr_pct")
        meta = row.get("meta") or {}
        lab = (macro_posture or {}).get("label")

        # expected value + range (win prob ~ conviction)
        p = _clamp((conv or 50) / 100.0, 0.05, 0.95)
        up = abs(tgt) if tgt is not None else None
        dn = abs(stop) if stop is not None else None
        ev = round(p * up - (1 - p) * dn, 1) if (up is not None and dn is not None) else None
        ret_range = {"downside_pct": (-dn if dn is not None else None),
                     "upside_pct": (up if up is not None else None)}

        # expected holding period — sessions to traverse the target at the typical daily move
        hold = round(_clamp((up / atr) if (up and atr) else 5, 1, 60)) if up else None

        # scores
        liq_tier = (row.get("liquidity") or {}).get("tier")
        liq_score = _TIER_SCORE.get(liq_tier, 60)
        risk_score = round(_clamp((atr or 4) / 8 * 100, 0, 100) * 0.7 + (100 - liq_score) * 0.3)
        macro_fit = (row.get("rank_factors") or {}).get("macrofit")

        # conflict flag
        conflict = (row.get("intraday_confirm") == "disagree"
                    or meta.get("decision") in ("delay",)
                    or bool((row.get("news_idea") or {}).get("ticker") and False))  # reserved

        # uncertainty (0-100) — what drives the meta-model to reduce/skip
        u = 0
        if lab == "Neutral":
            u += 25
        if liq_tier in ("thin", "moderate", "illiquid"):
            u += 25
        if conflict:
            u += 30
        if (conv or 100) < 60:
            u += 20
        if "High-volatility" in {t.get("tag") for t in (macro_posture or {}).get("tags", [])}:
            u += 15
        uncertainty = _clamp(u, 0, 100)
        u_band = "high" if uncertainty >= 60 else "moderate" if uncertainty >= 30 else "low"

        # size recommendation — from the meta verdict (which already folds in uncertainty)
        dec = meta.get("decision", "accept")
        sf = meta.get("size_factor", 1.0)
        size_rec = ("Skip" if dec in ("reject", "delay") else
                    "Full" if sf >= 0.99 else "Half" if sf >= 0.45 else "Quarter")

        return {
            "symbol": row.get("symbol"),
            "direction": direction,
            "action": row.get("action"),
            "confidence": conv,
            "expected_value_pct": ev,
            "return_range": ret_range,
            "rr": rr,
            "expected_hold_days": hold,
            "risk_score": risk_score,
            "liquidity_score": liq_score,
            "macro_fit": macro_fit,
            "conflict": bool(conflict),
            "uncertainty": uncertainty,
            "uncertainty_band": u_band,
            "news_read": (row.get("nlp") or {}).get("net"),
            "size_recommendation": size_rec,
            "meta_decision": dec,
            "kill_conditions": {"stop_pct": stop,
                                "book_drawdown_halt_pct": getattr(cfg, "dd_halt_pct", None),
                                "daily_loss_limit_pct": getattr(cfg, "daily_loss_limit_pct", None)},
        }
    except Exception:  # noqa: BLE001 — structured output must never break the build
        return None
