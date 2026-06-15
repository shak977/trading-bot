"""Adaptive asset ranking — allocate capital to the best setups, not every signal.

The confluence/conviction score answers "is this a good *trade*?" (probability/quality). This
ranker answers the allocation question on top of it: "given limited capital, which of today's
actionable names deserve it *first*?" It blends, transparently:

  • Quality      — the conviction score (which already folds in trend, momentum, fundamentals,
                   sentiment, earnings quality, liquidity and R:R).
  • Vol-adj reward — reward-to-risk, throttled when volatility is extreme (capital efficiency).
  • Macro fit    — does the trade's direction agree with the macro regime? (long in risk-on, etc.)
  • Liquidity    — execution quality / cost, by dollar-turnover tier.
  • Momentum     — trend strength (distance the right way from the trend line), direction-aware.

Each factor is 0–100, weighted into a single rank score. Fully rules-based and explainable — you
can see exactly why one name outranks another. Gated by cfg.rank_enabled; never raises.
"""
from __future__ import annotations

_TIER_SCORE = {"mega": 100, "very high": 90, "high": 80, "moderate": 55, "thin": 30, "illiquid": 10}

_WEIGHTS = {"quality": 0.45, "vreward": 0.20, "macrofit": 0.15, "liquidity": 0.10, "momentum": 0.10}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rank_rows(rows: list[dict], macro_posture: dict | None, cfg) -> list[dict]:
    """Score + sort actionable rows. Mutates each row in place (adds rank_score/rank/rank_factors)
    and returns a compact ranked list for display. Returns [] if disabled or nothing actionable."""
    if not getattr(cfg, "rank_enabled", True):
        return []
    try:
        reg = (macro_posture or {}).get("label")
        guard = float(getattr(cfg, "vol_guard_mult", 1.5)) * float(getattr(cfg, "vol_target_atr_pct", 4.0))
        scored = []
        for r in rows:
            if r.get("action") not in ("BUY", "SHORT", "HOLD LONG", "HOLD SHORT"):
                continue
            conv = (r.get("conviction") or {}).get("score_pct") or 0
            plan = r.get("plan") or {}
            ctx = r.get("context") or {}
            direction = r.get("direction", "LONG")
            short = direction == "SHORT"

            # 1) quality — the conviction score
            quality = _clamp(float(conv), 0, 100)

            # 2) vol-adjusted reward — reward:risk, throttled when volatility is extreme
            rr = plan.get("rr")
            atr = ctx.get("atr_pct")
            if rr is not None:
                vreward = _clamp(float(rr) / 3.0 * 100, 0, 100)
                if atr and atr > guard:
                    vreward *= guard / atr
            else:
                vreward = 50.0

            # 3) macro fit — does direction agree with the regime?
            if reg == "Risk-on":
                macrofit = 85 if not short else 30
            elif reg == "Risk-off":
                macrofit = 85 if short else 30
            else:
                macrofit = 60

            # 4) liquidity / execution cost
            liq = _TIER_SCORE.get((r.get("liquidity") or {}).get("tier"), 60)

            # 5) momentum / trend strength — distance the *right* way from the trend line
            vs = ctx.get("vs_slow_ma_pct")
            if vs is None:
                momentum = 50.0
            else:
                aligned = (-vs if short else vs)   # for a short, below-trend is strength
                momentum = _clamp(aligned / 15.0 * 100, 0, 100)

            score = (_WEIGHTS["quality"] * quality + _WEIGHTS["vreward"] * vreward
                     + _WEIGHTS["macrofit"] * macrofit + _WEIGHTS["liquidity"] * liq
                     + _WEIGHTS["momentum"] * momentum)
            factors = {"quality": round(quality), "vreward": round(vreward),
                       "macrofit": round(macrofit), "liquidity": round(liq), "momentum": round(momentum)}
            scored.append((r, round(score, 1), factors))

        scored.sort(key=lambda x: -x[1])
        ranked = []
        for i, (r, sc, f) in enumerate(scored, 1):
            r["rank_score"] = sc
            r["rank"] = i
            r["rank_factors"] = f
            ranked.append({"symbol": r["symbol"], "name": r.get("name", ""), "action": r["action"],
                           "direction": r.get("direction", "LONG"), "sector": r.get("sector"),
                           "rank_score": sc, "factors": f,
                           "conviction": (r.get("conviction") or {}).get("score_pct")})
        return ranked
    except Exception:  # noqa: BLE001 — ranking must never break the build
        return []
