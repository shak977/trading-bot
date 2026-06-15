"""Meta-signal model — the second opinion on every trade candidate.

The primary signal model (the confluence/conviction engine) answers "is there a setup here?" The
meta-model answers the harder question: "knowing the regime, the liquidity, the conflicts and how
setups like this have actually paid off — should we take it, and at what size?"

It returns one of four verdicts, the way a desk head signs off on a junior's idea:
    • accept  — clean; trade at full (risk-managed) size.
    • reduce  — take it, but smaller (a soft headwind: high-vol, counter-regime, a mild conflict,
                or a regime that has underperformed historically).
    • delay   — the setup is there but unconfirmed; wait a beat (no lower-timeframe agreement yet).
    • reject  — don't trade (a hard veto: too illiquid, earnings imminent).

It reuses the existing gates (no-trade vetoes, the feedback loop's regime win-rates, the macro
regime) rather than inventing new logic — so the verdict is transparent and consistent with the
rest of the system. The LLM and ML never sit here; this is rules. Gated by cfg.meta_enabled.
"""
from __future__ import annotations


def evaluate(row: dict, *, macro_posture: dict | None = None,
             track: dict | None = None, cfg=None) -> dict | None:
    """Return {decision, size_factor, reasons} for one actionable candidate, or None if disabled."""
    if cfg is not None and not getattr(cfg, "meta_enabled", True):
        return None
    try:
        direction = row.get("direction", "LONG")
        short = direction == "SHORT"
        lab = (macro_posture or {}).get("label")
        tags = {t.get("tag") for t in (macro_posture or {}).get("tags", [])}
        reasons: list[str] = []
        size_factor = 1.0
        decision = "accept"

        # --- hard reject: per-name no-trade veto (illiquid / earnings imminent) ---
        try:
            import notrade
            veto = notrade.symbol_veto(row, cfg)
        except Exception:  # noqa: BLE001
            veto = None
        if veto and veto.get("status") == "block":
            return {"decision": "reject", "size_factor": 0.0, "reasons": [veto["reason"]]}

        # --- delay: setup present but the lower timeframe hasn't confirmed it yet ---
        if row.get("intraday_confirm") == "disagree":
            return {"decision": "delay", "size_factor": 0.0,
                    "reasons": ["intraday trend disagrees with the daily call — wait for it to line up"]}

        # --- soft headwinds → reduce ---
        # 1) high-volatility regime
        if "High-volatility" in tags:
            size_factor = min(size_factor, 0.5)
            reasons.append("high-volatility regime — half size, wait for confirmation")
        # 2) counter to the macro regime (still actionable, but into a headwind)
        counter = (lab == "Risk-on" and short) or (lab == "Risk-off" and not short)
        if counter:
            size_factor = min(size_factor, 0.5)
            reasons.append(f"trade runs counter to the {lab} macro regime")
        # 3) this regime has underperformed historically (feedback loop, once there's a sample)
        g = ((track or {}).get("by_regime") or {}).get(lab) if lab else None
        min_n = int(getattr(cfg, "notrade_perf_min_n", 25)) if cfg else 25
        if g and (g.get("n") or 0) >= min_n and g.get("win_rate") is not None and g["win_rate"] < 45:
            size_factor = min(size_factor, 0.5)
            reasons.append(f"setups in this regime have only won {g['win_rate']}% historically")
        # 4) mild conflict flagged by the veto layer
        if veto and veto.get("status") == "caution":
            size_factor = min(size_factor, 0.6)
            reasons.append(veto["reason"])
        # 5) LLM news-text read opposes the trade (structured scores from headlines)
        nlp = row.get("nlp") or {}
        net = nlp.get("net")
        if net is not None:
            opposes = (net <= -0.5 and not short) or (net >= 0.5 and short)
            if opposes:
                size_factor = min(size_factor, 0.6)
                reasons.append(f"news-text read leans against the trade (net {net:+.2f})")

        if size_factor < 1.0:
            decision = "reduce"
        if not reasons:
            reasons.append("clean setup — no regime, liquidity or conflict headwinds")
        return {"decision": decision, "size_factor": round(size_factor, 2), "reasons": reasons}
    except Exception:  # noqa: BLE001 — meta must never break the build
        return None
