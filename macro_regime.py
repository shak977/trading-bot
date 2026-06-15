"""Macro regime → exposure engine.

The guiding principle from the spec: **macro data controls exposure, it is never a direct
buy/sell trigger.** This module reads the cross-asset backdrop (volatility, the yield curve,
credit spreads, the dollar) plus equity breadth, scores each driver from risk-off (−1) to
risk-on (+1), blends them into one composite, and maps that to an **exposure multiplier** that
scales how aggressively the bot sizes new positions:

  • Risk-on  → larger size, lean into momentum / cyclicals.
  • Neutral  → normal size.
  • Risk-off → smaller size, more cash, tighten up, favour defensives.

Each driver is transparent (you can see what pushed the dial), and the whole thing is wrapped so
missing data just drops that driver rather than breaking anything. Gated by cfg.macro_regime_enabled.
"""
from __future__ import annotations


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _vix_score(macro) -> tuple[float, str] | None:
    v = macro.get("vix")
    if v is None:
        return None
    if v < 16:
        s, txt = 0.9, f"VIX {v} — calm"
    elif v < 20:
        s, txt = 0.3, f"VIX {v} — normal"
    elif v < 25:
        s, txt = -0.3, f"VIX {v} — edgy"
    elif v < 30:
        s, txt = -0.7, f"VIX {v} — fearful"
    else:
        s, txt = -1.0, f"VIX {v} — stressed"
    if macro.get("vix_trend") == "rising":
        s -= 0.2; txt += ", rising"
    elif macro.get("vix_trend") == "falling":
        s += 0.2; txt += ", falling"
    return _clamp(s, -1, 1), txt


def _curve_score(macro) -> tuple[float, str] | None:
    c = macro.get("curve")
    if c is None:
        return None
    if c > 0.5:
        s, txt = 0.5, f"Curve +{c} — normal/steep"
    elif c >= 0:
        s, txt = 0.1, f"Curve +{c} — flat"
    elif c > -0.5:
        s, txt = -0.4, f"Curve {c} — slightly inverted"
    else:
        s, txt = -0.8, f"Curve {c} — deeply inverted"
    return s, txt


def _credit_score(macro) -> tuple[float, str] | None:
    oas = macro.get("hy_oas")
    if oas is None:
        return None
    if oas < 3.5:
        s, txt = 0.8, f"HY spread {oas}% — tight"
    elif oas < 5:
        s, txt = 0.2, f"HY spread {oas}% — normal"
    elif oas < 7:
        s, txt = -0.4, f"HY spread {oas}% — widening risk"
    else:
        s, txt = -1.0, f"HY spread {oas}% — credit stress"
    chg = macro.get("hy_oas_chg_1mo")
    if chg is not None:
        if chg > 0.3:
            s -= 0.4; txt += ", widening fast"
        elif chg > 0.05:
            s -= 0.2; txt += ", widening"
        elif chg < -0.05:
            s += 0.3; txt += ", tightening"
    return _clamp(s, -1, 1), txt


def _usd_score(macro) -> tuple[float, str] | None:
    chg = macro.get("dxy_chg_1mo")
    if chg is None:
        return None
    # A fast-strengthening dollar tightens global financial conditions → risk-off.
    if chg > 2:
        s, txt = -0.4, f"USD +{chg}%/mo — tightening"
    elif chg > 1:
        s, txt = -0.2, f"USD +{chg}%/mo — firmer"
    elif chg < -2:
        s, txt = 0.3, f"USD {chg}%/mo — easing"
    else:
        s, txt = 0.0, f"USD {chg:+}%/mo — steady"
    return s, txt


def _breadth_score(tape) -> tuple[float, str] | None:
    if not tape:
        return None
    b = tape.get("breadth")
    if b is None:
        return None
    if b >= 60:
        s, txt = 0.5, f"Breadth {b}% — broad strength"
    elif b <= 40:
        s, txt = -0.5, f"Breadth {b}% — broad weakness"
    else:
        s, txt = 0.0, f"Breadth {b}% — mixed"
    return s, txt


# driver -> weight (credit + vol are the most reliable risk gauges)
_WEIGHTS = {"VIX": 0.25, "Credit": 0.30, "Yield curve": 0.15, "US dollar": 0.10, "Breadth": 0.20}


def assess(macro: dict | None, tape_regime: dict | None, cfg) -> dict | None:
    """Blend the macro backdrop + equity breadth into a composite posture and an exposure
    multiplier. Returns None if disabled or there's nothing to score. Never raises."""
    if not getattr(cfg, "macro_regime_enabled", True):
        return None
    if not macro and not tape_regime:
        return None
    try:
        macro = macro or {}
        scorers = {
            "VIX": _vix_score(macro),
            "Credit": _credit_score(macro),
            "Yield curve": _curve_score(macro),
            "US dollar": _usd_score(macro),
            "Breadth": _breadth_score(tape_regime),
        }
        drivers, wsum, used = [], 0.0, 0.0
        for name, res in scorers.items():
            if not res:
                continue
            score, txt = res
            w = _WEIGHTS[name]
            wsum += score * w
            used += w
            drivers.append({"name": name, "score": round(score, 2), "read": txt, "weight": w})
        if used <= 0:
            return None
        composite = _clamp(wsum / used, -1, 1)   # renormalise by weights actually used

        if composite >= 0.25:
            label = "Risk-on"
            posture = ("Macro backdrop is supportive — sizing leans in; favour momentum, cyclicals "
                       "and higher-beta names.")
        elif composite <= -0.25:
            label = "Risk-off"
            posture = ("Macro backdrop is defensive — sizing pulls back, hold more cash, tighten risk "
                       "and favour defensives over high-beta.")
        else:
            label = "Neutral"
            posture = "Macro backdrop is mixed — sizing stays roughly normal; no strong macro tilt either way."

        base = float(getattr(cfg, "macro_exposure_base", 0.95))
        slope = float(getattr(cfg, "macro_exposure_slope", 0.30))
        lo = float(getattr(cfg, "macro_exposure_min", 0.50))
        hi = float(getattr(cfg, "macro_exposure_max", 1.20))
        exposure_mult = round(_clamp(base + slope * composite, lo, hi), 3)

        drivers.sort(key=lambda d: d["score"])  # most negative (risk-off) first
        return {
            "label": label,
            "score": round(composite, 2),
            "exposure_mult": exposure_mult,
            "posture": posture,
            "drivers": drivers,
            "cash_tilt_pct": round((1 - exposure_mult) * 100) if exposure_mult < 1 else 0,
        }
    except Exception:  # noqa: BLE001 — macro must never break the build
        return None
