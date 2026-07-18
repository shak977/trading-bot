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
def _finconditions_score(macro) -> tuple[float, str] | None:
    n = macro.get("nfci")
    if n is None:
        return None
    # NFCI: negative = looser-than-average conditions (risk-on), positive = tighter (risk-off); ~0 = average.
    s = _clamp(-n / 0.4, -1, 1)
    if n < -0.1:
        txt = f"Financial conditions {n:+.2f} — loose"
    elif n > 0.1:
        txt = f"Financial conditions {n:+.2f} — tight"
    else:
        txt = f"Financial conditions {n:+.2f} — average"
    if macro.get("nfci_trend") == "tightening":
        s -= 0.15; txt += ", tightening"
    elif macro.get("nfci_trend") == "easing":
        s += 0.15; txt += ", easing"
    return _clamp(s, -1, 1), txt


_WEIGHTS = {"VIX": 0.25, "Credit": 0.28, "Yield curve": 0.15, "US dollar": 0.08,
            "Breadth": 0.18, "Financial conditions": 0.22}


def _regime_tags(macro: dict, tape_regime: dict | None) -> list[dict]:
    """Secondary, possibly co-occurring regime labels beyond the risk-on/off axis.
    Each tag is {tag, why}. These describe the *character* of the tape, not just its risk level."""
    tags = []
    vix = macro.get("vix")
    curve = macro.get("curve")
    hy_widening = macro.get("hy_trend") == "widening"
    cpi = macro.get("cpi_yoy")
    oil = macro.get("oil")
    hy = macro.get("hy_oas")
    breadth = (tape_regime or {}).get("breadth")

    if vix is not None and (vix >= 25 or (vix >= 20 and macro.get("vix_trend") == "rising")):
        tags.append({"tag": "High-volatility", "why": f"VIX {vix}{' rising' if macro.get('vix_trend')=='rising' else ''} — choppy, fast tape."})
    # recessionary: inverted curve + a corroborating stress signal
    if curve is not None and curve < 0 and (hy_widening or (breadth is not None and breadth <= 40)):
        why = "inverted yield curve" + (", widening credit" if hy_widening else ", weak breadth")
        tags.append({"tag": "Recessionary", "why": f"{why} — late-cycle / contraction risk."})
    # inflationary: hot CPI (and/or firm oil)
    if cpi is not None and cpi >= 3.5:
        tags.append({"tag": "Inflationary", "why": f"CPI {cpi}% — sticky inflation; favours real assets over long-duration growth."})
    elif cpi is None and oil is not None and oil >= 95:
        tags.append({"tag": "Inflationary", "why": f"oil ${oil} firm — commodity-led price pressure."})
    # liquidity-driven: calm vol + tight credit + broad participation (easy-money risk appetite)
    if (vix is not None and vix < 16 and hy is not None and hy < 3.5
            and breadth is not None and breadth >= 60):
        tags.append({"tag": "Liquidity-driven", "why": "low vol, tight credit, broad breadth — risk appetite running on easy conditions."})
    return tags


def _strategy_bias(label: str, tags: list[dict]) -> dict:
    """Map the regime to which strategies to lean on vs. ease off — the 'strategy selection' the
    spec asks the regime to drive. Transparent, not a hard switch: it tilts emphasis, not rules."""
    tagset = {t["tag"] for t in tags}
    favored, caution = [], []

    if label == "Risk-on":
        favored += ["Momentum / trend", "Breakouts", "Long bias"]
        caution += ["Fresh shorts"]
    elif label == "Risk-off":
        favored += ["Defensives & quality", "Cash buffer", "Shorts on breakdowns"]
        caution += ["High-beta longs", "Adding gross exposure"]
    else:
        favored += ["Selective both ways", "Mean-reversion in range"]
        caution += ["Aggressive trend bets"]

    if "High-volatility" in tagset:
        favored += ["Mean-reversion / pairs", "Smaller size, wait for confirmation"]
        caution += ["Breakout chasing", "Oversized positions"]
    if "Recessionary" in tagset:
        favored += ["Defensives", "Reduced net long"]
        caution += ["Cyclical / high-beta longs"]
    if "Inflationary" in tagset:
        favored += ["Energy / materials / real assets", "Pricing-power names"]
        caution += ["Long-duration growth / rate-sensitive"]
    if "Liquidity-driven" in tagset:
        favored += ["Momentum (ride it)"]
        caution += ["Complacency — watch for a vol spike"]

    # de-dupe, preserve order
    favored = list(dict.fromkeys(favored))
    caution = list(dict.fromkeys(caution))
    note = "Regime tilts emphasis (favoured vs. caution); it never overrides the rules-based risk engine."
    return {"favored": favored, "caution": caution, "note": note}


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
            "Financial conditions": _finconditions_score(macro),
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
        tags = _regime_tags(macro, tape_regime)
        strategy_bias = _strategy_bias(label, tags)
        return {
            "label": label,
            "score": round(composite, 2),
            "exposure_mult": exposure_mult,
            "posture": posture,
            "drivers": drivers,
            "tags": tags,
            "strategy_bias": strategy_bias,
            "cash_tilt_pct": round((1 - exposure_mult) * 100) if exposure_mult < 1 else 0,
        }
    except Exception:  # noqa: BLE001 — macro must never break the build
        return None
