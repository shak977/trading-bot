"""SKILL BRIDGE — connects the vendored, already-tested trading skills to the live engine.

Context: 71 skills are vendored into this repo; only 9 were ever wired in. This module plugs in the
three that improve RISK, PLANNING and DISCIPLINE around the signals we already generate — chosen
deliberately over adding another entry signal, because two new signals (pullback, VCP-as-trigger)
both failed their backtests. These three can't lose money by being wrong; at worst they advise
something you ignore.

  1. breakout-trade-planner  -> Minervini-style rating bands + sizing multipliers + portfolio-heat
                                aware position sizing, applied to our best pattern (VCP).
  2. exposure-coach          -> a whole-account exposure CEILING from breadth + regime. We size
                                individual trades well but never capped total exposure.
  3. pre-trade-discipline-gate -> blocks planless / oversized / regime-blocked entries.

Every import is guarded: if a skill is missing the engine carries on untouched. Nothing here places
an order — it produces advice the dashboard shows and the sizer can respect.
"""
from __future__ import annotations

import os
import sys

_SKILLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "vendor", "claude-trading-skills", "skills")


def _skill_module(skill: str, script: str):
    """Import a script from a vendored skill without polluting sys.path permanently."""
    path = os.path.join(_SKILLS, skill, "scripts")
    if not os.path.isdir(path):
        return None
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        return __import__(script)
    except Exception:  # noqa: BLE001
        return None
    finally:
        if added and path in sys.path:
            sys.path.remove(path)


# ---------------------------------------------------------------- 1. breakout trade planner
def rate_setup(signal: dict) -> dict | None:
    """Grade a signal the way the breakout planner does: a rating band (textbook/strong/good/
    developing/weak) and the size multiplier that band earns. Driven by the model's P(win) when we
    have it (validated, AUC ~0.75) rather than the conviction score (proven inverted).

    Returns {band, size_mult, basis} or None if the skill isn't available.
    """
    rc = _skill_module("breakout-trade-planner", "risk_calculator")
    if rc is None:
        return None
    pw = signal.get("p_win")
    if isinstance(pw, (int, float)):
        composite, basis = pw * 100.0, "model win-probability"
    else:
        conv = (signal.get("conviction") or {}).get("score_pct")
        if not isinstance(conv, (int, float)):
            return None
        composite, basis = float(conv), "conviction (pre-model fallback)"
    try:
        band = rc.get_rating_band(composite)
        return {"band": band, "size_mult": rc.get_sizing_multiplier(band),
                "composite": round(composite, 1), "basis": basis}
    except Exception:  # noqa: BLE001
        return None


def plan_position(signal: dict, account_size: float, base_risk_pct: float,
                  sector_exposure_pct: float = 0.0) -> dict | None:
    """Portfolio-heat-aware position size for a signal, from the breakout planner's sizer. Respects
    per-position and per-sector caps and reports which constraint bound the size — something our own
    sizer doesn't surface."""
    rc = _skill_module("breakout-trade-planner", "risk_calculator")
    plan = signal.get("plan") or {}
    entry, stop = plan.get("entry"), plan.get("stop")
    if rc is None or not entry or not stop or entry <= stop:
        return None
    rating = rate_setup(signal)
    if not rating:
        return None
    try:
        out = rc.calculate_position_size(
            worst_entry=float(entry), stop_loss=float(stop), account_size=float(account_size),
            base_risk_pct=float(base_risk_pct * 100.0), sizing_multiplier=rating["size_mult"],
            current_sector_exposure=float(sector_exposure_pct))
        out["band"] = rating["band"]
        out["size_mult"] = rating["size_mult"]
        return out
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 2. exposure coach
def exposure_ceiling(breadth_pct: float | None, regime_label: str | None,
                     regime_score: float | None = None) -> dict | None:
    """How much of the account should be exposed AT ALL, given breadth + regime. Our engine sizes
    each trade but never capped the total — this is the missing whole-book brake.

    Returns {ceiling_pct, recommendation, composite, bias} or None.
    """
    ec = _skill_module("exposure-coach", "calculate_exposure")
    if ec is None:
        return None
    try:
        # Map our regime words onto the skill's 0-100 scale when no numeric score is supplied.
        if regime_score is None:
            regime_score = {"Risk-on": 75.0, "Neutral": 50.0, "Risk-off": 25.0}.get(regime_label or "", 50.0)
        breadth = float(breadth_pct) if breadth_pct is not None else 50.0
        # The skill weights several inputs; we supply the two we genuinely have (breadth, regime)
        # and let it apply its own haircut for the ones we don't, rather than inventing values.
        composite, provided, missing = ec.calculate_composite_score(
            {"breadth": breadth, "regime": float(regime_score)})
        ceiling = ec.determine_exposure_ceiling(float(composite))
        missing_critical = len(set(missing) & getattr(ec, "CRITICAL_INPUTS", set()))
        rec = ec.determine_recommendation(float(composite), None, missing_critical)
        return {"ceiling_pct": ceiling, "recommendation": str(rec),
                "composite": round(float(composite), 1),
                "inputs_used": provided, "inputs_missing": missing,
                "note": ("score is haircut for missing inputs — treat the ceiling as conservative"
                         if missing_critical else ""),
                "basis": {"breadth": breadth, "regime": regime_label}}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 3. pre-trade discipline gate
def discipline_check(signal: dict, *, exposure: dict | None = None,
                     open_positions: int = 0, max_open: int = 30,
                     recent_losses: int = 0) -> dict:
    """A local pre-trade checklist. Deliberately implemented here (rather than shelling out to the
    skill's CLI, which expects its own journal files) so it can run inline on every signal, but it
    encodes the same rules: no plan, oversized, regime-blocked, or revenge-risk entries get blocked.

    Returns {ok, blocks:[...], warns:[...]} — advisory; nothing is auto-cancelled.
    """
    blocks, warns = [], []
    plan = signal.get("plan") or {}
    if not (plan.get("entry") and plan.get("stop") and plan.get("target")):
        blocks.append("No complete plan (entry/stop/target) — never enter without one.")
    rr = plan.get("rr")
    if isinstance(rr, (int, float)) and rr < 1.0:
        blocks.append(f"Reward:risk is {rr:.2f} — risking more than the trade can pay.")
    if signal.get("earnings_gated") or (signal.get("conviction") or {}).get("earnings_gated"):
        blocks.append("Earnings inside the window — a coin flip, not a setup.")
    if exposure and isinstance(exposure.get("ceiling_pct"), (int, float)):
        if str(exposure.get("recommendation", "")).lower().startswith("cash"):
            blocks.append(f"Market posture says cash-priority (exposure ceiling "
                          f"{exposure['ceiling_pct']}%) — not an environment for new entries.")
    if open_positions >= max_open:
        blocks.append(f"Already at the position cap ({open_positions}/{max_open}).")
    if recent_losses >= 3:
        warns.append(f"{recent_losses} recent losses — check this isn't a revenge trade.")
    pw = signal.get("p_win")
    if isinstance(pw, (int, float)) and pw < 0.45:
        warns.append(f"Model win-probability only {pw*100:.0f}%.")
    return {"ok": not blocks, "blocks": blocks, "warns": warns}


# ---------------------------------------------------------------- 4. CANSLIM screener
def canslim_score(signal: dict, regime: dict | None = None) -> dict | None:
    """O'Neil's CANSLIM growth score, computed from data we already collect.

    Deliberately wired as a QUALITY OVERLAY, not a trade trigger. Two new entry signals (pullback,
    VCP-as-trigger) failed their backtests this month; a third unproven trigger is the wrong move.
    As a score it costs nothing to be wrong — it annotates a signal rather than creating one — and
    if it proves predictive in the record we can promote it later on evidence.

    Component mapping from our own fields:
      C  current earnings  <- quality.eps_growth (latest reported growth)
      A  annual growth     <- quality.rev_growth + margin/ROE quality
      N  newness/new highs <- context.pct_from_high (near highs = strong)
      M  market direction  <- regime breadth + label (the 'M' O'Neil says decides 3 of 4 trades)
    """
    sc = _skill_module("canslim-screener", "scorer")
    if sc is None:
        return None
    q = ((signal.get("fundamentals") or {}).get("quality") or {})
    ctx = signal.get("context") or {}
    if not q and not ctx:
        return None

    def band(v, lo, hi):
        """Map a raw value onto 0-100 between lo (=0) and hi (=100)."""
        if v is None:
            return 50.0
        return max(0.0, min(100.0, (float(v) - lo) / (hi - lo) * 100.0))

    c = band(q.get("eps_growth"), -20, 40)                 # -20% -> 0, +40% -> 100
    a_growth = band(q.get("rev_growth"), -5, 30)
    a_qual = (band(q.get("net_margin"), 0, 25) + band(q.get("roe"), 0, 30)) / 2
    a = 0.6 * a_growth + 0.4 * a_qual
    pfh = ctx.get("pct_from_high")                          # negative = below the high
    n = band(pfh, -35, 0) if pfh is not None else 50.0
    reg = regime or {}
    m_breadth = band(reg.get("breadth"), 20, 75)
    m_label = {"Risk-on": 85.0, "Neutral": 50.0, "Risk-off": 20.0}.get(reg.get("label"), 50.0)
    m = 0.5 * m_breadth + 0.5 * m_label
    try:
        out = sc.calculate_composite_score(c_score=c, a_score=a, n_score=n, m_score=m)
        out["components"] = {"C": round(c), "A": round(a), "N": round(n), "M": round(m)}
        out["advisory"] = True     # never used as a trigger — see docstring
        return out
    except Exception:  # noqa: BLE001
        return None


def available() -> dict:
    """Which bridged skills actually loaded — surfaced on the dashboard so a missing one is visible
    rather than silently doing nothing (the failure mode the guardrails exist to prevent)."""
    return {
        "breakout-trade-planner": _skill_module("breakout-trade-planner", "risk_calculator") is not None,
        "exposure-coach": _skill_module("exposure-coach", "calculate_exposure") is not None,
        "canslim-screener": _skill_module("canslim-screener", "scorer") is not None,
        "pre-trade-discipline-gate": True,   # implemented inline
    }
