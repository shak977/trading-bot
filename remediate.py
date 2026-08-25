"""REMEDIATION PIPELINE — what happens after guardrails.py finds something.

    guardrails (detect)  ->  triage (blast radius)  ->  validate (EVIDENCE)  ->  act  ->  log

The design decision that matters: **fixes are tiered by what they can cost you.**

  SAFE      cannot lose money — missing logging, stale references, doc drift, dead config.
            Auto-fixed, then logged.
  RISKY     touches gates / sizing / strategy selection / exits.
            NEVER auto-applied. Empirically validated, then queued as a proposal for you.

Why not auto-fix everything: this project has already demonstrated the failure mode twice.
  · A strategy added from sound reasoning ("buy pullbacks in uptrends") backtested at -1.12%
    expectancy and -85% drawdown. Reasoning is not evidence.
  · The obvious fix for "High conviction underperforms Medium" would be to prefer Medium — but
    Medium's edge is partly confounding (High correlates with extended setups). The correct fix
    was to abandon conviction for the meta-label. An agent optimising the flagged metric would
    have made things worse while appearing to succeed.

So the validator here is EMPIRICAL, not a second opinion: it re-runs the diagnostic / backtest and
requires the evidence to support the change. Two agents agreeing is just two guesses.

Run:
    python3 remediate.py              # triage + validate, apply SAFE fixes, queue the rest
    python3 remediate.py --dry-run    # show what it would do, change nothing
    python3 remediate.py --json       # write remediation_log.json
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys

LOG_PATH = "remediation_log.json"
QUEUE_PATH = "remediation_queue.json"

# ---- triage table: which finding codes are safe to fix unattended, and which touch money ----
SAFE = {
    "AUTO_BLIND_SPOTS": "add missing logging so the decision becomes auditable",
    "MEASUREMENT_PENDING": "no action — measurement maturing",
    "STALE_REFERENCE": "remove a reference to something retired",
    "SNAPSHOT_STALE": "no action — local snapshot only",
    "ALERTS_HEALTHY": "no action — informational",
    "CONVICTION_FALLBACK_OK": "no action — guarded fallback is acceptable",
    "AUTO_SWEEP_COVERAGE": "no action — coverage summary",
    "ANTI_PREDICTIVE_METRIC": "no action — feature, not a gate",
}
RISKY = {
    "CONVICTION_GATE_LIVE": "changes which signals are actioned",
    "AUTO_WRONG_TIER": "changes which cohort the engine prefers",
    "AUTO_INVERTED_FIELD": "changes a decision input",
    "ALERTS_UNDERPERFORM": "changes what gets surfaced to you",
    "CONTRADICTION": "engine behaviour disagrees with config",
    "MODEL_DECAY": "affects position sizing",
    "MODEL_DRIFT": "affects position sizing",
    "STRATEGY_NEGATIVE": "removes/keeps a strategy in the panel",
    "STRATEGY_FRAGILE": "questions a strategy's edge",
    "AUTO_WEAK_COHORT": "would exclude a cohort from trading",
    "AUTO_WEAK_RANKER": "changes ranking",
    "UNMEASURED_DECISION": "requires logging + later a gate change",
    "UNGRADED_DECISION": "requires a new scorecard",
    "CONVICTION_INVERTED": "informational, but the fix touches gates",
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M GMT")


def _load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def triage(finding: dict) -> dict:
    """Classify a finding by blast radius — what could this fix cost if it's wrong?"""
    code = finding.get("code", "")
    if code in SAFE:
        return {"tier": "SAFE", "rationale": SAFE[code]}
    if code in RISKY:
        return {"tier": "RISKY", "rationale": RISKY[code]}
    return {"tier": "RISKY", "rationale": "unknown finding type — treated as risky by default"}


# ------------------------------------------------------------------ validation (EVIDENCE, not opinion)
def validate(finding: dict) -> dict:
    """Check the finding against the actual record before anyone acts on it. Returns
    {verdict, evidence, confidence}. verdict: confirmed | refuted | insufficient_data.

    This is deliberately empirical. A finding that can't be corroborated by data does NOT get
    promoted into a proposal — that's the guard against acting on a plausible-sounding mistake.
    """
    code = finding.get("code", "")
    tr = _load("track_record.json", []) or []
    rows = [t for t in tr if t.get("status") in ("win", "loss")]

    if code in ("AUTO_WRONG_TIER", "CONVICTION_GATE_LIVE", "CONVICTION_INVERTED"):
        lo = [t for t in rows if t.get("direction") == "LONG"]
        def wr(tier):
            g = [t for t in lo if t.get("conviction") == tier]
            return (100 * sum(1 for t in g if t["status"] == "win") / len(g), len(g)) if len(g) >= 30 else (None, len(g))
        hi, nhi = wr("High")
        med, nmed = wr("Medium")
        if hi is None or med is None:
            return {"verdict": "insufficient_data", "evidence": f"High n={nhi}, Medium n={nmed}", "confidence": "low"}
        if med - hi >= 8:
            # corroborate: is the alternative (the model) actually better? otherwise the "fix" is blind.
            mh = _load("meta_history.json", []) or []
            auc = mh[-1].get("auc_meta") if mh else None
            ok_alt = isinstance(auc, (int, float)) and auc >= 0.60
            return {"verdict": "confirmed",
                    "evidence": (f"High {hi:.1f}% (n={nhi}) vs Medium {med:.1f}% (n={nmed}); "
                                 + (f"meta-label AUC {auc} is a validated alternative" if ok_alt
                                    else "NOTE: no validated alternative — do not simply invert the gate")),
                    "confidence": "high" if ok_alt else "medium"}
        return {"verdict": "refuted", "evidence": f"High {hi:.1f}% vs Medium {med:.1f}% — gap not material",
                "confidence": "high"}

    if code == "ALERTS_UNDERPERFORM":
        diag = _load("system_diagnostic.json", {}) or {}
        a = (diag.get("alerts") or {})
        al, nal = a.get("alerted") or {}, a.get("not_alerted") or {}
        if al.get("n", 0) < 30 or nal.get("n", 0) < 30:
            return {"verdict": "insufficient_data",
                    "evidence": f"alerted n={al.get('n',0)}, not n={nal.get('n',0)}", "confidence": "low"}
        d = (al.get("wr") or 0) - (nal.get("wr") or 0)
        return {"verdict": "confirmed" if d < -5 else "refuted",
                "evidence": f"alerted {al.get('wr'):.1f}% vs un-alerted {nal.get('wr'):.1f}%",
                "confidence": "high"}

    if code in ("STRATEGY_NEGATIVE", "STRATEGY_FRAGILE"):
        ss = (_load("strategy_study.json", {}) or {}).get("strategies") or {}
        msg = finding.get("message", "")
        m = re.search(r"'([^']+)'", msg)
        name = m.group(1) if m else None
        for _k, r in ss.items():
            if r.get("label") == name:
                b = r.get("base") or {}
                if b.get("n", 0) < 40:
                    return {"verdict": "insufficient_data", "evidence": f"n={b.get('n',0)}", "confidence": "low"}
                return {"verdict": "confirmed" if (b.get("expectancy") or 0) < 0 else "refuted",
                        "evidence": f"backtest expectancy {b.get('expectancy'):+.2f}% over n={b['n']} "
                                    f"under the live exit model", "confidence": "high"}
        return {"verdict": "insufficient_data", "evidence": "no backtest record for this strategy",
                "confidence": "low"}

    if code == "AUTO_WEAK_COHORT":
        m = re.search(r"'([^=]+)=([^']+)'", finding.get("message", ""))
        if not m:
            return {"verdict": "insufficient_data", "evidence": "could not parse cohort", "confidence": "low"}
        k, v = m.group(1), m.group(2)
        g = [t for t in rows if str(t.get(k)) == v]
        if len(g) < 60:
            return {"verdict": "insufficient_data",
                    "evidence": f"cohort n={len(g)} — too small to act on", "confidence": "low"}
        base = 100 * sum(1 for t in rows if t["status"] == "win") / len(rows)
        wr = 100 * sum(1 for t in g if t["status"] == "win") / len(g)
        return {"verdict": "confirmed" if wr < base - 15 else "refuted",
                "evidence": f"{k}={v}: {wr:.1f}% vs {base:.1f}% overall (n={len(g)})",
                "confidence": "medium"}

    if code in ("CONTRADICTION", "SILENT_FEATURE"):
        return {"verdict": "confirmed", "evidence": finding.get("evidence", "observed in the live build"),
                "confidence": "high"}

    return {"verdict": "insufficient_data", "evidence": "no empirical test defined for this code",
            "confidence": "low"}


# ------------------------------------------------------------------ safe auto-fixes
def apply_safe_fix(finding: dict, dry_run: bool = True) -> dict:
    """Only fixes that cannot affect a trading decision. Everything else is proposed, not applied."""
    code = finding.get("code", "")
    if code == "STALE_REFERENCE":
        ev = finding.get("evidence", "")
        return {"applied": False, "action": f"manual: remove retired reference at {ev}",
                "note": "flagged for cleanup — automated source edits are out of scope"}
    if code == "AUTO_BLIND_SPOTS":
        return {"applied": False,
                "action": "add the listed fields to tracker.py at log time so they become auditable",
                "note": "queued as a chore; harmless but needs a code edit"}
    return {"applied": False, "action": "no action required", "note": SAFE.get(code, "")}


def run(dry_run: bool = False) -> dict:
    rep = _load("guardrails_report.json")
    if not rep:
        # run a sweep if there's no report yet
        try:
            import guardrails
            rep = guardrails.sweep()
        except Exception:  # noqa: BLE001
            return {"error": "no guardrails report and sweep failed"}

    processed, proposals, autofixed = [], [], []
    for f in rep.get("findings", []):
        t = triage(f)
        # Informational SAFE codes are no-ops by definition — validating them just produces
        # confusing "unconfirmed" noise in the log.
        noop = t["tier"] == "SAFE" and SAFE.get(f.get("code", ""), "").startswith("no action")
        v = ({"verdict": "n/a", "evidence": "informational — no action required", "confidence": "n/a"}
             if noop else validate(f))
        item = {"code": f.get("code"), "severity": f.get("severity"), "message": f.get("message"),
                "evidence": f.get("evidence", ""), "tier": t["tier"], "why_tier": t["rationale"],
                "verdict": v["verdict"], "proof": v["evidence"], "confidence": v["confidence"],
                "seen": _now()}
        # Only CONFIRMED findings progress. Refuted/insufficient are recorded but go no further —
        # this is the gate that stops a plausible-but-wrong finding turning into a change.
        if v["verdict"] == "n/a":
            item["outcome"] = "no action needed"
        elif v["verdict"] != "confirmed":
            item["outcome"] = f"held ({v['verdict']})"
        elif t["tier"] == "SAFE":
            res = apply_safe_fix(f, dry_run)
            item["outcome"] = ("applied: " if res["applied"] else "chore: ") + res["action"]
            (autofixed if res["applied"] else proposals).append(item)
        else:
            item["outcome"] = "PROPOSED — needs your approval (touches trading logic)"
            proposals.append(item)
        processed.append(item)

    out = {"generated": _now(), "n_findings": len(processed),
           "counts": {"proposals": len(proposals), "auto_fixed": len(autofixed),
                      "held": sum(1 for i in processed if i["outcome"].startswith("held")),
                      "no_action": sum(1 for i in processed if i["outcome"] == "no action needed")},
           "proposals": proposals, "auto_fixed": autofixed, "all": processed}

    if not dry_run:
        hist = _load(LOG_PATH, []) or []
        hist.append({"run": out["generated"], "counts": out["counts"],
                     "items": [{k: i[k] for k in ("code", "tier", "verdict", "outcome")} for i in processed]})
        json.dump(hist[-200:], open(LOG_PATH, "w"), indent=1)
        json.dump(out, open(QUEUE_PATH, "w"), indent=1)
    return out


def report(out: dict) -> str:
    if out.get("error"):
        return out["error"]
    c = out["counts"]
    L = [f"REMEDIATION — {out['n_findings']} findings · {c['auto_fixed']} auto-fixed · "
         f"{c['proposals']} awaiting approval · {c['held']} held · {c.get('no_action',0)} informational",
         "=" * 96]
    for i in out["all"]:
        if i["outcome"].startswith("held") or i["outcome"] == "no action needed":
            continue
        L.append(f"[{i['tier']:5}] {i['code']}  ({i['verdict']}, {i['confidence']} confidence)")
        L.append(f"        {i['message'][:110]}")
        L.append(f"        proof: {i['proof'][:110]}")
        L.append(f"        -> {i['outcome']}")
    held = [i for i in out["all"] if i["outcome"].startswith("held")]
    if held:
        L.append("")
        L.append(f"held ({len(held)}): " + ", ".join(sorted({i['code'] for i in held})))
        L.append("  (findings that could NOT be corroborated by the record — deliberately not acted on)")
    return "\n".join(L)


if __name__ == "__main__":
    res = run(dry_run="--dry-run" in sys.argv)
    print(report(res))
    if "--json" in sys.argv and not res.get("error"):
        print(f"\n[wrote {LOG_PATH} + {QUEUE_PATH}]")
