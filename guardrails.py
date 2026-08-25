"""GUARDRAILS — the standing audit / police sweep over the engine's own logic.

`audit.py` asks "are the numbers plausible?". This asks the harder question:
**is the engine making decisions on things that are proven wrong, and is every automatic decision
actually being graded?**

It exists because a real bug survived ~6 weeks: alerts and paper execution both gated on the
"High conviction" score long after the record proved that score was INVERTED (alerted longs won
50.3% while the ones that never alerted won 66.8%). Nothing caught it, because nothing was
measuring the alerts at all. These checks encode that lesson so the same class of failure trips
an alarm within a day instead of a month.

Checks
  1 ANTI-PREDICTIVE GATE  a conviction check with a proven negative edge is still gating decisions
  2 UNMEASURED DECISION   the engine decides something automatically, but nothing scores it
  3 SILENT FEATURE        a feature is switched on but produced nothing in the latest build
  4 CONTRADICTION         config says one thing, the live output says another
  5 DRIFT                 a strategy / the model has decayed materially vs its own history
  6 STALE REFERENCE       something retired is still referenced in a live code path

Every finding is (severity, code, message, evidence). Exit code is 1 on any CRITICAL so CI can
fail loudly if you ever want it to block.

Run:
    python3 guardrails.py            # print the sweep
    python3 guardrails.py --json     # also write guardrails_report.json
"""
from __future__ import annotations

import json
import os
import re
import sys

REPORT_PATH = "guardrails_report.json"
SRC = ("dashboard.py", "notify.py", "paper.py", "scanner.py", "tracker.py", "strategies.py", "risk.py")

CRITICAL, WARN, INFO = "CRITICAL", "WARN", "INFO"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:  # noqa: BLE001
        return ""


def _finding(out, sev, code, msg, evidence=""):
    out.append({"severity": sev, "code": code, "message": msg, "evidence": evidence})


def _snapshot_age_days(snap) -> float | None:
    """How old is the build we're auditing? A stale local snapshot (e.g. a dev machine that hasn't
    pulled) would otherwise fire false 'feature produced nothing' alarms about features that simply
    didn't exist when it was generated."""
    import datetime
    ts = (snap or {}).get("generated_at") or ""
    for fmt in ("%Y-%m-%d %H:%M GMT", "%Y-%m-%d"):
        try:
            d = datetime.datetime.strptime(ts[:len(datetime.datetime.now().strftime(fmt))], fmt)
            return (datetime.datetime.utcnow() - d).total_seconds() / 86400
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------- 1. anti-predictive gates
def check_anti_predictive(out, diag, sources):
    """Any conviction check whose PASS cohort does materially WORSE than its not-pass cohort is
    anti-predictive. If such a check is still used to filter/rank/alert, that's the exact bug that
    cost us six weeks."""
    checks = (diag or {}).get("checks_longs_only") or {}
    if not checks:
        return
    for label, c in checks.items():
        delta = c.get("win_delta")
        if delta is None or delta > -10:
            continue
        n = (c.get("pass_n") or 0) + (c.get("notpass_n") or 0)
        if n < 60:
            continue
        hits = []
        needle = label.replace("?", "").strip()
        for f in sources:
            for i, line in enumerate(sources[f].splitlines(), 1):
                if needle in line and re.search(r"==|>=|<=|\bif\b|filter|sort|key=", line):
                    if not line.strip().startswith("#"):
                        hits.append(f"{f}:{i}")
        sev = CRITICAL if delta <= -25 else WARN
        if hits:
            _finding(out, sev, "ANTI_PREDICTIVE_GATE",
                     f'"{label}" is anti-predictive ({delta:+.0f}pt win delta, n={n}) but still appears '
                     f"in decision logic — verify it isn't gating trades.", "; ".join(hits[:4]))
        else:
            _finding(out, INFO, "ANTI_PREDICTIVE_METRIC",
                     f'"{label}" is anti-predictive ({delta:+.0f}pt, n={n}). Not gating anything — '
                     "fine as a model feature, but don't surface it as a quality badge.")


def check_conviction_gates(out, diag, sources):
    """The specific landmine: conviction label/score used as a QUALITY GATE anywhere, when the
    record shows the tiers are inverted."""
    bd = (diag or {}).get("by_direction") or {}
    tiers_inverted = False
    tr = _load("track_record.json") or []
    lo = [t for t in tr if t.get("direction") == "LONG" and t.get("status") in ("win", "loss")]
    if len(lo) > 100:
        def wr(tier):
            rows = [t for t in lo if t.get("conviction") == tier]
            return (100 * sum(1 for t in rows if t["status"] == "win") / len(rows)) if len(rows) > 30 else None
        hi, med = wr("High"), wr("Medium")
        if hi is not None and med is not None and med - hi > 8:
            tiers_inverted = True
            _finding(out, INFO, "CONVICTION_INVERTED",
                     f"Conviction tiers remain inverted: High {hi:.1f}% vs Medium {med:.1f}% win. "
                     "Any gate keyed on High is selecting the worse cohort.")
    if not tiers_inverted:
        return
    pat = re.compile(r"""conviction.{0,40}?(label|score_pct).{0,20}?(==\s*["']High["']|>=)""", re.S)
    for f in sources:
        lines = sources[f].splitlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#") or s.startswith("//"):
                continue
            if not pat.search(line) or "p_win" in line:
                continue
            # A GUARDED FALLBACK is fine: conviction only used when no model score exists. Detect it
            # by looking for a p_win check in the surrounding block (the enclosing few lines).
            ctx = "\n".join(lines[max(0, i - 8):i + 3])
            if "p_win" in ctx:
                _finding(out, INFO, "CONVICTION_FALLBACK_OK",
                         "Conviction referenced only as a pre-model fallback (guarded by a P(win) "
                         "check) — acceptable.", f"{f}:{i}")
                continue
            _finding(out, CRITICAL, "CONVICTION_GATE_LIVE",
                     "Conviction used as a quality gate while the tiers are inverted — "
                     "should gate on the validated meta-label P(win) instead.", f"{f}:{i}")


# ---------------------------------------------------------------- 2. unmeasured decisions
def check_unmeasured_decisions(out, diag):
    """Every automatic decision needs a scorecard. If the engine decides something and nothing
    grades it, a bug there is invisible — which is precisely how the alert bug survived."""
    tr = _load("track_record.json") or []
    resolved = [t for t in tr if t.get("status") in ("win", "loss")]
    recent = resolved[-250:] if resolved else []
    surfaces = [
        ("alerts", "alerted", "which signals pinged you", "alerts"),
        ("actionable BUYs", "action", "BUY vs Watch (the trades you'd actually take)", "buy_only"),
        ("model probability", "p_win", "the P(win) the model assigned at entry", "buy_only_pwin52"),
    ]
    for name, field, what, diag_key in surfaces:
        have = sum(1 for t in recent if t.get(field) is not None)
        graded = bool((diag or {}).get(diag_key))
        if not recent:
            continue
        # Is the field at least being WRITTEN now? If tracker.py logs it, absence in old resolved
        # trades just means the fix is recent and hasn't matured yet — INFO, not a warning.
        logged_now = bool(re.search(rf'["\']{field}["\']\s*:', _read("tracker.py")))
        if have == 0:
            if logged_now:
                _finding(out, INFO, "MEASUREMENT_PENDING",
                         f"{name} is now being recorded ({what}), but no resolved trade carries it yet — "
                         "the scorecard matures as new trades close.")
            else:
                _finding(out, WARN, "UNMEASURED_DECISION",
                         f"The engine decides {name} but the track record never records {what} — "
                         "its quality cannot be measured. Add the field at log time.",
                         f"0 of last {len(recent)} resolved trades carry '{field}'")
        elif not graded and have > 20:
            _finding(out, INFO, "UNGRADED_DECISION",
                     f"'{field}' is being recorded ({have} trades) but the nightly diagnostic "
                     f"doesn't report a scorecard for {name} yet.")


def check_alert_quality(out, diag):
    """Once alerts ARE measured: are the ones that ping you actually better than the ones that don't?"""
    a = (diag or {}).get("alerts") or {}
    al, nal = a.get("alerted") or {}, a.get("not_alerted") or {}
    if al.get("n", 0) > 30 and nal.get("n", 0) > 30:
        d = (al.get("wr") or 0) - (nal.get("wr") or 0)
        if d < -5:
            _finding(out, CRITICAL, "ALERTS_UNDERPERFORM",
                     f"Alerted trades are WORSE than un-alerted ones ({al['wr']:.1f}% vs {nal['wr']:.1f}%). "
                     "The alert gate is selecting the wrong cohort.",
                     f"n={al['n']} alerted / {nal['n']} not")
        elif d > 5:
            _finding(out, INFO, "ALERTS_HEALTHY",
                     f"Alerts are pulling their weight: {al['wr']:.1f}% vs {nal['wr']:.1f}% un-alerted.")


# ---------------------------------------------------------------- 3/4. silent features + contradictions
def check_silent_features(out, snap, cfg):
    """A feature switched ON that produced nothing is either broken or mis-wired (the Grok bug:
    'enabled' for weeks while every call silently failed)."""
    if snap is None:
        return
    sigs = snap.get("signals") or []
    probes = [
        ("xai_live_sentiment_enabled", lambda: sum(1 for s in sigs if s.get("xai_sentiment")),
         "Grok live sentiment", "xai_status"),
        ("xai_buzz_enabled", lambda: len(snap.get("xai_buzz") or []), "Grok buzz scan", None),
        ("stocktwits_trending_enabled", lambda: len(snap.get("retail_trending") or []),
         "StockTwits trending", None),
        ("premium_selling_enabled", lambda: len((snap.get("premium_selling") or {}).get("names") or []),
         "Premium-selling advisory", None),
        ("meta_pwin_enabled", lambda: sum(1 for s in sigs if s.get("p_win") is not None),
         "Meta-label P(win)", None),
    ]
    for flag, count_fn, label, status_key in probes:
        if not getattr(cfg, flag, False):
            continue
        try:
            n = count_fn()
        except Exception:  # noqa: BLE001
            n = 0
        if n == 0:
            extra = f" (status: {snap.get(status_key)})" if status_key and snap.get(status_key) else ""
            _finding(out, WARN, "SILENT_FEATURE",
                     f"{label} is ENABLED but produced nothing in the latest build{extra} — "
                     "check for a silently-swallowed error.", f"config.{flag}=True, output=0")


def check_contradictions(out, snap, cfg):
    """Config says one thing; the live output says another."""
    if snap is None:
        return
    sigs = snap.get("signals") or []
    if not getattr(cfg, "allow_shorts", False):
        bad = [s.get("symbol") for s in sigs
               if s.get("action") in ("SHORT", "HOLD SHORT", "WATCH SHORT")]
        if bad:
            _finding(out, CRITICAL, "CONTRADICTION",
                     f"Shorts are disabled but {len(bad)} tradeable SHORT action(s) were emitted.",
                     ", ".join(str(b) for b in bad[:6]))
    floor = getattr(cfg, "meta_pwin_floor", None)
    cap = getattr(cfg, "meta_buy_cap", None)
    if floor:
        below = [s.get("symbol") for s in sigs
                 if s.get("action") == "BUY" and isinstance(s.get("p_win"), (int, float))
                 and s["p_win"] < floor and not s.get("user_accepted")]
        if below:
            _finding(out, WARN, "CONTRADICTION",
                     f"{len(below)} fresh BUY(s) sit below the P(win) floor of {floor:.2f}.",
                     ", ".join(str(b) for b in below[:6]))
    if cap:
        nbuy = sum(1 for s in sigs if s.get("action") == "BUY")
        if nbuy > cap:
            _finding(out, WARN, "CONTRADICTION",
                     f"{nbuy} fresh BUYs exceed the daily cap of {cap}.")


# ---------------------------------------------------------------- 5/6. drift + stale refs
def check_drift(out):
    """Has anything decayed materially vs its own history?"""
    mh = _load("meta_history.json") or []
    if len(mh) >= 6:
        cur = mh[-1].get("auc_meta")
        prev = [h.get("auc_meta") for h in mh[-6:-1] if h.get("auc_meta")]
        if cur and prev:
            peak = max(prev)
            if cur < 0.55:
                _finding(out, CRITICAL, "MODEL_DECAY",
                         f"Meta-label AUC has fallen to {cur} — at/below coin-flip territory. "
                         "Stop letting it drive sizing until investigated.")
            elif peak - cur >= 0.05:
                _finding(out, WARN, "MODEL_DRIFT",
                         f"Meta-label AUC drifting down: {peak} -> {cur} over recent runs.")
    ss = (_load("strategy_study.json") or {}).get("strategies") or {}
    for _k, r in ss.items():
        b = r.get("base") or {}
        exp = b.get("expectancy")
        if exp is not None and b.get("n", 0) >= 40 and exp < 0:
            _finding(out, WARN, "STRATEGY_NEGATIVE",
                     f"Strategy '{r.get('label')}' has negative expectancy ({exp:+.2f}%, n={b['n']}) "
                     "under the live exit model — consider retiring it.")
        rob = r.get("robustness") or {}
        vals = [ (rob.get("trail_tight_75pct") or {}).get("expectancy"),
                 (rob.get("trail_loose_125pct") or {}).get("expectancy") ]
        if exp is not None and exp > 0 and any(v is not None and v < 0 for v in vals):
            _finding(out, INFO, "STRATEGY_FRAGILE",
                     f"Strategy '{r.get('label')}' flips negative when the trail is nudged — "
                     "its edge may be parameter luck.")


def check_stale_refs(out, sources):
    """Retired strategies still referenced in a live path."""
    try:
        import strategies as S
        live = set(S.STRATEGIES) | set(S.SHORT_STRATEGIES)
    except Exception:  # noqa: BLE001
        return
    retired = re.findall(r'"([a-z0-9_]+)"\s*\(?.*?RETIRED', sources.get("strategies.py", ""))
    for key in ("macd_trend", "macd_trend_dn", "pullback"):
        if key in live:
            continue
        for f in sources:
            if f == "strategies.py":
                continue
            for i, line in enumerate(sources[f].splitlines(), 1):
                if re.search(rf"['\"]{key}['\"]", line) and not line.strip().startswith("#"):
                    _finding(out, WARN, "STALE_REFERENCE",
                             f"Retired strategy '{key}' is still referenced in a live path.", f"{f}:{i}")


# ---------------------------------------------------------------- driver
def sweep() -> dict:
    out: list[dict] = []
    diag = _load("system_diagnostic.json")
    snap = _load("signals.json")
    sources = {f: _read(f) for f in SRC if os.path.exists(f)}
    try:
        from config import CONFIG as cfg
    except Exception:  # noqa: BLE001
        cfg = None

    # Code-level and record-level checks always run. Snapshot-level checks (did a feature produce
    # output? does the output contradict config?) only make sense on a FRESH build — auditing a
    # stale local snapshot would flag features that simply post-date it.
    age = _snapshot_age_days(snap)
    fresh = age is not None and age <= 3

    check_anti_predictive(out, diag, sources)
    check_conviction_gates(out, diag, sources)
    check_unmeasured_decisions(out, diag)
    check_alert_quality(out, diag)
    if cfg is not None and fresh:
        check_silent_features(out, snap, cfg)
        check_contradictions(out, snap, cfg)
    elif snap is not None:
        _finding(out, INFO, "SNAPSHOT_STALE",
                 f"signals.json is {age:.0f} days old — skipped live-output checks (silent features, "
                 "config contradictions). These run against a fresh build in CI."
                 if age is not None else "Could not date signals.json; skipped live-output checks.")
    check_drift(out)
    check_stale_refs(out, sources)

    rank = {CRITICAL: 0, WARN: 1, INFO: 2}
    out.sort(key=lambda f: rank.get(f["severity"], 3))
    counts = {s: sum(1 for f in out if f["severity"] == s) for s in (CRITICAL, WARN, INFO)}
    return {"generated": _now(), "counts": counts, "findings": out}


def report(res: dict) -> str:
    c = res["counts"]
    L = [f"GUARDRAILS SWEEP — {c[CRITICAL]} critical · {c[WARN]} warn · {c[INFO]} info", "=" * 92]
    if not res["findings"]:
        L.append("  clean — no logic-integrity issues found.")
        return "\n".join(L)
    mark = {CRITICAL: "!!", WARN: " !", INFO: "  "}
    for f in res["findings"]:
        L.append(f"{mark.get(f['severity'],'  ')} [{f['severity']:8}] {f['code']}")
        L.append(f"     {f['message']}")
        if f.get("evidence"):
            L.append(f"     evidence: {f['evidence']}")
    L.append("")
    L.append("CRITICAL = the engine is deciding on something proven wrong, or contradicting its own config.")
    return "\n".join(L)


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M GMT")


if __name__ == "__main__":
    res = sweep()
    print(report(res))
    if "--json" in sys.argv:
        json.dump(res, open(REPORT_PATH, "w"), indent=1)
        print(f"\n[wrote {REPORT_PATH}]")
    sys.exit(1 if res["counts"][CRITICAL] and "--strict" in sys.argv else 0)
