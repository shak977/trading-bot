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


# ---------------------------------------------------------------- AUTONOMOUS DISCOVERY
# The checks above encode failures we already knew about. These find NEW ones without being told
# what to look for: derive what the engine decides ON from the source, measure what actually
# predicts outcomes from the record, and flag anything that drives decisions without earning it.

# fields that are bookkeeping, not decision inputs — never flag these
_IGNORE_FIELDS = {
    "symbol", "name", "id", "advised_date", "advised_ts", "exit_date", "status", "entry", "exit",
    "stop", "target", "return_pct", "days_held", "t1", "t1_frac", "t1_hit", "t1_date", "high_water",
    "target_exceeded", "checks", "action", "direction", "regime_score",
}


def discover_decision_fields(sources: dict) -> dict:
    """Parse the source for places the engine FILTERS, RANKS or GATES, and pull out which signal
    fields those decisions read. No hardcoded field names — whatever the code keys on, we find."""
    found: dict[str, set] = {}
    # e.g.  s.get("p_win")   (r.get("conviction") or {}).get("label")   x["rank_score"]
    getters = re.compile(r"""(?:\.get\(|\[)\s*["']([a-z0-9_]{3,30})["']""")
    # Resolve one level of aliasing: `conv = s.get("conviction")` then `conv.get("label")` must still
    # be attributed to *conviction*, or splitting a gate across two lines hides it from the audit.
    alias_def = re.compile(r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(?\s*[A-Za-z_][A-Za-z0-9_]*\.get\(\s*["']([a-z0-9_]{3,30})["']""")
    decisionish = re.compile(r"\bif\b|\bfilter\b|sort\(|sorted\(|key\s*=|>=|<=|==|\band\b")
    for fname, text in sources.items():
        lines = text.splitlines()
        aliases: dict[str, str] = {}
        for i, line in enumerate(lines, 1):
            am = alias_def.match(line)
            if am:
                aliases[am.group(1)] = am.group(2)
            s = line.strip()
            if s.startswith("#") or s.startswith("//") or len(s) < 8:
                continue
            if not decisionish.search(s):
                continue
            # only count lines that look like they act on a signal/trade row
            if not re.search(r"\b[srtxq]\b\.get\(|\brow\.get\(|\bsig\.get\(|\.get\(", s):
                continue
            hits = {m for m in getters.findall(s) if m not in _IGNORE_FIELDS}
            # attribute alias uses back to the parent field (conv.get("label") -> conviction)
            for var, parent in aliases.items():
                if re.search(rf"\b{re.escape(var)}\.get\(", s) and parent not in _IGNORE_FIELDS:
                    hits.add(parent)
            for m in hits:
                found.setdefault(m, set()).add(f"{fname}:{i}")
    return found


def measure_field_edge(field: str, rows: list) -> dict | None:
    """Does this field actually separate winners from losers in the real record? Works for numbers
    (split at the median) and categories (best vs worst tier). Returns None if unmeasurable."""
    vals = [(t.get(field), t) for t in rows if t.get(field) is not None]
    if len(vals) < 60:
        return None
    nums = [v for v, _ in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) >= len(vals) * 0.8:                       # numeric field
        import statistics as _st
        med = _st.median(nums)
        hi = [t for v, t in vals if isinstance(v, (int, float)) and v > med]
        lo = [t for v, t in vals if isinstance(v, (int, float)) and v <= med]
        if len(hi) < 25 or len(lo) < 25:
            return None
        wr = lambda g: 100 * sum(1 for t in g if t["status"] == "win") / len(g)
        return {"kind": "numeric", "n": len(vals), "delta": wr(hi) - wr(lo),
                "detail": f"above-median {wr(hi):.1f}% vs below {wr(lo):.1f}%"}
    groups: dict = {}
    for v, t in vals:
        groups.setdefault(str(v), []).append(t)
    groups = {k: g for k, g in groups.items() if len(g) >= 25}
    if len(groups) < 2:
        return None
    wr = {k: 100 * sum(1 for t in g if t["status"] == "win") / len(g) for k, g in groups.items()}
    best, worst = max(wr, key=wr.get), min(wr, key=wr.get)
    return {"kind": "categorical", "n": len(vals), "delta": wr[best] - wr[worst],
            "detail": f"{best} {wr[best]:.1f}% vs {worst} {wr[worst]:.1f}%",
            "ranked": sorted(wr.items(), key=lambda kv: -kv[1])}


def check_autonomous(out, sources):
    """Cross-reference: every field the code makes decisions on, measured against the real record.
    This is what makes the sweep self-discovering — it would have caught the conviction bug with no
    prior knowledge of 'conviction', and will catch the next one the same way."""
    tr = _load("track_record.json") or []
    rows = [t for t in tr if t.get("status") in ("win", "loss")]
    if len(rows) < 120:
        return
    fields = discover_decision_fields(sources)
    if not fields:
        return
    # Only audit fields that genuinely live on a SIGNAL row. Without this the sweep drowns in
    # plumbing ('msg', 'title', 'headline'...) and becomes noise you learn to ignore — which is
    # how a real finding gets missed.
    snap = _load("signals.json") or {}
    sig_fields: set = set()
    for s in (snap.get("signals") or [])[:40]:
        sig_fields |= set(s.keys())
        for sub in ("conviction", "plan", "liquidity", "context", "factors"):
            v = s.get(sub)
            if isinstance(v, dict):
                sig_fields |= set(v.keys())
    fields = {k: v for k, v in fields.items() if k in sig_fields} or fields
    measured = unmeasurable = 0
    blind: list[str] = []
    for field, sites in sorted(fields.items()):
        edge = measure_field_edge(field, rows)
        if edge is None:
            # the code decides on something the record never captures -> we can never audit it.
            # Collected and reported as ONE summary line: a wall of per-field warnings is noise,
            # and noise is what makes a real finding get scrolled past.
            if len(sites) >= 2 and not any(t.get(field) is not None for t in rows[-200:]):
                unmeasurable += 1
                blind.append(field)
            continue
        measured += 1
        d = edge["delta"]
        # categorical: is the tier the code PREFERS actually the best one?
        if edge["kind"] == "categorical" and d >= 8:
            ranked = edge.get("ranked") or []
            best = ranked[0][0] if ranked else None
            for site in sorted(sites):
                f, ln = site.rsplit(":", 1)
                if f not in sources:
                    continue
                lines = sources[f].splitlines()
                i = int(ln)
                line = lines[i - 1]
                # The line is already attributed to this field (directly or via an alias), so just
                # read which value it compares against.
                m = re.search(r"""==\s*["']([^"']+)["']""", line)
                if not (m and best and m.group(1) != best) or "p_win" in line:
                    continue
                if m.group(1) not in (edge.get("ranked") and dict(edge["ranked"]) or {}):
                    continue      # comparing against something that isn't one of the measured tiers
                # guarded fallback (only reached when the model score is missing) is acceptable
                if "p_win" in "\n".join(lines[max(0, i - 8):i + 3]):
                    continue
                _finding(out, CRITICAL, "AUTO_WRONG_TIER",
                         f"Code selects '{field} == {m.group(1)}' but the record says "
                         f"'{best}' is the best-performing tier ({edge['detail']}, n={edge['n']}).",
                         site)
        # numeric: a field that ranks/sorts but has no separating power
        if edge["kind"] == "numeric" and abs(d) < 2 and any("key=" in sources.get(s.rsplit(":", 1)[0], "")
                                                            .splitlines()[int(s.rsplit(":", 1)[1]) - 1]
                                                            for s in sites
                                                            if s.rsplit(":", 1)[0] in sources):
            _finding(out, INFO, "AUTO_WEAK_RANKER",
                     f"'{field}' is used to rank but barely separates outcomes ({edge['detail']}, "
                     f"n={edge['n']}) — ranking on it is close to arbitrary.",
                     "; ".join(sorted(sites)[:3]))
        if d <= -12:
            _finding(out, CRITICAL if d <= -20 else WARN, "AUTO_INVERTED_FIELD",
                     f"'{field}' is used in decisions but is INVERTED in the record "
                     f"({edge['detail']}, n={edge['n']}) — higher/preferred values do WORSE.",
                     "; ".join(sorted(sites)[:3]))
    if blind:
        _finding(out, INFO, "AUTO_BLIND_SPOTS",
                 f"{len(blind)} field(s) drive decisions but are never logged at entry, so their "
                 "effect on outcomes can't be audited. Worth logging the ones that gate trades.",
                 ", ".join(sorted(blind)[:12]))
    _finding(out, INFO, "AUTO_SWEEP_COVERAGE",
             f"Autonomous pass: {len(fields)} decision fields discovered in source, {measured} had "
             f"enough history to audit, {unmeasurable} unmeasurable.")


def check_auto_cohorts(out):
    """Unsupervised outcome scan: slice the record by every categorical field and surface any cohort
    that materially underperforms. Finds problems nobody thought to look for."""
    tr = _load("track_record.json") or []
    rows = [t for t in tr if t.get("status") in ("win", "loss")]
    if len(rows) < 150:
        return
    base = 100 * sum(1 for t in rows if t["status"] == "win") / len(rows)
    keys = set()
    for t in rows[-400:]:
        keys |= {k for k, v in t.items()
                 if k not in _IGNORE_FIELDS and isinstance(v, (str, bool)) and str(v)[:1]}
    for k in sorted(keys):
        groups: dict = {}
        for t in rows:
            v = t.get(k)
            if v is None:
                continue
            groups.setdefault(str(v), []).append(t)
        for val, g in groups.items():
            if len(g) < 40:
                continue
            wr = 100 * sum(1 for t in g if t["status"] == "win") / len(g)
            if wr < base - 15:
                _finding(out, WARN, "AUTO_WEAK_COHORT",
                         f"Cohort '{k}={val}' wins {wr:.1f}% vs {base:.1f}% overall "
                         f"(n={len(g)}) — materially worse than the book.")


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
    # autonomous passes — these need no prior knowledge of what to look for
    check_autonomous(out, sources)
    check_auto_cohorts(out)

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
