"""Autonomous nightly analyst — the self-improving loop.

Runs after the close (its own GitHub Actions cron). Reviews the bot's OWN resolved trades across
every strategy bucket (daily / intraday / ORB), measures what's working and what isn't, and writes a
dated report with CONCRETE, prioritised proposed changes.

Advisory by design: it PROPOSES, you APPROVE. It never silently rewrites risk limits or config — the
bounded learned-weights loop already auto-adapts within ±0.5; this is the bigger-picture critic that
suggests the structural changes (thresholds, windows, which checks to retire) for a human to sign off.

Outputs analyst_report.json (latest) and appends a slim entry to analyst_history.json. The LLM
narrative is optional (ANTHROPIC_API_KEY) and fail-silent — the quantitative findings stand alone.
Never raises out of run().
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import attribution

_BUCKETS = (("daily", "Daily / swing", attribution.PATH),
            ("intraday", "Intraday", attribution.INTRADAY_PATH),
            ("orb", "ORB day-trade", attribution.ORB_PATH))

_RESOLVED = ("win", "loss", "expired", "eod")
_MIN_N = 12          # minimum resolved trades before a finding is trusted
_EDGE_HI, _EDGE_LO = 12.0, -12.0   # per-check edge thresholds for a proposal


def _load(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return json.load(f) or []
    except Exception:  # noqa: BLE001
        return []


def _bucket_stats(rows: list[dict]) -> dict:
    """Headline performance for one strategy bucket from its resolved trades."""
    resolved = [t for t in rows if t.get("status") in _RESOLVED]
    decided = [t for t in resolved if t.get("status") in ("win", "loss")]
    wins = [t for t in decided if t["status"] == "win"]
    out = {"logged": len(rows), "resolved": len(resolved), "decided": len(decided),
           "open": sum(1 for t in rows if t.get("status") == "open"),
           "win_rate": round(len(wins) / len(decided) * 100, 1) if decided else None}
    # by macro regime at entry (if tagged)
    reg = {}
    for t in decided:
        lab = t.get("regime") or "—"
        reg.setdefault(lab, [0, 0])
        reg[lab][1] += 1
        if t["status"] == "win":
            reg[lab][0] += 1
    out["by_regime"] = {k: {"n": v[1], "win_rate": round(v[0] / v[1] * 100, 1)}
                        for k, v in reg.items() if v[1] >= 5}
    # most recent resolved streak (by exit time when present)
    consec, kind = 0, None
    for t in sorted([t for t in resolved if t.get("exit_time") or t.get("exit_date")],
                    key=lambda x: x.get("exit_time") or x.get("exit_date") or "", reverse=True):
        st = t["status"]
        if st not in ("win", "loss"):
            continue
        if kind is None:
            kind = st
        if st == kind:
            consec += 1
        else:
            break
    out["streak"] = {"kind": kind, "n": consec} if kind else None
    return out


def _findings(scope: str, label: str, stats: dict, rep: list[dict], learned: dict) -> list[dict]:
    """Deterministic, explainable proposals for one bucket. Each: {severity, area, observation,
    proposal}. severity ∈ {info, watch, act}."""
    out = []
    n = stats.get("decided") or 0
    if n < _MIN_N:
        out.append({"severity": "info", "area": f"{label}: maturity",
                    "observation": f"Only {n} decided trades so far — not enough to judge.",
                    "proposal": "Keep accruing; treat current weights as provisional."})
        return out
    wr = stats.get("win_rate")
    # per-check edges that clear the evidence bar
    for r in rep:
        edge = r.get("edge")
        nn = min(r.get("n_pass", 0), r.get("n_fail", 0))
        if edge is None or nn < _MIN_N:
            continue
        if edge <= _EDGE_LO:
            out.append({"severity": "act", "area": f"{label}: check '{r['label']}'",
                        "observation": f"Win rate {r['win_rate_pass']}% when it passes vs "
                                       f"{r['win_rate_fail']}% when it fails (edge {edge:+.0f}pts) — it's "
                                       "not separating winners.",
                        "proposal": f"Down-weight or retire '{r['label']}' (currently "
                                    f"×{learned.get(r['label'], 1.0)})."})
        elif edge >= _EDGE_HI:
            out.append({"severity": "info", "area": f"{label}: check '{r['label']}'",
                        "observation": f"Strong edge {edge:+.0f}pts — earning its keep.",
                        "proposal": f"Keep / consider up-weighting '{r['label']}'."})
    # regime-conditional weakness
    for reg, g in (stats.get("by_regime") or {}).items():
        if g["win_rate"] is not None and wr is not None and g["win_rate"] <= wr - 15 and g["n"] >= 8:
            out.append({"severity": "act", "area": f"{label}: regime '{reg}'",
                        "observation": f"Win rate {g['win_rate']}% in '{reg}' vs {wr}% overall "
                                       f"({g['n']} trades) — this regime is hostile.",
                        "proposal": f"Raise the entry bar (or cut size) in the '{reg}' regime."})
    # loss streak
    st = stats.get("streak")
    if st and st["kind"] == "loss" and st["n"] >= 3:
        out.append({"severity": "watch", "area": f"{label}: streak",
                    "observation": f"{st['n']} resolved losses in a row.",
                    "proposal": "Confirm the no-trade / drawdown guards are doing their job; consider a pause."})
    # overall expectancy guard
    if wr is not None and wr < 40 and n >= 20:
        out.append({"severity": "watch", "area": f"{label}: overall",
                    "observation": f"Win rate {wr}% over {n} decided — below the 40% floor where R:R "
                                   "has to carry it.",
                    "proposal": "Verify average win ≥ ~1.5× average loss; if not, tighten signal selectivity."})
    return out


def _orb_window_note(rows: list[dict]) -> dict | None:
    """ORB-specific: which opening-range window is winning, from the shadow record's window tag."""
    byw = {}
    for t in rows:
        if t.get("status") not in ("win", "loss"):
            continue
        w = t.get("window_min")
        if w is None:
            continue
        byw.setdefault(w, [0, 0])
        byw[w][1] += 1
        if t["status"] == "win":
            byw[w][0] += 1
    byw = {w: {"n": v[1], "win_rate": round(v[0] / v[1] * 100, 1)} for w, v in byw.items() if v[1] >= 8}
    if len(byw) < 2:
        return None
    best = max(byw, key=lambda w: byw[w]["win_rate"])
    return {"severity": "info", "area": "ORB: opening-range window",
            "observation": f"By window: {byw}.",
            "proposal": f"{best}-min has the best win rate — consider it as orb_primary_window."}


def gather(cfg=None) -> dict:
    """Assemble the full quantitative review across buckets. Pure read; never raises."""
    mn = getattr(cfg, "adaptive_min_n", _MIN_N) if cfg else _MIN_N
    report = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT"),
              "buckets": {}, "findings": []}
    for scope, label, path in _BUCKETS:
        rows = _load(path)
        stats = _bucket_stats(rows)
        try:
            rep = attribution.report(scope=scope)
            learned = attribution.learned_weights(scope=scope, min_n=mn)
        except Exception:  # noqa: BLE001
            rep, learned = [], {}
        stats["learned_weights"] = learned
        report["buckets"][scope] = {"label": label, "stats": stats}
        report["findings"] += _findings(scope, label, stats, rep, learned)
        if scope == "orb":
            wn = _orb_window_note(rows)
            if wn:
                report["findings"].append(wn)
    # order: act first, then watch, then info
    _ord = {"act": 0, "watch": 1, "info": 2}
    report["findings"].sort(key=lambda f: _ord.get(f.get("severity"), 3))
    report["n_actions"] = sum(1 for f in report["findings"] if f.get("severity") == "act")
    return report


def run(cfg=None, write: bool = True) -> dict:
    """Build the report, attach an optional LLM narrative, persist it. Returns the report dict."""
    try:
        report = gather(cfg)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200], "findings": []}
    # optional LLM narrative — grounded only in the findings we computed
    try:
        if cfg is not None and getattr(cfg, "llm_enabled", False):
            import llm
            note = llm.analyst_review(report, cfg)
            if note:
                report["narrative"] = note
    except Exception:  # noqa: BLE001
        pass
    if write:
        try:
            with open(os.getenv("ANALYST_FILE", "analyst_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            hist_path = os.getenv("ANALYST_HISTORY_FILE", "analyst_history.json")
            hist = _load(hist_path)
            hist.append({"generated_at": report["generated_at"], "n_actions": report.get("n_actions", 0),
                         "findings": [{"severity": f["severity"], "area": f["area"],
                                       "proposal": f["proposal"]} for f in report["findings"][:12]]})
            with open(hist_path, "w") as f:
                json.dump(hist[-120:], f, indent=2)
        except Exception:  # noqa: BLE001
            pass
    return report


if __name__ == "__main__":
    from config import Config
    r = run(Config())
    print(f"[analyst] {r.get('n_actions', 0)} action items, {len(r.get('findings', []))} findings")
    for f in r.get("findings", [])[:12]:
        print(f"  [{f['severity']}] {f['area']}: {f['proposal']}")
