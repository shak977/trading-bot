"""Analyst memory — the loop that lets the nightly analyst grade its OWN past calls.

The bounded learned-weights loop and the directional gate already ADAPT trades automatically. What
was missing is whether those adaptations actually helped. This module is the analyst's memory: each
run it snapshots headline performance, remembers every proposal it has made (a ledger keyed by the
finding's area), and on later runs checks whether the metric that proposal targeted has improved
since — so the analyst can reinforce calls that worked, walk back calls that didn't, and report its
own hit rate instead of confidently repeating the same mistakes.

Reads/writes analyst_memory.json only. Deterministic, fail-silent, never raises.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

MEM_PATH = os.getenv("ANALYST_MEMORY_FILE", "analyst_memory.json")

_RECENT_DAYS = 7      # size of the "recent vs prior" comparison windows
_MIN_WIN = 5          # min decided trades in a window before a trend is trusted


def _load(path: str = MEM_PATH) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
            if isinstance(d, dict):
                d.setdefault("metrics", [])
                d.setdefault("ledger", {})
                return d
    except Exception:  # noqa: BLE001
        pass
    return {"metrics": [], "ledger": {}}


def _save(mem: dict, path: str = MEM_PATH) -> None:
    try:
        with open(path, "w") as f:
            json.dump(mem, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def _exit_day(t: dict) -> str:
    return (t.get("exit_date") or (t.get("exit_time") or "")[:10]) or ""


def _winrate(rows: list[dict]) -> float | None:
    dec = [t for t in rows if t.get("status") in ("win", "loss")]
    if not dec:
        return None
    return round(sum(1 for t in dec if t["status"] == "win") / len(dec) * 100, 1)


def windowed_winrate(rows: list[dict], start: str | None = None, end: str | None = None) -> dict:
    """Win rate over trades whose EXIT day falls in [start, end). Dates are ISO 'YYYY-MM-DD' so a
    plain string compare is a date compare. This is what lets us ask 'did it get better AFTER we
    changed something' rather than staring at a slow-moving cumulative number."""
    sel = []
    for t in rows:
        if t.get("status") not in ("win", "loss"):
            continue
        d = _exit_day(t)
        if not d or (start and d < start) or (end and d >= end):
            continue
        sel.append(t)
    return {"n": len(sel), "win_rate": _winrate(sel)}


def _scope_of(area: str, scopes) -> str | None:
    a = (area or "").lower()
    # findings are labelled e.g. "Daily / swing: check '...'", "Intraday: regime '...'"
    if a.startswith("daily") or "swing" in a.split(":")[0]:
        return "daily" if "daily" in scopes else None
    for s in scopes:
        if s != "daily" and a.startswith(s):
            return s
    return None


def review(report: dict, buckets_rows: dict, now: datetime | None = None, path: str = MEM_PATH) -> dict:
    """Grade the analyst against its own history and attach a self-review to `report`.

    `buckets_rows`: {scope: resolved-trade rows}. Mutates `report` to add `self_review` (a list of
    plain-English observations) and `self_confidence` (the analyst's own hit rate on past calls),
    updates the persisted memory, and returns it. Never raises."""
    try:
        now = now or datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        mem = _load(path)
        scopes = list(buckets_rows.keys())

        # 1) snapshot headline metrics for the time series
        snap = {"ts": now.strftime("%Y-%m-%d %H:%M GMT"), "date": today, "buckets": {}}
        for scope, rows in buckets_rows.items():
            snap["buckets"][scope] = {
                "win_rate": _winrate(rows),
                "decided": sum(1 for t in rows if t.get("status") in ("win", "loss")),
            }
        mem["metrics"].append(snap)
        mem["metrics"] = mem["metrics"][-180:]

        review_out: list[dict] = []

        # 2) recent-vs-prior trend per bucket (are things getting better lately?)
        d_recent = (now - timedelta(days=_RECENT_DAYS)).strftime("%Y-%m-%d")
        d_prior = (now - timedelta(days=2 * _RECENT_DAYS)).strftime("%Y-%m-%d")
        for scope, rows in buckets_rows.items():
            rec = windowed_winrate(rows, start=d_recent)
            pri = windowed_winrate(rows, start=d_prior, end=d_recent)
            if (rec["win_rate"] is not None and pri["win_rate"] is not None
                    and rec["n"] >= _MIN_WIN and pri["n"] >= _MIN_WIN):
                delta = round(rec["win_rate"] - pri["win_rate"], 1)
                trend = "improving" if delta >= 5 else "deteriorating" if delta <= -5 else "steady"
                review_out.append({
                    "kind": "trend", "scope": scope, "delta": delta,
                    "text": f"{scope}: last-{_RECENT_DAYS}d win rate {rec['win_rate']}% vs prior "
                            f"{pri['win_rate']}% ({delta:+.0f}pts) — {trend}.",
                })

        # 3) grade every recurring 'act' proposal against the metric it targeted
        ledger = mem["ledger"]
        for f in report.get("findings", []):
            if f.get("severity") != "act":
                continue
            key = f.get("area", "")
            if not key:
                continue
            scope = _scope_of(key, scopes)
            cur_wr = _winrate(buckets_rows.get(scope, [])) if scope else None
            e = ledger.get(key)
            if e is None:
                ledger[key] = {"first_seen": today, "first_wr": cur_wr, "times": 1, "last_seen": today}
            else:
                e["times"] = e.get("times", 1) + 1
                e["last_seen"] = today
                fw = e.get("first_wr")
                if fw is not None and cur_wr is not None:
                    d = round(cur_wr - fw, 1)
                    f["history"] = {"times_raised": e["times"], "since": e["first_seen"],
                                    "win_rate_delta": d}
                    verdict = ("improving since first flagged" if d > 0
                               else "still unresolved — win rate hasn't improved since first flagged")
                    review_out.append({
                        "kind": "proposal_grade", "scope": scope, "area": key, "delta": d,
                        "text": f"'{key}' raised {e['times']}× since {e['first_seen']}; bucket win "
                                f"rate {fw}%→{cur_wr}% ({d:+.0f}pts) — {verdict}.",
                    })

        # prune ledger entries not seen in a while so it doesn't grow unbounded
        cutoff = (now - timedelta(days=120)).strftime("%Y-%m-%d")
        for k in [k for k, v in ledger.items() if (v.get("last_seen") or "") < cutoff]:
            ledger.pop(k, None)

        # 4) the analyst's own hit rate: of graded proposals, how often did things improve after?
        graded = [r for r in review_out if r.get("kind") == "proposal_grade"]
        improved = [r for r in graded if r["delta"] > 0]
        report["self_review"] = review_out
        report["self_confidence"] = {
            "graded": len(graded), "improved": len(improved),
            "hit_rate": round(len(improved) / len(graded) * 100, 1) if graded else None,
        }
        _save(mem, path)
        return mem
    except Exception:  # noqa: BLE001 - self-review is additive; never break the analyst
        report.setdefault("self_review", [])
        report.setdefault("self_confidence", {"graded": 0, "improved": 0, "hit_rate": None})
        return {"metrics": [], "ledger": {}}
