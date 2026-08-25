"""Repeatable system diagnostic — reproduces the Jul-2026 deep dive on demand.

Run:  python3 system_diagnostic.py            # prints the report
      python3 system_diagnostic.py --json     # also writes system_diagnostic.json

Reads track_record.json (+ setups_study.json, timing_study.json if present). No network, no deps
beyond the stdlib. Everything is diagnostic — it changes nothing. The point is to answer, every
week on fresh data, the questions that actually move the P&L:
  1. Long book vs short book (direction is the biggest lever).
  2. R:R buckets — where do targets get reached vs stopped (the stop/target mismatch).
  3. Per-check cohorts, LONGS-ONLY (de-confounded from direction) — which checks actually predict.
  4. Setup edges vs baseline, and the timing gate.
"""
from __future__ import annotations
import json
import math
import statistics as st
import sys
from collections import Counter


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _wilson(k, n):
    if not n:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def _stats(rows):
    n = len(rows)
    if not n:
        return None
    w = sum(1 for t in rows if t["status"] == "win")
    ret = [t["return_pct"] for t in rows if isinstance(t.get("return_pct"), (int, float))]
    return {"n": n, "wr": 100 * w / n, "exp": (st.mean(ret) if ret else 0.0)}


def _has(t, label, statuses):
    return any(c.get("label") == label and c.get("status") in statuses for c in (t.get("checks") or []))


def run(track_path="track_record.json"):
    tr = _load(track_path) or []
    res = [t for t in tr if t.get("status") in ("win", "loss") and isinstance(t.get("return_pct"), (int, float))]
    out = {"resolved": len(res)}
    if not res:
        return out

    # 1. direction
    lo = [t for t in res if t.get("direction") == "LONG"]
    sh = [t for t in res if t.get("direction") == "SHORT"]
    out["by_direction"] = {"LONG": _stats(lo), "SHORT": _stats(sh), "ALL": _stats(res)}

    # 1b. ACTIONABLE win rate — only the fresh BUYs we'd actually take (needs 'action' recorded at entry,
    # so this populates forward). This is the number that matters, vs the raw all-longs book above.
    buys = [t for t in lo if t.get("action") == "BUY"]
    if buys:
        out["buy_only"] = _stats(buys)
        hp = [t for t in buys if isinstance(t.get("p_win"), (int, float)) and t["p_win"] >= 0.52]
        if hp:
            out["buy_only_pwin52"] = _stats(hp)

    # 1c. Are the ALERTS any good? The ones that ping you should beat the ones that don't — historically
    # they did the OPPOSITE (alerted longs 50.3%/+0.46% vs non-alerted 66.8%/+1.22%) because the ping was
    # gated on the inverted conviction score. Now gated on P(win); this watches whether the fix worked.
    al = [t for t in lo if t.get("alerted")]
    nal = [t for t in lo if t.get("alerted") is False]
    if al:
        out["alerts"] = {"alerted": _stats(al), "not_alerted": _stats(nal)}

    # 2. R:R buckets (longs incl. expired for exit mix)
    lo_all = [t for t in tr if t.get("direction") == "LONG" and t.get("status") in ("win", "loss", "expired")
              and isinstance(t.get("rr"), (int, float))]
    buckets = {}
    for lab, lo_, hi_ in (("rr<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), ("rr>=4", 4, 1e9)):
        rows = [t for t in lo_all if lo_ <= t["rr"] < hi_]
        n = len(rows)
        if n:
            c = Counter(t["status"] for t in rows)
            buckets[lab] = {"n": n, "target_hit_pct": round(100 * c["win"] / n),
                            "stopped_pct": round(100 * c["loss"] / n), "expired_pct": round(100 * c["expired"] / n),
                            "avg_ret": round(st.mean(t["return_pct"] for t in rows if isinstance(t.get("return_pct"), (int, float))), 2)}
    out["rr_buckets_longs"] = buckets

    # 3. per-check cohorts, LONGS-ONLY (de-confounded)
    labels = sorted({c["label"] for t in lo for c in (t.get("checks") or [])})
    checks = {}
    for lab in labels:
        P = [t for t in lo if _has(t, lab, ("pass",))]
        NP = [t for t in lo if _has(t, lab, ("warn", "fail"))]
        sp, snp = _stats(P), _stats(NP)
        if not sp or not snp or sp["n"] < 20 or snp["n"] < 20:
            continue
        plo, phi = _wilson(sum(1 for t in P if t["status"] == "win"), sp["n"])
        nlo, nhi = _wilson(sum(1 for t in NP if t["status"] == "win"), snp["n"])
        sig = "" if (phi >= nlo and plo <= nhi) else "SIG"
        checks[lab] = {"pass_n": sp["n"], "pass_wr": round(sp["wr"], 1), "pass_exp": round(sp["exp"], 2),
                       "notpass_n": snp["n"], "notpass_wr": round(snp["wr"], 1), "notpass_exp": round(snp["exp"], 2),
                       "win_delta": round(sp["wr"] - snp["wr"], 1), "exp_delta": round(sp["exp"] - snp["exp"], 2), "sig": sig}
    out["checks_longs_only"] = dict(sorted(checks.items(), key=lambda kv: kv[1]["win_delta"]))

    # 4. setups + timing (if present)
    ss = _load("setups_study.json")
    if ss:
        out["setups_verdict"] = ss.get("verdict")
    tsy = _load("timing_study.json")
    if tsy:
        out["timing_verdict"] = tsy.get("verdict")
    return out


def report(d):
    L = []
    L.append(f"SYSTEM DIAGNOSTIC — {d.get('resolved', 0)} resolved trades\n" + "=" * 60)
    bd = d.get("by_direction") or {}
    L.append("\nDIRECTION (the biggest lever):")
    for k in ("LONG", "SHORT", "ALL"):
        s = bd.get(k)
        if s:
            L.append(f"  {k:6}: n={s['n']:4d}  win={s['wr']:4.1f}%  exp={s['exp']:+5.2f}%")
    L.append("\nR:R BUCKETS (longs — where targets get reached vs stopped):")
    for k, b in (d.get("rr_buckets_longs") or {}).items():
        L.append(f"  {k:6}: n={b['n']:3d}  target-hit={b['target_hit_pct']:3d}%  stopped={b['stopped_pct']:3d}%  avgRet={b['avg_ret']:+5.2f}%")
    L.append("\nCHECKS, LONGS-ONLY (de-confounded — most anti-predictive first):")
    for lab, c in (d.get("checks_longs_only") or {}).items():
        L.append(f"  {lab[:26]:26} pass {c['pass_wr']:4.1f}%/{c['pass_exp']:+5.2f} (n={c['pass_n']})  "
                 f"not-pass {c['notpass_wr']:4.1f}%/{c['notpass_exp']:+5.2f} (n={c['notpass_n']})  "
                 f"winΔ {c['win_delta']:+5.1f} {c['sig']}")
    if d.get("setups_verdict"):
        L.append("\nSETUPS: " + str(d["setups_verdict"]))
    if d.get("timing_verdict"):
        L.append("\nTIMING: " + str(d["timing_verdict"]))
    return "\n".join(L)


if __name__ == "__main__":
    d = run()
    print(report(d))
    if "--json" in sys.argv:
        json.dump(d, open("system_diagnostic.json", "w"), indent=1)
        print("\n[wrote system_diagnostic.json]")
