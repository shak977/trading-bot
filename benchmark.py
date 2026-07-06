"""Weekly alpha benchmark — is the long book beating SPY, or just riding beta?

For every resolved LONG trade it compares the trade's return to SPY's move over the SAME holding
window (advised_date → exit_date), so the result isolates *selection* skill from market drift. Reports
average per-trade excess return (alpha), the share of trades that beat SPY, a by-conviction split, and
SPY buy-and-hold over the full window. Writes benchmark.json (latest) + appends benchmark_history.json
so we can watch whether the edge holds up as more (and eventually real) trades accrue.

Runs headless in the nightly analyst job. Fetches SPY from the Worker chart proxy (Yahoo fallback);
for offline testing set SPY_FILE to a saved {"bars":[{"t":ms,"c":px}]} JSON. Never raises out of run().
"""
from __future__ import annotations

import json
import os
import statistics as st
from bisect import bisect_right
from datetime import datetime, timezone

TRACK_FILE = os.getenv("TRACK_FILE", "track_record.json")


def _spy_from_worker_json(d: dict) -> dict:
    """Accept either the Worker chart shape {"bars":[{"t":ms,"c":px}]} or a raw Yahoo chart payload."""
    out = {}
    if isinstance(d.get("bars"), list):
        for b in d["bars"]:
            c = b.get("c")
            t = b.get("t")
            if c and t:
                out[datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d")] = c
        return out
    res = (d.get("chart") or {}).get("result") or []
    if res:
        ts = res[0].get("timestamp") or []
        cl = (((res[0].get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
        for t, c in zip(ts, cl):
            if c:
                out[datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")] = c
    return out


def fetch_spy(range_: str = "6mo") -> dict:
    """SPY {date: close}. Offline override via SPY_FILE; else Worker chart proxy, then Yahoo."""
    sf = os.getenv("SPY_FILE", "").strip()
    if sf:
        try:
            return _spy_from_worker_json(json.load(open(sf)))
        except Exception:  # noqa: BLE001
            return {}
    import urllib.request
    urls = []
    base = os.getenv("LIVE_QUOTES_URL", "").strip().rstrip("/")
    if base:
        urls.append(f"{base.split('?')[0]}?chart=SPY&range={range_}&interval=1d")
    urls.append(f"https://query1.finance.yahoo.com/v8/finance/chart/SPY?range={range_}&interval=1d")
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
            got = _spy_from_worker_json(d)
            if got:
                return got
        except Exception:  # noqa: BLE001
            continue
    return {}


def _clab(t: dict):
    c = t.get("conviction")
    return c.get("label") if isinstance(c, dict) else c


def compute(track: list[dict], spy: dict) -> dict:
    """Per-trade excess vs SPY over matched holding windows, for resolved LONG trades."""
    if not spy:
        return {"error": "no SPY data", "n": 0}
    days = sorted(spy)

    def spy_on(date: str):
        i = bisect_right(days, date) - 1
        return spy[days[i]] if i >= 0 else None

    longs = [t for t in track if (t.get("direction") or "LONG") == "LONG"
             and t.get("status") in ("win", "loss") and t.get("return_pct") is not None
             and t.get("advised_date") and t.get("exit_date")]
    rows = []
    for t in longs:
        se, sx = spy_on(t["advised_date"]), spy_on(t["exit_date"])
        if not se or not sx or se <= 0:
            continue
        spy_ret = (sx / se - 1) * 100
        rows.append({"ret": t["return_pct"], "spy": spy_ret,
                     "excess": t["return_pct"] - spy_ret, "conv": _clab(t)})
    n = len(rows)
    if not n:
        return {"error": "no matched trades", "n": 0}
    avg = lambda xs: round(st.mean(xs), 2) if xs else None  # noqa: E731
    beat = sum(1 for r in rows if r["excess"] > 0)
    ds = sorted(t["advised_date"] for t in longs)
    de = sorted(t["exit_date"] for t in longs)
    p0, p1 = spy_on(ds[0]), spy_on(de[-1])
    by_conv = {}
    for lab in ("High", "Medium", "Low"):
        g = [r for r in rows if r["conv"] == lab]
        if g:
            by_conv[lab] = {"n": len(g), "avg_excess": avg([r["excess"] for r in g]),
                            "beat_pct": round(sum(1 for r in g if r["excess"] > 0) / len(g) * 100, 1)}
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT"),
        "n": n, "window": {"from": ds[0], "to": de[-1]},
        "avg_trade_return": avg([r["ret"] for r in rows]),
        "avg_spy_matched": avg([r["spy"] for r in rows]),
        "avg_excess": avg([r["excess"] for r in rows]),
        "median_excess": round(st.median([r["excess"] for r in rows]), 2),
        "beat_spy_pct": round(beat / n * 100, 1),
        "spy_buy_hold_pct": round((p1 / p0 - 1) * 100, 2) if (p0 and p1) else None,
        "by_conviction": by_conv,
    }


def run(write: bool = True) -> dict:
    try:
        track = json.load(open(TRACK_FILE))
    except Exception as e:  # noqa: BLE001
        return {"error": f"track load: {str(e)[:120]}", "n": 0}
    rep = compute(track, fetch_spy())
    if write and not rep.get("error"):
        try:
            with open(os.getenv("BENCHMARK_FILE", "benchmark.json"), "w") as f:
                json.dump(rep, f, indent=2)
            hp = os.getenv("BENCHMARK_HISTORY_FILE", "benchmark_history.json")
            try:
                hist = json.load(open(hp))
            except Exception:  # noqa: BLE001
                hist = []
            today = rep["generated_at"][:10]
            hist = [h for h in hist if h.get("date") != today]  # one entry per day
            hist.append({"date": today, "n": rep["n"], "avg_excess": rep["avg_excess"],
                         "beat_spy_pct": rep["beat_spy_pct"], "spy_buy_hold_pct": rep["spy_buy_hold_pct"]})
            with open(hp, "w") as f:
                json.dump(hist[-260:], f, indent=2)
        except Exception:  # noqa: BLE001
            pass
    return rep


if __name__ == "__main__":
    r = run()
    if r.get("error"):
        print(f"[benchmark] {r['error']}")
    else:
        print(f"[benchmark] {r['n']} long trades {r['window']['from']}→{r['window']['to']}: "
              f"avg excess vs SPY {r['avg_excess']:+}%, beat {r['beat_spy_pct']}% "
              f"(SPY buy&hold {r['spy_buy_hold_pct']:+}%)")
        for lab, g in r["by_conviction"].items():
            print(f"  {lab}: n={g['n']} excess {g['avg_excess']:+}% beat {g['beat_pct']}%")
