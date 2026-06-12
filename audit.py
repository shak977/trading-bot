"""Data-integrity audit. Fetches the live signals.json (or reads a local file)
and sanity-checks every data point, flagging anything implausible.

Usage:
  python3 audit.py                      # audit the live published data
  python3 audit.py signals.json         # audit a local file
  python3 audit.py https://.../signals.json
"""
from __future__ import annotations

import json
import sys
import urllib.request

DEFAULT_URL = "https://shak977.github.io/trading-bot/signals.json"


def load(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=20) as r:
            return json.load(r)
    with open(src) as f:
        return json.load(f)


def audit_data(d):
    """Pure audit: returns (n_checks, [flag, ...]). No printing."""
    flags: list[dict] = []
    state = {"checks": 0}

    def flag(msg, level="error"):
        flags.append({"level": level, "msg": msg})

    def chk(cond, msg):
        state["checks"] += 1
        if not cond:
            flag(msg, "error")

    # ---- macro ----
    m = d.get("macro")
    if m:
        rng = {"y10": (0, 12), "y2": (0, 12), "cpi_yoy": (-5, 25),
               "unemployment": (0, 30), "fed_funds": (0, 12)}
        for k, (lo, hi) in rng.items():
            if m.get(k) is not None:
                chk(lo <= m[k] <= hi, f"macro {k}={m[k]} outside plausible {lo}..{hi}")

    charts = d.get("charts", {})
    for s in d.get("signals", []):
        sym, px = s.get("symbol"), s.get("price")
        def F(msg, level="warn"):  # noqa: E306  (extreme-but-plausible by default)
            flag(f"[{sym}] {msg}", level)

        # ---- core fields ----
        chk(isinstance(px, (int, float)) and px > 0, f"[{sym}] price not positive: {px}")
        rsi = s.get("rsi")
        chk(rsi is None or 0 <= rsi <= 100, f"[{sym}] RSI out of range: {rsi}")
        for k in ("fast_ma", "slow_ma"):
            if s.get(k) is not None:
                chk(s[k] > 0, f"[{sym}] {k} not positive: {s[k]}")
        rv = s.get("rel_volume")
        if rv is not None:
            chk(rv >= 0, f"[{sym}] rel_volume negative: {rv}")
            if rv > 25:
                F(f"rel_volume suspiciously high: {rv}x")

        # ---- trade plan (direction-aware: shorts invert stop/target geometry) ----
        p = s.get("plan") or {}
        e, st, tg, rr = p.get("entry"), p.get("stop"), p.get("target"), p.get("rr")
        is_short = (p.get("direction") or s.get("direction") or "LONG") == "SHORT"
        if is_short:
            if e and st:
                chk(st > e, f"[{sym}] short stop {st} should be > entry {e}")
            if e and tg:
                chk(0 < tg < e, f"[{sym}] short target {tg} should be 0<target<entry {e}")
        else:
            if e and st:
                chk(st < e and st > 0, f"[{sym}] stop {st} should be 0<stop<entry {e}")
            if e and tg:
                chk(tg > e, f"[{sym}] target {tg} should be > entry {e}")
        if rr is not None:
            chk(0 < rr < 50, f"[{sym}] risk:reward implausible: {rr}")

        # ---- conviction ----
        c = s.get("conviction") or {}
        if c:
            chk(0 <= c.get("score_pct", -1) <= 100, f"[{sym}] conviction score out of range")
            chk(len(c.get("checks", [])) > 0, f"[{sym}] conviction has no checks")

        # ---- context sanity ----
        ctx = s.get("context") or {}
        dc = ctx.get("day_change_pct")
        if dc is not None and abs(dc) > 40:
            F(f"day move {dc}% — possible bad/unadjusted data")
        pfh = ctx.get("pct_from_high")
        if pfh is not None and pfh > 3:
            F(f"price {pfh}% ABOVE its own 1y high (data issue?)")
        pfl = ctx.get("pct_from_low")
        if pfl is not None and pfl < -3:
            F(f"price {pfl}% BELOW its own 1y low (data issue?)")
        ap = ctx.get("atr_pct")
        if ap is not None and ap > 30:
            F(f"ATR {ap}% of price — extreme volatility, check data")

        # ---- edge ----
        edge = s.get("edge")
        if edge and edge.get("win_rate") is not None:
            chk(0 <= edge["win_rate"] <= 100, f"[{sym}] edge win_rate out of range")

        # ---- fundamentals ----
        fu = s.get("fundamentals") or {}
        if fu.get("pe") is not None and (fu["pe"] < -100 or fu["pe"] > 2000):
            F(f"P/E implausible: {fu['pe']}")
        tm = fu.get("target_mean")
        if tm and px and not (0.2 * px <= tm <= 5 * px):
            F(f"price target ${tm} far from price ${px} (>5x or <0.2x)")

        # ---- chart series integrity ----
        ch = charts.get(sym)
        if ch:
            o, h, l, cl, t = ch.get("open"), ch.get("high"), ch.get("low"), ch.get("close"), ch.get("t")
            n = len(cl or [])
            chk(n > 0, f"[{sym}] empty chart")
            # timestamps strictly increasing
            if t and any(t[i] >= t[i + 1] for i in range(len(t) - 1)):
                F("chart timestamps not strictly increasing", "error")
            # OHLC integrity + split-artifact detection
            bad_ohlc = day_jumps = 0
            for i in range(n):
                if None in (h[i], l[i], cl[i]):
                    continue
                if h[i] < l[i] - 1e-6 or cl[i] > h[i] + 1e-6 or cl[i] < l[i] - 1e-6:
                    bad_ohlc += 1
                if i > 0 and cl[i - 1]:
                    if abs(cl[i] / cl[i - 1] - 1) > 0.5:   # >50% one-day jump
                        day_jumps += 1
            if bad_ohlc:
                F(f"{bad_ohlc} bars violate OHLC (high<low etc.)", "error")
            if day_jumps:
                F(f"{day_jumps} one-day jumps >50% (likely unadjusted split)", "error")
            # Bollinger band order
            bu, bl = ch.get("bb_up"), ch.get("bb_lo")
            if bu and bl:
                viol = sum(1 for i in range(len(bu)) if bu[i] is not None and bl[i] is not None and bu[i] < bl[i])
                if viol:
                    F(f"{viol} Bollinger bars upper<lower", "error")
            # window return (overview-style). A *smooth* big move is a real rally,
            # not an error — so only flag a large 3mo move when it's driven by a
            # single-day discontinuity (the true split/bad-print signature), or when
            # it's absurd in absolute terms. Pair closes w/ timestamps to print prices.
            pairs = [(t[i] if t else None, cl[i]) for i in range(n) if cl[i] is not None]
            win = pairs[-60:]
            if len(win) > 2 and win[0][1]:
                base, last = win[0][1], win[-1][1]
                ret = (last / base - 1) * 100
                wv = [v for _, v in win]
                max_day = max((abs(wv[i] / wv[i - 1] - 1) for i in range(1, len(wv)) if wv[i - 1]), default=0)
                if abs(ret) > 250 or (abs(ret) > 80 and max_day > 0.45):
                    def _dt(ms):
                        if not ms:
                            return "?"
                        import datetime as _d
                        return _d.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
                    F(f"~3mo return {ret:+.0f}% (${base:.2f} {_dt(win[0][0])} -> "
                      f"${last:.2f} {_dt(win[-1][0])}, max 1-day {max_day*100:.0f}%) — verify split/data")

    return state["checks"], flags


def main(src):
    d = load(src)
    print(f"Auditing {d.get('mode')} build @ {d.get('generated_at')} — "
          f"{len(d.get('signals', []))} signals shown\n")
    checks, flags = audit_data(d)
    print(f"Ran {checks} structural checks.")
    errs = [f["msg"] for f in flags if f.get("level") == "error"]
    warns = [f["msg"] for f in flags if f.get("level") == "warn"]
    if errs:
        print(f"\n❌ {len(errs)} likely DATA ERROR(s) flagged:")
        for m in errs:
            print("  -", m)
    if warns:
        print(f"\n⚠  {len(warns)} extreme-but-likely-real item(s) (volatile names, verify):")
        for m in warns:
            print("  -", m)
    if not flags:
        print("\n✅ No anomalies found — all data points look sane.")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL))
