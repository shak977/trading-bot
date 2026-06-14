# Relative-Strength Engine + Factor-Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add relative-strength (RS) ranking vs a benchmark so the scan surfaces more & better names, weight it heavily in the conviction model, and add a factor-attribution loop that measures which conviction checks actually predict wins — so the wider funnel stays clean.

**Architecture:** A new pure module `rs.py` (mirrors `momentum.py`) computes per-symbol RS vs SPY and percentile-ranks the universe. `scan()` computes RS across all fetched rows, stashes it in `row["factors"]["rs"]`, and recomputes conviction. `_conviction()` gains a weighted RS check (reading `factors["rs"]`, like it already reads `factors["confluence"]`/`factors["edge"]`). `tracker.py` snapshots each call's conviction checks; a new `attribution.py` reads resolved trades and reports per-check win rates, surfaced on the dashboard.

**Tech Stack:** Python 3.13, pandas, numpy. Tests live in `selftest.py` using the `_ok(name, cond)` convention, run with `python3 selftest.py` (no pytest). Synthetic data via `data.synthetic_bars`.

---

## File Structure

- **Create** `rs.py` — pure RS math (no I/O), template = `momentum.py`.
- **Create** `attribution.py` — read `track_record.json`, compute per-check win rates (no I/O beyond reading the log path).
- **Modify** `config.py` — RS + benchmark knobs.
- **Modify** `scanner.py` — weighted conviction scoring, RS conviction check, RS computed across the universe in `scan()`, RS-ordered universe cap.
- **Modify** `tracker.py` — snapshot conviction checks on each logged call.
- **Modify** `dashboard.py` — RS percentile display + attribution panel.
- **Modify** `selftest.py` — one new test function per task, registered in `main()`.

---

## Task 1: Config knobs for RS

**Files:**
- Modify: `config.py:49` (after `rel_volume_window`, within the Universe/data block)

- [ ] **Step 1: Add config fields**

In `config.py`, immediately after line 49 (`rel_volume_window: int = 20 ...`), add:

```python
    # --- Relative strength (vs benchmark) ---
    benchmark: str = "SPY"                       # name ranked against; fetched once per scan
    rs_lookbacks: tuple[int, ...] = (21, 63, 126)  # trading-day windows for blended RS
    rs_weights: tuple[float, ...] = (0.2, 0.3, 0.5)  # weight per lookback (sums to 1.0)
    rs_conviction_weight: float = 2.0            # weight of the RS check in the conviction average
    rs_pass_pct: int = 70                        # RS percentile >= this = pass (long); <= 100-this = pass (short)
    rs_fail_pct: int = 40                        # RS percentile <= this = fail (long); >= 100-this = fail (short)
```

- [ ] **Step 2: Verify it imports**

Run: `python3 -c "from config import CONFIG; print(CONFIG.benchmark, CONFIG.rs_lookbacks, CONFIG.rs_conviction_weight)"`
Expected: `SPY (21, 63, 126) 2.0`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: add relative-strength knobs"
```

---

## Task 2: `rs.py` — RS score + universe ranking (pure)

**Files:**
- Create: `rs.py`
- Test: `selftest.py` (new `test_rs`)

- [ ] **Step 1: Write the failing test**

Add to `selftest.py` (above `def main():`):

```python
def test_rs():
    print("relative strength:")
    import numpy as np
    import pandas as pd
    import rs
    idx = pd.date_range("2024-01-01", periods=300)
    # benchmark: flat-ish +10% over the window
    bench = pd.Series(np.linspace(100, 110, 300), index=idx)
    strong = pd.Series(np.linspace(100, 200, 300), index=idx)   # way ahead of bench
    weak = pd.Series(np.linspace(100, 95, 300), index=idx)       # behind bench
    s_strong = rs.rs_score(strong, bench, (21, 63, 126), (0.2, 0.3, 0.5))
    s_weak = rs.rs_score(weak, bench, (21, 63, 126), (0.2, 0.3, 0.5))
    _ok("strong name has positive RS", s_strong > 0)
    _ok("weak name has negative RS", s_weak < 0)
    _ok("strong beats weak", s_strong > s_weak)
    _ok("short history -> None", rs.rs_score(strong.iloc[:50], bench.iloc[:50], (21, 63, 126), (0.2, 0.3, 0.5)) is None)
    ranks = rs.rank_universe({"A": s_strong, "B": s_weak, "C": 0.0})
    _ok("ranks cover all symbols", set(ranks) == {"A", "B", "C"})
    _ok("strong is top percentile", ranks["A"]["pct"] == 100)
    _ok("weak is bottom percentile", ranks["B"]["pct"] <= ranks["C"]["pct"] <= ranks["A"]["pct"])
    _ok("None scores excluded from ranking", "Z" not in rs.rank_universe({"A": s_strong, "Z": None}))
```

Register it in `main()` (add `test_rs()` near the top, after `test_indicators()`).

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'rs'`

- [ ] **Step 3: Implement `rs.py`**

```python
"""Relative strength vs a benchmark — cross-sectional leadership ranking.

Momentum tells you a name is rising; relative strength tells you it's rising
*faster than the market*. Leadership (high RS) is one of the most robust
momentum factors (O'Neil's RS line, AQR cross-sectional momentum). We blend a
few trailing windows, subtract the benchmark's return over each, then
percentile-rank the whole scanned universe so the conviction model can reward
leaders and penalise laggards. Pure functions, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_return(close: pd.Series, lookback: int) -> float | None:
    if close is None or len(close) < lookback + 1:
        return None
    past = close.iloc[-lookback - 1]
    last = close.iloc[-1]
    if not past or np.isnan(past) or np.isnan(last):
        return None
    return float(last / past - 1.0)


def rs_score(close: pd.Series, bench_close: pd.Series,
             lookbacks=(21, 63, 126), weights=(0.2, 0.3, 0.5)) -> float | None:
    """Blended excess return of `close` over `bench_close` across `lookbacks`.

    Returns None if either series lacks enough history for the longest lookback.
    Positive = the name outran the benchmark; negative = it lagged.
    """
    if bench_close is None or len(bench_close) < max(lookbacks) + 1:
        return None
    total, wsum = 0.0, 0.0
    for lb, w in zip(lookbacks, weights):
        r = _trailing_return(close, lb)
        b = _trailing_return(bench_close, lb)
        if r is None or b is None:
            return None
        total += w * (r - b)
        wsum += w
    if wsum <= 0:
        return None
    return total / wsum


def rank_universe(scores: dict) -> dict:
    """Percentile-rank RS scores across the universe (0-100, best = 100).

    Symbols whose score is None/NaN are dropped. Ties share the average percentile.
    Returns {symbol: {"rs": float, "pct": int}}.
    """
    clean = {s: float(v) for s, v in scores.items()
             if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if not clean:
        return {}
    ser = pd.Series(clean)
    pct = ser.rank(pct=True, method="average") * 100.0
    return {s: {"rs": round(clean[s], 4), "pct": int(round(pct[s]))} for s in clean}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — all `test_rs` lines show `ok`, ends `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add rs.py selftest.py
git commit -m "rs: relative-strength score + universe percentile ranking"
```

---

## Task 3: Weighted conviction scoring (backward-compatible)

**Files:**
- Modify: `scanner.py:500-501` (the `add()` helper) and `scanner.py:754-755` (the score sum)
- Test: `selftest.py` (new `test_conviction_weighting`)

This makes the conviction average support per-check weights, defaulting to 1.0 so existing scores are unchanged. Required before the RS check can carry 2× weight.

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_conviction_weighting():
    print("conviction weighting:")
    import scanner
    from data import synthetic_bars
    from config import CONFIG
    # baseline conviction with all default (1.0) weights must be unchanged vs before:
    row = scanner._analyse("MSFT", synthetic_bars("MSFT", n=CONFIG.lookback_days), CONFIG, CONFIG.starting_cash)
    checks = row["conviction"]["checks"]
    _ok("every check has a weight field defaulting to 1.0", all(c.get("weight", 1.0) == 1.0 for c in checks))
    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    manual = round(sum(pts[c["status"]] for c in checks) / len(checks) * 100)
    _ok("unweighted score matches simple average", row["conviction"]["score_pct"] == manual)
```

Register `test_conviction_weighting()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL on "every check has a weight field" (checks currently have no `weight` key, so `.get("weight",1.0)` is 1.0 — this passes; the meaningful failure is below). To force a true red, also assert presence:

Change the first `_ok` to:
```python
    _ok("every check carries an explicit weight", all("weight" in c for c in checks))
```
Run again — Expected: FAIL (`weight` not in checks yet).

- [ ] **Step 3: Implement weighted scoring**

In `scanner.py`, change the `add()` helper (around line 500) to accept a weight:

```python
    def add(label, status, note, weight=1.0):
        checks.append({"label": label, "status": status, "note": note, "weight": weight})
```

Then change the score computation (around line 754-755) from:

```python
    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = sum(pts[c["status"]] for c in checks) / len(checks)
```

to:

```python
    pts = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    wsum = sum(c.get("weight", 1.0) for c in checks)
    score = sum(pts[c["status"]] * c.get("weight", 1.0) for c in checks) / wsum if wsum else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_conviction_weighting` green, and **all pre-existing tests still pass** (default weights = 1.0 keep every score identical).

- [ ] **Step 5: Commit**

```bash
git add scanner.py selftest.py
git commit -m "scanner: weighted conviction checks (default 1.0, scores unchanged)"
```

---

## Task 4: RS conviction check (reads `factors["rs"]`)

**Files:**
- Modify: `scanner.py` `_conviction()` — add an RS check block; `scanner.py:769` `rescore()` already passes `factors` through.
- Test: `selftest.py` (new `test_rs_conviction`)

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_rs_conviction():
    print("rs conviction check:")
    import scanner
    from data import synthetic_bars
    from config import CONFIG
    df = synthetic_bars("MSFT", n=CONFIG.lookback_days)
    row = scanner._analyse("MSFT", df, CONFIG, CONFIG.starting_cash)
    # inject a strong-leader RS factor and recompute
    row.setdefault("factors", {})["rs"] = {"rs": 0.25, "pct": 92}
    scanner.rescore(row, CONFIG)
    labels = [c["label"] for c in row["conviction"]["checks"]]
    _ok("RS check appears once RS factor present", "Leading the market?" in labels)
    rs_check = next(c for c in row["conviction"]["checks"] if c["label"] == "Leading the market?")
    _ok("RS check is weighted heavily", rs_check["weight"] == CONFIG.rs_conviction_weight)
    # a long leader (high pct) should pass; a long laggard should fail
    _ok("high-RS long passes", rs_check["status"] == "pass")
    row["factors"]["rs"] = {"rs": -0.2, "pct": 8}
    scanner.rescore(row, CONFIG)
    rs_check = next(c for c in row["conviction"]["checks"] if c["label"] == "Leading the market?")
    _ok("low-RS long fails", rs_check["status"] == "fail")
```

Register `test_rs_conviction()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — "Leading the market?" not in labels.

- [ ] **Step 3: Implement the RS check**

In `scanner.py` `_conviction()`, after the trend check block (right after the "Is it trending up/down?" `add(...)`, before the regime block ~line 517), insert:

```python
    # Relative strength — is this name leading or lagging the market?
    rs_f = factors.get("rs")
    if rs_f and rs_f.get("pct") is not None:
        pct = rs_f["pct"]
        w = cfg.rs_conviction_weight
        if short:
            # for a short we want a laggard: low percentile is good
            if pct <= (100 - cfg.rs_pass_pct):
                add("Leading the market?", "pass",
                    f"Yes (for a short) — RS percentile {pct}; this name lags the market, so weakness has company.", w)
            elif pct >= (100 - cfg.rs_fail_pct):
                add("Leading the market?", "fail",
                    f"No — RS percentile {pct}; you'd be shorting a relative leader.", w)
            else:
                add("Leading the market?", "warn",
                    f"Mixed — RS percentile {pct}; no clear relative weakness yet.", w)
        else:
            if pct >= cfg.rs_pass_pct:
                add("Leading the market?", "pass",
                    f"Yes — RS percentile {pct}; this name is outrunning the market (leadership).", w)
            elif pct <= cfg.rs_fail_pct:
                add("Leading the market?", "fail",
                    f"No — RS percentile {pct}; this name lags the market (laggard).", w)
            else:
                add("Leading the market?", "warn",
                    f"Mixed — RS percentile {pct}; roughly in line with the market.", w)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_rs_conviction` green; existing tests unaffected (no `factors["rs"]` set elsewhere yet).

- [ ] **Step 5: Commit**

```bash
git add scanner.py selftest.py
git commit -m "scanner: RS conviction check (weighted), reads factors[rs]"
```

---

## Task 5: Compute RS across the universe in `scan()`

**Files:**
- Modify: `scanner.py` `scan()` (~line 866-920) — fetch benchmark, compute RS over fetched frames, set `row["factors"]["rs"]`, recompute conviction.
- Test: `selftest.py` (new `test_rs_in_scan`)

The `_df` frame is stashed on each row during `_analyse` and only popped after `show_top`. We compute RS before any pop.

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_rs_in_scan():
    print("rs wired into scan:")
    import scanner
    from config import CONFIG
    rows = scanner.scan(CONFIG, live=False)   # synthetic universe, deterministic
    _ok("scan returns rows", len(rows) > 0)
    rs_rows = [r for r in rows if (r.get("factors") or {}).get("rs")]
    _ok("at least one row carries an RS factor", len(rs_rows) > 0)
    pcts = [r["factors"]["rs"]["pct"] for r in rs_rows]
    _ok("RS percentiles are 0..100", all(0 <= p <= 100 for p in pcts))
    # a row with RS present should also have the RS conviction check folded in
    sample = rs_rows[0]
    _ok("RS check present in conviction", any(c["label"] == "Leading the market?" for c in sample["conviction"]["checks"]))
```

Register `test_rs_in_scan()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — no row carries an RS factor.

- [ ] **Step 3: Implement RS computation in `scan()`**

In `scanner.py`, add this import near the top with the others (after `import momentum` if present, else with module imports):

```python
import rs as rs_mod
```

Inside `scan()`, after the main symbol loop fills `rows` and before `rows.sort(key=_rank_key)` (line 901), insert:

```python
    # --- relative strength across the just-scanned universe (vs the benchmark) ---
    try:
        if live:
            bench_df = get_bars(cfg.benchmark, cfg)
        else:
            bench_df = synthetic_bars(cfg.benchmark, n=cfg.lookback_days)
        bench_close = bench_df["close"] if bench_df is not None and len(bench_df) else None
        if bench_close is not None:
            scores = {}
            for row in rows:
                df = row.get("_df")
                if df is not None and len(df):
                    scores[row["symbol"]] = rs_mod.rs_score(
                        df["close"], bench_close, cfg.rs_lookbacks, cfg.rs_weights)
            ranked = rs_mod.rank_universe(scores)
            for row in rows:
                rf = ranked.get(row["symbol"])
                if rf:
                    row.setdefault("factors", {})["rs"] = rf
                    rescore(row, cfg)   # fold the RS check into conviction
    except Exception:  # noqa: BLE001 - RS is additive; never break the scan
        pass
```

Note: `rescore(row, cfg)` recomputes conviction from `row["factors"]` (now including `rs`) and refreshes the desk read. It is safe to call with no research args (they default to None) at scan time; the later dashboard `rescore(..., sentiment=...)` call still reads `factors["rs"]` and keeps the RS check.

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_rs_in_scan` green. (Synthetic benchmark `SPY` is produced by `synthetic_bars`, which is deterministic per symbol name.)

- [ ] **Step 5: Commit**

```bash
git add scanner.py selftest.py
git commit -m "scanner: compute RS across the universe and fold into conviction"
```

---

## Task 6: RS-ordered universe cap (wider funnel, bounded cost)

**Files:**
- Modify: `config.py` (add a larger static pool flag) and `scanner.py` `build_universe()` (~line 161).
- Test: `selftest.py` (new `test_universe_cap`)

We expand the candidate pool but keep `max_candidates` so cost stays bounded. Since RS needs bars (not available at `build_universe` time), the bounded-cost lever here is simply: keep the existing cap, and let RS ordering happen post-fetch (Task 5) for ranking/display. This task adds the larger pool behind a flag without changing the cap semantics.

- [ ] **Step 1: Add config for the expanded pool**

In `config.py`, after the `show_top` line (51), add:

```python
    wide_universe: bool = False       # include the expanded liquid pool (S&P-500-ish) in live scans
```

- [ ] **Step 2: Write the failing test**

Add to `selftest.py`:

```python
def test_universe_cap():
    print("universe cap:")
    import dataclasses
    import scanner
    from config import CONFIG
    cfg = dataclasses.replace(CONFIG, wide_universe=True, max_candidates=50)
    u = scanner.build_universe(cfg)
    _ok("universe respects max_candidates", len(u) <= cfg.max_candidates)
    _ok("universe is de-duplicated", len(u) == len(set(u)))
    _ok("core names still present", "AAPL" in u or "MSFT" in u)
```

Register `test_universe_cap()` in `main()`.

- [ ] **Step 3: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `dataclasses.replace` rejects unknown field `wide_universe` until Step 1 is in; if Step 1 done, test passes only after Step 4 wiring. Run to confirm current behaviour, then implement.

- [ ] **Step 4: Implement expanded pool in `build_universe`**

In `scanner.py`, add a module-level constant near `CORE_WATCHLIST` (after line ~45):

```python
# Expanded liquid pool — large/mid-cap names beyond the core list, used when
# cfg.wide_universe is set. Static (checked in) so offline scans stay deterministic.
# Keep to liquid, optionable names; refresh manually.
WIDE_POOL = [
    "ORCL", "CRM", "ADBE", "CSCO", "ACN", "TXN", "QCOM", "INTC", "IBM", "NOW",
    "INTU", "AMAT", "MU", "LRCX", "ADI", "PANW", "SNPS", "CDNS", "KLAC", "ANET",
    "ABBV", "LLY", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "CVS",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "AXP", "SPGI",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "WMT", "COST", "PG",
    "KO", "PEP", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "CAT", "DE",
    "BA", "GE", "HON", "UNP", "UPS", "RTX", "LMT", "MMM", "EMR", "ETN",
]
```

Then modify `build_universe()` (line 161-177) to include the pool when flagged. Change:

```python
    syms: list[str] = list(CORE_WATCHLIST)
    try:
        syms += market.most_actives(cfg)
        syms += market.movers(cfg)
```

to:

```python
    syms: list[str] = list(CORE_WATCHLIST)
    if getattr(cfg, "wide_universe", False):
        syms += WIDE_POOL
    try:
        syms += market.most_actives(cfg)
        syms += market.movers(cfg)
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_universe_cap` green; cap and dedupe hold.

- [ ] **Step 6: Commit**

```bash
git add config.py scanner.py selftest.py
git commit -m "scanner: optional expanded liquid universe pool (wide_universe flag)"
```

---

## Task 7: Tracker snapshots conviction checks

**Files:**
- Modify: `tracker.py:68-77` (the record appended when logging a fresh call).
- Test: `selftest.py` (new `test_tracker_checks_snapshot`)

Attribution needs each tracked call to remember its per-check pass/warn/fail at advice time.

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_tracker_checks_snapshot():
    print("tracker stores conviction checks:")
    import tempfile, json
    import tracker
    from config import CONFIG
    tf = tempfile.mktemp(suffix=".json"); tracker.PATH = tf
    json.dump([], open(tf, "w"))
    sig = {"symbol": "X", "name": "X Co", "action": "BUY", "direction": "LONG",
           "price": 100.0,
           "plan": {"entry": 100.0, "stop": 95.0, "target": 115.0, "rr": 3.0},
           "conviction": {"label": "High", "checks": [
               {"label": "Is it trending up?", "status": "pass", "weight": 1.0},
               {"label": "Leading the market?", "status": "pass", "weight": 2.0}]}}
    tracker.run([sig], CONFIG, live=True, today="2026-06-12")
    rec = json.load(open(tf))[0]
    _ok("record stores conviction checks", isinstance(rec.get("checks"), list) and len(rec["checks"]) == 2)
    _ok("snapshot keeps label+status", set(rec["checks"][0]) >= {"label", "status"})
```

Register `test_tracker_checks_snapshot()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — record has no `checks` key.

- [ ] **Step 3: Implement the snapshot**

In `tracker.py`, in the record dict appended around line 68-74, add a `checks` field. Change the dict literal to include (alongside the existing `"conviction": (s.get("conviction") or {}).get("label"),` line):

```python
                "checks": [{"label": c.get("label"), "status": c.get("status")}
                           for c in ((s.get("conviction") or {}).get("checks") or [])],
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_tracker_checks_snapshot` green; existing tracker tests still pass (extra field is additive).

- [ ] **Step 5: Commit**

```bash
git add tracker.py selftest.py
git commit -m "tracker: snapshot conviction checks on each logged call"
```

---

## Task 8: `attribution.py` — per-check win rates

**Files:**
- Create: `attribution.py`
- Test: `selftest.py` (new `test_attribution`)

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_attribution():
    print("factor attribution:")
    import attribution
    log = [
        {"status": "win",  "checks": [{"label": "Leading the market?", "status": "pass"},
                                       {"label": "Room to rise?", "status": "fail"}]},
        {"status": "win",  "checks": [{"label": "Leading the market?", "status": "pass"},
                                       {"label": "Room to rise?", "status": "pass"}]},
        {"status": "loss", "checks": [{"label": "Leading the market?", "status": "fail"},
                                       {"label": "Room to rise?", "status": "pass"}]},
        {"status": "open", "checks": [{"label": "Leading the market?", "status": "pass"}]},  # ignored
    ]
    rep = attribution.attribute(log)
    by = {r["label"]: r for r in rep}
    _ok("only resolved trades counted", by["Leading the market?"]["n_pass"] == 2)
    _ok("RS pass win rate computed", by["Leading the market?"]["win_rate_pass"] == 100.0)
    _ok("RS fail win rate computed", by["Leading the market?"]["win_rate_fail"] == 0.0)
    _ok("edge = pass minus fail win rate", by["Leading the market?"]["edge"] == 100.0)
    _ok("report sorted by edge desc", rep == sorted(rep, key=lambda r: -(r["edge"] if r["edge"] is not None else -999)))
```

Register `test_attribution()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'attribution'`.

- [ ] **Step 3: Implement `attribution.py`**

```python
"""Factor attribution — which conviction checks actually predict wins?

Reads the resolved trades in track_record.json (each carries a snapshot of its
conviction checks) and, per check label, compares the win rate when that check
PASSED vs when it FAILED. The gap ("edge") tells you which checks earn their
weight and which are decoration. This is the loop that keeps a widening funnel
honest. Read-only; accumulates from the first run after the snapshot ships.
"""
from __future__ import annotations

import json
import os

PATH = os.getenv("TRACK_FILE", "track_record.json")

_RESOLVED = ("win", "loss", "expired")


def _win_rate(items: list[dict]) -> float | None:
    decided = [t for t in items if t["status"] in ("win", "loss")]
    if not decided:
        return None
    return round(sum(1 for t in decided if t["status"] == "win") / len(decided) * 100, 1)


def attribute(log: list[dict]) -> list[dict]:
    """Per-check win-rate-when-pass vs win-rate-when-fail across resolved trades.

    Returns a list of {label, n_pass, n_fail, win_rate_pass, win_rate_fail, edge},
    sorted by edge (pass minus fail win rate) descending. `edge` is None when either
    side has no decided trades.
    """
    resolved = [t for t in log if t.get("status") in _RESOLVED and t.get("checks")]
    labels = []
    seen = set()
    for t in resolved:
        for c in t["checks"]:
            lbl = c.get("label")
            if lbl and lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)
    out = []
    for lbl in labels:
        passed = [t for t in resolved if any(c.get("label") == lbl and c.get("status") == "pass" for c in t["checks"])]
        failed = [t for t in resolved if any(c.get("label") == lbl and c.get("status") == "fail" for c in t["checks"])]
        wp, wf = _win_rate(passed), _win_rate(failed)
        edge = round(wp - wf, 1) if (wp is not None and wf is not None) else None
        out.append({"label": lbl, "n_pass": len(passed), "n_fail": len(failed),
                    "win_rate_pass": wp, "win_rate_fail": wf, "edge": edge})
    out.sort(key=lambda r: -(r["edge"] if r["edge"] is not None else -999))
    return out


def load(path: str | None = None) -> list[dict]:
    """Read the track-record log (best-effort; [] on any error)."""
    try:
        with open(path or PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return []


def report(path: str | None = None) -> list[dict]:
    return attribute(load(path))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_attribution` green.

- [ ] **Step 5: Commit**

```bash
git add attribution.py selftest.py
git commit -m "attribution: per-check win-rate attribution from track record"
```

---

## Task 9: Dashboard — RS percentile + attribution panel

**Files:**
- Modify: `dashboard.py` — add an RS percentile badge to signal cards; add a "Which checks earn their keep" panel built from `attribution.report()`.
- Test: `selftest.py` (new `test_attribution_panel`)

- [ ] **Step 1: Write the failing test**

Add to `selftest.py`:

```python
def test_attribution_panel():
    print("attribution panel renders:")
    import dashboard
    rep = [{"label": "Leading the market?", "n_pass": 12, "n_fail": 5,
            "win_rate_pass": 64.0, "win_rate_fail": 30.0, "edge": 34.0},
           {"label": "Retail buzz?", "n_pass": 8, "n_fail": 3,
            "win_rate_pass": 40.0, "win_rate_fail": 45.0, "edge": -5.0}]
    html = dashboard._attribution_html(rep)
    _ok("panel names the best factor", "Leading the market?" in html)
    _ok("panel shows the edge value", "+34" in html or "34.0" in html)
    _ok("empty report yields a friendly note", "accruing" in dashboard._attribution_html([]).lower()
         or dashboard._attribution_html([]) != "")
```

Register `test_attribution_panel()` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `python3 selftest.py`
Expected: FAIL — `module 'dashboard' has no attribute '_attribution_html'`.

- [ ] **Step 3: Implement `_attribution_html` and wire it in**

In `dashboard.py`, add (near `_track_html`, following the `.trackrec` table style used there):

```python
def _attribution_html(rep: list[dict] | None) -> str:
    """Panel: which conviction checks actually predicted wins (pass vs fail win rate)."""
    intro = ('<h3 style="font-size:15px;margin:18px 0 8px;">Which checks earn their keep '
             '<span style="text-transform:none;font-weight:400;color:var(--muted);font-size:12px;">'
             '— win rate when each conviction check passed vs failed, on resolved calls</span></h3>')
    if not rep:
        return intro + ('<p style="color:var(--muted);font-size:13px;">Still accruing — per-check '
                        'win rates appear here once enough tracked calls resolve.</p>')
    rows = ""
    for r in rep:
        wp = "—" if r["win_rate_pass"] is None else f'{r["win_rate_pass"]:.0f}%'
        wf = "—" if r["win_rate_fail"] is None else f'{r["win_rate_fail"]:.0f}%'
        edge = r["edge"]
        ec = "buy" if (edge or 0) > 0 else "sell" if (edge or 0) < 0 else ""
        es = "—" if edge is None else f'{"+" if edge > 0 else ""}{edge:.0f} pts'
        rows += (f'<tr><td>{r["label"]}</td>'
                 f'<td style="text-align:right;">{r["n_pass"]}/{r["n_fail"]}</td>'
                 f'<td style="text-align:right;">{wp}</td><td style="text-align:right;">{wf}</td>'
                 f'<td style="text-align:right;" class="{ec}">{es}</td></tr>')
    return (intro + '<table class="trackrec"><thead><tr><th>Check</th>'
            '<th style="text-align:right;">Pass/Fail n</th><th style="text-align:right;">Win% pass</th>'
            '<th style="text-align:right;">Win% fail</th><th style="text-align:right;">Edge</th>'
            '</tr></thead><tbody>' + rows + '</tbody></table>')
```

Then, where the track-record section is assembled in the dashboard build (search for `_track_html(`), append the attribution panel right after it:

```python
    import attribution
    track_html += _attribution_html(attribution.report())
```

(If `track_html` is built differently, insert `_attribution_html(attribution.report())` into the same section string.)

For the RS badge on cards: in the per-signal card render, where factors/badges are shown, add — guarded — an RS chip when `sig["factors"]["rs"]` exists:

```python
    rs_f = (sig.get("factors") or {}).get("rs")
    rs_badge = (f'<span class="badge">RS {rs_f["pct"]}</span>') if rs_f else ""
```

and include `rs_badge` in the card's badge row (next to the existing alt-data badges).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 selftest.py`
Expected: PASS — `test_attribution_panel` green.

- [ ] **Step 5: Build the dashboard end-to-end (smoke)**

Run: `python3 dashboard.py` (offline/synthetic is fine)
Expected: writes `dashboard.html` + `signals.json` with no traceback; the page contains "Which checks earn their keep" and RS chips.

- [ ] **Step 6: Commit**

```bash
git add dashboard.py selftest.py
git commit -m "dashboard: RS percentile badge + factor-attribution panel"
```

---

## Task 10: A/B backtest variant for RS (validation)

**Files:**
- Modify: `backtest_compare.py` — add an RS-gated variant so RS must earn its place.
- Test: none new (this is an analysis tool); just confirm it runs.

- [ ] **Step 1: Add an RS variant**

In `backtest_compare.py`, locate the list of config variants (baseline / ADX / trailing / both). Add a variant that raises the RS check's influence so its effect is measurable, e.g.:

```python
    import dataclasses
    variants.append(("rs_weighted", dataclasses.replace(base, rs_conviction_weight=3.0)))
```

(Adapt to the file's actual variant-construction style; the point is one extra labelled config with a higher `rs_conviction_weight`.)

- [ ] **Step 2: Run it**

Run: `python3 backtest_compare.py`
Expected: prints a comparison table including the `rs_weighted` row (win%, avg return, drawdown, Sharpe) with no traceback.

- [ ] **Step 3: Commit**

```bash
git add backtest_compare.py
git commit -m "backtest_compare: add RS-weighted variant for A/B validation"
```

---

## Final verification

- [ ] **Run the whole suite**

Run: `python3 selftest.py`
Expected: ends with `ALL TESTS PASSED`.

- [ ] **Smoke the live-shaped path offline**

Run: `python3 dashboard.py`
Expected: `dashboard.html` + `signals.json` written, no traceback, RS chips + attribution panel present.

---

## Self-Review notes (author)

- **Spec coverage:** Unit A (`rs.py`) → Task 2. Unit B (universe + promotion) → Tasks 5–6 (RS computed across universe; bounded by `max_candidates`; expanded pool behind `wide_universe`). Unit C (RS conviction check + attribution loop) → Tasks 3–4 (check), 7–8 (snapshot + attribution), 9 (panel). Config → Task 1. A/B validation → Task 10.
- **Known deviation from spec:** the spec floated promoting only the top-RS *slice* into the expensive path. The plan keeps the existing `max_candidates` cap and uses RS for ordering/conviction/display rather than a separate pre-fetch promotion step — simpler, same cost ceiling, and avoids a second fetch pass. Revisit if scans exceed rate limits at `wide_universe` scale.
- **Type consistency:** `rs.rs_score(...)->float|None`, `rs.rank_universe(...)->{sym:{"rs","pct"}}`, `factors["rs"]` shape `{"rs","pct"}` is consumed identically in Tasks 4, 5, 9. Conviction check label `"Leading the market?"` is identical across Tasks 4, 5. `attribution.attribute(log)->[{label,n_pass,n_fail,win_rate_pass,win_rate_fail,edge}]` consumed identically in Tasks 8, 9.
- **No placeholders:** every code step shows the actual code; tests use the repo's `_ok` convention and are registered in `main()`.
